r"""
trek_index.py
=============
Object-id -> module INDEX for a TREK project, so traceability can resolve
"which module does requirement/test id X live in?" with an O(1) dict lookup
instead of probing candidate modules live (the old ~100s+ bottleneck in
trek_gui._detect_bridge_modules).

Adapted from TREKsynch/build_traceability_generic.py (the proven standalone
builder). Key ideas kept:
  * BATCH module names per request so the URL never exceeds the server limit
    (long moduleNames values get truncated to /Export/TestCase -> 404).
  * attributeNames=Key;ModuleName -> ask only for the light fields.
  * Requirements (/Export/Requirements) return the REAL module name in
    ModuleName -> use it directly. Fetched SEQUENTIALLY in batches (the
    endpoint serializes server-side; parallel was slower).
  * Retries on transient timeouts.

Scope: indexes ALL FOUR V-Model levels into one flat id->module map:
    SYR + SWR  (requirements, via /Export/Requirements -- real ModuleName)
    SYT + SWT  (test cases, via /Export/Links per module -- module = the one
                fetched, since /Export/TestCases 404s on this server)
So every id encountered while following links (a SWR requirement id, a SWT
test-case id, etc.) resolves to its module with one dict lookup -- no
name-guessing at any hop of the SYT->SYR->SWR->SWT chain.

Storage: the built map is cached as a single blob in the existing
trek_cache SQLite (key_index below), so it survives restarts and is only
rebuilt on demand ("Build Index" button) or when missing.

Public API
----------
    idx = TrekIndex(cache, project_id, campaign_id, config_id)
    idx.load()                       # load cached map (if any)
    idx.is_built(kind="SWR")         # bool
    idx.build(kinds=("SYR","SWR"), progress_cb=..., force=False)
    idx.resolve("SWR_PM_1042")       # -> "SWR_120_Power_Management" or None
    idx.modules_for_ids(ids, prefix="SWR")  # -> set of module names
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Set

import trek_cache


# Disciplines this index covers -- ALL FOUR V-Model levels, so any id met
# while following links (SYR/SWR requirement ids OR SYT/SWT test-case ids)
# resolves to its module with an O(1) lookup.
#   kind "requirement" (domain 2 = RM): SYR/SWR via /Export/Requirements,
#       whose rows carry the REAL ModuleName -> fast batched sequential.
#   kind "testcase"    (domain 4 = QM): SYT/SWT via /Export/Links per module
#       (the /Export/TestCases endpoint 404s on this server), so the module
#       IS the one we fetched -> parallel per-module fetch.
_DISCIPLINES = {
    "SYR": {"prefix": "SYR", "domain": "2", "kind": "requirement"},
    "SWR": {"prefix": "SWR", "domain": "2", "kind": "requirement"},
    "SYT": {"prefix": "SYT", "domain": "4", "kind": "testcase"},
    "SWT": {"prefix": "SWT", "domain": "4", "kind": "testcase"},
}

# Max module names per /Export/Requirements request (keeps URL well under the
# server's ~3.7KB truncation limit). 15 matches the standalone builder.
_BATCH_SIZE = 15

# Concurrency for per-module test-case (links) fetches. Each worker uses its
# own SSPI session (see the light-links fn wired from trek_gui).
_MAX_WORKERS = 8


def key_index(project_id, campaign_id) -> str:
    """Cache key for the whole id->module index of one project."""
    return f"index:{project_id}:{campaign_id}"


class TrekIndex:
    def __init__(self, cache, project_id, campaign_id, config_id):
        self.cache = cache
        self.project_id = str(project_id)
        self.campaign_id = str(campaign_id)
        self.config_id = str(config_id)
        # object_id -> module_name
        self._map: Dict[str, str] = {}
        # Reverse map for the SWT hop: SWR requirement id -> {SWT module names
        # whose test cases OUT-link to that SWR id}. Built while indexing SWT
        # module links (a SWT test case owns the OUT link to its SWR), so at
        # traceability time our resolved SWR ids -> exact SWT modules with no
        # name-guessing and no chicken-and-egg.
        self._swr_to_swt_modules: Dict[str, Set[str]] = {}
        # which kinds have been indexed, for is_built()
        self._built_kinds: Set[str] = set()
        self._updated_at: str = ""
        # lazily-built {area_prefix: module} for inferring dead ids' modules
        self._prefix_map: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load the cached index blob if present. Returns True if loaded."""
        cached = self.cache.get_blob(key_index(self.project_id, self.campaign_id))
        if cached is None:
            return False
        payload, updated_at = cached
        if isinstance(payload, dict):
            self._map = payload.get("object_to_module", {}) or {}
            self._built_kinds = set(payload.get("built_kinds", []) or [])
            # reverse map stored as {swr_id: [swt_module,...]} -> set
            raw_rev = payload.get("swr_to_swt_modules", {}) or {}
            self._swr_to_swt_modules = {k: set(v) for k, v in raw_rev.items()}
            self._updated_at = updated_at
            self._prefix_map = None   # rebuild lazily against the new map
            return bool(self._map)
        return False

    def _save(self):
        self.cache.set_blob(
            key_index(self.project_id, self.campaign_id),
            {
                "object_to_module": self._map,
                "swr_to_swt_modules": {k: sorted(v)
                                       for k, v in self._swr_to_swt_modules.items()},
                "built_kinds": sorted(self._built_kinds),
                "project_id": self.project_id,
                "campaign_id": self.campaign_id,
            },
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_built(self, kind: Optional[str] = None) -> bool:
        if kind is None:
            return bool(self._map)
        return kind in self._built_kinds

    def updated_at(self) -> str:
        return self._updated_at

    def size(self) -> int:
        return len(self._map)

    def resolve(self, object_id: str) -> Optional[str]:
        """Return the module name an id lives in, or None if not indexed."""
        return self._map.get(object_id)

    def modules_for_ids(self, ids: Iterable[str],
                        prefix: Optional[str] = None) -> Set[str]:
        """Resolve a set of ids to the set of modules they live in.

        Ids that aren't in the index are silently skipped (caller can detect
        misses via unresolved_ids()). If `prefix` is given, only ids starting
        with it are considered (e.g. only 'SWR_' ids)."""
        out: Set[str] = set()
        for oid in ids:
            if prefix and not str(oid).startswith(prefix):
                continue
            mod = self._map.get(oid)
            if mod:
                out.add(mod)
        return out

    def _area_prefix(self, object_id: str) -> str:
        """Return the 'family' prefix of an id: everything up to and including
        the last underscore before the trailing number, e.g.
        'SWR_SI_494' -> 'SWR_SI_', 'SWR_011_BSWICS_M1_10' -> 'SWR_011_BSWICS_M1_'.
        Used to infer the module of a dead/dangling id from its siblings."""
        import re as _re
        m = _re.match(r"^(.*_)\d+$", str(object_id))
        return m.group(1) if m else ""

    def _build_prefix_map(self):
        """Lazily build {area_prefix: module} from the id->module map, so an
        unresolved id can be mapped to the module its sibling ids live in.
        A prefix maps to a module only if ALL its ids share that one module
        (unambiguous); ambiguous prefixes are left out."""
        if getattr(self, "_prefix_map", None) is not None:
            return
        pref_mods: Dict[str, Set[str]] = {}
        for oid, mod in self._map.items():
            p = self._area_prefix(oid)
            if p:
                pref_mods.setdefault(p, set()).add(mod)
        self._prefix_map = {p: next(iter(mods)) for p, mods in pref_mods.items()
                            if len(mods) == 1}

    def infer_modules_for_ids(self, ids: Iterable[str],
                              prefix: Optional[str] = None) -> Set[str]:
        """Best-effort resolve ids NOT directly in the index, by matching
        their area-prefix (e.g. 'SWR_SI_') to the module its sibling ids
        live in. Handles dead/dangling links to deleted requirements whose
        family still clearly belongs to one module -- so a few stray ids no
        longer trigger a full live candidate probe."""
        self._build_prefix_map()
        out: Set[str] = set()
        for oid in ids:
            if prefix and not str(oid).startswith(prefix):
                continue
            p = self._area_prefix(oid)
            mod = self._prefix_map.get(p)
            if mod:
                out.add(mod)
        return out

    def swt_modules_for_swr_ids(self, swr_ids: Iterable[str]) -> Set[str]:
        """Given a set of SWR requirement ids, return the set of SWT modules
        whose test cases OUT-link to any of them (built during SWT indexing).
        Empty if the SWT level hasn't been indexed."""
        out: Set[str] = set()
        for sid in swr_ids:
            mods = self._swr_to_swt_modules.get(sid)
            if mods:
                out |= mods
        return out

    def ids_by_module(self, module_filter: Callable[[str], bool]) -> Dict[str, List[str]]:
        """Group every indexed id by its module, keeping only modules for
        which ``module_filter(module_name)`` is True. Used by the offline
        test-case text download to enumerate all SYT/SWT test-case ids
        without any network call (the index already knows them all)."""
        out: Dict[str, List[str]] = {}
        keep: Dict[str, bool] = {}
        for oid, mod in self._map.items():
            ok = keep.get(mod)
            if ok is None:
                ok = keep[mod] = bool(module_filter(mod))
            if ok:
                out.setdefault(mod, []).append(oid)
        for ids in out.values():
            ids.sort()
        return out

    def has_swt_reverse(self) -> bool:
        return bool(self._swr_to_swt_modules)

    def unresolved_ids(self, ids: Iterable[str],
                        prefix: Optional[str] = None) -> List[str]:
        """Return the ids that are NOT in the index (need a fallback)."""
        miss: List[str] = []
        for oid in ids:
            if prefix and not str(oid).startswith(prefix):
                continue
            if oid not in self._map:
                miss.append(oid)
        return miss

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, kinds: Iterable[str] = ("SYR", "SWR", "SYT", "SWT"),
              progress_cb: Optional[Callable[[str], None]] = None,
              force: bool = False,
              log=None,
              syt_prefixes: Optional[List[str]] = None) -> Dict[str, int]:
        """Build (or refresh) the index for the given V-Model levels.

        Defaults to all four (SYR/SWR requirements + SYT/SWT test cases) so a
        single "Build Index" run produces the complete id->module map. Uses one
        fresh API client for requirements; test-case levels fetch per-module
        concurrently. Returns a per-kind object-count report. Requires
        set_client_factory() (trek_gui wires it to TrekExportLinksClient).

        Args:
            syt_prefixes: Optional list of module-name prefixes for the SYT
                kind (e.g. ["SYT", "SYTS"]). When provided, _list_modules
                matches any module whose name starts with ANY of them instead
                of just the default "SYT". Other kinds (SYR/SWR/SWT) are
                unaffected. See trek_gui.DEFAULT_SYT_PREFIXES.
        """
        if _client_factory is None:
            raise RuntimeError(
                "trek_index client factory not set -- call "
                "trek_index.set_client_factory(TrekExportLinksClient) first."
            )
        client = _client_factory()
        report: Dict[str, int] = {}

        for kind in kinds:
            disc = _DISCIPLINES.get(kind)
            if disc is None:
                continue
            if self.is_built(kind) and not force:
                if progress_cb:
                    progress_cb(f"{kind} index already built ({self.size()} ids). Skipping.")
                continue

            # Use the project's custom SYT prefixes when indexing SYT modules,
            # so projects that name their system-test modules "SYTS_..." (etc.)
            # get them included in the index alongside the standard "SYT -".
            prefixes = disc["prefix"]
            if kind == "SYT" and syt_prefixes:
                prefixes = syt_prefixes

            if progress_cb:
                progress_cb(f"Listing {kind} modules...")
            modules = self._list_modules(client, disc["domain"], prefixes)
            if not modules:
                report[kind] = 0
                continue

            if disc["kind"] == "requirement":
                added = self._build_requirements(client, modules, kind, progress_cb, log)
            else:  # testcase -> per-module links
                added = self._build_testcases(modules, kind, disc["domain"],
                                              progress_cb, log)
            report[kind] = added
            self._built_kinds.add(kind)
            self._save()   # save after each kind so a crash keeps progress

        return report

    # ------------------------------------------------------------------
    # Internals (adapted from build_traceability_generic.py)
    # ------------------------------------------------------------------
    def _list_modules(self, client, domain: str,
                      prefix) -> List[str]:
        """List modules for a domain, filtered by prefix(es).

        Args:
            prefix: Either a single prefix string (e.g. "SYR") or a list
                of prefix strings (e.g. ["SYT", "SYTS"]). A module is
                included if its name starts with ANY of the prefixes
                (case-insensitive, matched on the first token up to the
                first space/underscore/hyphen -- same logic as
                trek_gui._is_syt_module).
        """
        import re as _re
        resp = client.get_modules(int(self.project_id), int(self.config_id),
                                   int(domain), timeout=120)
        if not getattr(resp, "success", False):
            return []
        # Normalise to a set of upper-cased prefixes for matching.
        if isinstance(prefix, str):
            wanted = {prefix.upper()}
        else:
            wanted = {p.upper() for p in prefix}
        result = []
        for m in resp.modules:
            if not isinstance(m, dict):
                continue
            name = str(m.get("Name", ""))
            # Extract the first token (up to the first space/underscore/hyphen)
            # so "SYTS_TS_Climate" matches prefix "SYTS" but "SYTHESIS" does
            # not match "SYT" -- same logic as trek_gui._is_syt_module.
            first_token = _re.split(r"[ _\-]", name.strip(), maxsplit=1)[0].upper()
            if first_token in wanted:
                result.append(name)
        return result

    def _build_requirements(self, client, modules: List[str], kind: str,
                            progress_cb, log) -> int:
        """Sequential batched /Export/Requirements(attributeNames=Key;ModuleName).
        ModuleName in each row is the real module name."""
        before = len(self._map)
        batches = [modules[i:i + _BATCH_SIZE]
                   for i in range(0, len(modules), _BATCH_SIZE)]
        for i, batch in enumerate(batches, 1):
            if progress_cb:
                progress_cb(f"Indexing {kind}: batch {i}/{len(batches)} "
                            f"({len(batch)} modules)...")
            rows = self._fetch_key_module(client, batch, log, kind)
            for row in rows:
                if isinstance(row, dict) and row.get("Key"):
                    mod = row.get("ModuleName") or batch[0]
                    self._map[row["Key"]] = mod
        return len(self._map) - before

    def _fetch_key_module(self, client, batch: List[str], log, kind) -> list:
        """Call get_requirements for a batch, asking only Key;ModuleName.

        The app's TrekExportLinksClient.get_requirements does not expose the
        attributeNames param, so we call the underlying HTTP session directly
        via the client's helpers when available, else fall back to the full
        get_requirements (still correct, just heavier).
        """
        # Prefer a lightweight raw call if the client exposes one.
        raw = _light_requirements(client, int(self.project_id),
                                  int(self.campaign_id), batch)
        if raw is not None:
            return raw
        # Fallback: full requirements (heavier but correct).
        resp = client.get_requirements(int(self.project_id),
                                        int(self.campaign_id), batch, timeout=300)
        if not getattr(resp, "success", False):
            return []
        return getattr(resp, "requirements", []) or []

    def _build_testcases(self, modules: List[str], kind: str, domain: str,
                         progress_cb, log) -> int:
        """Index SYT/SWT test-case ids via /Export/Links, fetched PER MODULE
        and CONCURRENTLY. The module identity is the module we asked for, so
        every item's Key maps to that module.

        For SWT specifically, ALSO build the reverse SWR-id -> {SWT module}
        map from each item's LinkKey that starts with 'SWR_' (a SWT test case
        owns the OUT link to its SWR requirement). This is what lets the
        traceability build resolve SWT modules from our SWR ids without
        name-guessing. Each worker uses its own SSPI session (thread-safe).
        """
        before = len(self._map)
        total = len(modules)
        done = 0
        build_reverse = (kind == "SWT")

        if _light_links_fn is None:
            if progress_cb:
                progress_cb(f"[{kind}] no links fetcher available -- skipped.")
            return 0

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        _local = threading.local()

        def _client():
            c = getattr(_local, "client", None)
            if c is None:
                c = _client_factory()
                _local.client = c
            return c

        def worker(mod):
            items = _light_links_fn(_client(), int(self.project_id),
                                    int(self.campaign_id), mod, int(domain))
            return mod, items

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = {ex.submit(worker, m): m for m in modules}
            for fut in as_completed(futures):
                mod = futures[fut]
                try:
                    mod, items = fut.result()
                except Exception:   # noqa: BLE001
                    items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    key = it.get("Key")
                    if key:
                        self._map[key] = mod
                    if build_reverse:
                        lk = it.get("LinkKey", "")
                        if lk.startswith("SWR_"):
                            self._swr_to_swt_modules.setdefault(lk, set()).add(mod)
                done += 1
                if progress_cb and (done % 10 == 0 or done == total):
                    progress_cb(f"Indexing {kind}: {done}/{total} modules...")

        return len(self._map) - before


# --------------------------------------------------------------------------
# Client factory + lightweight requirements call, wired by trek_gui so this
# module doesn't hard-import the API client (keeps it testable in isolation).
# --------------------------------------------------------------------------
_client_factory: Optional[Callable[[], object]] = None
_light_req_fn: Optional[Callable] = None
_light_links_fn: Optional[Callable] = None


def set_client_factory(factory: Callable[[], object]) -> None:
    global _client_factory
    _client_factory = factory


def set_light_requirements_fn(fn: Callable) -> None:
    """Optionally provide a fn(client, project_id, campaign_id, modules)->list
    that fetches ONLY Key;ModuleName (the fast path). If not set, build()
    falls back to the client's full get_requirements."""
    global _light_req_fn
    _light_req_fn = fn


def set_light_links_fn(fn: Callable) -> None:
    """Provide fn(client, project_id, campaign_id, module, domain)->set[str]
    that returns the set of test-case Keys in one module (via /Export/Links).
    Required for indexing SYT/SWT test-case levels."""
    global _light_links_fn
    _light_links_fn = fn


def _light_requirements(client, project_id, campaign_id, modules):
    if _light_req_fn is None:
        return None
    try:
        return _light_req_fn(client, project_id, campaign_id, modules)
    except Exception:
        return None
