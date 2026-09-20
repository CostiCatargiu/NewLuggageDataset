"""
TREK Local Cache
================
Small SQLite-backed cache used by trek_gui.py to avoid re-hitting the TREK
Web API for data that rarely changes (module list, SYT test-case IDs per
module, SYT->SYR->SWR->SWT link edges, and full test-case content).

Design
------
Two tables:

  * ``blobs``      -- generic key/value store for JSON-serialisable payloads
                       (module lists, TC-id lists, link-edge maps). Keyed by
                       an opaque string built by the caller (see the
                       ``*_key`` helpers below) so cache entries are scoped
                       per project/campaign/module as needed.

  * ``tc_content`` -- one row per TREK test-case ID holding its full content
                       (Name, PreCondition, Procedure, ...). This is a
                       dedicated table (rather than a blob) because content
                       is looked up/merged per-ID very frequently and is
                       shared across every SYT module / traceability run.

Every entry carries an ``updated_at`` ISO-8601 timestamp so the GUI can show
"cached 2h ago" next to data, and callers can decide to force a refresh.

The cache is intentionally dumb: it does not know about TREK semantics, only
about storing/retrieving JSON blobs and TC content by key. All key-building
and refresh policy lives in trek_gui.py.
"""

from __future__ import annotations

import array
import hashlib
import sys
import json
import sqlite3
import datetime
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import orjson as _fast_json   # 5-10x faster than stdlib json
    _fast_loads = _fast_json.loads
    def _fast_dumps(obj) -> str:
        return _fast_json.dumps(obj).decode("utf-8")
except ImportError:
    _fast_json = None
    _fast_loads = json.loads
    _fast_dumps = json.dumps

import trek_paths

DEFAULT_DB_PATH = trek_paths.data_file("trek_cache.sqlite3")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Network-share performance helpers
# ---------------------------------------------------------------------------
# Over SMB every SQLite page read is one network round trip (measured ~56 ms
# on the team share), and a large value (e.g. the 8.7 MB id index) is a
# CHAIN of pages read one after another -- 2,200 pages x 56 ms ~= 2 minutes.
# Two things cut the number of round trips:
#   * bigger pages (NETWORK_PAGE_SIZE for new databases / optimize), and
#   * zlib-compressing large blob values (JSON compresses ~5-10x).
NETWORK_PAGE_SIZE = 65536
_BLOB_COMPRESS_MIN = 8192        # compress blob JSON >= 8 KB; tiny values stay text


def _encode_blob_text(text: str):
    """Value to store in blobs.value_json: plain text for small values,
    zlib-compressed bytes (SQLite BLOB) for large ones."""
    if len(text) >= _BLOB_COMPRESS_MIN:
        return sqlite3.Binary(zlib.compress(text.encode("utf-8"), 6))
    return text


def _decode_blob_value(value):
    """Inverse of _encode_blob_text -- accepts old plain-text rows too."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = zlib.decompress(bytes(value))
    return _fast_loads(value)


# Human review states a pair can be in, plus "" = not reviewed. Kept here
# (not imported from trek_similarity) so the cache layer stays standalone.
_REVIEW_KEYS = ("same_scenario", "partial_overlap", "different_scenario", "")
_VERDICT_KEYS = ("same_scenario", "partial_overlap", "different_scenario", "error", "")


def compute_dupcheck_stats(result_dict: dict) -> dict:
    """Count a duplicate-check result into a small matrix
    {llm_verdict: {review_status: n}} (plus the pair total). Everything the
    Dashboard shows -- per view, per module -- is derived from this, so it
    never has to load the full result again."""
    xtab: Dict[str, Dict[str, int]] = {}
    total = 0
    for p in result_dict.get("pairs", []) or []:
        if not isinstance(p, dict):
            continue
        verdict = p.get("llm_verdict") or ""
        if verdict not in _VERDICT_KEYS:
            verdict = "error"
        review = p.get("review_status") or ""
        if review not in _REVIEW_KEYS:
            review = ""
        xtab.setdefault(verdict, {})
        xtab[verdict][review] = xtab[verdict].get(review, 0) + 1
        total += 1
    return {"version": 1, "total": total, "xtab": xtab}


def is_network_path(path) -> bool:
    r"""True for UNC paths (\\server\share, //server/share) AND for drive
    letters mapped to a network share (e.g. Z: -> \\server\share). The
    latter used to be treated as local, which enabled WAL + mmap -- both
    unsupported on network file systems (WAL can corrupt a shared file)."""
    s = str(path)
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    if sys.platform == "win32":
        try:
            import ctypes, os
            drive = os.path.splitdrive(os.path.abspath(s))[0]
            if drive:
                DRIVE_REMOTE = 4
                return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE
        except Exception:
            return False
    return False


def _pack_vector(vector: List[float]) -> bytes:
    """Pack a float vector into a compact float32 binary blob.

    An embedding vector stored as JSON text (e.g. "[-0.0234567891, ...]")
    costs roughly 18-20 bytes per float (sign, digits, decimal point,
    separator). Packed as raw float32, it costs exactly 4 bytes per float
    -- a ~4-5x size reduction, which matters a lot both for total database
    size AND for read/write latency when the .sqlite3 file lives on a
    network share (fewer bytes to transfer, and no JSON parsing of a huge
    string on every read). float32 precision (~7 significant digits) is
    ample for cosine-similarity scoring; embeddings are never displayed
    or used for exact reproduction.
    """
    return array.array("f", vector).tobytes()


def _unpack_vector(blob: bytes) -> List[float]:
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


class TrekCache:
    """Thin SQLite wrapper providing get/set for blobs and TC content."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._init_timings: Dict[str, float] = {}
        self._batch_depth: int = 0           # batch_writes() nesting depth
        self._blob_cache: Dict[str, Tuple[Any, str]] = {}  # in-memory LRU for get_blob
        self._blob_cache_max: int = 128      # max entries before eviction
        import time as _t

        # Detect whether the database lives on a network share (UNC path or
        # mapped drive whose root starts with \\). mmap and WAL both
        # misbehave over SMB -- disable them proactively instead of relying
        # on silent fallbacks that still cost round trips to discover.
        self._is_network = is_network_path(self.db_path)
        try:
            is_new = (not self.db_path.exists()) or self.db_path.stat().st_size == 0
        except OSError:
            is_new = True

        t0 = _t.perf_counter()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._init_timings["connect"] = _t.perf_counter() - t0
        if is_new:
            # Brand-new database: use big pages from the start so it stays
            # fast if it is ever moved to a network share (must be set
            # before the first table is created).
            try:
                self._conn.execute(f"PRAGMA page_size={NETWORK_PAGE_SIZE};")
            except sqlite3.Error:
                pass

        t0 = _t.perf_counter()
        # Network share: WAL needs shared memory (-shm) that SMB cannot
        # provide -- using it there can corrupt the file. A database copied
        # from a local WAL-mode file KEEPS WAL mode in its header, so the
        # mode is actively switched when it does not match.
        #
        # CHANGING the mode is expensive on a share (measured 4.2 s, every
        # single start), while READING it is just the file header we already
        # paid for -- so only write when it actually differs.
        wanted = "delete" if self._is_network else "wal"
        try:
            current = (self._conn.execute("PRAGMA journal_mode;").fetchone()[0] or "").lower()
            self.journal_mode_changed = current != wanted
            if current == wanted:
                self.journal_mode = current
            else:
                self.journal_mode = self._conn.execute(
                    f"PRAGMA journal_mode={wanted.upper()};").fetchone()[0]
        except sqlite3.Error:
            self.journal_mode = "delete"
        self._init_timings["journal_mode"] = _t.perf_counter() - t0

        # Without this, a second user writing at the same instant fails
        # immediately with "database is locked" instead of waiting their turn.
        self._conn.execute("PRAGMA busy_timeout=30000;")
        # --------------------------------------------------------------
        # Performance tuning. A shared cache (e.g. an SMB path like
        # \\server\share\Database) pays a network round trip for every
        # uncached page SQLite touches, so we tune aggressively. None of
        # this trades correctness: the cache is fully rebuildable.
        try:
            # ~64MB page cache instead of SQLite's default ~2MB.
            self._conn.execute("PRAGMA cache_size=-65536;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")

            if self._is_network:
                # mmap is unreliable/slow over SMB (Windows can't reliably
                # memory-map a remote file). Disable it entirely.
                self._conn.execute("PRAGMA mmap_size=0;")
            else:
                # Local disk: mmap avoids read()-per-page for large scans.
                self._conn.execute("PRAGMA mmap_size=268435456;")
        except sqlite3.Error:
            pass  # Best-effort tuning; never block cache startup on this.

        t0 = _t.perf_counter()
        self._init_schema()
        self._init_timings["schema_migrations"] = _t.perf_counter() - t0

    @property
    def is_shared(self) -> bool:
        """True when the cache is NOT in WAL mode, i.e. it lives on a network
        share -- writes serialize across users and are slower."""
        return str(self.journal_mode).lower() != "wal"

    # ------------------------------------------------------------------
    # Batched writes (commit grouping)
    # ------------------------------------------------------------------
    class _BatchContext:
        """Context manager that defers commits until the outermost batch
        exits. Nestable (ref-counted). On a network share every commit is
        3-5 SMB round trips (journal write + fsync + db write + fsync +
        journal delete), so batching 10 writes into one commit saves ~30-50
        round trips."""
        def __init__(self, cache: "TrekCache"):
            self._cache = cache
        def __enter__(self):
            self._cache._batch_depth += 1
            return self._cache
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._cache._batch_depth -= 1
            if self._cache._batch_depth == 0:
                self._cache._conn.commit()
            return False

    def batch_writes(self) -> "_BatchContext":
        """Return a context manager that defers ``commit()`` calls until the
        outermost ``with cache.batch_writes():`` block exits. Safe to nest.

        Usage::

            with CACHE.batch_writes():
                CACHE.set_blob(k1, d1)   # no commit here
                CACHE.set_blob(k2, d2)   # no commit here
                CACHE.record_operation_stat(...)  # no commit here
            # ONE commit here -- 1 round trip instead of 3
        """
        return self._BatchContext(self)

    def _commit(self):
        """Commit if not inside a batch_writes() block. When batching,
        the commit is deferred to the outermost __exit__."""
        if self._batch_depth == 0:
            self._conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                key         TEXT PRIMARY KEY,
                value_json  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                item_count  INTEGER NOT NULL DEFAULT -1
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tc_content (
                tc_id       TEXT PRIMARY KEY,
                content_json TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash   TEXT NOT NULL,
                model       TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                cost_usd    REAL NOT NULL DEFAULT 0.0,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (text_hash, model)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_judgments (
                key_hash    TEXT PRIMARY KEY,
                verdict     TEXT NOT NULL,
                reasoning   TEXT NOT NULL,
                cost_usd    REAL NOT NULL DEFAULT 0.0,
                updated_at  TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS duplicate_check_runs (
                module_key   TEXT PRIMARY KEY,
                syt_module   TEXT NOT NULL,
                result_json  TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                pair_count   INTEGER NOT NULL DEFAULT 0,
                llm_model    TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Separate table for the heavy result_json blobs so that listing
        # modules (which only needs the lightweight metadata in
        # duplicate_check_runs) never has to touch the multi-MB JSON
        # payloads -- SQLite stores all columns on the same page, so even
        # selecting only pair_count from a table that ALSO has a 50MB
        # result_json column means reading those pages over the network.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS duplicate_check_data (
                module_key   TEXT PRIMARY KEY,
                result_json  TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operation_stats (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                op_type           TEXT NOT NULL,
                module            TEXT,
                item_count        INTEGER NOT NULL DEFAULT 0,
                extra_count       INTEGER NOT NULL DEFAULT 0,
                duration_seconds  REAL NOT NULL DEFAULT 0.0,
                cost_usd          REAL NOT NULL DEFAULT 0.0,
                source            TEXT,
                details_json      TEXT,
                created_at        TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operation_stats_op_type "
            "ON operation_stats(op_type)"
        )
        self._conn.commit()
        self._migrate_add_cost_columns()
        self._migrate_add_details_json_column()
        self._migrate_add_vector_blob_column()
        self._migrate_add_dupcheck_summary_columns()
        self._migrate_add_dupcheck_stats_column()
        self._migrate_add_meta_indexes()
        self._migrate_split_dupcheck_data()
        self._migrate_add_item_count_column()

    def _migrate_add_cost_columns(self):
        """Add cost_usd to tables created before cost tracking existed --
        SQLite's CREATE TABLE IF NOT EXISTS above is a no-op on an already
        existing (older) table, so older on-disk caches need an explicit
        ALTER TABLE to gain the new column."""
        for table in ("embeddings", "llm_judgments"):
            cols = [row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "cost_usd" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0.0")
        self._conn.commit()

    def _migrate_add_details_json_column(self):
        """Add details_json to operation_stats for caches created before
        per-op-type structured details (e.g. traceability's SYR/SWR/SWT/
        related-SYT breakdown) existed."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(operation_stats)").fetchall()]
        if "details_json" not in cols:
            self._conn.execute("ALTER TABLE operation_stats ADD COLUMN details_json TEXT")
        self._conn.commit()

    def _migrate_add_vector_blob_column(self):
        """Add vector_blob (compact float32 binary) to the embeddings table
        for caches created before binary storage existed. vector_json is
        kept alongside it (nullable going forward) rather than dropped --
        SQLite can't cheaply drop/alter a column with data in it, and
        keeping both lets old rows keep working via get_embeddings()'s
        fallback path until optimize_storage() migrates and reclaims them.
        New rows are always written blob-only (vector_json = NULL) by
        set_embeddings(); see that method."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(embeddings)").fetchall()]
        if "vector_blob" not in cols:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN vector_blob BLOB")
        if "vector_json" in cols:
            # Older schema had vector_json NOT NULL; new rows only populate
            # vector_blob, so relax that constraint via a rebuild-free trick:
            # nothing to do here since SQLite's ALTER TABLE never enforces
            # NOT NULL retroactively on existing tables, but the constraint
            # WOULD reject future INSERTs with vector_json=NULL if we tried
            # to add a fresh NOT NULL column. Since vector_json already
            # exists as NOT NULL from CREATE TABLE, write an empty-string
            # placeholder instead of NULL for new blob-only rows (see
            # set_embeddings()) to satisfy that legacy constraint cheaply.
            pass
        self._conn.commit()

    def _migrate_add_dupcheck_stats_column(self):
        """Add stats_json to duplicate_check_runs: a tiny per-module count
        matrix (LLM verdict x human review status) written whenever a result
        is saved. The Dashboard reads THESE instead of loading every
        multi-MB result just to count verdicts."""
        cols = [row[1] for row in self._conn.execute(
            "PRAGMA table_info(duplicate_check_runs)").fetchall()]
        if "stats_json" not in cols:
            self._conn.execute(
                "ALTER TABLE duplicate_check_runs ADD COLUMN stats_json TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    def _migrate_add_meta_indexes(self):
        """Indexes that let the "Database" dialog summarise the cache without
        scanning the big value columns (on a network share a table scan of
        `blobs` drags every inline blob fragment across the wire)."""
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blobs_meta ON blobs(key, item_count, updated_at)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tc_content_updated ON tc_content(updated_at)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_legacy ON embeddings(text_hash) "
                "WHERE vector_blob IS NULL")
            self._conn.commit()
        except sqlite3.Error:
            pass      # best-effort; never block startup on an index

    def _migrate_add_dupcheck_summary_columns(self):
        """Add pair_count + llm_model summary columns to duplicate_check_runs
        so list_duplicate_check_modules() can populate the module combo
        WITHOUT reading and JSON-parsing every full result_json blob -- a
        huge latency win when the cache lives on a network share with many
        large results (each can be several MB of JSON).

        The backfill is tracked by a sentinel row in the blobs table
        ('_migration:dupcheck_summary_backfill') so it runs exactly ONCE
        per database, even if some rows legitimately have 0 pairs -- the
        previous approach (WHERE pair_count = 0) would re-scan those rows
        on every startup, pulling hundreds of MB of JSON over the network
        for no reason."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(duplicate_check_runs)").fetchall()]
        if "pair_count" not in cols:
            self._conn.execute("ALTER TABLE duplicate_check_runs ADD COLUMN pair_count INTEGER NOT NULL DEFAULT 0")
        if "llm_model" not in cols:
            self._conn.execute("ALTER TABLE duplicate_check_runs ADD COLUMN llm_model TEXT NOT NULL DEFAULT ''")

        # Check if backfill already ran (sentinel in blobs table).
        sentinel_key = "_migration:dupcheck_summary_backfill"
        cur = self._conn.execute("SELECT 1 FROM blobs WHERE key = ?", (sentinel_key,))
        if cur.fetchone() is None:
            # One-time backfill: read each result_json once to extract the
            # summary fields, then write a sentinel so this never runs again.
            cur2 = self._conn.execute(
                "SELECT module_key, result_json FROM duplicate_check_runs"
            )
            updates = []
            for module_key, result_json in cur2.fetchall():
                try:
                    parsed = json.loads(result_json)
                    pc = len(parsed.get("pairs", []))
                    lm = parsed.get("llm_model", "")
                except (json.JSONDecodeError, TypeError):
                    pc, lm = 0, ""
                updates.append((pc, lm, module_key))
            if updates:
                self._conn.executemany(
                    "UPDATE duplicate_check_runs SET pair_count = ?, llm_model = ? WHERE module_key = ?",
                    updates,
                )
            # Write sentinel so we never re-scan.
            self._conn.execute(
                "INSERT INTO blobs (key, value_json, updated_at) VALUES (?, ?, ?)",
                (sentinel_key, '"done"', _now_iso()),
            )
        self._conn.commit()

    def _migrate_split_dupcheck_data(self):
        """Move heavy result_json blobs from duplicate_check_runs into the
        separate duplicate_check_data table (created in _init_schema).

        SQLite stores all columns of a row on the same disk pages, so even
        a query selecting only pair_count from duplicate_check_runs has to
        READ through the pages containing the multi-MB result_json blobs.
        On a network share, that means listing 5 modules pulls hundreds of
        MB over SMB just to read 5 lightweight rows. Splitting the heavy
        column into its own table means listing touches only tiny pages.

        Tracked by a sentinel so it runs exactly once per database."""
        sentinel_key = "_migration:dupcheck_data_split"
        cur = self._conn.execute("SELECT 1 FROM blobs WHERE key = ?", (sentinel_key,))
        if cur.fetchone() is not None:
            return  # already done

        # Copy result_json from old table to new table (INSERT OR IGNORE
        # in case duplicate_check_data already has some rows from a
        # partial previous run / concurrent writer).
        self._conn.execute(
            """
            INSERT OR IGNORE INTO duplicate_check_data (module_key, result_json)
            SELECT module_key, result_json FROM duplicate_check_runs
            WHERE result_json != ''
            """
        )
        # Clear result_json in the old table to free up page space -- the
        # next VACUUM (manual or via "Optimize Storage") will physically
        # reclaim it. We keep the column itself (can't DROP COLUMN easily
        # in SQLite) but set it to '' so it costs ~0 bytes per row.
        self._conn.execute("UPDATE duplicate_check_runs SET result_json = ''")

        self._conn.execute(
            "INSERT INTO blobs (key, value_json, updated_at) VALUES (?, ?, ?)",
            (sentinel_key, '"done"', _now_iso()),
        )
        self._conn.commit()

    def _migrate_add_item_count_column(self):
        """Add item_count to the blobs table for caches created before
        pre-computed item counts existed. Old rows keep item_count=-1
        (the DEFAULT); list_blob_entries() handles that gracefully by
        falling back to JSON parsing only for those rows. New set_blob()
        calls always write the real count, so -1 rows disappear over time
        as data is naturally refreshed. NO BACKFILL -- reading and parsing
        every blob's value_json on a network share is catastrophically
        slow (the original backfill took 154s on a multi-GB cache)."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(blobs)").fetchall()]
        if "item_count" not in cols:
            self._conn.execute("ALTER TABLE blobs ADD COLUMN item_count INTEGER NOT NULL DEFAULT -1")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Generic blob cache (modules list, TC-id lists, link edge maps, ...)
    # ------------------------------------------------------------------
    def get_blob(self, key: str) -> Optional[Tuple[Any, str]]:
        """Return (data, updated_at_iso) for ``key``, or None if absent.

        Results are cached in memory so repeated reads of the same key
        (common during traceability builds where link/requirement data is
        read multiple times) skip both the SQLite query AND the JSON
        deserialization -- a large win when the .sqlite3 file lives on a
        network share (avoids 10-30 MB reads for the index blob, hundreds
        of KB for link data, etc.)."""
        # In-memory hit?
        cached = self._blob_cache.get(key)
        if cached is not None:
            return cached
        # SQLite fallback
        cur = self._conn.execute(
            "SELECT value_json, updated_at FROM blobs WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if not row:
            return None
        value_json, updated_at = row
        try:
            data = _decode_blob_value(value_json)
        except (json.JSONDecodeError, ValueError, zlib.error):
            return None
        result = (data, updated_at)
        # Store in memory cache (simple bounded dict).
        if len(self._blob_cache) >= self._blob_cache_max:
            # Evict oldest ~25% to avoid pathological single-eviction loops.
            evict_keys = list(self._blob_cache.keys())[:self._blob_cache_max // 4]
            for ek in evict_keys:
                del self._blob_cache[ek]
        self._blob_cache[key] = result
        return result

    def set_blob(self, key: str, data: Any, keep_in_memory: bool = True) -> str:
        """Store ``data`` (JSON-serialisable) under ``key``. Returns the
        updated_at timestamp that was written.

        ``keep_in_memory=False`` writes to SQLite only and drops any stale
        in-memory copy -- used by bulk downloads (offline preparation) so
        hundreds of MB of requirement payloads are not held in RAM."""
        updated_at = _now_iso()
        ic = len(data) if isinstance(data, (list, dict)) else 1
        self._conn.execute(
            """
            INSERT INTO blobs (key, value_json, updated_at, item_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at,
                item_count = excluded.item_count
            """,
            (key, _encode_blob_text(json.dumps(data)), updated_at, ic),
        )
        self._commit()
        # Invalidate/update the in-memory blob cache.
        if keep_in_memory:
            self._blob_cache[key] = (data, updated_at)
        else:
            self._blob_cache.pop(key, None)
        return updated_at

    def present_blob_keys(self, keys: List[str]) -> set:
        """Return which of ``keys`` exist in the blobs table, WITHOUT loading
        their payloads (cheap existence check for bulk downloads)."""
        present: set = set()
        keys = list(keys)
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT key FROM blobs WHERE key IN ({placeholders})", chunk)
            present.update(r[0] for r in cur.fetchall())
        return present

    def clear_blob(self, key: str) -> None:
        self._conn.execute("DELETE FROM blobs WHERE key = ?", (key,))
        self._commit()
        self._blob_cache.pop(key, None)

    def clear_blobs_prefix(self, prefix: str) -> None:
        """Delete all blob keys starting with ``prefix`` (e.g. to invalidate
        every cached artefact for one SYT module when it is refreshed)."""
        self._conn.execute("DELETE FROM blobs WHERE key LIKE ?", (prefix + "%",))
        self._commit()
        # Evict matching entries from the in-memory cache.
        for k in [k for k in self._blob_cache if k.startswith(prefix)]:
            del self._blob_cache[k]

    # ------------------------------------------------------------------
    # Embedding cache (content-addressed, global -- used by
    # trek_similarity.py for SYT vs SWT duplicate detection). Keyed by a
    # hash of the NORMALIZED text rather than by TC id, so two different
    # test cases that happen to share identical text (e.g. a SWT literally
    # copy-pasted from its SYT) reuse the same cached vector instead of
    # paying for two separate embedding API calls.
    # ------------------------------------------------------------------
    @staticmethod
    def text_hash(normalized_text: str) -> str:
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def get_embeddings(self, normalized_texts: List[str], model: str) -> Dict[str, List[float]]:
        """Return {normalized_text: vector} for whichever of the given
        (already-normalized) texts are cached under ``model``. Missing
        texts are simply absent from the result.

        Reads vector_blob (compact float32 binary) when present, falling
        back to the legacy vector_json text column for rows written before
        binary storage existed (see _migrate_add_vector_blob_column /
        optimize_storage) -- so old caches keep working without forcing a
        migration before first use."""
        if not normalized_texts:
            return {}
        hash_to_text = {self.text_hash(t): t for t in normalized_texts}
        found: Dict[str, List[float]] = {}
        CHUNK = 500
        hashes = list(hash_to_text.keys())
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT text_hash, vector_blob, vector_json FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                [model] + chunk,
            )
            for text_hash, vector_blob, vector_json in cur.fetchall():
                try:
                    if vector_blob is not None:
                        found[hash_to_text[text_hash]] = _unpack_vector(vector_blob)
                    elif vector_json:
                        found[hash_to_text[text_hash]] = json.loads(vector_json)
                except (json.JSONDecodeError, ValueError):
                    continue
        return found

    def set_embeddings(self, vectors_by_normalized_text: Dict[str, List[float]], model: str,
                        cost_by_text: Optional[Dict[str, float]] = None) -> None:
        """Store vectors, optionally with the per-text cost (USD) paid to
        compute them -- see trek_similarity.embed_texts()'s per_text_cost
        parameter, which splits a batch call's real reported cost evenly
        across the texts in that batch. Used to reconstruct the REAL
        historical cost of a cache hit later (see get_embedding_costs()),
        not just an estimate.

        Vectors are always written in the compact binary format
        (vector_blob); vector_json is left empty ("" to satisfy the legacy
        NOT NULL constraint without storing real duplicate data) for new
        rows -- see optimize_storage() for converting OLD json-only rows."""
        if not vectors_by_normalized_text:
            return
        cost_by_text = cost_by_text or {}
        updated_at = _now_iso()
        rows = [
            (self.text_hash(text), model, _pack_vector(vector), cost_by_text.get(text, 0.0), updated_at)
            for text, vector in vectors_by_normalized_text.items()
        ]
        self._conn.executemany(
            """
            INSERT INTO embeddings (text_hash, model, vector_json, vector_blob, cost_usd, updated_at)
            VALUES (?, ?, '', ?, ?, ?)
            ON CONFLICT(text_hash, model) DO UPDATE SET
                vector_json = '',
                vector_blob = excluded.vector_blob,
                cost_usd = excluded.cost_usd,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self._commit()

    def get_embedding_costs(self, normalized_texts: List[str], model: str) -> Dict[str, float]:
        """Return {normalized_text: cost_usd} for whichever of the given
        texts are cached -- the REAL cost recorded when that text was
        first embedded, so a 'Check Duplicates' run served entirely from
        cache can still report the true historical cost of what it's
        showing, not just this run's (zero) fresh spend."""
        if not normalized_texts:
            return {}
        hash_to_text = {self.text_hash(t): t for t in normalized_texts}
        found: Dict[str, float] = {}
        CHUNK = 500
        hashes = list(hash_to_text.keys())
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT text_hash, cost_usd FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                [model] + chunk,
            )
            for text_hash, cost_usd in cur.fetchall():
                found[hash_to_text[text_hash]] = cost_usd
        return found

    def embeddings_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM embeddings")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # LLM-judgment cache (content-addressed, global -- used by
    # trek_similarity.run_llm_judge_stage()). Keyed by a hash of a
    # composite string built from the normalized SYT/counterpart texts +
    # counterpart_type + model + judging instructions (see
    # trek_similarity.llm_judgment_cache_key()), so an identical pair
    # judged under the identical settings is never sent to the LLM twice,
    # while a changed model or changed instructions correctly misses the
    # cache and gets re-judged.
    # ------------------------------------------------------------------
    def get_llm_judgments(self, cache_keys: List[str]) -> Dict[str, Tuple[str, str]]:
        """Return {cache_key: (verdict, reasoning)} for whichever of the
        given cache keys are cached. Missing keys are simply absent."""
        if not cache_keys:
            return {}
        hash_to_key = {self.text_hash(k): k for k in cache_keys}
        found: Dict[str, Tuple[str, str]] = {}
        CHUNK = 500
        hashes = list(hash_to_key.keys())
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT key_hash, verdict, reasoning FROM llm_judgments "
                f"WHERE key_hash IN ({placeholders})",
                chunk,
            )
            for key_hash, verdict, reasoning in cur.fetchall():
                found[hash_to_key[key_hash]] = (verdict, reasoning)
        return found

    def set_llm_judgments(self, judgments_by_cache_key: Dict[str, Tuple[str, str]],
                           cost_by_cache_key: Optional[Dict[str, float]] = None) -> None:
        """Store verdicts, optionally with the per-pair cost (USD) paid to
        compute them -- see trek_similarity.run_llm_judge_stage()'s
        cost_cache parameter, which splits a batch call's real reported
        cost evenly across the pairs in that batch. Used to reconstruct
        the REAL historical cost of a cache hit later (see
        get_llm_judgment_costs()), not just an estimate."""
        if not judgments_by_cache_key:
            return
        cost_by_cache_key = cost_by_cache_key or {}
        updated_at = _now_iso()
        rows = [
            (self.text_hash(key), verdict, reasoning, cost_by_cache_key.get(key, 0.0), updated_at)
            for key, (verdict, reasoning) in judgments_by_cache_key.items()
        ]
        self._conn.executemany(
            """
            INSERT INTO llm_judgments (key_hash, verdict, reasoning, cost_usd, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key_hash) DO UPDATE SET
                verdict = excluded.verdict,
                reasoning = excluded.reasoning,
                cost_usd = excluded.cost_usd,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self._commit()

    def get_llm_judgment_costs(self, cache_keys: List[str]) -> Dict[str, float]:
        """Return {cache_key: cost_usd} for whichever of the given cache
        keys are cached -- the REAL cost recorded when that pair was
        first judged, so a run served entirely from cache can still
        report the true historical cost of what it's showing."""
        if not cache_keys:
            return {}
        hash_to_key = {self.text_hash(k): k for k in cache_keys}
        found: Dict[str, float] = {}
        CHUNK = 500
        hashes = list(hash_to_key.keys())
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT key_hash, cost_usd FROM llm_judgments "
                f"WHERE key_hash IN ({placeholders})",
                chunk,
            )
            for key_hash, cost_usd in cur.fetchall():
                found[hash_to_key[key_hash]] = cost_usd
        return found

    def llm_judgments_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM llm_judgments")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Persisted 'Check Duplicates' run results (one entry per SYT module --
    # the latest run overwrites the previous one). Lets the GUI browse
    # past duplicate-check results/metrics per module without re-running
    # anything -- see trek_gui.CachedDuplicateResultsDialog.
    # ------------------------------------------------------------------
    def set_duplicate_check_result(self, module_key: str, syt_module: str, result_dict: dict) -> str:
        """Store a full trek_similarity.DuplicateCheckResult.to_dict() payload
        under ``module_key`` (see key_duplicate_check()), overwriting any
        previous run for that module. Returns the updated_at timestamp.

        The heavy result_json is written to the separate
        duplicate_check_data table, while only the lightweight metadata
        (pair_count, llm_model, syt_module, updated_at) goes into
        duplicate_check_runs -- so list_duplicate_check_modules() never
        touches the multi-MB blobs."""
        updated_at = _now_iso()
        pair_count = len(result_dict.get("pairs", []))
        llm_model = result_dict.get("llm_model", "")
        stats_json = json.dumps(compute_dupcheck_stats(result_dict))
        result_json_str = _fast_dumps(result_dict)
        # Compress the JSON blob with zlib -- duplicate check results for
        # large modules are 50-100+ MB of JSON text (each pair carries full
        # test case content: syt_text, swt_text, syt_tc, swt_tc). JSON text
        # compresses extremely well (~10-15x) because test case text is
        # highly repetitive. The compressed blob transfers over a network
        # share in seconds instead of minutes -- the dominant bottleneck
        # was always the SMB read, not parsing or object creation.
        compressed = zlib.compress(result_json_str.encode("utf-8"), level=6)
        # Lightweight metadata row (small pages, fast to list).
        self._conn.execute(
            """
            INSERT INTO duplicate_check_runs
                (module_key, syt_module, result_json, updated_at, pair_count, llm_model, stats_json)
            VALUES (?, ?, '', ?, ?, ?, ?)
            ON CONFLICT(module_key) DO UPDATE SET
                syt_module = excluded.syt_module,
                result_json = '',
                updated_at = excluded.updated_at,
                pair_count = excluded.pair_count,
                llm_model = excluded.llm_model,
                stats_json = excluded.stats_json
            """,
            (module_key, syt_module, updated_at, pair_count, llm_model, stats_json),
        )
        # Heavy blob in its own table. Store as BLOB (compressed bytes)
        # in the result_json column -- the column name is historical; new
        # rows contain zlib-compressed bytes, old rows still contain plain
        # JSON text. get_duplicate_check_result() detects which format.
        self._conn.execute(
            """
            INSERT INTO duplicate_check_data (module_key, result_json)
            VALUES (?, ?)
            ON CONFLICT(module_key) DO UPDATE SET
                result_json = excluded.result_json
            """,
            (module_key, sqlite3.Binary(compressed)),
        )
        self._commit()
        return updated_at

    def get_duplicate_check_result(self, module_key: str) -> Optional[dict]:
        """Return the result_dict last stored under ``module_key``, or None
        if that module has no cached 'Check Duplicates' run.

        Reads from the separate duplicate_check_data table (where the
        heavy JSON blob lives), falling back to the old result_json column
        in duplicate_check_runs for databases that haven't been migrated
        yet by _migrate_split_dupcheck_data.

        Handles two storage formats:
          - NEW: zlib-compressed bytes (written by set_duplicate_check_result
            after the compression change). ~10-15x smaller than raw JSON,
            so reads over SMB finish in seconds instead of minutes.
          - OLD: plain JSON text (written by older versions). Still works
            transparently -- detected by checking if the data is bytes."""
        # Try the new split table first (post-migration path).
        cur = self._conn.execute(
            "SELECT result_json FROM duplicate_check_data WHERE module_key = ?", (module_key,)
        )
        row = cur.fetchone()
        if row and row[0]:
            raw = row[0]
            try:
                if isinstance(raw, bytes):
                    # New format: zlib-compressed bytes.
                    return _fast_loads(zlib.decompress(raw))
                else:
                    # Old format: plain JSON text. Parse it, then compress
                    # and re-save so next read is fast (opportunistic
                    # migration -- avoids a blocking startup migration).
                    parsed = _fast_loads(raw)
                    try:
                        compressed = zlib.compress(raw.encode("utf-8"), level=6)
                        self._conn.execute(
                            "UPDATE duplicate_check_data SET result_json = ? WHERE module_key = ?",
                            (sqlite3.Binary(compressed), module_key),
                        )
                        self._commit()
                    except Exception:
                        pass  # best-effort compression; never break the read
                    return parsed
            except (json.JSONDecodeError, ValueError, zlib.error):
                pass
        # Fallback: old table (pre-migration, or migration hasn't run yet).
        cur = self._conn.execute(
            "SELECT result_json FROM duplicate_check_runs WHERE module_key = ?", (module_key,)
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            return _fast_loads(row[0])
        except (json.JSONDecodeError, ValueError):
            return None

    def list_duplicate_check_modules(self) -> List[Dict[str, Any]]:
        """Return one row per module with a cached 'Check Duplicates' run:
        {module_key, syt_module, pair_count, llm_model, updated_at}, newest
        first -- used to populate the module switcher in
        trek_gui.CachedDuplicateResultsDialog.

        Reads pair_count / llm_model from dedicated summary columns
        (populated by set_duplicate_check_result / _migrate_add_dupcheck_summary_columns)
        so this query NEVER touches the large result_json blob -- critical
        for network-share caches where listing every module would otherwise
        pull hundreds of MB of JSON just to populate a dropdown."""
        cur = self._conn.execute(
            "SELECT module_key, syt_module, pair_count, llm_model, updated_at, "
            "       COALESCE(stats_json, '') "
            "FROM duplicate_check_runs ORDER BY updated_at DESC"
        )
        rows = []
        for module_key, syt_module, pair_count, llm_model, updated_at, stats_json in cur.fetchall():
            stats = None
            if stats_json:
                try:
                    stats = json.loads(stats_json)
                except (json.JSONDecodeError, TypeError):
                    stats = None
            rows.append({
                "module_key": module_key,
                "syt_module": syt_module,
                "pair_count": pair_count,
                "llm_model": llm_model or "",
                "updated_at": updated_at,
                "stats": stats,
            })
        return rows

    def ensure_dupcheck_stats(self, module_key: str) -> Optional[dict]:
        """Return the per-module count matrix, computing and storing it once
        for results saved by an older version (which have no stats yet)."""
        row = self._conn.execute(
            "SELECT COALESCE(stats_json, '') FROM duplicate_check_runs WHERE module_key = ?",
            (module_key,)).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass
        data = self.get_duplicate_check_result(module_key)      # heavy, once per module
        if data is None:
            return None
        stats = compute_dupcheck_stats(data)
        self._conn.execute(
            "UPDATE duplicate_check_runs SET stats_json = ? WHERE module_key = ?",
            (json.dumps(stats), module_key))
        self._commit()
        return stats

    def delete_duplicate_check_result(self, module_key: str) -> None:
        self._conn.execute("DELETE FROM duplicate_check_runs WHERE module_key = ?", (module_key,))
        self._conn.execute("DELETE FROM duplicate_check_data WHERE module_key = ?", (module_key,))
        self._commit()

    # ------------------------------------------------------------------
    # App-wide operation statistics (append-only -- unlike every other
    # table above, a row is never overwritten, so this is the one place
    # that lets the "App Report" dialog compute TRUE historical
    # averages/totals across every run ever performed, not just the
    # latest one per module). One row per completed operation instance:
    # a module-list fetch, a traceability build, or a duplicate-check
    # RAG/LLM stage -- see trek_gui.py's call sites for op_type values.
    # ------------------------------------------------------------------
    def record_operation_stat(self, op_type: str, module: Optional[str] = None,
                               item_count: int = 0, extra_count: int = 0,
                               duration_seconds: float = 0.0, cost_usd: float = 0.0,
                               source: Optional[str] = None, details: Optional[dict] = None) -> None:
        """``details`` holds op-type-specific structured extras (e.g.
        fetch_traceability's {syr_count, swr_count, swt_count,
        related_syt_count} breakdown) that don't fit the generic
        item_count/extra_count columns shared by every op_type."""
        self._conn.execute(
            """
            INSERT INTO operation_stats
                (op_type, module, item_count, extra_count, duration_seconds, cost_usd, source, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (op_type, module, item_count, extra_count, duration_seconds, cost_usd, source,
             json.dumps(details) if details else None, _now_iso()),
        )
        self._commit()

    def get_operation_stats(self, op_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return every recorded operation row (optionally filtered to one
        op_type), oldest first. ``details`` is the parsed details_json
        dict (empty dict if none was recorded)."""
        cols = ["op_type", "module", "item_count", "extra_count",
                "duration_seconds", "cost_usd", "source", "details_json", "created_at"]
        if op_type:
            cur = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM operation_stats WHERE op_type = ? ORDER BY created_at",
                (op_type,),
            )
        else:
            cur = self._conn.execute(f"SELECT {', '.join(cols)} FROM operation_stats ORDER BY created_at")
        rows = []
        for values in cur.fetchall():
            row = dict(zip(cols, values))
            details_json = row.pop("details_json")
            try:
                row["details"] = json.loads(details_json) if details_json else {}
            except json.JSONDecodeError:
                row["details"] = {}
            rows.append(row)
        return rows

    def get_operation_summary(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate every recorded operation by op_type only (no module
        breakdown) -- used for module-independent op types like
        fetch_modules (the global module LIST, not scoped to one module)."""
        summary: Dict[str, Dict[str, Any]] = {}
        for row in self.get_operation_stats():
            bucket = summary.setdefault(row["op_type"], {
                "count": 0, "total_seconds": 0.0, "total_items": 0, "total_cost_usd": 0.0,
            })
            bucket["count"] += 1
            bucket["total_seconds"] += row["duration_seconds"]
            bucket["total_items"] += row["item_count"]
            bucket["total_cost_usd"] += row["cost_usd"]
        for bucket in summary.values():
            bucket["avg_seconds"] = bucket["total_seconds"] / bucket["count"] if bucket["count"] else 0.0
            bucket["avg_seconds_per_item"] = (
                bucket["total_seconds"] / bucket["total_items"] if bucket["total_items"] else 0.0
            )
        return summary

    def get_operation_summary_by_module(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Aggregate every recorded operation by (module, op_type): run
        count, total/avg duration, avg duration per item, total cost, plus
        the most recently recorded row's extra_count/details/source (so
        the 'App Report' dialog can show e.g. the latest SYR/SWR/SWT/
        related-SYT breakdown alongside the historical averages). Rows
        with no module (e.g. fetch_modules, which fetches the global
        module list, not one specific module) are grouped under the
        ``None`` key.
        """
        by_module: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in self.get_operation_stats():
            module_bucket = by_module.setdefault(row["module"], {})
            bucket = module_bucket.setdefault(row["op_type"], {
                "count": 0, "total_seconds": 0.0, "total_items": 0, "total_cost_usd": 0.0,
                "last_extra_count": 0, "last_details": {}, "last_source": None, "last_created_at": None,
            })
            bucket["count"] += 1
            bucket["total_seconds"] += row["duration_seconds"]
            bucket["total_items"] += row["item_count"]
            bucket["total_cost_usd"] += row["cost_usd"]
            # Rows come back oldest-first, so the last one seen is the latest.
            bucket["last_extra_count"] = row["extra_count"]
            bucket["last_details"] = row["details"]
            bucket["last_source"] = row["source"]
            bucket["last_created_at"] = row["created_at"]
        for module_bucket in by_module.values():
            for bucket in module_bucket.values():
                bucket["avg_seconds"] = bucket["total_seconds"] / bucket["count"] if bucket["count"] else 0.0
                bucket["avg_seconds_per_item"] = (
                    bucket["total_seconds"] / bucket["total_items"] if bucket["total_items"] else 0.0
                )
        return by_module

    def clear_operation_stats(self) -> None:
        self._conn.execute("DELETE FROM operation_stats")
        self._commit()

    # ------------------------------------------------------------------
    # TC content cache (global, keyed by TC id)
    # ------------------------------------------------------------------
    def get_tc_contents(self, tc_ids: List[str]) -> Dict[str, dict]:
        """Return {tc_id: content_dict} for whichever of ``tc_ids`` are
        already cached. Missing IDs are simply absent from the result."""
        if not tc_ids:
            return {}
        found: Dict[str, dict] = {}
        # SQLite has a default limit of 999 host params; chunk defensively.
        CHUNK = 500
        for i in range(0, len(tc_ids), CHUNK):
            chunk = tc_ids[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT tc_id, content_json FROM tc_content "
                f"WHERE tc_id IN ({placeholders})",
                chunk,
            )
            for tc_id, content_json in cur.fetchall():
                try:
                    found[tc_id] = _fast_loads(content_json)
                except json.JSONDecodeError:
                    continue
        return found

    def cached_tc_id_set(self, tc_ids: List[str]) -> set:
        """Return the subset of ``tc_ids`` that already have content cached.
        Reads only the primary-key column (never the JSON payload), so it
        stays cheap even for tens of thousands of ids -- used by the
        offline "Download all test-case text" feature to skip what is
        already saved."""
        if not tc_ids:
            return set()
        present: set = set()
        CHUNK = 500
        ids = list(tc_ids)
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT tc_id FROM tc_content WHERE tc_id IN ({placeholders})",
                chunk,
            )
            present.update(r[0] for r in cur.fetchall())
        return present

    def set_tc_contents(self, content_map: Dict[str, dict]) -> None:
        if not content_map:
            return
        updated_at = _now_iso()
        rows = [
            (tc_id, json.dumps(content), updated_at)
            for tc_id, content in content_map.items()
        ]
        self._conn.executemany(
            """
            INSERT INTO tc_content (tc_id, content_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(tc_id) DO UPDATE SET
                content_json = excluded.content_json,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self._commit()

    def tc_content_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM tc_content")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Introspection (for the "View Database" dialog)
    # ------------------------------------------------------------------
    def list_blob_entries(self) -> List[Dict[str, Any]]:
        """Return one row per cached blob key: {key, kind, item_count,
        updated_at}. `kind` is the first colon-separated segment of the key
        (e.g. 'modules', 'links') and `item_count` is len(value) when the
        stored value is a list, else 1.

        Uses the pre-computed item_count column when available (>= 0).
        For pre-migration rows (item_count == -1) the count is shown as 0
        rather than parsing multi-MB JSON blobs over the network -- it will
        be populated automatically on the next set_blob() call."""
        cur = self._conn.execute(
            "SELECT key, item_count, updated_at FROM blobs ORDER BY updated_at DESC"
        )
        rows = []
        for key, stored_ic, updated_at in cur.fetchall():
            kind = key.split(":", 1)[0]
            rows.append({
                "key": key,
                "kind": kind,
                "item_count": stored_ic if stored_ic >= 0 else 0,
                "updated_at": updated_at,
            })
        return rows

    def summary(self) -> Dict[str, Any]:
        """Aggregate counts grouped by blob key 'kind' (modules/links/...),
        plus the tc_content row count and the most recent update time
        across the whole cache. Used to render the "View Database" dialog
        header without listing every single row."""
        entries = self.list_blob_entries()
        by_kind: Dict[str, Dict[str, Any]] = {}
        latest = None
        for e in entries:
            k = e["kind"]
            bucket = by_kind.setdefault(k, {"kind": k, "key_count": 0, "item_count": 0, "latest": None})
            bucket["key_count"] += 1
            bucket["item_count"] += e["item_count"]
            if bucket["latest"] is None or e["updated_at"] > bucket["latest"]:
                bucket["latest"] = e["updated_at"]
            if latest is None or e["updated_at"] > latest:
                latest = e["updated_at"]

        tc_count = self.tc_content_count()
        cur = self._conn.execute("SELECT MAX(updated_at) FROM tc_content")
        tc_latest = cur.fetchone()[0]
        if tc_latest and (latest is None or tc_latest > latest):
            latest = tc_latest

        try:
            file_size_bytes = self.db_path.stat().st_size
        except OSError:
            file_size_bytes = 0

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE vector_blob IS NULL AND vector_json != ''"
        )
        unoptimized_embeddings = cur.fetchone()[0]

        return {
            "db_path": str(self.db_path),
            "blob_kinds": sorted(by_kind.values(), key=lambda b: b["kind"]),
            "tc_content_count": tc_count,
            "tc_content_latest": tc_latest,
            "latest_overall": latest,
            "file_size_bytes": file_size_bytes,
            "embeddings_count": self.embeddings_count(),
            "unoptimized_embeddings": unoptimized_embeddings,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        """Wipe EVERY table -- blobs, TC content, embeddings, LLM
        judgments, persisted duplicate-check results, and operation
        stats. Used by the "Delete All Cache Data" button for a full
        reset, e.g. when the underlying TREK data has changed enough
        that stale cached data would be actively misleading."""
        for table in ("blobs", "tc_content", "embeddings", "llm_judgments",
                                             "duplicate_check_runs", "duplicate_check_data", "operation_stats"):
            self._conn.execute(f"DELETE FROM {table}")
        self._commit()

    def optimize_storage(self, progress_cb: Optional[Callable[[int, int], None]] = None,
                         page_size: int = NETWORK_PAGE_SIZE,
                         drop_error_judgments: bool = False) -> Dict[str, Any]:
        """One-time maintenance: convert any legacy JSON-only embedding rows
        (written before binary vector storage existed) to the compact
        vector_blob format, then VACUUM the database file to physically
        reclaim the freed space.

        This is the fix for a cache that has grown large (e.g. several GB)
        mostly because of embeddings stored as JSON text -- see
        _pack_vector()'s docstring for the ~4-5x size difference. Safe to
        call on an already-optimized cache (it will just do nothing, then
        VACUUM). Safe to call repeatedly / interrupt-and-retry: each row
        conversion is committed in its own batch.

        Args:
            progress_cb: optional callback(done_count, total_count) invoked
                         after each batch, so a GUI progress dialog can show
                         movement on multi-GB caches with millions of rows.

        Returns a dict with before/after file sizes and how many rows were
        converted, for a confirmation message to the user.
        """
        try:
            size_before = self.db_path.stat().st_size
        except OSError:
            size_before = 0

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE vector_blob IS NULL AND vector_json != ''"
        )
        total = cur.fetchone()[0]
        converted = 0

        if total:
            CHUNK = 500
            while True:
                rows = self._conn.execute(
                    "SELECT text_hash, model, vector_json FROM embeddings "
                    "WHERE vector_blob IS NULL AND vector_json != '' LIMIT ?",
                    (CHUNK,),
                ).fetchall()
                if not rows:
                    break
                updates = []
                for text_hash, model, vector_json in rows:
                    try:
                        vector = json.loads(vector_json)
                        updates.append((_pack_vector(vector), text_hash, model))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # Corrupt row -- clear vector_json so it stops being
                        # picked up by this loop forever; the (now empty)
                        # entry will simply miss on next lookup and get
                        # re-embedded on demand.
                        updates.append((None, text_hash, model))
                self._conn.executemany(
                    "UPDATE embeddings SET vector_blob = ?, vector_json = '' "
                    "WHERE text_hash = ? AND model = ?",
                    updates,
                )
                self._commit()
                converted += len(rows)
                if progress_cb:
                    progress_cb(converted, total)

        stats = self._compress_large_values(drop_error_judgments=drop_error_judgments,
                                            progress_cb=progress_cb)

        # VACUUM physically rewrites the file to reclaim space freed by the
        # UPDATEs above (SQLite doesn't shrink the file on its own). This
        # requires roughly as much free space as the database currently
        # uses and an exclusive lock, so it can take a while on a multi-GB
        # network-share file -- but it's a one-time cost. It is also the
        # only moment the page size can change (not allowed in WAL mode, so
        # switch to DELETE for the duration and restore afterwards).
        old_page = self._conn.execute("PRAGMA page_size;").fetchone()[0]
        change_page = bool(page_size) and old_page != page_size
        if change_page:
            self._conn.execute("PRAGMA journal_mode=DELETE;")
            self._conn.execute(f"PRAGMA page_size={int(page_size)};")
        self._conn.execute("PRAGMA temp_store=FILE;")     # never build a GB-sized temp copy in RAM
        try:
            self._conn.execute("VACUUM;")
        finally:
            self._conn.execute("PRAGMA temp_store=MEMORY;")
            if change_page and not self._is_network:
                try:
                    self.journal_mode = self._conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
                except sqlite3.Error:
                    pass
        self._blob_cache.clear()

        try:
            size_after = self.db_path.stat().st_size
        except OSError:
            size_after = size_before

        return {
            "converted_rows": converted,
            "size_before_bytes": size_before,
            "size_after_bytes": size_after,
            "page_size_before": old_page,
            "page_size_after": self._conn.execute("PRAGMA page_size;").fetchone()[0],
            **stats,
        }

    def _compress_large_values(self, drop_error_judgments: bool = False,
                               progress_cb: Optional[Callable[[int, int], None]] = None) -> Dict[str, int]:
        """Rewrite large plain-text values in the compact compressed format
        (fewer pages = fewer network round trips): blob values >= 8 KB, and
        'Check Duplicates' results still stored as plain JSON text by old
        versions. Optionally drop cached LLM verdicts of type 'error' (never
        reused -- the judge re-asks for them anyway). Committed in batches."""
        out = {"blobs_compressed": 0, "results_compressed": 0, "error_judgments_dropped": 0}
        keys = [r[0] for r in self._conn.execute(
            "SELECT key FROM blobs WHERE typeof(value_json) = 'text' AND length(value_json) >= ?",
            (_BLOB_COMPRESS_MIN,)).fetchall()]
        for i, key in enumerate(keys, 1):
            (text,) = self._conn.execute("SELECT value_json FROM blobs WHERE key = ?", (key,)).fetchone()
            self._conn.execute("UPDATE blobs SET value_json = ? WHERE key = ?",
                               (_encode_blob_text(text), key))
            if i % 50 == 0:
                self._commit()
                if progress_cb:
                    progress_cb(i, len(keys))
        self._commit()
        out["blobs_compressed"] = len(keys)

        mkeys = [r[0] for r in self._conn.execute(
            "SELECT module_key FROM duplicate_check_data WHERE typeof(result_json) = 'text'").fetchall()]
        for mk in mkeys:
            (text,) = self._conn.execute(
                "SELECT result_json FROM duplicate_check_data WHERE module_key = ?", (mk,)).fetchone()
            self._conn.execute("UPDATE duplicate_check_data SET result_json = ? WHERE module_key = ?",
                               (sqlite3.Binary(zlib.compress(text.encode("utf-8"), level=6)), mk))
            self._commit()
        out["results_compressed"] = len(mkeys)

        if drop_error_judgments:
            cur = self._conn.execute("DELETE FROM llm_judgments WHERE verdict = 'error'")
            out["error_judgments_dropped"] = cur.rowcount or 0
            self._commit()
        return out

    def export_optimized_copy(self, dst_path, page_size: int = NETWORK_PAGE_SIZE,
                              drop_error_judgments: bool = False,
                              progress_cb: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Write a network-share-ready COPY of this database to ``dst_path``:
        large values compressed (done in this database too, format-compatible),
        64 KB pages, no free space, rollback journal (DELETE) mode. The copy
        is verified (quick_check + row counts) before returning. The
        original file is not otherwise changed. ``dst_path`` must not exist."""
        dst = Path(dst_path)
        if dst.exists():
            raise FileExistsError(f"{dst} already exists -- choose a new file name.")
        try:
            size_before = self.db_path.stat().st_size
        except OSError:
            size_before = 0
        stats = self._compress_large_values(drop_error_judgments=drop_error_judgments,
                                            progress_cb=progress_cb)
        self._conn.execute(f"PRAGMA page_size={int(page_size)};")   # applies to VACUUM INTO
        self._conn.execute("VACUUM INTO ?;", (str(dst),))

        check = sqlite3.connect(str(dst))
        try:
            check.execute("PRAGMA journal_mode=DELETE;")
            ok = check.execute("PRAGMA quick_check;").fetchone()[0]
            tables = [r[0] for r in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            mismatch = []
            for t in tables:
                a = self._conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                b = check.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                if a != b:
                    mismatch.append(f"{t}: {a} vs {b}")
            new_page = check.execute("PRAGMA page_size;").fetchone()[0]
        finally:
            check.close()
        if ok != "ok" or mismatch:
            raise RuntimeError(f"Verification of the copy failed: {ok}; {mismatch}")
        return {"size_before_bytes": size_before, "size_after_bytes": dst.stat().st_size,
                "page_size_after": new_page, "path": str(dst), **stats}

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Key builders -- centralised so trek_gui.py never hand-rolls a cache key.
# ---------------------------------------------------------------------------
def key_modules(project_id: int, config_id: int) -> str:
    return f"modules:{project_id}:{config_id}"


def key_links(project_id: int, campaign_id: int, module_name: str, domain_type: int) -> str:
    """Raw ``Export/Links`` item list for one module at one
    ``local_config_domain_type``.

    This single generic key covers every raw link fetch in the app because
    the same (module, domain_type) call is reused for several purposes:
      - SYT module,  domain 4 -> TC-id extraction (Step 2) AND SYT->SYR
        edges (Step 3) share the exact same raw response.
      - SWR module,  domain 2 -> SWR-candidate probing (Step 2) AND
        SYR->SWR bridge edges (Step 3) share the exact same raw response.
      - SWT module,  domain 4 -> SWR->SWT edges (Step 3).
    Caching by (module, domain_type) therefore de-duplicates network calls
    across steps, not just across GUI sessions.
    """
    return f"links:{project_id}:{campaign_id}:{module_name}:{domain_type}"


def key_requirements(project_id: int, campaign_id: int, module_name: str) -> str:
    """Full ``Export/Requirements`` item list for one module (e.g.
    ``SYR - Infrastructure``). Requirement content is cached per-module the
    same way link items are, rather than per-ID like TC content, since
    fetching a whole module's requirements is one API call regardless of
    how many individual requirement IDs are actually referenced by the
    current traceability run.
    """
    return f"requirements:{project_id}:{campaign_id}:{module_name}"


def key_bridge_map(project_id: int, campaign_id: int, source_module: str, target_kind: str) -> str:
    """Cached list of AUTO-DETECTED downstream bridge module name(s) for
    one upstream module -- e.g. which SWR module(s) a given SYR module's
    own links actually point into (see trek_gui._get_bridge_modules_cached()).

    ``target_kind`` (e.g. "SYR", "SWR", "SWT") disambiguates the target
    level being bridged to, since a single source module can bridge to
    more than one kind of downstream module -- most notably a SYT module
    bridges to BOTH its SYR requirement module (target_kind="SYR") and its
    SWT/SWIT test module(s) (target_kind="SWT") independently.

    Why this needs its own persistent, per-source-module cache entry:
    correctly identifying which downstream module a source module bridges
    to requires cross-referencing that source module's full link export
    against every candidate module at the target level (potentially
    dozens of live API probes) -- and, critically, the CORRECT answer
    depends only on the source module's own content, never on which
    specific test cases happen to be selected in any particular
    traceability run. Caching it here means the detection only ever runs
    ONCE per source module for the lifetime of the cache (or until the
    user forces a refresh), instead of being re-derived from a
    run-specific (and potentially small/unrepresentative) subset every
    single time -- which is what previously allowed a handful of stray
    cross-references from an unrelated functional area to occasionally
    outvote the genuine bridge module when only a few test cases were
    selected.

    See also key_bridge_map_manual() -- a separate, higher-priority cache
    entry for mappings the USER has explicitly confirmed/corrected via the
    ModuleMappingDialog, for cases where automatic detection (name-based
    or ID-token-based) cannot reliably guess the right answer at all (e.g.
    a domain-specific ID abbreviation like "RWW" for "Rear Wiper" that has
    no textual relationship to the module's display name).
    """
    return f"bridge_map:{project_id}:{campaign_id}:{source_module}:{target_kind}"


def key_bridge_map_manual(project_id: int, campaign_id: int, source_module: str, target_kind: str) -> str:
    """User-confirmed override for key_bridge_map() -- see that function's
    docstring. Always checked FIRST and takes priority over the
    auto-detected cache entry when present; never invalidated by a
    'Force Refresh' (auto-detection may need to re-run against fresh
    data, but a manually-confirmed mapping is a deliberate user decision
    that should persist until the user explicitly edits it again via the
    mapping dialog).
    """
    return f"bridge_map_manual:{project_id}:{campaign_id}:{source_module}:{target_kind}"


def key_duplicate_check(project_id: int, campaign_id: int, syt_module: str, llm_model: str = "") -> str:
    """Key for one SYT module's persisted 'Check Duplicates' run (see
    set_duplicate_check_result()/get_duplicate_check_result()) -- scoped
    per project/campaign so the same module name in a different
    project/campaign never collides, AND per ``llm_model`` (falling back
    to "no-llm" when the LLM-judge stage wasn't used) so running the same
    module with several different LLM models keeps a SEPARATE stored
    version per model instead of the newest model overwriting the others
    -- lets the user compare how different models judged the same
    module. Re-running the SAME model (regardless of which/how-many pairs
    were included in that particular run) still overwrites that model's
    own version, since the key doesn't depend on the pair set.
    """
    return f"dupcheck:{project_id}:{campaign_id}:{syt_module}:{llm_model or 'no-llm'}"


def format_age(updated_at_iso: str) -> str:
    """Turn an ISO timestamp into a short human string like '3m ago',
    '2h ago', '5d ago'. Falls back to the raw string on parse failure."""
    try:
        then = datetime.datetime.fromisoformat(updated_at_iso)
    except ValueError:
        return updated_at_iso
    delta = datetime.datetime.now() - then
    secs = int(delta.total_seconds())
    if secs < 5:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"
