"""
TREK Project Store
==================
Small JSON-backed store for saved TREK project configurations (Project ID /
Campaign ID / Config ID), so trek_gui.py no longer hardcodes a single
project. Lets the user configure one or more projects on first run, switch
between them, and have the choice persist across app restarts.

Each project entry ALSO carries a JWT Token for the LLM embedding gateway
(entered once at project setup time, see ProjectSetupDialog in trek_gui.py)
used by trek_similarity.py for the SYT/SWT duplicate-detection feature.
This is MANDATORY, not optional: discovering duplicate test cases is this
application's core purpose, so every project requires a valid JWT Token to
be saved at all. The gateway URL itself is hardcoded in trek_similarity.py
(LLM_GATEWAY_URL) since there is only ever one gateway to talk to.

Similarity thresholds and BM25/vector weighting are intentionally NOT
stored per-project here -- the user sets those interactively each time
they run "Check Duplicates" (see DuplicateCheckSettingsDialog in
trek_gui.py), since sensitivity may reasonably differ per run/module
rather than being a fixed project-level setting.

Each project MAY ALSO carry ``syt_prefixes``: a comma-separated string of
module-name prefixes (default "SYT") used by Step 1's module filter (see
trek_gui._is_syt_module). Some TREK projects name their system-test
modules "SYT - <Subsystem>", others use a different convention entirely
(e.g. a colleague's project uses "SYTS_..."). Rather than hardcode a
single prefix, the Step 1 panel exposes an editable "Module prefix(es)"
field (see TrekMainWindow._make_step1_panel / _on_prefix_changed) that
persists here per-project so it's remembered next time this project is
loaded, without touching ProjectSetupDialog's other required fields.

Each project MAY ALSO carry a custom ``db_path`` pointing at a
trek_cache.sqlite3 file to use for that project instead of the app-wide
default (see trek_paths.data_file / trek_cache.DEFAULT_DB_PATH). This
lets a project's cached TREK data + duplicate-check history live on a
shared network location so a team collaborates through one cache file,
while other projects keep using the default local cache -- see
trek_gui.ProjectSetupDialog and trek_gui._apply_active_project(), which
re-points the global CACHE handle at this path whenever the active
project changes. An empty/missing db_path means "use the default
location."

File format (trek_projects.json, next to this script):
    {
        "active": "<project key>",
        "projects": {
            "<project key>": {
                "name": "BMW ZIM Rear",
                "project_id": 607,
                "campaign_id": 102766584,
                "config_id": 8579,
                "jwt_token": "...",  # mandatory
                "db_path": "",       # optional, "" = default cache location
                "syt_prefixes": "SYT"  # optional, "" falls back to "SYT"
            },
            ...
        }
    }

The "key" is an internal identifier (derived from the name, de-duplicated)
used to reference a project without repeating its full config; it is not
shown to the user directly (the "name" is what's displayed).

SECURITY NOTE: jwt_token is stored in plain text in this JSON file, same
as other Zeus config files (e.g. ~/.zeus/zeus.json). trek_projects.json is
already gitignored (per-machine state, never committed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import trek_paths

DEFAULT_PROJECTS_PATH = trek_paths.data_file("trek_projects.json")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "project"


class TrekProjectStore:
    """Loads/saves trek_projects.json and provides CRUD + active-project
    helpers. All state is kept in memory and written back to disk on every
    mutating call, so callers never need to explicitly "save"."""

    def __init__(self, path: Path = DEFAULT_PROJECTS_PATH):
        self.path = Path(path)
        self._data: Dict = {"active": None, "projects": {}}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "projects" in data:
                    self._data = data
            except (json.JSONDecodeError, OSError):
                # Corrupt/unreadable file -- start fresh rather than crash
                # the whole app; the user will be prompted to (re)configure.
                self._data = {"active": None, "projects": {}}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def has_projects(self) -> bool:
        return bool(self._data.get("projects"))

    def list_projects(self) -> List[dict]:
        """Return all saved projects as a list of dicts, each including its
        internal 'key', sorted by name for stable UI ordering."""
        items = []
        for key, proj in self._data.get("projects", {}).items():
            items.append({"key": key, **proj})
        return sorted(items, key=lambda p: p.get("name", "").lower())

    def get_active(self) -> Optional[dict]:
        """Return the active project dict (including its 'key'), or None
        if no project is configured yet / the stored active key is stale."""
        key = self._data.get("active")
        proj = self._data.get("projects", {}).get(key)
        if proj is None:
            return None
        return {"key": key, **proj}

    def get(self, key: str) -> Optional[dict]:
        proj = self._data.get("projects", {}).get(key)
        return {"key": key, **proj} if proj is not None else None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_project(self, name: str, project_id: int, campaign_id: int,
                     config_id: int, jwt_token: str, make_active: bool = True,
                     db_path: str = "") -> str:
        """Add (or overwrite) a project entry. Returns its internal key.

        Args:
            db_path: Optional path to a trek_cache.sqlite3 file this
                     project should use instead of the app-wide default
                     (e.g. a shared network location). Empty string means
                     "use the default location" (see trek_paths.data_file).

        Raises:
            ValueError: if jwt_token is empty/blank -- a JWT Token is
                        mandatory for every project (see module docstring).
        """
        if not jwt_token or not jwt_token.strip():
            raise ValueError(
                "A JWT Token is required for every project (duplicate "
                "detection is this application's core purpose)."
            )

        base_key = _slugify(name)
        key = base_key
        n = 2
        projects = self._data.setdefault("projects", {})
        # Avoid clobbering a different existing project that happens to
        # slugify to the same key (e.g. two projects both named "Test").
        while key in projects and projects[key].get("name") != name:
            key = f"{base_key}_{n}"
            n += 1

        projects[key] = {
            "name": name,
            "project_id": int(project_id),
            "campaign_id": int(campaign_id),
            "config_id": int(config_id),
            "jwt_token": jwt_token,
            "db_path": (db_path or "").strip(),
        }
        if make_active or not self._data.get("active"):
            self._data["active"] = key
        self._save()
        return key

    def update_project(self, key: str, name: str, project_id: int,
                        campaign_id: int, config_id: int, jwt_token: str,
                        db_path: str = "") -> None:
        if key not in self._data.get("projects", {}):
            raise KeyError(f"Unknown project key: {key}")
        if not jwt_token or not jwt_token.strip():
            raise ValueError(
                "A JWT Token is required for every project (duplicate "
                "detection is this application's core purpose)."
            )
        self._data["projects"][key] = {
            "name": name,
            "project_id": int(project_id),
            "campaign_id": int(campaign_id),
            "config_id": int(config_id),
            "jwt_token": jwt_token,
            "db_path": (db_path or "").strip(),
        }
        self._save()

    def remove_project(self, key: str) -> None:
        self._data.get("projects", {}).pop(key, None)
        if self._data.get("active") == key:
            remaining = list(self._data.get("projects", {}).keys())
            self._data["active"] = remaining[0] if remaining else None
        self._save()

    def set_active(self, key: str) -> None:
        if key not in self._data.get("projects", {}):
            raise KeyError(f"Unknown project key: {key}")
        self._data["active"] = key
        self._save()

    # ------------------------------------------------------------------
    # LLM embedding gateway JWT token (per project, mandatory)
    # ------------------------------------------------------------------
    def get_jwt_token(self, key: str) -> str:
        """Return this project's JWT Token (empty string if somehow unset,
        e.g. a project saved before this field existed)."""
        proj = self._data.get("projects", {}).get(key, {})
        return proj.get("jwt_token", "")

    # ------------------------------------------------------------------
    # Custom cache database path (per project, optional)
    # ------------------------------------------------------------------
    def get_db_path(self, key: str) -> str:
        """Return this project's custom trek_cache.sqlite3 path, or "" if
        unset (meaning: use the app-wide default location). Empty string
        for projects saved before this field existed, same as get_jwt_token."""
        proj = self._data.get("projects", {}).get(key, {})
        return proj.get("db_path", "")

    # ------------------------------------------------------------------
    # Step 1 module-name prefix(es) (per project, optional)
    # ------------------------------------------------------------------
    def get_syt_prefixes(self, key: str) -> str:
        """Return this project's comma-separated Step 1 module-name
        prefix list (e.g. "SYT" or "SYT, SYTS"), or "SYT" if unset --
        matching the original hardcoded behaviour for projects saved
        before this field existed."""
        proj = self._data.get("projects", {}).get(key, {})
        return proj.get("syt_prefixes", "") or "SYT"

    def set_syt_prefixes(self, key: str, prefixes: str) -> None:
        """Persist this project's Step 1 module-name prefix list. Does
        nothing if the key is unknown (e.g. project was deleted mid-edit)."""
        proj = self._data.get("projects", {}).get(key)
        if proj is None:
            return
        proj["syt_prefixes"] = (prefixes or "").strip()
        self._save()
