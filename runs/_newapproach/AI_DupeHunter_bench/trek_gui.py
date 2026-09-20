"""
TREK Traceability GUI
=====================
PySide6 application for browsing SYT modules, selecting test cases,
viewing the SYT->SYR->SWR->SWT traceability chain, and exporting to JSON.

Layout:
  ┌────────────────────────────────────────────────────────────────────┐
  │  TREK Traceability                                   [header bar]  │
  ├──────────────┬─────────────────────────────────────────────────────┤
  │              │                                                     │
  │  STEP 1      │  STEP 2              STEP 3                         │
  │  Modules     │  Test Cases          Results                        │
  │              │                                                     │
  │  [list]      │  [table+checkboxes]  [splitter: tree | detail]      │
  │              │                                                     │
  │  [Load][⟳]   │  [All][None][Rand]   [ ] Force refresh              │
  │              │  [⟳ next to info]    [Run] [Export JSON]            │
  └──────────────┴─────────────────────────────────────────────────────┘

Local caching
-------------
Module lists, SYT/SWR/SWT link edges, and full test-case content are cached
in a local SQLite database (trek_cache.py -> trek_cache.sqlite3, next to
this script). Every fetch point in the UI has a plain "Load" action that
reads the cache when present (near-instant) and only calls the live TREK
API on a cache miss, plus an explicit "⟳ Refresh" / "Force refresh"
control that bypasses the cache and overwrites it with fresh data. See
trek_cache.py for the cache schema and key scheme.
"""

import sys
import re
import json
import random
import time
import datetime
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import trek_cache
from trek_cache import TrekCache
import trek_projects
import trek_similarity
import trek_log
import trek_paths
import trek_theme
import trek_index

from PySide6.QtCore import (
    Qt, QThread, Signal, QSortFilterProxyModel, QSize, QTimer, QElapsedTimer,
    QAbstractTableModel, QModelIndex, QEvent, QObject
)
from PySide6.QtGui import (
    QColor, QFont, QStandardItem, QStandardItemModel, QPalette, QIcon, QPainter
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QStatusBar, QFrame, QScrollArea, QSizePolicy,
    QCheckBox, QGroupBox, QMessageBox, QTabWidget, QAbstractItemView,
    QDialog, QComboBox, QFormLayout, QInputDialog, QTableView,
)
try:
    from PySide6.QtCharts import QChart, QChartView, QPieSeries
    _HAS_QTCHARTS = True
except ImportError:
    _HAS_QTCHARTS = False

# ---------------------------------------------------------------------------
# Load the TREK API client
# ---------------------------------------------------------------------------
_API_PATH = trek_paths.resource_file("tal", "KeywordDrivenBase", "Addons", "WorkspaceUpdate", "TrekExportLinksAPI.py")
_spec = importlib.util.spec_from_file_location("TrekExportLinksAPI", _API_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrekExportLinksClient = _mod.TrekExportLinksClient

PROJECT_STORE = trek_projects.TrekProjectStore()

# These three are intentionally mutable module globals (not a frozen
# constant) so that switching the active project at runtime -- via
# _apply_active_project() below -- takes effect everywhere they're
# referenced. Every function/method in this file that uses PROJECT_ID /
# CAMPAIGN_ID / CONFIG_ID reads them as a bare name resolved from this
# module's global namespace at call time (not captured by value at import
# time), so reassigning them here is sufficient to redirect every
# subsequent API call, cache key, and UI label to the newly active
# project -- no need to thread a "current project" parameter through 20+
# call sites. Real values are populated by _apply_active_project() once a
# project is configured (see main()); None here just means "not
# configured yet" during the brief window before the first-run dialog
# runs.
PROJECT_ID:  Optional[int] = None
CAMPAIGN_ID: Optional[int] = None
CONFIG_ID:   Optional[int] = None


def _apply_active_project(load_index: bool = True):
    """Refresh the PROJECT_ID / CAMPAIGN_ID / CONFIG_ID globals from
    PROJECT_STORE's currently active project. Called once at startup after
    the first-run setup dialog (if needed), and again every time the user
    switches projects via the header dropdown.

    Also re-points the global CACHE at this project's custom database path
    (ProjectSetupDialog's "Cache Database" field), if one is set -- falling
    back to the app-wide default location (trek_cache.DEFAULT_DB_PATH)
    otherwise. This lets one project's cache live on a shared network path
    while other projects keep using the local default, with zero changes
    needed anywhere else (every call site reads the bare CACHE name, so
    reassigning it here redirects every subsequent cache read/write)."""
    global PROJECT_ID, CAMPAIGN_ID, CONFIG_ID, INDEX, CACHE
    active = PROJECT_STORE.get_active()
    if active:
        PROJECT_ID  = active["project_id"]
        CAMPAIGN_ID = active["campaign_id"]
        CONFIG_ID   = active["config_id"]

        desired_db_path = Path(active.get("db_path") or trek_cache.DEFAULT_DB_PATH)
        if desired_db_path != CACHE.db_path:
            old_cache = CACHE
            try:
                CACHE = TrekCache(desired_db_path)
            except Exception as exc:
                LOG.log(
                    "Cache", f"Failed to open custom cache DB '{desired_db_path}': {exc} "
                    "-- falling back to default location.", level="ERROR",
                )
                CACHE = TrekCache(trek_cache.DEFAULT_DB_PATH)
            old_cache.close()

        # Rebuild the id->module INDEX handle for the newly active project
        # (and possibly-new CACHE) and load whatever is already cached for
        # it. resolve() lookups then replace the old live SWR candidate
        # probing (see FetchLinksWorker).
        INDEX = trek_index.TrekIndex(CACHE, PROJECT_ID, CAMPAIGN_ID, CONFIG_ID)
        # load_index=False: the caller loads it in the background instead
        # (TrekMainWindow._start_index_load) -- reading the multi-MB index
        # from a network share used to block startup for ~2 minutes.
        if load_index:
            INDEX.load()

# ---------------------------------------------------------------------------
# Local cache (SQLite, next to this script) -- avoids re-hitting TREK for
# data that rarely changes. See trek_cache.py. Every worker below accepts a
# `force_refresh` flag: False (default) reads from cache when present and
# only calls TREK on a cache miss; True always calls TREK and overwrites
# the cache with the fresh result.
# ---------------------------------------------------------------------------
CACHE = TrekCache()

# ---------------------------------------------------------------------------
# Object-id -> module INDEX (see trek_index.py). Lets traceability resolve
# which module an SWR/SYR id lives in with an O(1) lookup instead of probing
# candidate modules live. Rebuilt per active project by _apply_active_project();
# a placeholder handle is created here so references are always valid.
# ---------------------------------------------------------------------------
INDEX: "trek_index.TrekIndex" = trek_index.TrekIndex(CACHE, 0, 0, 0)

# Wire the API client into trek_index without it hard-importing the client.
trek_index.set_client_factory(lambda: TrekExportLinksClient())
trek_index.set_light_requirements_fn(
    lambda client, pid, cid, modules: client.get_requirements_keys_only(pid, cid, modules)
)


def _index_links_items(client, pid, cid, module, domain):
    """Return the raw /Export/Links items for one module (each with Key,
    LinkType, LinkKey). Used by trek_index to build BOTH:
      - the test-case id -> module map (from each item's Key), and
      - the reverse SWR-id -> SWT-module map (from each item's LinkKey that
        starts with 'SWR_', since a SWT test case owns the OUT link to its
        SWR requirement -- see FetchLinksWorker step 3).
    Reuses the app's cached link fetch so the index and traceability share
    the same cache entries (no double network cost)."""
    items, _from_cache = _get_links_items(client, module, domain, False, timeout=200)
    return items


trek_index.set_light_links_fn(
    lambda client, pid, cid, module, domain: _index_links_items(client, pid, cid, module, domain)
)

# ---------------------------------------------------------------------------
# Activity log (trek_log.py -> trek_activity.log, next to this script).
# Records every TREK API call, cache hit/miss, and worker stage with
# elapsed timing, so the "Show Log" viewer (header button) can show exactly
# what the app did and how long each step took, without having to watch
# the status bar live. Thread-safe -- safe to call LOG.log()/LOG.timed()
# from any worker QThread, including the parallelized fetch loops in
# FetchLinksWorker.
# ---------------------------------------------------------------------------
LOG = trek_log.TrekLog()

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# These module-level constants are read by 100+ inline f-string styles
# throughout this file at widget-construction time. They are now populated
# from the ACTIVE theme (see trek_theme.py) instead of being hardcoded, so
# the whole app can be re-skinned by choosing a different theme. The default
# theme ("Charcoal") reproduces the exact original palette, so nothing
# changes visually unless the user opts into another theme.
#
# _apply_theme_constants() (re)fills these from a theme colour dict; it is
# called once at import time (below) and again whenever the user switches
# themes via the header dropdown. Because Python module globals are looked up
# by name at call time, reassigning them here redirects every subsequent
# widget style that reads them.
DARK_BG      = "#1e1e2e"
PANEL_BG     = "#252535"
ACCENT       = "#7c6af7"
ACCENT_HOVER = "#9d8ff9"
SUCCESS      = "#3ddc84"
WARNING      = "#f5c518"
DANGER       = "#ff6b6b"
TEXT         = "#e0e0f0"
TEXT_DIM     = "#888899"
BORDER       = "#3a3a55"
ROW_ALT      = "#2a2a3e"
# Derived colours used by the HTML detail renderers (code boxes / requirement
# headings). Previously hardcoded dark (#1a1a2e / #7cb9e8) which stayed dark
# and unreadable on light themes -- now theme-derived. See trek_theme.
CODE_BG      = "#1a1a2e"
REQ_COLOR    = "#7cb9e8"
# Semantic accent colours for tree nodes / headings (SWR, related-SYT, amber).
# Previously hardcoded (#f5c518 / #c792ea / #e8b84b) which were unreadable on
# light themes -- now theme-derived (readable on both). See trek_theme.
SWR_COLOR     = "#f5c518"
RELATED_COLOR = "#c792ea"
AMBER_COLOR   = "#e8b84b"
# Soft accent-tinted row selection (keeps item's own fg readable) + a more
# legible 'dim' text colour than TEXT_DIM (which is too faint on light themes).
SELECTION_BG  = "#7c6af7"
DIM_TEXT      = "#888899"
# Text-only danger/success (darker than the button-fill DANGER/SUCCESS on
# light themes) -- used for classification/verdict labels. See trek_theme.
DANGER_TEXT   = "#ff6b6b"
SUCCESS_TEXT  = "#3ddc84"


def _apply_theme_constants(theme: dict):
    """Fill the module-level colour constants from a theme colour dict."""
    global DARK_BG, PANEL_BG, ACCENT, ACCENT_HOVER, SUCCESS, WARNING
    global DANGER, TEXT, TEXT_DIM, BORDER, ROW_ALT, CODE_BG, REQ_COLOR
    global SWR_COLOR, RELATED_COLOR, AMBER_COLOR, SELECTION_BG, DIM_TEXT
    global DANGER_TEXT, SUCCESS_TEXT
    DARK_BG      = theme["bg"]
    PANEL_BG     = theme["panel"]
    ACCENT       = theme["accent"]
    ACCENT_HOVER = theme["accent_hover"]
    SUCCESS      = theme["success"]
    WARNING      = theme["warning"]
    DANGER       = theme["danger"]
    TEXT         = theme["text"]
    TEXT_DIM     = theme["text_dim"]
    BORDER       = theme["border"]
    ROW_ALT      = theme["row_alt"]
    CODE_BG      = trek_theme.code_bg(theme)
    REQ_COLOR    = trek_theme.req_color(theme)
    SWR_COLOR    = trek_theme.swr_color(theme)
    RELATED_COLOR = trek_theme.related_color(theme)
    AMBER_COLOR  = trek_theme.amber_color(theme)
    SELECTION_BG = trek_theme.selection_bg(theme)
    DIM_TEXT     = trek_theme.dim_text(theme)
    # Text-only variants of danger/success -- darker than the theme's
    # button-fill danger/success on light themes, since small coloured TEXT
    # needs more contrast than a solid button fill with white text on top.
    # See _themed_classification_color() below, which uses these for every
    # CLASSIFICATION_LABELS / LLM_VERDICT_LABELS colour (duplicate/distinct/
    # same_scenario/different_scenario), not just the amber ones.
    DANGER_TEXT  = trek_theme.danger_text_color(theme)
    SUCCESS_TEXT = trek_theme.success_text_color(theme)


# Populate constants from the persisted active theme before the STYLESHEET
# below (and any widgets) are built.
_apply_theme_constants(trek_theme.get_active())


# trek_similarity.py is a theme-agnostic logic/data module, so its
# CLASSIFICATION_LABELS / LLM_VERDICT_LABELS ship with FIXED colours (red/
# orange/yellow/green, e.g. "distinct" -> hardcoded #2ecc71). Those fixed
# hex values were tuned for a dark background and have very poor contrast
# on light themes (e.g. #2ecc71 green is only ~2:1 against white -- nearly
# invisible; the yellow/orange pair used by "similar"/"near_duplicate"/
# "partial_overlap" has the same problem, already solved elsewhere via
# SWR_COLOR/AMBER_COLOR). Map EVERY key to its theme-derived equivalent
# wherever these labels are rendered, instead of hardcoding a theme
# dependency into trek_similarity itself.
_CLASSIFICATION_COLOR_OVERRIDES = {
    "duplicate":          "DANGER_TEXT",
    "near_duplicate":     "AMBER_COLOR",
    "similar":            "AMBER_COLOR",
    "distinct":           "SUCCESS_TEXT",
    "not_scored":         "DIM_TEXT",
    "same_scenario":      "DANGER_TEXT",
    "partial_overlap":    "AMBER_COLOR",
    "different_scenario": "SUCCESS_TEXT",
    "error":              "DIM_TEXT",
}


def _themed_classification_color(key: str, fallback_color: str) -> str:
    """Return a theme-correct colour for a CLASSIFICATION_LABELS/
    LLM_VERDICT_LABELS entry, overriding every fixed colour known to wash
    out on light themes with the corresponding theme-derived module
    constant (read by name at call time, so a runtime theme switch is
    picked up automatically -- see _apply_theme_constants)."""
    const_name = _CLASSIFICATION_COLOR_OVERRIDES.get(key)
    if const_name is not None:
        return globals().get(const_name, fallback_color)
    return fallback_color

# The global stylesheet is now generated from the active theme's colours
# (see trek_theme.build_stylesheet). The default "Charcoal" theme reproduces
# the original palette exactly, so the default look is unchanged. Rebuilt on
# theme switch via TrekMainWindow._change_theme().
STYLESHEET = trek_theme.build_stylesheet(trek_theme.get_active())

# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class WorkerCancelled(Exception):
    """Raised internally when a _CancellableWorker notices request_cancel()
    was called; caught by the worker's own run()/_run_inner() to exit
    cleanly (emitting a 'Cancelled by user.' error) instead of continuing
    or looking like a hard failure."""


class _CancellableWorker:
    """Mixin adding cooperative cancellation to a QThread worker. QThread
    has no safe way to forcibly kill a running thread mid-network-call, so
    instead this exposes a flag that long-running loops (one iteration per
    module/batch/pair-group) check between iterations and abort via
    WorkerCancelled once noticed -- typically within one in-flight
    request of the user clicking "⏹ Stop", not instantly, but without the
    corruption risk of actually terminating the thread.
    """
    _cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def raise_if_cancelled(self):
        if self._cancel_requested:
            raise WorkerCancelled()


class BuildIndexWorker(QThread, _CancellableWorker):
    """Build the object-id -> module INDEX (all 4 levels: SYR/SWR/SYT/SWT)
    for the active project off the UI thread. See trek_index.py. Once built,
    every hop of the traceability chain resolves ids to modules via O(1)
    lookups instead of live candidate probing / name-guessing."""
    progress = Signal(str)
    done     = Signal(dict)   # per-kind report {"SYR": n, "SWR": m, ...}
    error    = Signal(str)

    def __init__(self, force: bool = False, syt_prefixes: Optional[List[str]] = None):
        super().__init__()
        self.force = force
        self.syt_prefixes = syt_prefixes

    def run(self):
        start = time.perf_counter()
        with CACHE.batch_writes(), \
             LOG.timed("Index", "Build id->module index",
                        force=self.force) as t:
            try:
                # Log each level to the activity log as it completes (not
                # just the final summary), so the whole build is visible in
                # "Show Log" instead of appearing to do nothing for minutes.
                # Each kind is separately timed for a per-level duration.
                def _log_progress(msg: str):
                    self.progress.emit(msg)          # status bar (live)
                    LOG.log("Index", msg)            # activity log (persisted)

                report: Dict[str, int] = {}
                for kind in ("SYR", "SWR", "SYT", "SWT"):
                    k_start = time.perf_counter()
                    part = INDEX.build(kinds=(kind,),
                                       progress_cb=_log_progress,
                                       force=self.force,
                                       syt_prefixes=self.syt_prefixes)
                    added = part.get(kind, 0)
                    report[kind] = added
                    LOG.log("Index",
                            f"Indexed {kind}: +{added} ids "
                            f"({time.perf_counter() - k_start:.1f}s). "
                            f"Running total: {INDEX.size()} ids.")

                t.details.update(
                    syr=report.get("SYR", 0), swr=report.get("SWR", 0),
                    syt=report.get("SYT", 0), swt=report.get("SWT", 0),
                    total_ids=INDEX.size(),
                )
                # Record to the App Report (operation_stats) so the index
                # build shows up there alongside the other phases.
                CACHE.record_operation_stat(
                    "build_index", module=None,
                    item_count=INDEX.size(),
                    duration_seconds=time.perf_counter() - start,
                    source="live" if self.force else "cache-aware",
                    details={"syr": report.get("SYR", 0), "swr": report.get("SWR", 0),
                             "syt": report.get("SYT", 0), "swt": report.get("SWT", 0)},
                )
                self.done.emit(report)
            except Exception as exc:   # noqa: BLE001
                self.error.emit(f"Index build failed: {exc}")


def _key_tc_not_returned(project_id=None, campaign_id=None) -> str:
    """Cache key: ids TREK ANSWERED for but did not return in a previous
    offline download (deleted/archived/manual-only). Skipped next time
    unless the user asks to retry them, so every run doesn't re-request
    the same dead ids."""
    return (f"tc_not_returned:{PROJECT_ID if project_id is None else project_id}:"
            f"{CAMPAIGN_ID if campaign_id is None else campaign_id}")


class DownloadTcContentWorker(QThread, _CancellableWorker):
    """Offline preparation: download the text (Name / PreCondition /
    Procedure / Expected_result / Module_Path ...) of EVERY SYT and/or SWT
    test case known to the id->module INDEX, so Get Test Cases and Build
    Traceability later work fully from the local cache -- chapters, SYT/SWT
    detail panels and Check Duplicates included -- even without TREK.

    * Only ids NOT already cached are fetched (primary-key check, cheap).
    * Fetched in chunks; every chunk is saved (committed) immediately, so
      Stop / a crash / a network drop loses nothing and the next run simply
      continues with what is still missing.
    * Ids TREK answered for but did not return are remembered
      (_key_tc_not_returned) and skipped next time unless retry is asked.
    * Aborts early if TREK is clearly unreachable (whole chunks failing).
    * Optional phase 2 (req_modules): every SYR module's links + requirement
      text and every SWR module's requirement text -- the per-module data
      Build Traceability otherwise fetches live on first use.
    """
    progress = Signal(str)
    done     = Signal(dict)
    error    = Signal(str)

    CHUNK_SIZE = 600                 # ids per chunk (= 12 API batches of 50)
    MAX_UNREACHABLE_CHUNKS = 2       # consecutive fully-failed chunks -> stop

    def __init__(self, id_groups: Dict[str, List[str]], retry_not_returned: bool = False,
                 req_modules: Optional[Dict[str, List[str]]] = None):
        super().__init__()
        self.id_groups = id_groups            # {"SYT": [...], "SWT": [...]}
        self.retry_not_returned = retry_not_returned
        # {"SYR": [module, ...], "SWR": [module, ...]} -- requirement modules
        # whose data Build Traceability needs: SYR -> links (domain 2) +
        # requirement text; SWR -> requirement text. Empty/None = skip.
        self.req_modules = req_modules or {}
        # Pin the project + cache DB this run belongs to: switching projects
        # in the header while the download runs must not redirect it.
        self.project_id = PROJECT_ID
        self.campaign_id = CAMPAIGN_ID
        self.cache = CACHE

    def run(self):
        start = time.perf_counter()
        with LOG.timed("Offline", "Download all test-case text",
                        kinds=",".join(self.id_groups)) as t:
            try:
                report = self._run_inner()
                report["duration_seconds"] = time.perf_counter() - start
                t.details.update(downloaded=report["downloaded"],
                                 not_returned=report["not_returned"],
                                 failed=report["failed"],
                                 cancelled=report["cancelled"],
                                 unreachable=report["unreachable"])
                self.cache.record_operation_stat(
                    "download_tc_content", module=None,
                    item_count=report["downloaded"], extra_count=report["not_returned"],
                    duration_seconds=report["duration_seconds"], source="live",
                    details=dict(report),
                )
                self.done.emit(report)
            except Exception as exc:   # noqa: BLE001
                t.level = "ERROR"
                t.details["error"] = str(exc)
                self.error.emit(f"Offline download failed: {exc}")

    def _run_inner(self) -> dict:
        report = {"kinds": {}, "to_download": 0, "downloaded": 0, "not_returned": 0,
                  "failed": 0, "skipped_known_missing": 0,
                  "cancelled": False, "unreachable": False, "error": "",
                  "req": {}}
        if self.id_groups:
            self._download_tc_text(report)
        if self.req_modules and not report["cancelled"] and not report["unreachable"]:
            self._download_requirement_modules(report)
        return report

    def _download_tc_text(self, report: dict) -> dict:

        cached_nr = self.cache.get_blob(_key_tc_not_returned(self.project_id, self.campaign_id))
        prev_not_returned = set((cached_nr[0] if cached_nr else []) or [])

        # 1. Work out what is still missing, per kind (no network).
        todo: List[str] = []
        kind_of: Dict[str, str] = {}
        for kind, ids in self.id_groups.items():
            if self.is_cancelled:
                report["cancelled"] = True
                return report
            self.progress.emit(f"Checking which of {len(ids):,} {kind} test cases are already saved...")
            have = self.cache.cached_tc_id_set(ids)
            missing = [i for i in ids if i not in have]
            skipped = 0
            if not self.retry_not_returned and prev_not_returned:
                before = len(missing)
                missing = [i for i in missing if i not in prev_not_returned]
                skipped = before - len(missing)
            report["skipped_known_missing"] += skipped
            report["kinds"][kind] = {"total": len(ids), "already_saved": len(have),
                                     "to_download": len(missing), "downloaded": 0,
                                     "not_returned": 0, "failed": 0,
                                     "skipped_known_missing": skipped}
            for i in missing:
                if i not in kind_of:
                    kind_of[i] = kind
                    todo.append(i)
        total = len(todo)
        report["to_download"] = total
        if not total:
            return report

        # 2. Download in chunks, saving each chunk right away.
        client = TrekExportLinksClient()
        got: set = set()
        not_returned: set = set()
        consecutive_unreachable = 0
        processed = 0
        t0 = time.perf_counter()
        for start in range(0, total, self.CHUNK_SIZE):
            if self.is_cancelled:
                report["cancelled"] = True
                break
            ids = todo[start:start + self.CHUNK_SIZE]
            fetched = client.get_test_case_content(
                self.project_id, self.campaign_id, ids, batch_size=50, verbose=False,
            ) or {}
            fetched = {k: v for k, v in fetched.items() if k in kind_of}
            failed = set(getattr(client, "last_failed_ids", []) or []) - set(fetched)
            if fetched:
                self.cache.set_tc_contents(fetched)
            nr = [i for i in ids if i not in fetched and i not in failed]
            got.update(fetched)
            not_returned.update(nr)
            for i in fetched:
                report["kinds"][kind_of[i]]["downloaded"] += 1
            for i in nr:
                report["kinds"][kind_of[i]]["not_returned"] += 1
            for i in failed:
                report["kinds"][kind_of[i]]["failed"] += 1
            report["downloaded"] += len(fetched)
            report["failed"] += len(failed)
            processed += len(ids)

            if ids and len(failed) == len(ids):
                consecutive_unreachable += 1
                if consecutive_unreachable >= self.MAX_UNREACHABLE_CHUNKS:
                    errors = getattr(client, "last_fetch_errors", []) or []
                    report["unreachable"] = True
                    report["error"] = str(errors[0])[:300] if errors else "all requests failed"
                    break
            else:
                consecutive_unreachable = 0

            elapsed = time.perf_counter() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            m, sec = divmod(int(eta), 60)
            self.progress.emit(
                f"Offline download: {processed:,}/{total:,} test cases "
                f"({processed * 100 // total}%) · saved {report['downloaded']:,}"
                + (f" · not in TREK {len(not_returned):,}" if not_returned else "")
                + (f" · failed {report['failed']:,}" if report["failed"] else "")
                + f" · ~{m}m {sec:02d}s left"
            )

        report["not_returned"] = len(not_returned)

        # 3. Remember ids TREK did not return (and forget ones it now did).
        new_nr = (prev_not_returned - got) | not_returned
        if new_nr != prev_not_returned:
            self.cache.set_blob(_key_tc_not_returned(self.project_id, self.campaign_id), sorted(new_nr))
        if not_returned:
            LOG.log("Offline", f"TREK returned no text for {len(not_returned)} test case(s), "
                               f"e.g. {sorted(not_returned)[:10]}", level="WARN")
        return report

    REQ_WORKERS = 4                  # parallel module downloads (SSPI-safe level)
    MAX_FAILED_MODULES_IN_ROW = 3    # consecutive failed modules -> TREK unreachable

    def _download_requirement_modules(self, report: dict) -> None:
        """Phase 2: for every SYR module -> its links (domain 2: SYR->SWR and
        SYR->other-SYT edges) + its full requirement export; for every SWR
        module -> its full requirement export. Exactly the per-module data
        Build Traceability would otherwise fetch live the first time a module
        is used (same cache keys as _get_links_items/_get_requirements_cached),
        so afterwards traceability downloads nothing. Already-saved entries
        are skipped (key-existence check only, no payload load)."""
        import threading
        pid, cid = self.project_id, self.campaign_id

        # (module, what) jobs -- what in {"links", "reqs"}
        jobs: List[tuple] = []
        for level, mods in self.req_modules.items():
            for m in mods:
                jobs.append((m, "reqs"))
                if level == "SYR":
                    jobs.append((m, "links"))
        key_of = {
            (m, w): (trek_cache.key_links(pid, cid, m, 2) if w == "links"
                     else trek_cache.key_requirements(pid, cid, m))
            for (m, w) in jobs
        }
        self.progress.emit(f"Checking which of {len(jobs)} requirement-module downloads are already saved...")
        present = self.cache.present_blob_keys(list(key_of.values()))
        todo = [j for j in jobs if key_of[j] not in present]
        n_modules = sum(len(v) for v in self.req_modules.values())
        rq = report["req"] = {"modules": n_modules, "jobs_total": len(jobs),
                              "already_saved": len(jobs) - len(todo), "to_download": len(todo),
                              "downloaded": 0, "failed": 0, "failed_items": []}
        if not todo:
            return

        local = threading.local()

        def _client():
            c = getattr(local, "client", None)
            if c is None:
                c = local.client = TrekExportLinksClient()
            return c

        def _one(job):
            """Network only -- runs in a pool thread. Returns the payload;
            the SQLite write happens back on this worker's own thread (one
            shared connection must never commit from several threads)."""
            mod, what = job
            last_err = ""
            for attempt in range(2):                     # one retry
                if self.is_cancelled:
                    return job, None, "cancelled"
                c = _client()
                if what == "links":
                    r = c.get_links(pid, cid, [mod], 2, timeout=200)
                    if getattr(r, "success", False):
                        items = (r.links[0].get("Data", [])
                                 if r.links and isinstance(r.links[0], dict) else [])
                        return job, items, ""
                else:
                    r = c.get_requirements(pid, cid, [mod], timeout=300)
                    if getattr(r, "success", False):
                        return job, (r.requirements or []), ""
                last_err = getattr(r, "error_message", "") or f"HTTP {getattr(r, 'status_code', '?')}"
                time.sleep(2)
            return job, None, last_err

        t0 = time.perf_counter()
        done_n = 0
        failed_in_row = 0
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        with ThreadPoolExecutor(max_workers=self.REQ_WORKERS) as ex:
            futures = [ex.submit(_one, j) for j in todo]
            for fut in _as_completed(futures):
                job, payload, err = fut.result()
                ok = payload is not None
                if ok:
                    self.cache.set_blob(key_of[job], payload, keep_in_memory=False)
                done_n += 1
                if ok:
                    rq["downloaded"] += 1
                    failed_in_row = 0
                elif err != "cancelled":
                    rq["failed"] += 1
                    rq["failed_items"].append(f"{job[0]} ({job[1]}): {err}"[:200])
                    failed_in_row += 1
                stop = self.is_cancelled or failed_in_row >= self.MAX_FAILED_MODULES_IN_ROW
                if stop:
                    for f in futures:
                        f.cancel()
                    if self.is_cancelled:
                        report["cancelled"] = True
                    else:
                        report["unreachable"] = True
                        report["error"] = err
                    break
                elapsed = time.perf_counter() - t0
                eta = elapsed / done_n * (len(todo) - done_n) if done_n else 0
                m, sec = divmod(int(eta), 60)
                self.progress.emit(
                    f"Offline download (requirements): {done_n}/{len(todo)} "
                    f"module downloads · saved {rq['downloaded']}"
                    + (f" · failed {rq['failed']}" if rq["failed"] else "")
                    + f" · ~{m}m {sec:02d}s left"
                )
        if rq["failed_items"]:
            LOG.log("Offline", f"{rq['failed']} requirement-module download(s) failed, "
                               f"e.g. {rq['failed_items'][:5]}", level="WARN")


class FetchModulesWorker(QThread, _CancellableWorker):
    """Step 1: resolve the full module list.

    IMPORTANT: the TREK /Modules endpoint is scoped by
    ``localConfigurationDomainType`` just like /Export/Links is. Domain
    type 4 returns the SYT/SWT-family modules; domain type 2 returns the
    SYR/SWR-family modules. Calling get_modules() with only the default
    (domain type 4) means SYR/SWR modules are NEVER present in the result
    -- which silently broke SYR auto-detection (it scanned a module list
    that structurally could not contain any "SYR -"/"SWR -" entries).
    We therefore fetch both domain types here and merge them (de-duplicated
    by Name) into a single module list that Step 2's auto-detection can
    scan for every V-Model level.

    Cache-aware: unless ``force_refresh`` is set, a cache hit short-circuits
    the network calls entirely (no TrekExportLinksClient is even created).
    """
    done    = Signal(list, bool, str)   # (modules, from_cache, updated_at)
    error   = Signal(str)

    def __init__(self, force_refresh: bool = False):
        super().__init__()
        self.force_refresh = force_refresh

    def run(self):
        start = time.perf_counter()
        with CACHE.batch_writes(), \
             LOG.timed("Modules", "Load module list",
                        force_refresh=self.force_refresh) as t:
            try:
                key = trek_cache.key_modules(PROJECT_ID, CONFIG_ID)
                if not self.force_refresh:
                    cached = CACHE.get_blob(key)
                    if cached is not None:
                        modules, updated_at = cached
                        t.details.update(source="cache", module_count=len(modules))
                        CACHE.record_operation_stat(
                            "fetch_modules", item_count=len(modules),
                            duration_seconds=time.perf_counter() - start, source="cache",
                        )
                        self.done.emit(modules, True, updated_at)
                        return

                client = TrekExportLinksClient()
                merged: Dict[str, dict] = {}
                for domain_type in (4, 2):
                    self.raise_if_cancelled()
                    with LOG.timed("Modules", f"GET /Modules (domain_type={domain_type})") as t2:
                        resp = client.get_modules(PROJECT_ID, CONFIG_ID, domain_type, timeout=160)
                        t2.details["domain_type"] = domain_type
                        if resp.success:
                            t2.details["module_count"] = len(resp.modules)
                    if not resp.success:
                        # Only fail hard if BOTH domain types fail; a single
                        # domain type erroring shouldn't hide the other half.
                        if not merged and domain_type == 2:
                            t.details.update(source="live", error=resp.error_message)
                            self.error.emit(resp.error_message)
                            return
                        continue
                    for m in resp.modules:
                        name = m.get("Name")
                        if name:
                            merged[name] = m

                modules = list(merged.values())
                updated_at = CACHE.set_blob(key, modules)
                t.details.update(source="live", module_count=len(modules))
                CACHE.record_operation_stat(
                    "fetch_modules", item_count=len(modules),
                    duration_seconds=time.perf_counter() - start, source="live",
                )
                self.done.emit(modules, False, updated_at)
            except WorkerCancelled:
                t.level = "ERROR"
                t.details["error"] = "cancelled"
                self.error.emit("Cancelled by user.")
            except Exception as e:
                t.level = "ERROR"
                t.details["error"] = str(e)
                self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Module-name convention helper
# ---------------------------------------------------------------------------
DEFAULT_SYT_PREFIXES = ("SYT",)

def _parse_syt_prefixes(text: str) -> List[str]:
    prefixes: List[str] = []
    for part in (text or "").split(","):
        p = part.strip().upper()
        if p and p not in prefixes:
            prefixes.append(p)
    return prefixes or list(DEFAULT_SYT_PREFIXES)

def _is_syt_module(name: str, prefixes: Iterable[str] = DEFAULT_SYT_PREFIXES) -> bool:
    """True if `name` is a SYT (system-test) module, across the different
    naming conventions used by different TREK projects.

    Some projects name SYT modules "SYT - <Subsystem>" (dash-space), others
    use an underscore convention like "SYT_TS_412_Climate_Control". We treat
    a name as SYT when its first token -- everything up to the first space,
    underscore or hyphen -- matches one of the given *prefixes*
    (case-insensitive). That matches "SYT - X", "SYT_TS_X" and "SYT-X"
    while rejecting unrelated names such as "SYTHESIS". Keeping this in one
    place means the Step-1 filter stays convention-agnostic instead of
    hardcoding a single project's prefix.
    """
    if not name:
        return False
    first_token = re.split(r"[ _\-]", name.strip(), maxsplit=1)[0]
    return first_token.upper() in {p.upper() for p in prefixes}


# ---------------------------------------------------------------------------
# Shared cache-aware fetch helpers (used by multiple workers below so that
# Step 2 and Step 3 never re-fetch the exact same Export/Links call).
# ---------------------------------------------------------------------------
def _get_links_items(client, module_name: str, domain_type: int,
                      force_refresh: bool, timeout: int = 120):
    """Return (items, from_cache) for one module / domain_type Export/Links
    call. Transparently uses the local cache unless force_refresh is True.

    Mirrors the original (pre-cache) behaviour: a failed/empty API response
    is NOT an error here -- it just yields an empty item list (the SWR/SWT
    probing logic relies on this to silently skip modules that don't exist
    for a given subsystem). Only a successful response is written to the
    cache, so a transient failure never poisons the cache with an empty
    result for a module that does actually have data.
    """
    with LOG.timed("Links", f"Links for '{module_name}' (domain={domain_type})",
                    module=module_name, domain_type=domain_type) as t:
        key = trek_cache.key_links(PROJECT_ID, CAMPAIGN_ID, module_name, domain_type)
        if not force_refresh:
            cached = CACHE.get_blob(key)
            if cached is not None:
                items, _updated_at = cached
                t.details.update(source="cache", item_count=len(items))
                return items, True

        r = client.get_links(PROJECT_ID, CAMPAIGN_ID, [module_name], domain_type, timeout=timeout)
        items = (r.links[0].get("Data", [])
                 if r.links and isinstance(r.links[0], dict) else [])
        if r.success:
            CACHE.set_blob(key, items)
        t.details.update(source="live", item_count=len(items), success=r.success)
        return items, False


def _get_requirements_cached(client, module_name: str, force_refresh: bool,
                              timeout: int = 120):
    """Return (requirements_list, from_cache) for one module's full
    Export/Requirements payload (Name/text, Type, Maturity, SIL,
    TestCoverage, VerificationMethod, DOORS Url, etc). Cached per-module
    just like _get_links_items, so re-opening the same module's
    requirement content across Step 3 clicks/sessions doesn't re-fetch."""
    with LOG.timed("Requirements", f"Requirements for '{module_name}'", module=module_name) as t:
        key = trek_cache.key_requirements(PROJECT_ID, CAMPAIGN_ID, module_name)
        if not force_refresh:
            cached = CACHE.get_blob(key)
            if cached is not None:
                reqs, _updated_at = cached
                t.details.update(source="cache", item_count=len(reqs))
                return reqs, True

        r = client.get_requirements(PROJECT_ID, CAMPAIGN_ID, [module_name], timeout=timeout)
        if not r.success:
            t.details.update(source="live", success=False)
            return [], False
        CACHE.set_blob(key, r.requirements)
        t.details.update(source="live", item_count=len(r.requirements))
        return r.requirements, False


def _parallel_fetch_per_module(module_names: List[str], fetch_one, max_workers: int = 5) -> List[tuple]:
    """Run fetch_one(module_name) for every module in module_names
    CONCURRENTLY (up to max_workers at a time) instead of one-at-a-time,
    since fetching different modules' links/requirements are independent
    network calls with no dependency on each other's results.

    Each call gets its OWN TrekExportLinksClient() (fresh requests.Session)
    -- never share one client/session across threads here, because
    requests_negotiate_sspi's HttpNegotiateAuth caches mutable per-instance
    state (self._host) on first use, so concurrent requests through a
    single shared auth object risk a data race. Windows SSPI credentials
    are ambient (tied to the logged-on user, not stored per-session), so a
    fresh client per worker is cheap and correct -- same pattern used by
    TrekExportLinksAPI.get_test_case_content()'s internal batch parallelism.

    Performance: a cache-first pass on the CALLING thread resolves every
    module that's already cached WITHOUT creating any TrekExportLinksClient
    (and its expensive SSPI Negotiate handshake). Only genuinely uncached
    modules are submitted to the thread pool with a real client. On a
    fully-cached Build Traceability this means ZERO SSPI handshakes and
    ZERO thread-pool overhead, reducing the time from ~30-60s to <1s.

    Args:
        module_names: modules to fetch, each processed independently.
        fetch_one: callable(client, module_name) -> result, invoked once
                   per module in a worker thread with its own client.
        max_workers: max concurrent in-flight requests.

    Returns:
        List of (module_name, result) tuples, in COMPLETION order (not
        necessarily the same order as module_names) -- callers that need
        a specific order should sort/index by module_name afterward.
    """
    if not module_names:
        return []

    # ---- Cache-first pass (no client, no threads) ----
    # Run fetch_one with client=None for every module on the calling thread.
    # The fetch_one callbacks (_get_links_items, _get_requirements_cached)
    # check the local cache BEFORE accessing the client, so a cache hit
    # returns immediately without ever touching `client`. A cache miss
    # tries to call client.get_links() on None -> AttributeError, which
    # we catch to mark the module as needing a real fetch.
    results: List[tuple] = []
    uncached: List[str] = []
    for name in module_names:
        try:
            results.append((name, fetch_one(None, name)))
        except (AttributeError, TypeError):
            # Cache miss -- client=None was accessed -> needs a real fetch.
            uncached.append(name)

    if not uncached:
        # Everything was cached -- skip the thread pool entirely.
        return results

    def _run_one(name: str, retries: int = 2):
        # Windows SSPI (requests_negotiate_sspi) occasionally raises a
        # transient "bad parameter or other API misuse" pywin32 error under
        # concurrent Negotiate handshakes (multiple fresh sessions
        # authenticating at once). It's not a real request failure -- retry
        # with a fresh client/session a couple of times before giving up.
        last_exc = None
        for attempt in range(retries + 1):
            try:
                client = TrekExportLinksClient()
                return name, fetch_one(client, name)
            except Exception as exc:   # noqa: BLE001
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise last_exc

    with LOG.timed("Parallel Fetch", f"{len(uncached)} uncached of {len(module_names)} module(s)",
                    module_count=len(module_names), max_workers=max_workers) as t:
        failed: List[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(uncached)))) as executor:
            futures = {executor.submit(_run_one, name): name for name in uncached}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:   # noqa: BLE001
                    # One module's failure (even after retries) must NOT
                    # discard every other module that already succeeded --
                    # log it, skip it, and let the caller see a hole for
                    # just that module instead of losing the whole batch.
                    failed.append(name)
                    LOG.log("Parallel Fetch", f"Module '{name}' failed: {exc}", level="ERROR")
        t.details["modules"] = ", ".join(module_names[:10]) + (
            f" (+{len(module_names) - 10} more)" if len(module_names) > 10 else ""
        )
        t.details["cached"] = len(module_names) - len(uncached)
        t.details["uncached"] = len(uncached)
        if failed:
            t.details["failed_modules"] = ", ".join(failed)
            t.level = "WARN"
    return results


# Step 2 chapter bucket for test cases whose text could not be downloaded
# because TREK was unreachable (vs. "(no chapter)" = text present but no
# Module_Path). See _get_tc_content_cached(failed_out=...).
NOT_DOWNLOADED_CHAPTER = "(text not downloaded -- TREK unreachable)"


def _get_tc_content_cached(client, tc_ids: List[str], force_refresh: bool,
                            batch_size: int = 50,
                            failed_out: Optional[set] = None) -> Dict[str, dict]:
    """Return {tc_id: content} for tc_ids, using the global TC-content cache
    unless force_refresh is True. Only IDs missing from the cache (or all
    IDs, when forcing) are actually fetched from TREK.

    ``failed_out`` (optional set): receives the ids whose download FAILED
    (TREK unreachable, timeout, HTTP error) -- as opposed to ids TREK
    answered for but did not return. The UI uses this to label them
    "not downloaded" instead of the misleading "NOT FOUND IN TREK"."""
    if not tc_ids:
        return {}

    with LOG.timed("TC Content", f"Content for {len(tc_ids)} TC(s)",
                    requested=len(tc_ids), batch_size=batch_size) as timed_op:
        if force_refresh:
            to_fetch: List[str] = list(tc_ids)
            content_map: Dict[str, dict] = {}
        else:
            content_map = CACHE.get_tc_contents(tc_ids)
            to_fetch = [tid for tid in tc_ids if tid not in content_map]

        timed_op.details.update(from_cache=len(content_map), to_fetch=len(to_fetch))

        if to_fetch:
            fetched = client.get_test_case_content(
                PROJECT_ID, CAMPAIGN_ID, to_fetch, batch_size=batch_size, verbose=False
            )
            CACHE.set_tc_contents(fetched)
            content_map.update(fetched)
            timed_op.details["fetched_live"] = len(fetched)
            failed = [t for t in getattr(client, "last_failed_ids", []) or []
                      if t not in content_map]
            if failed and force_refresh:
                # A forced refresh that could not reach TREK must not throw
                # away content we already have: fall back to the cached copy
                # for exactly the ids whose download failed.
                fallback = CACHE.get_tc_contents(failed)
                if fallback:
                    content_map.update(fallback)
                    timed_op.details["cache_fallback"] = len(fallback)
                    failed = [t for t in failed if t not in fallback]
            if failed:
                timed_op.details["not_downloaded"] = len(failed)
                errors = getattr(client, "last_fetch_errors", []) or []
                if errors:
                    timed_op.details["fetch_error"] = str(errors[0])[:200]
                if failed_out is not None:
                    failed_out.update(failed)

        return content_map


class FetchLinksWorker(QThread, _CancellableWorker):
    progress = Signal(str)
    done     = Signal(dict)    # full result dict
    error    = Signal(str)

    def __init__(self, syt_module, syr_modules, swt_modules, selected_tc_ids,
                 all_swr_module_names=None, force_refresh: bool = False):
        super().__init__()
        self.syt_module           = syt_module
        self.syr_modules          = syr_modules
        self.swt_modules          = swt_modules
        self.selected_tc_ids      = selected_tc_ids
        self.all_swr_module_names = all_swr_module_names or []
        self.force_refresh        = force_refresh

    def run(self):
        start = time.perf_counter()
        with CACHE.batch_writes(), \
             LOG.timed("Build Traceability", f"'{self.syt_module}' ({len(self.selected_tc_ids)} TCs selected)",
                        syt_module=self.syt_module, selected_tc_count=len(self.selected_tc_ids),
                        force_refresh=self.force_refresh) as overall_t:
            self._run_inner(overall_t, start)

    def _run_inner(self, overall_t, start):
        try:
            fr = self.force_refresh
            # Track whether ANY sub-fetch actually hit the network (vs.
            # everything served from cache) so the source= field in the
            # operation_stats record is accurate.
            _any_live = False

            client = TrekExportLinksClient()

            # 1. SYT links
            self.raise_if_cancelled()
            self.progress.emit(f"{'Refreshing' if fr else 'Fetching'} SYT links for '{self.syt_module}'...")
            syt_items, from_cache = _get_links_items(client, self.syt_module, 4, fr)
            if not from_cache:
                _any_live = True
            if from_cache:
                self.progress.emit(f"SYT links for '{self.syt_module}' loaded from cache.")

            # 2. SYR -> SWR. Confirmed directly in TREK that the SYR
            self.raise_if_cancelled()
            # requirement is cross-linked with the SWR requirement it's
            # refined into (SYR_INFRA_310 <-> SWR_INFRA_SWU_4225 via a
            # "syr-swr" link group). HOWEVER, unlike the SYT->SYR hop
            # (where the child/test always owns the OUT link), SWR
            # *derives from* SYR -- so SWR is the "child" here and, by the
            # same convention, it is SWR that owns the OUT link, meaning
            # the SYR module's own export sees it as an INCOMING link, not
            # OUT. Rather than gamble on "IN" being exactly right either,
            # match by ID prefix only (LinkKey=SWR_*) regardless of
            # LinkType, since that pairing is unambiguous either way. The
            # SYR side's own id is NOT prefix-checked -- it's already
            # guaranteed valid by construction (every item here comes from
            # a module already confirmed as this SYT's SYR bridge, see
            # self.syr_modules), which also makes this work for a
            # confirmed SYR-like module whose own ids don't use the "SYR_"
            # convention at all (see _EXTRA_SYR_LIKE_MODULES).
            # Fetched CONCURRENTLY (independent per-module requests -- see
            # _parallel_fetch_per_module docstring) instead of one module
            # at a time; a selection tracing through several SYR modules
            # previously paid full network latency per module sequentially.
            self.progress.emit(
                f"{'Refreshing' if fr else 'Fetching'} {len(self.syr_modules)} SYR bridge module(s)..."
            )
            syr_to_swr = defaultdict(set)
            # Links from a bridge SYR to OTHER SYT test cases (different
            # from the one we started tracing from) -- e.g. a SYR shared by
            # multiple SYT test cases across modules/functional areas.
            # Matched by ID prefix only (LinkKey=SYT_*) regardless of
            # LinkType, same rationale as the SYR<->SWR pairing above.
            syr_to_syt = defaultdict(set)
            syr_module_items: Dict[str, list] = {}
            for syr_mod, (syr_items, from_cache) in _parallel_fetch_per_module(
                self.syr_modules, lambda c, name: _get_links_items(c, name, 2, fr)
            ):
                if not from_cache:
                    _any_live = True
                if from_cache:
                    self.progress.emit(f"SYR bridge '{syr_mod}' loaded from cache.")
                syr_module_items[syr_mod] = syr_items
                for lnk in syr_items:
                    key, lkey = lnk.get("Key", ""), lnk.get("LinkKey", "")
                    if lkey.startswith("SWR_"):
                        syr_to_swr[key].add(lkey)
                    elif lkey.startswith("SYT_"):
                        syr_to_syt[key].add(lkey)

            # Build syt_to_syr AFTER the SYR module fetch above so a link
            # can be verified against the confirmed SYR module(s)' OWN ids
            # (ground truth) instead of assuming the "SYR_" ID prefix --
            # needed for a confirmed SYR-like module whose ids don't
            # follow that convention at all (see _EXTRA_SYR_LIKE_MODULES).
            all_confirmed_syr_keys = {
                lnk.get("Key", "") for items in syr_module_items.values() for lnk in items
            }
            syt_to_syr = defaultdict(set)
            for lnk in syt_items:
                if lnk.get("LinkType") != "OUT":
                    continue
                lkey = lnk.get("LinkKey", "")
                if lkey.startswith("SYR_") or lkey in all_confirmed_syr_keys:
                    syt_to_syr[lnk["Key"]].add(lkey)

            # 2b. Auto-detect the real SWR module(s) bridged to by each SYR
            # module, using that SYR module's ENTIRE link export -- not
            # just the SWR ids reachable from the currently SELECTED test
            # cases. Using only the selection is what previously allowed a
            # handful of stray cross-references from an unrelated
            # functional area to occasionally outvote the genuine bridge
            # module when few TCs were selected (see a real incident:
            # "SYT - Rear Window Heating" with 227 TCs selected, where 3
            # stray SWR_SUP_* ids from a single outlier TC out-detected the
            # correct SWR module reachable by the other 226 -- full-module
            # data has 128 RWH-area references vs. those same 3 outliers,
            # a much clearer signal). See _detect_bridge_modules()'s
            # docstring for the acronym/overlap-count logic that also
            # protects against this.
            #
            # Cached per SOURCE SYR module (see _get_bridge_modules_cached)
            # so this detection -- which can involve probing several
            # candidate SWR modules live -- only ever runs ONCE per SYR
            # module for the lifetime of the cache, regardless of how many
            # different TC selections/runs reference that SYR module
            # afterward.
            # FAST PATH: resolve SWR bridge modules via the object-id -> module
            # INDEX (see trek_index.py). Every SWR id referenced by the SYR
            # modules is looked up directly -> we know EXACTLY which SWR
            # module(s) to fetch, with no live candidate probing. This
            # replaces _detect_bridge_modules, whose probing of ~17 candidate
            # modules (incl. huge unrelated ones like a 145k-item Diagnosis
            # module) was the dominant ~100s+ cost of a Build Traceability run.
            #
            # Manual mappings still take priority (a user override always
            # wins). Any SWR ids the index cannot resolve fall back to the old
            # candidate-probing so nothing breaks for unindexed ids.
            swr_modules: List[str] = []
            bridge_link_memo: Dict[tuple, list] = {}

            # Gather all referenced SWR ids per SYR module up front.
            swr_ids_by_syr: Dict[str, set] = {}
            all_referenced_swr_ids: set = set()
            for syr_mod in self.syr_modules:
                syr_items = syr_module_items.get(syr_mod, [])
                module_swr_ids = {
                    lnk.get("LinkKey", "") for lnk in syr_items
                    if lnk.get("Key", "").startswith("SYR_") and lnk.get("LinkKey", "").startswith("SWR_")
                }
                swr_ids_by_syr[syr_mod] = module_swr_ids
                all_referenced_swr_ids |= module_swr_ids

            index_built = INDEX.is_built("SWR")
            index_hits = 0
            dead_swr_ids_total = 0
            for syr_mod in self.syr_modules:
                module_swr_ids = swr_ids_by_syr.get(syr_mod, set())
                if not module_swr_ids:
                    continue

                # 1) Manual override wins.
                manual = get_manual_bridge_mapping(syr_mod, "SWR")
                if manual is not None:
                    self.progress.emit(f"SWR bridge for '{syr_mod}' from manual mapping.")
                    for name in manual:
                        if name not in swr_modules:
                            swr_modules.append(name)
                    continue

                # 2) INDEX lookup (O(1) per id, no network).
                resolved = INDEX.modules_for_ids(module_swr_ids, prefix="SWR")
                unresolved = INDEX.unresolved_ids(module_swr_ids, prefix="SWR")

                if index_built:
                    # Index is built -> an id NOT in it means the SWR
                    # requirement no longer exists in TREK (a dead/dangling
                    # link to a deleted/archived requirement). There is no
                    # module to fetch and no content to retrieve, so we simply
                    # IGNORE unresolved ids -- no probing. (Any module a dead
                    # id might have belonged to is already covered by its live
                    # sibling ids, so nothing real is lost.) This removes the
                    # slow candidate-probing entirely once the index exists.
                    if resolved:
                        index_hits += 1
                        for name in resolved:
                            if name not in swr_modules:
                                swr_modules.append(name)
                    if unresolved:
                        dead_swr_ids_total += len(unresolved)
                        LOG.log("Index",
                                f"Ignoring {len(unresolved)} dead SWR id(s) for "
                                f"'{syr_mod}' (not in TREK): {unresolved[:5]}")
                else:
                    # Index NOT built -> fall back to live candidate probing
                    # for everything (legacy behaviour, correctness first).
                    for name in resolved:
                        if name not in swr_modules:
                            swr_modules.append(name)
                    probe_ids = set(unresolved) if unresolved else module_swr_ids
                    if probe_ids:
                        self.progress.emit(
                            f"Index not built -- probing SWR candidates for '{syr_mod}'..."
                        )
                        bridged, source = _get_bridge_modules_cached(
                            client, syr_mod, probe_ids, self.all_swr_module_names,
                            domain_type=2, force_refresh=fr, target_kind="SWR",
                            progress_cb=self.progress.emit, label="SWR",
                            run_link_memo=bridge_link_memo,
                        )
                        for name in bridged:
                            if name not in swr_modules:
                                swr_modules.append(name)

            if index_hits:
                msg = f"Resolved SWR bridges for {index_hits} SYR module(s) instantly from the index."
                if dead_swr_ids_total:
                    msg += f" Ignored {dead_swr_ids_total} dead link(s)."
                self.progress.emit(msg)

            # 3. SWT links. Determine WHICH SWT modules to fetch. The SWT
            # test case owns the OUT link to its SWR, so SWT ids only live
            # inside SWT modules -- we can't look them up beforehand. Instead
            # the index built a reverse SWR-id -> {SWT module} map while
            # indexing SWT links. Resolve the SWT modules from ALL the SWR
            # ids our resolved SWR modules point to, and UNION that with the
            # name-guessed candidates passed in (so a miss on either side is
            # covered). This removes the reliance on SWT name-guessing.
            self.raise_if_cancelled()
            all_swr_ids_referenced: set = set()
            for _syr, _swrs in syr_to_swr.items():
                all_swr_ids_referenced |= _swrs
            index_swt_modules = INDEX.swt_modules_for_swr_ids(all_swr_ids_referenced)
            swt_modules_to_fetch = list(dict.fromkeys(
                list(self.swt_modules) + sorted(index_swt_modules)
            ))
            if index_swt_modules:
                extra = sorted(set(index_swt_modules) - set(self.swt_modules))
                if extra:
                    self.progress.emit(
                        f"Index added {len(extra)} SWT module(s) the name-match missed: "
                        f"{', '.join(extra[:5])}{'...' if len(extra) > 5 else ''}"
                    )

            self.progress.emit(
                f"{'Refreshing' if fr else 'Fetching'} {len(swt_modules_to_fetch)} SWT module(s)..."
            )
            swr_to_swt = defaultdict(set)
            _swt_cache_hits = 0
            _swt_live = 0
            for swt_mod, (swt_items, from_cache) in _parallel_fetch_per_module(
                swt_modules_to_fetch, lambda c, name: _get_links_items(c, name, 4, fr)
            ):
                if from_cache:
                    _swt_cache_hits += 1
                else:
                    _swt_live += 1
                    _any_live = True
                for lnk in swt_items:
                    if lnk.get("LinkType") == "OUT" and lnk.get("LinkKey", "").startswith("SWR_"):
                        swr_to_swt[lnk["LinkKey"]].add(lnk["Key"])
            LOG.log("Links",
                    f"SWT phase: {_swt_cache_hits} from cache, {_swt_live} live "
                    f"(force_refresh={fr}). modules={len(swt_modules_to_fetch)}")

            # 4. Build rows for selected TCs
            self.raise_if_cancelled()
            rows = []
            for syt_id in self.selected_tc_ids:
                syr_ids = sorted(syt_to_syr.get(syt_id, set()))
                swr_ids: set = set()
                for syr in syr_ids:
                    swr_ids |= syr_to_swr.get(syr, set())
                swt_ids: set = set()
                for swr in swr_ids:
                    swt_ids |= swr_to_swt.get(swr, set())

                chain = []
                related_syt_ids: set = set()
                for syr in syr_ids:
                    swrs = sorted(syr_to_swr.get(syr, set()))
                    swts_here: set = set()
                    for swr in swrs:
                        swts_here |= swr_to_swt.get(swr, set())
                    # Other SYT test cases that also link to this SYR --
                    # kept as "related" to the SYT we started from, since
                    # sharing a system requirement is a meaningful signal
                    # even though they weren't in the original selection.
                    related_here = sorted(syr_to_syt.get(syr, set()) - {syt_id})
                    related_syt_ids.update(related_here)
                    chain.append({
                        "syr_id": syr, "swr_ids": swrs, "swt_ids": sorted(swts_here),
                        "related_syt_ids": related_here,
                    })

                rows.append({
                    "syt_id":       syt_id,
                    "has_syr_link": len(syr_ids) > 0,
                    "has_swt":      len(swt_ids) > 0,
                    "syr_count":    len(syr_ids),
                    "swr_count":    len(swr_ids),
                    "swt_count":    len(swt_ids),
                    "syr_ids":      syr_ids,
                    "swr_ids":      sorted(swr_ids),
                    "swt_ids":      sorted(swt_ids),
                    "related_syt_ids": sorted(related_syt_ids),
                    "related_syt_count": len(related_syt_ids),
                    "chain":        chain,
                })

            # 5. Fetch TC content (global cache keyed by TC id, shared across
            self.raise_if_cancelled()
            #    every SYT module / run -- content is rarely re-fetched once
            #    seen). SYT and SWT content are independent of each other,
            #    so fetch both concurrently instead of one-after-the-other.
            #    Related SYT ids (found via a shared SYR, see step 4) are
            #    fetched together with the original selection -- content
            #    lookup is by TC id only, module-independent, so a related
            #    SYT from a different module resolves the same way.
            all_swt_ids = sorted({s for r in rows for s in r["swt_ids"]})
            all_related_syt_ids = sorted({s for r in rows for s in r["related_syt_ids"]})
            all_syt_ids = sorted(set(self.selected_tc_ids) | set(all_related_syt_ids))

            # Unique comparison-pair counts -- SAME dedup rule
            # DuplicateCheckWorker itself will later apply when building
            # its actual (syt, counterpart) pairs, so this number in the
            # traceability banner matches what "Check Duplicates" will
            # really compare: every SYT-SWT combo is its own pair, but a
            # SYT-SYT (related) pair is UNORDERED (A-related-to-B is the
            # same pair as B-related-to-A), so it's only counted once.
            syt_swt_pair_count = sum(len(r["swt_ids"]) for r in rows)
            seen_related_pairs: set = set()
            for r in rows:
                for related_id in r["related_syt_ids"]:
                    seen_related_pairs.add(frozenset((r["syt_id"], related_id)))
            syt_syt_pair_count = len(seen_related_pairs)

            self.progress.emit(
                f"Resolving content for {len(all_syt_ids)} SYT "
                f"({len(all_related_syt_ids)} related) + {len(all_swt_ids)} SWT TCs..."
            )
            syt_not_downloaded: set = set()
            swt_not_downloaded: set = set()
            with ThreadPoolExecutor(max_workers=2) as executor:
                syt_future = executor.submit(_get_tc_content_cached, TrekExportLinksClient(), all_syt_ids, fr,
                                             failed_out=syt_not_downloaded)
                swt_future = executor.submit(_get_tc_content_cached, TrekExportLinksClient(), all_swt_ids, fr,
                                             failed_out=swt_not_downloaded)
                syt_content = syt_future.result()
                swt_content = swt_future.result()
            # ids we could not download (TREK unreachable) -- labelled
            # "not downloaded" in the tree, NOT "NOT FOUND IN TREK".
            not_downloaded_ids = syt_not_downloaded | swt_not_downloaded

            # 6. Fetch SYR + SWR requirement content (Name/text, Type,
            self.raise_if_cancelled()
            # Maturity, SIL, TestCoverage, VerificationMethod, DOORS Url)
            # for every SYR/SWR module resolved -- fetched concurrently per
            # module (same rationale as the link fetches above), then
            # indexed by Key for O(1) lookup per row.
            # Fetch SYR + SWR requirement content in ONE combined parallel
            # batch instead of two sequential ones. SYR and SWR requirements
            # are independent, so overlapping them (rather than finishing all
            # SYR, then all SWR) shortens the critical path. We tag each
            # module with its level so the results still land in the right
            # content map. max_workers=5 (NOT higher): Windows SSPI
            # (requests_negotiate_sspi) has shown transient "bad parameter or
            # other API misuse" failures under many concurrent Negotiate
            # handshakes (each worker authenticates a fresh session) -- 5 is
            # the historically stable level; _parallel_fetch_per_module also
            # now retries a failed module a couple of times before giving up,
            # and one module failing no longer discards the whole batch.
            self.progress.emit(
                f"Resolving requirement content for {len(self.syr_modules)} SYR + {len(swr_modules)} SWR module(s)..."
            )
            syr_content: Dict[str, dict] = {}
            swr_content: Dict[str, dict] = {}
            _syr_set = set(self.syr_modules)
            _all_req_modules = list(self.syr_modules) + list(swr_modules)
            for req_mod, (reqs, from_cache) in _parallel_fetch_per_module(
                _all_req_modules,
                lambda c, name: _get_requirements_cached(c, name, fr),
                max_workers=5,
            ):
                target = syr_content if req_mod in _syr_set else swr_content
                if from_cache:
                    self.progress.emit(f"Requirement content for '{req_mod}' loaded from cache.")
                else:
                    _any_live = True
                for req in reqs:
                    req_key = req.get("Key")
                    if req_key:
                        target[req_key] = req

            for r in rows:
                r["not_downloaded_ids"] = sorted(
                    t for t in [r["syt_id"], *r["swt_ids"], *r["related_syt_ids"]]
                    if t in not_downloaded_ids
                )
                r["syt_content"] = syt_content.get(r["syt_id"])
                r["swt_content"] = [swt_content[s] for s in r["swt_ids"] if s in swt_content]
                r["related_syt_content"] = [
                    syt_content[s] for s in r["related_syt_ids"] if s in syt_content
                ]
                r["syr_content"] = {
                    syr_id: syr_content[syr_id] for syr_id in r["syr_ids"] if syr_id in syr_content
                }
                r["swr_content"] = {
                    swr_id: swr_content[swr_id] for swr_id in r["swr_ids"] if swr_id in swr_content
                }

            # Actually-comparable pair counts, i.e. what "Check Duplicates"
            # will really build -- can be LESS than syt_swt_pair_count /
            # syt_syt_pair_count above if TREK returned no content for some
            # SWT/related-SYT id (archived/deleted/restricted test case),
            # since compare_syt_swt_pairs() only ever sees swt_content /
            # related_syt_content, never the raw id lists.
            syt_swt_pair_count_resolved = sum(len(r["swt_content"]) for r in rows)
            seen_related_resolved: set = set()
            for r in rows:
                for related in r["related_syt_content"]:
                    seen_related_resolved.add(frozenset((r["syt_id"], related.get("Key", ""))))
            syt_syt_pair_count_resolved = len(seen_related_resolved)

            result = {
                "meta": {
                    "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
                    "project_id":     PROJECT_ID,
                    "campaign_id":    CAMPAIGN_ID,
                    "config_id":      CONFIG_ID,
                    "syt_module":     self.syt_module,
                    "syr_modules_used": self.syr_modules,
                    "swr_modules_used": swr_modules,
                    "swt_modules_used": swt_modules_to_fetch,
                },
                "summary": {
                    "selected_tc_count":        len(self.selected_tc_ids),
                    "tcs_with_syr_link":        len([r for r in rows if r["has_syr_link"]]),
                    "tcs_without_syr_link":     len([r for r in rows if not r["has_syr_link"]]),
                    "tcs_with_swt_coverage":    len([r for r in rows if r["has_swt"]]),
                    "tcs_without_swt_coverage": len([r for r in rows if not r["has_swt"]]),
                    "unique_syr_ids_found":     len({s for r in rows for s in r["syr_ids"]}),
                    "unique_swr_ids_found":     len({s for r in rows for s in r["swr_ids"]}),
                    "unique_swt_ids_found":     len(all_swt_ids),
                    "related_syt_ids_found":    len(all_related_syt_ids),
                    "syt_swt_pair_count":       syt_swt_pair_count,
                    "syt_syt_pair_count":       syt_syt_pair_count,
                    "syt_swt_pair_count_resolved": syt_swt_pair_count_resolved,
                    "syt_syt_pair_count_resolved": syt_syt_pair_count_resolved,
                    "syt_content_fetched":      len(syt_content),
                    "swt_content_fetched":      len(swt_content),
                    "syr_content_fetched":      len(syr_content),
                    "swr_content_fetched":      len(swr_content),
                    "tc_not_downloaded":        len(not_downloaded_ids),
                },
                "traceability": rows,
            }
            overall_t.details.update(result["summary"])
            CACHE.record_operation_stat(
                "fetch_traceability", module=self.syt_module,
                item_count=len(self.selected_tc_ids),
                extra_count=len(all_swt_ids) + len(all_related_syt_ids),
                duration_seconds=time.perf_counter() - start,
                source="live" if _any_live else "cache",
                details={
                    "syr_count": result["summary"]["unique_syr_ids_found"],
                    "swr_count": result["summary"]["unique_swr_ids_found"],
                    "swt_count": len(all_swt_ids),
                    "related_syt_count": len(all_related_syt_ids),
                },
            )
            self.done.emit(result)

        except WorkerCancelled:
            overall_t.level = "ERROR"
            overall_t.details["error"] = "cancelled"
            self.error.emit("Cancelled by user.")
        except Exception as e:
            overall_t.level = "ERROR"
            overall_t.details["error"] = str(e)
            self.error.emit(str(e))


class DuplicateCheckWorker(QThread, _CancellableWorker):
    """Step 3 'Check Duplicates': embed each SYT test case's text and each
    of its linked SWT test cases' text, then compute cosine similarity per
    SYT<->SWT pair to flag likely copy-pasted/redundant coverage.

    See trek_similarity.py for the embedding/scoring logic itself (adapted
    from the RAG pipeline in the vehicleConvert AI test-generation project)
    -- this worker's job is just to: build the (syt_id, text, swt_id, text)
    pairs from the already-fetched traceability rows, load/save the
    embedding cache (trek_cache.sqlite3, content-addressed so identical
    text across different TCs is only ever embedded once), and run the
    comparison off the UI thread since it makes network calls.
    """
    progress = Signal(str)
    done     = Signal(object)   # trek_similarity.DuplicateCheckResult
    error    = Signal(str)

    def __init__(self, traceability_rows: list, settings: Optional[dict] = None):
        super().__init__()
        self.traceability_rows = traceability_rows
        # Merged dict of: jwt_token (mandatory, from the active project's
        # saved credential, see TrekMainWindow._current_jwt_token) +
        # bm25_weight/vec_weight/sim_* (from DuplicateCheckSettingsDialog,
        # entered fresh for each run). The LLM gateway URL itself is
        # hardcoded in trek_similarity.LLM_GATEWAY_URL, not part of settings.
        self.settings = settings or {}

    def run(self):
        with CACHE.batch_writes(), \
             LOG.timed("Duplicate Check", f"{len(self.traceability_rows)} traceability row(s)",
                        row_count=len(self.traceability_rows)) as overall_t:
            self._run_inner(overall_t)

    def _run_inner(self, overall_t):
        try:
            use_rag = self.settings.get("use_rag_score", True)
            use_llm = self.settings.get("use_llm_judge", False)
            include_syt_swt = self.settings.get("include_syt_swt", True)
            include_syt_syt = self.settings.get("include_syt_syt", True)
            if not use_rag and not use_llm:
                raise ValueError("Neither RAG scoring nor LLM verification is enabled.")

            self.progress.emit("Connecting to embedding service...")
            with LOG.timed("Duplicate Check", "Connect to embedding gateway"):
                client = trek_similarity.get_embeddings_client(
                    jwt_token=self.settings.get("jwt_token"),
                )

            pairs = []
            seen_related_pairs = set()   # dedup unordered (syt_id, related_id) -- avoid A-vs-B AND B-vs-A
            for row in self.traceability_rows:
                syt_content = row.get("syt_content")
                syt_text = trek_similarity.build_comparison_text(syt_content)
                syt_methods = trek_similarity.extract_method_call_sequence(syt_content)
                if include_syt_swt:
                    for swt_content in row.get("swt_content", []):
                        swt_id = swt_content.get("Key", "")
                        swt_text = trek_similarity.build_comparison_text(swt_content)
                        swt_methods = trek_similarity.extract_method_call_sequence(swt_content)
                        pairs.append((row["syt_id"], syt_text, swt_id, swt_text, syt_methods, swt_methods, "SWT", syt_content, swt_content))
                # Related SYT test cases (share a SYR with this one, see
                # FetchLinksWorker) are judged the same way, tagged so the
                # LLM prompt and results table label them distinctly.
                # Symmetric relationship -- SYT_A related to SYT_B also
                # means SYT_B is related to SYT_A, so without deduping by
                # the UNORDERED pair, both directions would be compared
                # (double the embedding/LLM cost for the same comparison).
                if include_syt_syt:
                    for related_content in row.get("related_syt_content", []):
                        related_id = related_content.get("Key", "")
                        pair_key = frozenset((row["syt_id"], related_id))
                        if pair_key in seen_related_pairs:
                            continue
                        seen_related_pairs.add(pair_key)
                        related_text = trek_similarity.build_comparison_text(related_content)
                        related_methods = trek_similarity.extract_method_call_sequence(related_content)
                        pairs.append((row["syt_id"], syt_text, related_id, related_text, syt_methods, related_methods, "related_syt", syt_content, related_content))

            if not pairs:
                self.done.emit(trek_similarity.DuplicateCheckResult())
                return

            if use_rag:
                self.raise_if_cancelled()
                rag_force_refresh = self.settings.get("rag_force_refresh", False)
                self.progress.emit(f"Loading cached embeddings for {len(pairs) * 2} texts...")
                all_texts = set()
                for _syt_id, syt_text, _swt_id, swt_text, _sm, _wm, _ct, *_tc_dicts in pairs:
                    if syt_text:
                        all_texts.add(syt_text)
                    if swt_text:
                        all_texts.add(swt_text)
                normalized_texts = list({
                    re.sub(r"\s+", " ", t.strip().lower()).rstrip("."): t for t in all_texts
                }.values())
                norm_to_raw = {re.sub(r"\s+", " ", t.strip().lower()).rstrip("."): t for t in normalized_texts}
                cached = (
                    {} if rag_force_refresh else
                    CACHE.get_embeddings(list(norm_to_raw.keys()), trek_similarity.EMBEDDING_MODEL)
                )
                embedding_cache = dict(cached)   # normalized_text -> vector

                self.progress.emit(f"Comparing {len(pairs)} SYT/SWT pairs "
                                    f"({len(normalized_texts) - len(cached)} new texts to embed)...")
                new_text_count = len(normalized_texts) - len(cached)
                per_text_cost: dict = {}
                rag_start = time.perf_counter()
                with LOG.timed("Duplicate Check", f"Score {len(pairs)} pair(s)",
                                pair_count=len(pairs), cached_embeddings=len(cached),
                                new_embeddings=new_text_count, rag_force_refresh=rag_force_refresh) as score_t:
                    result = trek_similarity.compare_syt_swt_pairs(
                        client, pairs, embedding_cache,
                        duplicate_threshold=self.settings.get(
                            "sim_duplicate", trek_similarity.SIMILARITY_DUPLICATE),
                        near_duplicate_threshold=self.settings.get(
                            "sim_near_duplicate", trek_similarity.SIMILARITY_NEAR_DUPLICATE),
                        similar_threshold=self.settings.get(
                            "sim_similar", trek_similarity.SIMILARITY_SIMILAR),
                        bm25_weight=self.settings.get(
                            "bm25_weight", trek_similarity.DEFAULT_BM25_WEIGHT),
                        vec_weight=self.settings.get(
                            "vec_weight", trek_similarity.DEFAULT_VEC_WEIGHT),
                        seq_weight=self.settings.get(
                            "seq_weight", trek_similarity.DEFAULT_SEQ_WEIGHT),
                        per_text_cost=per_text_cost,
                    )
                    result.cached_texts_embedded = len(cached)
                    score_t.details.update(
                        tokens_embedded=result.total_tokens_embedded,
                        texts_embedded=result.total_texts_embedded,
                        cached_texts_embedded=result.cached_texts_embedded,
                        cost_usd=result.total_cost_usd,
                    )
                result.rag_duration_seconds = time.perf_counter() - rag_start

                # Persist any newly-computed embeddings back to the cache so a
                # re-run (or a different traceability run sharing text) never
                # re-embeds the same content. Always persisted (even after a
                # forced refresh) so the cache reflects the freshest result.
                # cost_by_norm maps the SAME normalized-text keys as
                # new_entries, recording the REAL per-text cost so a future
                # cache-hit run can still report the true historical cost.
                new_entries = {k: v for k, v in embedding_cache.items() if k not in cached}
                if new_entries:
                    cost_by_norm = {
                        trek_similarity._normalize_for_exact_match(text): cost
                        for text, cost in per_text_cost.items()
                    }
                    CACHE.set_embeddings(new_entries, trek_similarity.EMBEDDING_MODEL, cost_by_text=cost_by_norm)

                # Historical cost = what was actually paid this run for new
                # embeddings + the REAL recorded cost of every cache-hit text
                # (not an estimate) -- so a fully-cached re-run still shows
                # the true cost of what it's displaying.
                cached_costs = CACHE.get_embedding_costs(list(cached.keys()), trek_similarity.EMBEDDING_MODEL)
                result.historical_embed_cost_usd = result.total_cost_usd + sum(cached_costs.values())
            else:
                self.progress.emit(f"RAG scoring disabled -- skipping embedding/BM25 for {len(pairs)} pair(s)...")
                with LOG.timed("Duplicate Check", f"Build {len(pairs)} bare pair(s) (RAG disabled)",
                                pair_count=len(pairs)):
                    result = trek_similarity.build_bare_pairs(pairs)

            if use_llm and result.pairs:
                self.raise_if_cancelled()
                llm_model = self.settings.get("llm_model", trek_similarity.CHAT_MODEL)
                llm_instructions = self.settings.get("llm_instructions", "")
                llm_force_refresh = self.settings.get("llm_force_refresh", False)
                cache_keys = [
                    trek_similarity.llm_judgment_cache_key(
                        p.syt_text, p.swt_text, p.counterpart_type, llm_model, llm_instructions
                    )
                    for p in result.pairs
                ]
                llm_cached = {} if llm_force_refresh else CACHE.get_llm_judgments(cache_keys)
                llm_cache = dict(llm_cached)
                # Pre-seed cost_cache with each cache-hit pair's REAL recorded
                # cost (not just newly-judged ones) so every pair -- cached or
                # fresh -- ends up with a real per-pair llm_cost_usd, letting
                # a merged/superset result sum true historical cost with no
                # double counting (see trek_similarity.merge_duplicate_check_results()).
                cost_cache: dict = {} if llm_force_refresh else dict(CACHE.get_llm_judgment_costs(cache_keys))

                self.progress.emit(f"Starting LLM verification of {len(result.pairs)} pair(s)...")
                llm_start = time.perf_counter()
                with LOG.timed("Duplicate Check", f"LLM verification of {len(result.pairs)} pair(s)",
                                pair_count=len(result.pairs), llm_force_refresh=llm_force_refresh) as llm_t:
                    trek_similarity.run_llm_judge_stage(
                        client, result,
                        model=llm_model, instructions=llm_instructions,
                        batch_size=self.settings.get("llm_batch_size", 1),
                        max_workers=self.settings.get("llm_max_workers", 20),
                        progress_cb=self.progress.emit, llm_cache=llm_cache,
                        cost_cache=cost_cache,
                        should_cancel=lambda: self.is_cancelled,
                    )
                    llm_t.details.update(
                        llm_calls=result.llm_calls, llm_cached_pairs=result.llm_cached_pairs,
                        llm_tokens=result.total_llm_tokens, llm_cost_usd=result.total_llm_cost_usd,
                    )
                result.llm_duration_seconds = time.perf_counter() - llm_start

                # Persist any newly-computed judgments so an identical pair
                # (same model/instructions) is never sent to the LLM twice.
                new_llm_entries = {k: v for k, v in llm_cache.items() if k not in llm_cached}
                if new_llm_entries:
                    CACHE.set_llm_judgments(new_llm_entries, cost_by_cache_key=cost_cache)

                # Historical cost = what was actually paid this run for new
                # judgments + the REAL recorded cost of every cache-hit pair.
                cached_llm_costs = CACHE.get_llm_judgment_costs(list(llm_cached.keys()))
                result.historical_llm_cost_usd = result.total_llm_cost_usd + sum(cached_llm_costs.values())

                # Whatever was judged (and persisted above) before the user
                # clicked Stop is kept, but the run itself is still reported
                # as cancelled rather than a normal completion.
                self.raise_if_cancelled()

            counts = result.summary_counts()
            overall_t.details.update(
                pair_count=len(pairs), **counts, skipped=len(result.skipped),
                tokens_embedded=result.total_tokens_embedded,
                texts_embedded=result.total_texts_embedded,
                cost_usd=result.total_cost_usd,
                llm_batch_requests=result.llm_calls,
                llm_pairs_judged=sum(1 for p in result.pairs if p.llm_verdict),
                llm_cached_pairs=result.llm_cached_pairs,
                llm_tokens=result.total_llm_tokens,
                llm_cost_usd=result.total_llm_cost_usd,
            )
            self.done.emit(result)
        except WorkerCancelled:
            overall_t.level = "ERROR"
            overall_t.details["error"] = "cancelled"
            self.error.emit("Cancelled by user.")
        except Exception as e:
            overall_t.level = "ERROR"
            overall_t.details["error"] = str(e)
            self.error.emit(str(e))


class FetchSytTcsWorker(QThread, _CancellableWorker):
    """Step 2: resolve SYT TC ids for the selected module, then auto-detect
    the real SYR bridge module(s) FROM THE ACTUAL OUT LINK DATA -- not by
    guessing a module name from the SYT subsystem's display name.

    Why: TREK's naming conventions between V-Model levels are inconsistent
    (e.g. "SYT - SWUpdate" is linked to SYR requirements filed under the
    "Infrastructure" functional area, not anything containing "SWUpdate").
    Guessing f"SYR - {subsystem}" or matching the subsystem string against
    module names is fundamentally unreliable. Instead:

      1. Fetch the SYT module's own links -- this gives the exact SYR_*
         requirement IDs the selected test cases actually reference
         (ground truth, no guessing).
      2. Derive the functional-area token from those IDs, e.g.
         "SYR_INFRA_310" -> "INFRA".
      3. Shortlist candidate SYR modules from the already-loaded module
         list whose normalized name contains that token (e.g.
         "SYR - Infrastructure" contains "infra").
      4. VERIFY each candidate live: fetch its own items and confirm at
         least one of the actually-referenced SYR IDs appears as a Key.
         Only verified modules are trusted -- this eliminates false
         positives from token collisions.

    Cache-aware via the shared _get_links_items() helper (same cache
    entries as FetchLinksWorker's Step 3 fetch, so selecting a module in
    Step 2 and then running Step 3 never re-fetches the same data twice).
    """
    progress = Signal(str)
    done  = Signal(list, list, dict, dict)   # (syt_ids, valid_syr_names, chapter_by_tc_id, syr_ids_by_tc)
    error = Signal(str)

    def __init__(self, syt_module, all_syr_module_names, force_refresh: bool = False):
        super().__init__()
        self.syt_module           = syt_module
        self.all_syr_module_names = all_syr_module_names
        self.force_refresh        = force_refresh

    def run(self):
        start = time.perf_counter()
        with LOG.timed("Load TCs", f"Resolve TCs for '{self.syt_module}'",
                        syt_module=self.syt_module, force_refresh=self.force_refresh) as overall_t:
            self._run_inner(overall_t, start)

    def _run_inner(self, overall_t, start):
        try:
            client = TrekExportLinksClient()
            fr = self.force_refresh

            # 1. Fetch SYT's own links -- gives both the TC ids AND the
            # real SYR ids those TCs are actually linked to.
            self.raise_if_cancelled()
            self.progress.emit(f"{'Refreshing' if fr else 'Loading'} SYT TC IDs for '{self.syt_module}'...")
            items, from_cache = _get_links_items(client, self.syt_module, 4, fr)
            if from_cache:
                self.progress.emit(f"SYT TC IDs for '{self.syt_module}' loaded from cache.")
            syt_ids = sorted({lnk["Key"] for lnk in items})
            referenced_syr_ids = {
                lnk["LinkKey"] for lnk in items
                if lnk.get("LinkType") == "OUT" and lnk.get("LinkKey", "").startswith("SYR_")
            }

            # Per-TC SYR ids, so the Step 2 list can show how many SYR
            # requirements each individual test case links to (the "SYR
            # links" column) -- not just the module-wide total.
            syr_ids_by_tc: Dict[str, set] = defaultdict(set)
            for lnk in items:
                if lnk.get("LinkType") == "OUT" and lnk.get("LinkKey", "").startswith("SYR_"):
                    syr_ids_by_tc[lnk["Key"]].add(lnk["LinkKey"])

            # 2. Fetch full TC content for every TC in the module upfront,
            # so the Step 2 list can be grouped by its real DOORS chapter
            # (Module_Path, e.g. "TestCaseSpecification.SWUpdate.Intake").
            # This is the same TC-content cache used by FetchLinksWorker
            # for Step 3, keyed globally by TC id -- so a TC's content is
            # never fetched twice regardless of which step triggers it
            # first. First load of a large module is slower (fetches ALL
            # TCs, not just the ones later selected); every subsequent
            # load/module switch reuses the cache.
            self.raise_if_cancelled()
            self.progress.emit(f"Resolving chapters for {len(syt_ids)} TCs...")
            not_downloaded: set = set()
            tc_content = _get_tc_content_cached(client, syt_ids, fr, failed_out=not_downloaded)
            chapter_by_tc_id = {
                tc_id: content.get("Module_Path", "") or "(no chapter)"
                for tc_id, content in tc_content.items()
            }
            for tc_id in not_downloaded:
                chapter_by_tc_id.setdefault(tc_id, NOT_DOWNLOADED_CHAPTER)
            if not_downloaded:
                overall_t.details["tc_text_not_downloaded"] = len(not_downloaded)

            syr_ids_by_tc_sorted = {tc_id: sorted(syrs) for tc_id, syrs in syr_ids_by_tc.items()}
            overall_t.details["tc_count"] = len(syt_ids)

            # A manually-confirmed mapping (see ModuleMappingDialog) always
            # wins, even with ZERO "SYR_"-prefixed referenced ids -- some
            # confirmed SYR-like modules (see _EXTRA_SYR_LIKE_MODULES) use
            # a completely different ID convention, so referenced_syr_ids
            # being empty must NOT bypass a manual mapping the user already
            # confirmed for this exact SYT module.
            manual_syr = get_manual_bridge_mapping(self.syt_module, "SYR")
            if not referenced_syr_ids and manual_syr is None:
                CACHE.record_operation_stat(
                    "fetch_testcases", module=self.syt_module, item_count=len(syt_ids),
                    duration_seconds=time.perf_counter() - start, source="cache" if from_cache else "live",
                )
                self.done.emit(syt_ids, [], chapter_by_tc_id, syr_ids_by_tc_sorted)
                return

            # 3-5. Resolve the real SYR module(s) the referenced SYR ids live
            # in. Priority:
            #   1) Manual mapping (user override) -- always wins.
            #   2) INDEX lookup (O(1) per id, no network) -- the fast path
            #      once "Build Index" has run for this project.
            #   3) Fallback: live candidate probing via
            #      _get_bridge_modules_cached, but ONLY for ids the index
            #      couldn't resolve (unbuilt index / brand-new ids), so
            #      nothing breaks when the index is missing or incomplete.
            self.raise_if_cancelled()
            if manual_syr is not None:
                valid_syr, syr_source = list(manual_syr), "manual"
            else:
                valid_syr = []
                resolved = INDEX.modules_for_ids(referenced_syr_ids, prefix="SYR")
                resolved -= _SYR_EXCLUDED_MODULES
                unresolved = INDEX.unresolved_ids(referenced_syr_ids, prefix="SYR")
                for name in resolved:
                    if name not in valid_syr:
                        valid_syr.append(name)

                if INDEX.is_built("SYR"):
                    # Index built -> unresolved ids are dead links to SYR
                    # requirements deleted from TREK. Ignore them (no module,
                    # no content, nothing to process); no probing.
                    syr_source = "index"
                    if unresolved:
                        LOG.log("Index",
                                f"Ignoring {len(unresolved)} dead SYR id(s) for "
                                f"'{self.syt_module}' (not in TREK): {unresolved[:5]}")
                    self.progress.emit(
                        f"Resolved SYR module(s) for '{self.syt_module}' from the index"
                        + (f" (ignored {len(unresolved)} dead link(s))" if unresolved else "")
                        + "."
                    )
                else:
                    # Index not built -> legacy live candidate probing.
                    probe_ids = set(unresolved) if unresolved else referenced_syr_ids
                    self.progress.emit(
                        f"Index not built -- probing SYR candidates for '{self.syt_module}'..."
                    )
                    probed, probe_source = _get_bridge_modules_cached(
                        client, self.syt_module, probe_ids, self.all_syr_module_names,
                        domain_type=2, force_refresh=fr, target_kind="SYR",
                        progress_cb=self.progress.emit, label="SYR",
                    )
                    for name in probed:
                        if name not in valid_syr:
                            valid_syr.append(name)
                    syr_source = "index+probe" if resolved else probe_source
            overall_t.details["syr_modules_found"] = len(valid_syr)
            overall_t.details["syr_source"] = syr_source

            CACHE.record_operation_stat(
                "fetch_testcases", module=self.syt_module, item_count=len(syt_ids),
                duration_seconds=time.perf_counter() - start, source="cache" if from_cache else "live",
            )
            self.done.emit(syt_ids, valid_syr, chapter_by_tc_id, syr_ids_by_tc_sorted)
        except WorkerCancelled:
            overall_t.level = "ERROR"
            overall_t.details["error"] = "cancelled"
            self.error.emit("Cancelled by user.")
        except Exception as e:
            overall_t.level = "ERROR"
            overall_t.details["error"] = str(e)
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Helper: subsystem-name normalization for SWT/SWIT auto-detection
# ---------------------------------------------------------------------------
def _normalize_subsystem(text: str) -> str:
    """Fold a subsystem/module name down to bare lowercase alphanumerics so
    naming-convention differences between modules don't break matching.

    TREK module names are not consistent about spacing, e.g.:
        SYT - Rear Window Heating
        SWT - RearWindowHeating        (no spaces)
        SWR - Rear_Window-Heating      (underscores/hyphens, hypothetically)
    A plain substring check ("rear window heating" in "rearwindowheating")
    fails on these even though they clearly refer to the same subsystem.
    Stripping all non-alphanumeric characters before comparing fixes this.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Modules known to hold real SYR-like requirement content but that don't
# follow the "SYR -" naming convention, so the prefix filter in
# _on_module_selected() would otherwise hide them entirely -- including
# them here just makes them SELECTABLE in "Edit SYR/SWT Mapping" (manual
# confirmation still required per SYT module, same as any other SYR).
_EXTRA_SYR_LIKE_MODULES = {
    "BMW_SP25_25_ZCU_ZMH_Generic_Stress_and_Robustness",
}

# Modules that START with "SYR_" (so they pass the prefix filter and get
# indexed) but do NOT represent a real SYR bridge for any specific SYT
# subsystem -- they are cross-cutting audit/stress/robustness modules whose
# requirements get referenced by many SYT modules as stray cross-links,
# polluting the SYR bridge detection with a huge unrelated module and
# diverting the SYR->SWR->SWT chain away from the real functional area.
# Excluded from both auto-detection AND the index-based SYR resolution.
_SYR_EXCLUDED_MODULES = {
    "SYR_AUD_MLB3_21_IBC_HCP4_Stress_and_Robustness",
}


def _area_tokens(ids: set) -> Dict[str, int]:
    """Derive functional-area token(s) from a set of TREK object IDs, e.g.
    'SYR_INFRA_1442' -> {'INFRA': 1}, 'SWR_INFRA_SWU_4225' -> {'INFRA_SWU': 1, 'INFRA': 1, 'SWU': 1}.

    ID format is PREFIX_AREA_NUMBER where AREA itself may contain further
    underscores (the leading prefix and trailing numeric segment are
    stripped). Both the FULL compound area (e.g. "INFRA_SWU") and each
    individual underscore-separated segment within it (e.g. "INFRA", "SWU")
    are returned as separate candidate tokens: module names don't always
    contain the compound as one contiguous substring once normalized (e.g.
    "SWR - Infrastructure SW Update" -> "swrinfrastructureswupdate" does
    NOT contain "infraswu" contiguously because "structure" sits in
    between "infra" and "swu"), but usually does contain at least one of
    the individual segments.

    Returns a dict of {token: frequency} rather than a bare set, where
    frequency = the number of distinct ids that produced this token. This
    lets callers prioritize the dominant subsystem when a batch of ids
    contains a handful of stray cross-references to an unrelated
    functional area (e.g. 128 "SYT - Rear Window Heating" test cases
    referencing RWH-prefixed SYR ids, plus 1 outlier TC that also happens
    to reference 3 SUPport-Functions-prefixed SYR ids) -- without this,
    a rare/coincidental token can dominate detection just as easily as
    the token that actually represents almost all the real data.
    """
    tokens: Dict[str, int] = defaultdict(int)
    for obj_id in ids:
        parts = obj_id.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            area_parts = parts[1:-1]
        elif len(parts) >= 2:
            area_parts = parts[1:]
        else:
            area_parts = []
        if area_parts:
            tokens["_".join(area_parts)] += 1   # full compound
            for part in area_parts:
                tokens[part] += 1               # each individual segment
    return dict(tokens)


def _subsystem_initials(module_name: str) -> str:
    """Return the initials of a module's subsystem words, e.g.
    "SYR - Rear Window Heating" -> "RWH", "SWT - RearWindowHeating" -> "R"
    (CamelCase without spaces only yields one "word" to split(), so this
    only helps when the module name has genuine word separators -- see
    _module_matches_token()'s CamelCase-splitting fallback for the other
    case).

    Used to catch functional-area tokens that are ACRONYMS of the
    subsystem name rather than truncated prefixes of it: e.g. an area
    token "RWH" (derived from a SYR id like SYR_RWH_366) has NO
    contiguous substring relationship with the normalized module name
    "rearwindowheating" (a plain _normalize_subsystem() substring check
    always misses it), but IS exactly the first-letter-of-each-word
    acronym of "Rear Window Heating". Truncation-style tokens like
    "INFRA" (from "Infrastructure") are already handled by the plain
    substring check in _detect_bridge_modules() and don't need this.
    """
    subsystem = re.sub(r"^(SYT|SYR|SWR|SWT|SWIT)\s*-\s*", "", module_name, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z]+", subsystem)
    return "".join(w[0] for w in words).upper()


def _detect_bridge_modules(client, referenced_ids: set, candidate_module_names: List[str],
                            domain_type: int, force_refresh: bool, progress_cb=None,
                            label: str = "candidate", run_link_memo: Optional[dict] = None) -> List[str]:
    """Auto-detect which module(s) from ``candidate_module_names`` actually
    contain the referenced ids -- used for both SYR and SWR bridge-module
    detection.

    Why this approach instead of guessing a module name from a subsystem
    string: TREK's naming conventions are not consistent between V-Model
    levels (e.g. a SYT subsystem named "SWUpdate" links to SYR/SWR
    requirements filed under an unrelated "Infrastructure" functional
    area). Deriving the functional-area token directly from real
    IDs already known to be referenced, then shortlisting + verifying
    candidates from the real module list, is reliable regardless of
    naming drift.

    Two refinements on top of the basic "derive a token, substring-match
    module names, accept the first candidate with any overlap" approach,
    both needed to fix a real false-positive: a "SYT - Rear Window
    Heating" selection whose 555 links reference 131 SYR ids, 128 of them
    RWH-prefixed and 3 of them a stray cross-reference to an unrelated
    SUP(port Functions)-prefixed area from a single outlier test case.
    "SUP" substring-matches "SYR - Support Functions" and got accepted
    outright, while "RWH" -- an ACRONYM of "Rear Window Heating", not a
    truncated prefix of it -- has no substring relationship with the
    normalized module name at all and was never even shortlisted:

      1. Shortlist candidates via BOTH a plain substring match (handles
         truncated-prefix tokens like "INFRA" for "Infrastructure") AND
         an acronym match (handles initials-style tokens like "RWH" for
         "Rear Window Heating" -- see _subsystem_initials()).
      2. Verify every shortlisted candidate live, but instead of
         accepting the FIRST one with any overlap, count how many
         referenced ids each candidate actually contains and return only
         the module(s) achieving the MAXIMUM overlap count. This makes
         detection immune to noisy stray cross-references (3 overlapping
         ids loses to 128) regardless of how a candidate was shortlisted,
         without needing the token-frequency heuristic to be perfect.

    Args:
        referenced_ids: object IDs (e.g. SYR_* or SWR_*) known to be
                        referenced by the level above (ground truth).
        candidate_module_names: full list of real module names at the
                                 target level (e.g. all "SYR -" or "SWR -"
                                 modules) to shortlist and verify against.
        domain_type: local_config_domain_type to probe candidates with.
        progress_cb: optional callable(str) for progress messages.
        label: human-readable name for progress messages (e.g. "SYR", "SWR").
    """
    if not referenced_ids:
        return []

    token_frequency = _area_tokens(referenced_ids)
    # Try higher-frequency tokens first (purely cosmetic for the progress
    # messages -- overlap-count verification below is what actually
    # decides the winner, so ordering doesn't affect correctness).
    ordered_tokens = sorted(token_frequency, key=lambda t: token_frequency[t], reverse=True)

    candidates: List[str] = []
    for token in ordered_tokens:
        token_norm = _normalize_subsystem(token)
        if not token_norm:
            continue
        for n in candidate_module_names:
            if n in candidates:
                continue
            if token_norm in _normalize_subsystem(n) or token.upper() == _subsystem_initials(n):
                candidates.append(n)

    # Verify all shortlisted candidates CONCURRENTLY instead of one-at-a-
    # time. Each candidate is an independent Export/Links call (~20s live
    # per module against TREK), so probing e.g. 9 candidates sequentially
    # cost ~190s -- by far the dominant term in a "Build Traceability" run.
    # These calls have no dependency on each other (we only intersect each
    # module's own keys against referenced_ids afterwards), so they run in
    # parallel via _parallel_fetch_per_module (fresh client per worker, see
    # its docstring for the SSPI-auth thread-safety rationale). Cache hits
    # short-circuit inside _get_links_items, so already-cached candidates
    # stay ~instant.
    total = len(candidates)
    if progress_cb:
        progress_cb(f"Verifying {total} {label} candidate(s) concurrently...")

    # Per-RUN in-memory memo of candidate link fetches. A single "Build
    # Traceability" run detects the SWR bridge once PER SYR module, and
    # different SYR modules shortlist heavily-overlapping candidate lists.
    # Under Force Refresh (force_refresh=True) the persistent cache is
    # bypassed, so without this memo the SAME candidate module was fetched
    # LIVE once per SYR module -- e.g. the log showed SWR_160/SWR_720/... 
    # fetched twice (~130s wasted) for a 2-SYR selection. The memo makes
    # each unique candidate probed at most ONCE per run regardless of how
    # many SYR modules shortlist it. Keyed by (module_name, domain_type).
    if run_link_memo is None:
        run_link_memo = {}

    to_probe = [n for n in candidates if (n, domain_type) not in run_link_memo]

    def _probe(c, n):
        items, from_cache = _get_links_items(c, n, domain_type, force_refresh, timeout=160)
        return items

    if to_probe:
        for name, items in _parallel_fetch_per_module(
            to_probe, _probe,
            max_workers=5,   # kept at the historically-stable level -- higher
                             # concurrency has triggered transient Windows
                             # SSPI "bad parameter or other API misuse"
                             # failures under many simultaneous Negotiate
                             # handshakes. This path is now a rare fallback
                             # (the id->module index resolves almost
                             # everything), so 5 costs little.
        ):
            run_link_memo[(name, domain_type)] = items

    overlap_counts: Dict[str, int] = {}
    for name in candidates:
        items = run_link_memo.get((name, domain_type), [])
        keys_in_module = {lnk.get("Key") for lnk in items}
        overlap = len(keys_in_module & referenced_ids)
        if overlap > 0:
            overlap_counts[name] = overlap

    if not overlap_counts:
        return []

    best_overlap = max(overlap_counts.values())
    return [name for name, count in overlap_counts.items() if count == best_overlap]


def get_manual_bridge_mapping(source_module: str, target_kind: str) -> Optional[List[str]]:
    """Return the user-confirmed bridge module list for (source_module,
    target_kind) if one has been saved via ModuleMappingDialog, else None.
    Exposed at module level (not nested in _get_bridge_modules_cached) so
    the GUI can check "does a manual mapping already exist?" without
    needing a client/referenced_ids/candidate list on hand.
    """
    key = trek_cache.key_bridge_map_manual(PROJECT_ID, CAMPAIGN_ID, source_module, target_kind)
    cached = CACHE.get_blob(key)
    if cached is None:
        return None
    modules, _updated_at = cached
    return modules


def set_manual_bridge_mapping(source_module: str, target_kind: str, modules: List[str]) -> None:
    """Persist a user-confirmed bridge module list for (source_module,
    target_kind), saved via ModuleMappingDialog. This takes priority over
    auto-detection (see _get_bridge_modules_cached()) and is NOT cleared
    by 'Force Refresh' -- only by the user explicitly editing the mapping
    again."""
    key = trek_cache.key_bridge_map_manual(PROJECT_ID, CAMPAIGN_ID, source_module, target_kind)
    CACHE.set_blob(key, modules)


def _get_bridge_modules_cached(client, source_module: str, referenced_ids: set,
                                candidate_module_names: List[str], domain_type: int,
                                force_refresh: bool, target_kind: str = "candidate",
                                progress_cb=None, label: str = "candidate",
                                run_link_memo: Optional[dict] = None) -> "tuple[List[str], str]":
    """Cache-aware wrapper around _detect_bridge_modules(), persisted per
    SOURCE module (e.g. "SYR - Rear Window Heating") rather than per
    traceability RUN or per selected-TC subset.

    Why source-module scoping matters, not just for speed but for
    correctness: the right bridge module(s) for a source module depend
    only on that source module's own full link content, never on which
    specific test cases a user happens to have selected. An earlier
    version of this detection derived its `referenced_ids` from only the
    currently SELECTED test cases (as a perf shortcut, to avoid probing
    every candidate module live on every run) -- but a small/unlucky
    selection could then let a handful of stray cross-references to an
    unrelated functional area outvote the genuine bridge module (see
    _detect_bridge_modules()'s docstring for the real "SYT - Rear Window
    Heating" incident this caused: 3 stray SWR_SUP_* ids from one outlier
    test case out-detected the correct SWR module reachable by the other
    226). Persistently caching the result per source module -- using that
    module's ENTIRE link export, not a selection-scoped slice -- fixes
    both problems at once: the answer no longer depends on selection size,
    and detection (which may involve probing dozens of candidate modules
    live) only ever runs once per source module for the lifetime of the
    cache, not once per "Build Traceability" click.

    A THIRD tier sits above both auto-detection and its cache: a
    user-confirmed manual mapping (see get_manual_bridge_mapping() /
    set_manual_bridge_mapping()), for cases automatic detection cannot
    reliably solve at all -- e.g. "SYT - Rear Wiper" links to SYR ids
    prefixed "RWW" (German "Wischen und Waschen" / wipe-and-wash), which
    has no textual relationship whatsoever to the module name
    "SYR - Rear Wiper" (fails both substring and acronym matching, and
    would need probing every one of ~20 candidate SYR modules live to
    even have a chance of finding it by overlap alone). The manual
    mapping, once saved via ModuleMappingDialog, is checked FIRST and
    is never overwritten by auto-detection or invalidated by Force
    Refresh -- only by the user editing it again.

    Returns (bridge_module_names, source) where source is one of
    "manual", "cache", or "live".
    """
    manual = get_manual_bridge_mapping(source_module, target_kind)
    if manual is not None:
        return manual, "manual"

    key = trek_cache.key_bridge_map(PROJECT_ID, CAMPAIGN_ID, source_module, target_kind)
    if not force_refresh:
        cached = CACHE.get_blob(key)
        if cached is not None:
            modules, _updated_at = cached
            return modules, "cache"

    modules = _detect_bridge_modules(
        client, referenced_ids, candidate_module_names,
        domain_type=domain_type, force_refresh=force_refresh,
        progress_cb=progress_cb, label=label, run_link_memo=run_link_memo,
    )
    CACHE.set_blob(key, modules)
    return modules, "live"


# ---------------------------------------------------------------------------
# Traceability tree rendering constants/helpers
# ---------------------------------------------------------------------------
# Reused per node instead of constructing a QFont per item -- a big module
# builds 50k+ nodes, and each QFont() does a family lookup.
_FONT_TREE_TOP         = QFont("Consolas", 11, QFont.Bold)
_FONT_TREE_CHILD       = QFont("Consolas", 10)
_FONT_TREE_CHILD_BOLD  = QFont("Consolas", 10, QFont.Bold)

# Above this many nodes, don't auto-expand the tree after a run -- laying
# out every node blocks the UI thread (see TrekMainWindow._expand_all_tree).
_TREE_AUTO_EXPAND_LIMIT = 3000

# Marks a top-level SYT node whose children have already been built lazily.
_ROLE_CHILDREN_BUILT = Qt.UserRole + 1


def _estimate_tree_nodes(rows: List[dict]) -> int:
    """Total nodes the tree WOULD have fully expanded -- computed from the
    traceability data rather than by walking items, since children are
    only built on demand (see TrekMainWindow._build_tree_row)."""
    total = 0
    for row in rows:
        total += 1
        for entry in row.get("chain", []):
            total += 1 + len(entry.get("swr_ids", ())) + len(entry.get("swt_ids", ())) \
                     + len(entry.get("related_syt_ids", ()))
    return total


# ---------------------------------------------------------------------------
# Helper: status legend tooltip (RAG classification + LLM verdict meanings)
# ---------------------------------------------------------------------------
def _status_legend_html() -> str:
    """Rich-text tooltip explaining what every RAG classification and LLM
    verdict status actually means -- attached to the SYT/SWT content
    headers in SideBySideResultsWidget so the meaning is always one hover
    away, right where the statuses are shown."""
    rag_rows, llm_rows = [], []
    for key, (label, color) in trek_similarity.CLASSIFICATION_LABELS.items():
        color = _themed_classification_color(key, color)
        explanation = trek_similarity.CLASSIFICATION_EXPLANATIONS.get(key, "")
        rag_rows.append(f'<tr><td style="color:{color};font-weight:bold;white-space:nowrap">{label}</td>'
                         f'<td>&nbsp;{explanation}</td></tr>')
    for key, (label, color) in trek_similarity.LLM_VERDICT_LABELS.items():
        color = _themed_classification_color(key, color)
        explanation = trek_similarity.LLM_VERDICT_EXPLANATIONS.get(key, "")
        llm_rows.append(f'<tr><td style="color:{color};font-weight:bold;white-space:nowrap">{label}</td>'
                         f'<td>&nbsp;{explanation}</td></tr>')
    return (
        '<div style="max-width:420px">'
        '<b>RAG classification</b> (hybrid BM25/vector/sequence score)<br>'
        f'<table cellspacing="4">{"".join(rag_rows)}</table>'
        '<b>LLM verdict</b> (optional, reads actual test logic)<br>'
        f'<table cellspacing="4">{"".join(llm_rows)}</table>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Helper: format TC content as HTML
# ---------------------------------------------------------------------------
def _tc_to_html(tc: dict, level_color: str = ACCENT) -> str:
    def _block(label, text):
        if not text or text.strip() in ("", "not used"):
            return ""
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted = escaped.replace("\n", "<br>")
        return (
            f'<div style="margin-bottom:20px;padding-top:12px;'
            f'border-top:1px solid {BORDER}">'
            f'<div style="color:{level_color};font-weight:bold;font-size:11px;'
            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">{label}</div>'
            f'<div style="background:{CODE_BG};border-radius:5px;padding:10px;'
            f'font-family:Consolas,monospace;font-size:12px;color:{TEXT};'
            f'white-space:pre-wrap">{formatted}</div>'
            f'</div>'
        )

    name   = tc.get("Name", "").strip()
    key    = tc.get("Key", "")
    mpath  = tc.get("Module_Path", "")
    doors  = tc.get("Doors_Path", "")

    html = (
        f'<body style="background-color:{PANEL_BG};color:{TEXT}">'
        f'<div style="padding:4px">'
        f'<div style="font-size:15px;font-weight:bold;color:{TEXT};margin-bottom:2px">'
        f'{name}</div>'
        f'<div style="color:{TEXT_DIM};font-size:11px;margin-bottom:12px">'
        f'{key} &nbsp;|&nbsp; {mpath}</div>'
    )
    html += _block("Pre-Condition",  tc.get("PreCondition", ""))
    html += _block("Procedure",      tc.get("Procedure", ""))
    html += _block("Post-Condition", tc.get("Postcondition", ""))
    html += _block("Expected Result",tc.get("Expected_result", ""))
    if doors:
        html += (f'<div style="margin-top:8px;color:{TEXT_DIM};font-size:11px">'
                 f'🔗 <a href="{doors}" style="color:{TEXT_DIM}">{doors}</a></div>')
    html += '</div></body>'
    return html


# ---------------------------------------------------------------------------
# Helper: format a requirement (SYR/SWR, from Export/Requirements) as HTML
# ---------------------------------------------------------------------------
def _req_to_html(req: dict, level_color: Optional[str] = None) -> str:
    """Render a requirement object returned by /Export/Requirements.

    Full field shape confirmed against live TREK data for SYR requirements
    (e.g. SYR_INFRA_1442, SYR_INFRA_2058) -- the same shape applies to SWR
    requirements from the same endpoint:

      AbsoluteNumber, ArtefactType, BindingRegulation, Carline,
      ChapterNumber, CLS, CreatedBy, CreatedTime, Discipline, Function,
      FunctionalArea, Id, ImplementationStatus, ImportTime, Key, Maturity,
      ModifiedBy, ModifiedTime, ModuleName, Name (requirement text itself),
      Release, ReviewComment, SecurityAndPrivacy, SIL, SpecificationId,
      SPL, TestCoverage, TestSeverity, Type, Url (DOORS Rational URN),
      Variant, VariantCarline, VerificationMethod.

    All of the above (not just ReviewComment) are surfaced here, grouped
    into: header/text, classification chips, engineering attributes,
     variant/release scope, review history, and the DOORS link.
    """
    if level_color is None:
        level_color = REQ_COLOR
    def _chip(label, value):
        # Rendered as its own block-level line (not an inline-block span)
        # because Qt's QTextEdit rich-text renderer does not reliably wrap
        # inline-block spans the way a browser does -- attributes would
        # otherwise run together on one line with no visual separator
        # (e.g. "Maturity: project_acceptedImpl. Status: implemented").
        # A <div> per attribute guarantees one attribute per line.
        if not value or str(value).strip() in ("", "#TBD#"):
            return ""
        return (
            f'<div style="background:{CODE_BG};border-radius:3px;padding:3px 8px;'
            f'margin-bottom:3px;font-size:11px;color:{DIM_TEXT}">{label}: '
            f'<span style="color:{TEXT}">{value}</span></div>'
        )

    def _block(label, text):
        if not text or str(text).strip() in ("", "not used"):
            return ""
        escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted = escaped.replace("\n", "<br>")
        return (
            f'<div style="margin-top:12px">'
            f'<div style="color:{level_color};font-weight:bold;font-size:11px;'
            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">{label}</div>'
            f'<div style="background:{CODE_BG};border-radius:4px;padding:8px;'
            f'font-family:Consolas,monospace;font-size:12px;color:{TEXT};'
            f'white-space:pre-wrap">{formatted}</div>'
            f'</div>'
        )

    def _section(label):
        return (f'<div style="color:{level_color};font-weight:bold;font-size:10px;'
                f'text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px 0;'
                f'border-top:1px solid {BORDER};padding-top:10px">{label}</div>')

    key   = req.get("Key", "")
    text  = req.get("Name", "").strip()
    url   = req.get("Url", "")

    html = (
        f'<body style="background-color:{PANEL_BG};color:{TEXT}">'
        f'<div style="padding:4px">'
        f'<div style="color:{level_color};font-size:11px;font-weight:bold;'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">'
        f'{key} &nbsp;<span style="color:{TEXT_DIM};font-weight:normal">'
        f'({req.get("ModuleName", "")})</span></div>'
    )
    html += (
        f'<div style="background:{CODE_BG};border-radius:4px;padding:10px;'
        f'font-size:13px;color:{TEXT};white-space:pre-wrap;margin-bottom:6px">{text}</div>'
    ) if text else ""

    # Classification
    html += '<div style="margin-bottom:6px">'
    html += _chip("Area", req.get("FunctionalArea"))
    html += _chip("Type", req.get("Type"))
    html += _chip("Discipline", req.get("Discipline"))
    html += _chip("Maturity", req.get("Maturity"))
    html += _chip("Impl. Status", req.get("ImplementationStatus"))
    html += '</div>'

    # Engineering attributes
    html += _section("Engineering Attributes")
    html += '<div style="margin-bottom:6px">'
    html += _chip("Verification", req.get("VerificationMethod"))
    html += _chip("Test Coverage", req.get("TestCoverage"))
    html += _chip("Test Severity", req.get("TestSeverity"))
    html += _chip("SIL", req.get("SIL"))
    html += _chip("SPL", req.get("SPL"))
    html += _chip("Security/Privacy", req.get("SecurityAndPrivacy"))
    html += _chip("Binding Regulation", req.get("BindingRegulation"))
    html += _chip("Chapter", req.get("ChapterNumber"))
    html += _chip("Abs. Number", req.get("AbsoluteNumber"))
    html += _chip("Spec Id", req.get("SpecificationId"))
    html += _chip("Artefact Type", req.get("ArtefactType"))
    html += '</div>'

    # Variant / release scope
    html += _section("Variant / Release Scope")
    html += '<div style="margin-bottom:6px">'
    html += _chip("Release", req.get("Release"))
    html += _chip("Function", req.get("Function"))
    html += '</div>'
    html += _block("Variant", req.get("Variant", ""))
    html += _block("Carline", req.get("Carline", ""))
    html += _block("Variant Carline", req.get("VariantCarline", ""))

    # Provenance
    html += _section("Provenance")
    html += '<div style="margin-bottom:6px">'
    html += _chip("Created By", req.get("CreatedBy"))
    html += _chip("Created", (req.get("CreatedTime") or "")[:10])
    html += _chip("Modified By", req.get("ModifiedBy"))
    html += _chip("Modified", (req.get("ModifiedTime") or "")[:10])
    html += '</div>'

    # Review history
    html += _block("Review Comment", req.get("ReviewComment", ""))

    if url:
        html += (f'<div style="margin-top:12px;color:{TEXT_DIM};font-size:11px">'
                 f'🔗 <a href="{url}" style="color:{TEXT_DIM}">{url}</a></div>')
    html += '</div></body>'
    return html


# ---------------------------------------------------------------------------
# View Database dialog
# ---------------------------------------------------------------------------
class ProjectSetupDialog(QDialog):
    """Add/edit a saved TREK project: Project ID / Campaign ID / Config ID
    + a friendly display name, PLUS a JWT Token for the LLM embedding
    gateway used by "Check Duplicates" (SYT vs SWT similarity). Used both
    for the first-run setup (blocking, no Cancel) and for "+ Add
    Project..." / "Edit Project..." from the header dropdown (normal,
    cancellable).

    ALL fields including the JWT Token are required and validated on Save
    -- discovering duplicate SYT/SWT test cases is this application's core
    purpose, not an optional add-on, so a project cannot be created
    without the credential needed to actually do that. The LLM gateway
    URL itself is hardcoded (see trek_similarity.LLM_GATEWAY_URL) and not
    shown here since there is only ever one gateway to talk to.

    The Database Path field is OPTIONAL: leaving it blank uses the app's
    default cache location (see trek_paths.data_dir()). Setting it points
    this project's trek_cache.sqlite3 (module lists, TC content, embeddings,
    duplicate-check results, etc.) at a custom file -- typically a shared
    network path so a team collaborates through one cache -- without
    affecting any other saved project.
    """

    def __init__(self, parent=None, existing: Optional[dict] = None, allow_cancel: bool = True):
        super().__init__(parent)
        self.setWindowTitle("TREK Project Setup" if existing is None else "Edit TREK Project")
        self.resize(460, 400)
        self._existing = existing
        self._result: Optional[dict] = None
        self._build_ui(allow_cancel)
        if existing:
            self._name_edit.setText(existing.get("name", ""))
            self._project_edit.setText(str(existing.get("project_id", "")))
            self._campaign_edit.setText(str(existing.get("campaign_id", "")))
            self._config_edit.setText(str(existing.get("config_id", "")))
            self._jwt_edit.setText(existing.get("jwt_token", ""))
            self._db_path_edit.setText(existing.get("db_path", ""))

    def _build_ui(self, allow_cancel: bool):
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Enter your TREK project details. You can add more projects "
            "later and switch between them from the header."
            if self._existing is None else
            "Update this project's TREK connection details."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-bottom:6px;")
        lay.addWidget(intro)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. BMW ZIM Rear")
        form.addRow("Display Name:", self._name_edit)

        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("e.g. 607")
        form.addRow("Project ID:", self._project_edit)

        self._campaign_edit = QLineEdit()
        self._campaign_edit.setPlaceholderText("e.g. 102766584")
        form.addRow("Campaign ID:", self._campaign_edit)

        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("e.g. 8579")
        form.addRow("Config ID:", self._config_edit)

        lay.addLayout(form)

        jwt_lbl = QLabel("Duplicate Detection (required)")
        jwt_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:12px;margin-top:12px;")
        lay.addWidget(jwt_lbl)
        jwt_hint = QLabel(
            "This application's purpose is discovering duplicate SYT/SWT "
            "test cases -- a JWT Token for the LLM embedding gateway is "
            "required for every project."
        )
        jwt_hint.setWordWrap(True)
        jwt_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-bottom:4px;")
        lay.addWidget(jwt_hint)

        jwt_form = QFormLayout()
        self._jwt_edit = QLineEdit()
        self._jwt_edit.setPlaceholderText("Required")
        self._jwt_edit.setEchoMode(QLineEdit.Password)
        jwt_form.addRow("JWT Token:", self._jwt_edit)

        self._show_token_chk = QCheckBox("Show token")
        self._show_token_chk.toggled.connect(
            lambda checked: self._jwt_edit.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        jwt_form.addRow("", self._show_token_chk)
        lay.addLayout(jwt_form)

        db_lbl = QLabel("Cache Database (optional)")
        db_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:12px;margin-top:12px;")
        lay.addWidget(db_lbl)
        db_hint = QLabel(
            "Leave blank to use the default cache location. Set a custom "
            "path (e.g. a shared network folder) to point this project's "
            "cache -- modules, test-case content, embeddings, duplicate-check "
            "results -- at a specific trek_cache.sqlite3 file, e.g. to share "
            "it with teammates. Other projects are unaffected."
        )
        db_hint.setWordWrap(True)
        db_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-bottom:4px;")
        lay.addWidget(db_hint)

        db_row = QHBoxLayout()
        self._db_path_edit = QLineEdit()
        self._db_path_edit.setPlaceholderText("Default cache location")
        db_row.addWidget(self._db_path_edit, 1)
        btn_load_db = QPushButton("Load...")
        btn_load_db.setObjectName("btn_secondary")
        btn_load_db.setToolTip("Open an existing cache database file.")
        btn_load_db.clicked.connect(self._on_load_db_path)
        db_row.addWidget(btn_load_db)
        btn_new_db = QPushButton("New...")
        btn_new_db.setObjectName("btn_secondary")
        btn_new_db.setToolTip("Choose where to create a new cache database file.")
        btn_new_db.clicked.connect(self._on_new_db_path)
        db_row.addWidget(btn_new_db)
        lay.addLayout(db_row)

        lay.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if allow_cancel:
            btn_cancel = QPushButton("Cancel")
            btn_cancel.clicked.connect(self.reject)
            btn_row.addWidget(btn_cancel)

        btn_save = QPushButton("Save")
        btn_save.setObjectName("btn_success")
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_save)
        lay.addLayout(btn_row)

    def _on_load_db_path(self):
        """Let the user pick an existing cache database file."""
        start_dir = self._db_path_edit.text().strip() or str(trek_paths.data_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Existing Cache Database", start_dir,
            "SQLite Database (*.sqlite3);;All Files (*)",
        )
        if path:
            self._db_path_edit.setText(path)

    def _on_new_db_path(self):
        """Let the user choose where to create a new cache database file."""
        start_dir = self._db_path_edit.text().strip() or str(trek_paths.data_dir())
        path, _ = QFileDialog.getSaveFileName(
            self, "Create New Cache Database", start_dir,
            "SQLite Database (*.sqlite3);;All Files (*)",
        )
        if path:
            self._db_path_edit.setText(path)

    def _on_save(self):
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a display name for this project.")
            return

        try:
            project_id  = int(self._project_edit.text().strip())
            campaign_id = int(self._campaign_edit.text().strip())
            config_id   = int(self._config_edit.text().strip())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid IDs",
                "Project ID, Campaign ID, and Config ID must all be whole numbers."
            )
            return

        jwt_token = self._jwt_edit.text().strip()
        if not jwt_token:
            QMessageBox.warning(
                self, "Missing JWT Token",
                "A JWT Token is required for every project -- discovering "
                "duplicate SYT/SWT test cases is this application's core "
                "purpose."
            )
            return

        self._result = {
            "name": name,
            "project_id": project_id,
            "campaign_id": campaign_id,
            "config_id": config_id,
            "jwt_token": jwt_token,
            "db_path": self._db_path_edit.text().strip(),
        }
        self.accept()

    def result_data(self) -> Optional[dict]:
        return self._result


class ScoringSettingsWidget(QWidget):
    """Reusable BM25/vector/sequence weight + similarity threshold controls,
    shared by DuplicateCheckSettingsDialog (real 'Check Duplicates' runs)
    and TestAlgorithmDialog (manual synthetic-data sandbox) so both use
    IDENTICAL controls and IDENTICAL validation -- there is only one
    implementation of "how do these settings behave" to keep in sync.

    Three weights always sum to 1.0:
      - BM25 (lexical overlap -- shared words/DFR tokens/method names)
      - Vector (semantic embedding similarity)
      - Sequence (order-sensitive method-call sequence similarity --
        compares the ACTUAL scripted Component.method(...) calls in the
        Procedure, see trek_similarity.extract_method_call_sequence()).
        Only applies to pairs where both sides have a scripted procedure;
        automatically excluded (weights renormalized) for prose-only
        test cases -- see ensemble_score()'s docstring.
    Adjusting one spinbox proportionally redistributes the remaining
    budget across the other two, so all three always sum to 1.0.
    """

    def __init__(self, parent=None, current: Optional[dict] = None):
        super().__init__(parent)
        self._updating = False
        self._build_ui()
        current = current or {}
        # Set _updating=True while pre-filling initial values so
        # _on_weight_changed's proportional-redistribution logic doesn't
        # fire and corrupt them (e.g. setting bm25 first would otherwise
        # immediately redistribute vec/seq before they're set to their own
        # intended initial values).
        self._updating = True
        self._bm25_spin.setValue(current.get("bm25_weight", trek_similarity.DEFAULT_BM25_WEIGHT))
        self._vec_spin.setValue(current.get("vec_weight", trek_similarity.DEFAULT_VEC_WEIGHT))
        self._seq_spin.setValue(current.get("seq_weight", trek_similarity.DEFAULT_SEQ_WEIGHT))
        self._updating = False
        self._dup_spin.setValue(current.get("sim_duplicate", trek_similarity.SIMILARITY_DUPLICATE))
        self._near_spin.setValue(current.get("sim_near_duplicate", trek_similarity.SIMILARITY_NEAR_DUPLICATE))
        self._similar_spin.setValue(current.get("sim_similar", trek_similarity.SIMILARITY_SIMILAR))

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        weight_lbl = QLabel("Hybrid Score Weighting -- 3 Detection Mechanisms")
        weight_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:12px;")
        lay.addWidget(weight_lbl)

        # Compact one-liner per mechanism, laid out as 3 columns so this
        # stays short/square instead of a tall paragraph block; the full
        # detailed explanation (with examples/caveats) is available as a
        # hover tooltip on each column for anyone who wants more depth.
        mechanisms_row = QHBoxLayout()

        bm25_col = QLabel("<b>1. BM25</b><br><span style='font-size:10px'>Shared vocabulary/tokens</span>")
        bm25_col.setToolTip(
            "BM25 (lexical) -- measures shared VOCABULARY: how many of the "
            "same words, DFR tokens, and identifiers appear in both test "
            "cases, weighted so rare/distinctive words count more than "
            "common ones.\n\nGood at catching copy-paste text; weak when "
            "tests describe the same idea in different words."
        )

        vector_col = QLabel("<b>2. Vector</b><br><span style='font-size:10px'>Semantic meaning (AI)</span>")
        vector_col.setToolTip(
            "Vector (semantic) -- measures MEANING via AI embeddings: "
            "converts each test case's text into a numeric representation "
            "and compares them mathematically (cosine similarity), so "
            "reworded/paraphrased tests that mean the same thing still "
            "score high even with zero shared words.\n\nCan also score two "
            "DIFFERENT tests on the same topic as similar, since it "
            "captures general meaning, not exact intent."
        )

        seq_col = QLabel("<b>3. Sequence</b><br><span style='font-size:10px'>Method-call order</span>")
        seq_col.setToolTip(
            "Sequence (method calls) -- measures STRUCTURE: extracts the "
            "ordered list of scripted Component.method(...) calls from the "
            "Procedure and compares the two sequences directly (same "
            "calls, same order = high score).\n\nThe most literal 'was "
            "this copy-pasted' signal, since genuinely different tests "
            "almost never call the exact same methods in the exact same "
            "order. Only applies when BOTH test cases have a scripted "
            "procedure -- for prose-only test cases this signal is "
            "skipped and BM25/Vector automatically absorb its share of "
            "the weight."
        )

        for col in (bm25_col, vector_col, seq_col):
            col.setStyleSheet(
                f"background:{CODE_BG};border-radius:4px;padding:6px;color:{TEXT};font-size:11px;"
            )
            col.setWordWrap(True)
            mechanisms_row.addWidget(col)
        lay.addLayout(mechanisms_row)

        mechanisms_hint = QLabel("💡 Hover a box above for a detailed explanation of each mechanism.")
        mechanisms_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;margin-bottom:6px;")
        lay.addWidget(mechanisms_hint)

        weight_form = QFormLayout()
        self._bm25_spin = self._make_weight_spin()
        self._bm25_spin.valueChanged.connect(lambda v: self._on_weight_changed(self._bm25_spin, v))
        weight_form.addRow("BM25 (lexical) weight:", self._bm25_spin)

        self._vec_spin = self._make_weight_spin()
        self._vec_spin.valueChanged.connect(lambda v: self._on_weight_changed(self._vec_spin, v))
        weight_form.addRow("Vector (semantic) weight:", self._vec_spin)

        self._seq_spin = self._make_weight_spin()
        self._seq_spin.valueChanged.connect(lambda v: self._on_weight_changed(self._seq_spin, v))
        weight_form.addRow("Sequence (method calls) weight:", self._seq_spin)
        lay.addLayout(weight_form)

        weight_hint = QLabel(
            "All three weights always sum to 1.0 -- changing one "
            "proportionally redistributes the rest. Ensemble score = "
            "BM25_weight×BM25 + Vector_weight×Vector + Sequence_weight× "
            "Sequence, always -- your weighting is never silently "
            "overridden. Sequence similarity only applies when BOTH test "
            "cases have a scripted procedure; for prose-only pairs, BM25 "
            "and Vector are automatically renormalized to fill the gap."
        )
        weight_hint.setWordWrap(True)
        weight_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-bottom:8px;")
        lay.addWidget(weight_hint)

        thresh_lbl = QLabel("Similarity Classification Thresholds")
        thresh_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:12px;margin-top:6px;")
        lay.addWidget(thresh_lbl)

        thresh_form = QFormLayout()
        self._dup_spin = self._make_threshold_spin()
        thresh_form.addRow("🔴 Duplicate ≥", self._dup_spin)
        self._near_spin = self._make_threshold_spin()
        thresh_form.addRow("🟠 Near-duplicate ≥", self._near_spin)
        self._similar_spin = self._make_threshold_spin()
        thresh_form.addRow("🟡 Similar ≥", self._similar_spin)
        lay.addLayout(thresh_form)

        hint = QLabel(
            "Thresholds apply to the final ensemble score (0.0-1.0). Scores "
            "at or above 'Duplicate' are flagged red, down to 'Similar' "
            "(yellow); anything lower is green (distinct)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-top:4px;")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setObjectName("btn_secondary")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    @staticmethod
    def _make_weight_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        return spin

    def _on_weight_changed(self, changed_spin, new_value):
        """Keep bm25/vec/seq weights summing to 1.0 by proportionally
        redistributing the remaining budget across the other two spins
        (proportional to their CURRENT relative values, so e.g. raising
        BM25 shrinks vector and sequence by the same ratio they already
        had to each other, rather than always taking from just one)."""
        if self._updating:
            return
        self._updating = True
        try:
            others = [s for s in (self._bm25_spin, self._vec_spin, self._seq_spin) if s is not changed_spin]
            remaining_budget = max(0.0, 1.0 - new_value)
            others_sum = sum(s.value() for s in others)
            if others_sum <= 0:
                # Both others are at 0 -- split the remaining budget evenly.
                for s in others:
                    s.setValue(round(remaining_budget / len(others), 2))
            else:
                for s in others:
                    proportion = s.value() / others_sum
                    s.setValue(round(remaining_budget * proportion, 2))
        finally:
            self._updating = False

    @staticmethod
    def _make_threshold_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        return spin

    def _reset_defaults(self):
        self._updating = True
        self._bm25_spin.setValue(trek_similarity.DEFAULT_BM25_WEIGHT)
        self._vec_spin.setValue(trek_similarity.DEFAULT_VEC_WEIGHT)
        self._seq_spin.setValue(trek_similarity.DEFAULT_SEQ_WEIGHT)
        self._updating = False
        self._dup_spin.setValue(trek_similarity.SIMILARITY_DUPLICATE)
        self._near_spin.setValue(trek_similarity.SIMILARITY_NEAR_DUPLICATE)
        self._similar_spin.setValue(trek_similarity.SIMILARITY_SIMILAR)

    def validate(self, parent_for_dialog=None) -> bool:
        """Check threshold ordering, showing a warning dialog if invalid.
        Returns True if valid (safe to proceed), False otherwise."""
        dup, near, similar = self._dup_spin.value(), self._near_spin.value(), self._similar_spin.value()
        if not (dup >= near >= similar):
            QMessageBox.warning(
                parent_for_dialog, "Invalid Thresholds",
                "Thresholds must be in descending order:\n"
                "Duplicate ≥ Near-duplicate ≥ Similar.\n\n"
                f"Got: Duplicate={dup:.2f}, Near-duplicate={near:.2f}, Similar={similar:.2f}"
            )
            return False
        return True

    def get_settings(self) -> dict:
        return {
            "bm25_weight": self._bm25_spin.value(),
            "vec_weight": self._vec_spin.value(),
            "seq_weight": self._seq_spin.value(),
            "sim_duplicate": self._dup_spin.value(),
            "sim_near_duplicate": self._near_spin.value(),
            "sim_similar": self._similar_spin.value(),
        }


class DuplicateCheckSettingsDialog(QDialog):
    """Shown right before running 'Check Duplicates': wraps
    ScoringSettingsWidget with Cancel/Run Check buttons. These are
    deliberately per-run settings (not project-level) -- sensitivity may
    reasonably differ per module/run.
    """

    def __init__(self, parent=None, current: Optional[dict] = None, traceability_rows: Optional[list] = None):
        super().__init__(parent)
        self.setWindowTitle("Check Duplicates -- Scoring Settings")
        self.resize(1100, 720)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        lay = QVBoxLayout(self)

        # Real pair count/avg text length/SYT-group count from the actual
        # traceability rows (when available) so the LLM cost estimate below
        # reflects THIS run's real data instead of a generic guess. Pairs
        # include both SWT coverage and any related-SYT test cases (see
        # FetchLinksWorker) -- both are judged the same way.
        self._num_pairs = 0
        self._num_syt_groups = 0
        self._avg_pair_chars = 1200
        if traceability_rows:
            total_chars, count, groups = 0, 0, 0
            for row in traceability_rows:
                syt_text = trek_similarity.build_comparison_text(row.get("syt_content"))
                counterparts = list(row.get("swt_content", [])) + list(row.get("related_syt_content", []))
                row_has_pair = False
                for counterpart_content in counterparts:
                    counterpart_text = trek_similarity.build_comparison_text(counterpart_content)
                    if syt_text and counterpart_text:
                        total_chars += len(syt_text) + len(counterpart_text)
                        count += 1
                        row_has_pair = True
                if row_has_pair:
                    groups += 1
            self._num_pairs = count
            self._num_syt_groups = groups
            if count:
                self._avg_pair_chars = total_chars // count

        intro = QLabel(
            "Tune how SYT vs SWT similarity is scored for this run: the "
            "balance between lexical overlap (BM25), semantic overlap "
            "(vector embeddings), structural overlap (method-call "
            "sequence), and the classification sensitivity."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-bottom:6px;")
        lay.addWidget(intro)

        # --- Pair type + count (centered row) ---
        pair_box = QHBoxLayout()
        pair_box.addStretch()

        pair_lbl = QLabel("Compare:")
        pair_lbl.setStyleSheet("font-weight:600;")
        pair_box.addWidget(pair_lbl)

        self._chk_syt_swt = QCheckBox("SYT ↔ SWT pairs")
        self._chk_syt_swt.setChecked(bool((current or {}).get("include_syt_swt", True)))
        self._chk_syt_swt.setToolTip(
            "Compare each SYT test case against its linked SWT test cases -- "
            "the main duplicate-detection path (system test vs. software test)."
        )
        pair_box.addWidget(self._chk_syt_swt)

        self._chk_syt_syt = QCheckBox("SYT ↔ SYT pairs (related)")
        self._chk_syt_syt.setChecked(bool((current or {}).get("include_syt_syt", True)))
        self._chk_syt_syt.setToolTip(
            "Compare SYT test cases that share a SYR requirement -- finds "
            "redundant system tests covering the same requirement differently."
        )
        pair_box.addWidget(self._chk_syt_syt)

        self._pair_count_lbl = QLabel("")
        self._pair_count_lbl.setStyleSheet(f"font-size:11px;color:{ACCENT};font-weight:600;margin-left:8px;")
        pair_box.addWidget(self._pair_count_lbl)

        pair_box.addStretch()

        # Update pair count when checkboxes toggle
        self._chk_syt_swt.toggled.connect(lambda: self._update_pair_count())
        self._chk_syt_syt.toggled.connect(lambda: self._update_pair_count())
        lay.addLayout(pair_box)

        # Compute per-type pair counts for the label
        self._num_syt_swt_pairs = 0
        self._num_syt_syt_pairs = 0
        if traceability_rows:
            seen_related = set()
            for row in traceability_rows:
                syt_text = trek_similarity.build_comparison_text(row.get("syt_content"))
                for c in row.get("swt_content", []):
                    if syt_text and trek_similarity.build_comparison_text(c):
                        self._num_syt_swt_pairs += 1
                for c in row.get("related_syt_content", []):
                    rid = c.get("Key", "")
                    pk = frozenset((row.get("syt_id", ""), rid))
                    if pk not in seen_related and syt_text and trek_similarity.build_comparison_text(c):
                        seen_related.add(pk)
                        self._num_syt_syt_pairs += 1
        self._update_pair_count()

        # Left (RAG) / right (LLM + prompt) columns in a resizable splitter
        # instead of one long vertical stack -- the judging-instructions
        # prompt used to end up squeezed at the very bottom of a tall
        # narrow dialog and needed scrolling to even see; giving it its own
        # column with real vertical room fixes that.
        splitter = QSplitter(Qt.Horizontal)

        rag_panel = QWidget()
        rag_lay = QVBoxLayout(rag_panel)
        rag_lay.setContentsMargins(0, 0, 0, 0)

        self._rag_checkbox = QCheckBox("📊 Run hybrid BM25/vector/sequence scoring (RAG)")
        self._rag_checkbox.setChecked(bool((current or {}).get("use_rag_score", True)))
        self._rag_checkbox.setToolTip(
            "The fast/cheap lexical+semantic+structural ensemble score below. "
            "Uncheck to skip it entirely (no embedding/BM25 calls at all) and "
            "rely only on the LLM verification stage."
        )
        rag_lay.addWidget(self._rag_checkbox)

        self._rag_force_refresh_chk = QCheckBox("🔄 Force re-embed (ignore cached RAG embeddings)")
        self._rag_force_refresh_chk.setChecked(bool((current or {}).get("rag_force_refresh", False)))
        self._rag_force_refresh_chk.setToolTip(
            "By default, identical text already embedded in a previous run is "
            "reused for free. Check this to re-embed everything from scratch "
            "for this run (e.g. after changing the embedding model, or to "
            "verify the cache isn't stale). The cache is still updated "
            "afterward with the fresh result."
        )
        rag_lay.addWidget(self._rag_force_refresh_chk)

        self._settings_widget = ScoringSettingsWidget(self, current=current)
        rag_lay.addWidget(self._settings_widget)
        self._rag_checkbox.toggled.connect(self._settings_widget.setEnabled)
        self._rag_checkbox.toggled.connect(self._rag_force_refresh_chk.setEnabled)
        self._settings_widget.setEnabled(self._rag_checkbox.isChecked())
        self._rag_force_refresh_chk.setEnabled(self._rag_checkbox.isChecked())
        rag_lay.addStretch()
        splitter.addWidget(rag_panel)

        llm_panel = QWidget()
        lay2 = QVBoxLayout(llm_panel)
        lay2.setContentsMargins(0, 0, 0, 0)

        llm_lbl = QLabel("LLM Verification (optional, extra cost)")
        llm_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:12px;")
        lay2.addWidget(llm_lbl)

        self._llm_checkbox = QCheckBox(
            "🧠 Verify every pair with an LLM -- reads the actual test logic "
            "instead of just lexical/semantic overlap"
        )
        self._llm_checkbox.setChecked(bool((current or {}).get("use_llm_judge", False)))
        self._llm_checkbox.setToolTip(
            "Sends every compared pair's SYT/SWT text to a chat model, asking "
            "whether they test the SAME scenario -- catches redundant coverage "
            "the hybrid BM25/vector/sequence score misses when phrasing differs "
            "a lot, and vice versa. Adds one chat-completion call per pair "
            "(real token/cost tracked and shown in the results summary). This "
            "is a SEPARATE result from the BM25/vector/sequence classification "
            "above -- it never overrides it, both are shown side by side."
        )
        lay2.addWidget(self._llm_checkbox)

        model_form = QFormLayout()
        self._llm_model_combo = QComboBox()
        self._llm_model_combo.addItems(trek_similarity.AVAILABLE_CHAT_MODELS)
        current_model = (current or {}).get("llm_model", trek_similarity.CHAT_MODEL)
        if current_model in trek_similarity.AVAILABLE_CHAT_MODELS:
            self._llm_model_combo.setCurrentText(current_model)
        model_form.addRow("Chat model:", self._llm_model_combo)

        self._llm_batch_spin = QSpinBox()
        self._llm_batch_spin.setRange(1, 50)
        self._llm_batch_spin.setValue((current or {}).get("llm_batch_size", 1))
        self._llm_batch_spin.setToolTip(
            "Number of SYT test cases judged per single LLM request -- each "
            "SYT is sent together with ALL of its pairs (linked SWT test "
            "cases AND any related SYT test cases sharing a SYR), never "
            "split across two requests. Higher = fewer, cheaper requests "
            "(less repeated prompt overhead) but a larger prompt per call "
            "and a bad/partial response affects more SYTs at once. 1 means "
            "one SYT (with all its pairs) per request."
        )
        model_form.addRow("SYT groups per LLM request (batch size):", self._llm_batch_spin)

        self._llm_workers_spin = QSpinBox()
        self._llm_workers_spin.setRange(1, 50)
        self._llm_workers_spin.setValue((current or {}).get("llm_max_workers", 20))
        self._llm_workers_spin.setToolTip(
            "Maximum number of LLM batch requests sent concurrently. Higher "
            "= faster (more requests in flight at once) but may hit the "
            "gateway's rate limit. 20 is a good default for most gateways; "
            "lower it if you see 429 (rate limit) errors."
        )
        model_form.addRow("Concurrent LLM requests:", self._llm_workers_spin)
        lay2.addLayout(model_form)

        self._llm_force_refresh_chk = QCheckBox("🔄 Force re-judge (ignore cached LLM verdicts)")
        self._llm_force_refresh_chk.setChecked(bool((current or {}).get("llm_force_refresh", False)))
        self._llm_force_refresh_chk.setToolTip(
            "By default, a pair already judged before under the same model + "
            "judging instructions is served from cache for free. Check this "
            "to re-judge everything from scratch for this run (e.g. after "
            "tweaking the instructions and wanting to compare). The cache is "
            "still updated afterward with the fresh verdicts."
        )
        lay2.addWidget(self._llm_force_refresh_chk)

        self._cost_estimate_lbl = QLabel("")
        self._cost_estimate_lbl.setWordWrap(True)
        self._cost_estimate_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-bottom:4px;")
        lay2.addWidget(self._cost_estimate_lbl)

        instructions_lbl = QLabel("Judging instructions (editable -- JSON output format is always enforced separately):")
        instructions_lbl.setWordWrap(True)
        instructions_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;margin-top:4px;")
        lay2.addWidget(instructions_lbl)

        disclaimer_lbl = QLabel(
            "⚠️ These are GENERIC default instructions shared across every "
            "module. For meaningfully better verdicts, customize them per "
            "module -- e.g. mention this module's specific fault types, "
            "naming/ID conventions, or what counts as \"redundant\" in this "
            "domain -- and save your own wording before running."
        )
        disclaimer_lbl.setWordWrap(True)
        disclaimer_lbl.setStyleSheet(
            f"color:{AMBER_COLOR};font-size:11px;font-weight:bold;margin:2px 0 4px 0;"
        )
        lay2.addWidget(disclaimer_lbl)

        self._llm_instructions_edit = QTextEdit()
        self._llm_instructions_edit.setPlainText(
            (current or {}).get("llm_instructions") or trek_similarity.LLM_JUDGE_DEFAULT_INSTRUCTIONS
        )
        lay2.addWidget(self._llm_instructions_edit, 1)
        splitter.addWidget(llm_panel)
        splitter.setSizes([380, 620])
        lay.addWidget(splitter, 1)

        self._llm_checkbox.toggled.connect(self._update_cost_estimate)
        self._llm_checkbox.toggled.connect(self._llm_force_refresh_chk.setEnabled)
        self._llm_force_refresh_chk.setEnabled(self._llm_checkbox.isChecked())
        self._llm_model_combo.currentTextChanged.connect(self._update_cost_estimate)
        self._llm_batch_spin.valueChanged.connect(self._update_cost_estimate)
        self._llm_instructions_edit.textChanged.connect(self._update_cost_estimate)
        self._update_cost_estimate()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_run = QPushButton("Run Check")
        btn_run.setObjectName("btn_success")
        btn_run.clicked.connect(self._on_save)
        btn_row.addWidget(btn_run)
        lay.addLayout(btn_row)

    def _update_cost_estimate(self):
        """Refresh the pre-run LLM cost estimate label -- recomputed live as
        the user changes the model/batch-size/instructions/checkbox.

        Three tiers, most to least accurate:
          1. REAL learned cost-per-pair from this EXACT model's own run
             history (see AppStatsDialog's "llm_stage" operation_stats).
          2. REAL learned tokens-per-pair from ANY model's run history,
             re-priced at THIS model's own output rate -- output/reasoning
             tokens dominate both token count and cost for the chat models
             used here, and can be many times a generic guess (e.g. hidden
             reasoning tokens some models bill for), but the actual TOKEN
             VOLUME needed to judge one pair is roughly comparable across
             models doing the same judging task, so this is a much better
             starting point than a char-count guess for a brand-new model.
          3. trek_similarity.estimate_llm_judge_cost()'s char-count
             heuristic (using the real pair count/average text length from
             THIS run's actual traceability rows, see __init__) -- last
             resort, only when NO llm_stage history exists at all yet.
        """
        if not self._llm_checkbox.isChecked() or self._num_pairs <= 0:
            self._cost_estimate_lbl.setText("")
            return
        model = self._llm_model_combo.currentText()
        batch_size = self._llm_batch_spin.value()
        num_batches = -(-self._num_syt_groups // max(1, batch_size))

        all_llm_runs = CACHE.get_operation_stats("llm_stage")
        exact_model_runs = [
            row for row in all_llm_runs
            if (row["details"] or {}).get("model") == model and row["item_count"] > 0
        ]
        exact_new_pairs = sum(row["item_count"] for row in exact_model_runs)

        any_model_runs = [row for row in all_llm_runs if row["item_count"] > 0]
        any_new_pairs = sum(row["item_count"] for row in any_model_runs)
        any_tokens = sum((row["details"] or {}).get("tokens", 0) for row in any_model_runs)

        pricing = trek_similarity.MODEL_PRICING.get(model)
        if exact_model_runs and exact_new_pairs > 0:
            cost_per_pair = sum(row["cost_usd"] for row in exact_model_runs) / exact_new_pairs
            estimate = cost_per_pair * self._num_pairs
            basis = f"based on {len(exact_model_runs)} previous run(s) with {model}"
        elif any_model_runs and any_new_pairs > 0 and any_tokens > 0 and pricing is not None:
            tokens_per_pair = any_tokens / any_new_pairs
            estimate = tokens_per_pair * self._num_pairs * pricing["output_cost"]
            basis = (
                f"based on token volume from {len(any_model_runs)} previous run(s) with other "
                f"models, re-priced at {model}'s rate -- no history for {model} itself yet"
            )
        else:
            estimate = trek_similarity.estimate_llm_judge_cost(
                self._num_pairs, model, batch_size,
                instructions=self._llm_instructions_edit.toPlainText(),
                avg_pair_chars=self._avg_pair_chars,
                num_groups=self._num_syt_groups,
            )
            basis = "rough heuristic -- no prior LLM-judge run history at all yet"

        if estimate is None:
            self._cost_estimate_lbl.setText(
                f"💵 Estimated cost: unavailable (no pricing for '{model}') -- "
                f"{self._num_pairs} pair(s) across {self._num_syt_groups} SYT(s), "
                f"~{num_batches} batch request(s)."
            )
        else:
            self._cost_estimate_lbl.setText(
                f"💵 Estimated LLM cost: ~${estimate:.4f} for {self._num_pairs} pair(s) "
                f"across {self._num_syt_groups} SYT(s) (~{num_batches} batch request(s)) "
                f"-- {basis}. Approximate -- exact cost is shown after the run."
            )

    def _update_pair_count(self):
        """Update the pair-count label based on which pair types are checked."""
        n = 0
        parts = []
        if self._chk_syt_swt.isChecked():
            n += self._num_syt_swt_pairs
            parts.append(f"{self._num_syt_swt_pairs} SYT↔SWT")
        if self._chk_syt_syt.isChecked():
            n += self._num_syt_syt_pairs
            parts.append(f"{self._num_syt_syt_pairs} SYT↔SYT")
        self._num_pairs = n
        self._pair_count_lbl.setText(f"({' + '.join(parts)} = {n} pairs)" if parts else "(no pairs selected)")
        # Guard: _llm_checkbox may not exist yet during __init__ (the pair
        # type section is built before the LLM panel).
        if hasattr(self, "_llm_checkbox"):
            self._update_cost_estimate()

    def _on_save(self):
        if not self._rag_checkbox.isChecked() and not self._llm_checkbox.isChecked():
            QMessageBox.warning(
                self, "Nothing to Run",
                "Enable at least one of 'Run hybrid BM25/vector/sequence "
                "scoring (RAG)' or 'Verify every pair with an LLM'."
            )
            return
        if not self._chk_syt_swt.isChecked() and not self._chk_syt_syt.isChecked():
            QMessageBox.warning(
                self, "No Pair Types Selected",
                "Enable at least one pair type to compare "
                "(SYT ↔ SWT and/or SYT ↔ SYT)."
            )
            return
        if self._rag_checkbox.isChecked() and not self._settings_widget.validate(self):
            return
        self._result = dict(
            self._settings_widget.get_settings(),
            include_syt_swt=self._chk_syt_swt.isChecked(),
            include_syt_syt=self._chk_syt_syt.isChecked(),
            use_rag_score=self._rag_checkbox.isChecked(),
            rag_force_refresh=self._rag_force_refresh_chk.isChecked(),
            use_llm_judge=self._llm_checkbox.isChecked(),
            llm_model=self._llm_model_combo.currentText(),
            llm_batch_size=self._llm_batch_spin.value(),
            llm_max_workers=self._llm_workers_spin.value(),
            llm_instructions=self._llm_instructions_edit.toPlainText().strip(),
            llm_force_refresh=self._llm_force_refresh_chk.isChecked(),
        )
        self.accept()

    def result_data(self) -> Optional[dict]:
        return getattr(self, "_result", None)


class SideBySideResultsWidget(QWidget):
    """Reusable results display for trek_similarity.DuplicateCheckResult:
    a sortable list of all compared pairs (SYT id, SWT id, final ensemble
    score, classification) on the left; selecting a row shows the full
    SYT and SWT comparison text side by side on the right, along with the
    score breakdown (BM25 / Vector / Sequence / Ensemble) that the
    classification was based on.

    Shared by DuplicateResultsDialog (real 'Check Duplicates' runs against
    TREK data) and TestAlgorithmDialog (manual synthetic-data sandbox) so
    both present results in the EXACT same way -- one implementation of
    "how results are displayed" to keep in sync, per the user's request
    that the sandbox use the identical results view as the real feature.
    """

    def __init__(self, parent=None, on_review_changed=None):
        super().__init__(parent)
        self._result: Optional["trek_similarity.DuplicateCheckResult"] = None
        self._on_review_changed_callback = on_review_changed
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._summary_lbl = QLabel("Run a comparison to see results.")
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        lay.addWidget(self._summary_lbl)

        splitter = QSplitter(Qt.Horizontal)

        # Left: sortable pair list
        list_frame = QGroupBox("Compared Pairs (click a row for details)")
        list_lay = QVBoxLayout(list_frame)

        # Per-column filter row (not just sort) -- e.g. typing "syt" in the
        # counterpart filter narrows to related-SYT pairs only (their IDs
        # look like "SYT_3"), since real SWT ids never contain "syt".
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        self._filter_syt = QLineEdit()
        self._filter_syt.setPlaceholderText("Filter SYT...")
        self._filter_syt.setFixedWidth(110)
        self._filter_syt.textChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_syt)

        self._filter_swt = QLineEdit()
        self._filter_swt.setPlaceholderText("Filter SWT/related SYT...")
        self._filter_swt.setFixedWidth(170)
        self._filter_swt.textChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_swt)

        self._filter_score_min = QLineEdit()
        self._filter_score_min.setPlaceholderText("min score")
        self._filter_score_min.setFixedWidth(70)
        self._filter_score_min.textChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_score_min)

        self._filter_classification = QComboBox()
        self._filter_classification.setFixedWidth(150)
        self._filter_classification.addItem("All classifications", None)
        for key, (label, _color) in trek_similarity.CLASSIFICATION_LABELS.items():
            self._filter_classification.addItem(label, key)
        self._filter_classification.currentIndexChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_classification)

        self._filter_llm = QComboBox()
        self._filter_llm.addItem("All LLM verdicts", None)
        for key, (label, _color) in trek_similarity.LLM_VERDICT_LABELS.items():
            self._filter_llm.addItem(label, key)
        self._filter_llm.currentIndexChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_llm)

        self._filter_review = QComboBox()
        self._filter_review.addItem("All reviews", None)
        self._filter_review.addItem("⬜ Unreviewed", "unreviewed")
        self._filter_review.addItem("🔴 Same scenario", "same_scenario")
        self._filter_review.addItem("🟡 Partial overlap", "partial_overlap")
        self._filter_review.addItem("🟢 Different scenario", "different_scenario")
        self._filter_review.currentIndexChanged.connect(self._apply_pair_filters)
        filter_row.addWidget(self._filter_review, 1)
        list_lay.addLayout(filter_row)

        self._pairs_table = QTableWidget(0, 6)
        self._pairs_table.setHorizontalHeaderLabels(["SYT", "SWT / Related SYT", "Score", "Classification", "LLM Verdict", "Review"])
        self._pairs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self._pairs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self._pairs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self._pairs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self._pairs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Interactive)
        self._pairs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._pairs_table.setColumnWidth(0, 110)
        self._pairs_table.setColumnWidth(1, 170)
        self._pairs_table.setColumnWidth(2, 70)
        self._pairs_table.setColumnWidth(3, 150)
        self._pairs_table.setColumnWidth(4, 170)
        self._pairs_table.setAlternatingRowColors(True)
        self._pairs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pairs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._pairs_table.verticalHeader().setVisible(False)
        self._pairs_table.setSortingEnabled(True)
        self._pairs_table.itemSelectionChanged.connect(self._on_row_selected)
        list_lay.addWidget(self._pairs_table)
        splitter.addWidget(list_frame)

        # Right: side-by-side content + score breakdown
        detail_frame = QGroupBox("Side-by-Side Comparison")
        detail_lay = QVBoxLayout(detail_frame)

        self._score_breakdown_lbl = QLabel("Select a pair to see details.")
        self._score_breakdown_lbl.setWordWrap(True)
        self._score_breakdown_lbl.setStyleSheet(
            f"background:{CODE_BG};border-radius:4px;padding:8px;color:{TEXT};font-size:12px;"
        )
        detail_lay.addWidget(self._score_breakdown_lbl)

        # Always-visible legend (not just a hover tooltip) explaining what
        # every RAG classification / LLM verdict status actually means --
        # collapsible so it doesn't eat vertical space once learned.
        legend_box = QGroupBox("ℹ️  What do these statuses mean?")
        legend_box.setCheckable(True)
        legend_box.setChecked(False)   # collapsed by default
        legend_box_lay = QVBoxLayout(legend_box)
        legend_lbl = QLabel(_status_legend_html())
        legend_lbl.setWordWrap(True)
        legend_lbl.setTextFormat(Qt.RichText)
        legend_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        legend_lbl.setVisible(False)   # start hidden (matches collapsed state)
        legend_box_lay.addWidget(legend_lbl)
        legend_box.toggled.connect(legend_lbl.setVisible)
        detail_lay.addWidget(legend_box)

        side_by_side = QHBoxLayout()
        syt_box = QVBoxLayout()
        syt_lbl = QLabel("SYT Content  ℹ️")
        syt_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        syt_lbl.setToolTip(_status_legend_html())
        syt_box.addWidget(syt_lbl)
        self._syt_text_view = QTextEdit()
        self._syt_text_view.setReadOnly(True)
        syt_box.addWidget(self._syt_text_view)

        swt_box = QVBoxLayout()
        self._swt_lbl = QLabel("SWT Content  ℹ️")
        self._swt_lbl.setStyleSheet(f"color:{SUCCESS_TEXT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        self._swt_lbl.setToolTip(_status_legend_html())
        swt_box.addWidget(self._swt_lbl)
        self._swt_text_view = QTextEdit()
        self._swt_text_view.setReadOnly(True)
        swt_box.addWidget(self._swt_text_view)

        side_by_side.addLayout(syt_box)
        side_by_side.addLayout(swt_box)
        detail_lay.addLayout(side_by_side, 1)

        # ── Manual review panel (collapsible) ──
        review_frame = QGroupBox("📋  Manual Review")
        review_frame.setCheckable(True)
        review_frame.setChecked(True)
        review_main_lay = QVBoxLayout(review_frame)
        review_main_lay.setContentsMargins(8, 6, 8, 6)
        review_main_lay.setSpacing(4)

        # Status label (full width)
        self._review_status_lbl = QLabel("No pair selected")
        self._review_status_lbl.setStyleSheet("font-size:11px;font-weight:600;")
        self._review_status_lbl.setWordWrap(True)
        review_main_lay.addWidget(self._review_status_lbl)

        # Verdict buttons row
        review_btn_row = QHBoxLayout()
        review_btn_row.setSpacing(4)

        self._btn_review_same = QPushButton("🔴 Same")
        self._btn_review_same.setObjectName("btn_danger")
        self._btn_review_same.setToolTip("Both tests exercise the same scenario — redundant coverage.")
        self._btn_review_same.setEnabled(False)
        self._btn_review_same.clicked.connect(lambda: self._set_review("same_scenario"))
        review_btn_row.addWidget(self._btn_review_same)

        self._btn_review_partial = QPushButton("🟡 Partial")
        self._btn_review_partial.setToolTip("Tests share the same area but differ meaningfully in scope or method.")
        self._btn_review_partial.setEnabled(False)
        self._btn_review_partial.clicked.connect(lambda: self._set_review("partial_overlap"))
        review_btn_row.addWidget(self._btn_review_partial)

        self._btn_review_different = QPushButton("🟢 Different")
        self._btn_review_different.setObjectName("btn_success")
        self._btn_review_different.setToolTip("Tests exercise genuinely different scenarios — distinct coverage.")
        self._btn_review_different.setEnabled(False)
        self._btn_review_different.clicked.connect(lambda: self._set_review("different_scenario"))
        review_btn_row.addWidget(self._btn_review_different)

        self._btn_clear_review = QPushButton("X Clear")
        self._btn_clear_review.setObjectName("btn_secondary")
        self._btn_clear_review.setToolTip("Clear review verdict and comment.")
        self._btn_clear_review.setEnabled(False)
        self._btn_clear_review.clicked.connect(self._clear_review)
        review_btn_row.addWidget(self._btn_clear_review)

        review_btn_row.addSpacing(8)

        review_main_lay.addLayout(review_btn_row)

        # Comment field — separate row, larger
        self._review_comment_edit = QLineEdit()
        self._review_comment_edit.setPlaceholderText("Comment (optional — explain your verdict, note what the AI got wrong, remarks...)  ")
        self._review_comment_edit.setEnabled(False)
        self._review_comment_edit.setMinimumHeight(30)
        review_main_lay.addWidget(self._review_comment_edit)
        detail_lay.addWidget(review_frame)

        splitter.addWidget(detail_frame)
        splitter.setSizes([420, 680])
        lay.addWidget(splitter, 1)

    def refresh_theme(self):
        """Re-apply inline styles and re-render content with new theme colours."""
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        self._score_breakdown_lbl.setStyleSheet(
            f"color:{TEXT};font-size:12px;padding:4px 0;"
        )
        self._swt_lbl.setStyleSheet(f"color:{SUCCESS_TEXT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        if self._result is not None:
            self.set_result(self._result)

    def set_result(self, result: "trek_similarity.DuplicateCheckResult"):
        self._result = result

        # A merged/superset cached result (see
        # trek_similarity.merge_duplicate_check_results()) can span
        # several different runs -- total_cost_usd/total_llm_cost_usd only
        # ever reflect the LATEST run's fresh spend, which is misleading
        # to show as "the" cost of a list that's actually accumulated over
        # multiple runs. Only show cost here for a genuine single-run
        # result (every pair sharing the same checked_at, or none at all);
        # use the Metrics dialog's historical totals for a merged list.
        is_merged = len({p.checked_at for p in result.pairs if p.checked_at}) > 1

        counts = result.summary_counts()
        if counts["not_scored"] == len(result.pairs) and result.pairs:
            summary_text = f"⚪ RAG scoring disabled -- {counts['not_scored']} pair(s) not scored"
        else:
            summary_text = (
                f"🔴 {counts['duplicate']} duplicate  |  "
                f"🟠 {counts['near_duplicate']} near-duplicate  |  "
                f"🟡 {counts['similar']} similar  |  "
                f"🟢 {counts['distinct']} distinct"
            )
        if result.total_tokens_embedded:
            summary_text += (
                f"  |  🔢 ~{result.total_tokens_embedded:,} tokens embedded "
                f"({result.total_texts_embedded} new text(s))"
            )
            if result.total_cost_usd and not is_merged:
                summary_text += f"  |  💵 ${result.total_cost_usd:.4f}"
        llm_judged = sum(1 for p in result.pairs if p.llm_verdict)
        if llm_judged:
            llm_counts = result.llm_verdict_counts()
            summary_text += (
                f"  |  🧠🔴 {llm_counts['same_scenario']} same scenario  |  "
                f"🧠🟡 {llm_counts['partial_overlap']} partial overlap  |  "
                f"🧠🟢 {llm_counts['different_scenario']} different scenario"
            )
            if llm_counts["error"]:
                summary_text += f"  |  🧠⚠️ {llm_counts['error']} judge error(s)"
            cache_part = f", {result.llm_cached_pairs} from cache" if result.llm_cached_pairs else ""
            model_part = f", model: {result.llm_model}" if result.llm_model else ""
            summary_text += f"  |  ({llm_judged} LLM-verified{model_part}, {result.llm_calls} batch request(s){cache_part})"
            if result.total_llm_cost_usd and not is_merged:
                summary_text += f" (${result.total_llm_cost_usd:.4f})"
        if result.skipped:
            summary_text += f"  |  ⚠️ {len(result.skipped)} skipped (no content)"
        if is_merged:
            summary_text += "  |  💵 see Metrics for historical cost (this list spans multiple runs)"
        self._summary_lbl.setText(summary_text)

        # Sort worst-first (highest ensemble score = most likely redundant)
        # so the pairs needing attention appear at the top by default.
        sorted_pairs = sorted(result.pairs, key=lambda p: p.score, reverse=True)

        # Merged/superset cached results (see
        # trek_similarity.merge_duplicate_check_results()) can mix pairs
        # from several different runs -- flag whichever pairs share the
        # MOST RECENT checked_at so it's clear which ones are fresh vs.
        # carried over from an earlier run. Only shown when the list
        # actually mixes runs (a single live run has one timestamp for
        # every pair, so the flag would be meaningless noise there).
        checked_ats = {p.checked_at for p in result.pairs if p.checked_at}
        latest_checked_at = max(checked_ats) if checked_ats else None
        show_latest_flag = len(checked_ats) > 1

        # --- High-performance table population --------------------------------
        # For large results (9000+ pairs) the old row-by-row insertRow() +
        # setItem() loop was catastrophically slow (15-20 minutes) because
        # each call triggers Qt's internal model/view notification machinery.
        # Fix: pre-allocate all rows, block signals/updates during fill,
        # and unblock at the end -- turns 15 minutes into <2 seconds.
        table = self._pairs_table
        table.setSortingEnabled(False)
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        table.setRowCount(len(sorted_pairs))
        for row, pair in enumerate(sorted_pairs):
            is_latest = show_latest_flag and pair.checked_at == latest_checked_at
            syt_item = QTableWidgetItem(("🆕 " if is_latest else "") + pair.syt_id)
            syt_item.setData(Qt.UserRole, pair)
            table.setItem(row, 0, syt_item)

            counterpart_item = QTableWidgetItem(
                pair.swt_id + ("  (related SYT)" if pair.counterpart_type == "related_syt" else "")
            )
            if pair.counterpart_type == "related_syt":
                counterpart_item.setForeground(QColor(RELATED_COLOR))
            table.setItem(row, 1, counterpart_item)

            score_item = QTableWidgetItem()
            score_item.setData(Qt.EditRole, round(pair.score, 4))
            score_item.setText(f"{pair.score:.3f}")
            table.setItem(row, 2, score_item)

            label, color = trek_similarity.CLASSIFICATION_LABELS.get(
                pair.classification, (pair.classification, TEXT_DIM)
            )
            color = _themed_classification_color(pair.classification, color)
            class_item = QTableWidgetItem(label + ("  (exact text match)" if pair.exact_match else ""))
            class_item.setForeground(QColor(color))
            table.setItem(row, 3, class_item)

            llm_label, llm_color = trek_similarity.LLM_VERDICT_LABELS.get(
                pair.llm_verdict, ("—", TEXT_DIM)
            ) if pair.llm_verdict else ("—", TEXT_DIM)
            if pair.llm_verdict:
                llm_color = _themed_classification_color(pair.llm_verdict, llm_color)
            llm_item = QTableWidgetItem(llm_label)
            llm_item.setForeground(QColor(llm_color))
            table.setItem(row, 4, llm_item)

            review_item = QTableWidgetItem(self._review_display(pair))
            review_item.setForeground(QColor(self._review_color(pair.review_status)))
            table.setItem(row, 5, review_item)

        table.blockSignals(False)
        table.setUpdatesEnabled(True)
        table.setSortingEnabled(True)

        self._syt_text_view.clear()
        self._swt_text_view.clear()
        self._score_breakdown_lbl.setText("Select a pair to see details.")
        self._apply_pair_filters()
        if sorted_pairs:
            self._pairs_table.selectRow(0)

    def _apply_pair_filters(self):
        """Hide rows that don't match every active per-column filter (text
        filters are case-insensitive substring matches; classification/LLM
        verdict are exact-match dropdowns). All filters combine with AND.

        For large tables (9000+ rows) we block updates during the loop to
        avoid Qt repainting after every setRowHidden call."""
        syt_filter = self._filter_syt.text().strip().lower()
        swt_filter = self._filter_swt.text().strip().lower()
        min_score = None
        min_score_text = self._filter_score_min.text().strip()
        if min_score_text:
            try:
                min_score = float(min_score_text)
            except ValueError:
                min_score = None
        classification_filter = self._filter_classification.currentData()
        llm_filter = self._filter_llm.currentData()
        review_filter = self._filter_review.currentData()

        table = self._pairs_table
        table.setUpdatesEnabled(False)
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            pair = item.data(Qt.UserRole) if item else None
            if pair is None:
                continue
            visible = True
            if syt_filter and syt_filter not in pair.syt_id.lower():
                visible = False
            if visible and swt_filter and swt_filter not in pair.swt_id.lower():
                visible = False
            if visible and min_score is not None and pair.score < min_score:
                visible = False
            if visible and classification_filter and pair.classification != classification_filter:
                visible = False
            if visible and llm_filter and pair.llm_verdict != llm_filter:
                visible = False
            if visible and review_filter:
                pair_status = pair.review_status or "unreviewed"
                if pair_status != review_filter:
                    visible = False
            table.setRowHidden(row, not visible)
        table.setUpdatesEnabled(True)

    # -- Review helpers --
    # Review statuses mirror LLM verdicts so human vs AI can be compared
    # directly in the Metrics dialog.
    _REVIEW_LABELS = {
        "same_scenario":      ("🔴 Same scenario",      DANGER_TEXT),
        "partial_overlap":    ("🟡 Partial overlap",     AMBER_COLOR),
        "different_scenario": ("🟢 Different scenario",  SUCCESS_TEXT),
    }

    def _review_display(self, pair) -> str:
        """Human-readable review status for the table cell."""
        entry = self._REVIEW_LABELS.get(pair.review_status)
        if entry:
            label = entry[0]
            if pair.review_comment:
                comment = pair.review_comment[:40] + ("..." if len(pair.review_comment) > 40 else "")
                return f"{label}: {comment}"
            return label
        return ""

    def _review_color(self, status: str) -> str:
        entry = self._REVIEW_LABELS.get(status)
        return entry[1] if entry else TEXT_DIM

    def _clear_review(self):
        """Clear the review on the current pair."""
        self._review_comment_edit.clear()
        self._set_review("")

    def _set_review(self, status: str):
        """Set the review status on the currently selected pair, update the
        table cell, and persist to cache via the parent dialog's save hook."""
        items = self._pairs_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        pair = self._pairs_table.item(row, 0).data(Qt.UserRole)
        if pair is None:
            return

        import os as _os
        from datetime import datetime as _dt
        pair.review_status = status
        pair.review_comment = self._review_comment_edit.text().strip() if status else ""
        pair.reviewer = _os.environ.get("USERNAME", _os.environ.get("USER", ""))
        pair.reviewed_at = _dt.now().isoformat(timespec="seconds") if status else ""

        # Update the table cell
        review_item = self._pairs_table.item(row, 5)
        if review_item:
            review_item.setText(self._review_display(pair))
            review_item.setForeground(QColor(self._review_color(status)))

        # Clear the comment field after saving (ready for next pair)
        self._review_comment_edit.clear()

        # Update the review status label
        self._update_review_panel(pair)

        # Notify parent to persist (CachedDuplicateResultsDialog hooks this)
        if self._on_review_changed_callback:
            self._on_review_changed_callback()

    def _update_review_panel(self, pair):
        """Refresh the review panel controls for the given pair."""
        has_review = bool(pair.review_status)
        self._btn_review_same.setEnabled(True)
        self._btn_review_partial.setEnabled(True)
        self._btn_review_different.setEnabled(True)
        self._btn_clear_review.setEnabled(has_review)
        self._review_comment_edit.setEnabled(True)
        self._review_comment_edit.setText(pair.review_comment or "")

        entry = self._REVIEW_LABELS.get(pair.review_status)
        if entry:
            label_text, color = entry
            self._review_status_lbl.setText(
                f"{label_text}  —  reviewed by {pair.reviewer} ({pair.reviewed_at})"
            )
            self._review_status_lbl.setStyleSheet(f"font-size:12px;font-weight:600;color:{color};")
        else:
            self._review_status_lbl.setText("⬜ Not yet reviewed")
            self._review_status_lbl.setStyleSheet("font-size:12px;font-weight:600;")

    def _on_row_selected(self):
        items = self._pairs_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        pair = self._pairs_table.item(row, 0).data(Qt.UserRole)
        if pair is None:
            return

        # Structured content (labeled Pre-Condition/Procedure/Post-Condition
        # sections) when available; falls back to the flat comparison text
        # used for scoring (older cached runs predate syt_tc/swt_tc).
        if pair.syt_tc:
            self._syt_text_view.setHtml(_tc_to_html(pair.syt_tc, level_color=ACCENT))
        else:
            self._syt_text_view.setPlainText(pair.syt_text or "(no content)")
        if pair.swt_tc:
            self._swt_text_view.setHtml(_tc_to_html(pair.swt_tc, level_color=SUCCESS_TEXT))
        else:
            self._swt_text_view.setPlainText(pair.swt_text or "(no content)")
        self._swt_lbl.setText(
            "Related SYT Content" if pair.counterpart_type == "related_syt" else "SWT Content"
        )

        label, color = trek_similarity.CLASSIFICATION_LABELS.get(
            pair.classification, (pair.classification, TEXT_DIM)
        )
        color = _themed_classification_color(pair.classification, color)
        if pair.exact_match:
            breakdown = (
                f'<b>{pair.syt_id}</b> vs <b>{pair.swt_id}</b><br>'
                f'<span style="color:{color};font-weight:bold">{label}</span> '
                f'&mdash; identical text after normalization (no scoring needed).'
            )
        elif pair.classification == "not_scored":
            breakdown = (
                f'<b>{pair.syt_id}</b> vs <b>{pair.swt_id}</b><br>'
                f'<span style="color:{color};font-weight:bold">{label}</span> '
                f'&mdash; RAG scoring was disabled for this run.'
            )
        else:
            seq_part = (
                f'Sequence: <b>{pair.sequence_score:.3f}</b> &nbsp;|&nbsp; '
                if pair.has_sequence else
                f'Sequence: <span style="color:{TEXT_DIM}">N/A (no scripted procedure)</span> &nbsp;|&nbsp; '
            )
            rag_explanation = trek_similarity.CLASSIFICATION_EXPLANATIONS.get(pair.classification, "")
            breakdown = (
                f'<b>{pair.syt_id}</b> vs <b>{pair.swt_id}</b><br>'
                f'<span style="color:{color};font-weight:bold">{label}</span>'
                f'{f" &mdash; {rag_explanation}" if rag_explanation else ""}<br><br>'
                f'BM25 (lexical) score: <b>{pair.bm25_score:.3f}</b> &nbsp;|&nbsp; '
                f'Vector (semantic) score: <b>{pair.vector_score:.3f}</b> &nbsp;|&nbsp; '
                f'{seq_part}'
                f'<span style="color:{ACCENT}">Ensemble score: <b>{pair.score:.3f}</b></span>'
            )
        if pair.llm_verdict:
            llm_label, llm_color = trek_similarity.LLM_VERDICT_LABELS.get(
                pair.llm_verdict, (pair.llm_verdict, TEXT_DIM)
            )
            llm_explanation = trek_similarity.LLM_VERDICT_EXPLANATIONS.get(pair.llm_verdict, "")
            breakdown += (
                f'<br><br><span style="color:{llm_color};font-weight:bold">{llm_label}</span>'
                f'{f" &mdash; {llm_explanation}" if llm_explanation else ""}<br>'
                f'{pair.llm_reasoning}'
            )
            # Only meaningful when BOTH the RAG classification and the LLM
            # verdict are actually available for this pair -- comparing
            # against a "not_scored" (RAG-disabled) result would be comparing
            # against nothing.
            rag_scored = pair.exact_match or pair.classification != "not_scored"
            if rag_scored and pair.llm_verdict != "error":
                rag_flagged = pair.exact_match or pair.classification in ("duplicate", "near_duplicate", "similar")
                if rag_flagged and pair.llm_verdict == "same_scenario":
                    agree_color, agree_text = SUCCESS_TEXT, "✅ RAG and LLM agree -- both flag this as likely redundant coverage."
                elif not rag_flagged and pair.llm_verdict == "different_scenario":
                    agree_color, agree_text = SUCCESS_TEXT, "✅ RAG and LLM agree -- both consider this pair distinct."
                else:
                    agree_color, agree_text = AMBER_COLOR, "⚠️ RAG and LLM disagree -- worth a closer manual look."
                breakdown += f'<br><br><span style="color:{agree_color};font-weight:bold">{agree_text}</span>'
        self._score_breakdown_lbl.setText(breakdown)

        # Update the review panel for the selected pair
        self._update_review_panel(pair)


def _open_independent_window(owner, dlg: QDialog) -> None:
    """Show ``dlg`` as its own top-level, non-modal window (no parent)
    instead of a blocking modal child -- on Windows this gives it a real,
    separate taskbar entry that's grouped with the app's other open
    windows, so hovering the taskbar icon lets the user pick any open
    window by its title. Keeps a reference on ``owner._open_windows`` (a
    single shared list, created lazily) so the window isn't garbage
    collected while shown, and drops that reference once the window is
    actually closed/destroyed.
    """
    if not hasattr(owner, "_open_windows"):
        owner._open_windows = []
    dlg.setAttribute(Qt.WA_DeleteOnClose)
    # These are parentless top-level windows, so they don't inherit the main
    # window's widget-level stylesheet -- explicitly apply the CURRENT global
    # stylesheet so they always match the active theme (otherwise a dialog
    # opened after a theme switch could render with stale/dark styling).
    app = QApplication.instance()
    if app is not None:
        dlg.setStyleSheet(app.styleSheet())

    def _cleanup():
        if dlg in owner._open_windows:
            owner._open_windows.remove(dlg)

    owner._open_windows.append(dlg)
    dlg.destroyed.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _export_dict_to_json(parent, data: dict, default_filename: str) -> None:
    """Prompt for a save path and write ``data`` as nicely indented,
    human-readable JSON (nested sections rather than a flat dump -- see
    each dialog's _export_to_json() for the actual grouping). Shows a
    QMessageBox on success/failure; does nothing if the save is cancelled.
    """
    default_filename = default_filename if default_filename.lower().endswith(".json") else default_filename + ".json"
    path, _ = QFileDialog.getSaveFileName(parent, "Export to JSON", default_filename, "JSON files (*.json)")
    if not path:
        return
    if not path.lower().endswith(".json"):
        path += ".json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        QMessageBox.critical(parent, "Export Failed", f"Could not write JSON file:\n{exc}")
        return
    QMessageBox.information(parent, "Export Complete", f"Exported to:\n{path}")


class DuplicateMetricsDialog(QDialog):
    """Aggregate metrics view for a 'Check Duplicates' run: per-method
    (RAG/LLM) classification percentages, how often the two methods AGREE
    on a pair, pair coverage by counterpart kind (SWT vs related SYT --
    see FetchLinksWorker's related-SYT detection, which applies even when
    a SYR has no SWR/SWT bridge at all), and the REAL total cost of
    everything shown (this run's fresh spend + the recorded cost of
    anything served from cache -- see historical_embed_cost_usd /
    historical_llm_cost_usd on DuplicateCheckResult).
    """

    def __init__(self, result: "trek_similarity.DuplicateCheckResult", module_name: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duplicate Check -- Metrics")
        self.resize(1280, 900)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        lay = QVBoxLayout(self)

        self._result = result
        metrics = result.compute_metrics()
        self._metrics = metrics

        by_type = metrics.get("breakdown_by_counterpart_type", {})
        self._chart_scope_combo = None
        self._chart_type_combo = None
        self._review_filter_checks = {}
        if _HAS_QTCHARTS:
            scope_row = QHBoxLayout()
            scope_row.addWidget(QLabel("Scope:"))
            self._chart_scope_combo = QComboBox()
            self._chart_scope_combo.addItem(f"Overall -- {metrics['total_pairs']} pair(s)", None)
            swt_total = by_type.get("SWT", {}).get("total_pairs", 0)
            if swt_total:
                self._chart_scope_combo.addItem(f"SYT ↔ SWT -- {swt_total} pair(s)", "SWT")
            related_total = by_type.get("related_syt", {}).get("total_pairs", 0)
            if related_total:
                self._chart_scope_combo.addItem(f"SYT ↔ SYT (related) -- {related_total} pair(s)", "related_syt")
            self._chart_scope_combo.currentIndexChanged.connect(self._refresh_charts)
            scope_row.addWidget(self._chart_scope_combo)

            scope_row.addSpacing(20)
            scope_row.addWidget(QLabel("View:"))
            self._chart_type_combo = QComboBox()
            self._chart_type_combo.addItem("AI Analysis  (RAG + LLM + RAG vs LLM)", "ai_analysis")
            self._chart_type_combo.addItem("Human Review  (Human + LLM + Human vs LLM)", "human_review")
            self._chart_type_combo.addItem("LLM Accuracy Breakdown  (per LLM verdict → human result)", "llm_accuracy_breakdown")
            self._chart_type_combo.currentIndexChanged.connect(self._refresh_charts)
            scope_row.addWidget(self._chart_type_combo)
            scope_row.addStretch()
            lay.addLayout(scope_row)

            # Human review verdict filter — select which reviewed pairs to include
            filter_row = QHBoxLayout()
            filter_row.addWidget(QLabel("Filter by human review:"))
            for key, label in [("same_scenario", "🔴 Same"),
                               ("partial_overlap", "🟡 Partial"),
                               ("different_scenario", "🟢 Different"),
                               ("unreviewed", "⬜ Unreviewed")]:
                chk = QCheckBox(label)
                chk.setChecked(True)
                chk.toggled.connect(self._refresh_charts)
                filter_row.addWidget(chk)
                self._review_filter_checks[key] = chk
            self._filter_count_lbl = QLabel("")
            self._filter_count_lbl.setStyleSheet("font-size:11px;margin-left:8px;")
            filter_row.addWidget(self._filter_count_lbl)
            filter_row.addStretch()
            lay.addLayout(filter_row)

        self._charts_wrapper = QWidget()
        self._charts_wrapper.setMinimumHeight(340)
        self._charts_wrapper.setMaximumHeight(420)

        view = QTextEdit()
        view.setReadOnly(True)
        view.setHtml(self._build_html(metrics, module_name))

        # Charts and text detail in a vertical splitter so user can
        # resize the boundary between them
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(self._charts_wrapper)
        content_splitter.addWidget(view)
        content_splitter.setSizes([380, 400])
        lay.addWidget(content_splitter, 1)

        self._refresh_charts()

        btn_row = QHBoxLayout()
        self._btn_maximize = QPushButton("⛶  Maximize")
        self._btn_maximize.setObjectName("btn_secondary")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        btn_row.addWidget(self._btn_maximize)
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_maximize.setText("⛶  Maximize")
        else:
            self.showMaximized()
            self._btn_maximize.setText("🗗  Restore")

    def _refresh_charts(self):
        """(Re)build the pie-chart row for whichever scope is selected in
        self._chart_scope_combo -- 'Overall', 'SYT ↔ SWT', or 'SYT ↔ SYT
        (related)'. QCharts don't support swapping data cleanly in place,
        so the old chart widgets are detached onto a throwaway widget
        (deleteLater'd, taking its children with it) and fresh ones built
        via _build_charts() every time the scope changes."""
        old_layout = self._charts_wrapper.layout()
        if old_layout is not None:
            trash = QWidget()
            trash.setLayout(old_layout)
            trash.deleteLater()

        scope = self._chart_scope_combo.currentData() if self._chart_scope_combo else None
        category = self._chart_type_combo.currentData() if self._chart_type_combo else "ai_analysis"

        # Filter pairs by selected human review verdicts + scope
        allowed_reviews = set()
        for key, chk in self._review_filter_checks.items():
            if chk.isChecked():
                allowed_reviews.add(key)

        all_pairs = self._result.pairs
        filtered = []
        for p in all_pairs:
            # Scope filter
            if scope == "SWT" and p.counterpart_type != "SWT":
                continue
            if scope == "related_syt" and p.counterpart_type != "related_syt":
                continue
            # Human review filter
            status = p.review_status or "unreviewed"
            if status not in allowed_reviews:
                continue
            filtered.append(p)

        # Update count label
        if hasattr(self, "_filter_count_lbl"):
            self._filter_count_lbl.setText(f"{len(filtered)} of {len(all_pairs)} pairs")

        # Recompute metrics on filtered subset
        from trek_similarity import DuplicateCheckResult
        filtered_result = DuplicateCheckResult(pairs=filtered)
        m = filtered_result.compute_metrics()

        new_lay = QHBoxLayout()
        if m and filtered:
            if category == "ai_analysis":
                chart_types = ["rag", "llm", "agreement"]
            elif category == "llm_accuracy_breakdown":
                chart_types = ["llm_acc_same", "llm_acc_partial", "llm_acc_different"]
            else:
                chart_types = ["review", "llm", "llm_vs_human"]
            for ct in chart_types:
                chart = self._build_single_chart(m, ct)
                if chart:
                    new_lay.addWidget(chart, 1)
        self._charts_wrapper.setLayout(new_lay)

    @staticmethod
    def _make_pie_chart(title: str, slices: List[tuple]) -> Optional[QWidget]:
        """Build a widget containing a clean pie chart (no slice labels,
        no built-in legend) + a custom legend underneath with large
        coloured bullets showing label, count, and percentage."""
        if not _HAS_QTCHARTS:
            return None
        non_zero = [(label, value, color) for label, value, color in slices if value > 0]
        if not non_zero:
            return None

        chart_total = sum(value for _, value, _ in non_zero)

        # --- Pie chart (clean, labels appear on hover) ---
        series = QPieSeries()
        series.setPieSize(0.58)   # small enough to leave room for labels on all sides
        label_data = {}
        for label, value, color in non_zero:
            pct = (value / chart_total * 100.0) if chart_total else 0.0
            piece = series.append(label, value)
            piece.setLabelVisible(False)
            piece.setBrush(QColor(color))
            piece.setLabel(f"{label}: {value} ({pct:.0f}%)")
            piece.setLabelColor(QColor(TEXT))
            piece.setLabelFont(QFont("Inter", 8))
            label_data[id(piece)] = piece
            piece.hovered.connect(
                lambda is_hovered, p=piece: (
                    p.setLabelVisible(is_hovered),
                    p.setExploded(is_hovered),
                    p.setExplodeDistanceFactor(0.05 if is_hovered else 0),
                )
            )

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle(title)
        chart.setTitleBrush(QColor(TEXT))
        chart.setTitleFont(QFont("Inter", 10, QFont.Bold))
        chart.legend().setVisible(False)
        chart.setBackgroundBrush(QColor(PANEL_BG))
        chart.setBackgroundRoundness(8)
        from PySide6.QtCore import QMargins
        chart.setMargins(QMargins(10, 24, 10, 24))   # vertical padding for hover labels
        chart.setAnimationOptions(QChart.SeriesAnimations)

        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setMinimumHeight(260)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Custom legend with colour bullets ---
        legend_widget = QWidget()
        legend_lay = QVBoxLayout(legend_widget)
        legend_lay.setContentsMargins(10, 0, 10, 4)
        legend_lay.setSpacing(3)
        for label, value, color in non_zero:
            pct = (value / chart_total * 100.0) if chart_total else 0.0
            row = QHBoxLayout()
            row.setSpacing(6)
            bullet = QLabel("●")
            bullet.setStyleSheet(f"color:{color};font-size:22px;")
            bullet.setFixedWidth(26)
            row.addWidget(bullet)
            text = QLabel(f"{label}: <b>{value}</b>  ({pct:.1f}%)")
            text.setStyleSheet(f"color:{TEXT};font-size:11px;")
            row.addWidget(text, 1)
            legend_lay.addLayout(row)

        # --- Combined widget ---
        container = QWidget()
        container_lay = QVBoxLayout(container)
        container_lay.setContentsMargins(0, 0, 0, 0)
        container_lay.setSpacing(0)
        container_lay.addWidget(view, 1)
        container_lay.addWidget(legend_widget)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return container

    @classmethod
    def _build_single_chart(cls, m: dict, chart_type: str) -> Optional["QChartView"]:
        """Build a single pie chart based on the selected chart type."""
        if chart_type == "rag":
            rag = m.get("rag_counts", {})
            return cls._make_pie_chart("RAG Classification", [
                ("Duplicate", rag.get("duplicate", 0), DANGER),
                ("Near-duplicate", rag.get("near_duplicate", 0), WARNING),
                ("Similar", rag.get("similar", 0), AMBER_COLOR),
                ("Distinct", rag.get("distinct", 0), SUCCESS),
                ("Not scored", rag.get("not_scored", 0), TEXT_DIM),
            ])
        elif chart_type == "llm":
            llm = m.get("llm_counts", {})
            return cls._make_pie_chart("LLM Verdict", [
                ("Same scenario", llm.get("same_scenario", 0), DANGER),
                ("Partial overlap", llm.get("partial_overlap", 0), WARNING),
                ("Different scenario", llm.get("different_scenario", 0), SUCCESS),
                ("Error", llm.get("error", 0), TEXT_DIM),
            ])
        elif chart_type == "agreement":
            agreement = m.get("agreement_counts", {})
            return cls._make_pie_chart("RAG / LLM Agreement", [
                ("Agree", agreement.get("agree", 0), SUCCESS),
                ("Disagree", agreement.get("disagree", 0), DANGER),
                ("Not applicable", agreement.get("not_applicable", 0), TEXT_DIM),
            ])
        elif chart_type == "review":
            rc = m.get("review_counts", {})
            return cls._make_pie_chart("Human Review Verdicts", [
                ("Same scenario", rc.get("same_scenario", 0), DANGER),
                ("Partial overlap", rc.get("partial_overlap", 0), WARNING),
                ("Different scenario", rc.get("different_scenario", 0), SUCCESS),
                ("Unreviewed", rc.get("unreviewed", 0), TEXT_DIM),
            ])
        elif chart_type == "llm_vs_human":
            acc = m.get("llm_vs_human", {})
            return cls._make_pie_chart("LLM vs Human Agreement", [
                ("LLM correct", acc.get("agree", 0), SUCCESS),
                ("LLM wrong", acc.get("disagree", 0), DANGER),
                ("Not comparable", acc.get("not_comparable", 0), TEXT_DIM),
            ])
        # --- LLM Accuracy Breakdown: one pie per LLM verdict ---
        elif chart_type.startswith("llm_acc_"):
            breakdown = m.get("llm_accuracy_breakdown", {})
            verdict_key = chart_type.replace("llm_acc_", "")
            # Map short key to full key
            key_map = {"same": "same_scenario", "partial": "partial_overlap", "different": "different_scenario"}
            full_key = key_map.get(verdict_key, verdict_key)
            title_map = {"same_scenario": "LLM: Same scenario", "partial_overlap": "LLM: Partial overlap", "different_scenario": "LLM: Different scenario"}
            bucket = breakdown.get(full_key, {})
            total = bucket.get("total", 0)
            if total == 0:
                return None
            human_said = bucket.get("human_said", {})
            return cls._make_pie_chart(
                f"{title_map.get(full_key, full_key)} ({total} pairs)",
                [
                    (f"Human: Same", human_said.get("same_scenario", 0), DANGER),
                    (f"Human: Partial", human_said.get("partial_overlap", 0), WARNING),
                    (f"Human: Different", human_said.get("different_scenario", 0), SUCCESS),
                    (f"Not reviewed", bucket.get("not_reviewed", 0), TEXT_DIM),
                ]
            )
        return None

    @staticmethod
    def _build_html(m: dict, module_name: Optional[str] = None) -> str:
        def _section(title: str) -> str:
            return f'<div style="color:{ACCENT};font-weight:bold;font-size:13px;margin-top:14px;">{title}</div>'

        def _row(label: str, count, pct=None) -> str:
            pct_part = f" &nbsp; <span style=\"color:{TEXT_DIM}\">({pct:.1f}%)</span>" if pct is not None else ""
            return f'<div style="padding:2px 0 2px 8px">{label}: <b>{count}</b>{pct_part}</div>'

        total = m["total_pairs"]
        html = f'<div style="font-size:14px;color:{TEXT}">Total pairs compared: <b>{total}</b></div>'

        html += _section("Pairs by Counterpart Type")
        cp_counts = m["counterpart_type_counts"]
        html += _row("SWT", cp_counts.get("SWT", 0))
        html += _row("Related SYT (shared SYR, no SWR/SWT bridge required)", cp_counts.get("related_syt", 0))

        rag_labels = {
            "duplicate": "🔴 Duplicate", "near_duplicate": "🟠 Near-duplicate",
            "similar": "🟡 Similar", "distinct": "🟢 Distinct", "not_scored": "⚪ Not scored",
        }
        llm_labels = {
            "same_scenario": "🧠🔴 Same scenario", "partial_overlap": "🧠🟡 Partial overlap",
            "different_scenario": "🧠🟢 Different scenario", "error": "🧠⚠️ Judge error",
        }

        html += _section("🔴🟠🟡🟢 RAG Classification")
        for key, label in rag_labels.items():
            html += _row(label, m["rag_counts"].get(key, 0), m["rag_percentages"].get(key, 0.0))

        html += _section(f"🧠 LLM Verdict ({m['llm_judged_total']} pair(s) judged)")
        if m["llm_judged_total"]:
            for key, label in llm_labels.items():
                html += _row(label, m["llm_counts"].get(key, 0), m["llm_percentages"].get(key, 0.0))
        else:
            html += f'<div style="padding:2px 0 2px 8px;color:{TEXT_DIM}">LLM verification was not run.</div>'

        html += _section("🤝 RAG / LLM Agreement")
        agreement = m["agreement_counts"]
        applicable = agreement["agree"] + agreement["disagree"]
        if applicable:
            html += _row("✅ Agree", agreement["agree"], m["agreement_percentage"])
            html += _row("⚠️ Disagree", agreement["disagree"], 100.0 - m["agreement_percentage"])
        html += _row("— Not applicable (one side missing/errored)", agreement["not_applicable"])

        # Same three breakdowns again, but split by counterpart kind -- so
        # "SYT vs SWT" duplicates and "SYT vs SYT" (related, shared-SYR)
        # duplicates are never averaged together into one misleading bucket.
        by_type = m.get("breakdown_by_counterpart_type", {})
        type_titles = {"SWT": "SYT ↔ SWT", "related_syt": "SYT ↔ SYT (related, shared SYR)"}
        for ctype, title in type_titles.items():
            sub = by_type.get(ctype)
            if not sub or not sub["total_pairs"]:
                continue
            html += _section(f"📎 {title} -- {sub['total_pairs']} pair(s)")
            for key, label in rag_labels.items():
                html += _row(label, sub["rag_counts"].get(key, 0), sub["rag_percentages"].get(key, 0.0))
            if sub["llm_judged_total"]:
                html += _row("<i>LLM verdict</i>", f"{sub['llm_judged_total']} judged")
                for key, label in llm_labels.items():
                    html += _row(f"&nbsp;&nbsp;{label}", sub["llm_counts"].get(key, 0), sub["llm_percentages"].get(key, 0.0))
            sub_agreement = sub["agreement_counts"]
            sub_applicable = sub_agreement["agree"] + sub_agreement["disagree"]
            if sub_applicable:
                html += _row("&nbsp;&nbsp;✅ Agree", sub_agreement["agree"], sub["agreement_percentage"])
                html += _row("&nbsp;&nbsp;⚠️ Disagree", sub_agreement["disagree"], 100.0 - sub["agreement_percentage"])

        html += _section("⏱ Timing")
        html += _row("RAG (BM25 + vector + sequence) stage", DuplicateMetricsDialog._fmt_duration(m.get("rag_duration_seconds", 0.0)))
        html += _row("LLM-judge stage", DuplicateMetricsDialog._fmt_duration(m.get("llm_duration_seconds", 0.0)))

        html += _section("🤖 Model & Tokens")
        html += _row("LLM model used", m.get("llm_model") or "(LLM stage not run)")
        html += _row("Embedding model", trek_similarity.EMBEDDING_MODEL)
        html += _row("Tokens embedded this run", f"{m.get('embed_tokens', 0):,}")
        html += _row("LLM tokens used this run", f"{m.get('llm_tokens', 0):,}")

        html += _section("💵 Cost")
        html += _row("Embeddings -- last run", f"${m['embed_cost_usd']:.4f}")
        html += _row("Embeddings -- current list value", f"${m['embed_cost_historical_usd']:.4f}")
        html += _row("Embeddings served from cache this run", m["embed_cached_texts"])
        html += _row("LLM -- last run", f"${m['llm_cost_usd']:.4f}")
        html += _row("LLM -- current list value", f"${m['llm_cost_historical_usd']:.4f}")
        html += _row("LLM pairs served from cache this run", m["llm_cached_pairs"])
        html += (
            f'<div style="color:{TEXT_DIM};font-size:10px;margin-top:8px">'
            "\"Current list value\" is the REAL cost of what's shown right "
            "now: each pair's cost as of its MOST RECENT judgment/embedding "
            "-- if a pair gets re-checked, its OLD cost is replaced, not "
            "added to. It can go up OR down between runs; it is NOT a "
            "running total of every dollar ever spent (see below for that)."
            "</div>"
        )

        if module_name:
            spent = DuplicateMetricsDialog._true_total_spent(module_name)
            if spent is not None:
                embed_spent, llm_spent, run_count = spent
                html += _section("🧾 Total Money Ever Spent (this module, all models/runs)")
                html += _row("Embeddings -- all runs, ever", f"${embed_spent:.4f}")
                html += _row("LLM -- all runs, ever", f"${llm_spent:.4f}")
                html += _row("Recorded RAG/LLM runs counted", run_count)
                html += (
                    f'<div style="color:{TEXT_DIM};font-size:10px;margin-top:4px">'
                    "TRUE cumulative real spend from trek_cache's append-only "
                    "operation_stats log (see App Report) -- includes money "
                    "spent on pairs later re-checked/overwritten, so it only "
                    "ever goes UP, never down, unlike \"current list value\" above."
                    "</div>"
                )
        return html

    @staticmethod
    def _true_total_spent(module_name: str) -> Optional["tuple[float, float, int]"]:
        """Sum REAL cost across every rag_stage/llm_stage run ever recorded
        for this module (see trek_cache's append-only operation_stats),
        regardless of which model or how many times pairs were re-checked
        -- unlike historical_*_cost_usd (current list value), this never
        decreases. Returns None if operation_stats has nothing for this
        module yet."""
        rows = [r for r in CACHE.get_operation_stats() if r["module"] == module_name
                and r["op_type"] in ("rag_stage", "llm_stage")]
        if not rows:
            return None
        embed_spent = sum(r["cost_usd"] for r in rows if r["op_type"] == "rag_stage")
        llm_spent = sum(r["cost_usd"] for r in rows if r["op_type"] == "llm_stage")
        return embed_spent, llm_spent, len(rows)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if not seconds:
            return "not run / instant (fully cached)"
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}m {secs:.0f}s"


class _LoadDashboardDataWorker(QThread):
    """Background worker: load ALL duplicate-check result_json blobs from
    the cache DB. On a network share with many modules this is the dominant
    bottleneck for opening the Dashboard (each result can be several MB of
    JSON), so doing it off the UI thread keeps the app responsive."""
    done = Signal(list)   # list of {module, model, pairs, updated_at}
    error = Signal(str)

    def __init__(self, cache: TrekCache):
        super().__init__()
        self._cache = cache

    def run(self):
        try:
            with LOG.timed("Dashboard", "Load all cached duplicate-check results") as t:
                entries = self._cache.list_duplicate_check_modules()
                t.details["module_count"] = len(entries)
                raw_modules = []
                for entry in entries:
                    data = self._cache.get_duplicate_check_result(entry["module_key"])
                    if not data:
                        continue
                    result = trek_similarity.DuplicateCheckResult.from_dict(data)
                    if not result.pairs:
                        continue
                    raw_modules.append({
                        "module": entry["syt_module"],
                        "model": entry.get("llm_model", ""),
                        "pairs": result.pairs,
                        "updated_at": entry.get("updated_at", ""),
                    })
                t.details["loaded_modules"] = len(raw_modules)
                t.details["total_pairs"] = sum(len(m["pairs"]) for m in raw_modules)
                self.done.emit(raw_modules)
        except Exception as exc:
            self.error.emit(str(exc))


class OverviewDashboardDialog(QDialog):
    """Management dashboard: bird's-eye view of LLM performance and human
    review progress across ALL cached modules. Shows summary cards, a
    per-module breakdown table, and aggregate charts."""

    def __init__(self, cache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overview Dashboard — All Modules")
        self.resize(1400, 850)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._cache = cache
        self._raw_modules = []
        self._load_worker = None
        lay = QVBoxLayout(self)

        # --- Top: summary cards ---
        self._cards_row = QHBoxLayout()
        lay.addLayout(self._cards_row)

        # --- Filter row ---
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItem("All pairs (full overview)", "all")
        self._view_combo.addItem("LLM: Same scenario — review breakdown", "llm_same")
        self._view_combo.addItem("LLM: Partial overlap — review breakdown", "llm_partial")
        self._view_combo.addItem("LLM: Different scenario — review breakdown", "llm_different")
        self._view_combo.addItem("Only reviewed pairs", "reviewed_only")
        self._view_combo.currentIndexChanged.connect(self._refresh_view)
        filter_row.addWidget(self._view_combo)
        filter_row.addStretch()
        self._filter_info_lbl = QLabel("")
        self._filter_info_lbl.setStyleSheet("font-size:11px;")
        filter_row.addWidget(self._filter_info_lbl)
        lay.addLayout(filter_row)

        # --- Middle: per-module table ---
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table, 1)

        # --- Bottom: buttons ---
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._load_data)
        btn_row.addWidget(btn_refresh)

        btn_export = QPushButton("Export JSON")
        btn_export.setObjectName("btn_secondary")
        btn_export.clicked.connect(self._export_json)
        btn_row.addWidget(btn_export)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._load_data()

    def _load_data(self):
        """Kick off a background load of all cached duplicate-check results.
        The heavy lifting (reading and parsing each result_json over the
        network) happens in _LoadDashboardDataWorker; once it finishes,
        _on_data_loaded() populates the dashboard."""
        if self._load_worker is not None and self._load_worker.isRunning():
            return  # already loading
        self._filter_info_lbl.setText("Loading data...")
        self._table.setRowCount(0)
        self._load_worker = _LoadDashboardDataWorker(self._cache)
        self._load_worker.done.connect(self._on_data_loaded)
        self._load_worker.error.connect(lambda msg: self._filter_info_lbl.setText(f"Error: {msg}"))
        self._load_worker.start()

    def _on_data_loaded(self, raw_modules: list):
        self._raw_modules = raw_modules
        self._refresh_view()

    def _refresh_view(self):
        """Filter pairs based on the selected view and recompute everything."""
        view = self._view_combo.currentData() if hasattr(self, "_view_combo") else "all"

        # Build per-module stats from filtered pairs
        self._module_data = []
        for raw in self._raw_modules:
            # Apply view filter
            if view == "llm_same":
                pairs = [p for p in raw["pairs"] if p.llm_verdict == "same_scenario"]
            elif view == "llm_partial":
                pairs = [p for p in raw["pairs"] if p.llm_verdict == "partial_overlap"]
            elif view == "llm_different":
                pairs = [p for p in raw["pairs"] if p.llm_verdict == "different_scenario"]
            elif view == "reviewed_only":
                pairs = [p for p in raw["pairs"] if p.review_status]
            else:
                pairs = raw["pairs"]

            total = len(pairs)
            if total == 0:
                continue

            reviewed = [p for p in pairs if p.review_status]
            reviewed_count = len(reviewed)
            llm_agree = sum(1 for p in reviewed if p.llm_verdict and p.llm_verdict == p.review_status)
            llm_disagree = sum(1 for p in reviewed if p.llm_verdict and p.llm_verdict != p.review_status and p.llm_verdict != "error")
            human_same = sum(1 for p in reviewed if p.review_status == "same_scenario")
            human_partial = sum(1 for p in reviewed if p.review_status == "partial_overlap")
            human_different = sum(1 for p in reviewed if p.review_status == "different_scenario")

            llm_same = sum(1 for p in pairs if p.llm_verdict == "same_scenario")
            llm_partial = sum(1 for p in pairs if p.llm_verdict == "partial_overlap")
            llm_different = sum(1 for p in pairs if p.llm_verdict == "different_scenario")

            accuracy_denom = llm_agree + llm_disagree
            accuracy_pct = round(llm_agree / accuracy_denom * 100, 1) if accuracy_denom else None

            self._module_data.append({
                "module": raw["module"],
                "model": raw["model"],
                "total": total,
                "reviewed": reviewed_count,
                "review_pct": round(reviewed_count / total * 100, 1) if total else 0,
                "llm_agree": llm_agree,
                "llm_disagree": llm_disagree,
                "accuracy_pct": accuracy_pct,
                "human_same": human_same,
                "human_partial": human_partial,
                "human_different": human_different,
                "llm_same": llm_same,
                "llm_partial": llm_partial,
                "llm_different": llm_different,
                "updated_at": raw["updated_at"],
            })

        # Update info label
        total_all = sum(len(r["pairs"]) for r in self._raw_modules)
        total_filtered = sum(m["total"] for m in self._module_data)
        view_labels = {
            "all": "All pairs",
            "llm_same": "LLM: Same scenario",
            "llm_partial": "LLM: Partial overlap",
            "llm_different": "LLM: Different scenario",
            "reviewed_only": "Reviewed pairs only",
        }
        self._filter_info_lbl.setText(
            f"{view_labels.get(view, view)}: {total_filtered} of {total_all} pairs across {len(self._module_data)} modules"
        )

        self._update_cards()
        self._update_table()

    def _update_cards(self):
        """Build/rebuild the summary cards row."""
        # Clear old cards
        while self._cards_row.count():
            item = self._cards_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_pairs = sum(m["total"] for m in self._module_data)
        total_reviewed = sum(m["reviewed"] for m in self._module_data)
        total_agree = sum(m["llm_agree"] for m in self._module_data)
        total_disagree = sum(m["llm_disagree"] for m in self._module_data)
        review_pct = round(total_reviewed / total_pairs * 100, 1) if total_pairs else 0
        accuracy_denom = total_agree + total_disagree
        accuracy_pct = round(total_agree / accuracy_denom * 100, 1) if accuracy_denom else None

        cards = [
            ("Modules", str(len(self._module_data)), ACCENT),
            ("Total Pairs", f"{total_pairs:,}", TEXT),
            ("Reviewed", f"{total_reviewed:,} ({review_pct}%)", AMBER_COLOR),
            ("LLM Accuracy", f"{accuracy_pct}%" if accuracy_pct is not None else "N/A", SUCCESS_TEXT if (accuracy_pct or 0) >= 70 else DANGER_TEXT),
            ("LLM Correct", str(total_agree), SUCCESS_TEXT),
            ("LLM Wrong", str(total_disagree), DANGER_TEXT),
        ]

        for title, value, color in cards:
            card = QGroupBox()
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(12, 8, 12, 8)
            val_lbl = QLabel(value)
            val_lbl.setStyleSheet(f"font-size:22px;font-weight:bold;color:{color};")
            val_lbl.setAlignment(Qt.AlignCenter)
            card_lay.addWidget(val_lbl)
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-size:11px;")
            title_lbl.setAlignment(Qt.AlignCenter)
            card_lay.addWidget(title_lbl)
            self._cards_row.addWidget(card)

    def _update_table(self):
        """Populate the per-module breakdown table with consolidated columns."""
        headers = [
            "Module", "Model", "Pairs", "Reviewed",
            "LLM Accuracy", "Same (H/L)", "Partial (H/L)",
            "Different (H/L)", "Updated",
        ]
        self._table.setSortingEnabled(False)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(0)

        def _add_row(row_data, bold=False):
            row = self._table.rowCount()
            self._table.insertRow(row)
            font = QFont("Inter", 11, QFont.Bold) if bold else QFont()

            # Module
            item = QTableWidgetItem(row_data["module"])
            if bold:
                item.setFont(font)
            self._table.setItem(row, 0, item)

            # Model
            self._table.setItem(row, 1, QTableWidgetItem(row_data.get("model", "")))

            # Pairs
            item = QTableWidgetItem()
            item.setData(Qt.EditRole, row_data["total"])
            if bold:
                item.setFont(font)
            self._table.setItem(row, 2, item)

            # Reviewed — "45 (1.3%)"
            rev = row_data["reviewed"]
            pct = row_data["review_pct"]
            item = QTableWidgetItem(f"{rev} ({pct}%)")
            item.setData(Qt.EditRole, rev)
            item.setForeground(QColor(AMBER_COLOR))
            if bold:
                item.setFont(font)
            self._table.setItem(row, 3, item)

            # LLM Accuracy — "84% (38✔ 7✘)"
            acc = row_data["accuracy_pct"]
            agree = row_data["llm_agree"]
            disagree = row_data["llm_disagree"]
            if acc is not None:
                item = QTableWidgetItem(f"{acc}% ({agree}✔ {disagree}✘)")
                item.setData(Qt.EditRole, acc)
                item.setForeground(QColor(SUCCESS_TEXT if acc >= 70 else DANGER_TEXT))
            else:
                item = QTableWidgetItem("N/A")
                item.setForeground(QColor(TEXT_DIM))
            if bold:
                item.setFont(font)
            self._table.setItem(row, 4, item)

            # Same (H/L)
            item = QTableWidgetItem(f"{row_data['human_same']}/{row_data['llm_same']}")
            item.setForeground(QColor(DANGER_TEXT))
            if bold:
                item.setFont(font)
            self._table.setItem(row, 5, item)

            # Partial (H/L)
            item = QTableWidgetItem(f"{row_data['human_partial']}/{row_data['llm_partial']}")
            item.setForeground(QColor(AMBER_COLOR))
            if bold:
                item.setFont(font)
            self._table.setItem(row, 6, item)

            # Different (H/L)
            item = QTableWidgetItem(f"{row_data['human_different']}/{row_data['llm_different']}")
            item.setForeground(QColor(SUCCESS_TEXT))
            if bold:
                item.setFont(font)
            self._table.setItem(row, 7, item)

            # Updated
            self._table.setItem(row, 8, QTableWidgetItem(row_data.get("updated_at", "")))

        for row_data in self._module_data:
            _add_row(row_data)

        # Totals row
        if self._module_data:
            totals = {"module": "TOTAL", "model": "", "updated_at": ""}
            for key in ("total", "reviewed", "llm_agree", "llm_disagree",
                        "human_same", "human_partial", "human_different",
                        "llm_same", "llm_partial", "llm_different"):
                totals[key] = sum(m[key] for m in self._module_data)
            t = totals["total"]
            r = totals["reviewed"]
            totals["review_pct"] = round(r / t * 100, 1) if t else 0
            acc_d = totals["llm_agree"] + totals["llm_disagree"]
            totals["accuracy_pct"] = round(totals["llm_agree"] / acc_d * 100, 1) if acc_d else None
            _add_row(totals, bold=True)

        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

    def _export_json(self):
        """Export dashboard data as JSON."""
        if not self._module_data:
            QMessageBox.information(self, "Nothing to Export", "No module data loaded.")
            return

        total_pairs = sum(m["total"] for m in self._module_data)
        total_reviewed = sum(m["reviewed"] for m in self._module_data)
        total_agree = sum(m["llm_agree"] for m in self._module_data)
        total_disagree = sum(m["llm_disagree"] for m in self._module_data)
        acc_denom = total_agree + total_disagree

        data = {
            "export_meta": {
                "type": "overview_dashboard",
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "modules_count": len(self._module_data),
            },
            "summary": {
                "total_pairs": total_pairs,
                "total_reviewed": total_reviewed,
                "review_percentage": round(total_reviewed / total_pairs * 100, 1) if total_pairs else 0,
                "llm_correct": total_agree,
                "llm_wrong": total_disagree,
                "llm_accuracy_pct": round(total_agree / acc_denom * 100, 1) if acc_denom else None,
            },
            "modules": self._module_data,
        }
        _export_dict_to_json(self, data, "overview_dashboard.json")


class DuplicateResultsDialog(QDialog):
    """Wraps SideBySideResultsWidget in a standalone dialog with a Close
    button, shown automatically when a real 'Check Duplicates' run against
    TREK data completes.

    Resizable/maximizable: Qt's QDialog has no title-bar maximize button
    by default, only a system menu with minimal resizing affordances --
    explicitly enabling WindowMinMaxButtonsHint gives it a real title-bar
    maximize button, and the in-window "⛶ Maximize" button is a reliable
    fallback in case the window manager doesn't render title-bar hints
    consistently.
    """

    def __init__(self, result: "trek_similarity.DuplicateCheckResult", module_name: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duplicate Check Results")
        self.resize(1300, 750)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._result = result
        self._module_name = module_name
        lay = QVBoxLayout(self)

        self._results_widget = SideBySideResultsWidget(self, on_review_changed=self._persist_review)
        lay.addWidget(self._results_widget, 1)
        self._results_widget.set_result(result)

        btn_row = QHBoxLayout()
        self._btn_maximize = QPushButton("⛶  Maximize")
        self._btn_maximize.setObjectName("btn_secondary")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        btn_row.addWidget(self._btn_maximize)

        btn_metrics = QPushButton("📊  Metrics")
        btn_metrics.setObjectName("btn_secondary")
        btn_metrics.clicked.connect(self._show_metrics)
        btn_row.addWidget(btn_metrics)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _persist_review(self):
        """Save the current result (with updated review fields) to cache."""
        if self._result and self._module_name:
            try:
                module_key = trek_cache.key_duplicate_check(
                    PROJECT_ID, CAMPAIGN_ID,
                    self._module_name,
                    self._result.llm_model or "unknown",
                )
                CACHE.set_duplicate_check_result(
                    module_key, self._module_name, self._result.to_dict(),
                )
            except Exception:
                pass

    def _show_metrics(self):
        dlg = DuplicateMetricsDialog(self._result, module_name=self._module_name)
        _open_independent_window(self, dlg)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_maximize.setText("⛶  Maximize")
        else:
            self.showMaximized()
            self._btn_maximize.setText("🗗  Restore")


class _LoadDupCheckResultWorker(QThread):
    """Background worker: read and parse a single duplicate-check result_json
    blob from the cache DB. This is the single most expensive I/O on a
    network-share cache (each result can be several MB of JSON that must
    travel over SMB), so doing it off the UI thread keeps the app responsive
    while the user waits, instead of freezing the entire GUI."""
    done = Signal(str, object, str)   # (module_key, DuplicateCheckResult, syt_module)
    error = Signal(str)

    def __init__(self, cache: TrekCache, module_key: str, syt_module: str):
        super().__init__()
        self._cache = cache
        self._module_key = module_key
        self._syt_module = syt_module

    def run(self):
        try:
            with LOG.timed("Results", f"Load cached result for '{self._syt_module}'") as t:
                data = self._cache.get_duplicate_check_result(self._module_key)
                if data is None:
                    t.details["status"] = "not_found"
                    self.error.emit(f"No cached result for {self._syt_module}")
                    return
                result = trek_similarity.DuplicateCheckResult.from_dict(data)
                t.details["pair_count"] = len(result.pairs)
                self.done.emit(self._module_key, result, self._syt_module)
        except Exception as exc:
            self.error.emit(str(exc))


class CachedDuplicateResultsDialog(QDialog):
    """Browse every SYT module that currently has a persisted 'Check
    Duplicates' run in trek_cache.sqlite3 (see
    TrekMainWindow._on_duplicate_check_ready(), which saves the full
    result after each real run via CACHE.set_duplicate_check_result()),
    switching between modules via a combo box without re-running
    anything -- reuses the same SideBySideResultsWidget/
    DuplicateMetricsDialog as a live run.
    """

    def __init__(self, cache: TrekCache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cached Duplicate Check Results")
        self.resize(1300, 750)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._cache = cache
        self._module_entries: Dict[str, dict] = {}
        self._current_result: Optional["trek_similarity.DuplicateCheckResult"] = None
        # In-memory cache of loaded results so switching back to a
        # previously-viewed module is instant (no network round trip).
        # Key = module_key, value = (DuplicateCheckResult, syt_module).
        self._result_cache: Dict[str, tuple] = {}
        lay = QVBoxLayout(self)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Module:"))
        self._module_combo = QComboBox()
        self._module_combo.setMinimumWidth(320)
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        top_row.addWidget(self._module_combo)

        top_row.addWidget(QLabel("Show:"))
        self._verdict_filter = QComboBox()
        self._verdict_filter.addItem("All verdicts", "all")
        self._verdict_filter.addItem("🔴 Same scenario", "same_scenario")
        self._verdict_filter.addItem("🟡 Partial overlap", "partial_overlap")
        self._verdict_filter.addItem("🔴🟡 Same + Partial", "same_partial")
        self._verdict_filter.addItem("🟢 Different scenario", "different_scenario")
        self._verdict_filter.setToolTip(
            "Pre-filter which LLM verdicts to display.\n"
            "Uses the already-loaded result in memory — switching is instant.\n"
            "Useful for large modules where you only care about duplicates."
        )
        self._verdict_filter.currentIndexChanged.connect(self._on_verdict_filter_changed)
        top_row.addWidget(self._verdict_filter)

        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet(f"color:{TEXT_DIM}")
        top_row.addWidget(self._lbl_info, 1)
        btn_refresh = QPushButton("↻  Refresh List")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._reload_modules)
        top_row.addWidget(btn_refresh)
        lay.addLayout(top_row)

        self._results_widget = SideBySideResultsWidget(self, on_review_changed=self._persist_review)
        lay.addWidget(self._results_widget, 1)

        btn_row = QHBoxLayout()
        self._btn_maximize = QPushButton("⛶  Maximize")
        self._btn_maximize.setObjectName("btn_secondary")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        btn_row.addWidget(self._btn_maximize)

        btn_metrics = QPushButton("📊  Metrics")
        btn_metrics.setObjectName("btn_secondary")
        btn_metrics.clicked.connect(self._show_metrics)
        btn_row.addWidget(btn_metrics)

        btn_compare = QPushButton("🔀  Compare Model Versions")
        btn_compare.setObjectName("btn_secondary")
        btn_compare.setToolTip(
            "Compare two cached versions of the same module that used\n"
            "different LLM models, pair by pair."
        )
        btn_compare.clicked.connect(self._show_compare_dialog)
        btn_row.addWidget(btn_compare)

        btn_export_json = QPushButton("📄  Export to JSON")
        btn_export_json.setObjectName("btn_secondary")
        btn_export_json.setToolTip("Export the currently selected module's full result (summary + cost + pairs) as structured JSON.")
        btn_export_json.clicked.connect(self._export_to_json)
        btn_row.addWidget(btn_export_json)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._reload_modules()

    def _reload_modules(self):
        with LOG.timed("Results", "List cached duplicate-check modules") as t:
            entries = self._cache.list_duplicate_check_modules()
            t.details["module_count"] = len(entries)
        self._module_entries = {e["module_key"]: e for e in entries}
        self._module_combo.blockSignals(True)
        self._module_combo.clear()
        # Add a placeholder prompt so no module is auto-loaded on open --
        # let the user choose which module to load (matters when each load
        # can take 4-14s over a network share).
        self._module_combo.addItem("— Select a module —", None)
        for e in entries:
            model_part = f", {e['llm_model']}" if e.get("llm_model") else ""
            self._module_combo.addItem(f"{e['syt_module']}  ({e['pair_count']} pairs{model_part})", e["module_key"])
        self._module_combo.blockSignals(False)
        if not entries:
            self._lbl_info.setText("No cached 'Check Duplicates' results yet -- run it at least once.")
        else:
            self._lbl_info.setText(f"{len(entries)} module(s) available — select one to load.")

    def _on_module_changed(self, idx: int):
        if idx < 0 or not self._module_entries:
            return
        module_key = self._module_combo.itemData(idx)
        if module_key is None:
            return  # placeholder "— Select a module —"
        entry = self._module_entries.get(module_key)
        if entry is None:
            return
        # Serve from in-memory cache if we've already loaded this module
        # (switching back to a previously-viewed module = instant, zero
        # network I/O — critical for a 50+ MB blob on a network share
        # that can take 4-77 seconds to re-read depending on SMB caching).
        if module_key in self._result_cache:
            cached_result, cached_syt = self._result_cache[module_key]
            self._lbl_info.setText("Loading from memory...")
            QApplication.processEvents()
            t0 = time.perf_counter()
            self._on_result_loaded(module_key, cached_result, cached_syt)
            ms = (time.perf_counter() - t0) * 1000
            LOG.log("Results", f"Load '{cached_syt}' from memory cache",
                    duration_ms=ms, pair_count=len(cached_result.pairs))
            return
        # Fetch from the network-share cache in the background.
        self._load_start = time.perf_counter()
        self._lbl_info.setText("Loading from database...  (00:00)")
        QApplication.processEvents()
        self._module_combo.setEnabled(False)

        # Live elapsed timer so the user sees the loading isn't frozen.
        if not hasattr(self, "_load_timer"):
            self._load_timer = QTimer(self)
            self._load_timer.timeout.connect(self._update_load_elapsed)
        self._load_timer.start(1000)

        self._load_worker = _LoadDupCheckResultWorker(self._cache, module_key, entry["syt_module"])
        self._load_worker.done.connect(self._on_result_loaded)
        self._load_worker.error.connect(self._on_result_load_error)
        self._load_worker.finished.connect(lambda: self._module_combo.setEnabled(True))
        self._load_worker.start()

    def _update_load_elapsed(self):
        elapsed = time.perf_counter() - self._load_start
        mins, secs = divmod(int(elapsed), 60)
        self._lbl_info.setText(f"Loading from database...  ({mins:02d}:{secs:02d})")

    def _on_result_loaded(self, module_key: str, result: "trek_similarity.DuplicateCheckResult", syt_module: str):
        if hasattr(self, "_load_timer"):
            self._load_timer.stop()
        entry = self._module_entries.get(module_key)
        self._current_result = result
        self._current_module_name = syt_module
        # Store in memory so switching back is instant.
        self._result_cache[module_key] = (result, syt_module)
        self._module_combo.setEnabled(True)
        # Apply the current verdict filter before rendering.
        self._apply_verdict_filter()
        # Update the combo label with the real pair count.
        real_count = len(result.pairs)
        idx = self._module_combo.currentIndex()
        if idx >= 0:
            model_part = f", {entry['llm_model']}" if entry and entry.get("llm_model") else ""
            self._module_combo.setItemText(idx, f"{syt_module}  ({real_count} pairs{model_part})")

    def _apply_verdict_filter(self):
        """Filter the current result by the selected LLM verdict and pass
        the filtered view to the results widget. The full unfiltered result
        stays in self._current_result / self._result_cache so switching
        filters is instant (no re-load from DB)."""
        if self._current_result is None:
            return
        verdict_key = self._verdict_filter.currentData()
        all_pairs = self._current_result.pairs
        if verdict_key == "same_scenario":
            shown = [p for p in all_pairs if p.llm_verdict == "same_scenario"]
        elif verdict_key == "partial_overlap":
            shown = [p for p in all_pairs if p.llm_verdict == "partial_overlap"]
        elif verdict_key == "same_partial":
            shown = [p for p in all_pairs if p.llm_verdict in ("same_scenario", "partial_overlap")]
        elif verdict_key == "different_scenario":
            shown = [p for p in all_pairs if p.llm_verdict == "different_scenario"]
        else:
            shown = all_pairs

        # Build a shallow copy with the filtered pairs list so the widget
        # shows only the filtered subset, but cost/token totals stay correct.
        import dataclasses
        filtered = dataclasses.replace(self._current_result, pairs=shown)
        self._results_widget.set_result(filtered)

        entry = self._module_entries.get(self._module_combo.currentData())
        filter_label = self._verdict_filter.currentText()
        total = len(all_pairs)
        showing = len(shown)
        age_info = ""
        if entry:
            age_info = f"  ·  cached {trek_cache.format_age(entry['updated_at'])} ({entry['updated_at']})"
        if verdict_key == "all":
            self._lbl_info.setText(f"Loaded {total:,} pairs{age_info}")
        else:
            self._lbl_info.setText(f"Showing {showing:,} / {total:,} pairs ({filter_label}){age_info}")

    def _on_verdict_filter_changed(self, _idx: int):
        """Re-filter the already-loaded result — instant, no DB access."""
        self._apply_verdict_filter()

    def _on_result_load_error(self, msg: str):
        if hasattr(self, "_load_timer"):
            self._load_timer.stop()
        self._lbl_info.setText(f"Error loading result: {msg}")

    def _persist_review(self):
        """Save the current result (with updated review fields) to cache."""
        if self._current_result and hasattr(self, "_current_module_name"):
            try:
                module_key = self._module_combo.currentData()
                if module_key:
                    self._cache.set_duplicate_check_result(
                        module_key,
                        self._current_module_name,
                        self._current_result.to_dict(),
                    )
            except Exception:
                pass

    def refresh_theme(self):
        """Re-render all baked-in colours after a live theme switch.
        Called from TrekMainWindow._change_theme() for every open dialog."""
        self._lbl_info.setStyleSheet(f"color:{TEXT_DIM}")
        if self._current_result is not None:
            self._results_widget.set_result(self._current_result)

    def _show_metrics(self):
        if self._current_result is None:
            return
        dlg = DuplicateMetricsDialog(self._current_result, module_name=getattr(self, "_current_module_name", None))
        _open_independent_window(self, dlg)

    def _export_to_json(self):
        if not self._current_result or not self._current_result.pairs:
            QMessageBox.information(self, "Nothing to Export", "No cached pairs for the selected module.")
            return

        # Pop up a small dialog with checkboxes
        dlg = QDialog(self)
        dlg.setWindowTitle("Export JSON")
        dlg.setFixedSize(360, 160)
        dlg_lay = QVBoxLayout(dlg)

        dlg_lay.addWidget(QLabel("Select which reports to export:"))
        chk_full = QCheckBox("Full Analysis Report (all data)")
        chk_full.setChecked(True)
        dlg_lay.addWidget(chk_full)
        chk_review = QCheckBox("Human Review Report (reviewed pairs only)")
        chk_review.setChecked(True)
        dlg_lay.addWidget(chk_review)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)
        btn_export = QPushButton("Export")
        btn_export.setObjectName("btn_success")
        btn_export.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_export)
        dlg_lay.addLayout(btn_row)

        if dlg.exec() != QDialog.Accepted:
            return

        module_name = getattr(self, "_current_module_name", None) or "module"
        safe = module_name.replace(" ", "_").replace("/", "-")

        if chk_full.isChecked():
            self._export_full_json(module_name, safe)
        if chk_review.isChecked():
            self._export_human_review_json(module_name, safe)

    def _export_human_review_json(self, module_name: str, safe_name: str):
        """Export only reviewed pairs in a clean, easy-to-track format."""
        reviewed = [p for p in self._current_result.pairs if p.review_status]
        if not reviewed:
            QMessageBox.information(self, "No Reviews", "No pairs have been reviewed yet.")
            return

        review_counts = {"same_scenario": 0, "partial_overlap": 0, "different_scenario": 0}
        llm_agree, llm_disagree = 0, 0
        pairs_json = []
        for p in reviewed:
            review_counts[p.review_status] = review_counts.get(p.review_status, 0) + 1
            if p.llm_verdict and p.llm_verdict != "error":
                if p.llm_verdict == p.review_status:
                    llm_agree += 1
                else:
                    llm_disagree += 1
            pairs_json.append({
                "syt_id": p.syt_id,
                "counterpart_id": p.swt_id,
                "counterpart_type": p.counterpart_type,
                "score": round(p.score, 4),
                "rag_classification": p.classification,
                "llm_verdict": p.llm_verdict or None,
                "human_verdict": p.review_status,
                "llm_agrees_with_human": (
                    p.llm_verdict == p.review_status
                    if p.llm_verdict and p.llm_verdict != "error" else None
                ),
                "comment": p.review_comment or None,
                "reviewer": p.reviewer,
                "reviewed_at": p.reviewed_at,
            })

        data = {
            "export_meta": {
                "type": "human_review",
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "module": module_name,
                "llm_model": self._current_result.llm_model or None,
            },
            "summary": {
                "total_pairs_in_module": len(self._current_result.pairs),
                "reviewed_pairs": len(reviewed),
                "human_verdicts": review_counts,
                "llm_accuracy": {
                    "agree": llm_agree,
                    "disagree": llm_disagree,
                    "accuracy_pct": round(llm_agree / (llm_agree + llm_disagree) * 100, 1) if (llm_agree + llm_disagree) else None,
                },
            },
            "reviewed_pairs": pairs_json,
        }
        _export_dict_to_json(self, data, f"human_review_{safe_name}.json")

    def _export_full_json(self, module_name: str, safe_name: str):
        """Full analysis export with all data."""
        module_key = self._module_combo.currentData()
        entry = self._module_entries.get(module_key) or {}
        m = self._current_result.compute_metrics()
        spent = DuplicateMetricsDialog._true_total_spent(module_name)

        pairs_json = []
        for p in self._current_result.pairs:
            pairs_json.append({
                "syt_id": p.syt_id,
                "counterpart_id": p.swt_id,
                "counterpart_type": p.counterpart_type,
                "rag_classification": p.classification,
                "exact_match": p.exact_match,
                "scores": {
                    "final": round(p.score, 4),
                    "bm25": round(p.bm25_score, 4),
                    "vector": round(p.vector_score, 4),
                    "sequence": round(p.sequence_score, 4) if p.has_sequence else None,
                },
                "llm": {
                    "verdict": p.llm_verdict,
                    "reasoning": p.llm_reasoning,
                    "cost_usd": round(p.llm_cost_usd, 6),
                    "checked_at": p.checked_at or None,
                } if p.llm_verdict or p.llm_reasoning else None,
                "human_review": {
                    "verdict": p.review_status,
                    "comment": p.review_comment,
                    "reviewer": p.reviewer,
                    "reviewed_at": p.reviewed_at or None,
                } if p.review_status else None,
                "syt_text": p.syt_text,
                "counterpart_text": p.swt_text,
            })

        data = {
            "export_meta": {
                "type": "full_analysis",
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "module": module_name,
                "llm_model": m.get("llm_model") or None,
                "cached_at": entry.get("updated_at"),
            },
            "summary": {
                "total_pairs": m["total_pairs"],
                "by_counterpart_type": m["counterpart_type_counts"],
                "rag_classification": {"counts": m["rag_counts"], "percentages": m["rag_percentages"]},
                "llm_verdict": {
                    "judged_total": m["llm_judged_total"],
                    "counts": m["llm_counts"],
                    "percentages": m["llm_percentages"],
                },
                "agreement": {"counts": m["agreement_counts"], "agreement_percentage": m["agreement_percentage"]},
                "human_review": {
                    "counts": m.get("review_counts", {}),
                    "llm_vs_human": m.get("llm_vs_human", {}),
                },
                "timing_seconds": {"rag_stage": m["rag_duration_seconds"], "llm_stage": m["llm_duration_seconds"]},
            },
            "cost": {
                "embeddings": {
                    "last_run_usd": m["embed_cost_usd"],
                    "current_list_value_usd": m["embed_cost_historical_usd"],
                    "served_from_cache_this_run": m["embed_cached_texts"],
                },
                "llm": {
                    "last_run_usd": m["llm_cost_usd"],
                    "current_list_value_usd": m["llm_cost_historical_usd"],
                    "served_from_cache_this_run": m["llm_cached_pairs"],
                },
            },
            "pairs": pairs_json,
            "skipped_ids": self._current_result.skipped,
        }
        if spent is not None:
            embed_spent, llm_spent, run_count = spent
            data["cost"]["total_money_ever_spent"] = {
                "embeddings_usd": embed_spent,
                "llm_usd": llm_spent,
                "recorded_runs_counted": run_count,
                "note": (
                    "True cumulative real spend across every run ever recorded for this "
                    "module (all models) -- only ever increases, unlike current_list_value_usd."
                ),
            }

        _export_dict_to_json(self, data, f"trek_cached_results_{safe_name}.json")

    def _show_compare_dialog(self):
        dlg = CompareModelVersionsDialog(self._cache)
        _open_independent_window(self, dlg)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_maximize.setText("⛶  Maximize")
        else:
            self.showMaximized()
            self._btn_maximize.setText("🗗  Restore")


# Workers kept alive until they finish, even if the dialog that started
# them is closed first (a QThread destroyed while running crashes the app).
_BG_COMPARE_WORKERS: set = set()


class _CompareLoadWorker(QThread):
    """Background load for CompareModelVersionsDialog: read + decompress +
    parse BOTH cached results, build the pair-by-pair comparison rows and
    both versions' metrics -- all off the UI thread. A large module (e.g.
    85k pairs, one version stored as a ~640 MB legacy JSON blob) used to
    freeze the whole app for minutes doing exactly this on the UI thread."""
    done  = Signal(int, object)    # (request_seq, payload dict)
    error = Signal(int, str)

    def __init__(self, cache, seq: int, key_a: str, key_b: str):
        super().__init__()
        self._cache, self._seq, self._key_a, self._key_b = cache, seq, key_a, key_b

    def run(self):
        try:
            with LOG.timed("Compare", f"Load comparison {self._key_a} vs {self._key_b}") as t:
                data_a = self._cache.get_duplicate_check_result(self._key_a)
                data_b = (data_a if self._key_b == self._key_a
                          else self._cache.get_duplicate_check_result(self._key_b))
                if data_a is None or data_b is None:
                    self.error.emit(self._seq, "One of the selected versions has no cached result.")
                    return
                res_a = trek_similarity.DuplicateCheckResult.from_dict(data_a)
                res_b = trek_similarity.DuplicateCheckResult.from_dict(data_b)
                del data_a, data_b

                by_a = {(p.syt_id, p.swt_id): p for p in res_a.pairs}
                by_b = {(p.syt_id, p.swt_id): p for p in res_b.pairs}
                all_keys = sorted(set(by_a) | set(by_b))
                labels = trek_similarity.LLM_VERDICT_LABELS
                agree = disagree = only_a = only_b = 0
                agree_by_verdict = {"same_scenario": 0, "partial_overlap": 0, "different_scenario": 0}
                rows = []
                for key in all_keys:
                    pa, pb = by_a.get(key), by_b.get(key)
                    ref = pa or pb
                    va = pa.llm_verdict if pa else None
                    vb = pb.llm_verdict if pb else None
                    if pa is None:
                        agreement, fkey = "only in B", "only_b"; only_b += 1
                    elif pb is None:
                        agreement, fkey = "only in A", "only_a"; only_a += 1
                    elif va and vb:
                        if va == vb:
                            agreement, fkey = "✅ Agree", "agree"; agree += 1
                            if va in agree_by_verdict:
                                agree_by_verdict[va] += 1
                        else:
                            agreement, fkey = "⚠️ Disagree", "disagree"; disagree += 1
                    else:
                        agreement, fkey = "n/a", "na"
                    ctype = ref.counterpart_type
                    rows.append((
                        key[0],
                        key[1] + ("  (related SYT)" if ctype == "related_syt" else ""),
                        ctype,
                        labels.get(va, (va or "—", None))[0],
                        labels.get(vb, (vb or "—", None))[0],
                        agreement, fkey, pa, pb, va, vb,
                    ))
                payload = {
                    "res_a": res_a, "res_b": res_b,
                    "by_a": by_a, "by_b": by_b, "all_keys": all_keys, "rows": rows,
                    "agree": agree, "disagree": disagree, "only_a": only_a, "only_b": only_b,
                    "agree_by_verdict": agree_by_verdict,
                    "m_a": res_a.compute_metrics(), "m_b": res_b.compute_metrics(),
                }
                t.details.update(pairs=len(rows))
            self.done.emit(self._seq, payload)
        except Exception as exc:   # noqa: BLE001
            self.error.emit(self._seq, str(exc))


class _CompareTableModel(QAbstractTableModel):
    """Virtual table for the compared pairs: Qt only asks for the cells that
    are visible, so 85k rows show instantly (a QTableWidget needed ~500k
    item objects built on the UI thread). Row tuple layout -- see
    _CompareLoadWorker: (syt, counterpart, type, label_a, label_b,
    agreement, filter_key, pa, pb, verdict_a, verdict_b)."""
    HEADERS = ["SYT ID", "Counterpart", "Type", "Version A Verdict", "Version B Verdict", "Agreement"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []
        self.headers = list(self.HEADERS)

    def set_rows(self, rows: list, header_a: str, header_b: str):
        self.beginResetModel()
        self.rows = rows
        self.headers = ["SYT ID", "Counterpart", "Type", header_a, header_b, "Agreement"]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 6

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self.rows[index.row()][index.column()]
        if role == Qt.TextAlignmentRole and index.column() >= 3:
            return int(Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        """Sort the row list in Python (one key per row) -- Qt's proxy
        sorting calls data() millions of times for 85k rows (~20 s)."""
        if column < 0 or column > 5 or not self.rows:
            return
        self.layoutAboutToBeChanged.emit()
        self.rows.sort(key=lambda r: (r[column] or "").lower(),
                       reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()


class _CompareFilterProxy(QSortFilterProxyModel):
    """AND-combination of the dialog's per-column filters, evaluated straight
    on the source row tuples (no per-cell model calls)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.syt = self.counterpart = ""
        self.ctype = self.verdict_a = self.verdict_b = self.agreement = None

    def sort(self, column, order=Qt.AscendingOrder):
        # Delegate to the source model's fast Python sort; the proxy keeps
        # source order, so it never runs its own per-comparison sort.
        self.sourceModel().sort(column, order)

    def _active(self) -> bool:
        return bool(self.syt or self.counterpart or self.ctype or self.verdict_a
                    or self.verdict_b or self.agreement is not None)

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._active():
            return True
        r = self.sourceModel().rows[source_row]
        if self.syt and self.syt not in r[0].lower():
            return False
        if self.counterpart and self.counterpart not in r[1].lower():
            return False
        if self.ctype and r[2] != self.ctype:
            return False
        if self.verdict_a and (r[7] is None or r[9] != self.verdict_a):
            return False
        if self.verdict_b and (r[8] is None or r[10] != self.verdict_b):
            return False
        if self.agreement is not None and r[6] != self.agreement:
            return False
        return True


class CompareModelVersionsDialog(QDialog):
    """Compare two persisted 'Check Duplicates' versions of the SAME SYT
    module that used DIFFERENT LLM models (see key_duplicate_check()'s
    per-model versioning in trek_cache.py) pair by pair -- same layout as
    SideBySideResultsWidget (sortable pair list on the left, side-by-side
    content on the right), but with both models' verdict/reasoning shown
    together per pair so it's easy to see which model agrees/disagrees on
    which specific pairs, and by how much overall.
    """

    def __init__(self, cache: TrekCache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Model Versions")
        self.resize(1400, 820)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._cache = cache
        self._entries_by_module: Dict[str, List[dict]] = {}
        self._result_a: Optional["trek_similarity.DuplicateCheckResult"] = None
        self._result_b: Optional["trek_similarity.DuplicateCheckResult"] = None
        self._m_a: Optional[dict] = None
        self._m_b: Optional[dict] = None
        self._metrics_args: Optional[tuple] = None
        self._all_keys: list = []
        self._load_seq = 0
        lay = QVBoxLayout(self)

        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Module:"))
        self._module_combo = QComboBox()
        self._module_combo.setMinimumWidth(260)
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        pick_row.addWidget(self._module_combo)
        pick_row.addWidget(QLabel("  Version A:"))
        self._combo_a = QComboBox()
        self._combo_a.setMinimumWidth(220)
        self._combo_a.currentIndexChanged.connect(self._reload_comparison)
        pick_row.addWidget(self._combo_a)
        pick_row.addWidget(QLabel("  Version B:"))
        self._combo_b = QComboBox()
        self._combo_b.setMinimumWidth(220)
        self._combo_b.currentIndexChanged.connect(self._reload_comparison)
        pick_row.addWidget(self._combo_b)
        pick_row.addStretch()
        btn_refresh = QPushButton("↻  Refresh")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._reload_modules)
        pick_row.addWidget(btn_refresh)
        lay.addLayout(pick_row)

        self._summary_lbl = QLabel("Pick a module with 2+ cached model versions to compare.")
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(self._summary_lbl)

        self._metrics_lbl = QLabel("")
        self._metrics_lbl.setStyleSheet(
            f"background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;"
            f"padding:8px;color:{TEXT};font-size:12px;"
        )
        self._metrics_lbl.setWordWrap(True)
        self._metrics_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(self._metrics_lbl)

        # Left: sortable/filterable pair list | Right: side-by-side content
        # -- same layout convention as SideBySideResultsWidget.
        splitter = QSplitter(Qt.Horizontal)

        list_frame = QGroupBox("Compared Pairs (click a row for details, click a header to sort)")
        list_lay = QVBoxLayout(list_frame)

        # Per-column filter row -- same layout as SideBySideResultsWidget's
        # Compared Pairs table, so both tables behave consistently.
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        self._filter_syt = QLineEdit()
        self._filter_syt.setPlaceholderText("Filter SYT...")
        self._filter_syt.setFixedWidth(110)
        # Text filters are debounced (applied 300 ms after typing stops) so
        # a big comparison isn't re-filtered on every keystroke.
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(300)
        self._filter_timer.timeout.connect(self._apply_filter)
        self._filter_syt.textChanged.connect(self._filter_timer.start)
        filter_row.addWidget(self._filter_syt)

        self._filter_counterpart = QLineEdit()
        self._filter_counterpart.setPlaceholderText("Filter counterpart...")
        self._filter_counterpart.setFixedWidth(170)
        self._filter_counterpart.textChanged.connect(self._filter_timer.start)
        filter_row.addWidget(self._filter_counterpart)

        self._filter_type = QComboBox()
        self._filter_type.setFixedWidth(90)
        self._filter_type.addItem("All types", None)
        self._filter_type.addItem("SWT", "SWT")
        self._filter_type.addItem("related_syt", "related_syt")
        self._filter_type.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_type)

        self._filter_verdict_a = QComboBox()
        self._filter_verdict_a.addItem("All Verdicts (A)", None)
        for key, (label, _color) in trek_similarity.LLM_VERDICT_LABELS.items():
            self._filter_verdict_a.addItem(label, key)
        self._filter_verdict_a.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_verdict_a, 1)

        self._filter_verdict_b = QComboBox()
        self._filter_verdict_b.addItem("All Verdicts (B)", None)
        for key, (label, _color) in trek_similarity.LLM_VERDICT_LABELS.items():
            self._filter_verdict_b.addItem(label, key)
        self._filter_verdict_b.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_verdict_b, 1)

        self._filter_combo = QComboBox()
        self._filter_combo.setFixedWidth(100)
        self._filter_combo.addItems(["All", "✅ Agree", "⚠️ Disagree", "Only in A", "Only in B"])
        self._filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_combo)
        list_lay.addLayout(filter_row)

        self._model = _CompareTableModel(self)
        self._proxy = _CompareFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Interactive)
        self._table.setColumnWidth(0, 110)
        self._table.setColumnWidth(1, 170)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(5, 100)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(-1, Qt.AscendingOrder)   # keep worker order until user sorts
        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)
        list_lay.addWidget(self._table)
        splitter.addWidget(list_frame)

        detail_frame = QGroupBox("Side-by-Side Comparison")
        detail_lay = QVBoxLayout(detail_frame)
        self._detail_lbl = QLabel("Select a pair to see both models' verdicts and reasoning.")
        self._detail_lbl.setWordWrap(True)
        self._detail_lbl.setTextFormat(Qt.RichText)
        self._detail_lbl.setStyleSheet(
            f"background:{CODE_BG};border-radius:4px;padding:8px;color:{TEXT};font-size:12px;"
        )
        detail_lay.addWidget(self._detail_lbl)

        side_by_side = QHBoxLayout()
        syt_box = QVBoxLayout()
        syt_lbl = QLabel("SYT Content")
        syt_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        syt_box.addWidget(syt_lbl)
        self._syt_text_view = QTextEdit()
        self._syt_text_view.setReadOnly(True)
        syt_box.addWidget(self._syt_text_view)

        counterpart_box = QVBoxLayout()
        self._counterpart_lbl = QLabel("Counterpart Content")
        self._counterpart_lbl.setStyleSheet(f"color:{SUCCESS_TEXT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        counterpart_box.addWidget(self._counterpart_lbl)
        self._counterpart_text_view = QTextEdit()
        self._counterpart_text_view.setReadOnly(True)
        counterpart_box.addWidget(self._counterpart_text_view)

        side_by_side.addLayout(syt_box)
        side_by_side.addLayout(counterpart_box)
        detail_lay.addLayout(side_by_side, 1)

        splitter.addWidget(detail_frame)
        splitter.setSizes([560, 720])
        lay.addWidget(splitter, 1)

        btn_row = QHBoxLayout()
        btn_export_json = QPushButton("📄  Export to JSON")
        btn_export_json.setObjectName("btn_secondary")
        btn_export_json.setToolTip("Export the full comparison (summary + per-model metrics + pairs) as structured JSON.")
        btn_export_json.clicked.connect(self._export_to_json)
        btn_row.addWidget(btn_export_json)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._reload_modules()

    def refresh_theme(self):
        """Re-render all baked-in colours after a live theme switch."""
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        self._metrics_lbl.setStyleSheet(
            f"color:{TEXT};font-size:12px;padding:4px 0;"
        )
        self._detail_lbl.setStyleSheet(
            f"color:{TEXT};font-size:12px;padding:4px 0;"
        )
        self._counterpart_lbl.setStyleSheet(f"color:{SUCCESS_TEXT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        # Re-render the colour-baked metrics HTML from what is already
        # loaded (never re-load both results just for a theme change).
        if self._metrics_args is not None:
            self._metrics_lbl.setText(self._build_metrics_html(*self._metrics_args))

    def _reload_modules(self):
        entries = self._cache.list_duplicate_check_modules()
        by_module: Dict[str, List[dict]] = {}
        for e in entries:
            by_module.setdefault(e["syt_module"], []).append(e)
        for versions in by_module.values():
            versions.sort(key=lambda v: v["updated_at"], reverse=True)
        self._entries_by_module = by_module

        self._module_combo.blockSignals(True)
        self._module_combo.clear()
        smallest_idx, smallest_pairs = 0, None
        for module in sorted(by_module.keys()):
            versions = by_module[module]
            if len(versions) < 2:
                continue   # need at least 2 model versions to compare
            max_pairs = max(int(v.get("pair_count") or 0) for v in versions)
            self._module_combo.addItem(
                f"{module}  ({len(versions)} versions, up to {max_pairs:,} pairs)", module)
            if smallest_pairs is None or max_pairs < smallest_pairs:
                smallest_idx, smallest_pairs = self._module_combo.count() - 1, max_pairs
        self._module_combo.blockSignals(False)

        if self._module_combo.count():
            # Open on the SMALLEST module so the dialog appears instantly; a
            # big one (85k pairs) loads in the background when picked.
            self._module_combo.blockSignals(True)
            self._module_combo.setCurrentIndex(smallest_idx)
            self._module_combo.blockSignals(False)
            self._on_module_changed(smallest_idx)
        else:
            self._model.set_rows([], "Version A Verdict", "Version B Verdict")
            self._metrics_lbl.setText("")
            self._summary_lbl.setText(
                "No module has 2+ cached model versions yet -- run 'Check Duplicates' on the "
                "same module with a different LLM model to build a second version to compare."
            )

    def _on_module_changed(self, idx: int):
        if idx < 0:
            return
        module = self._module_combo.itemData(idx)
        versions = self._entries_by_module.get(module, [])
        for combo in (self._combo_a, self._combo_b):
            combo.blockSignals(True)
            combo.clear()
            for v in versions:
                model_label = v["llm_model"] or "(no LLM)"
                combo.addItem(
                    f"{model_label}  ({v['pair_count']:,} pairs, {trek_cache.format_age(v['updated_at'])})",
                    v["module_key"],
                )
            if combo is self._combo_b and len(versions) > 1:
                combo.setCurrentIndex(1)     # still blocked -> no extra load
            combo.blockSignals(False)
        self._reload_comparison()

    def _reload_comparison(self):
        """Start loading the selected A/B versions in the background
        (_CompareLoadWorker). The UI stays responsive; a newer selection
        simply supersedes an older in-flight load (its result is ignored)."""
        key_a = self._combo_a.currentData()
        key_b = self._combo_b.currentData()
        if not key_a or not key_b:
            return
        self._load_seq += 1
        seq = self._load_seq
        big = [int(x.replace(",", "")) for x in re.findall(r"\(([\d,]+) pairs",
               self._combo_a.currentText() + " " + self._combo_b.currentText())]
        n = max(big) if big else 0
        self._summary_lbl.setText(
            f"⏳ Loading comparison{f' ({n:,} pairs)' if n else ''}... "
            + ("large result -- this can take a minute; the window stays usable."
               if n > 20000 else "")
        )
        self._metrics_lbl.setText("")
        self._detail_lbl.setText("Loading...")
        self._syt_text_view.clear()
        self._counterpart_text_view.clear()
        self._model.set_rows([], "Version A Verdict", "Version B Verdict")

        worker = _CompareLoadWorker(self._cache, seq, key_a, key_b)
        worker.done.connect(self._on_comparison_loaded)
        worker.error.connect(self._on_comparison_error)
        _BG_COMPARE_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _BG_COMPARE_WORKERS.discard(w))
        worker.start()

    def _on_comparison_error(self, seq: int, msg: str):
        if seq != self._load_seq:
            return
        self._summary_lbl.setText(f"⚠️ Could not load comparison: {msg}")
        self._detail_lbl.setText("")

    def _on_comparison_loaded(self, seq: int, p: dict):
        if seq != self._load_seq:
            return   # user picked something else meanwhile
        self._result_a, self._result_b = p["res_a"], p["res_b"]
        self._m_a, self._m_b = p["m_a"], p["m_b"]
        # kept for _export_to_json -- exports every compared pair regardless of the on-screen filter
        self._by_pair_a, self._by_pair_b, self._all_keys = p["by_a"], p["by_b"], p["all_keys"]
        model_a = self._result_a.llm_model or "(no LLM)"
        model_b = self._result_b.llm_model or "(no LLM)"
        self._model_a, self._model_b = model_a, model_b

        self._model.set_rows(p["rows"], f"{model_a} Verdict", f"{model_b} Verdict")

        agree, disagree = p["agree"], p["disagree"]
        applicable = agree + disagree
        agree_pct = (agree / applicable * 100.0) if applicable else 0.0
        self._summary_lbl.setText(
            f"Comparing <b>{model_a}</b> vs <b>{model_b}</b>: "
            f"{len(p['all_keys']):,} unique pair(s)  |  ✅ {agree:,} agree ({agree_pct:.1f}%)  |  "
            f"⚠️ {disagree:,} disagree  |  only in A: {p['only_a']:,}  |  only in B: {p['only_b']:,}"
        )
        self._metrics_args = (model_a, model_b, agree, applicable, agree_pct, p["agree_by_verdict"])
        self._metrics_lbl.setText(self._build_metrics_html(*self._metrics_args))
        self._detail_lbl.setText("Select a pair to see both models' verdicts and reasoning.")
        for w in (self._filter_syt, self._filter_counterpart, self._filter_type,
                  self._filter_verdict_a, self._filter_verdict_b, self._filter_combo):
            w.blockSignals(True)
        self._filter_syt.clear()
        self._filter_counterpart.clear()
        self._filter_type.setCurrentIndex(0)
        self._filter_verdict_a.setCurrentIndex(0)
        self._filter_verdict_b.setCurrentIndex(0)
        self._filter_combo.setCurrentIndex(0)
        for w in (self._filter_syt, self._filter_counterpart, self._filter_type,
                  self._filter_verdict_a, self._filter_verdict_b, self._filter_combo):
            w.blockSignals(False)
        self._apply_filter()
        if self._proxy.rowCount():
            self._table.selectRow(0)

    def _apply_filter(self):
        """Combine every per-column filter (AND) -- text filters are
        case-insensitive substring matches; Type/Verdict A/Verdict B/
        Agreement are exact-match dropdowns. Evaluated by _CompareFilterProxy
        on the raw rows, so it stays fast for 85k pairs."""
        filter_map = {0: None, 1: "agree", 2: "disagree", 3: "only_a", 4: "only_b"}
        px = self._proxy
        px.syt = self._filter_syt.text().strip().lower()
        px.counterpart = self._filter_counterpart.text().strip().lower()
        px.ctype = self._filter_type.currentData()
        px.verdict_a = self._filter_verdict_a.currentData()
        px.verdict_b = self._filter_verdict_b.currentData()
        px.agreement = filter_map.get(self._filter_combo.currentIndex())
        px.invalidateFilter()

    def _export_to_json(self):
        if not self._all_keys:
            QMessageBox.information(self, "Nothing to Export", "No compared pairs to export -- pick a module first.")
            return
        model_a, model_b = self._model_a, self._model_b
        module = self._module_combo.currentData() or "module"
        m_a = self._m_a if self._m_a is not None else self._result_a.compute_metrics()
        m_b = self._m_b if self._m_b is not None else self._result_b.compute_metrics()

        def _model_summary(m: dict) -> dict:
            return {
                "llm_model": m.get("llm_model") or None,
                "judged_total": m["llm_judged_total"],
                "verdict_counts": m["llm_counts"],
                "verdict_percentages": m["llm_percentages"],
                "cost": {"last_run_usd": m["llm_cost_usd"], "current_list_value_usd": m["llm_cost_historical_usd"]},
                "llm_duration_seconds": m.get("llm_duration_seconds", 0.0),
            }

        agree = disagree = only_a = only_b = 0
        pairs_json = []
        for syt_id, counterpart_id in self._all_keys:
            pa = self._by_pair_a.get((syt_id, counterpart_id))
            pb = self._by_pair_b.get((syt_id, counterpart_id))
            ref_pair = pa or pb
            if pa is None:
                agreement = "only_in_b"
                only_b += 1
            elif pb is None:
                agreement = "only_in_a"
                only_a += 1
            elif pa.llm_verdict and pb.llm_verdict:
                if pa.llm_verdict == pb.llm_verdict:
                    agreement = "agree"
                    agree += 1
                else:
                    agreement = "disagree"
                    disagree += 1
            else:
                agreement = "not_applicable"
            pairs_json.append({
                "syt_id": syt_id,
                "counterpart_id": counterpart_id,
                "counterpart_type": ref_pair.counterpart_type,
                "agreement": agreement,
                "version_a": ({
                    "verdict": pa.llm_verdict, "reasoning": pa.llm_reasoning, "score": round(pa.score, 4),
                    "llm_cost_usd": round(pa.llm_cost_usd, 6), "checked_at": pa.checked_at or None,
                } if pa else None),
                "version_b": ({
                    "verdict": pb.llm_verdict, "reasoning": pb.llm_reasoning, "score": round(pb.score, 4),
                    "llm_cost_usd": round(pb.llm_cost_usd, 6), "checked_at": pb.checked_at or None,
                } if pb else None),
                "syt_text": ref_pair.syt_text,
                "counterpart_text": ref_pair.swt_text,
            })

        applicable = agree + disagree
        data = {
            "export_meta": {
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "module": module,
                "version_a_model": model_a,
                "version_b_model": model_b,
            },
            "summary": {
                "unique_pairs": len(self._all_keys),
                "agree": agree,
                "disagree": disagree,
                "only_in_a": only_a,
                "only_in_b": only_b,
                "agreement_percentage": round(agree / applicable * 100.0, 1) if applicable else 0.0,
            },
            "per_model_metrics": {model_a: _model_summary(m_a), model_b: _model_summary(m_b)},
            "pairs": pairs_json,
        }
        safe = str(module).replace(" ", "_").replace("/", "-")
        _export_dict_to_json(self, data, f"trek_compare_{safe}_{model_a}_vs_{model_b}.json")

    def _build_metrics_html(self, model_a: str, model_b: str, agree: int, applicable: int, agree_pct: float,
                             agree_by_verdict: Dict[str, int]) -> str:
        """Per-model verdict-percentage breakdown + cost, plus the overall
        "common feedback" (agreement) percentage across the two versions --
        reuses DuplicateCheckResult.compute_metrics() so the numbers match
        exactly what the single-version Metrics dialog would show.
        """
        m_a = self._m_a if self._m_a is not None else self._result_a.compute_metrics()
        m_b = self._m_b if self._m_b is not None else self._result_b.compute_metrics()

        def _side(label: str, m: dict) -> str:
            counts = m["llm_counts"]
            pct = m["llm_percentages"]
            total = m["llm_judged_total"]
            return (
                f"<b>{label}</b> — {total} pair(s) judged<br>"
                f"🔴 Same: {counts['same_scenario']} ({pct['same_scenario']:.1f}%) &nbsp; "
                f"🟡 Partial: {counts['partial_overlap']} ({pct['partial_overlap']:.1f}%) &nbsp; "
                f"🟢 Different: {counts['different_scenario']} ({pct['different_scenario']:.1f}%) &nbsp; "
                f"⚠️ Error: {counts['error']} ({pct['error']:.1f}%)<br>"
                f"💵 Cost — last run: ${m['llm_cost_usd']:.4f}, "
                f"historical total: ${m['llm_cost_historical_usd']:.4f}<br>"
                f"⏱ LLM-judge time: {DuplicateMetricsDialog._fmt_duration(m.get('llm_duration_seconds', 0.0))}"
            )

        same_n = agree_by_verdict.get("same_scenario", 0)
        partial_n = agree_by_verdict.get("partial_overlap", 0)
        different_n = agree_by_verdict.get("different_scenario", 0)
        pct = lambda n: (n / applicable * 100.0) if applicable else 0.0

        return (
            f'<table width="100%" cellpadding="4"><tr>'
            f'<td valign="top" width="50%">{_side(model_a, m_a)}</td>'
            f'<td valign="top" width="50%">{_side(model_b, m_b)}</td>'
            f"</tr></table>"
            f'<div style="margin-top:6px;color:{ACCENT};font-weight:bold">'
            f"🤝 Common feedback (same verdict on the same pair): {agree}/{applicable} "
            f"({agree_pct:.1f}%)</div>"
            f'<div style="margin-top:4px;color:{TEXT}">'
            f"&nbsp;&nbsp;🔴 Both said Same: {same_n} ({pct(same_n):.1f}%) &nbsp; "
            f"🟡 Both said Partial: {partial_n} ({pct(partial_n):.1f}%) &nbsp; "
            f"🟢 Both said Different: {different_n} ({pct(different_n):.1f}%)"
            f"</div>"
            f'<div style="margin-top:6px;color:{AMBER_COLOR};font-weight:bold">'
            f"{self._build_cost_comparison_html(model_a, model_b, m_a, m_b)}</div>"
            f'<div style="margin-top:4px;color:{AMBER_COLOR};font-weight:bold">'
            f"{self._build_time_comparison_html(model_a, model_b, m_a, m_b)}</div>"
        )

    @staticmethod
    def _build_cost_comparison_html(model_a: str, model_b: str, m_a: dict, m_b: dict) -> str:
        """Lead with whichever model is actually MORE expensive, then state
        how much pricier it is than the cheaper one, in percent -- reads
        clearer than always anchoring the comparison to "model B vs A"
        regardless of which one costs more. Uses historical cost (not
        just last-run) so a fully-cached run's $0 fresh spend doesn't hide
        the real cost."""
        cost_a = m_a["llm_cost_historical_usd"]
        cost_b = m_b["llm_cost_historical_usd"]
        if cost_a == 0 and cost_b == 0:
            return "💰 Cost comparison: both versions cost $0.0000 (nothing to compare)."
        if min(cost_a, cost_b) == 0:
            cheap_name, cheap_cost = (model_a, cost_a) if cost_a <= cost_b else (model_b, cost_b)
            pricey_name, pricey_cost = (model_b, cost_b) if cost_a <= cost_b else (model_a, cost_a)
            return (
                f"💰 Cost comparison: <b>{cheap_name}</b> cost $0.0000 while "
                f"<b>{pricey_name}</b> cost ${pricey_cost:.4f} -- percentage difference not meaningful (division by zero)."
            )

        if cost_a >= cost_b:
            pricey_name, pricey_cost, cheap_name, cheap_cost = model_a, cost_a, model_b, cost_b
        else:
            pricey_name, pricey_cost, cheap_name, cheap_cost = model_b, cost_b, model_a, cost_a
        diff_pct = (pricey_cost - cheap_cost) / cheap_cost * 100.0
        if diff_pct < 0.05:
            return f"💰 Cost comparison: <b>{model_a}</b> (${cost_a:.4f}) and <b>{model_b}</b> (${cost_b:.4f}) cost about the same."
        return (
            f"💰 Cost comparison: <b>{pricey_name}</b> (${pricey_cost:.4f}) is "
            f"<b>{diff_pct:.1f}% more expensive</b> than <b>{cheap_name}</b> (${cheap_cost:.4f})."
        )

    @staticmethod
    def _build_time_comparison_html(model_a: str, model_b: str, m_a: dict, m_b: dict) -> str:
        """Same "lead with the more extreme one" phrasing as the cost
        comparison above, but for LLM-judge wall-clock duration -- speed
        is a real factor when picking a model too (a cheaper model that's
        also much slower may not actually be the better trade-off)."""
        time_a = m_a.get("llm_duration_seconds", 0.0)
        time_b = m_b.get("llm_duration_seconds", 0.0)
        if time_a == 0 and time_b == 0:
            return "⏱ Time comparison: neither version's LLM-judge stage ran (nothing to compare)."
        if min(time_a, time_b) == 0:
            fast_name, fast_time = (model_a, time_a) if time_a <= time_b else (model_b, time_b)
            slow_name, slow_time = (model_b, time_b) if time_a <= time_b else (model_a, time_a)
            return (
                f"⏱ Time comparison: <b>{fast_name}</b> took "
                f"{DuplicateMetricsDialog._fmt_duration(fast_time)} (fully cached/not run) while "
                f"<b>{slow_name}</b> took {DuplicateMetricsDialog._fmt_duration(slow_time)}."
            )

        if time_a >= time_b:
            slow_name, slow_time, fast_name, fast_time = model_a, time_a, model_b, time_b
        else:
            slow_name, slow_time, fast_name, fast_time = model_b, time_b, model_a, time_a
        diff_pct = (slow_time - fast_time) / fast_time * 100.0
        if diff_pct < 0.05:
            return (
                f"⏱ Time comparison: <b>{model_a}</b> "
                f"({DuplicateMetricsDialog._fmt_duration(time_a)}) and <b>{model_b}</b> "
                f"({DuplicateMetricsDialog._fmt_duration(time_b)}) took about the same time."
            )
        return (
            f"⏱ Time comparison: <b>{slow_name}</b> "
            f"({DuplicateMetricsDialog._fmt_duration(slow_time)}) took "
            f"<b>{diff_pct:.1f}% longer</b> than <b>{fast_name}</b> "
            f"({DuplicateMetricsDialog._fmt_duration(fast_time)})."
        )

    def _on_row_selected(self, *_):
        sel = self._table.selectionModel().selectedRows()
        if not sel:
            return
        src = self._proxy.mapToSource(sel[0])
        if not src.isValid() or src.row() >= len(self._model.rows):
            return
        row_t = self._model.rows[src.row()]
        pa, pb = row_t[7], row_t[8]
        if pa is None and pb is None:
            return
        model_a = (self._result_a.llm_model if self._result_a else "") or "(no LLM)"
        model_b = (self._result_b.llm_model if self._result_b else "") or "(no LLM)"

        def _side(model_name: str, pair) -> str:
            if pair is None:
                return f"<b>{model_name}</b>: not present in this version."
            label, color = trek_similarity.LLM_VERDICT_LABELS.get(
                pair.llm_verdict, (pair.llm_verdict or "not judged", TEXT_DIM)
            )
            return (
                f'<b>{model_name}</b>: <span style="color:{color};font-weight:bold">{label}</span><br>'
                f'{pair.llm_reasoning or "(no reasoning)"}'
            )

        ref_pair = pa or pb
        self._detail_lbl.setText(
            f"<b>{ref_pair.syt_id}</b> vs <b>{ref_pair.swt_id}</b>"
            f"{'  (related SYT)' if ref_pair.counterpart_type == 'related_syt' else ''}<br><br>"
            f"{_side(model_a, pa)}<br><br>{_side(model_b, pb)}"
        )
        self._counterpart_lbl.setText(
            "Related SYT Content" if ref_pair.counterpart_type == "related_syt" else "SWT Content"
        )
        # Same labeled Pre-Condition/Procedure/Post-Condition rendering as
        # SideBySideResultsWidget, falling back to flat text for older
        # cached runs that predate syt_tc/swt_tc.
        if ref_pair.syt_tc:
            self._syt_text_view.setHtml(_tc_to_html(ref_pair.syt_tc, level_color=ACCENT))
        else:
            self._syt_text_view.setPlainText(ref_pair.syt_text or "(no content)")
        if ref_pair.swt_tc:
            self._counterpart_text_view.setHtml(_tc_to_html(ref_pair.swt_tc, level_color=SUCCESS_TEXT))
        else:
            self._counterpart_text_view.setPlainText(ref_pair.swt_text or "(no content)")


_OP_TYPE_LABELS = {
    "fetch_modules":       "📂 Fetch Modules (global module list)",
    "build_index":         "🧭 Build Index (id → module)",
    "fetch_testcases":     "📥 Get Test Cases",
    "fetch_traceability":  "🔗 Build Traceability",
    "rag_stage":           "🧮 Check Duplicates -- RAG Scoring",
    "llm_stage":           "🧠 Check Duplicates -- LLM Judge",
}


class AppStatsDialog(QDialog):
    """App-wide performance/cost report, broken down PER MODULE: for every
    SYT module that has recorded activity, shows every individual run of
    "Get Test Cases" (Step 2), "Build Traceability" (Step 3, with the
    SYR/SWR/SWT/related-SYT counts found), and "Check Duplicates" (RAG +
    LLM stages, with cost) -- full history, not just the latest run (see
    trek_cache's append-only operation_stats table, populated by
    FetchSytTcsWorker/FetchLinksWorker/DuplicateCheckWorker). An optional
    "Show averages" toggle adds one extra summary row per phase with the
    average time/cost across all its runs.
    """

    def __init__(self, cache: TrekCache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App Report -- Performance & Cost Statistics (per module)")
        self.resize(1100, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._cache = cache
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Full history of every run recorded so far (this and previous sessions), grouped by "
            "module then phase. Enable \"Show averages\" to add a summary row per phase."
        )
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(["Module / Phase / Run", "Items Found", "Time", "Cost", "Source / When"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            self._tree.header().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        lay.addWidget(self._tree, 1)

        btn_row = QHBoxLayout()
        self._chk_averages = QCheckBox("📐  Show averages")
        self._chk_averages.toggled.connect(self._reload)
        btn_row.addWidget(self._chk_averages)

        btn_expand = QPushButton("+  Expand All")
        btn_expand.setObjectName("btn_secondary")
        btn_expand.setToolTip("Expand every module/phase row.")
        btn_expand.clicked.connect(self._tree.expandAll)
        btn_row.addWidget(btn_expand)

        btn_collapse = QPushButton("−  Collapse All")
        btn_collapse.setObjectName("btn_secondary")
        btn_collapse.setToolTip("Collapse all rows to the top-level modules/phases.")
        btn_collapse.clicked.connect(self._tree.collapseAll)
        btn_row.addWidget(btn_collapse)

        btn_refresh = QPushButton("↻  Refresh")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._reload)
        btn_row.addWidget(btn_refresh)

        btn_clear = QPushButton("🗑  Clear Statistics")
        btn_clear.setObjectName("btn_secondary")
        btn_clear.clicked.connect(self._clear_stats)
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._reload()

    @staticmethod
    def _items_found(op_type: str, item_count: int, extra_count: int, details: dict) -> str:
        if op_type == "fetch_modules":
            return f"{item_count} module(s)"
        if op_type == "build_index":
            d = details or {}
            return (f"{item_count} ids  |  {d.get('syr', 0)} SYR, {d.get('swr', 0)} SWR, "
                    f"{d.get('syt', 0)} SYT, {d.get('swt', 0)} SWT")
        if op_type == "fetch_testcases":
            return f"{item_count} TC(s)"
        if op_type == "fetch_traceability":
            d = details or {}
            return (f"{item_count} TC  |  {d.get('syr_count', 0)} SYR, {d.get('swr_count', 0)} SWR, "
                    f"{d.get('swt_count', 0)} SWT, {d.get('related_syt_count', 0)} related SYT")
        if op_type == "rag_stage":
            return f"{item_count} new text(s) embedded  /  {extra_count} pair(s) compared"
        if op_type == "llm_stage":
            return f"{item_count} new pair(s) judged  /  {extra_count} pair(s) total"
        return f"{item_count} item(s)"

    @classmethod
    def _run_row(cls, op_type: str, row: dict) -> List[str]:
        model = (row["details"] or {}).get("model")
        prefix = "  ·  ".join(p for p in (row["source"], model) if p)
        return [
            "",
            cls._items_found(op_type, row["item_count"], row["extra_count"], row["details"]),
            DuplicateMetricsDialog._fmt_duration(row["duration_seconds"]),
            f"${row['cost_usd']:.4f}" if row["cost_usd"] else "--",
            f"{prefix + '  ·  ' if prefix else ''}{row['created_at']}",
        ]

    @classmethod
    def _average_row(cls, op_type: str, runs: List[dict]) -> QTreeWidgetItem:
        """Compute a TRUE per-field average across every run in ``runs`` --
        including the SYR/SWR/SWT/related-SYT breakdown for
        fetch_traceability -- rather than reusing the latest run's raw
        counts, so this row genuinely represents "average", not "last"."""
        n = len(runs) or 1
        avg_item_count = round(sum(r["item_count"] for r in runs) / n)
        avg_extra_count = round(sum(r["extra_count"] for r in runs) / n)
        total_seconds = sum(r["duration_seconds"] for r in runs)
        total_cost = sum(r["cost_usd"] for r in runs)
        avg_seconds = total_seconds / n
        avg_cost = total_cost / n

        if op_type == "fetch_traceability":
            avg_syr = round(sum((r["details"] or {}).get("syr_count", 0) for r in runs) / n)
            avg_swr = round(sum((r["details"] or {}).get("swr_count", 0) for r in runs) / n)
            avg_swt = round(sum((r["details"] or {}).get("swt_count", 0) for r in runs) / n)
            avg_rel = round(sum((r["details"] or {}).get("related_syt_count", 0) for r in runs) / n)
            items_found = (f"{avg_item_count} TC  |  {avg_syr} SYR, {avg_swr} SWR, "
                           f"{avg_swt} SWT, {avg_rel} related SYT")
        else:
            items_found = cls._items_found(op_type, avg_item_count, avg_extra_count, {})

        # "Average" here means the same thing for every column: the mean
        # across the ``n`` runs -- e.g. avg cost = mean(cost per run), NOT
        # a weighted total-cost/total-items ratio, so it lines up with the
        # avg pair/TC count and avg time shown alongside it. Cost-per-item
        # is normalized by the NEW-work item_count (not the total pair
        # count in extra_count), so it stays comparable across runs with
        # different cache-hit rates.
        avg_cost_per_item = avg_cost / avg_item_count if avg_item_count else 0.0
        cost_text = (
            f"${avg_cost:.4f} avg  (${avg_cost_per_item:.4f}/new item, total ${total_cost:.4f})"
            if total_cost else "--"
        )

        item = QTreeWidgetItem([
            "Σ  Average",
            items_found,
            DuplicateMetricsDialog._fmt_duration(avg_seconds),
            cost_text,
            f"across {len(runs)} run(s)",
        ])
        for col in range(5):
            font = item.font(col)
            font.setItalic(True)
            item.setFont(col, font)
            item.setForeground(col, QColor(ACCENT))
        return item

    def _add_token_usage_section(self, all_rows: List[dict]):
        """Top-level summary: total tokens (and cost) used PER MODEL across
        every RAG/LLM run ever recorded, plus a grand total across all
        models -- lets you compare different LLM/embedding models' token
        usage directly."""
        per_model: Dict[str, Dict[str, Any]] = {}
        for row in all_rows:
            if row["op_type"] not in ("rag_stage", "llm_stage"):
                continue
            model = (row["details"] or {}).get("model") or "(unknown model)"
            bucket = per_model.setdefault(model, {"tokens": 0, "cost": 0.0, "runs": 0})
            bucket["tokens"] += (row["details"] or {}).get("tokens", 0)
            bucket["cost"] += row["cost_usd"]
            bucket["runs"] += 1
        if not per_model:
            return

        header = QTreeWidgetItem(["🤖  Token Usage by Model", "", "", "", ""])
        font = header.font(0)
        font.setBold(True)
        header.setFont(0, font)
        self._tree.addTopLevelItem(header)

        for model in sorted(per_model.keys()):
            b = per_model[model]
            header.addChild(QTreeWidgetItem([
                model, f"{b['tokens']:,} tokens", "",
                f"${b['cost']:.4f}" if b["cost"] else "--", f"{b['runs']} run(s)",
            ]))

        grand_tokens = sum(b["tokens"] for b in per_model.values())
        grand_cost = sum(b["cost"] for b in per_model.values())
        grand_runs = sum(b["runs"] for b in per_model.values())
        total_item = QTreeWidgetItem([
            "Σ  Grand total (all models)", f"{grand_tokens:,} tokens", "",
            f"${grand_cost:.4f}" if grand_cost else "--", f"{grand_runs} run(s)",
        ])
        for col in range(5):
            f = total_item.font(col)
            f.setItalic(True)
            total_item.setFont(col, f)
            total_item.setForeground(col, QColor(ACCENT))
        header.addChild(total_item)
        header.setExpanded(True)

    def _reload(self):
        self._tree.clear()
        show_avg = self._chk_averages.isChecked()
        all_rows = self._cache.get_operation_stats()
        op_order = ["fetch_testcases", "fetch_traceability", "rag_stage", "llm_stage"]

        self._add_token_usage_section(all_rows)

        grouped: Dict[Optional[str], Dict[str, List[dict]]] = {}
        for row in all_rows:
            grouped.setdefault(row["module"], {}).setdefault(row["op_type"], []).append(row)

        def _add_phase(parent, module_key, op_type, runs):
            phase_item = QTreeWidgetItem([
                f"{_OP_TYPE_LABELS.get(op_type, op_type)}  ({len(runs)} run(s))", "", "", "", "",
            ])
            if parent is None:
                self._tree.addTopLevelItem(phase_item)
            else:
                parent.addChild(phase_item)
            for row in runs:
                phase_item.addChild(QTreeWidgetItem(self._run_row(op_type, row)))
            if show_avg:
                phase_item.addChild(self._average_row(op_type, runs))
            phase_item.setExpanded(True)

        # Global (module-independent) phases first, e.g. fetch_modules.
        for op_type, runs in grouped.pop(None, {}).items():
            _add_phase(None, None, op_type, runs)

        for module in sorted(grouped.keys()):
            module_ops = grouped[module]
            mod_item = QTreeWidgetItem([module, "", "", "", ""])
            font = mod_item.font(0)
            font.setBold(True)
            mod_item.setFont(0, font)
            self._tree.addTopLevelItem(mod_item)
            for op_type in op_order:
                if op_type in module_ops:
                    _add_phase(mod_item, module, op_type, module_ops[op_type])
            mod_item.setExpanded(True)

        if self._tree.topLevelItemCount() == 0:
            placeholder = QTreeWidgetItem([
                "No operations recorded yet -- load modules, get test cases, "
                "build traceability, or run 'Check Duplicates' at least once.",
                "", "", "", "",
            ])
            self._tree.addTopLevelItem(placeholder)

    def _clear_stats(self):
        reply = QMessageBox.question(
            self, "Clear Statistics",
            "Delete all recorded performance/cost statistics? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._cache.clear_operation_stats()
            self._reload()


class TestAlgorithmWorker(QThread):
    """Background worker for TestAlgorithmDialog: runs the exact same
    trek_similarity.compare_syt_swt_pairs() pipeline used by real
    'Check Duplicates' runs, but against user-entered synthetic SYT/SWT
    text instead of real TREK data. Uses the SAME embedding cache
    (content-addressed, so repeated experiments with the same text don't
    re-embed) and the active project's JWT Token."""
    done  = Signal(object)   # trek_similarity.DuplicateCheckResult
    error = Signal(str)

    def __init__(self, pairs: list, settings: dict, jwt_token: str):
        super().__init__()
        self.pairs = pairs   # list of (syt_id, syt_text, swt_id, swt_text, syt_methods, swt_methods, counterpart_type)
        self.settings = settings
        self.jwt_token = jwt_token

    def run(self):
        try:
            client = trek_similarity.get_embeddings_client(jwt_token=self.jwt_token)

            all_texts = set()
            for _syt_id, syt_text, _swt_id, swt_text, _sm, _wm, _ct, *_tc_dicts in self.pairs:
                if syt_text:
                    all_texts.add(syt_text)
                if swt_text:
                    all_texts.add(swt_text)
            norm_to_raw = {re.sub(r"\s+", " ", t.strip().lower()).rstrip("."): t for t in all_texts}
            cached = CACHE.get_embeddings(list(norm_to_raw.keys()), trek_similarity.EMBEDDING_MODEL)
            embedding_cache = dict(cached)

            result = trek_similarity.compare_syt_swt_pairs(
                client, self.pairs, embedding_cache,
                duplicate_threshold=self.settings["sim_duplicate"],
                near_duplicate_threshold=self.settings["sim_near_duplicate"],
                similar_threshold=self.settings["sim_similar"],
                bm25_weight=self.settings["bm25_weight"],
                vec_weight=self.settings["vec_weight"],
                seq_weight=self.settings["seq_weight"],
            )

            new_entries = {k: v for k, v in embedding_cache.items() if k not in cached}
            if new_entries:
                CACHE.set_embeddings(new_entries, trek_similarity.EMBEDDING_MODEL)

            self.done.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TestAlgorithmDialog(QDialog):
    """Manual sandbox for experimenting with the duplicate-detection
    algorithm using imported synthetic/real data, independent of any real
    TREK traceability run. The user imports a JSON file describing one SYT
    test case and one-or-more SWT test cases; imported test cases appear
    in a clickable list with a side-by-side content preview (no manual
    typing/editing forms -- import-only, per user preference), then
    "Score Pairs" runs the same hybrid BM25/vector/sequence pipeline used
    by real 'Check Duplicates' runs (via the shared ScoringSettingsWidget)
    and shows results in the same SideBySideResultsWidget used by the
    production dialog.

    JSON import format (see _import_json()):
        {
            "syt": {"name": "...", "preconditions": "...", "procedure": "...", "postconditions": "..."},
            "swts": [
                {"name": "...", "preconditions": "...", "procedure": "...", "postconditions": "..."},
                ...
            ]
        }
    Field names are matched case-insensitively and tolerate the singular
    forms too (precondition/postcondition), so hand-written JSON doesn't
    need to match an exact casing/pluralization convention.
    """

    def __init__(self, parent=None, jwt_token: str = ""):
        super().__init__(parent)
        self.setWindowTitle("🧪 Test Duplicate-Detection Algorithm")
        self.resize(1300, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._jwt_token = jwt_token
        self._syt_tc: Optional[dict] = None    # {"Name","PreCondition","Procedure","Postcondition"}
        self._swt_tcs: List[dict] = []         # same shape, one per imported SWT
        self._build_ui()

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_maximize.setText("⛶  Maximize")
        else:
            self.showMaximized()
            self._btn_maximize.setText("🗗  Restore")

    def _build_ui(self):
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Import a JSON file with one SYT and one-or-more SWT test "
            "cases to see exactly how the duplicate-detection algorithm "
            "scores them -- useful for verifying the algorithm behaves as "
            "expected (e.g. a hand-crafted copy-paste pair should score as "
            "'Duplicate') before trusting it on real TREK data."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-bottom:6px;")
        lay.addWidget(intro)

        import_row = QHBoxLayout()
        btn_import = QPushButton("📂  Import JSON")
        btn_import.setObjectName("btn_secondary")
        btn_import.setToolTip(
            'Load {"syt": {...}, "swts": [{...}, ...]} with '
            "name/preconditions/procedure/postconditions fields."
        )
        btn_import.clicked.connect(self._import_json)
        import_row.addWidget(btn_import)

        self._loaded_summary_lbl = QLabel("No test cases imported yet.")
        self._loaded_summary_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-left:10px;")
        import_row.addWidget(self._loaded_summary_lbl)
        import_row.addStretch()
        lay.addLayout(import_row)

        # Tabs: "Preview" (imported test case list + side-by-side content,
        # no scores yet) and "Results" (SideBySideResultsWidget -- the SAME
        # results view used by the real 'Check Duplicates' feature).
        self._tabs = QTabWidget()
        lay.addWidget(self._tabs, 1)

        # --- Preview tab ---------------------------------------------------
        preview_tab = QWidget()
        preview_tab_lay = QHBoxLayout(preview_tab)

        # Left: list of imported test cases (SYT + each SWT), click to preview
        list_frame = QGroupBox("Imported Test Cases (click to preview)")
        list_lay = QVBoxLayout(list_frame)
        self._tc_list = QListWidget()
        self._tc_list.itemSelectionChanged.connect(self._on_preview_selection_changed)
        list_lay.addWidget(self._tc_list)
        preview_tab_lay.addWidget(list_frame, 1)

        # Middle: side-by-side content preview (SYT always shown left; the
        # selected SWT shown right -- same visual pattern as the Results
        # tab's side-by-side comparison, just without scores yet).
        preview_frame = QGroupBox("Side-by-Side Preview")
        preview_lay = QVBoxLayout(preview_frame)
        side_by_side = QHBoxLayout()

        syt_box = QVBoxLayout()
        syt_lbl = QLabel("SYT Content")
        syt_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        syt_box.addWidget(syt_lbl)
        self._syt_preview = QTextEdit()
        self._syt_preview.setReadOnly(True)
        syt_box.addWidget(self._syt_preview)

        swt_box = QVBoxLayout()
        swt_lbl = QLabel("SWT Content")
        swt_lbl.setStyleSheet(f"color:{SUCCESS_TEXT};font-weight:bold;font-size:11px;text-transform:uppercase;")
        swt_box.addWidget(swt_lbl)
        self._swt_preview = QTextEdit()
        self._swt_preview.setReadOnly(True)
        swt_box.addWidget(self._swt_preview)

        side_by_side.addLayout(syt_box)
        side_by_side.addLayout(swt_box)
        preview_lay.addLayout(side_by_side, 1)
        preview_tab_lay.addWidget(preview_frame, 2)

        # Right: scoring settings + run button
        right_frame = QWidget()
        right_lay = QVBoxLayout(right_frame)

        self._settings_widget = ScoringSettingsWidget(self)
        right_lay.addWidget(self._settings_widget)

        self._btn_run = QPushButton("▶  Score Pairs")
        self._btn_run.setObjectName("btn_success")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run_test)
        right_lay.addWidget(self._btn_run)
        right_lay.addStretch()

        preview_tab_lay.addWidget(right_frame, 1)
        self._tabs.addTab(preview_tab, "Preview")

        # --- Results tab -------------------------------------------------
        self._results_widget = SideBySideResultsWidget(self)
        self._tabs.addTab(self._results_widget, "Results")

        btn_row = QHBoxLayout()
        self._btn_maximize = QPushButton("⛶  Maximize")
        self._btn_maximize.setObjectName("btn_secondary")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        btn_row.addWidget(self._btn_maximize)
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    @staticmethod
    def _get_field_ci(entry: dict, *names: str) -> str:
        """Look up a value in a JSON object by field name, tolerant of
        case and the singular/plural variants users might reasonably type
        (e.g. 'precondition' vs 'preconditions')."""
        lowered = {k.lower(): v for k, v in entry.items()} if isinstance(entry, dict) else {}
        for name in names:
            if name.lower() in lowered:
                val = lowered[name.lower()]
                return "" if val is None else str(val)
        return ""

    def _entry_to_tc_dict(self, entry: dict) -> dict:
        return {
            "Name": self._get_field_ci(entry, "name"),
            "PreCondition": self._get_field_ci(entry, "preconditions", "precondition"),
            "Procedure": self._get_field_ci(entry, "procedure", "procedures"),
            "Postcondition": self._get_field_ci(entry, "postconditions", "postcondition"),
        }

    @staticmethod
    def _tc_preview_text(tc: dict) -> str:
        parts = []
        if tc.get("Name"):
            parts.append(f"Name: {tc['Name']}")
        if tc.get("PreCondition"):
            parts.append(f"\nPreCondition:\n{tc['PreCondition']}")
        if tc.get("Procedure"):
            parts.append(f"\nProcedure:\n{tc['Procedure']}")
        if tc.get("Postcondition"):
            parts.append(f"\nPostcondition:\n{tc['Postcondition']}")
        return "\n".join(parts) if parts else "(empty)"

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Test Case JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Import Failed", f"Could not read/parse JSON file:\n{e}")
            return

        syt_entry = data.get("syt") if isinstance(data, dict) else None
        swts_entry = data.get("swts") if isinstance(data, dict) else None
        if not isinstance(syt_entry, dict) or not isinstance(swts_entry, list) or not swts_entry:
            QMessageBox.critical(
                self, "Invalid JSON Format",
                'Expected {"syt": {...}, "swts": [{...}, ...]} with each '
                "object containing name/preconditions/procedure/"
                "postconditions fields."
            )
            return

        self._syt_tc = self._entry_to_tc_dict(syt_entry)
        self._swt_tcs = [
            self._entry_to_tc_dict(swt_entry)
            for swt_entry in swts_entry if isinstance(swt_entry, dict)
        ]

        self._populate_tc_list()
        self._btn_run.setEnabled(bool(self._swt_tcs))
        self._loaded_summary_lbl.setText(
            f"Loaded 1 SYT + {len(self._swt_tcs)} SWT test case(s) from '{Path(path).name}'."
        )
        self._tabs.setCurrentIndex(0)

    def _populate_tc_list(self):
        self._tc_list.clear()

        syt_item = QListWidgetItem(f"🔷 SYT: {self._syt_tc.get('Name') or '(no name)'}")
        syt_item.setData(Qt.UserRole, {"type": "syt", "tc": self._syt_tc})
        self._tc_list.addItem(syt_item)

        for i, swt_tc in enumerate(self._swt_tcs, 1):
            swt_item = QListWidgetItem(f"🔶 SWT #{i}: {swt_tc.get('Name') or '(no name)'}")
            swt_item.setData(Qt.UserRole, {"type": "swt", "tc": swt_tc, "index": i})
            self._tc_list.addItem(swt_item)

        self._syt_preview.setPlainText(self._tc_preview_text(self._syt_tc))
        self._swt_preview.clear()
        if self._tc_list.count():
            self._tc_list.setCurrentRow(0)

    def _on_preview_selection_changed(self):
        items = self._tc_list.selectedItems()
        if not items:
            return
        data = items[0].data(Qt.UserRole)
        if not data:
            return
        # SYT content is always shown on the left regardless of which row
        # is selected; only the right (SWT) pane changes -- selecting the
        # SYT row itself just clears the right pane.
        if data["type"] == "swt":
            self._swt_preview.setPlainText(self._tc_preview_text(data["tc"]))
        else:
            self._swt_preview.clear()

    def _run_test(self):
        if not self._settings_widget.validate(self):
            return
        if not self._jwt_token:
            QMessageBox.warning(
                self, "Missing JWT Token",
                "The active project has no JWT Token configured. Set one via "
                "'Edit Project' (header ✎ button) before testing the algorithm."
            )
            return
        if not self._syt_tc or not self._swt_tcs:
            QMessageBox.warning(self, "No Data Imported", "Import a JSON file with a SYT and at least one SWT first.")
            return

        syt_text = trek_similarity.build_comparison_text(self._syt_tc, include_postcondition=True)
        if not syt_text:
            QMessageBox.warning(self, "Empty SYT", "The imported SYT test case has no usable content.")
            return
        syt_methods = trek_similarity.extract_method_call_sequence(self._syt_tc)

        pairs = []
        for i, swt_tc in enumerate(self._swt_tcs, 1):
            swt_text = trek_similarity.build_comparison_text(swt_tc, include_postcondition=True)
            if not swt_text:
                continue
            swt_methods = trek_similarity.extract_method_call_sequence(swt_tc)
            swt_id = f"SWT_#{i}"
            pairs.append(("SYT_TEST", syt_text, swt_id, swt_text, syt_methods, swt_methods, "SWT", self._syt_tc, swt_tc))

        if not pairs:
            QMessageBox.warning(self, "No SWT Content", "None of the imported SWT test cases have usable content.")
            return

        settings = self._settings_widget.get_settings()
        self._btn_run.setEnabled(False)
        self._btn_run.setText("Scoring...")

        worker = TestAlgorithmWorker(pairs, settings, self._jwt_token)
        worker.done.connect(self._on_test_ready)
        worker.error.connect(self._on_test_error)
        worker.finished.connect(lambda: self._cleanup_test_worker(worker))
        self._worker = worker
        worker.start()

    def _cleanup_test_worker(self, worker):
        self._btn_run.setEnabled(True)
        self._btn_run.setText("▶  Score Pairs")

    def _on_test_error(self, msg: str):
        QMessageBox.critical(self, "Scoring Failed", msg)

    def _on_test_ready(self, result: "trek_similarity.DuplicateCheckResult"):
        self._results_widget.set_result(result)
        self._tabs.setCurrentWidget(self._results_widget)
        self._tabs.setCurrentWidget(self._results_widget)

        if result.skipped:
            QMessageBox.information(
                self, "Some Pairs Skipped",
                f"{len(result.skipped)} item(s) had no usable text and were skipped."
            )


class ModuleMappingDialog(QDialog):
    """Manual review/correction UI for a SYT module's SYR and SWT/SWIT
    bridge module mapping, for the cases automatic detection cannot
    reliably solve at all -- e.g. a domain-specific ID abbreviation like
    "RWW" (German "Wischen und Waschen" / wipe-and-wash) for
    "SYT - Rear Wiper", which has no textual relationship whatsoever to
    the module's display name and so never gets shortlisted by either the
    substring or acronym matching in _detect_bridge_modules().

    Shows two checkable lists (all real "SYR -" modules, all real
    "SWT -"/"SWIT -" modules), pre-checked with whatever auto-detection
    already found (possibly nothing, as in the RWW case above) so the
    user only needs to correct what's wrong rather than start from
    scratch. Saving persists the confirmed selection via
    set_manual_bridge_mapping() -- a manual mapping always takes priority
    over auto-detection afterward (see _get_bridge_modules_cached()) and
    is never silently overwritten by 'Force Refresh'; only re-opening this
    dialog and saving again changes it.
    """

    def __init__(self, syt_module: str, all_syr_names: List[str], all_swt_names: List[str],
                 auto_syr: List[str], auto_swt: List[str], parent=None,
                 all_swr_names: Optional[List[str]] = None,
                 auto_swr: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit SYR/SWR/SWT Mapping -- {syt_module}")
        self.resize(860, 560)
        self.syt_module = syt_module
        self._build_ui(all_syr_names, all_swt_names, auto_syr, auto_swt,
                       all_swr_names or [], auto_swr or [])

    def _build_ui(self, all_syr_names, all_swt_names, auto_syr, auto_swt,
                  all_swr_names, auto_swr):
        lay = QVBoxLayout(self)

        intro = QLabel(
            f"Confirm which SYR requirement module, SWR requirement module(s) "
            f"and SWT/SWIT test module(s) '{self.syt_module}' actually bridges "
            f"to. Auto-detected candidates are pre-checked; correct them if "
            f"wrong or empty. Setting the SWR module(s) manually is especially "
            f"valuable for speed: it lets 'Build Traceability' SKIP probing "
            f"dozens of candidate SWR modules live (which can include huge "
            f"unrelated modules that take minutes to download). "
            f"Your selection is saved permanently and always takes priority "
            f"over auto-detection afterward."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-bottom:6px;")
        lay.addWidget(intro)

        splitter = QSplitter(Qt.Horizontal)

        syr_box = QGroupBox(f"SYR Module{'s' if len(auto_syr) != 1 else ''} "
                             f"({'auto-detected: ' + ', '.join(auto_syr) if auto_syr else 'none auto-detected'})")
        syr_lay = QVBoxLayout(syr_box)
        self._syr_search = QLineEdit()
        self._syr_search.setPlaceholderText("Filter SYR modules...")
        self._syr_search.textChanged.connect(lambda t: self._filter_list(self._syr_list, t))
        syr_lay.addWidget(self._syr_search)
        self._syr_list = QListWidget()
        for name in sorted(all_syr_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in auto_syr else Qt.Unchecked)
            self._syr_list.addItem(item)
        syr_lay.addWidget(self._syr_list, 1)
        splitter.addWidget(syr_box)

        # SWR bridge module(s). Manually setting this is the key speed lever:
        # see _get_bridge_modules_cached() -- a manual mapping short-circuits
        # the live candidate probing entirely.
        swr_box = QGroupBox(f"SWR Module(s) "
                             f"({'auto-detected: ' + ', '.join(auto_swr) if auto_swr else 'none auto-detected'})")
        swr_lay = QVBoxLayout(swr_box)
        self._swr_search = QLineEdit()
        self._swr_search.setPlaceholderText("Filter SWR modules...")
        self._swr_search.textChanged.connect(lambda t: self._filter_list(self._swr_list, t))
        swr_lay.addWidget(self._swr_search)
        self._swr_list = QListWidget()
        for name in sorted(all_swr_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in auto_swr else Qt.Unchecked)
            self._swr_list.addItem(item)
        swr_lay.addWidget(self._swr_list, 1)
        splitter.addWidget(swr_box)

        swt_box = QGroupBox(f"SWT/SWIT Module(s) "
                             f"({'auto-detected: ' + ', '.join(auto_swt) if auto_swt else 'none auto-detected'})")
        swt_lay = QVBoxLayout(swt_box)
        self._swt_search = QLineEdit()
        self._swt_search.setPlaceholderText("Filter SWT/SWIT modules...")
        self._swt_search.textChanged.connect(lambda t: self._filter_list(self._swt_list, t))
        swt_lay.addWidget(self._swt_search)
        self._swt_list = QListWidget()
        for name in sorted(all_swt_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in auto_swt else Qt.Unchecked)
            self._swt_list.addItem(item)
        swt_lay.addWidget(self._swt_list, 1)
        splitter.addWidget(swt_box)

        lay.addWidget(splitter, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_save = QPushButton("Save Mapping")
        btn_save.setObjectName("btn_success")
        btn_save.clicked.connect(self.accept)
        btn_row.addWidget(btn_save)
        lay.addLayout(btn_row)

    @staticmethod
    def _filter_list(list_widget: QListWidget, text: str):
        text = text.lower().strip()
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    @staticmethod
    def _checked_items(list_widget: QListWidget) -> List[str]:
        return [
            list_widget.item(i).text()
            for i in range(list_widget.count())
            if list_widget.item(i).checkState() == Qt.Checked
        ]

    def get_selected_syr(self) -> List[str]:
        return self._checked_items(self._syr_list)

    def get_selected_swr(self) -> List[str]:
        return self._checked_items(self._swr_list)

    def get_selected_swt(self) -> List[str]:
        return self._checked_items(self._swt_list)


def _format_bytes(n: int) -> str:
    """Human-readable byte count, e.g. 3221225472 -> '3.00 GB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.2f} TB"


class TrekDatabaseDialog(QDialog):
    """Read-only viewer for the local trek_cache.sqlite3 contents.

    Shows an aggregate summary (per blob "kind" -- modules/links -- with
    key counts, item counts, and freshness) plus a per-key breakdown table
    for the blob cache so the user can see exactly which SYT/SYR/SWR/SWT
    modules currently have cached data and how old it is.

    Also surfaces the on-disk file size and an "Optimize Storage" action
    (see TrekCache.optimize_storage()) -- for caches that predate binary
    embedding storage, embeddings are the single biggest contributor to
    file size (JSON-text vectors are ~4-5x larger than the packed float32
    format used for new writes), which matters especially for a cache
    living on a network share, where every extra byte costs a round trip.
    """

    def __init__(self, cache: TrekCache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TREK Local Database")
        self.resize(760, 620)
        self._cache = cache
        self._build_ui()
        self._reload()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        self._path_lbl = QLabel()
        self._path_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self._path_lbl.setWordWrap(True)
        lay.addWidget(self._path_lbl)

        self._summary_lbl = QLabel()
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        self._summary_lbl.setWordWrap(True)
        lay.addWidget(self._summary_lbl)

        self._optimize_hint_lbl = QLabel()
        self._optimize_hint_lbl.setStyleSheet(f"color:{AMBER_COLOR};font-size:11px;padding:2px 0;")
        self._optimize_hint_lbl.setWordWrap(True)
        lay.addWidget(self._optimize_hint_lbl)

        # Per-entry table: key / kind / items / age
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Cache Key", "Kind", "Items", "Last Updated"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("⟳  Refresh View")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._reload)
        btn_row.addWidget(btn_refresh)

        self._btn_optimize = QPushButton("🗜  Optimize Storage...")
        self._btn_optimize.setObjectName("btn_secondary")
        self._btn_optimize.setToolTip(
            "Convert older embeddings to compact binary storage and reclaim "
            "disk space (VACUUM). Recommended for large caches, especially "
            "on a network share. Can take a while on multi-GB databases."
        )
        self._btn_optimize.clicked.connect(self._on_optimize_storage)
        btn_row.addWidget(self._btn_optimize)
        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _reload(self):
        summary = self._cache.summary()
        size_str = _format_bytes(summary.get("file_size_bytes", 0))
        self._path_lbl.setText(f"Database file: {summary['db_path']}  ·  {size_str} on disk")

        tc_count = summary["tc_content_count"]
        tc_age = (trek_cache.format_age(summary["tc_content_latest"])
                  if summary["tc_content_latest"] else "—")
        overall_age = (trek_cache.format_age(summary["latest_overall"])
                       if summary["latest_overall"] else "—")

        kind_parts = "  |  ".join(
            f"{b['kind']}: {b['key_count']} keys / {b['item_count']} items ({trek_cache.format_age(b['latest'])})"
            for b in summary["blob_kinds"]
        ) or "(no cached link/module data yet)"

        self._summary_lbl.setText(
            f"🧾 TC content cached: {tc_count} test cases (latest {tc_age})\n"
            f"📦 {kind_parts}\n"
            f"🧮 Embeddings cached: {summary.get('embeddings_count', 0):,}\n"
            f"🕒 Most recent write overall: {overall_age}"
        )

        unoptimized = summary.get("unoptimized_embeddings", 0)
        if unoptimized:
            self._optimize_hint_lbl.setText(
                f"⚠ {unoptimized:,} embedding(s) are still stored in the older, larger "
                "JSON format. Click 'Optimize Storage' below to shrink the database "
                "and speed up reads (especially important on a network-share cache)."
            )
            self._optimize_hint_lbl.show()
        else:
            self._optimize_hint_lbl.hide()

        entries = self._cache.list_blob_entries()
        self._table.setRowCount(0)
        for e in entries:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(e["key"]))
            self._table.setItem(row, 1, QTableWidgetItem(e["kind"]))
            self._table.setItem(row, 2, QTableWidgetItem(str(e["item_count"])))
            self._table.setItem(row, 3, QTableWidgetItem(
                f"{trek_cache.format_age(e['updated_at'])}  ({e['updated_at']})"
            ))

    def _on_optimize_storage(self):
        summary = self._cache.summary()
        unoptimized = summary.get("unoptimized_embeddings", 0)
        reply = QMessageBox.question(
            self, "Optimize Storage",
            (f"This will convert {unoptimized:,} older embedding row(s) to compact "
             "binary storage, " if unoptimized else "This will ") +
            "compress large cached entries, switch the file to 64 KB pages and "
            "VACUUM it to reclaim free space.\n\n"
            "Fewer, bigger pages = far fewer network round trips: on a network "
            "share, startup and opening results become many times faster.\n\n"
            "It rewrites the whole file: best done while the database is on the "
            "LOCAL disk (then copy it to the share). The app is unresponsive until "
            "it finishes -- a few minutes for a multi-GB file. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._btn_optimize.setEnabled(False)
        self._btn_optimize.setText("Optimizing... please wait")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            result = self._cache.optimize_storage()
        except Exception as exc:
            QMessageBox.critical(self, "Optimize Storage Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._btn_optimize.setEnabled(True)
            self._btn_optimize.setText("🗜  Optimize Storage...")

        before = _format_bytes(result["size_before_bytes"])
        after = _format_bytes(result["size_after_bytes"])
        QMessageBox.information(
            self, "Optimize Storage Complete",
            f"Converted {result['converted_rows']:,} embedding row(s), compressed "
            f"{result.get('blobs_compressed', 0):,} large entr(y/ies) and "
            f"{result.get('results_compressed', 0):,} old result(s).\n"
            f"Page size: {result.get('page_size_before', '?')} → {result.get('page_size_after', '?')} bytes\n\n"
            f"Database size: {before} → {after}",
        )
        self._reload()


class TrekLogDialog(QDialog):
    """Activity/timing log viewer (trek_log.py -> LOG). Shows every TREK
    API call, cache hit/miss, and worker stage recorded during this
    session (plus prior sessions, since trek_activity.log is append-only
    on disk -- though this viewer only shows what's currently held in
    memory via LOG.get_entries(); very old entries beyond the in-memory
    cap are only in the on-disk file).

    Columns: Time / Level / Category / Duration / Message / Details.
    ERROR rows are highlighted red, WARN rows yellow, so failures/slow
    spots are visible at a glance without reading every row. A live
    keyword filter narrows by category/message/details text. "⟳ Refresh"
    reloads from LOG (useful while a background worker is still running
    and logging more entries); "🗑 Clear View" empties the in-memory log
    (does NOT touch the on-disk trek_activity.log unless the user also
    checks "also delete on-disk file"); "💾 Export..." saves the currently
    filtered view as a plain-text file.
    """

    def __init__(self, log: "trek_log.TrekLog", parent=None):
        super().__init__(parent)
        self.setWindowTitle("TREK Activity Log")
        self.resize(1100, 620)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._log = log
        self._all_entries: List["trek_log.LogEntry"] = []
        self._build_ui()
        self._reload()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        self._path_lbl = QLabel(f"Log file: {self._log.path}")
        self._path_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self._path_lbl.setWordWrap(True)
        lay.addWidget(self._path_lbl)

        self._summary_lbl = QLabel()
        self._summary_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        self._summary_lbl.setWordWrap(True)
        lay.addWidget(self._summary_lbl)

        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter by category, message, or details...")
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit)

        self._errors_only_chk = QCheckBox("Errors/warnings only")
        self._errors_only_chk.toggled.connect(self._apply_filter)
        filter_row.addWidget(self._errors_only_chk)
        lay.addLayout(filter_row)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Time", "Level", "Category", "Duration", "Message / Details"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("⟳  Refresh")
        btn_refresh.setObjectName("btn_secondary")
        btn_refresh.clicked.connect(self._reload)
        btn_row.addWidget(btn_refresh)

        btn_clear = QPushButton("🗑  Clear View")
        btn_clear.setObjectName("btn_secondary")
        btn_clear.setToolTip("Clears the in-memory log shown here. The on-disk trek_activity.log is kept.")
        btn_clear.clicked.connect(self._clear_view)
        btn_row.addWidget(btn_clear)

        btn_copy = QPushButton("📋  Copy")
        btn_copy.setObjectName("btn_secondary")
        btn_copy.setToolTip("Copy the currently shown log entries to the clipboard.")
        btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_row.addWidget(btn_copy)

        btn_export = QPushButton("💾  Export...")
        btn_export.setObjectName("btn_secondary")
        btn_export.clicked.connect(self._export)
        btn_row.addWidget(btn_export)

        self._btn_maximize = QPushButton("⛶  Maximize")
        self._btn_maximize.setObjectName("btn_secondary")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        btn_row.addWidget(self._btn_maximize)

        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_maximize.setText("⛶  Maximize")
        else:
            self.showMaximized()
            self._btn_maximize.setText("🗗  Restore")

    def _reload(self):
        self._all_entries = self._log.get_entries()
        self._update_summary()
        self._apply_filter()

    def _update_summary(self):
        total = len(self._all_entries)
        errors = sum(1 for e in self._all_entries if e.level == "ERROR")
        warnings = sum(1 for e in self._all_entries if e.level == "WARN")
        timed_entries = [e for e in self._all_entries if e.duration_ms is not None]
        total_time_s = sum(e.duration_ms for e in timed_entries) / 1000.0
        self._summary_lbl.setText(
            f"📋 {total} entries  |  ⏱️ {total_time_s:.1f}s total across {len(timed_entries)} timed operations  |  "
            f"🔴 {errors} error(s)  |  🟡 {warnings} warning(s)"
        )

    def _apply_filter(self):
        text = self._filter_edit.text().strip().lower()
        errors_only = self._errors_only_chk.isChecked()

        filtered = []
        for e in self._all_entries:
            if errors_only and e.level not in ("ERROR", "WARN"):
                continue
            if text:
                haystack = f"{e.category} {e.message} {e.format_details()}".lower()
                if text not in haystack:
                    continue
            filtered.append(e)

        self._table.setRowCount(0)
        for e in filtered:
            row = self._table.rowCount()
            self._table.insertRow(row)

            time_item = QTableWidgetItem(e.timestamp.split("T")[-1][:12])
            self._table.setItem(row, 0, time_item)

            level_item = QTableWidgetItem(e.level)
            if e.level == "ERROR":
                level_item.setForeground(QColor(DANGER_TEXT))
            elif e.level == "WARN":
                level_item.setForeground(QColor(AMBER_COLOR))
            self._table.setItem(row, 1, level_item)

            self._table.setItem(row, 2, QTableWidgetItem(e.category))

            dur_item = QTableWidgetItem(e.format_duration())
            self._table.setItem(row, 3, dur_item)

            detail_str = e.format_details()
            msg_text = f"{e.message}    {detail_str}" if detail_str else e.message
            msg_item = QTableWidgetItem(msg_text)
            if e.level == "ERROR":
                msg_item.setForeground(QColor(DANGER_TEXT))
            self._table.setItem(row, 4, msg_item)

    def _clear_view(self):
        reply = QMessageBox.question(
            self, "Clear Log View",
            "Clear the in-memory log shown here?\n\n"
            "The on-disk trek_activity.log file is NOT deleted -- full "
            "history remains there.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._log.clear(clear_file=False)
            self._reload()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", "trek_activity_export.txt", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        text = self._log.export_text(self._all_entries)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Export Complete", f"Log exported to:\n{path}")
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _copy_to_clipboard(self):
        """Copy the currently shown log entries to the system clipboard,
        using the same text format as Export."""
        text = self._log.export_text(self._all_entries)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            self.setWindowTitle(
                f"TREK Activity Log  —  copied {len(self._all_entries)} entries to clipboard"
            )
        else:
            QMessageBox.warning(self, "Copy Failed", "Clipboard is unavailable.")


class UnresolvedPairsDialog(QDialog):
    """Lists every (syt_id, counterpart_id) pair that TREK returned a LINK
    for but NO usable content for the counterpart -- i.e. exactly the
    pairs silently dropped from "Unique comparison pairs" in the
    traceability summary (see FetchLinksWorker's syt_swt_pair_count vs.
    syt_swt_pair_count_resolved, and the '_' variants for related SYT).
    Gives the user the exact TC id and side to go check manually in TREK
    (archived/deleted/access-restricted test case are the usual cause).
    """

    def __init__(self, unresolved_pairs: List[dict], syt_module: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unresolved Pairs -- Content Not Found")
        self.resize(900, 560)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._unresolved_pairs = unresolved_pairs
        self._syt_module = syt_module
        lay = QVBoxLayout(self)

        lbl = QLabel(
            f"{len(unresolved_pairs)} pair(s) have a LINK in TREK but NO usable content "
            "was returned for the counterpart side (archived/deleted/restricted test "
            "case are the usual cause) -- these are excluded from 'Check Duplicates'."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{TEXT};font-size:12px;padding:4px 0;")
        lay.addWidget(lbl)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["SYT ID", "Missing Counterpart ID", "Counterpart Type"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self._table.setColumnWidth(0, 160)
        self._table.setColumnWidth(2, 140)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(unresolved_pairs))
        for row, pair in enumerate(unresolved_pairs):
            self._table.setItem(row, 0, QTableWidgetItem(pair["syt_id"]))
            self._table.setItem(row, 1, QTableWidgetItem(pair["counterpart_id"]))
            self._table.setItem(row, 2, QTableWidgetItem(pair["counterpart_type"]))
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("📄  Export to JSON")
        btn_export.setObjectName("btn_secondary")
        btn_export.clicked.connect(self._export_to_json)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _export_to_json(self):
        data = {
            "export_meta": {
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "module": self._syt_module,
            },
            "unresolved_pairs": self._unresolved_pairs,
        }
        safe = self._syt_module.replace(" ", "_").replace("/", "-")
        _export_dict_to_json(self, data, f"trek_unresolved_pairs_{safe}.json")


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
def _is_swt_module(name: str) -> bool:
    """SWT/SWIT (software test) module, by first name token."""
    first = re.split(r"[ _\-]", (name or "").strip(), maxsplit=1)[0].upper()
    return first in ("SWT", "SWIT")


def _is_review_or_template_module(name: str) -> bool:
    low = (name or "").lower()
    return "review" in low or "template" in low


class OfflineDownloadDialog(QDialog):
    """Options for 'Offline' -> download the text of all SYT/SWT test cases
    and, optionally, every SYR/SWR requirement module's links + requirement
    text (see DownloadTcContentWorker). Counts come from the in-memory INDEX, so
    opening this dialog costs no network or database time; which of those
    ids are already saved is checked by the worker itself."""

    def __init__(self, parent=None, syt_prefixes: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Offline data -- download all test-case text")
        self.setMinimumWidth(560)
        self._syt_prefixes = list(syt_prefixes or DEFAULT_SYT_PREFIXES)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Downloads the text (name, steps, expected result, chapter) of every "
            "test case in the index, so Get Test Cases, Build Traceability and "
            "Check Duplicates work fully offline.\n\n"
            "• Test cases already saved are skipped -- only what is missing is downloaded.\n"
            "• Runs in the background: you can keep working.\n"
            "• ⏹ Stop at any time -- everything downloaded so far stays saved, and the "
            "next run continues from there."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        box = QGroupBox("What to download")
        form = QVBoxLayout(box)
        self._chk_syt = QCheckBox()
        self._chk_syt.setChecked(True)
        self._chk_swt = QCheckBox()
        self._chk_swt.setChecked(True)
        self._chk_syr = QCheckBox()
        self._chk_syr.setChecked(True)
        self._chk_syr.setToolTip(
            "SYR modules: links (SYR->SWR, SYR->other SYT) + requirement text.\n"
            "What Build Traceability otherwise downloads the first time a SYR\n"
            "module is used (~20-30 s per module). Already-saved modules are skipped.")
        self._chk_swr = QCheckBox()
        self._chk_swr.setChecked(True)
        self._chk_swr.setToolTip(
            "SWR modules: requirement text (shown in the SWR Requirement tab).\n"
            "What Build Traceability otherwise downloads the first time a SWR\n"
            "module is used (~20-30 s per module). Already-saved modules are skipped.")
        self._chk_skip_review = QCheckBox("Skip 'Review' and 'Template' modules")
        self._chk_skip_review.setChecked(True)
        self._chk_retry = QCheckBox("Retry test cases TREK did not return last time")
        self._chk_retry.setChecked(False)
        self._chk_retry.setToolTip(
            "Ids TREK answered for but did not return in a previous run (deleted, "
            "archived, restricted) are skipped by default so every run doesn't "
            "re-request them.")
        for w in (self._chk_syt, self._chk_swt, self._chk_syr, self._chk_swr,
                  self._chk_skip_review, self._chk_retry):
            form.addWidget(w)
        lay.addWidget(box)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self._btn_start = QPushButton("Start Download")
        self._btn_start.setObjectName("btn_success")
        self._btn_start.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_start)
        lay.addLayout(btn_row)

        for w in (self._chk_syt, self._chk_swt, self._chk_syr, self._chk_swr, self._chk_skip_review):
            w.toggled.connect(self._refresh_counts)
        self._refresh_counts()

    def _groups(self) -> Dict[str, Dict[str, List[str]]]:
        skip = self._chk_skip_review.isChecked()
        syt = INDEX.ids_by_module(
            lambda m: _is_syt_module(m, self._syt_prefixes)
            and not (skip and _is_review_or_template_module(m)))
        swt = INDEX.ids_by_module(
            lambda m: _is_swt_module(m)
            and not (skip and _is_review_or_template_module(m)))
        return {"SYT": syt, "SWT": swt}

    def _req_module_names(self) -> Dict[str, List[str]]:
        """SYR / SWR requirement modules known to the index (the same modules
        Build Traceability can bridge through)."""
        skip = self._chk_skip_review.isChecked()
        mods = set(INDEX.ids_by_module(lambda m: True).keys()) if INDEX.is_built() else set()
        out: Dict[str, List[str]] = {"SYR": [], "SWR": []}
        for m in sorted(mods):
            if skip and _is_review_or_template_module(m):
                continue
            first = re.split(r"[ _\-]", m.strip(), maxsplit=1)[0].upper()
            if (first == "SYR" or m in _EXTRA_SYR_LIKE_MODULES) and m not in _SYR_EXCLUDED_MODULES:
                out["SYR"].append(m)
            elif first == "SWR":
                out["SWR"].append(m)
        return out

    def _refresh_counts(self):
        g = self._groups()
        n_syt = sum(len(v) for v in g["SYT"].values())
        n_swt = sum(len(v) for v in g["SWT"].values())
        self._chk_syt.setText(f"SYT test cases -- {n_syt:,} in {len(g['SYT'])} modules")
        self._chk_swt.setText(f"SWT test cases -- {n_swt:,} in {len(g['SWT'])} modules")
        rm = self._req_module_names()
        self._chk_syr.setText(
            f"SYR requirement modules -- {len(rm['SYR'])} modules (links + requirement text)")
        self._chk_swr.setText(
            f"SWR requirement modules -- {len(rm['SWR'])} modules (requirement text)")
        total = (n_syt if self._chk_syt.isChecked() else 0) + (n_swt if self._chk_swt.isChecked() else 0)
        n_req = ((len(rm["SYR"]) if self._chk_syr.isChecked() else 0)
                 + (len(rm["SWR"]) if self._chk_swr.isChecked() else 0))
        have = CACHE.tc_content_count() if hasattr(CACHE, "tc_content_count") else None
        self._summary.setText(
            f"Selected: <b>{total:,}</b> test cases"
            + (f" (about {have:,} test-case texts are already saved in the database)" if have else "")
            + (f" + <b>{n_req}</b> requirement modules" if n_req else "")
            + ".<br>Already-saved data is skipped. Progress and time remaining are shown "
              "in the status bar. A first full run can take 15-40 minutes, depending on TREK."
        )
        self._summary.setTextFormat(Qt.RichText)
        self._btn_start.setEnabled(total > 0 or n_req > 0)

    def id_groups(self) -> Dict[str, List[str]]:
        g = self._groups()
        out: Dict[str, List[str]] = {}
        if self._chk_syt.isChecked():
            out["SYT"] = [i for ids in g["SYT"].values() for i in ids]
        if self._chk_swt.isChecked():
            out["SWT"] = [i for ids in g["SWT"].values() for i in ids]
        return out

    def retry_not_returned(self) -> bool:
        return self._chk_retry.isChecked()

    def req_modules(self) -> Dict[str, List[str]]:
        rm = self._req_module_names()
        out: Dict[str, List[str]] = {}
        if self._chk_syr.isChecked() and rm["SYR"]:
            out["SYR"] = rm["SYR"]
        if self._chk_swr.isChecked() and rm["SWR"]:
            out["SWR"] = rm["SWR"]
        return out


class CheckableComboBox(QComboBox):
    """Multi-select dropdown: every row has a checkbox, clicking a row
    toggles it and the list STAYS OPEN, and the box shows the checked
    values ("SYT, SWT"). A plain QComboBox with checkable model items
    selects-and-closes on click and can only display one row's text.
    Emits checkedChanged(list_of_texts) after every toggle."""
    checkedChanged = Signal(list)

    def __init__(self, parent=None, empty_text: str = "(none)"):
        super().__init__(parent)
        self._empty_text = empty_text
        self.setEditable(True)                      # only to display custom text
        self.setInsertPolicy(QComboBox.NoInsert)
        le = self.lineEdit()
        le.setReadOnly(True)
        le.setCursor(Qt.PointingHandCursor)
        le.installEventFilter(self)
        self.setModel(QStandardItemModel(self))
        self.view().viewport().installEventFilter(self)
        # Checkbox look comes from the global theme stylesheet
        # ("QComboBox QAbstractItemView::indicator" in trek_theme.py).

    # -- items ---------------------------------------------------------
    def set_items(self, texts: List[str], checked: Iterable[str]):
        checked = set(checked)
        m = self.model()
        m.blockSignals(True)
        m.clear()
        for t in texts:
            it = QStandardItem(t)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setData(Qt.Checked if t in checked else Qt.Unchecked, Qt.CheckStateRole)
            m.appendRow(it)
        m.blockSignals(False)
        self.view().reset()
        self._update_text()

    def checked_items(self) -> List[str]:
        m = self.model()
        return [m.item(r).text() for r in range(m.rowCount())
                if m.item(r).checkState() == Qt.Checked]

    def set_checked(self, text: str, on: bool):
        m = self.model()
        for r in range(m.rowCount()):
            if m.item(r).text() == text:
                m.item(r).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._update_text()

    # -- behaviour -----------------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.lineEdit() and ev.type() == QEvent.MouseButtonRelease:
            self.showPopup()
            return True
        if obj is self.view().viewport() and ev.type() == QEvent.MouseButtonRelease:
            idx = self.view().indexAt(ev.position().toPoint())
            item = self.model().itemFromIndex(idx) if idx.isValid() else None
            if item is not None:
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
                self._update_text()
                self.checkedChanged.emit(self.checked_items())
            return True   # swallow the release -> the list stays open
        return super().eventFilter(obj, ev)

    def hidePopup(self):
        super().hidePopup()
        self._update_text()          # QComboBox may have rewritten the text

    def _update_text(self):
        checked = self.checked_items()
        text = ", ".join(checked) if checked else self._empty_text
        fm = self.lineEdit().fontMetrics()
        self.lineEdit().setText(fm.elidedText(text, Qt.ElideRight, max(40, self.lineEdit().width() - 4)))
        self.lineEdit().setToolTip(text)
        self.setToolTip("Module-name prefixes shown in the list: " + text +
                        "\nClick to choose (several can be selected).")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_text()


class _IndexLoadBridge(QObject):
    """Delivers a background index load result to the UI thread."""
    done = Signal(int, object, tuple, float)    # (seq, TrekIndex, (pid, cid, cfg), seconds)


def _load_index_in_thread(bridge: "_IndexLoadBridge", seq: int, cache, pid, cid, cfg):
    """Runs in a DAEMON Python thread: read-only, so the app can exit at any
    moment without waiting for it (a running QThread aborts the process when
    the app closes -- this load can take minutes on an unoptimized share)."""
    t0 = time.perf_counter()
    idx = trek_index.TrekIndex(cache, pid, cid, cfg)
    try:
        idx.load()
    except Exception as exc:   # noqa: BLE001
        LOG.log("Index", f"Background index load failed: {exc}", level="ERROR")
    try:
        bridge.done.emit(seq, idx, (pid, cid, cfg), time.perf_counter() - t0)
    except RuntimeError:
        pass      # window already gone


class TrekMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI DupeHunter")
        self.resize(1400, 860)
        self.setMinimumSize(1000, 600)

        # State
        self._modules:       list = []
        self._syt_modules:   list = []
        self._selected_mod:  Optional[dict] = None
        self._all_syt_ids:   list = []
        self._syr_candidates:list = []
        self._swt_candidates:list = []
        self._all_syr_module_names: list = []   # every real "SYR -" module (for ModuleMappingDialog)
        self._all_swt_module_names: list = []   # every real "SWT -"/"SWIT -" module (for ModuleMappingDialog)
        self._tc_chapters:   Dict[str, str] = {}   # tc_id -> DOORS chapter (Module_Path)
        self._result:        Optional[dict] = None
        self._workers = []   # keep alive
        self._index_worker = None            # BuildIndexWorker (kept alive)
        self._pending_tc_load = None         # queued TC load after index build
        self._index_loading = False          # background INDEX load in progress
        self._index_load_seq = 0
        self._index_bridge = _IndexLoadBridge(self)
        self._index_bridge.done.connect(self._on_index_loaded)
        self._after_index_load = None        # action to resume once the INDEX is loaded
        self._tc_table_signal_connected = False   # tracks itemChanged wiring for the TC tree
        self._syt_prefixes: List[str] = list(DEFAULT_SYT_PREFIXES)

        # Busy/elapsed-time indicator (⏳ <stage> (MM:SS)) shown in the status
        # bar while any background worker is running, so the user always
        # sees what's happening next and how long it's been running.
        self._busy_stage_text = ""
        self._busy_elapsed = QElapsedTimer()
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(1000)
        self._busy_timer.timeout.connect(self._tick_busy_indicator)

        self._load_syt_prefixes_for_active_project()
        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        if not INDEX.is_built():
            self._start_index_load()

    def closeEvent(self, event):
        """Stop background workers cleanly before the window closes -- a
        QThread still running at exit aborts the whole process. Long jobs
        (Offline download) stop at the next batch; everything already
        downloaded is saved."""
        running = [w for w in list(self._workers)
                   if isinstance(w, QThread) and w.isRunning()]
        if running:
            self._set_status("Stopping background tasks...")
            for w in running:
                if isinstance(w, _CancellableWorker):
                    w.request_cancel()
            deadline = time.perf_counter() + 20
            for w in running:
                w.wait(max(0, int((deadline - time.perf_counter()) * 1000)))
        super().closeEvent(event)

    # -----------------------------------------------------------------------
    # Background INDEX loading
    # -----------------------------------------------------------------------
    def _start_index_load(self):
        """Load the active project's id->module INDEX in the background so
        the window is usable immediately (on a network share the index read
        took ~2 minutes and used to block startup / project switching)."""
        import threading
        self._index_loading = True
        self._index_load_seq += 1
        threading.Thread(
            target=_load_index_in_thread, name="index-load", daemon=True,
            args=(self._index_bridge, self._index_load_seq, CACHE, PROJECT_ID, CAMPAIGN_ID, CONFIG_ID),
        ).start()
        if getattr(self, "_process_lbl", None) is not None:
            self._set_status("⏳ Loading the ID index in the background -- you can already start working.")

    def _on_index_loaded(self, seq: int, idx, key: tuple, seconds: float):
        global INDEX
        if seq != self._index_load_seq or key != (PROJECT_ID, CAMPAIGN_ID, CONFIG_ID):
            return      # project switched meanwhile; a newer load is running
        INDEX = idx
        self._index_loading = False
        LOG.log("Index", f"Index loaded in background: {idx.size():,} ids in {seconds:.1f}s",
                duration_ms=seconds * 1000)
        if idx.size():
            self._set_status(f"ID index ready ({idx.size():,} ids, loaded in {seconds:.1f}s).")
        action, self._after_index_load = self._after_index_load, None
        if action is not None:
            action()

    def _wait_for_index(self, action, what: str) -> bool:
        """If the INDEX is still loading, remember ``action`` to run as soon
        as it is ready and return True (caller should stop)."""
        if not self._index_loading:
            return False
        self._after_index_load = action
        self._set_status(f"⏳ Waiting for the ID index to finish loading -- {what} will start automatically.")
        return True

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        root.addWidget(self._make_header())

        # Body: 3-pane horizontal splitter
        self._body_splitter = QSplitter(Qt.Horizontal)
        self._body_splitter.setHandleWidth(2)
        self._body_splitter.addWidget(self._make_step1_panel())
        self._body_splitter.addWidget(self._make_step2_panel())
        self._body_splitter.addWidget(self._make_step3_panel())
        self._body_splitter.setSizes([280, 320, 800])
        self._step1_expanded = True
        self._step1_saved_width = 280
        root.addWidget(self._body_splitter, 1)

        # Status bar. Taller, with a dedicated prominent "current process"
        # label (accent-coloured, bold) plus a styled progress bar, so the
        # live action is easy to read and looks professional.
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setSizeGripEnabled(False)
        self.status.setMinimumHeight(44)
        self.status.setStyleSheet(
            f"QStatusBar {{ min-height:44px; }} "
            f"QStatusBar::item {{ border:none; }}"
        )

        # Prominent live-action label (left side). Bold accent text so the
        # user can always see what the app is doing right now. Shown via
        # addWidget (not showMessage) so we control its styling fully.
        self._process_lbl = QLabel("Ready — fetch modules to start.")
        self._process_lbl.setObjectName("process_lbl")
        self._process_lbl.setStyleSheet(
            f"#process_lbl {{ color:{TEXT}; font-size:14px; font-weight:600; "
            f"padding:0 10px; }}"
        )
        self.status.addWidget(self._process_lbl, 1)

        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setObjectName("btn_secondary")
        self._btn_stop.setToolTip("Cancel the currently running action (fetch/sync/Check Duplicates).")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_current_action)
        self.status.addPermanentWidget(self._btn_stop)

        # Professional-looking indeterminate progress bar: rounded track,
        # accent chunk, comfortable size. Hidden until an action runs.
        self._progress = QProgressBar()
        self._progress.setObjectName("busy_progress")
        self._progress.setFixedWidth(300)
        self._progress.setMinimumHeight(20)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setStyleSheet(
            f"#busy_progress {{ border:1px solid {BORDER}; border-radius:10px; "
            f"background:{PANEL_BG}; }} "
            f"#busy_progress::chunk {{ border-radius:9px; margin:1px; "
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0, "
            f"stop:0 {ACCENT}, stop:1 {ACCENT_HOVER}); }}"
        )
        self.status.addPermanentWidget(self._progress)

        # Which cache this session is using -- matters when a team points
        # several installs at one shared folder (see trek_paths.data_dir),
        # or when the active project has its own custom database path (see
        # ProjectSetupDialog's "Cache Database" field / _apply_active_project).
        self._cache_lbl = QLabel()
        self._cache_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;padding:0 8px;")
        self.status.addPermanentWidget(self._cache_lbl)
        self._refresh_cache_label()
        self._set_status("Ready — fetch modules to start.")

    def _refresh_cache_label(self):
        """(Re)populate the status-bar cache indicator from the CURRENT
        global CACHE handle -- called at startup and again after switching/
        editing a project, since _apply_active_project() may have re-pointed
        CACHE at that project's custom database path."""
        scope = "shared" if CACHE.is_shared else "local"
        self._cache_lbl.setText(f"🗄 {scope} cache")
        active = PROJECT_STORE.get_active()
        custom_path = (active or {}).get("db_path") or ""
        if custom_path:
            location_line = f"Custom cache DB for this project:\n{CACHE.db_path}"
        else:
            location_line = f"Cache / log / projects folder:\n{trek_paths.data_dir()}"
        self._cache_lbl.setToolTip(
            f"{location_line}\n\n"
            f"SQLite journal mode: {CACHE.journal_mode}\n\n"
            "Point colleagues at the same folder/file to share extracted "
            "data: set the TREK_DATA_DIR environment variable, put a "
            "trek_data_dir.txt file next to the .exe, or set a per-project "
            "Cache Database path in Edit Project."
        )

    # -----------------------------------------------------------------------
    # Busy / elapsed-time indicator
    # -----------------------------------------------------------------------
    def _busy_start(self, stage: str):
        """Start (or restart) the ⏳ elapsed-time indicator with an initial
        stage description. Call this right before starting a worker."""
        self._busy_stage_text = stage
        self._busy_elapsed.start()
        self._render_busy_indicator()
        self._busy_timer.start()
        self._btn_stop.setEnabled(True)

    def _busy_stage_update(self, stage: str):
        """Update the stage text shown next to the elapsed time, without
        resetting the clock. Wire this to a worker's `progress` signal so
        the user sees what's happening next as a multi-step fetch runs."""
        self._busy_stage_text = stage
        self._render_busy_indicator()

    def _busy_stop(self):
        self._busy_timer.stop()
        self._btn_stop.setEnabled(False)
        # Reset the live-action label back to the neutral (non-accent) style.
        self._process_lbl.setStyleSheet(
            f"#process_lbl {{ color:{TEXT}; font-size:14px; font-weight:600; "
            f"padding:0 10px; }}"
        )

    def _set_status(self, text: str):
        """Update the prominent status-bar process label. Replaces
        status.showMessage() so our styled label isn't hidden by Qt's
        temporary-message mechanism (showMessage hides non-permanent
        widgets while a message is shown)."""
        if getattr(self, "_process_lbl", None) is not None:
            self._process_lbl.setText(text)
        else:
            self.status.showMessage(text)

    def _stop_current_action(self):
        """Cooperatively cancel whatever worker(s) are currently running
        (see _CancellableWorker) -- fetch modules, get test cases, build
        traceability, or Check Duplicates. Cancellation is checked between
        stages/batches, not instant, so this may take a moment to actually
        stop; the worker's own error handler shows "Cancelled by user."
        instead of a hard failure once it notices."""
        cancelled_any = False
        for worker in list(self._workers):
            if isinstance(worker, _CancellableWorker):
                worker.request_cancel()
                cancelled_any = True
        if cancelled_any:
            self._busy_stage_update("Stopping (finishing current request)...")
        self._btn_stop.setEnabled(False)

    def _tick_busy_indicator(self):
        self._render_busy_indicator()

    def _render_busy_indicator(self):
        secs = self._busy_elapsed.elapsed() // 1000
        mins, secs = divmod(int(secs), 60)
        # Prominent accent-coloured live text with elapsed timer.
        self._process_lbl.setStyleSheet(
            f"#process_lbl {{ color:{ACCENT}; font-size:14px; font-weight:700; "
            f"padding:0 10px; }}"
        )
        self._process_lbl.setText(f"⏳  {self._busy_stage_text}   ·   {mins:02d}:{secs:02d}")

    def _make_header(self):
        hdr = QWidget()
        hdr.setFixedHeight(54)
        hdr.setStyleSheet(f"background:{PANEL_BG};border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(20, 0, 20, 0)
        # Keep references to the header + its inline-styled bits so a live
        # theme switch can recolour them (their inline styles bake the
        # colour constants at construction time). See _change_theme().
        self._header_widget = hdr

        icon_lbl = QLabel("🔍")
        icon_lbl.setStyleSheet(f"font-size:22px;color:{ACCENT};")
        lay.addWidget(icon_lbl)
        self._header_icon_lbl = icon_lbl

        title = QLabel("AI DupeHunter")
        title.setStyleSheet(f"font-size:14px;font-weight:bold;color:{TEXT};margin-left:6px;")
        lay.addWidget(title)
        self._header_title_lbl = title
        lay.addStretch()

        self._btn_show_log = QPushButton("Log")
        self._btn_show_log.setObjectName("btn_secondary")
        self._btn_show_log.setToolTip("Show activity log — every API call, cache hit/miss, timing.")
        self._btn_show_log.clicked.connect(self._show_log_dialog)
        lay.addWidget(self._btn_show_log)

        btn_database = QPushButton("Database")
        btn_database.setObjectName("btn_secondary")
        btn_database.setToolTip("View cache database — file size, cached entries, embeddings, and Optimize Storage.")
        btn_database.clicked.connect(self._show_database_dialog)
        lay.addWidget(btn_database)

        btn_clear_cache = QPushButton("Delete")
        btn_clear_cache.setObjectName("btn_secondary")
        btn_clear_cache.setToolTip("Delete ALL cached data (modules, links, embeddings, LLM judgments, results, stats).")
        btn_clear_cache.clicked.connect(self._confirm_and_clear_cache)
        lay.addWidget(btn_clear_cache)

        btn_cached_results = QPushButton("Results")
        btn_cached_results.setObjectName("btn_secondary")
        btn_cached_results.setToolTip("Browse cached 'Check Duplicates' results per module.")
        btn_cached_results.clicked.connect(self._show_cached_results_dialog)
        lay.addWidget(btn_cached_results)

        btn_compare_models = QPushButton("Compare")
        btn_compare_models.setObjectName("btn_secondary")
        btn_compare_models.setToolTip("Compare two cached LLM model versions for the same module.")
        btn_compare_models.clicked.connect(self._show_compare_models_dialog)
        lay.addWidget(btn_compare_models)

        btn_app_stats = QPushButton("Report")
        btn_app_stats.setObjectName("btn_secondary")
        btn_app_stats.setToolTip("Performance & cost statistics across all runs.")
        btn_app_stats.clicked.connect(self._show_app_stats_dialog)
        lay.addWidget(btn_app_stats)

        self._btn_build_index = QPushButton("Build")
        self._btn_build_index.setObjectName("btn_secondary")
        self._btn_build_index.setToolTip("Build the id→module index (SYR/SWR/SYT/SWT). One-time fetch, cached.")
        self._btn_build_index.clicked.connect(self._build_index)
        lay.addWidget(self._btn_build_index)

        self._btn_offline = QPushButton("Offline")
        self._btn_offline.setObjectName("btn_secondary")
        self._btn_offline.setToolTip(
            "Download ALL SYT/SWT test-case text + SYR/SWR requirement modules\n"
            "(only what is missing) so everything works without TREK.\n"
            "Runs in the background; Stop keeps what was downloaded and the\n"
            "next run continues from there.")
        self._btn_offline.clicked.connect(self._download_all_tc_text)
        lay.addWidget(self._btn_offline)

        btn_dashboard = QPushButton("Dashboard")
        btn_dashboard.setObjectName("btn_secondary")
        btn_dashboard.setToolTip("Overview dashboard — LLM accuracy & review progress across all cached modules.")
        btn_dashboard.clicked.connect(self._show_overview_dashboard)
        lay.addWidget(btn_dashboard)

        # Project switcher: shows every saved project (name -- Project X ·
        # Campaign Y) plus a trailing "+ Add Project..." entry. Selecting a
        # saved project switches PROJECT_ID/CAMPAIGN_ID/CONFIG_ID globally
        # (see _apply_active_project) and refreshes Step 1's module list;
        # selecting "+ Add Project..." opens ProjectSetupDialog instead of
        # actually switching.
        proj_lbl = QLabel("Project:")
        proj_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-left:16px;")
        lay.addWidget(proj_lbl)

        self._project_combo = QComboBox()
        self._project_combo.setMinimumWidth(220)
        self._project_combo.currentIndexChanged.connect(self._on_project_combo_changed)
        lay.addWidget(self._project_combo)

        btn_edit_proj = QPushButton("✎  Edit Project")
        btn_edit_proj.setObjectName("btn_secondary")
        btn_edit_proj.setToolTip(
            "Edit the currently selected project's connection details\n"
            "(name, Project ID, Campaign ID, Config ID, JWT Token)."
        )
        btn_edit_proj.clicked.connect(self._edit_current_project)
        lay.addWidget(btn_edit_proj)

        # Theme switcher: choose the app colour theme. The choice is
        # persisted (trek_theme) and re-applied immediately; a short note
        # reminds the user a restart makes it perfect everywhere.
        theme_lbl = QLabel("Theme:")
        theme_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;margin-left:16px;")
        lay.addWidget(theme_lbl)

        self._theme_combo = QComboBox()
        self._theme_combo.setMinimumWidth(150)
        self._theme_combo.setToolTip("Change the application colour theme.")
        self._theme_combo.addItems(trek_theme.theme_names())
        # Block signals while setting the initial value: _make_header() runs
        # before self.status exists, and currentTextChanged -> _change_theme
        # would otherwise fire during construction and touch self.status.
        self._theme_combo.blockSignals(True)
        self._theme_combo.setCurrentText(trek_theme.get_active_name())
        self._theme_combo.blockSignals(False)
        self._theme_combo.currentTextChanged.connect(self._change_theme)
        lay.addWidget(self._theme_combo)

        self._refresh_project_combo()
        return hdr

    def _change_theme(self, name: str):
        """Persist and apply the chosen colour theme. Re-fills the module
        colour constants, rebuilds the global stylesheet, recolours the
        header's inline-styled widgets, and forces a full repaint so the
        change is visible immediately. A restart makes every remaining
        baked-in inline colour perfect too."""
        trek_theme.set_active(name)
        theme = trek_theme.get_theme(name)
        _apply_theme_constants(theme)
        global STYLESHEET
        STYLESHEET = trek_theme.build_stylesheet(theme)

        app = QApplication.instance()
        if app is not None:
            # Override the application palette's Highlight/HighlightedText
            # so Qt's native selection drawing (which bleeds through even
            # when QSS ::item:selected is set) uses the theme's soft
            # selection colour instead of the OS default purple/blue.
            from PySide6.QtGui import QPalette
            pal = app.palette()
            pal.setColor(QPalette.Highlight, QColor(SELECTION_BG))
            pal.setColor(QPalette.HighlightedText, QColor(TEXT))
            app.setPalette(pal)
            # Clearing first forces Qt to fully re-parse & re-polish every
            # widget when the new sheet is set (re-setting the same-ish
            # sheet is otherwise sometimes treated as a no-op and existing
            # widgets keep their old palette until they next repaint).
            app.setStyleSheet("")
            app.setStyleSheet(STYLESHEET)
        # CRUCIAL: the main window ALSO has its own stylesheet set in
        # __init__ (self.setStyleSheet(STYLESHEET)). A widget-level sheet
        # takes precedence over the app-level one, so unless we re-set it
        # here too the window keeps rendering the OLD theme regardless of
        # the app-level update above.
        self.setStyleSheet(STYLESHEET)

        # Recolour the header pieces whose inline styles baked the old
        # constants at construction time.
        if getattr(self, "_header_widget", None) is not None:
            self._header_widget.setStyleSheet(
                f"background:{PANEL_BG};border-bottom:1px solid {BORDER};"
            )
        if getattr(self, "_header_icon_lbl", None) is not None:
            self._header_icon_lbl.setStyleSheet(f"font-size:22px;color:{ACCENT};")
        if getattr(self, "_header_title_lbl", None) is not None:
            self._header_title_lbl.setStyleSheet(
                f"font-size:17px;font-weight:bold;color:{TEXT};margin-left:8px;"
            )
        # Recolour the status-bar process label + progress bar for the theme.
        if getattr(self, "_process_lbl", None) is not None:
            self._process_lbl.setStyleSheet(
                f"#process_lbl {{ color:{TEXT}; font-size:14px; font-weight:600; "
                f"padding:0 10px; }}"
            )
        if getattr(self, "_progress", None) is not None:
            self._progress.setStyleSheet(
                f"#busy_progress {{ border:1px solid {BORDER}; border-radius:10px; "
                f"background:{PANEL_BG}; }} "
                f"#busy_progress::chunk {{ border-radius:9px; margin:1px; "
                f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0, "
                f"stop:0 {ACCENT}, stop:1 {ACCENT_HOVER}); }}"
            )

        # Re-apply the stylesheet to EVERY open top-level window (dialogs,
        # tool windows, etc.) -- not just the main window. A dialog that was
        # opened before the theme switch inherits the OLD app stylesheet at
        # construction time, and a widget-level sheet takes precedence over
        # the app-level one, so unless we explicitly re-set it here those
        # windows keep the old theme colours.
        #
        # Additionally, dialogs whose content is rendered with baked-in
        # colour constants (inline HTML, setForeground, etc.) need a full
        # content re-render -- those implement a refresh_theme() method.
        if app is not None:
            for tlw in app.topLevelWidgets():
                if tlw is not self and tlw.isVisible():
                    tlw.setStyleSheet(STYLESHEET)
                    if hasattr(tlw, "refresh_theme"):
                        tlw.refresh_theme()

        # Force a style re-polish on VISIBLE top-level windows only (not
        # every single widget in the app -- the old allWidgets() loop was
        # iterating thousands of widgets and was the main reason theme
        # switches felt sluggish). Setting the stylesheet on each top-level
        # widget already cascades to its children; we only need to fixup
        # QTabBars whose geometry gets stale after a stylesheet swap.
        if app is not None:
            from PySide6.QtWidgets import QTabBar
            for tlw in app.topLevelWidgets():
                if tlw.isVisible():
                    for tab in tlw.findChildren(QTabBar):
                        tab.adjustSize()
                        tab.updateGeometry()

        # Update Step-3 tree item colours IN PLACE (without rebuilding the
        # tree, which would lose expand/collapse state and selection). The
        # foreground colours on tree items are baked at build time via
        # setForeground, so we walk existing items and re-apply the current
        # theme constants. Detail panels (HTML) are cleared so they
        # re-render with new colours on next click.
        if getattr(self, "_result", None):
            try:
                # Walk tree items and re-apply foreground colours in place
                # (preserves expand/collapse state and selection).
                _tree = getattr(self, "_tree", None)
                if _tree is not None:
                    self._recolor_tree_items(_tree.invisibleRootItem())
                    # Re-render the detail panel for the currently selected
                    # item so it picks up new theme colours immediately
                    # (instead of showing a blank panel until the user
                    # clicks something).
                    current = _tree.currentItem()
                    if current is not None:
                        self._on_tree_item_clicked(current, 0)
            except Exception:   # noqa: BLE001 - never let a repaint break theme switch
                pass

        if getattr(self, "status", None) is not None:
            self._set_status(f"Theme set to '{name}'.")

    # -----------------------------------------------------------------------
    # Project switching
    # -----------------------------------------------------------------------
    def _refresh_project_combo(self):
        """Repopulate the header project dropdown from PROJECT_STORE and
        select whichever project is currently active, without triggering
        a spurious project switch while doing so."""
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        active_key = PROJECT_STORE.get_active().get("key") if PROJECT_STORE.get_active() else None
        select_index = 0
        for i, proj in enumerate(PROJECT_STORE.list_projects()):
            label = f"{proj['name']}  (Project {proj['project_id']} · Campaign {proj['campaign_id']})"
            self._project_combo.addItem(label, proj["key"])
            if proj["key"] == active_key:
                select_index = i
        self._project_combo.addItem("+ Add Project...", "__add_new__")
        self._project_combo.setCurrentIndex(select_index)
        self._project_combo.blockSignals(False)

    def _on_project_combo_changed(self, index: int):
        key = self._project_combo.itemData(index)
        if key is None:
            return
        if key == "__add_new__":
            self._add_new_project()
            return

        active = PROJECT_STORE.get_active()
        if active and active["key"] == key:
            return  # already active, nothing to do

        PROJECT_STORE.set_active(key)
        _apply_active_project(load_index=False)
        self._start_index_load()
        self._on_active_project_switched()

    def _add_new_project(self):
        dlg = ProjectSetupDialog(parent=self, allow_cancel=True)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.result_data()
            PROJECT_STORE.add_project(
                data["name"], data["project_id"], data["campaign_id"], data["config_id"],
                jwt_token=data["jwt_token"], make_active=True,
                db_path=data.get("db_path", ""),
            )
            _apply_active_project(load_index=False)
            self._start_index_load()
            self._refresh_project_combo()
            self._on_active_project_switched()
        else:
            # User cancelled "+ Add Project..." -- snap the combo back to
            # whatever project is actually still active instead of leaving
            # it stuck on the "+ Add Project..." entry.
            self._refresh_project_combo()

    def _edit_current_project(self):
        active = PROJECT_STORE.get_active()
        if not active:
            self._add_new_project()
            return
        dlg = ProjectSetupDialog(parent=self, existing=active, allow_cancel=True)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.result_data()
            # Detect whether connection-relevant fields changed -- if only
            # cosmetic fields (name, JWT) were edited, skip the expensive
            # full reset (module re-fetch, state wipe) and just refresh the
            # combo + status bar.
            old_conn = (active.get("project_id"), active.get("campaign_id"),
                        active.get("config_id"), active.get("db_path", ""))
            new_conn = (data["project_id"], data["campaign_id"],
                        data["config_id"], data.get("db_path", ""))
            PROJECT_STORE.update_project(
                active["key"], data["name"], data["project_id"],
                data["campaign_id"], data["config_id"], jwt_token=data["jwt_token"],
                db_path=data.get("db_path", ""),
            )
            _apply_active_project(load_index=False)
            self._start_index_load()
            self._refresh_project_combo()
            if old_conn != new_conn:
                self._on_active_project_switched()
            else:
                self._set_status(f"Project '{data['name']}' updated.")

    def _build_index(self, force: bool = False):
        """Kick off the id->module index build (SYR + SWR + SYT + SWT) off the
        UI thread. Once built, every hop of Build Traceability resolves ids to
        modules instantly. Called directly by the 'Build Index' button, and
        automatically by the first 'Get Test Cases' when the index is missing
        (see _get_test_cases_clicked / _pending_tc_load)."""
        if getattr(self, "_index_worker", None) is not None and self._index_worker.isRunning():
            self._set_status("Index build already running...")
            return
        if self._wait_for_index(lambda: self._build_index(force), "Build"):
            return
        fully_built = (INDEX.is_built("SYR") and INDEX.is_built("SWR")
                       and INDEX.is_built("SYT") and INDEX.is_built("SWT"))
        # If the user clicked the 'Build Index' button while it's already
        # fully built, ask whether to rebuild. (When called from Get Test
        # Cases with a pending load, we only get here BECAUSE it wasn't fully
        # built, so this prompt won't fire in that path.)
        if fully_built and not force:
            resp = QMessageBox.question(
                self, "Index Already Built",
                f"The index is already built ({INDEX.size()} ids).\n\n"
                f"Do you want to rebuild it from TREK (ignores cache)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
            force = True

        self._btn_build_index.setEnabled(False)
        self._busy_start("Building id→module index (SYR + SWR + SYT + SWT)...")
        self._progress.setVisible(True)
        worker = BuildIndexWorker(force=force, syt_prefixes=self._syt_prefixes)
        worker.progress.connect(self._busy_stage_update)
        worker.done.connect(self._on_index_built)
        worker.error.connect(self._on_index_error)
        self._index_worker = worker
        worker.start()

    def _download_all_tc_text(self):
        """Header 'Offline' button: download the text of every SYT/SWT test
        case in the index that is not saved yet (DownloadTcContentWorker)."""
        if self._wait_for_index(self._download_all_tc_text, "Offline"):
            return
        running = getattr(self, "_offline_worker", None)
        if running is not None and running.isRunning():
            self._set_status("Offline download already running -- see progress above.")
            return
        if not (INDEX.is_built("SYT") or INDEX.is_built("SWT")):
            QMessageBox.information(
                self, "Build the index first",
                "The offline download takes its list of test cases from the index.\n\n"
                "Press 'Build' first (one-time), then 'Offline'.")
            return
        dlg = OfflineDownloadDialog(self, syt_prefixes=self._syt_prefixes)
        if dlg.exec() != QDialog.Accepted:
            return
        groups = dlg.id_groups()
        req_modules = dlg.req_modules()
        if not any(groups.values()) and not any(req_modules.values()):
            return

        worker = DownloadTcContentWorker(groups, retry_not_returned=dlg.retry_not_returned(),
                                         req_modules=req_modules)
        worker.progress.connect(self._busy_stage_update)
        worker.done.connect(self._on_offline_download_done)
        worker.error.connect(self._on_offline_download_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._offline_worker = worker
        self._workers.append(worker)
        self._btn_offline.setEnabled(False)
        self._progress.setVisible(True)
        self._busy_start("Offline download: checking what is already saved...")
        worker.start()

    def _on_offline_download_done(self, report: dict):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_offline.setEnabled(True)
        self._refresh_cache_label()

        mins, secs = divmod(int(report.get("duration_seconds", 0)), 60)
        lines = []
        for kind, k in report.get("kinds", {}).items():
            lines.append(
                f"{kind}: {k['total']:,} total · {k['already_saved']:,} already saved · "
                f"{k['downloaded']:,} downloaded"
                + (f" · {k['not_returned']:,} not returned by TREK" if k["not_returned"] else "")
                + (f" · {k['failed']:,} failed" if k["failed"] else "")
                + (f" · {k['skipped_known_missing']:,} skipped (not in TREK last time)"
                   if k.get("skipped_known_missing") else "")
            )
        rq = report.get("req") or {}
        if rq:
            lines.append(
                f"SYR/SWR modules: {rq.get('modules', 0)} modules · "
                f"{rq.get('already_saved', 0)} downloads already saved · "
                f"{rq.get('downloaded', 0)} downloaded"
                + (f" · {rq['failed']} failed" if rq.get("failed") else "")
            )
        body = "\n".join(lines) + f"\n\nTime: {mins}m {secs:02d}s"

        if report.get("unreachable"):
            title, icon = "Offline download stopped -- TREK unreachable", QMessageBox.Warning
            body = ("TREK could not be reached, so the download stopped. Everything "
                    "downloaded before that is saved; run 'Offline' again when connected.\n\n"
                    f"Error: {report.get('error', '')}\n\n" + body)
        elif report.get("cancelled"):
            title, icon = "Offline download stopped", QMessageBox.Information
            body = ("Stopped by you. Everything downloaded so far is saved; run 'Offline' "
                    "again to continue with what is still missing.\n\n" + body)
        elif report.get("to_download", 0) == 0 and not rq.get("to_download"):
            title, icon = "Offline data is complete", QMessageBox.Information
            body = "Nothing to download -- all selected test cases are already saved.\n\n" + body
        else:
            title, icon = "Offline download finished", QMessageBox.Information
            if report.get("not_returned"):
                body += ("\n\n'Not returned by TREK' = TREK answered but has no text for "
                         "these ids (deleted, archived, restricted). They are listed in the "
                         "Log and skipped next time.")
            if report.get("failed") or rq.get("failed"):
                body += "\n\nSome requests failed -- run 'Offline' again to retry them."

        self._set_status(
            f"{title}. {report.get('downloaded', 0):,} test-case text(s)"
            + (f" and {rq.get('downloaded', 0)} requirement-module download(s)" if rq else "")
            + " saved.")
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(title)
        box.setInformativeText(body)
        box.exec()

    def _on_offline_download_error(self, msg: str):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_offline.setEnabled(True)
        self._set_status(msg)
        QMessageBox.warning(self, "Offline download failed", msg)

    def _on_index_built(self, report: dict):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_build_index.setEnabled(True)
        syr = report.get("SYR", 0)
        swr = report.get("SWR", 0)
        syt = report.get("SYT", 0)
        swt = report.get("SWT", 0)
        self._set_status(
            f"Index built: {INDEX.size()} ids total "
            f"(SYR +{syr}, SWR +{swr}, SYT +{syt}, SWT +{swt}). "
            f"Traceability is now fast."
        )
        # If Get Test Cases triggered this build, continue loading now.
        pending = getattr(self, "_pending_tc_load", None)
        if pending is not None:
            self._pending_tc_load = None
            self._load_tcs_for_module(force_refresh=pending.get("force_refresh", False))

    def _on_index_error(self, msg: str):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_build_index.setEnabled(True)
        # Drop any pending TC load so the user can retry manually; the slower
        # fallback (probing/name-guessing) still works if they press Get
        # Test Cases again and choose to proceed.
        self._pending_tc_load = None
        self._set_status(msg)
        QMessageBox.warning(self, "Index Build Failed", msg)

    def _on_active_project_switched(self):
        """Reset all Step 1/2/3 state and reload the module list for the
        newly active project. Existing cache data is NOT cleared -- every
        cache key already includes project_id/campaign_id/config_id (see
        trek_cache.py), so switching projects simply starts reading/writing
        a different slice of the cache file now in effect (the app-wide
        default, or this project's own custom database path -- see
        _apply_active_project(), which may have re-pointed the global
        CACHE handle before this method runs)."""
        self._load_syt_prefixes_for_active_project()
        self._refresh_prefix_combo(self._syt_prefixes)
        self._refresh_cache_label()
        self._modules = []
        self._syt_modules = []
        self._selected_mod = None
        self._all_syt_ids = []
        self._syr_candidates = []
        self._swt_candidates = []
        self._all_syr_module_names = []
        self._all_swt_module_names = []
        self._tc_chapters = {}
        self._result = None
        self._unresolved_pairs = []
        self._mod_list.clear()
        self._mod_info.setText("← Pick a module first")
        self._tc_tree.clear()
        self._tree.clear()
        self._detail_syt.clear()
        self._detail_syr.clear()
        self._detail_swr.clear()
        self._detail_swt.clear()
        self._btn_get_tcs.setEnabled(False)
        self._btn_edit_mapping.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_unresolved.setEnabled(False)
        self._btn_check_duplicates.setEnabled(False)
        self._mod_cache_lbl.setText("Modules source: (not loaded)")
        active = PROJECT_STORE.get_active()
        if active:
            self._set_status(
                f"Switched to project '{active['name']}' "
                f"(Project {active['project_id']} · Campaign {active['campaign_id']}). "
                f"Click 'Load Modules' to begin."
            )
        self._fetch_modules(force_refresh=False)

    # --- STEP 1: Module selection -------------------------------------------
    def _make_step1_panel(self):
        self._step1_box = QGroupBox("① Select SYT Module")
        lay = QVBoxLayout(self._step1_box)
        lay.setSpacing(8)

        # Prefix dropdown (checkable combo box for multi-prefix filtering)
        self._prefix_row_widget = QWidget()
        prefix_row = QHBoxLayout(self._prefix_row_widget)
        prefix_row.setContentsMargins(0, 0, 0, 0)
        prefix_lbl = QLabel("Prefixes:")
        prefix_lbl.setStyleSheet("font-size:11px;")
        prefix_row.addWidget(prefix_lbl)
        self._prefix_combo = CheckableComboBox()
        self._prefix_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Show the project's saved prefixes straight away (the full list of
        # available prefixes is filled in once modules are loaded).
        self._prefix_combo.set_items(list(self._syt_prefixes), self._syt_prefixes)
        self._prefix_combo.checkedChanged.connect(self._on_prefix_checked_changed)
        prefix_row.addWidget(self._prefix_combo, 1)
        lay.addWidget(self._prefix_row_widget)

        # Search filter
        self._mod_search = QLineEdit()
        self._mod_search.setPlaceholderText("Filter modules...")
        self._mod_search.textChanged.connect(self._filter_modules)
        lay.addWidget(self._mod_search)

        # Module list
        self._mod_list = QListWidget()
        self._mod_list.setAlternatingRowColors(True)
        self._mod_list.currentRowChanged.connect(self._on_module_selected)
        lay.addWidget(self._mod_list, 1)

        # Cache status label
        self._mod_cache_lbl = QLabel("Modules source: (not loaded)")
        self._mod_cache_lbl.setStyleSheet("font-size:11px;")
        lay.addWidget(self._mod_cache_lbl)

        # Fetch / Refresh / Collapse buttons
        btn_row = QHBoxLayout()
        self._btn_fetch_mods = QPushButton("📂  Load Modules")
        self._btn_fetch_mods.setToolTip("Load from local cache if available, otherwise fetch from TREK.")
        self._btn_fetch_mods.clicked.connect(lambda: self._fetch_modules(force_refresh=False))
        btn_row.addWidget(self._btn_fetch_mods)

        self._btn_refresh_mods = QPushButton("⟳  Force Refresh")
        self._btn_refresh_mods.setObjectName("btn_secondary")
        self._btn_refresh_mods.setToolTip("Force refresh module list from TREK (ignores cache).")
        self._btn_refresh_mods.clicked.connect(lambda: self._fetch_modules(force_refresh=True))
        btn_row.addWidget(self._btn_refresh_mods)
        lay.addLayout(btn_row)

        # Collapse toggle -- hides the module list to free space for Steps 2-3
        self._btn_collapse_step1 = QPushButton("«  Hide")
        self._btn_collapse_step1.setObjectName("btn_secondary")
        self._btn_collapse_step1.setToolTip("Collapse the module list to free space for test cases and results.")
        self._btn_collapse_step1.clicked.connect(self._toggle_step1_panel)
        lay.addWidget(self._btn_collapse_step1)

        # Collapsed state: a single clickable vertical strip with the section
        # name painted top-to-bottom. Clicking ANYWHERE on it expands.
        class _CollapsedStrip(QWidget):
            """Thin clickable strip with vertically-rotated text."""
            clicked = Signal()

            def __init__(self, text, parent=None):
                super().__init__(parent)
                self._text = text
                self.setCursor(Qt.PointingHandCursor)
                self.setToolTip("Click to show the module list")
                self.setFixedWidth(28)

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter, QFontMetrics
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                p.setPen(QColor(ACCENT))
                font = self.font()
                font.setPointSize(9)
                font.setBold(True)
                p.setFont(font)
                # Rotate -90° = text reads top-to-bottom (natural for a
                # left sidebar: eye scans downward).
                p.translate(0, self.height())
                p.rotate(-90)
                p.drawText(0, 0, self.height(), self.width(),
                           Qt.AlignCenter, self._text)
                p.end()

            def mousePressEvent(self, event):
                self.clicked.emit()

            def minimumSizeHint(self):
                from PySide6.QtCore import QSize
                return QSize(28, 80)

            def sizeHint(self):
                from PySide6.QtCore import QSize
                return QSize(28, 200)

        self._step1_collapsed_bar = _CollapsedStrip("»  ① Modules")
        self._step1_collapsed_bar.clicked.connect(self._toggle_step1_panel)
        self._step1_collapsed_bar.hide()
        lay.addWidget(self._step1_collapsed_bar)

        return self._step1_box

    def _toggle_step1_panel(self):
        """Collapse or expand the Step 1 (module list) panel."""
        sizes = self._body_splitter.sizes()
        if self._step1_expanded:
            # Collapse: save current width, hide content, keep a narrow
            # strip (~40px) with the expand button always visible.
            self._step1_saved_width = sizes[0] if sizes[0] > 60 else self._step1_saved_width
            self._prefix_row_widget.hide()
            self._mod_search.hide()
            self._mod_list.hide()
            self._mod_cache_lbl.hide()
            self._btn_fetch_mods.hide()
            self._btn_refresh_mods.hide()
            self._step1_box.setTitle("")
            self._btn_collapse_step1.hide()
            self._step1_collapsed_bar.show()
            # Keep 32px for the collapsed strip, redistribute the rest
            freed = sizes[0] - 32
            if freed < 0:
                freed = 0
            self._body_splitter.setSizes([32, sizes[1] + freed, sizes[2]])
            self._step1_box.setMaximumWidth(32)
            self._step1_expanded = False
        else:
            # Expand: restore saved width and show all content
            self._step1_collapsed_bar.hide()
            self._step1_box.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            restored = self._step1_saved_width or 280
            extra = restored - sizes[0]
            if extra < 0:
                extra = 0
            self._body_splitter.setSizes([restored, sizes[1] - extra, sizes[2]])
            self._prefix_row_widget.show()
            self._mod_search.show()
            self._mod_list.show()
            self._mod_cache_lbl.show()
            self._btn_fetch_mods.show()
            self._btn_refresh_mods.show()
            self._btn_collapse_step1.show()
            self._step1_box.setTitle("① Select SYT Module")
            self._step1_expanded = True

    # --- STEP 2: TC selection -----------------------------------------------
    def _make_step2_panel(self):
        box = QGroupBox("② Select Test Cases")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Module info label
        self._mod_info = QLabel("← Pick a module first")
        self._mod_info.setStyleSheet("font-size:12px;")
        self._mod_info.setWordWrap(True)
        lay.addWidget(self._mod_info)

        # Manual SYR/SWT mapping override -- see ModuleMappingDialog. Only
        # enabled once a module's TCs have been loaded (so auto-detection
        # results, even if empty, exist to pre-populate the dialog with).
        self._btn_edit_mapping = QPushButton("✎  Edit SYR/SWR/SWT Mapping")
        self._btn_edit_mapping.setObjectName("btn_secondary")
        self._btn_edit_mapping.setEnabled(False)
        self._btn_edit_mapping.setToolTip(
            "Manually confirm/correct which SYR, SWR and SWT/SWIT module(s)\n"
            "this SYT module bridges to. Setting the SWR module(s) makes\n"
            "'Build Traceability' MUCH faster: it skips probing dozens of\n"
            "candidate SWR modules live (some huge and unrelated). Also use\n"
            "this when auto-detection finds nothing or the wrong module\n"
            "(e.g. an ID prefix like 'RWW' unrelated to the display name)."
        )
        self._btn_edit_mapping.clicked.connect(self._show_module_mapping_dialog)
        lay.addWidget(self._btn_edit_mapping)

        # "Use cached data" checkbox + explicit "Get Test Cases" action.
        # Selecting a module in Step 1 no longer auto-fetches -- the user
        # decides source (cache vs. a fresh copy from TREK) then triggers
        # the fetch here.
        self._chk_use_cache = QCheckBox("Use cached data")
        self._chk_use_cache.setChecked(True)
        self._chk_use_cache.setToolTip(
            "Checked: load TC IDs / SYR candidates from the local cache when\n"
            "available (fast). Unchecked: always fetch a fresh copy from TREK\n"
            "and overwrite the cache."
        )
        lay.addWidget(self._chk_use_cache)

        self._btn_get_tcs = QPushButton("📥  Get Test Cases")
        self._btn_get_tcs.setEnabled(False)
        self._btn_get_tcs.setToolTip(
            "Load TC IDs for the selected module -- from cache if checked\n"
            "above, otherwise as a fresh copy fetched live from TREK."
        )
        self._btn_get_tcs.clicked.connect(self._get_test_cases_clicked)
        lay.addWidget(self._btn_get_tcs)

        # TC filter
        self._tc_search = QLineEdit()
        self._tc_search.setPlaceholderText("Filter TC IDs or chapter...")
        self._tc_search.textChanged.connect(self._filter_tcs)
        lay.addWidget(self._tc_search)

        # TC tree: grouped by real DOORS chapter (Module_Path), e.g.
        # "TestCaseSpecification.SWUpdate.Intake". Top-level nodes are
        # chapter headers whose checkbox toggles every TC beneath them;
        # each TC is a checkable leaf.
        tc_tree_btn_row = QHBoxLayout()
        tc_tree_btn_row.setSpacing(4)
        btn_tc_expand = QPushButton("+  Expand All")
        btn_tc_expand.setObjectName("btn_secondary")
        btn_tc_expand.setToolTip("Expand every chapter to show its test cases.")
        btn_tc_expand.clicked.connect(lambda: self._tc_tree.expandAll())
        tc_tree_btn_row.addWidget(btn_tc_expand)
        btn_tc_collapse = QPushButton("−  Collapse All")
        btn_tc_collapse.setObjectName("btn_secondary")
        btn_tc_collapse.setToolTip("Collapse every chapter -- selections are kept.")
        btn_tc_collapse.clicked.connect(lambda: self._tc_tree.collapseAll())
        tc_tree_btn_row.addWidget(btn_tc_collapse)
        tc_tree_btn_row.addStretch()
        lay.addLayout(tc_tree_btn_row)

        self._tc_tree = QTreeWidget()
        self._tc_tree.setHeaderLabels(["TC ID / Chapter"])
        self._tc_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tc_tree.setAlternatingRowColors(True)
        lay.addWidget(self._tc_tree, 1)

        # Selection buttons row
        btn_row = QHBoxLayout()
        self._btn_all  = QPushButton("All")
        self._btn_none = QPushButton("None")
        self._btn_all.setObjectName("btn_secondary")
        self._btn_none.setObjectName("btn_secondary")
        self._btn_all.setToolTip("Select ALL test cases in the list for the traceability build.")
        self._btn_none.setToolTip("Deselect all test cases.")
        self._btn_all.clicked.connect(self._select_all_tcs)
        self._btn_none.clicked.connect(self._select_none_tcs)
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)

        # Random N
        self._spin_random = QSpinBox()
        self._spin_random.setRange(1, 9999)
        self._spin_random.setValue(10)
        self._spin_random.setFixedWidth(60)
        self._spin_random.setToolTip("How many test cases the 'Random' button selects.")
        # Styled via the global QSpinBox QSS rule in trek_theme (no baked colours).
        self._btn_random = QPushButton("Random")
        self._btn_random.setObjectName("btn_secondary")
        self._btn_random.setToolTip(
            "Randomly select N test cases (N from the box on the left).\n"
            "Handy for a quick sample run instead of the whole module."
        )
        self._btn_random.clicked.connect(self._select_random_tcs)
        btn_row.addWidget(self._spin_random)
        btn_row.addWidget(self._btn_random)
        lay.addLayout(btn_row)

        # Selected count label
        self._sel_count_lbl = QLabel("0 selected")
        self._sel_count_lbl.setStyleSheet("font-size:11px;")
        lay.addWidget(self._sel_count_lbl)

        # Force-refresh option for the traceability build
        self._chk_force_refresh = QCheckBox("Force refresh from TREK (ignore cache)")
        self._chk_force_refresh.setToolTip(
            "When checked, Build Traceability re-fetches links and test-case\n"
            "content live from TREK and overwrites the local cache, instead of\n"
            "reusing previously cached results."
        )
        lay.addWidget(self._chk_force_refresh)

        # Run button
        self._btn_run = QPushButton("▶  Build Traceability")
        self._btn_run.setEnabled(False)
        self._btn_run.setToolTip(
            "Build the full SYT → SYR → SWR → SWT traceability chain for the\n"
            "selected test cases. Resolves every requirement/test module via\n"
            "the id→module index (build it once when prompted). Enable after\n"
            "'Get Test Cases' and selecting at least one TC."
        )
        self._btn_run.clicked.connect(self._run_traceability)
        lay.addWidget(self._btn_run)

        return box

    # --- STEP 3: Results ----------------------------------------------------
    def _make_step3_panel(self):
        box = QGroupBox("③ Traceability Results")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        # Summary bar
        self._summary_lbl = QLabel("Run traceability to see results.")
        self._summary_lbl.setStyleSheet("font-size:12px;padding:4px 0;")
        lay.addWidget(self._summary_lbl)

        # Horizontal splitter: tree (left) | detail tabs (right)
        splitter = QSplitter(Qt.Horizontal)

        # Left: traceability tree
        tree_frame = QGroupBox("Chain  SYT → SYR → SWR → SWT")
        tree_lay   = QVBoxLayout(tree_frame)

        tree_btn_row = QHBoxLayout()
        btn_expand_all = QPushButton("+  Expand All")
        btn_expand_all.setObjectName("btn_secondary")
        btn_expand_all.setToolTip(
            "Expand every node of the traceability tree.\n"
            "(For very large results this is disabled to keep the UI responsive.)"
        )
        btn_expand_all.clicked.connect(self._expand_all_tree)
        tree_btn_row.addWidget(btn_expand_all)
        btn_collapse_all = QPushButton("−  Collapse All")
        btn_collapse_all.setObjectName("btn_secondary")
        btn_collapse_all.setToolTip("Collapse the traceability tree back to the top-level SYT nodes.")
        btn_collapse_all.clicked.connect(lambda: self._tree.collapseAll())
        tree_btn_row.addWidget(btn_collapse_all)
        tree_btn_row.addStretch()
        tree_lay.addLayout(tree_btn_row)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["ID", "Type", "Info"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self._tree.header().setSectionResizeMode(1, QHeaderView.Interactive)
        self._tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self._tree.setColumnWidth(0, 220)
        self._tree.setColumnWidth(1, 110)
        # Big modules produce tens of thousands of nodes (a single SYR can
        # be shared by 200+ related SYTs) -- ResizeToContents would re-measure
        # every one of them on each insert/expand, and non-uniform row
        # heights force a per-row layout pass. Both are O(n) per operation
        # and froze the UI outright on 'SYT - Infrastructure'.
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
        self._tree.itemExpanded.connect(self._on_tree_item_expanded)
        tree_lay.addWidget(self._tree)
        splitter.addWidget(tree_frame)

        # Right: detail tabs -- scrollable so all 5 tabs are reachable on
        # small/laptop screens where the tab bar would otherwise clip.
        self._detail_tabs = QTabWidget()
        self._detail_tabs.setTabBarAutoHide(False)
        self._detail_tabs.setUsesScrollButtons(True)
        self._detail_tabs.setElideMode(Qt.ElideNone)

        self._detail_syt = QTextEdit()
        self._detail_syt.setReadOnly(True)
        self._detail_tabs.addTab(self._detail_syt, "SYT Content")

        self._detail_syr = QTextEdit()
        self._detail_syr.setReadOnly(True)
        self._detail_tabs.addTab(self._detail_syr, "SYR Requirement")

        self._detail_swr = QTextEdit()
        self._detail_swr.setReadOnly(True)
        self._detail_tabs.addTab(self._detail_swr, "SWR Requirement")

        self._detail_swt = QTextEdit()
        self._detail_swt.setReadOnly(True)
        self._detail_tabs.addTab(self._detail_swt, "SWT Content")

        self._detail_related_syt = QTextEdit()
        self._detail_related_syt.setReadOnly(True)
        self._detail_tabs.addTab(self._detail_related_syt, "Related SYT")

        splitter.addWidget(self._detail_tabs)
        splitter.setSizes([420, 580])
        lay.addWidget(splitter, 1)

        # Export + duplicate-check buttons. "Check Duplicates" opens
        # DuplicateCheckSettingsDialog (BM25/vector weighting + similarity
        # thresholds for this run) and, on completion, automatically opens
        # DuplicateResultsDialog -- a dedicated side-by-side results window
        # (see _on_duplicate_check_ready) rather than an in-panel tab.
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_check_duplicates = QPushButton("🔁  Check Duplicates")
        self._btn_check_duplicates.setObjectName("btn_secondary")
        self._btn_check_duplicates.setEnabled(False)
        self._btn_check_duplicates.setToolTip(
            "Compare each SYT test case against its linked SWT test case(s)\n"
            "using hybrid BM25 + embedding similarity, to flag likely\n"
            "copy-pasted / redundant test coverage. Requires this project's\n"
            "JWT Token (set in Edit Project)."
        )
        self._btn_check_duplicates.clicked.connect(self._run_duplicate_check)
        btn_row.addWidget(self._btn_check_duplicates)

        self._btn_export = QPushButton("💾  Export JSON")
        self._btn_export.setObjectName("btn_success")
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip(
            "Export the full traceability result (SYT→SYR→SWR→SWT chain,\n"
            "counts, and content) for the selected test cases to a JSON file.\n"
            "Enabled after a traceability build completes."
        )
        self._btn_export.clicked.connect(self._export_json)
        btn_row.addWidget(self._btn_export)

        self._btn_unresolved = QPushButton("⚠️  Unresolved Pairs")
        self._btn_unresolved.setObjectName("btn_secondary")
        self._btn_unresolved.setEnabled(False)
        self._btn_unresolved.setToolTip(
            "Pairs that have a link in TREK but no usable content was\n"
            "returned for the counterpart -- excluded from Check Duplicates."
        )
        self._btn_unresolved.clicked.connect(self._show_unresolved_pairs)
        btn_row.addWidget(self._btn_unresolved)
        lay.addLayout(btn_row)

        return box

    # -----------------------------------------------------------------------
    # Step 1 logic
    # -----------------------------------------------------------------------
    def _fetch_modules(self, force_refresh: bool = False):
        self._btn_fetch_mods.setEnabled(False)
        self._btn_refresh_mods.setEnabled(False)
        self._progress.setVisible(True)
        self._busy_start(
            "Refreshing module list from TREK..." if force_refresh
            else "Loading module list (cache if available)..."
        )

        worker = FetchModulesWorker(force_refresh=force_refresh)
        worker.done.connect(self._on_modules_fetched)
        worker.error.connect(self._on_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_modules_fetched(self, modules: list, from_cache: bool, updated_at: str):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_fetch_mods.setEnabled(True)
        self._btn_refresh_mods.setEnabled(True)

        self._modules = modules
        # Auto-detect which prefixes are present in this project's modules
        # (only SY*/SW* style prefixes) and refresh the prefix combo so the
        # user can toggle which ones are visible in the module list.
        detected = self._detect_syt_prefixes(modules)
        self._refresh_prefix_combo(detected)
        self._apply_syt_prefix_filter()
        age = trek_cache.format_age(updated_at)
        source = f"cache ({age})" if from_cache else "TREK (live)"
        self._mod_cache_lbl.setText(f"Modules source: {source}")
        self._set_status(f"Loaded {len(self._syt_modules)} SYT modules from {source}.")

    def _apply_syt_prefix_filter(self):
        """Rebuild self._syt_modules from self._modules using the currently
        checked prefixes in the prefix combo, then repopulate the list."""
        checked = self._get_checked_prefixes()
        if not checked:
            checked = list(DEFAULT_SYT_PREFIXES)
        self._syt_prefixes = checked
        self._syt_modules = [
            m for m in self._modules
            if m.get("Total", 0) > 0 and _is_syt_module(m.get("Name", ""), prefixes=checked)
        ]
        # Keep whatever is typed in "Filter modules..." applied.
        self._filter_modules(self._mod_search.text() if hasattr(self, "_mod_search") else "")

    def _detect_syt_prefixes(self, modules) -> List[str]:
        """Auto-detect SY*/SW*-style prefixes present in the module list."""
        prefixes: List[str] = []
        for m in modules:
            name = m.get("Name", "")
            if not name:
                continue
            first = re.split(r"[ _\-]", name.strip(), maxsplit=1)[0].upper()
            if first and (first.startswith("SY") or first.startswith("SW")):
                if first not in prefixes:
                    prefixes.append(first)
        return sorted(prefixes) if prefixes else list(DEFAULT_SYT_PREFIXES)

    def _refresh_prefix_combo(self, available: List[str]):
        """Fill the prefix dropdown with *available* prefixes (plus any saved
        ones not currently detected, so a saved choice is never silently
        dropped), checking those in self._syt_prefixes."""
        items = list(available)
        for p in self._syt_prefixes:
            if p not in items:
                items.append(p)
        self._prefix_combo.set_items(sorted(items), self._syt_prefixes)

    def _get_checked_prefixes(self) -> List[str]:
        return self._prefix_combo.checked_items()

    def _on_prefix_checked_changed(self, checked: List[str]):
        """A prefix was ticked/unticked: keep at least one selected, save the
        choice to the active project, and re-filter the module list."""
        if not checked:
            # Never leave the list empty by accident -- restore the last one.
            for p in self._syt_prefixes:
                self._prefix_combo.set_checked(p, True)
            self._set_status("At least one prefix must stay selected.")
            return
        self._apply_syt_prefix_filter()
        active = PROJECT_STORE.get_active()
        if active:
            try:
                PROJECT_STORE.set_syt_prefixes(active["key"], ", ".join(checked))
            except Exception as exc:   # noqa: BLE001
                LOG.log("Modules", f"Could not save SYT prefixes: {exc}", level="WARN")

    def _load_syt_prefixes_for_active_project(self):
        """Load syt_prefixes from the active project's settings, falling
        back to DEFAULT_SYT_PREFIXES if none are stored."""
        active = PROJECT_STORE.get_active()
        if active:
            raw = active.get("syt_prefixes", "")
            if raw:
                self._syt_prefixes = _parse_syt_prefixes(raw)
                return
        self._syt_prefixes = list(DEFAULT_SYT_PREFIXES)

    def _populate_module_list(self, modules):
        self._mod_list.clear()
        for m in modules:
            name  = m["Name"]
            total = m.get("Total", 0)
            item  = QListWidgetItem(f"{name}  [{total} TCs]")
            item.setData(Qt.UserRole, m)
            self._mod_list.addItem(item)

    def _filter_modules(self, text):
        filtered = [m for m in self._syt_modules
                    if text.lower() in m["Name"].lower()]
        self._populate_module_list(filtered)

    def _on_module_selected(self, row):
        if row < 0:
            return
        item = self._mod_list.item(row)
        if not item:
            return
        self._selected_mod = item.data(Qt.UserRole)

        # Selecting a module no longer auto-fetches anything -- it just
        # shows the module info and arms the "Get Test Cases" button. The
        # user decides (via the checkbox) whether to read cached data or
        # force a fresh copy from TREK, then clicks the button explicitly.
        mod  = self._selected_mod
        name = mod["Name"]
        tc_count = mod.get("Total", 0)
        self._mod_info.setText(f"{name}\n{tc_count} test cases (in TREK)\n← Click 'Get Test Cases' to load")
        self._btn_run.setEnabled(False)
        self._tc_tree.clear()
        self._all_syt_ids = []
        self._syr_candidates = []
        self._swt_candidates = []
        self._btn_get_tcs.setEnabled(True)
        self._btn_edit_mapping.setEnabled(False)

    # -----------------------------------------------------------------------
    # Step 2 logic
    # -----------------------------------------------------------------------
    def _get_test_cases_clicked(self):
        """Handler for the '📥 Get Test Cases' button. Whether this reads
        the local cache or forces a fresh copy from TREK is controlled by
        the 'Use cached data' checkbox.

        If the id->module INDEX has not been built yet for this project, build
        it FIRST (once), then continue to load the test cases automatically.
        The index is what lets SYR/SWR/SWT modules be resolved by lookup
        instead of slow live probing/name-guessing, and SYR resolution
        already happens here at Get Test Cases -- so building it now, up
        front, is the right moment."""
        if not self._selected_mod:
            return
        if self._wait_for_index(self._get_test_cases_clicked, "Get Test Cases"):
            return
        use_cache = self._chk_use_cache.isChecked()
        force_refresh = not use_cache

        # Build the index first if it's missing, then chain the TC load.
        if not (INDEX.is_built("SYR") and INDEX.is_built("SWR")
                and INDEX.is_built("SWT")):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Build Index First — this can take a while")
            box.setText(
                "The object-id → module index for this project has not been "
                "built yet."
            )
            box.setInformativeText(
                "It will be built now as a ONE-TIME step (then cached and "
                "reused every time).\n\n"
                "⏳ This can take SEVERAL MINUTES the first time, because it "
                "reads every SYR / SWR / SYT / SWT module from TREK.\n\n"
                "Why it's worth it: afterwards, traceability resolves every "
                "module by instant lookup instead of slow live probing.\n\n"
                "You can watch progress in the bar at the bottom. Your test "
                "cases will load automatically as soon as the index is ready.\n\n"
                "Build the index now?"
            )
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            box.button(QMessageBox.Ok).setText("Build Index Now")
            box.setDefaultButton(QMessageBox.Ok)
            if box.exec() != QMessageBox.Ok:
                return
            # Remember to load these TCs once the index build completes.
            self._pending_tc_load = {"force_refresh": force_refresh}
            self._build_index()
            return

        self._load_tcs_for_module(force_refresh=force_refresh)

    def _load_tcs_for_module(self, force_refresh: bool = False):
        mod  = self._selected_mod
        name = mod["Name"]
        tc_count = mod.get("Total", 0)

        self._mod_info.setText(f"{name}\n{tc_count} test cases")
        self._btn_run.setEnabled(False)
        self._btn_get_tcs.setEnabled(False)
        self._tc_tree.clear()
        self._all_syt_ids = []
        self._syr_candidates = []
        self._swt_candidates = []

        # Reset Step 3 (traceability) results entirely -- switching/reloading
        # the Step 2 module must never leave stale results/stats from a
        # PREVIOUS module's "Build Traceability" run visible on screen.
        self._result = None
        self._unresolved_pairs = []
        self._last_dup_result = None
        self._tree.clear()
        self._summary_lbl.setText("Run traceability to see results.")
        self._detail_syt.clear()
        self._detail_syr.clear()
        self._detail_swr.clear()
        self._detail_swt.clear()
        self._detail_related_syt.clear()
        self._btn_export.setEnabled(False)
        self._btn_unresolved.setEnabled(False)
        self._btn_check_duplicates.setEnabled(False)

        self._busy_start(f"{'Refreshing' if force_refresh else 'Loading'} links for '{name}'...")
        self._progress.setVisible(True)

        subsystem = name.replace("SYT -", "").strip()

        # Keep the full candidate lists around as instance state -- both
        # for FetchSytTcsWorker's SYR detection below, and so
        # _show_module_mapping_dialog() can offer the complete real module
        # lists to pick from, not just whatever auto-detection guessed.
        subsystem_norm = _normalize_subsystem(subsystem)
        all_names = [m["Name"] for m in self._modules]
        self._all_syr_module_names = [
            n for n in all_names
            if (n.startswith("SYR") or n in _EXTRA_SYR_LIKE_MODULES)
            and "Review" not in n
            and n not in _SYR_EXCLUDED_MODULES
        ]
        self._all_swt_module_names = [
            n for n in all_names
            if (n.startswith("SWT") or n.startswith("SWIT")) and "Review" not in n and "Template" not in n
        ]

        # SWT/SWIT: check for a user-confirmed manual mapping first (see
        # ModuleMappingDialog) -- it always takes priority. Otherwise fall
        # back to normalized substring match (letters/digits only, no
        # spaces/underscores/hyphens) because TREK module names are not
        # consistent about word separators, e.g. "SYT - Rear Window
        # Heating" vs. "SWT - RearWindowHeating". This heuristic can still
        # miss cases with no textual relationship at all (the same failure
        # mode as SYR detection) -- "Edit SYR/SWT Mapping" lets the user
        # correct those.
        manual_swt = get_manual_bridge_mapping(name, "SWT")
        if manual_swt is not None:
            self._swt_candidates = manual_swt
        else:
            self._swt_candidates = [
                n for n in self._all_swt_module_names
                if subsystem_norm in _normalize_subsystem(n)
            ]

        # SYR is NOT detected by guessing a name from the SYT subsystem --
        # that assumption doesn't hold in TREK (e.g. "SYT - SWUpdate" links
        # to SYR requirements filed under an unrelated "Infrastructure"
        # functional area, and "SYT - Rear Wiper" references a "RWW"
        # abbreviation with no textual relationship to any module name at
        # all). Instead FetchSytTcsWorker derives the real SYR module from
        # the SYT module's own OUT link data (ground truth) via
        # _get_bridge_modules_cached(), which also checks for a
        # user-confirmed manual mapping first -- so it just needs the full
        # list of real SYR module names to check auto-detected candidates
        # against.
        self._fetch_syt_links_and_detect_syr(name, self._all_syr_module_names, force_refresh)

    def _fetch_syt_links_and_detect_syr(self, syt_module, syr_module_names, force_refresh: bool = False):
        """Fetch SYT links to get TC IDs, then auto-detect the real SYR
        bridge module(s) from the actual OUT link data (see
        FetchSytTcsWorker docstring for why this replaces name-guessing)."""
        worker = FetchSytTcsWorker(syt_module, syr_module_names, force_refresh)
        worker.progress.connect(self._busy_stage_update)
        worker.done.connect(self._on_syt_ids_fetched)
        worker.error.connect(self._on_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_syt_ids_fetched(self, syt_ids, valid_syr, chapter_by_tc_id, syr_ids_by_tc):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_get_tcs.setEnabled(True)
        self._btn_edit_mapping.setEnabled(True)
        self._all_syt_ids    = syt_ids
        self._syr_candidates = valid_syr
        self._tc_chapters    = chapter_by_tc_id

        self._populate_tc_tree(syt_ids, chapter_by_tc_id, syr_ids_by_tc)

        # Update the left panel's module list item with the REAL TC count
        # (from /Export/Links) instead of the inflated Total from /Modules
        # (which includes headings, comments, deleted items, etc.).
        mod_name = self._selected_mod["Name"]
        current_row = self._mod_list.currentRow()
        if current_row >= 0:
            list_item = self._mod_list.item(current_row)
            if list_item:
                list_item.setText(f"{mod_name}  [{len(syt_ids)} TCs]")

        syr_str  = ", ".join(self._syr_candidates) if self._syr_candidates else "(none found)"
        self._mod_info.setText(
            f"{mod_name}\n"
            f"{len(syt_ids)} TCs found\n"
            f"SYR: {syr_str}"
        )
        self._btn_run.setEnabled(len(syt_ids) > 0)
        self._update_sel_count()
        not_dl = sum(1 for c in chapter_by_tc_id.values() if c == NOT_DOWNLOADED_CHAPTER)
        self._set_status(
            f"{len(syt_ids)} TCs loaded. "
            f"SYR: {self._syr_candidates or 'none'}."
            + (f"  ⚠️ TREK unreachable: {not_dl} TC(s) without text (not downloaded yet)."
               if not_dl else "")
        )

        # Auto-detection found nothing for SYR (the failure mode ID-prefix
        # abbreviations like "RWW" for "Rear Wiper" cause, since they have
        # no textual relationship to any module's display name at all) --
        # prompt the user to confirm the mapping manually right away
        # instead of silently letting "Build Traceability" fail later.
        if not self._syr_candidates and syt_ids:
            self._show_module_mapping_dialog()

    def _show_module_mapping_dialog(self):
        if not self._selected_mod:
            return
        mod_name = self._selected_mod["Name"]

        # All real SWR modules (exclude Review copies), for the SWR list.
        all_swr_names = [
            n for n in (m["Name"] for m in self._modules)
            if n.startswith("SWR") and "Review" not in n
        ]
        # Pre-check the SWR list with any SWR mapping already saved for this
        # SYT's SYR module(s) -- the SWR bridge is keyed PER SYR module (see
        # _get_bridge_modules_cached), so union whatever is stored for each.
        auto_swr: List[str] = []
        for syr_mod in self._syr_candidates:
            existing = get_manual_bridge_mapping(syr_mod, "SWR")
            if existing:
                for name in existing:
                    if name not in auto_swr:
                        auto_swr.append(name)

        dlg = ModuleMappingDialog(
            syt_module=mod_name,
            all_syr_names=getattr(self, "_all_syr_module_names", []),
            all_swt_names=getattr(self, "_all_swt_module_names", []),
            auto_syr=self._syr_candidates,
            auto_swt=self._swt_candidates,
            all_swr_names=all_swr_names,
            auto_swr=auto_swr,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            selected_syr = dlg.get_selected_syr()
            selected_swr = dlg.get_selected_swr()
            selected_swt = dlg.get_selected_swt()
            set_manual_bridge_mapping(mod_name, "SYR", selected_syr)
            set_manual_bridge_mapping(mod_name, "SWT", selected_swt)
            # Save the SWR bridge for EACH SYR module of this SYT, since the
            # traceability build looks up the SWR bridge per SYR module. Only
            # save when the user actually picked SWR module(s); an empty
            # selection leaves auto-detection in charge (avoids accidentally
            # pinning "no SWR" and breaking the chain).
            if selected_swr:
                for syr_mod in selected_syr:
                    set_manual_bridge_mapping(syr_mod, "SWR", selected_swr)
            self._syr_candidates = selected_syr
            self._swt_candidates = selected_swt

            syr_str = ", ".join(selected_syr) if selected_syr else "(none)"
            swr_str = ", ".join(selected_swr) if selected_swr else "(auto)"
            swt_str = ", ".join(selected_swt) if selected_swt else "(none)"
            self._mod_info.setText(
                f"{mod_name}\n"
                f"{len(self._all_syt_ids)} TCs found\n"
                f"SYR: {syr_str}  (manual)\n"
                f"SWR: {swr_str}\n"
                f"SWT: {swt_str}  (manual)"
            )
            self._btn_run.setEnabled(len(self._all_syt_ids) > 0)
            self._set_status(f"Saved manual SYR/SWR/SWT mapping for '{mod_name}'.")

    def _populate_tc_tree(self, syt_ids, chapter_by_tc_id, syr_ids_by_tc=None):
        """Rebuild the Step 2 TC tree, grouping TCs by their real DOORS
        chapter (Module_Path, e.g. "TestCaseSpecification.SWUpdate.Intake").
        Chapter nodes are non-checkable group headers; each TC is a
        checkable leaf under its chapter. TCs with unknown chapter (e.g.
        content fetch failed for that one id) fall under "(no chapter)".

        Disconnects itemChanged first so refreshing the same module doesn't
        stack duplicate connections (tracked via an explicit flag rather
        than try/except disconnect(), because PySide6 prints a
        RuntimeWarning to stderr on a no-op disconnect instead of raising
        a catchable exception).
        """
        if self._tc_table_signal_connected:
            self._tc_tree.itemChanged.disconnect(self._on_tc_item_changed)
            self._tc_table_signal_connected = False

        self._tc_tree.clear()
        chapter_nodes: Dict[str, QTreeWidgetItem] = {}

        for tc_id in syt_ids:
            chapter = chapter_by_tc_id.get(tc_id, "(no chapter)")
            chapter_node = chapter_nodes.get(chapter)
            if chapter_node is None:
                chapter_node = QTreeWidgetItem([chapter])
                chapter_node.setFlags(chapter_node.flags() | Qt.ItemIsUserCheckable)
                chapter_node.setCheckState(0, Qt.Unchecked)
                chapter_node.setFont(0, QFont("Segoe UI", 9, QFont.Bold))
                chapter_node.setForeground(0, QColor(ACCENT))
                chapter_node.setToolTip(0, "Check to select every test case in this chapter.")
                self._tc_tree.addTopLevelItem(chapter_node)
                chapter_nodes[chapter] = chapter_node

            leaf = QTreeWidgetItem([tc_id])
            leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.Unchecked)
            leaf.setFont(0, QFont("Consolas", 10))
            chapter_node.addChild(leaf)

        for chapter, node in chapter_nodes.items():
            node.setText(0, f"{chapter}  [{node.childCount()} TCs]")

        self._tc_tree.expandAll()
        self._tc_tree.itemChanged.connect(self._on_tc_item_changed)
        self._tc_table_signal_connected = True

    def _on_tc_item_changed(self, item: QTreeWidgetItem, column: int):
        """Cascade a chapter checkbox down to its test cases, and reflect a
        leaf change back up as checked/unchecked/partially-checked."""
        self._tc_tree.itemChanged.disconnect(self._on_tc_item_changed)
        try:
            if item.parent() is None:
                state = item.checkState(0)
                if state != Qt.PartiallyChecked:
                    for j in range(item.childCount()):
                        leaf = item.child(j)
                        if not leaf.isHidden():
                            leaf.setCheckState(0, state)
            else:
                self._sync_chapter_check_state(item.parent())
        finally:
            self._tc_tree.itemChanged.connect(self._on_tc_item_changed)
        self._update_sel_count()

    @staticmethod
    def _sync_chapter_check_state(chapter_node: QTreeWidgetItem):
        total = chapter_node.childCount()
        checked = sum(chapter_node.child(j).checkState(0) == Qt.Checked for j in range(total))
        if checked == 0:
            chapter_node.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            chapter_node.setCheckState(0, Qt.Checked)
        else:
            chapter_node.setCheckState(0, Qt.PartiallyChecked)

    def _iter_tc_leaves(self):
        """Yield every TC leaf item across all chapter groups in the tree."""
        for i in range(self._tc_tree.topLevelItemCount()):
            chapter_node = self._tc_tree.topLevelItem(i)
            for j in range(chapter_node.childCount()):
                yield chapter_node.child(j)

    def _filter_tcs(self, text):
        text = text.lower().strip()
        for i in range(self._tc_tree.topLevelItemCount()):
            chapter_node = self._tc_tree.topLevelItem(i)
            chapter_matches = text in chapter_node.text(0).lower()
            any_child_visible = False
            for j in range(chapter_node.childCount()):
                leaf = chapter_node.child(j)
                visible = chapter_matches or (text in leaf.text(0).lower())
                leaf.setHidden(not visible)
                any_child_visible = any_child_visible or visible
            chapter_node.setHidden(not any_child_visible)

    def _select_all_tcs(self):
        self._set_leaf_check_states(lambda leaf: Qt.Checked if not leaf.isHidden() else leaf.checkState(0))

    def _select_none_tcs(self):
        self._set_leaf_check_states(lambda leaf: Qt.Unchecked)

    def _select_random_tcs(self):
        n = self._spin_random.value()
        visible_leaves = [leaf for leaf in self._iter_tc_leaves() if not leaf.isHidden()]
        chosen = set(random.sample(visible_leaves, min(n, len(visible_leaves))))
        self._set_leaf_check_states(lambda leaf: Qt.Checked if leaf in chosen else Qt.Unchecked)

    def _set_leaf_check_states(self, state_for):
        """Apply state_for(leaf) to every TC leaf with the itemChanged signal
        muted, then refresh chapter tri-states and the selection count once."""
        self._tc_tree.itemChanged.disconnect(self._on_tc_item_changed)
        try:
            for leaf in self._iter_tc_leaves():
                leaf.setCheckState(0, state_for(leaf))
            for i in range(self._tc_tree.topLevelItemCount()):
                self._sync_chapter_check_state(self._tc_tree.topLevelItem(i))
        finally:
            self._tc_tree.itemChanged.connect(self._on_tc_item_changed)
        self._update_sel_count()

    def _update_sel_count(self):
        count = sum(1 for leaf in self._iter_tc_leaves() if leaf.checkState(0) == Qt.Checked)
        self._sel_count_lbl.setText(f"{count} selected")
        self._btn_run.setEnabled(count > 0)

    def _get_selected_tc_ids(self) -> List[str]:
        return [leaf.text(0) for leaf in self._iter_tc_leaves() if leaf.checkState(0) == Qt.Checked]

    # -----------------------------------------------------------------------
    # Step 3 logic
    # -----------------------------------------------------------------------
    def _run_traceability(self):
        selected_ids = self._get_selected_tc_ids()
        if not selected_ids:
            self._set_status("No TCs selected.")
            return
        if self._wait_for_index(self._run_traceability, "Build Traceability"):
            return

        # Make sure the in-memory INDEX reflects the latest built/cached data
        # before the worker reads it. The build may have run in a separate
        # step/worker; reloading here guarantees resolve() sees all ids
        # instead of silently falling back to slow candidate probing.
        if not INDEX.is_built("SWR"):
            INDEX.load()
        LOG.log("Index", f"INDEX ready for traceability: {INDEX.size()} ids "
                         f"(built: {sorted(INDEX._built_kinds)})")

        if not self._syr_candidates:
            QMessageBox.warning(
                self, "No SYR Module",
                "No SYR module was found for this subsystem.\n"
                "The traceability chain requires a SYR bridge module\n"
                "(SYR carries the OUT link to SWR).\n\n"
                "Please verify the module naming in TREK."
            )
            return

        self._btn_run.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_check_duplicates.setEnabled(False)
        self._progress.setVisible(True)
        self._tree.clear()
        self._detail_syt.clear()
        self._detail_syr.clear()
        self._detail_swr.clear()
        self._detail_swt.clear()

        force_refresh = self._chk_force_refresh.isChecked()
        all_swr_module_names = [
            n for n in (m["Name"] for m in self._modules)
            if n.startswith("SWR") and "Review" not in n
        ]
        self._busy_start("Starting traceability build...")
        worker = FetchLinksWorker(
            syt_module           = self._selected_mod["Name"],
            syr_modules          = self._syr_candidates,
            swt_modules          = self._swt_candidates,
            selected_tc_ids      = selected_ids,
            all_swr_module_names = all_swr_module_names,
            force_refresh        = force_refresh,
        )
        worker.progress.connect(self._busy_stage_update)
        worker.done.connect(self._on_results_ready)
        worker.error.connect(self._on_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    # Color map: tree item type -> theme colour constant name.
    _TREE_TYPE_COLORS = {
        "SYT": "ACCENT",
        "SYR": "REQ_COLOR",
        "SWR": "SWR_COLOR",
    }

    def _recolor_tree_items(self, parent):
        """Walk tree items and re-apply foreground colours from current
        theme constants, preserving expand/collapse state and selection."""
        for i in range(parent.childCount()):
            item = parent.child(i)
            data = item.data(0, Qt.UserRole)
            if isinstance(data, dict):
                item_type = data.get("type", "")
                if item_type == "SYT":
                    item.setForeground(0, QColor(ACCENT))
                elif item_type == "SYR":
                    item.setForeground(0, QColor(REQ_COLOR))
                    item.setForeground(2, QColor(REQ_COLOR))
                elif item_type == "SWR":
                    item.setForeground(0, QColor(SWR_COLOR))
                elif item_type == "SWT":
                    # SWT items: green if resolved, red if not found
                    note = item.text(2)
                    not_found = bool(note)
                    item.setForeground(0, QColor(DANGER_TEXT if not_found else SUCCESS_TEXT))
                    if not_found:
                        item.setForeground(2, QColor(DANGER_TEXT))
                elif item_type == "related_syt":
                    note = item.text(2)
                    not_found = bool(note)
                    item.setForeground(0, QColor(DANGER_TEXT if not_found else RELATED_COLOR))
                    if not_found:
                        item.setForeground(2, QColor(DANGER_TEXT))
            # Recurse into children
            if item.childCount() > 0:
                self._recolor_tree_items(item)

    def _on_results_ready(self, result: dict):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._result = result
        self._btn_check_duplicates.setEnabled(True)

        summary = result["summary"]
        no_syr_count = summary.get("tcs_without_syr_link", 0)
        no_syr_part = (
            f'  |  <span style="color:{DANGER_TEXT};font-weight:bold">'
            f"🚫 {no_syr_count} TC(s) with NO SYR link (=> no SWR/SWT possible)</span>"
            if no_syr_count else ""
        )
        # "Resolved" pairs are what "Check Duplicates" will actually build
        # -- can be LESS than the raw ID-based pair count if TREK returned
        # no content for some SWT/related-SYT id (see FetchLinksWorker).
        swt_potential = summary.get("syt_swt_pair_count", 0)
        swt_resolved = summary.get("syt_swt_pair_count_resolved", swt_potential)
        related_potential = summary.get("syt_syt_pair_count", 0)
        related_resolved = summary.get("syt_syt_pair_count_resolved", related_potential)
        missing_total = (swt_potential - swt_resolved) + (related_potential - related_resolved)
        missing_part = (
            f'  <span style="color:{AMBER_COLOR}">'
            f"(⚠️ {missing_total} pair(s) excluded -- content not resolved for at least one side)</span>"
            if missing_total else ""
        )
        not_dl = summary.get("tc_not_downloaded", 0)
        if not_dl:
            missing_part += (
                f'<br><span style="color:{AMBER_COLOR};font-weight:bold">'
                f"⏳ TREK unreachable: {not_dl} test case(s) not downloaded -- "
                f"shown without text. Run again while connected.</span>"
            )

        # Actual (syt_id, missing_counterpart_id) list behind missing_total
        # above -- each SWT/related-SYT id that appears in the raw link
        # list but has no matching entry in swt_content/related_syt_content
        # (see FetchLinksWorker step 6).
        self._unresolved_pairs: List[dict] = []
        for row in result["traceability"]:
            resolved_swt_ids = {c.get("Key") for c in row.get("swt_content", [])}
            for swt_id in row["swt_ids"]:
                if swt_id not in resolved_swt_ids:
                    self._unresolved_pairs.append({
                        "syt_id": row["syt_id"], "counterpart_id": swt_id, "counterpart_type": "SWT",
                    })
            resolved_related_ids = {c.get("Key") for c in row.get("related_syt_content", [])}
            for related_id in row["related_syt_ids"]:
                if related_id not in resolved_related_ids:
                    self._unresolved_pairs.append({
                        "syt_id": row["syt_id"], "counterpart_id": related_id, "counterpart_type": "related_syt",
                    })

        self._summary_lbl.setTextFormat(Qt.RichText)
        self._summary_lbl.setText(
            f"📋 {summary['selected_tc_count']} SYT selected  |  "
            f"🔗 {summary.get('unique_syr_ids_found', 0)} SYR, "
            f"{summary.get('unique_swr_ids_found', 0)} SWR, "
            f"{summary.get('unique_swt_ids_found', 0)} SWT, "
            f"{summary.get('related_syt_ids_found', 0)} related SYT"
            f"{no_syr_part}<br>"
            f"🔁 Unique comparison pairs: {swt_resolved} SYT-SWT, "
            f"{related_resolved} SYT-SYT (related){missing_part}"
        )

        self._tree.clear()
        self._tree.setUpdatesEnabled(False)
        try:
            top_items = [self._build_tree_row(row) for row in result["traceability"]]
            self._tree.addTopLevelItems(top_items)
        finally:
            self._tree.setUpdatesEnabled(True)

        # Auto-expanding is only safe for small trees -- a big module can
        # produce 50k+ nodes (see _build_tree_row), where expandAll() has to
        # lay out every one of them and hangs the UI. Use the Expand All
        # button for those instead; it asks first.
        self._tree_node_count = _estimate_tree_nodes(result["traceability"])
        if self._tree_node_count <= _TREE_AUTO_EXPAND_LIMIT:
            self._populate_all_tree_children()
            self._tree.expandAll()
        self._btn_export.setEnabled(True)
        self._btn_unresolved.setEnabled(bool(self._unresolved_pairs))
        self._set_status(
            f"Done. {summary['selected_tc_count']} SYT TCs processed, "
            f"{summary['tcs_with_swt_coverage']} have SWT coverage."
            + (f"  ⚠️ TREK unreachable: {not_dl} TC(s) not downloaded." if not_dl else "")
        )

    def _show_unresolved_pairs(self):
        if not getattr(self, "_unresolved_pairs", None):
            return
        mod_name = self._result["meta"].get("syt_module", "module") if self._result else "module"
        dlg = UnresolvedPairsDialog(self._unresolved_pairs, mod_name)
        _open_independent_window(self, dlg)

    def _expand_all_tree(self):
        """Expand every node, guarding the huge-tree case -- expandAll() on
        a 50k-node tree blocks the UI thread for a long time, so confirm
        first rather than letting the app look hung."""
        count = getattr(self, "_tree_node_count", 0)
        if count > _TREE_AUTO_EXPAND_LIMIT:
            reply = QMessageBox.question(
                self, "Expand All",
                f"This traceability tree has ~{count:,} nodes. Expanding all of "
                f"them can take a while and the window will be unresponsive "
                f"until it finishes.\n\nExpand anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._tree.setUpdatesEnabled(False)
        try:
            self._populate_all_tree_children()
            self._tree.expandAll()
        finally:
            self._tree.setUpdatesEnabled(True)
            QApplication.restoreOverrideCursor()

    def _populate_all_tree_children(self):
        """Force the lazy children of every top-level node into existence --
        expandAll() alone wouldn't expand them, since they're only created
        while its own itemExpanded signal is being handled."""
        for i in range(self._tree.topLevelItemCount()):
            self._on_tree_item_expanded(self._tree.topLevelItem(i))

    def _build_tree_row(self, row: dict) -> QTreeWidgetItem:
        """Build ONLY the top-level SYT node. Its SYR/SWR/SWT/related-SYT
        children are created lazily on first expand (see
        _on_tree_item_expanded) -- a big module reaches 50k+ child nodes
        (one SYR can be shared by 200+ related SYTs), and building them all
        up front blocked the UI thread for long enough that Windows marked
        the app "Not Responding"."""
        syt_id  = row["syt_id"]
        has_swt = row["has_swt"]
        has_syr = row["has_syr_link"]

        # Status icon
        if has_swt:
            icon, color = "✅", SUCCESS
        elif has_syr:
            icon, color = "⚠️", WARNING
        else:
            icon, color = "❌", DANGER

        top = QTreeWidgetItem([
            syt_id,
            "SYT",
            f"{icon}  SYR:{row['syr_count']}  SWR:{row['swr_count']}  SWT:{row['swt_count']}  Related:{row.get('related_syt_count', 0)}"
        ])
        top.setForeground(0, QColor(color))
        top.setForeground(2, QColor(color))
        top.setData(0, Qt.UserRole, {"type": "SYT", "row": row})
        top.setFont(0, _FONT_TREE_TOP)
        if syt_id in row.get("not_downloaded_ids", ()):
            top.setToolTip(0, "Text not downloaded -- TREK was unreachable. "
                              "Run again while connected to TREK.")
        if row.get("chain"):
            top.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        return top

    def _on_tree_item_expanded(self, item: QTreeWidgetItem):
        data = item.data(0, Qt.UserRole)
        if not data or data.get("type") != "SYT" or item.data(0, _ROLE_CHILDREN_BUILT):
            return
        item.setData(0, _ROLE_CHILDREN_BUILT, True)
        item.addChildren(self._build_chain_children(data["row"]))

    @staticmethod
    def _build_chain_children(row: dict) -> List[QTreeWidgetItem]:
        NOT_FOUND_NOTE = "⚠️ NOT FOUND IN TREK -- check manually in DOORS/TREK UI"
        NOT_DOWNLOADED_NOTE = "⏳ not downloaded -- TREK unreachable (run again online)"
        not_downloaded = set(row.get("not_downloaded_ids", ()))
        NO_SWR_NOTE = "⚠️ no SWR link in TREK (not refined into software) => no SWT possible"
        UNRESOLVED_SYR_NOTE = ("⚠️ SYR not in any fetched SYR module -- it likely belongs to "
                               "another module, so its SWR/SWT links could not be followed")
        resolved_swt_ids = {c.get("Key") for c in row.get("swt_content", [])}
        resolved_related_ids = {c.get("Key") for c in row.get("related_syt_content", [])}
        resolved_syr_ids = set(row.get("syr_content", {}))
        syr_nodes = []
        for chain_entry in row.get("chain", []):
            syr_id = chain_entry["syr_id"]
            # A SYR with no children is ambiguous on its own: either we never
            # fetched its module, or TREK genuinely has no SWR refinement for
            # it. Say which, instead of showing a silently empty node.
            if syr_id not in resolved_syr_ids:
                syr_note, syr_color = UNRESOLVED_SYR_NOTE, DANGER
            elif not chain_entry.get("swr_ids"):
                syr_note, syr_color = NO_SWR_NOTE, WARNING
            else:
                syr_note, syr_color = "", REQ_COLOR
            syr_node = QTreeWidgetItem([syr_id, "SYR", syr_note])
            syr_node.setForeground(0, QColor(syr_color))
            syr_node.setForeground(2, QColor(syr_color))
            syr_node.setFont(0, _FONT_TREE_CHILD_BOLD if syr_note else _FONT_TREE_CHILD)
            syr_node.setData(0, Qt.UserRole, {"type": "SYR", "syr_id": syr_id, "row": row})

            child_nodes = []
            for swr_id in chain_entry.get("swr_ids", []):
                swr_node = QTreeWidgetItem([swr_id, "SWR", ""])
                swr_node.setForeground(0, QColor(SWR_COLOR))
                swr_node.setFont(0, _FONT_TREE_CHILD)
                swr_node.setData(0, Qt.UserRole, {"type": "SWR", "swr_id": swr_id, "row": row})
                child_nodes.append(swr_node)

            for swt_id in chain_entry.get("swt_ids", []):
                not_found = swt_id not in resolved_swt_ids
                offline = not_found and swt_id in not_downloaded
                note = (NOT_DOWNLOADED_NOTE if offline else NOT_FOUND_NOTE) if not_found else ""
                miss_color = AMBER_COLOR if offline else DANGER_TEXT
                swt_node = QTreeWidgetItem([swt_id, "SWT", note])
                swt_node.setForeground(0, QColor(miss_color if not_found else SUCCESS_TEXT))
                swt_node.setForeground(2, QColor(miss_color))
                swt_node.setFont(0, _FONT_TREE_CHILD_BOLD if not_found else _FONT_TREE_CHILD)
                swt_node.setData(0, Qt.UserRole, {"type": "SWT", "swt_id": swt_id, "row": row})
                child_nodes.append(swt_node)

            # Other SYT test cases that also link to this SYR -- kept as
            # "related" to the SYT we started from (see FetchLinksWorker
            # step 4), shown as siblings of the SWR/SWT children so it's
            # clear they were discovered via this specific SYR.
            for related_id in chain_entry.get("related_syt_ids", []):
                not_found = related_id not in resolved_related_ids
                offline = not_found and related_id in not_downloaded
                note = (NOT_DOWNLOADED_NOTE if offline else NOT_FOUND_NOTE) if not_found else ""
                miss_color = AMBER_COLOR if offline else DANGER_TEXT
                related_node = QTreeWidgetItem([related_id, "Related SYT", note])
                related_node.setForeground(0, QColor(miss_color if not_found else RELATED_COLOR))
                related_node.setForeground(2, QColor(miss_color))
                related_node.setFont(0, _FONT_TREE_CHILD_BOLD if not_found else _FONT_TREE_CHILD)
                related_node.setData(0, Qt.UserRole, {"type": "RELATED_SYT", "related_syt_id": related_id, "row": row})
                child_nodes.append(related_node)

            syr_node.addChildren(child_nodes)
            syr_nodes.append(syr_node)
        return syr_nodes

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, col: int):
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        row = data.get("row", {})

        if data.get("type") == "SYT":
            tc = row.get("syt_content")
            if tc:
                self._detail_syt.setHtml(_tc_to_html(tc, ACCENT))
            elif row.get("syt_id") in row.get("not_downloaded_ids", ()):
                self._detail_syt.setPlainText(
                    f"⏳ NOT DOWNLOADED: {row.get('syt_id')}\n\n"
                    "This test case was NOT DOWNLOADED: TREK could not be reached when "
                    "its text was requested (offline, VPN down, or a timeout). It is "
                    "not missing from TREK. Run Build Traceability again while "
                    "connected (or use 'Offline' -> download all test-case text); "
                    "until then it is excluded from 'Check Duplicates'."
                )
            else:
                self._detail_syt.setPlainText("No content available.")
            # Show all linked SWT content
            swt_list = row.get("swt_content", [])
            if swt_list:
                html = "".join(_tc_to_html(tc, SUCCESS_TEXT) + f"<hr style='border-color:{BORDER}'>"
                               for tc in swt_list)
                self._detail_swt.setHtml(html)
            else:
                self._detail_swt.setPlainText("No linked SWT test cases.")
            self._detail_tabs.setCurrentIndex(0)

        elif data.get("type") == "SYR":
            syr_id  = data.get("syr_id")
            req     = row.get("syr_content", {}).get(syr_id)
            if req:
                self._detail_syr.setHtml(_req_to_html(req, REQ_COLOR))
            else:
                self._detail_syr.setPlainText(
                    f"Requirement content not available for {syr_id}.\n"
                    "(The SYR module's requirement export may not have been fetched, "
                    "or this ID was not found in it.)"
                )
            self._detail_tabs.setCurrentIndex(1)

        elif data.get("type") == "SWR":
            swr_id  = data.get("swr_id")
            req     = row.get("swr_content", {}).get(swr_id)
            if req:
                self._detail_swr.setHtml(_req_to_html(req, SWR_COLOR))
            else:
                self._detail_swr.setPlainText(
                    f"Requirement content not available for {swr_id}.\n"
                    "(The SWR module's requirement export may not have been fetched, "
                    "or this ID was not found in it.)"
                )
            self._detail_tabs.setCurrentIndex(2)

        elif data.get("type") == "SWT":
            swt_id  = data.get("swt_id")
            swt_list= row.get("swt_content", [])
            tc      = next((t for t in swt_list if t.get("Key") == swt_id), None)
            if tc:
                self._detail_swt.setHtml(_tc_to_html(tc, SUCCESS_TEXT))
            elif swt_id in row.get("not_downloaded_ids", ()):
                self._detail_swt.setPlainText(
                    f"⏳ NOT DOWNLOADED: {swt_id}\n\n"
                    "This test case was NOT DOWNLOADED: TREK could not be reached when "
                    "its text was requested (offline, VPN down, or a timeout). It is "
                    "not missing from TREK. Run Build Traceability again while "
                    "connected (or use 'Offline' -> download all test-case text); "
                    "until then it is excluded from 'Check Duplicates'."
                )
            else:
                self._detail_swt.setPlainText(
                    f"⚠️ NOT FOUND IN TREK: {swt_id}\n\n"
                    "This SWT test case is LINKED (via SYR -> SWR -> SWT) but TREK "
                    "returned no usable content for it -- likely archived, deleted, "
                    "or access-restricted. Please check it manually in DOORS or the "
                    "TREK UI; it is excluded from 'Check Duplicates' until then."
                )
            self._detail_tabs.setCurrentIndex(3)

        elif data.get("type") == "RELATED_SYT":
            related_id   = data.get("related_syt_id")
            related_list = row.get("related_syt_content", [])
            tc = next((t for t in related_list if t.get("Key") == related_id), None)
            if tc:
                self._detail_related_syt.setHtml(_tc_to_html(tc, RELATED_COLOR))
            elif related_id in row.get("not_downloaded_ids", ()):
                self._detail_related_syt.setPlainText(
                    f"⏳ NOT DOWNLOADED: {related_id}\n\n"
                    "This test case was NOT DOWNLOADED: TREK could not be reached when "
                    "its text was requested (offline, VPN down, or a timeout). It is "
                    "not missing from TREK. Run Build Traceability again while "
                    "connected (or use 'Offline' -> download all test-case text); "
                    "until then it is excluded from 'Check Duplicates'."
                )
            else:
                self._detail_related_syt.setPlainText(
                    f"⚠️ NOT FOUND IN TREK: {related_id}\n\n"
                    "This SYT test case links to a SYR shared with the original "
                    "selection, but TREK returned no usable content for it -- "
                    "likely archived, deleted, or access-restricted. Please check "
                    "it manually in DOORS or the TREK UI; it is excluded from "
                    "'Check Duplicates' until then."
                )
            self._detail_tabs.setCurrentIndex(4)

    # -----------------------------------------------------------------------
    # Duplicate detection (SYT vs SWT similarity)
    # -----------------------------------------------------------------------
    def _current_jwt_token(self) -> str:
        active = PROJECT_STORE.get_active()
        if not active:
            return ""
        return PROJECT_STORE.get_jwt_token(active["key"])

    def _run_duplicate_check(self):
        if not self._result or not self._result.get("traceability"):
            self._set_status("Run 'Build Traceability' first.")
            return

        jwt_token = self._current_jwt_token()
        if not jwt_token:
            QMessageBox.warning(
                self, "Missing JWT Token",
                "The active project has no JWT Token configured. A JWT "
                "Token is required for 'Check Duplicates' -- edit this "
                "project (header ✎ button) to set one."
            )
            return

        # Let the user tune hybrid weighting + thresholds for this
        # specific run before spending any embedding/BM25 calls.
        settings_dlg = DuplicateCheckSettingsDialog(
            parent=self, current=getattr(self, "_last_dup_settings", None),
            traceability_rows=self._result["traceability"],
        )
        if settings_dlg.exec() != QDialog.Accepted:
            return
        run_settings = settings_dlg.result_data()
        self._last_dup_settings = run_settings   # remember for next time this session

        worker_settings = dict(run_settings, jwt_token=jwt_token)

        self._btn_check_duplicates.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._busy_start("Starting duplicate check...")

        worker = DuplicateCheckWorker(self._result["traceability"], settings=worker_settings)
        worker.progress.connect(self._busy_stage_update)
        worker.done.connect(self._on_duplicate_check_ready)
        worker.error.connect(self._on_duplicate_check_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_duplicate_check_error(self, msg: str):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_check_duplicates.setEnabled(True)
        self._btn_run.setEnabled(True)
        self._set_status(f"Duplicate check failed: {msg}")

        if msg == "Cancelled by user.":
            self._set_status("Duplicate check stopped -- any results already computed were still saved to cache.")
            return

        # Give a targeted hint based on what actually went wrong, instead
        # of always pointing at the JWT Token -- e.g. a Vertex AI 400
        # INVALID_ARGUMENT "input token count" error means the request was
        # too large, which has nothing to do with credentials.
        lower_msg = msg.lower()
        if "token count" in lower_msg or "invalid_argument" in lower_msg:
            hint = ("The embedding request exceeded the model's per-request "
                    "size limit. This should now be handled automatically by "
                    "splitting large batches -- if you still see this, try "
                    "selecting fewer test cases per 'Check Duplicates' run.")
        elif "no jwt token" in lower_msg or "jwt token" in lower_msg:
            hint = "Check this project's JWT Token (header ✎ Edit Project button)."
        else:
            hint = "Check this project's JWT Token (header ✎ Edit Project button) and network connectivity."

        QMessageBox.critical(
            self, "Duplicate Check Failed",
            f"{msg}\n\n{hint}"
        )

    def _on_duplicate_check_ready(self, result: "trek_similarity.DuplicateCheckResult"):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_check_duplicates.setEnabled(True)
        self._btn_run.setEnabled(True)
        self._last_dup_result = result   # kept for _export_json's cost/report data

        counts = result.summary_counts()
        token_part = (
            f", ~{result.total_tokens_embedded:,} tokens embedded"
            f"{f' (${result.total_cost_usd:.4f})' if result.total_cost_usd else ''}"
            if result.total_tokens_embedded else ""
        )
        llm_judged = sum(1 for p in result.pairs if p.llm_verdict)
        llm_part = (
            f", {llm_judged} pairs LLM-verified ({result.llm_calls} batch request(s))"
            f"{f' (${result.total_llm_cost_usd:.4f})' if result.total_llm_cost_usd else ''}"
            if llm_judged else ""
        )
        self._set_status(
            f"Duplicate check complete: {len(result.pairs)} pairs compared, "
            f"{counts['duplicate'] + counts['near_duplicate']} flagged as duplicate/near-duplicate"
            f"{token_part}{llm_part}."
        )

        # Persist the full result so it can be browsed later per module
        # (see CachedDuplicateResultsDialog) without re-running anything.
        # Keyed per (module, LLM model) -- see key_duplicate_check() -- so
        # running the same module with different models keeps a SEPARATE
        # stored version per model instead of overwriting each other.
        #
        # MERGED (not overwritten) with whatever's already cached for this
        # (module, model): re-running on a smaller/different selection of
        # test cases must never discard the rest of a larger previously
        # -cached list -- only the pairs actually part of THIS run are
        # updated (and stamped with checked_at so the results view can
        # flag them as the latest), every other previously-cached pair is
        # kept as-is. See trek_similarity.merge_duplicate_check_results().
        mod_name = self._result["meta"].get("syt_module") if self._result else None
        if mod_name and result.pairs:
            module_key = trek_cache.key_duplicate_check(PROJECT_ID, CAMPAIGN_ID, mod_name, result.llm_model)
            existing_data = CACHE.get_duplicate_check_result(module_key)
            run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            if existing_data:
                existing_result = trek_similarity.DuplicateCheckResult.from_dict(existing_data)
                to_persist = trek_similarity.merge_duplicate_check_results(existing_result, result, run_timestamp)
            else:
                for p in result.pairs:
                    p.checked_at = run_timestamp
                to_persist = result
            CACHE.set_duplicate_check_result(module_key, mod_name, to_persist.to_dict())

        # Append-only stats for the "App Report" -- unlike the module-keyed
        # result above (overwritten each run), these accumulate forever so
        # the report can show a TRUE historical average across every run.
        # ``item_count`` is the NEW work actually done this run (NOT the
        # total pair count) so cost/time-per-item stays comparable across
        # runs -- otherwise a run that's mostly cache hits looks
        # deceptively "cheap per pair" next to a run with less caching.
        # ``extra_count`` keeps the total pair count for reference.
        # ``details.model``/``details.tokens`` let the App Report break
        # down token usage per model (see AppStatsDialog's token section).
        if result.rag_duration_seconds:
            CACHE.record_operation_stat(
                "rag_stage", module=mod_name, item_count=result.total_texts_embedded,
                extra_count=len(result.pairs),
                duration_seconds=result.rag_duration_seconds, cost_usd=result.total_cost_usd,
                details={"model": trek_similarity.EMBEDDING_MODEL, "tokens": result.total_tokens_embedded},
            )
        if result.llm_duration_seconds:
            new_pairs_judged = len(result.pairs) - result.llm_cached_pairs
            CACHE.record_operation_stat(
                "llm_stage", module=mod_name, item_count=new_pairs_judged,
                extra_count=len(result.pairs),
                duration_seconds=result.llm_duration_seconds, cost_usd=result.total_llm_cost_usd,
                details={"model": result.llm_model, "tokens": result.total_llm_tokens},
            )

        # Show the dedicated side-by-side results window automatically.
        dlg = DuplicateResultsDialog(result, module_name=mod_name)
        _open_independent_window(self, dlg)

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    def _export_json(self):
        if not self._result:
            return
        mod_name = self._result["meta"].get("syt_module", "export")
        safe     = mod_name.replace(" ", "_").replace("/", "-")
        default  = f"trek_traceability_{safe}.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", default, "JSON files (*.json)"
        )
        if not path:
            return
        export_data = dict(self._result)
        # Include the last 'Check Duplicates' run's cost/token data (RAG
        # embeddings + LLM verification) so a later report can track
        # spend over time without having to re-run anything.
        last_dup_result = getattr(self, "_last_dup_result", None)
        if last_dup_result is not None:
            export_data["duplicate_check"] = last_dup_result.to_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(export_data, fh, indent=2, ensure_ascii=False)
        self._set_status(f"Exported to {path}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _on_error(self, msg: str):
        self._busy_stop()
        self._progress.setVisible(False)
        self._btn_fetch_mods.setEnabled(True)
        self._btn_refresh_mods.setEnabled(True)
        self._btn_get_tcs.setEnabled(self._selected_mod is not None)
        self._btn_run.setEnabled(True)
        if msg == "Cancelled by user.":
            self._set_status("Stopped by user.")
            return
        self._set_status(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    def _show_database_dialog(self):
        dlg = TrekDatabaseDialog(CACHE)
        _open_independent_window(self, dlg)

    def _show_log_dialog(self):
        dlg = TrekLogDialog(LOG)
        _open_independent_window(self, dlg)

    def _show_cached_results_dialog(self):
        dlg = CachedDuplicateResultsDialog(CACHE)
        _open_independent_window(self, dlg)

    def _show_compare_models_dialog(self):
        dlg = CompareModelVersionsDialog(CACHE)
        _open_independent_window(self, dlg)

    def _show_app_stats_dialog(self):
        dlg = AppStatsDialog(CACHE)
        _open_independent_window(self, dlg)

    def _show_overview_dashboard(self):
        dlg = OverviewDashboardDialog(CACHE)
        _open_independent_window(self, dlg)

    def _confirm_and_clear_cache(self):
        reply = QMessageBox.warning(
            self, "Delete All Cache Data",
            "This will PERMANENTLY delete EVERYTHING in the local cache:\n"
            "- cached modules, links, and test-case content\n"
            "- embeddings and LLM judgments\n"
            "- persisted 'Check Duplicates' results\n"
            "- app performance/cost statistics\n\n"
            "This cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        CACHE.clear_all()

        # Same Step 1/2/3 reset as switching projects (see
        # _on_active_project_switched), since every cached artefact those
        # steps depend on was just wiped.
        self._modules = []
        self._syt_modules = []
        self._selected_mod = None
        self._all_syt_ids = []
        self._syr_candidates = []
        self._swt_candidates = []
        self._all_syr_module_names = []
        self._all_swt_module_names = []
        self._tc_chapters = {}
        self._result = None
        self._unresolved_pairs = []
        self._last_dup_result = None
        self._mod_list.clear()
        self._mod_info.setText("← Pick a module first")
        self._tc_tree.clear()
        self._tree.clear()
        self._detail_syt.clear()
        self._detail_syr.clear()
        self._detail_swr.clear()
        self._detail_swt.clear()
        self._btn_get_tcs.setEnabled(False)
        self._btn_edit_mapping.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_unresolved.setEnabled(False)
        self._btn_check_duplicates.setEnabled(False)
        self._mod_cache_lbl.setText("Modules source: (not loaded)")
        self._set_status("All cache data deleted. Reloading modules...")
        self._fetch_modules(force_refresh=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setApplicationName("AI DupeHunter")
    icon_path = trek_paths.resource_file("dupehunter_icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    # Without an explicit AppUserModelID Windows groups the app under the
    # host python.exe and shows ITS icon in the taskbar; the frozen exe needs
    # it too so pinning survives rebuilds.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MAIA.AIDupeHunter")
        except Exception:
            pass

    # The module-level STYLESHEET (built at import time, BEFORE this
    # QApplication existed) intentionally has no checkbox tick/dash icons --
    # rendering those requires a live QApplication (constructing a QPixmap
    # earlier is a Qt fatal error, not a catchable exception; see
    # trek_theme._icon_path). Now that the QApplication exists, rebuild the
    # stylesheet once so the real icon files get baked into the QSS.
    global STYLESHEET
    STYLESHEET = trek_theme.build_stylesheet(trek_theme.get_active())

    # Applied at the QApplication level (not just TrekMainWindow) so
    # independent top-level windows (see _open_independent_window(),
    # which shows them without a parent so each gets its own taskbar
    # entry) still pick up the dark theme via stylesheet cascade.
    app.setStyleSheet(STYLESHEET)

    # Override the application palette's Highlight/HighlightedText so Qt's
    # native selection drawing matches the theme's soft selection colour.
    from PySide6.QtGui import QPalette
    pal = app.palette()
    pal.setColor(QPalette.Highlight, QColor(SELECTION_BG))
    pal.setColor(QPalette.HighlightedText, QColor(TEXT))
    app.setPalette(pal)

    # Packaged builds only -- keeps a source checkout free of generated files.
    if trek_paths.is_frozen():
        trek_paths.ensure_config_template()

    # First-run setup: block until the user configures at least one TREK
    # project (Project ID / Campaign ID / Config ID + JWT Token). On
    # subsequent runs, PROJECT_STORE already has a saved active project
    # and this is skipped entirely. allow_cancel=False here because the
    # app cannot do anything useful without at least one configured
    # project, and the JWT Token itself is mandatory (see
    # ProjectSetupDialog / TrekProjectStore.add_project) since discovering
    # duplicate SYT/SWT test cases is this application's core purpose.
    if not PROJECT_STORE.has_projects():
        setup = ProjectSetupDialog(allow_cancel=False)
        setup.setWindowTitle("Welcome to AI DupeHunter -- Project Setup")
        if setup.exec() == QDialog.Accepted:
            data = setup.result_data()
            PROJECT_STORE.add_project(
                data["name"], data["project_id"], data["campaign_id"], data["config_id"],
                jwt_token=data["jwt_token"], make_active=True,
                db_path=data.get("db_path", ""),
            )
        else:
            # allow_cancel=False means _on_save() is the only way to close
            # this dialog, so reaching here would mean the window was
            # force-closed (e.g. Alt+F4) without saving -- exit cleanly
            # rather than launching the main window with no project.
            sys.exit(0)

    # Apply the active project (which may open a slow network-share cache
    # DB). Show a lightweight splash label so the user sees immediate
    # feedback instead of nothing while SQLite connects + runs migrations.
    active = PROJECT_STORE.get_active()
    db_path_display = (active or {}).get("db_path", "") or str(trek_paths.data_dir())
    _splash_start = time.perf_counter()

    splash = QLabel(f"⏳  Connecting to cache database...  (00:00)\n{db_path_display}")
    splash.setWindowTitle("AI DupeHunter")
    splash.setAlignment(Qt.AlignCenter)
    splash.setStyleSheet(
        f"background:{DARK_BG};color:{TEXT};font-size:14px;padding:40px 60px;"
        f"border:2px solid {ACCENT};border-radius:12px;"
    )
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    splash.adjustSize()
    splash.show()
    app.processEvents()

    def _update_splash_elapsed():
        elapsed = time.perf_counter() - _splash_start
        mins, secs = divmod(int(elapsed), 60)
        splash.setText(
            f"⏳  Connecting to cache database...  ({mins:02d}:{secs:02d})\n{db_path_display}"
        )

    splash_timer = QTimer()
    splash_timer.timeout.connect(_update_splash_elapsed)
    splash_timer.start(1000)

    # Run _apply_active_project() in a worker thread so the main thread's
    # event loop stays alive to repaint the splash timer. A local QEventLoop
    # blocks main() here (so the main window can't open with the wrong DB)
    # but still processes QTimer/repaint events every second.
    from PySide6.QtCore import QEventLoop
    _startup_loop = QEventLoop()
    _startup_error = []

    class _StartupCacheWorker(QThread):
        def run(self):
            try:
                # Only open the database here; the (large) id index is loaded
                # in the background once the main window is visible.
                _apply_active_project(load_index=False)
            except Exception as exc:
                _startup_error.append(str(exc))

    _cache_worker = _StartupCacheWorker()
    _cache_worker.finished.connect(_startup_loop.quit)

    with LOG.timed("Startup", f"Connect to cache database ({db_path_display})") as t:
        _cache_worker.start()
        _startup_loop.exec()  # blocks here, but timer events still fire
        t.details["db_path"] = db_path_display
        t.details["journal_mode"] = CACHE.journal_mode
        t.details["shared"] = CACHE.is_shared
        t.details.update({f"init_{k}": f"{v:.2f}s" for k, v in CACHE._init_timings.items()})
        if _startup_error:
            t.details["error"] = _startup_error[0]

    splash_timer.stop()
    splash.close()

    if _startup_error:
        QMessageBox.warning(None, "Cache Connection Error",
            f"Failed to connect to cache database:\n{_startup_error[0]}\n\n"
            "Falling back to default local cache.")

    win = TrekMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
