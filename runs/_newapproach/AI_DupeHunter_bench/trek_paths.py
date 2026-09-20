"""
Runtime path resolution for both source and frozen (PyInstaller) runs.

Two different directories matter once the app is packaged as an .exe:

* **Bundle dir** -- read-only resources unpacked by PyInstaller into a temp
  folder (``sys._MEIPASS``). Anything shipped *with* the exe lives here,
  e.g. the vendored ``TrekExportLinksAPI.py``.
* **Data dir** -- somewhere the user can actually write. The temp bundle dir
  is wiped when the exe exits, so the SQLite cache, activity log, and the
  project list must NOT go there or they would be lost on every run (and
  writes into Program Files would be blocked anyway).

Running from source keeps the historical behaviour -- everything stays next
to the scripts -- so a developer checkout is unaffected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

APP_NAME = "AI_DupeHunter"

# Drop a file with this name next to the .exe containing a single line -- a
# path -- to point the whole team at one shared cache/log/project folder,
# e.g. \\server\share\TrekTraceability. Overridden by the TREK_DATA_DIR
# environment variable if that is set.
DATA_DIR_MARKER = "trek_data_dir.txt"
DATA_DIR_ENV = "TREK_DATA_DIR"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Directory holding read-only resources shipped with the app."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def app_dir() -> Path:
    """Directory the .exe itself sits in (NOT the temp unpack dir)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _configured_data_dir() -> Optional[Path]:
    """Explicit data-dir override, or None if the user hasn't set one.

    A relative path is resolved against the folder holding the .exe (NOT the
    process working directory), so `.` means "portable -- keep everything
    next to the app" no matter where it was launched from.
    """
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    raw = env
    if not raw:
        marker = app_dir() / DATA_DIR_MARKER
        if marker.is_file():
            for line in marker.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    raw = line
                    break
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (app_dir() / path).resolve()
    return path


def data_dir() -> Path:
    """Writable directory for the cache / log / project list.

    Resolution order:
      1. TREK_DATA_DIR env var or trek_data_dir.txt (team share override)
      2. PORTABLE default: the folder holding the .exe (or the script dir
         when running from source). Data files live right next to the app
         so copying the folder to another PC carries everything.
      3. Fallback: %LOCALAPPDATA%\\APP_NAME (only if the portable dir is
         not writable, e.g. installed under Program Files).
    """
    configured = _configured_data_dir()
    if configured is not None:
        try:
            configured.mkdir(parents=True, exist_ok=True)
            return configured
        except OSError:
            pass
    # Portable default: data next to the exe / script.
    portable = app_dir()
    try:
        portable.mkdir(parents=True, exist_ok=True)
        # Quick write-test to make sure the directory is actually writable
        # (it won't be under Program Files or a read-only share).
        _probe = portable / ".write_test"
        _probe.write_text("ok", encoding="utf-8")
        _probe.unlink()
        return portable
    except OSError:
        pass
    # Fallback: per-user app-data directory.
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    root = Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_file(name: str) -> Path:
    return data_dir() / name


def resource_file(*parts: str) -> Path:
    return bundle_dir().joinpath(*parts)


CONFIG_TEMPLATE = """\
# TrekTraceability -- data folder configuration
# ---------------------------------------------
# Controls where trek_cache.sqlite3, trek_projects.json and
# trek_activity.log are kept. Uncomment ONE line below.
#
# 1) PORTABLE -- keep everything next to this .exe. Copy the whole folder to
#    a USB stick or another PC and your settings and cache travel with it.
#
#.
#
# 2) SHARED -- put everything on a folder the whole team can reach, so data
#    one person extracts is instantly available to the others.
#
#\\\\server\\share\\{app}
#
# 3) DEFAULT (nothing uncommented) -- each user gets a private copy in
#    %LOCALAPPDATA%\\{app}
#
# Notes:
#   * Relative paths are resolved against this .exe's folder, so "." is
#     portable mode and works no matter where the app is started from.
#   * The TREK_DATA_DIR environment variable overrides this file.
#   * If the folder is unreachable the app silently falls back to the
#     per-user default, so a broken share never blocks startup.
#   * On a network share SQLite drops out of WAL mode: concurrent writes
#     still work but serialize, so extraction is slower when several people
#     refresh at the same moment. Reading cached data stays fast.
#   * The status bar (bottom right) shows whether the cache is local or
#     shared -- hover it to see the exact folder in use.
#   * trek_projects.json holds your JWT token -- don't hand that file to
#     someone else; each user enters their own on first launch.
""".format(app=APP_NAME)


def ensure_config_template() -> Optional[Path]:
    """Drop a commented config file next to the exe so the shared-folder
    option is discoverable instead of being an undocumented convention.
    Never overwrites an existing file; returns None if the folder is
    read-only (e.g. installed under Program Files)."""
    target = app_dir() / DATA_DIR_MARKER
    if target.exists():
        return target
    try:
        target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        return target
    except OSError:
        return None
