"""
TREK Activity Log
==================
Lightweight in-memory + on-disk activity/timing log for trek_gui.py, so the
user can see exactly what the app did during a run (every TREK API call,
cache hit/miss, worker stage) and how long each step took -- directly
answering "how much time did X take" without needing to watch the status
bar in real time or guess from the busy-indicator text.

Design
------
- ``TrekLog`` keeps a bounded in-memory deque of ``LogEntry`` records
  (newest last) for fast display in the GUI, AND appends the same entries
  to a persistent JSON-Lines file (trek_activity.log, next to this script
  by default) so history survives app restarts and can be inspected
  outside the GUI (e.g. with any text editor) if needed.

- Thread-safe: FetchLinksWorker / DuplicateCheckWorker / FetchModulesWorker
  / etc. all run on background QThreads and call ``log()`` directly (no Qt
  signal round-trip required to record an entry) -- a plain
  ``threading.Lock`` protects the shared deque and file handle so
  concurrent workers (see trek_gui.py's ThreadPoolExecutor-based
  parallelization) never corrupt the log.

- ``timed()`` is a context manager that logs a single entry with the
  elapsed wall-clock time automatically computed, so instrumenting a call
  site is a one-line::

      with LOG.timed("SYT Links", f"Fetch links for '{name}'") as t:
          ... do the work ...
          t.details["item_count"] = len(items)   # optional, added before the log line is written

  rather than manually calling ``time.perf_counter()`` and building the
  LogEntry by hand at every call site.

Same design precedent as trek_cache.py / trek_projects.py: a plain
module-level default path resolved via ``Path(__file__).parent`` (robust
to working directory), instantiated explicitly by the caller (trek_gui.py
creates ``LOG = trek_log.TrekLog()``) rather than a hidden global singleton,
so tests can redirect ``DEFAULT_LOG_PATH`` to a temp location before
importing trek_gui and never touch the real log file on disk.
"""

from __future__ import annotations

import json
import threading
import time
import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import trek_paths

DEFAULT_LOG_PATH = trek_paths.data_file("trek_activity.log")

# Cap on in-memory entries kept for the GUI viewer -- the on-disk file is
# append-only and NOT capped (so long-term history is never lost), but
# holding tens of thousands of entries in memory for display would be
# wasteful. 5000 is generous for even a very long working session.
DEFAULT_MAX_ENTRIES = 5000

LEVELS = ("INFO", "WARN", "ERROR")


@dataclass
class LogEntry:
    timestamp: str                      # ISO-8601, e.g. "2026-08-16T14:32:07.123456"
    level: str                          # "INFO" | "WARN" | "ERROR"
    category: str                       # e.g. "Modules", "SYT Links", "TC Content", "Duplicate Check"
    message: str                        # human-readable description
    duration_ms: Optional[float] = None # elapsed time for this operation, if timed
    details: Dict[str, Any] = field(default_factory=dict)   # arbitrary extra structured info

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_duration(self) -> str:
        if self.duration_ms is None:
            return "-"
        if self.duration_ms < 1000:
            return f"{self.duration_ms:.0f} ms"
        return f"{self.duration_ms / 1000:.2f} s"

    def format_details(self) -> str:
        if not self.details:
            return ""
        return "  ".join(f"{k}={v}" for k, v in self.details.items())


class _TimedOperation:
    """Context manager returned by TrekLog.timed(). Records elapsed
    wall-clock time on exit and writes the log entry, including any
    exception message (as an ERROR-level entry) if the wrapped block
    raised. ``details`` can be freely mutated by the caller before the
    ``with`` block exits -- e.g. to attach a result count discovered only
    after the operation completed.
    """

    def __init__(self, log: "TrekLog", category: str, message: str,
                 level: str = "INFO", **details: Any):
        self._log = log
        self.category = category
        self.message = message
        self.level = level
        self.details: Dict[str, Any] = dict(details)
        self._start: Optional[float] = None

    def __enter__(self) -> "_TimedOperation":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self.details.setdefault("error", str(exc_val))
            self._log.log(self.category, self.message, level="ERROR",
                           duration_ms=duration_ms, **self.details)
            return False   # do not suppress the exception
        self._log.log(self.category, self.message, level=self.level,
                       duration_ms=duration_ms, **self.details)
        return False


class TrekLog:
    """Thread-safe in-memory + append-only-file activity/timing log."""

    def __init__(self, path: Path = DEFAULT_LOG_PATH, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.path = Path(path)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: Deque[LogEntry] = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def log(self, category: str, message: str, level: str = "INFO",
            duration_ms: Optional[float] = None, **details: Any) -> LogEntry:
        """Record one log entry immediately (no timing). Use timed() for
        the common case of "log how long this operation took"."""
        if level not in LEVELS:
            level = "INFO"
        entry = LogEntry(
            timestamp=datetime.datetime.now().isoformat(timespec="milliseconds"),
            level=level,
            category=category,
            message=message,
            duration_ms=duration_ms,
            details=details,
        )
        with self._lock:
            self._entries.append(entry)
            self._append_to_file(entry)
        return entry

    def timed(self, category: str, message: str, level: str = "INFO", **details: Any) -> _TimedOperation:
        """Return a context manager that logs ``message`` with the elapsed
        wall-clock time of the ``with`` block automatically attached."""
        return _TimedOperation(self, category, message, level=level, **details)

    def _append_to_file(self, entry: LogEntry) -> None:
        """Best-effort append to the on-disk JSON-Lines file. Never raises
        -- a disk write failure (e.g. read-only filesystem, disk full)
        should never crash the app or interrupt whatever operation is
        being logged; the in-memory record is still available either way."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get_entries(self) -> List[LogEntry]:
        """Return a snapshot copy of all currently held in-memory entries,
        oldest first."""
        with self._lock:
            return list(self._entries)

    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear(self, clear_file: bool = False) -> None:
        """Clear the in-memory entries. If clear_file=True, also truncate
        the on-disk log file (off by default -- clearing the GUI view
        should not normally destroy the persistent history)."""
        with self._lock:
            self._entries.clear()
            if clear_file:
                try:
                    open(self.path, "w", encoding="utf-8").close()
                except OSError:
                    pass

    def export_text(self, entries: Optional[List[LogEntry]] = None) -> str:
        """Format entries as human-readable lines, e.g. for a 'Save As...'
        export or copy-to-clipboard action in the GUI. Defaults to all
        currently held in-memory entries."""
        if entries is None:
            entries = self.get_entries()
        lines = []
        for e in entries:
            dur = e.format_duration()
            det = e.format_details()
            line = f"[{e.timestamp}] [{e.level:5s}] [{e.category}] {e.message}"
            if dur != "-":
                line += f"  ({dur})"
            if det:
                line += f"  | {det}"
            lines.append(line)
        return "\n".join(lines)
