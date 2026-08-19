"""
disk_space_scan.py

Scans a drive (default C:\\) and reports the biggest folders and files,
so you can see what's actually eating your space before deleting anything.

This script is READ-ONLY. It never deletes or moves anything - it only
reads folder/file sizes and prints/saves a report.

USAGE (run from Command Prompt / PowerShell):
    python disk_space_scan.py
    python disk_space_scan.py --drive "C:\\" --depth 2 --top 30
    python disk_space_scan.py --drive "C:\\Users\\YourName" --top 50

Notes:
    - Run as Administrator for best results, otherwise some system/protected
      folders (e.g. C:\\Windows\\WinSxS, other users' folders) will be skipped
      with a permission error and just excluded from the totals.
    - First run on a full C:\\ drive can take a few minutes depending on how
      many files you have.
"""

import argparse
import os
import sys
from pathlib import Path


def get_size(path: Path) -> int:
    """Return total size in bytes of a file or all files under a directory.
    Silently skips anything that can't be read (permission errors, broken
    symlinks, etc.) instead of crashing the whole scan.
    """
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        for entry in os.scandir(path):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_size(Path(entry.path))
            except (PermissionError, FileNotFoundError, OSError):
                continue
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return total


def human_size(num_bytes: float) -> str:
    """Convert bytes to a human-readable string (KB/MB/GB/TB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def scan_top_level(root: Path, depth: int):
    """Walk `depth` levels deep from root, computing size of every folder/file
    found at that level. depth=1 means only immediate children of root.
    """
    results = []
    print(f"Scanning {root} (this can take a while)...")

    def walk(current: Path, level: int):
        try:
            entries = list(os.scandir(current))
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"  [skip] {current} ({e.__class__.__name__})")
            return
        for entry in entries:
            p = Path(entry.path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if level == depth:
                        size = get_size(p)
                        results.append((size, p, "DIR"))
                    else:
                        walk(p, level + 1)
                elif entry.is_file(follow_symlinks=False) and level == depth:
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                        results.append((size, p, "FILE"))
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
            except (PermissionError, FileNotFoundError, OSError):
                continue

    walk(root, 1)
    return results


def largest_files(root: Path, min_size_mb: float, limit: int):
    """Walk the entire tree and return the largest individual files found,
    regardless of which folder they're in. Useful for spotting single huge
    files (ISOs, VM disks, old backups) that a folder-level scan can hide.
    """
    big_files = []
    min_bytes = min_size_mb * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                if fpath.is_symlink():
                    continue
                size = fpath.stat().st_size
                if size >= min_bytes:
                    big_files.append((size, fpath))
            except (PermissionError, FileNotFoundError, OSError):
                continue
    big_files.sort(key=lambda x: x[0], reverse=True)
    return big_files[:limit]


def main():
    parser = argparse.ArgumentParser(description="Find what's taking up space on a drive.")
    parser.add_argument("--drive", default="C:\\", help="Root path to scan (default: C:\\)")
    parser.add_argument("--depth", type=int, default=2, help="How many folder levels deep to break down (default: 2)")
    parser.add_argument("--top", type=int, default=25, help="How many top entries to show (default: 25)")
    parser.add_argument("--big-files", type=int, default=20, help="How many largest individual files to list (default: 20)")
    parser.add_argument("--min-file-mb", type=float, default=100.0, help="Ignore files smaller than this when hunting big files (default: 100MB)")
    parser.add_argument("--skip-big-files-scan", action="store_true", help="Skip the full-tree largest-files pass (faster, folder breakdown only)")
    parser.add_argument("--save", default="disk_report.txt", help="File to save the report to (default: disk_report.txt)")
    args = parser.parse_args()

    root = Path(args.drive)
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    if os.name != "nt":
        print("Note: this script is written with Windows drives (C:\\) in mind, "
              "but will work on any OS - just pass a normal folder path with --drive.")

    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    out(f"=== Disk space report for {root} ===\n")

    # --- Folder/file breakdown at the requested depth ---
    results = scan_top_level(root, args.depth)
    results.sort(key=lambda x: x[0], reverse=True)
    total_scanned = sum(r[0] for r in results)

    out(f"\nTop {args.top} biggest items at depth {args.depth} under {root}:")
    out(f"{'SIZE':>10}  {'TYPE':<5}  PATH")
    out("-" * 70)
    for size, path, kind in results[: args.top]:
        out(f"{human_size(size):>10}  {kind:<5}  {path}")

    out(f"\nTotal scanned (this level, sum of listed items' full contents): {human_size(total_scanned)}")

    # --- Largest individual files anywhere in the tree ---
    if not args.skip_big_files_scan:
        out(f"\nScanning for the {args.big_files} largest individual files "
            f"(>= {args.min_file_mb} MB) - this is the slow part...")
        big = largest_files(root, args.min_file_mb, args.big_files)
        out(f"\nLargest individual files under {root}:")
        out(f"{'SIZE':>10}  PATH")
        out("-" * 70)
        for size, path in big:
            out(f"{human_size(size):>10}  {path}")

    # --- Save report ---
    try:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to: {os.path.abspath(args.save)}")
    except OSError as e:
        print(f"\nCould not save report: {e}")

    print("\nDone. This script only READS sizes - nothing was deleted or moved.")
    print("Review the list above before deleting anything, especially inside")
    print("C:\\Windows, C:\\Program Files, or any folder you don't recognize.")


if __name__ == "__main__":
    main()