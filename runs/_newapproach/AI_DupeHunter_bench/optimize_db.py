"""Optimize the TREK cache database for use on a network share.

Close AI DupeHunter first (the database must not be in use).

  python optimize_db.py
      Optimize D:/TREKflow/trek_cache.sqlite3 IN PLACE: compress large
      entries, switch to 64 KB pages, drop free space. Also makes local use
      a bit faster.

  python optimize_db.py --copy-to D:/TREKflow/trek_cache_network.sqlite3
      Leave the original as it is (except compressing large entries, which
      stays fully compatible) and write an optimized, verified COPY. Then
      copy that file to the share with Explorer and point the project's
      "Cache Database" (Edit Project) at it.

  python optimize_db.py SOURCE.sqlite3 [--copy-to TARGET.sqlite3]
      Same, for another database file.

  --drop-error-judgments   also delete cached LLM verdicts of type 'error'
                           (never reused by the app; saves a little space).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trek_cache  # noqa: E402


def _mb(n):
    return f"{n / 1024 / 1024:,.0f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="D:/TREKflow/trek_cache.sqlite3")
    ap.add_argument("--copy-to", dest="copy_to", default="")
    ap.add_argument("--drop-error-judgments", action="store_true")
    a = ap.parse_args()

    src = Path(a.source)
    if not src.exists():
        sys.exit(f"Database not found: {src}")
    for side in ("-wal", "-shm", "-journal"):
        p = Path(str(src) + side)
        if p.exists() and p.stat().st_size > 0 and side != "-shm":
            print(f"Note: {p.name} exists -- make sure AI DupeHunter is closed.")

    print(f"Database : {src}  ({_mb(src.stat().st_size)})")
    t0 = time.perf_counter()
    cache = trek_cache.TrekCache(src)
    print(f"Page size: {cache._conn.execute('PRAGMA page_size').fetchone()[0]} bytes, "
          f"free pages: {cache._conn.execute('PRAGMA freelist_count').fetchone()[0]:,}")

    def progress(done, total):
        print(f"  compressing large entries: {done}/{total}", end="\r", flush=True)

    try:
        if a.copy_to:
            print(f"Writing optimized copy -> {a.copy_to} ...")
            r = cache.export_optimized_copy(a.copy_to, drop_error_judgments=a.drop_error_judgments,
                                            progress_cb=progress)
        else:
            print("Optimizing in place (this rewrites the whole file; can take a few minutes) ...")
            r = cache.optimize_storage(progress_cb=progress, drop_error_judgments=a.drop_error_judgments)
    finally:
        cache.close()

    print(" " * 60, end="\r")
    print(f"Done in {time.perf_counter() - t0:,.0f} s")
    print(f"  large entries compressed : {r.get('blobs_compressed', 0)}")
    print(f"  old results compressed   : {r.get('results_compressed', 0)}")
    if a.drop_error_judgments:
        print(f"  error verdicts dropped   : {r.get('error_judgments_dropped', 0):,}")
    print(f"  page size                : {r.get('page_size_after')} bytes")
    print(f"  size                     : {_mb(r['size_before_bytes'])} -> {_mb(r['size_after_bytes'])}"
          + (f"  ({r['path']})" if r.get("path") else ""))
    if a.copy_to:
        print("\nNext: copy the new file to the network share (Explorer is fine), then in the app use\n"
              "Edit Project -> Cache Database to point at it.")


if __name__ == "__main__":
    main()
