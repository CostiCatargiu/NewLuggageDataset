"""Time each step of reading the ID index from a (network) database.  Read-only.

Usage:  python check_share_db.py "\\\\iads220n\\D\\Users\\Costi\\Database\\trek_cache.sqlite3"
Writes check_share_db.txt next to this script and prints it.
"""
import json, sqlite3, sys, time, zlib
from pathlib import Path

db = sys.argv[1] if len(sys.argv) > 1 else r"\\iads220n\D\Users\Costi\Database\trek_cache.sqlite3"
out = []
def p(*a):
    s = " ".join(str(x) for x in a); out.append(s); print(s, flush=True)

t = time.perf_counter()
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
con.execute("PRAGMA mmap_size=0")
p(f"DB: {db}")
p(f"open                  : {time.perf_counter()-t:6.2f} s")
try:
    size = Path(db).stat().st_size
except OSError:
    size = 0
t = time.perf_counter()
page = con.execute("PRAGMA page_size").fetchone()[0]
jm = con.execute("PRAGMA journal_mode").fetchone()[0]
free = con.execute("PRAGMA freelist_count").fetchone()[0]
p(f"header                : {time.perf_counter()-t:6.2f} s   page_size={page}  journal={jm}  "
  f"file={size/1e6:,.0f} MB  free_pages={free:,}")

t = time.perf_counter()
rows = con.execute("SELECT key, typeof(value_json), length(value_json), updated_at FROM blobs "
                   "WHERE key LIKE 'index:%'").fetchall()
p(f"find index row        : {time.perf_counter()-t:6.2f} s   {rows}")
for key, typ, ln, upd in rows:
    t = time.perf_counter()
    (v,) = con.execute("SELECT value_json FROM blobs WHERE key=?", (key,)).fetchone()
    t_read = time.perf_counter() - t
    pages = (ln or 0) / page
    p(f"READ index bytes      : {t_read:6.2f} s   {ln/1e6:.1f} MB stored as {typ} "
      f"(~{pages:,.0f} pages -> {t_read/max(pages,1)*1000:.0f} ms/page)")
    t = time.perf_counter()
    if isinstance(v, bytes):
        v = zlib.decompress(v)
    d = json.loads(v)
    p(f"decompress+parse      : {time.perf_counter()-t:6.2f} s   {len(d.get('object_to_module', {})):,} ids")

t = time.perf_counter()
try:
    n = con.execute("SELECT COUNT(*) FROM duplicate_check_runs").fetchone()[0]
except sqlite3.Error:
    n = "n/a"
p(f"list results (small)  : {time.perf_counter()-t:6.2f} s   {n} rows")
t = time.perf_counter()
c = con.execute("SELECT typeof(value_json), COUNT(*), SUM(length(value_json)) FROM blobs "
                "WHERE length(value_json) >= 8192 GROUP BY 1").fetchall()
p(f"large entries by type : {time.perf_counter()-t:6.2f} s   {c}   (text = not compressed)")
con.close()

# Raw network speed, without SQLite: 20 x 64 KB random reads + 16 MB sequential
import os, random
try:
    with open(db, "rb", buffering=0) as fh:
        t = time.perf_counter()
        for _ in range(20):
            fh.seek(random.randrange(0, max(size - 65536, 1))); fh.read(65536)
        lat = (time.perf_counter() - t) / 20
        t = time.perf_counter(); fh.seek(0); got = 0
        while got < 16 * 1024 * 1024:
            b = fh.read(1024 * 1024)
            if not b:
                break
            got += len(b)
        dt = time.perf_counter() - t
    p(f"raw random 64KB read  : {lat*1000:6.0f} ms each")
    p(f"raw sequential read   : {got/1e6/dt if dt else 0:6.1f} MB/s")
except OSError as e:
    p(f"raw read test failed: {e}")
Path(__file__).with_name("check_share_db.txt").write_text("\n".join(out), encoding="utf-8")
