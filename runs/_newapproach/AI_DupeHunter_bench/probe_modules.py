"""Why does TREK return 0 modules?  Shows the RAW answer for a project/config.

Run it from the folder that holds trek_gui.py (needs tal\\...\\TrekExportLinksAPI.py).

  python probe_modules.py --project 607 --config 8579
  python probe_modules.py --project 607 --config 8579 --campaign 102766584
  python probe_modules.py --project 607 --configs 8579,8580,1234     # try several

It prints, per domain type (4 = test cases, 2 = requirements): HTTP status,
how many modules came back, the first module names, and the first 400
characters of the raw response when the list is empty.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
api_path = here / "tal" / "KeywordDrivenBase" / "Addons" / "WorkspaceUpdate" / "TrekExportLinksAPI.py"
if not api_path.exists():
    sys.exit(f"TrekExportLinksAPI.py not found at {api_path} -- run this from the app folder.")
spec = importlib.util.spec_from_file_location("TrekExportLinksAPI", api_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--project", type=int, required=True)
ap.add_argument("--config", type=int)
ap.add_argument("--configs", default="")
ap.add_argument("--campaign", type=int, help="also try one /Export/Links call")
a = ap.parse_args()

configs = [int(c) for c in a.configs.split(",") if c.strip()] or ([a.config] if a.config else [])
if not configs:
    sys.exit("Give --config or --configs")

client = mod.TrekExportLinksClient()
print(f"TREK: {client.base_url}")
for cfg in configs:
    for domain in (4, 2):
        r = client.get_modules(a.project, cfg, domain, timeout=120)
        names = [m.get("Name") for m in (r.modules or []) if isinstance(m, dict)][:8]
        print(f"\nproject={a.project} config={cfg} domain={domain}: "
              f"HTTP {r.status_code} success={r.success} modules={len(r.modules or [])}")
        if names:
            print("   e.g.", ", ".join(str(n) for n in names))
        else:
            raw = (getattr(r, "raw_response", "") or getattr(r, "error_message", "") or "")[:400]
            print("   raw:", raw if raw else "(empty response body)")

if a.campaign and configs:
    print("\n--- one /Export/Links call (needs a real module name to be useful) ---")
    r = client.get_links(a.project, a.campaign, ["NoSuchModule"], 4, timeout=60)
    print(f"links: HTTP {r.status_code} success={r.success} "
          f"raw: {(getattr(r, 'raw_response', '') or '')[:200]}")
