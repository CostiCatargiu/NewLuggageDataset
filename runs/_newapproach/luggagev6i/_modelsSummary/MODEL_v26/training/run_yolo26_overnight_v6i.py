#!/usr/bin/env python3
"""
YOLO26 OVERNIGHT — resolution and capacity, the two axes nobody has touched.

=============================================================================
WHY THESE FIVE AND NOT MORE MODULE VARIANTS
=============================================================================
This is the fallback for "the architecture runs did nothing". If y26_lsshift
and y26_gctxp3 land on top of y26_p2_b32, the custom modules are dead on YOLO26
and ablating them further is wasted GPU time. What has NEVER been tested on this
dataset, on either detector, is:

  RESOLUTION   every one of the ~60 v6i runs was at 640. On v5i, four controlled
               640 -> 896 pairs gave +1.26 / +1.58 / +1.36 / +1.44 — mean +1.41,
               positive 4/4. That is an order of magnitude more than the entire
               YOLO26 loss campaign produced (best +0.35 against a ~0.45 floor).

  CAPACITY     every v6i run is scale 's'. run_model_scale.py planned an m/l probe
               and never executed it. Your paper's abstract claims YOLOv12m; every
               experiment behind it used yolov12s.

Both are orthogonal to everything tried so far: they cannot be redundant with a
loss reweighting or with a P2 head, which is exactly the trap that made SWA and
LB-TAL substitutive rather than additive.

=============================================================================
THE DESIGN — one batch for all five, so every delta is attributable
=============================================================================
    run                imgsz  scale  head        isolates
    y26_3lvl_640_b16     640    s    3-level     THE IN-SET ANCHOR
    y26_3lvl_896_b16     896    s    3-level     resolution, alone
    y26_p2_896_b16       896    s    p2          the P2 head at 896
    y26_m_640_b16        640    m    3-level     capacity, alone
    y26_m_p2_640_b16     640    m    p2          the P2 head at m

    2 - 1  = resolution          4 - 1 = capacity
    3 - 2  = P2 head @896        5 - 4 = P2 head @m

!! BATCH IS FIXED AT 16 FOR ALL FIVE, ON PURPOSE. 896 and yolo26m each force a
   smaller batch than 640/s does; letting batch float would confound every
   comparison with an effective-LR change. Holding it at the minimum that fits
   the heaviest run costs speed and makes the set internally clean. Run 1 is the
   anchor because NOTHING here is comparable to the b32 arch runs or the b82
   loss runs. Do not quote a delta across batches — that mistake is already
   sitting in this project's history.

=============================================================================
CAVEAT ON 896 THAT BELONGS IN THE WRITE-UP
=============================================================================
The source images are natively 640x360. Training at 896 is UPSAMPLING: more
pixels to compute over, not more information. The v5i gains were measured on the
same 640-wide source, so they should transfer — but say it that way rather than
claiming added detail, or a reviewer will say it for you.

Roughly 3-4 h per run at b16, so ~17 h — a full overnight. Trim with:
    python run_yolo26_overnight_v6i.py y26_3lvl_640_b16 y26_3lvl_896_b16
which still gives the resolution answer, the single most valuable delta here.

REQUIRES: nothing custom. All five use SHIPPED ultralytics topologies and stock
loss, so this script runs even if the module port is still not on the import
path. That is deliberate — it is the fallback.

Usage:
    python run_yolo26_overnight_v6i.py
    python run_yolo26_overnight_v6i.py y26_3lvl_896_b16
"""

import gc
import hashlib
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
PROJECT_DIR = "runs_yolo26_overnight_v6i"
EPOCHS = 70
BATCH = 16                 # held for ALL runs — see the design note above
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# For context only. Different batch, NOT a reference for anything below.
CONTEXT = {"yolo26 stock @640 b82": 0.5524, "y26_swa_a06_03 @640 b82": 0.5559,
           "ls_shift_gctxP3 (YOLOv12) @640 b32": 0.5602}

RUNS = [
    {"name": "y26_3lvl_640_b16", "cfg": "yolo26s.yaml", "weights": "yolo26s.pt",
     "imgsz": 640,
     "label": "yolo26s, 3 levels, 640 — THE IN-SET ANCHOR",
     "why": "Nothing in this file is comparable to the b32 arch runs or the b82 "
            "loss runs, so the set carries its own reference. Every delta below "
            "is measured against this run and only against this run."},

    {"name": "y26_3lvl_896_b16", "cfg": "yolo26s.yaml", "weights": "yolo26s.pt",
     "imgsz": 896,
     "label": "yolo26s, 3 levels, 896 — RESOLUTION, alone",
     "why": "The single highest-expected-value run left in the project. Four "
            "controlled 640->896 pairs on v5i gave +1.26/+1.58/+1.36/+1.44, "
            "positive 4/4, mean +1.41. Nothing else measured anywhere in this "
            "work comes close. Minus run 1, this is resolution with the head, "
            "the scale, the loss and the batch all held."},

    {"name": "y26_p2_896_b16", "cfg": "yolo26-p2.yaml", "weights": "yolo26s.pt",
     "imgsz": 896,
     "label": "yolo26-p2, 4 levels, 896 — resolution + the P2 head",
     "why": "Are resolution and a stride-4 head additive, or do they both just "
            "supply small-object detail and saturate? This is the same "
            "substitution question that made SWA redundant with the P2 head on "
            "YOLOv12, asked on the one axis where the effect is large enough to "
            "see. Minus run 2 = the P2 head at 896."},

    {"name": "y26_m_640_b16", "cfg": "yolo26m.yaml", "weights": "yolo26m.pt",
     "imgsz": 640,
     "label": "yolo26m, 3 levels, 640 — CAPACITY, alone",
     "why": "Every run in this project is scale 's'. run_model_scale.py planned "
            "an m/l probe and never ran it, and the paper's abstract claims "
            "YOLOv12m while every experiment behind it used yolov12s. Minus run "
            "1, this is capacity with everything else held — and it also closes "
            "the gap between what the manuscript says and what was measured."},

    {"name": "y26_m_p2_640_b16", "cfg": "yolo26-p2.yaml", "weights": "yolo26m.pt",
     "imgsz": 640, "scale_override": "m",
     "label": "yolo26m + p2, 4 levels, 640 — capacity + the P2 head",
     "why": "Minus run 4 gives the P2 head at m scale. If the head helps at s but "
            "not at m, the head was compensating for capacity rather than adding "
            "resolution — a cleaner explanation than anything the loss campaign "
            "produced, and testable in one run."},
]


def resolve_cfg(rc):
    """yolo26-p2.yaml at scale m must be requested as yolo26m-p2.yaml."""
    s = rc.get("scale_override")
    if not s:
        return rc["cfg"]
    stem, ext = os.path.splitext(rc["cfg"])
    base = stem.split("-")[0]
    rest = stem[len(base):]
    return f"{base}{s}{rest}{ext}"


def env_provenance():
    info = {"ultralytics_path": None, "loss_md5": None, "version": None}
    try:
        import ultralytics
        import ultralytics.utils.loss as _lm
        info["ultralytics_path"] = os.path.dirname(ultralytics.__file__)
        info["version"] = getattr(ultralytics, "__version__", None)
        p = getattr(_lm, "__file__", None)
        if p and os.path.exists(p):
            info["loss_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    except Exception as e:
        info["import_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  ultralytics : {ENV.get('ultralytics_path')}  v{ENV.get('version')}")
    print(f"  loss.py md5 : {ENV.get('loss_md5')}")
    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import ultralytics: {ENV['import_error']}")
        return False

    print(f"\n  [!] BATCH={BATCH} is held for ALL runs so the set is internally")
    print(f"      attributable. These numbers are NOT comparable to the b32 arch")
    print(f"      runs or the b82 loss runs. Compare only against y26_3lvl_640_b16.")
    print(f"  [!] 896 is UPSAMPLING — the source images are 640x360.")

    missing_w = sorted({r["weights"] for r in todo
                        if not os.path.exists(r["weights"])})
    if missing_w:
        print(f"\n  [note] weights not found locally, ultralytics will download: {missing_w}")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def run_one(rc):
    name, cfg, imgsz = rc["name"], resolve_cfg(rc), rc["imgsz"]
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}  weights={rc['weights']}  imgsz={imgsz}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    try:
        model.load(rc["weights"])
    except Exception as e:
        print(f"  [warn] weight transfer failed: {e} — training from scratch")

    nl = len(getattr(model.model.model[-1], "stride", [])) or "?"
    npar = sum(p.numel() for p in model.model.parameters())
    print(f"  built: {nl} detection levels, {npar / 1e6:.2f}M params")

    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=imgsz, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=name, patience=PATIENCE, close_mosaic=CLOSE_MOSAIC,
                          seed=SEED, deterministic=True, exist_ok=OVERWRITE_EXISTING)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "cfg": cfg, "weights_init": rc["weights"], "imgsz": imgsz,
           "batch": BATCH, "seed": SEED, "levels": nl, "params_M": round(npar / 1e6, 2),
           "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "overnight_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        # eval MUST use the training imgsz — mixing them is the 896 lesson
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=imgsz, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    by = {r["name"]: r["test_map5095"] for r in res if r["test_map5095"] == r["test_map5095"]}
    a = by.get("y26_3lvl_640_b16")
    print(f"\n{'=' * 92}\n  YOLO26 OVERNIGHT — v6i, batch {BATCH} throughout, seed {SEED}\n{'=' * 92}")
    print(f"{'run':<20}{'imgsz':>6}{'par M':>7}{'lvl':>5}{'mAP50':>8}{'mAP50-95':>10}"
          f"{'vs anchor':>11}{'h':>6}")
    print('-' * 92)
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        vs = f"{(r['test_map5095'] - a) * 100:+11.2f}" if a else f"{'n/a':>11}"
        print(f"{r['name']:<20}{r['imgsz']:>6}{r['params_M']:>7.1f}{str(r['levels']):>5}"
              f"{r['test_map50'] * 100:>8.2f}{r['test_map5095'] * 100:>10.2f}{vs}{r['hours']:>6.1f}")
    print('-' * 92)
    print("  THE FOUR DELTAS THIS FILE EXISTS TO MEASURE:")

    def d(x, y, lbl):
        if x in by and y in by:
            print(f"    {lbl:<28}{(by[x] - by[y]) * 100:+7.2f} pp")
        else:
            print(f"    {lbl:<28}   (needs {x} and {y})")

    d("y26_3lvl_896_b16", "y26_3lvl_640_b16", "resolution 640 -> 896")
    d("y26_p2_896_b16", "y26_3lvl_896_b16", "P2 head, at 896")
    d("y26_m_640_b16", "y26_3lvl_640_b16", "capacity s -> m")
    d("y26_m_p2_640_b16", "y26_m_640_b16", "P2 head, at m scale")
    print("\n  v5i reference for the resolution delta: +1.26 / +1.58 / +1.36 / +1.44")
    print("  (four controlled pairs, mean +1.41). Anything near that replicates.")
    print("\n  For context ONLY — different batch, not a baseline:")
    for k, v in CONTEXT.items():
        print(f"    {k:<38}{v * 100:.2f}")
    print("\n  Read per size: resolution should move SMALL most. Per-size via")
    print("  CocoEvalAllFolders_luggage.py — and evaluate at the TRAINING imgsz.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 92}\n  YOLO26 OVERNIGHT — {len(todo)} runs (~{3.4 * len(todo):.0f} GPU-h at b{BATCH})")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print(f"  Resolution and capacity: the only two axes untested on this dataset.")
    print(f"{'=' * 92}\n")
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "cfg": r["cfg"], "imgsz": r["imgsz"],
                        "batch": BATCH, "params_M": float("nan"), "levels": "?",
                        "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
