#!/usr/bin/env python3
"""
YOLO26 ARCH ROUND 3 — two configs around the winner. Both cheap, both untested.

=============================================================================
WHERE THE WINNER CAME FROM
=============================================================================
The round-2 ablation, anchored on y26_p2_b32 = 55.03 (all b32, s, 640):

    y26_p2_dysample    55.94   +0.91   <- DySample does ALL the work
    y26_lsshift        55.58   +0.55       and it is the CHEAPEST module
    y26_p2_snake_p3p4  55.07   +0.04
    y26_p2_b32         55.03    ANCHOR
    y26_p2_dys_snake   54.86   -0.17   <- adding the snake goes BELOW the anchor
    y26_levelspec      51.93   -3.10

    DySample  = dysample  - p2         +0.91
    snake     = dys_snake - dysample   -1.08   (actively harmful, and expensive)
    gctx(P2)  = lsshift   - dys_snake  +0.72

=============================================================================
RUN 1 — the arithmetic says this should be the best model in the project
=============================================================================
gctx's +0.72 was measured on top of DySample+snake. On top of DySample ALONE it
has never been run. If it carries:

    55.94 + 0.72 = ~56.6

which would beat everything measured on either detector, including YOLOv12's
ls_shift_gctxP3 at 56.02. Even if it only half-carries it still wins.

The reason to doubt it: y26_gctxp3 (gctx at P3) was -0.49 and cost 4.99 on large,
so gctx is not uniformly good here. But that was gctx at P3; this is gctx at P2,
which is the placement that has only ever appeared alongside the snake.

No snake -> ~1.6 h, and near-zero inference cost. This is the run to do first.

=============================================================================
RUN 2 — the port was incomplete, and the missing piece is the module that works
=============================================================================
v12's arch_ls_shift replaces ALL THREE upsamples with DySample:

    v12 layer  9  DySample  P5->P4        my port  layer 11  nn.Upsample  MISSING
    v12 layer 12  DySample  P4->P3        my port  layer 14  nn.Upsample  MISSING
    v12 layer 21  DySample  P3->P2        my port  layer 17  DySample     present

Everything measured so far used ONE of the three. Since that one is worth +0.91,
the two that were left out are the largest untested quantity in the project.
Channels check out at scale s: 512 and 256, both divisible by groups=4.

If DySample is additive, three could be worth ~+2 over the anchor. If the P3->P2
one is special — it is the only one feeding the stride-4 level where the small
objects are — the other two will do nothing, and that is worth knowing too.

=============================================================================
READ LARGE
=============================================================================
y26_p2_dysample gains +1.37 small and +0.64 medium but loses 3.77 on large (vs
the b32 anchor). Nothing in 22 YOLO26 runs has improved large. If either config
here holds small while recovering large, that is the more useful result than
another 0.5 pp of aggregate mAP.

Batch 32, imgsz 640, seed 0 — matches round 1 and round 2 exactly.
REQUIRES the module port on the import path.

Usage:
    python run_yolo26_arch3_v6i.py
    python run_yolo26_arch3_v6i.py y26_p2_dys_gctx
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
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_arch3_v6i"
YAML_DIR = "arch_yamls_y26_r3"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CUSTOM_MODULES = ("DySample", "ZGGlobalContext2", "ZGDSConv")

ANCHOR_B32 = 0.5503        # y26_p2_b32, stock p2
BEST_B32 = 0.5594          # y26_p2_dysample  <- the run to beat
BEST_SMALL = 0.5239        # y26_p2_dysample, best small in the project
LSSHIFT_B32 = 0.5558
V12_BEST = 0.5602          # ls_shift_gctxP3, best on YOLOv12

DYS_GCTX_YAML = """# BEST + gctx(P2), still NO snake
nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [19, 1, ZGGlobalContext2, [128]] # 29  P2 + gated global context
  - [[29, 22, 25, 28], 1, Detect, [nc]]
"""

DYS3_YAML = """# DySample at ALL THREE upsample points — the faithful arch_ls_shift upsampler
nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, DySample, [2]] # 11  content-aware P5 -> P4
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, DySample, [2]] # 14  content-aware P4 -> P3
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""


RUNS = [
    {"name": "y26_p2_dys_gctx", "cfg": "y26_p2_dys_gctx.yaml", "yaml": DYS_GCTX_YAML,
     "expect": {"DySample": 1, "ZGGlobalContext2": 1},
     "label": "DySample + ZGGlobalContext2(P2), NO snake — the arithmetic favourite",
     "why": "The two modules that HELPED, without the one that hurt. gctx's +0.72 "
            "was only ever measured on top of DySample+snake; on top of DySample "
            "alone it is untested. If it carries, 55.94 + 0.72 = ~56.6, past "
            "YOLOv12's 56.02 and the best model in the project. Caveat: gctx at P3 "
            "was -0.49 here, so gctx is not uniformly good on YOLO26 — but that was "
            "P3, and this is P2. Near-zero inference cost either way."},

    {"name": "y26_p2_dys3", "cfg": "y26_p2_dys3.yaml", "yaml": DYS3_YAML,
     "expect": {"DySample": 3},
     "label": "DySample at ALL THREE upsamples — the faithful arch_ls_shift upsampler",
     "why": "Every YOLO26 run so far used ONE DySample; v12's arch_ls_shift uses "
            "THREE (both top-down upsamples as well). That was an omission in my "
            "port. Since the one we did use is worth +0.91 — the largest single "
            "architecture effect measured on YOLO26 — the two left out are the "
            "biggest untested quantity here. If DySample is additive this could "
            "reach ~+2 over the anchor; if the P3->P2 one is special because it is "
            "the only one feeding stride 4, the others do nothing and that is a "
            "clean mechanism result."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def env_provenance():
    info = {"ultralytics_path": None, "tasks_md5": None, "modules": {}, "registered": {}}
    try:
        import ultralytics
        import ultralytics.nn.modules as M
        import ultralytics.nn.modules.block as _b
        import ultralytics.nn.tasks as _t
        info["ultralytics_path"] = os.path.dirname(ultralytics.__file__)
        p = getattr(_t, "__file__", None)
        if p and os.path.exists(p):
            info["tasks_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
        for n in CUSTOM_MODULES:
            info["modules"][n] = hasattr(M, n) or hasattr(_b, n)
        src = open(_t.__file__, encoding="utf-8").read()
        info["registered"] = {"ZGGlobalContext2": "ZGGlobalContext2," in src,
                              "DySample": "elif m is DySample:" in src}
    except Exception as e:
        info["import_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  ultralytics : {ENV.get('ultralytics_path')}")
    print(f"  tasks.py md5: {ENV.get('tasks_md5')}")
    print(f"  importable  : {ENV.get('modules')}")
    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import ultralytics: {ENV['import_error']}")
        return False
    missing = [k for k, v in ENV["modules"].items() if not v]
    if missing:
        print(f"\n  [ABORT] not importable: {missing}. Run patch_ultralytics_modules.py.")
        return False
    unreg = [k for k, v in ENV["registered"].items() if not v]
    if unreg:
        print(f"\n  [ABORT] not registered in parse_model: {unreg}")
        return False
    print(f"\n  [!] b{BATCH}/{IMG_SIZE}/seed{SEED} — matches rounds 1 and 2.")
    print(f"      Anchor y26_p2_b32 {ANCHOR_B32 * 100:.2f} | to beat: y26_p2_dysample "
          f"{BEST_B32 * 100:.2f} | project best (YOLOv12) {V12_BEST * 100:.2f}")
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


def count_custom(model):
    seen = {}
    for m in model.modules():
        n = type(m).__name__
        if n in CUSTOM_MODULES:
            seen[n] = seen.get(n, 0) + 1
    return seen


def run_one(rc):
    name = rc["name"]
    cfg = save_yaml(rc["yaml"], os.path.join(YAML_DIR, rc["cfg"]))
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    built = count_custom(model.model)
    print(f"  custom layers: {built}   expected: {rc['expect']}")
    if built != rc["expect"]:
        raise RuntimeError(
            f"{name}: graph has {built}, the yaml declares {rc['expect']}. parse_model "
            f"dropped or duplicated a module.")

    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] weight transfer failed: {e}")

    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "cfg": rc["cfg"], "cfg_path": cfg, "custom_layers": built,
           "imgsz": IMG_SIZE, "batch": BATCH, "seed": SEED, "hours": hours,
           "weights": weights, "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch3_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
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
    print(f"\n{'=' * 84}\n  YOLO26 ARCH ROUND 3 — s/{IMG_SIZE}/b{BATCH}, seed {SEED}\n{'=' * 84}")
    print(f"{'run':<22}{'modules':<26}{'mAP50':>8}{'mAP50-95':>10}{'vs best':>9}{'h':>6}")
    print('-' * 84)
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        mods = ", ".join(f"{k[:9]}x{v}" for k, v in (r.get("custom_layers") or {}).items())
        print(f"{r['name']:<22}{mods:<26}{r['test_map50'] * 100:>8.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{(r['test_map5095'] - BEST_B32) * 100:>+9.2f}{r['hours']:>6.1f}")
    print('-' * 84)
    print(f"{'y26_p2_dysample (best)':<22}{'DySample x1':<26}{'':>8}{BEST_B32 * 100:>10.2f}")
    print(f"{'y26_lsshift':<22}{'all three':<26}{'':>8}{LSSHIFT_B32 * 100:>10.2f}")
    print(f"{'y26_p2_b32 (anchor)':<22}{'-':<26}{'':>8}{ANCHOR_B32 * 100:>10.2f}")
    print(f"\n  vs anchor {ANCHOR_B32 * 100:.2f}:")
    for n in ("y26_p2_dys_gctx", "y26_p2_dys3"):
        if n in by:
            print(f"    {n:<20}{(by[n] - ANCHOR_B32) * 100:+7.2f} pp")
    if "y26_p2_dys_gctx" in by:
        print(f"\n    gctx(P2) on top of DySample = {(by['y26_p2_dys_gctx'] - BEST_B32) * 100:+.2f} pp")
        print(f"      (it was +0.72 on top of DySample+snake — does it carry?)")
    if "y26_p2_dys3" in by:
        print(f"    2 extra DySamples           = {(by['y26_p2_dys3'] - BEST_B32) * 100:+.2f} pp")
        print(f"      (the first one alone was +0.91 — additive, or is P3->P2 special?)")
    top = max(by.values()) if by else 0
    if top > V12_BEST:
        print(f"\n  *** {top * 100:.2f} beats YOLOv12's best ({V12_BEST * 100:.2f}) — "
              f"seed-confirm before reporting it.")
    print(f"\n  READ LARGE. y26_p2_dysample is -3.77 there vs the anchor and nothing in")
    print(f"  22 runs has improved it. Holding small while recovering large beats")
    print(f"  another 0.5 pp of aggregate. Per-size: CocoEvalAllFolders_luggage.py")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 84}\n  YOLO26 ARCH ROUND 3 — {len(todo)} runs (~{1.7 * len(todo):.1f} GPU-h, no snake convs)")
    print(f"  {', '.join(r['name'] for r in todo)}\n{'=' * 84}\n")
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
            res.append({"name": r["name"], "cfg": r["cfg"], "hours": float("nan"),
                        "error": str(e), "test_map50": float("nan"),
                        "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
