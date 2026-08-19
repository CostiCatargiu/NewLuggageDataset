#!/usr/bin/env python3
"""
YOLO26 DYSAMPLE SWEEP — tune the module that carried the architecture gain.

=============================================================================
WHY DYSAMPLE
=============================================================================
Decomposed at matched batch (b32, all single-factor from y26_p2_b32 = 55.03):

    DySample x1 at P3->P2      +1.05    (56.08, n=10 -> the only module that won)
    DySample x3                -0.54
    + ZGGlobalContext2         -1.16
    + ZGDSConv (snake)         -1.08
    P2 branch 128 -> 256 ch    -0.50

One module, one placement, one count. Everything added on top made it worse.

=============================================================================
WHAT IS ACTUALLY UNTUNED
=============================================================================
Every run used the library default:

    DySample(c1=128, scale=2, groups=4)
        offset = Conv2d(128, 2*groups*scale*scale = 32, k=1)  ->  4,128 params

    scale=2   fixed by the upsampling factor, not a free parameter
    groups=4  NEVER varied in any run. c1=128 at P2, so 1/2/4/8/16/32/64/128
              are all legal.
    count     1 measured (56.08), 3 measured (54.49). TWO never measured.

`groups` sets how many independent offset fields the upsampler predicts. It is
the module's only real knob and it has been at its default throughout.

=============================================================================
WHAT A FLAT RESULT WOULD MEAN — read this before hoping for a win
=============================================================================
The realistic outcome is that all four land near 56.08. That is not a wasted
night: "the gain comes from DySample at its DEFAULT configuration and is
insensitive to `groups` over 2-16" is a STRONGER claim than a tuned constant,
because it means the result transfers without retuning. Reviewers trust a broad
optimum more than a fitted one.

A win is possible but the prior is modest: the offset conv is 4k parameters and
groups only rescales it. Expect characterisation, not a jump.

=============================================================================
CONTROL — no new baseline needed
=============================================================================
    y26_p2_dysample (groups=4)   56.08 +- 0.19   n=10, b32/640/seed0
    y26_p2_b32 (no DySample)     55.03           n=1,  same batch

Every run below is a single-factor change from the 56.08 config, so that
10-replicate mean IS the control. Apparent noise on this family is 0.19 pp
(a LOWER bound - those ten shared seed 0).

    beats 56.08 by > 0.40   real, worth adopting
    within 0.20 of 56.08    flat -> the default is fine, report insensitivity
    below 55.70             that direction is wrong

PRIORITY NOTE. This refines the side already measured ten times. The reference
side - stock yolo26s at b32 - is still n=1 at the WRONG batch (b82), so the
headline "+0.84 vs baseline" stays confounded no matter how this sweep lands.
If you have one night, run the two b32 baselines first.

Batch 32, imgsz 640, seed 0 — matches every recorded architecture run.
REQUIRES the module port (DySample) on the import path. No new code.

Usage:
    python run_yolo26_dysample_sweep_v6i.py
    python run_yolo26_dysample_sweep_v6i.py y26_dys_g8 y26_dys_2of3
"""

import gc
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_dysample_sweep_v6i"
YAML_DIR = "arch_yamls_y26_dyssweep"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CENTRE = 56.08      # y26_p2_dysample, groups=4, n=10
CENTRE_SD = 0.19
NO_DYSAMPLE = 55.03  # y26_p2_b32, same batch, n=1
DYS3 = 54.49         # three DySamples, for the count axis

YAML_G8 = """nc: 3
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

  - [-1, 1, DySample, [2, 8]] # 17  P3 -> P2, groups=8
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

YAML_G2 = """nc: 3
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

  - [-1, 1, DySample, [2, 2]] # 17  P3 -> P2, groups=2
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

YAML_G16 = """nc: 3
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

  - [-1, 1, DySample, [2, 16]] # 17  P3 -> P2, groups=16
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

YAML_2OF3 = """nc: 3
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
    {"name": "y26_dys_g8", "yaml": YAML_G8, "expect": {"n_dys": 1, "groups": 8},
     "label": "groups 4 -> 8  (offset conv 32 -> 64 ch, 4,128 -> 8,256 params)",
     "why": "The first half of the only knob this module has. More groups means "
            "more independent offset fields per location, so the upsampler can "
            "warp different channel subsets differently - the plausible direction "
            "if the P3->P2 gain is limited by offset expressiveness rather than "
            "by the idea itself. Paired with g2 below it makes groups a curve; "
            "alone it would only show a direction."},

    {"name": "y26_dys_g2", "yaml": YAML_G2, "expect": {"n_dys": 1, "groups": 2},
     "label": "groups 4 -> 2  (offset conv 32 -> 16 ch, 4,128 -> 2,064 params)",
     "why": "The other side, and the more useful null. If HALVING the offset "
            "capacity costs nothing, the mechanism does not depend on offset "
            "expressiveness at all and the claim becomes 'a minimal content-aware "
            "upsampler suffices' - cheaper to deploy and more portable than a "
            "tuned value. With g8 and the default at 4 this gives three points."},

    {"name": "y26_dys_2of3", "yaml": YAML_2OF3, "expect": {"n_dys": 2, "groups": 4},
     "label": "DySample at P4->P3 AND P3->P2 (layers 14 and 17), groups=4",
     "why": "The missing middle of the count axis. One DySample gives 56.08, "
            "three gives 54.49 - so more is worse, but TWO has never been "
            "measured and the useful count is somewhere in 1-2. This adds it at "
            "the P4->P3 upsample, the one feeding the level where COCO-small "
            "objects mostly live, while leaving P5->P4 (which serves large) "
            "alone. If two matches one, the mechanism is specifically about the "
            "stride-4 path and should be described that way."},

    {"name": "y26_dys_g16", "yaml": YAML_G16, "expect": {"n_dys": 1, "groups": 16},
     "label": "groups 4 -> 16  (offset conv 32 -> 128 ch, 4,128 -> 16,512 params)",
     "why": "Extends the groups axis to four points so a trend can be told from "
            "scatter. Lowest prior of the four: at 16 groups each field covers "
            "only 8 of the 128 channels, and the offset conv starts to rival the "
            "C3k2 blocks around it in parameter count, which is where a "
            "near-zero-init module usually stops behaving like a drop-in. Run it "
            "last, or skip it if the first three are flat."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def preflight(todo):
    try:
        import ultralytics
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    if not hasattr(M, "DySample"):
        print("  [ABORT] DySample not importable — run patch_ultralytics_modules.py")
        return False
    src = open(T.__file__, encoding="utf-8").read()
    if "elif m is DySample:" not in src:
        print("  [ABORT] DySample not registered in parse_model")
        return False
    if "args = [c1, *args]" not in src:
        print("  [ABORT] parse_model does not forward extra DySample args, so the")
        print("  groups value in the yaml would be IGNORED and every run would be")
        print("  a replicate of the default. Fix tasks.py before running.")
        return False
    print("  DySample importable + registered + args forwarded    OK")
    print()
    print(f"  MODEL {MODEL_WEIGHTS}  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  control y26_p2_dysample (groups=4) = {CENTRE:.2f} +- {CENTRE_SD:.2f}  (n=10)")
    print(f"  no DySample (y26_p2_b32)           = {NO_DYSAMPLE:.2f}   -> the module is worth +{CENTRE - NO_DYSAMPLE:.2f}")
    print(f"  beating {CENTRE:.2f} by >0.40 is real; within 0.20 means the default is fine.")
    print(f"  [!] the b32 STOCK baseline is still missing — the headline '+0.84 vs")
    print(f"      baseline' stays confounded regardless of how this sweep lands.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print()
        print("  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def check_graph(model, rc):
    """Verify the BUILT graph has the requested DySample count and groups."""
    mods = [m for m in model.model.modules() if type(m).__name__ == "DySample"]
    n, want = len(mods), rc["expect"]
    if n != want["n_dys"]:
        raise RuntimeError(f"{rc['name']}: graph has {n} DySample, yaml declares {want['n_dys']}")
    for m in mods:
        g = int(getattr(m, "groups", -1))
        if g != want["groups"]:
            raise RuntimeError(
                f"{rc['name']}: DySample built with groups={g}, expected {want['groups']}. "
                f"parse_model is not forwarding the yaml arg — this run would be a "
                f"replicate of the default.")
    ch = [int(m.offset.in_channels) for m in mods]
    off = [int(m.offset.out_channels) for m in mods]
    print(f"  [guard] DySample x{n}  groups={want['groups']}  in_ch={ch}  offset_out={off}")
    return {"n_dys": n, "groups": want["groups"], "in_ch": ch, "offset_out": off}


def run_one(rc):
    name = rc["name"]
    cfg = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{name}.yaml"))
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  cfg={cfg}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}")
    print()
    t0 = time.time()

    model = YOLO(cfg)
    built = check_graph(model, rc)
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
    out = {"name": name, "built": built, "expect": rc["expect"], "seed": SEED,
           "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "dys_sweep_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    by = {r["name"]: r["test_map5095"] * 100 for r in ok}
    print()
    print("=" * 84)
    print(f"  DYSAMPLE SWEEP — b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 84)
    print(f"{'run':<16}{'n':>3}{'groups':>8}{'mAP50':>9}{'mAP50-95':>10}{'vs centre':>11}   verdict")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        d = v - CENTRE
        vd = ("BETTER" if d > 0.40 else "wrong direction" if v < 55.70 else "flat")
        print(f"{r['name']:<16}{r['expect']['n_dys']:>3}{r['expect']['groups']:>8}"
              f"{r['test_map50'] * 100:>9.2f}{v:>10.2f}{d:>+11.2f}   {vd}")
    print("-" * 84)
    print(f"{'y26_p2_dysample':<16}{1:>3}{4:>8}{'':>9}{CENTRE:>10.2f}   <- centre, n=10 +-{CENTRE_SD}")
    print(f"{'y26_p2_b32':<16}{0:>3}{'-':>8}{'':>9}{NO_DYSAMPLE:>10.2f}   <- no DySample")

    print("\n  groups axis (1 DySample at P3->P2):")
    for lbl, n in (("2", "y26_dys_g2"), ("4", None), ("8", "y26_dys_g8"), ("16", "y26_dys_g16")):
        v = CENTRE if n is None else by.get(n)
        if v:
            print(f"    groups={lbl:<3} {v:.2f}{'   <- centre (default)' if n is None else ''}")
    print("\n  count axis (groups=4):")
    print(f"    0 DySample   {NO_DYSAMPLE:.2f}")
    print(f"    1 DySample   {CENTRE:.2f}   <- centre")
    if "y26_dys_2of3" in by:
        print(f"    2 DySample   {by['y26_dys_2of3']:.2f}")
    print(f"    3 DySample   {DYS3:.2f}")

    g = [by[n] for n in ("y26_dys_g2", "y26_dys_g8", "y26_dys_g16") if n in by]
    if g and max(g + [CENTRE]) - min(g + [CENTRE]) < 0.40:
        print("\n  groups is FLAT across the tested range. Report it that way: the gain")
        print("  comes from DySample at its default configuration and does not require")
        print("  tuning. That is a stronger and more portable claim than a fitted value.")
    elif g:
        print(f"\n  groups spans {max(g + [CENTRE]) - min(g + [CENTRE]):.2f} pp — there is a")
        print("  trend. Confirm the winner's per-size profile matches the centre's")
        print("  (small up, large down) before adopting it.")
    print(f"\n  Per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print(f"  Do NOT read LARGE at n=1 — sd 2.11 pp on this dataset.")
    print(f"  The b32 stock baseline is still the missing piece for the headline.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<16} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print()
    print("=" * 84)
    print(f"  YOLO26 DYSAMPLE SWEEP — {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 84)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "expect": r["expect"], "seed": SEED,
                        "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
