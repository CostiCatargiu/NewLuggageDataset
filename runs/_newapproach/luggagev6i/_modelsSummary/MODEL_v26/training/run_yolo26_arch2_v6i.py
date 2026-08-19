#!/usr/bin/env python3
"""
YOLO26 ARCHITECTURE ROUND 2 — the control, plus four configs the DATA points at.

=============================================================================
WHERE THIS COMES FROM
=============================================================================
Round 1 produced two numbers and no way to read them:

    yolo26_custom-9   55.24  stock 3-level          b82
    y26_lsshift       55.58  p2 + DySample+gctx+snake  b32   <- +0.34, caused by WHAT?
    y26_gctxp3        55.09  as lsshift, + gctx at P3  b32   <- -0.49, large -4.99

Runs 1 and 2 below fix the attribution. Runs 3-5 follow the only three signals
in the YOLO26 data that point anywhere.

=============================================================================
SIGNAL 1 — gctx at P3 is HARMFUL here, and it was the BEST config on YOLOv12
=============================================================================
    YOLOv12  ls_shift_gctxP3  56.02  <- top of the entire v6i campaign
    YOLO26   y26_gctxp3       55.09  <- -0.49 vs lsshift, large -4.99

A clean inversion. If gctx is the liability on YOLO26, dropping it entirely
should beat lsshift -> y26_p2_dys_snake tests exactly that.

=============================================================================
SIGNAL 2 — NOTHING improves large. 0 of 17 runs. Mean -3.1 pp.
=============================================================================
Across every YOLO26 config, loss and architecture alike, the stock anchor still
holds the best large (60.87). The one module that touches the P5 feature is the
snake. y26_p2_snake_p3p4 removes it from P5 only and keeps everything else, so
if large recovers the snake is the cause and the fix is a one-line YAML change.

=============================================================================
SIGNAL 3 — v12's third-best architecture was never ported
=============================================================================
    ls_shift_gctxP3  56.02   ported (y26_gctxp3)
    arch_ls_shift    55.98   ported (y26_lsshift)
    arch_levelspec   55.90   NEVER PORTED   <- y26_levelspec
    ls_shift_k5      55.87
It is level-specific rather than uniform: global context on the FINE levels
where small objects live, shape prior on the COARSE levels where the elongated
trolleys live. Different hypothesis from the other two, and 0.08 pp from the top
on v12 — inside noise there, so it was never really beaten.

=============================================================================
!! TWO BATCHES, ON PURPOSE — read this before comparing anything
=============================================================================
EVERY run here is b32, matching y26_lsshift 55.58 and y26_gctxp3 55.09, and all
five are anchored on run 1. The stock baseline 55.24 was trained at b82, so the
"free P2 head" number carries a batch caveat — state it rather than hide it.

    (1)         - 55.24  = the free P2 head        (batch caveat)
    (2,3,4,5)   - (1)    = what each module adds    (clean)

Cost ~10.7 h. If you only run two, make them y26_p2_b32 and y26_p2_dysample:
together they give the anchor plus the cheapest module, in ~3.2 h.

REQUIRES the module port on the import path for runs 3-5. Runs 1 and 2 use the
SHIPPED yolo26-p2.yaml and need nothing custom.

Usage:
    python run_yolo26_arch2_v6i.py                    # all five
    python run_yolo26_arch2_v6i.py y26_p2_b32 y26_p2_dysample
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
PROJECT_DIR = "runs_yolo26_arch2_v6i"
YAML_DIR = "arch_yamls_y26_r2"
EPOCHS = 70
IMG_SIZE = 640
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CUSTOM_MODULES = ("DySample", "ZGGlobalContext2", "ZGDSConv")

# measured, for the summary
BASELINE_B82 = 0.5524      # yolo26_custom-9, stock 3-level
LSSHIFT_B32 = 0.5558       # y26_lsshift
GCTXP3_B32 = 0.5509        # y26_gctxp3

DYS_SNAKE_YAML = """# DySample + snake on P3/P4/P5, NO gctx
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

  - [22, 1, ZGDSConv, [256, 9]] # 29  P3 + snake
  - [25, 1, ZGDSConv, [512, 9]] # 30  P4 + snake
  - [28, 1, ZGDSConv, [1024, 9]] # 31  P5 + snake
  - [[19, 29, 30, 31], 1, Detect, [nc]]
"""

SNAKE_P3P4_YAML = """# lsshift MINUS the P5 snake — P5 left untouched
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

  - [19, 1, ZGGlobalContext2, [128]] # 29  P2 + gctx
  - [22, 1, ZGDSConv, [256, 9]] # 30  P3 + snake
  - [25, 1, ZGDSConv, [512, 9]] # 31  P4 + snake
  - [[29, 30, 31, 28], 1, Detect, [nc]]   # P5 = raw C3k2, no snake
"""

DYSAMPLE_YAML = """# p2 + DySample ONLY — the content-aware P3->P2 upsample, nothing else
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

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

LEVELSPEC_YAML = """# arch_levelspec analogue — gctx on P2/P3, snake on P4/P5
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

  - [19, 1, ZGGlobalContext2, [128]] # 29  P2 + gctx   (small)
  - [22, 1, ZGGlobalContext2, [256]] # 30  P3 + gctx   (small)
  - [25, 1, ZGDSConv, [512, 9]] # 31  P4 + snake  (medium)
  - [28, 1, ZGDSConv, [1024, 9]] # 32  P5 + snake  (large)
  - [[29, 30, 31, 32], 1, Detect, [nc]]
"""


RUNS = [
    {"name": "y26_p2_b32", "cfg": "yolo26-p2.yaml", "yaml": None, "batch": 32,
     "expect": {},
     "label": "stock yolo26-p2, 4 levels — THE ANCHOR for everything below",
     "why": "yolo26-p2.yaml SHIPS with ultralytics; nothing of ours is in it. That "
            "is exactly why it is needed: it measures how much of y26_lsshift's "
            "55.58 came from a free 4-level head rather than from "
            "DySample/gctx/snake. Batch 32 matches y26_lsshift and y26_gctxp3, so "
            "it anchors all five module runs. Comparing it to the b82 baseline "
            "(55.24) carries a batch caveat — state it, do not hide it."},

    {"name": "y26_p2_dysample", "cfg": "y26_p2_dysample.yaml", "yaml": DYSAMPLE_YAML,
     "batch": 32, "expect": {"DySample": 1},
     "label": "p2 + DySample only — the cheapest module, isolated",
     "why": "Completes the decomposition. With the anchor and y26_p2_dys_snake "
            "this pins every module separately:\n"
            "    DySample = dysample - p2\n"
            "    snake    = dys_snake - dysample\n"
            "    gctx(P2) = lsshift - dys_snake\n"
            "Also the cheapest run here (~1.6 h) and the one with the clearest "
            "mechanism: nearest-neighbour upsampling blurs exactly the detail "
            "small objects need at stride 4. Near-zero-init offsets, so it starts "
            "as bilinear and only departs if that helps."},

    {"name": "y26_p2_dys_snake", "cfg": "y26_p2_dys_snake.yaml", "yaml": DYS_SNAKE_YAML,
     "batch": 32, "expect": {"DySample": 1, "ZGDSConv": 3},
     "label": "p2 + DySample + snake, NO gctx — is the context gate the liability?",
     "why": "SIGNAL 1. gctx at P3 cost 0.49 overall and 4.99 on large, while the "
            "same change was the BEST config on YOLOv12 (56.02). If gctx is what "
            "hurts on an NMS-free detector, removing it entirely should beat "
            "lsshift's 55.58. Minus y26_lsshift = the gctx(P2) contribution, "
            "with sign."},

    {"name": "y26_p2_snake_p3p4", "cfg": "y26_p2_snake_p3p4.yaml", "yaml": SNAKE_P3P4_YAML,
     "batch": 32, "expect": {"DySample": 1, "ZGGlobalContext2": 1, "ZGDSConv": 2},
     "label": "lsshift MINUS the P5 snake — the large-object probe",
     "why": "SIGNAL 2. Not one of 17 YOLO26 runs improves large; the stock anchor "
            "still holds the best value at 60.87 and the mean deficit is -3.1 pp. "
            "The snake is the only module touching the P5 feature. This removes "
            "it from P5 alone and changes nothing else, so a large recovery "
            "identifies the cause and the fix is one YAML line."},

    {"name": "y26_levelspec", "cfg": "y26_levelspec.yaml", "yaml": LEVELSPEC_YAML,
     "batch": 32, "expect": {"DySample": 1, "ZGGlobalContext2": 2, "ZGDSConv": 2},
     "label": "arch_levelspec analogue — gctx on P2/P3, snake on P4/P5  [v12 55.90]",
     "why": "SIGNAL 3. v12's third-best architecture and the only one of the top "
            "three never ported. Level-SPECIFIC rather than uniform: context "
            "where small objects live, shape prior where the elongated trolleys "
            "do. On v12 it sat 0.12 pp from the top, inside noise, so it was "
            "never actually beaten. Note it keeps gctx at P3, which signal 1 "
            "says is harmful here — so it is also the cleanest test of whether "
            "that reading is right."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def resolve_cfg(rc):
    return save_yaml(rc["yaml"], os.path.join(YAML_DIR, rc["cfg"])) if rc["yaml"] else rc["cfg"]


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
                              "ZGDSConv": "ZGDSConv," in src,
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
    needs_custom = [r["name"] for r in todo if r["yaml"]]
    missing = [k for k, v in ENV["modules"].items() if not v]
    if needs_custom and missing:
        print(f"\n  [ABORT] {missing} not importable, needed by {needs_custom}.")
        print(f"          Run patch_ultralytics_modules.py, or run only")
        print(f"          y26_p2_b32 — it uses the SHIPPED yaml and needs nothing custom.")
        return False
    unreg = [k for k, v in ENV["registered"].items() if not v]
    if needs_custom and unreg:
        print(f"\n  [ABORT] not registered in parse_model: {unreg}")
        return False

    print(f"\n  [!] ALL RUNS AT b32 — comparable to y26_lsshift 55.58 and y26_gctxp3 55.09.")
    print(f"      The b82 baseline 55.24 carries a batch caveat; state it.")

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
    name, cfg, batch = rc["name"], resolve_cfg(rc), rc["batch"]
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}" + ("   (written here)" if rc["yaml"] else "   (shipped)"))
    print(f"  imgsz={IMG_SIZE}  batch={batch}  epochs={EPOCHS}  seed={SEED}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    built = count_custom(model.model)
    print(f"  custom layers: {built or 'none'}   expected: {rc['expect'] or 'none'}")
    if built != rc["expect"]:
        raise RuntimeError(
            f"{name}: graph has {built}, the yaml declares {rc['expect']}. parse_model "
            f"dropped or duplicated a module — this run would measure the wrong thing.")

    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] weight transfer failed: {e}")

    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "cfg": rc["cfg"], "cfg_path": cfg, "custom_layers": built,
           "imgsz": IMG_SIZE, "batch": batch, "seed": SEED, "hours": hours,
           "weights": weights, "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch2_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
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
    print(f"\n{'=' * 92}\n  YOLO26 ARCH ROUND 2 — s/{IMG_SIZE}, seed {SEED}\n{'=' * 92}")
    print(f"{'run':<22}{'b':>4}{'modules':<28}{'mAP50':>8}{'mAP50-95':>10}{'h':>6}")
    print('-' * 92)
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        mods = ", ".join(f"{k[:9]}x{v}" for k, v in (r.get("custom_layers") or {}).items()) or "-"
        print(f"{r['name']:<22}{r['batch']:>4}{mods:<28}{r['test_map50'] * 100:>8.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{r['hours']:>6.1f}")
    print('-' * 92)
    print(f"{'yolo26_custom-9 (known)':<22}{82:>4}{'-':<28}{'':>8}{BASELINE_B82 * 100:>10.2f}")
    print(f"{'y26_lsshift (known)':<22}{32:>4}{'all three':<28}{'':>8}{LSSHIFT_B32 * 100:>10.2f}")
    print(f"{'y26_gctxp3 (known)':<22}{32:>4}{'all three + gctxP3':<28}{'':>8}{GCTXP3_B32 * 100:>10.2f}")

    a = by.get("y26_p2_b32")
    print("\n  THE DELTAS — everything below is b32, anchored on y26_p2_b32:")
    if a:
        print(f"    free P2 head vs the b82 baseline   {(a - BASELINE_B82) * 100:+7.2f} pp"
              f"   <- BATCH CAVEAT, b32 vs b82")
        for n, lbl in (("y26_p2_dysample", "DySample alone"),
                       ("y26_p2_dys_snake", "DySample + snake, no gctx"),
                       ("y26_p2_snake_p3p4", "lsshift minus the P5 snake"),
                       ("y26_levelspec", "levelspec, gctx P2/P3 + snake P4/P5")):
            if n in by:
                print(f"    {lbl:<35}{(by[n] - a) * 100:+7.2f} pp")
        print(f"    {'all three (y26_lsshift, known)':<35}{(LSSHIFT_B32 - a) * 100:+7.2f} pp")
        print(f"    {'+ gctxP3 (y26_gctxp3, known)':<35}{(GCTXP3_B32 - a) * 100:+7.2f} pp")

        print("\n  PER-MODULE, by subtraction:")
        if "y26_p2_dysample" in by:
            print(f"    DySample = dysample - p2           {(by['y26_p2_dysample'] - a) * 100:+7.2f} pp")
        if "y26_p2_dysample" in by and "y26_p2_dys_snake" in by:
            print(f"    snake    = dys_snake - dysample    "
                  f"{(by['y26_p2_dys_snake'] - by['y26_p2_dysample']) * 100:+7.2f} pp")
        if "y26_p2_dys_snake" in by:
            print(f"    gctx(P2) = lsshift - dys_snake     "
                  f"{(LSSHIFT_B32 - by['y26_p2_dys_snake']) * 100:+7.2f} pp")
        if "y26_p2_snake_p3p4" in by:
            print(f"    P5 snake = lsshift - snake_p3p4    "
                  f"{(LSSHIFT_B32 - by['y26_p2_snake_p3p4']) * 100:+7.2f} pp")
        print("    if the singles do NOT sum to lsshift's delta, the modules interact")
    else:
        print("    (needs y26_p2_b32 — nothing here is readable without it)")
    print("\n  READ LARGE PER SIZE. Anchor 60.87; no run of 17 has beaten it.")
    print("  y26_p2_snake_p3p4 is the one config that could. Per-size:")
    print("  CocoEvalAllFolders_luggage.py on best.pt")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    h = sum(1.6 if not r["yaml"] else 2.5 for r in todo)
    print(f"\n{'=' * 92}\n  YOLO26 ARCH ROUND 2 — {len(todo)} runs (~{h:.1f} GPU-h)")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print(f"  If you only run two: y26_p2_b32 and y26_p2_dysample.")
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
            res.append({"name": r["name"], "cfg": r["cfg"], "batch": r["batch"],
                        "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
