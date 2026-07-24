#!/usr/bin/env python3
"""
70% Ablation — Round 24: ATTACK RANKING, NOT FINDING (2 new archs).

Every prior architecture (rounds 1-23) improved how the model FINDS objects
(receptive field, scale, fusion, deep supervision). But the bottleneck is not
finding -- "other" has recall ~0.84. It is RANKING / precision -- AP ~0.51, i.e.
the model finds "other" objects and scores them wrong. This round is the first
to target classification/ranking at the architecture level:

  1. r24_decoupled_70  DetectDecoupled -- box branch reads the main neck, cls
     branch reads a DEDICATED cls feature pathway (separate conv weights). Tests
     whether cls features are weak because they're shared with the box objective
     (r18's deeper cls head couldn't isolate this -- it used the shared feature).

  2. r24_obj_70        DetectObj -- re-adds an explicit objectness (foreground/
     background) branch (YOLOv12 has none). score = sigmoid(cls + obj) at
     inference, so background-like anchors are suppressed -- directly attacking
     the false-positive / precision failure. Objectness is supervised by BCE vs
     the TAL foreground mask (DetectObjLoss).

Both yolov12s/640, default TAL, fixed batch 48. Read per-class "other" AP50 and
PRECISION (the metric these target). Confirm any winner at seeds.

Honest prior: ~15-20% -- these aim at the right failure mode (the first in the
search to do so), but the residual ceiling still looks like an "other" label
problem that no architecture fixes.

NOTE: framework-level changes (custom head + custom loss for DetectObj).
SMOKE-TEST FIRST (snippet below) -- could not be executed in authoring env.

Bars: 78.45 baseline | 79.40 r11_widefuse | 80.45 v5_topk15_beta3 (loss-only).

Usage:
  python run_round24_ranking.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 50

BASE_0_20 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head (box)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up (box)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head (box)
"""

# 1: DetectDecoupled — dedicated cls feature pathway (separate C3k2 per scale)
ARCH_DECOUPLED = BASE_0_20 + """
  - [14, 1, C3k2, [256, False]]                       # 21 — cls_p3 (dedicated)
  - [17, 1, C3k2, [512, False]]                       # 22 — cls_p4
  - [20, 1, C3k2, [1024, True]]                       # 23 — cls_p5
  - [[14, 17, 20, 21, 22, 23], 1, DetectDecoupled, [nc]]  # 24 — box: 14/17/20, cls: 21/22/23
"""

# 2: DetectObj — explicit objectness branch (Detect stays at index 21)
ARCH_OBJ = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectObj, [nc]]                # 21 — + objectness branch
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # {"name": "r24_decoupled_70",
    #  "desc": "[1/2] DetectDecoupled — box uses main neck, cls uses dedicated cls pathway",
    #  "yaml_content": ARCH_DECOUPLED, "batch": BATCH, "seed": 0, "epochs": 80},
    {"name": "r24_obj_70",
     "desc": "[2/2] DetectObj — explicit objectness branch, score=sigmoid(cls+obj) (precision/FP suppression)",
     "yaml_content": ARCH_OBJ, "batch": BATCH, "seed": 0, "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() then remap Detect keys if the index shifted. Transfers
    backbone + neck + the box branch (cv2/dfl); new branches train fresh."""
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v
                for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} Detect keys transferred on top")
    return model


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Seed: {run['seed']}   Epochs: {run.get('epochs', 80)}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=run.get("epochs", 80),
            imgsz=IMG_SIZE,
            batch=run["batch"],
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=run["seed"],
            deterministic=True,
            **TAL_DEFAULT,
            alpha_start=0.0,
            alpha_end=0.0,
            alpha_min=0.0,
            alpha_max=0.0,
            iou_clip_start=999.0,
            iou_clip_end=999.0,
            dfl_clip_start=999.0,
            dfl_clip_end=999.0,
            small_obj_boost=1.0,
            small_obj_px=0,
            center_loss_weight_init=0.0,
            center_loss_weight_min=0.0,
            use_vfl=False,
        )

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — ROUND 24: RANKING-TARGETED ARCHS (decoupled + objectness)")
    print(f"  Bars: 78.45 baseline | 79.40 r11_widefuse | 80.45 v5_topk15_beta3")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<20} batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(RUNS):
        print(f"\n>>> Run {i+1}/{len(RUNS)}: {run['name']}")
        results.append(run_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<20} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
