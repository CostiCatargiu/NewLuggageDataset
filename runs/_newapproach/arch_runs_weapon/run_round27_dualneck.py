#!/usr/bin/env python3
"""
70% Ablation — Round 27: DUAL-NECK DECOUPLED (escalate the one signal).

The decoupled cls head is the only consistent positive: top-2 on validation
(r25_widefuse_decoupled 78.98, r26_decoupledobj 78.75) and highest precision in
the project. But it only half-decouples -- cls gets a per-scale conv whose
features still inherit the BOX neck's cross-scale fusion. This round fully
decouples: a parallel, dedicated CLS feature pyramid (its own top-down+bottom-up
neck), so classification gets multi-scale fusion optimized purely for
discrimination, independent of localization from the backbone up.

  box branch  <- standard PAN  (box P3/P4/P5)
  cls branch  <- separate cls neck (cls P3/P4/P5)
  DetectDecoupled routes box features -> cv2, cls features -> cv3.

Two runs (yolov12s/640, default TAL, batch 48):
  1. r27_dualneck_stock_70     stock backbone + dual neck
  2. r27_dualneck_widefuse_70  widefuse backbone + dual neck (best backbone)

Read precision + "other" AP50 vs the decoupled runs (r25/r26) -- does a full cls
neck beat the half-decoupled per-scale version? Seed-check any winner.

No new modules (DetectDecoupled already implemented). SMOKE-TEST FIRST -- the
dual-neck index routing is the risky spot.

Bars: 79.40 r11_widefuse | r25_decoupled val 78.98 | r26_decoupledobj val 78.75.

Usage:
  python run_round27_dualneck.py
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

# 1: stock backbone + dual neck (box neck 9-20, cls neck 21-32, Detect 33)
ARCH_DUALNECK_STOCK = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4 — backbone P3
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — backbone P4
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8 — backbone P5

head:
  # ---- BOX neck (standard PAN) ----
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]             # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]             # 14 — box-P3
  - [-1, 1, Conv, [256, 3, 2]]                   # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]             # 17 — box-P4
  - [-1, 1, Conv, [512, 3, 2]]                   # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                  # 20 — box-P5
  # ---- CLS neck (parallel, dedicated to classification) ----
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]    # 21
  - [[-1, 6], 1, Concat, [1]]                     # 22
  - [-1, 2, C3k2, [512, False, 0.25]]            # 23 — cls-P4-td
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 24
  - [[-1, 4], 1, Concat, [1]]                     # 25
  - [-1, 2, C3k2, [256, False, 0.25]]            # 26 — cls-P3
  - [-1, 1, Conv, [256, 3, 2]]                   # 27
  - [[-1, 23], 1, Concat, [1]]                    # 28
  - [-1, 2, C3k2, [512, False, 0.25]]            # 29 — cls-P4
  - [-1, 1, Conv, [512, 3, 2]]                   # 30
  - [[-1, 8], 1, Concat, [1]]                     # 31
  - [-1, 2, C3k2, [1024, True]]                  # 32 — cls-P5
  - [[14, 17, 20, 26, 29, 32], 1, DetectDecoupled, [nc]]   # 33
"""

# 2: widefuse backbone + dual neck (widefuse @ 21, cls neck 22-33, Detect 34)
ARCH_DUALNECK_WIDEFUSE = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4 — backbone P3
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — backbone P4
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8 — backbone P5

head:
  # ---- BOX neck (standard PAN) ----
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]             # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]             # 14 — box-P3
  - [-1, 1, Conv, [256, 3, 2]]                   # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]             # 17 — box-P4 (raw)
  - [-1, 1, Conv, [512, 3, 2]]                   # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                  # 20 — box-P5
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]       # 21 — box-P4 (widefuse = r11)
  # ---- CLS neck (parallel, dedicated to classification) ----
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]    # 22
  - [[-1, 6], 1, Concat, [1]]                     # 23
  - [-1, 2, C3k2, [512, False, 0.25]]            # 24 — cls-P4-td
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 25
  - [[-1, 4], 1, Concat, [1]]                     # 26
  - [-1, 2, C3k2, [256, False, 0.25]]            # 27 — cls-P3
  - [-1, 1, Conv, [256, 3, 2]]                   # 28
  - [[-1, 24], 1, Concat, [1]]                    # 29
  - [-1, 2, C3k2, [512, False, 0.25]]            # 30 — cls-P4
  - [-1, 1, Conv, [512, 3, 2]]                   # 31
  - [[-1, 8], 1, Concat, [1]]                     # 32
  - [-1, 2, C3k2, [1024, True]]                  # 33 — cls-P5
  - [[14, 21, 20, 27, 30, 33], 1, DetectDecoupled, [nc]]   # 34
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r27_dualneck_stock_70",
     "desc": "[1/2] stock backbone + dual neck (separate box & cls feature pyramids)",
     "yaml_content": ARCH_DUALNECK_STOCK, "batch": BATCH, "seed": 0, "epochs": 80},
    {"name": "r27_dualneck_widefuse_70",
     "desc": "[2/2] widefuse backbone + dual neck (best backbone + full cls decoupling)",
     "yaml_content": ARCH_DUALNECK_WIDEFUSE, "batch": BATCH, "seed": 0, "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() then remap Detect keys if the index shifted. Box neck +
    backbone transfer; the dedicated cls neck and cls head train fresh."""
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
    print(f"  70% ABLATION — ROUND 27: DUAL-NECK DECOUPLED (separate cls pyramid)")
    print(f"  Bars: 79.40 r11_widefuse | r25_decoupled val 78.98 | r26_decoupledobj val 78.75")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<26} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<26} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
