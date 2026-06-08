#!/usr/bin/env python3
"""
Full Dataset — 4 Architecture proposals for weapon detection.

All use the best training config: topk=13, alpha=0.7, beta=4.0, cls=1.2, tight clips
All trained from yolov12s.pt pretrained weights.

CURRENT BEST: v5_tal07_full = 83.12% mAP50 (standard YOLOv12s architecture)

WEAKNESS ANALYSIS:
  - Small objects: mAP50=63.25% vs overall 83.12% (20% gap)
  - "Other" class: 60% mAP50 (hardest class)
  - P3 head (small objects) has NO attention — just A2C2f with False
  - P5 head uses C3k2 instead of A2C2f — no attention for large objects

4 PROPOSALS (ordered by risk, low to high):
  1. P3 attention:      Enable attention at P3 head (one flag change)
  2. LD-Redistribute:   More capacity at P3, less at P5 (proven on luggage)
  3. P3 attention + LD: Combine both
  4. Deep P3 + Bidi P4: Deep P3 + bidirectional P4 fusion (most complex)

Usage:
  python run_arch_proposals.py
"""

import time
import gc
import os
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58

# Best training config (v5_tal07)
TRAIN_PARAMS = {
    "cls": 1.2,
    "alpha_start": 0.7,
    "alpha_end": 0.3,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "small_obj_px": 40,
    "small_obj_boost": 2.5,
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    "iou_clip_start": 20.0,
    "iou_clip_end": 10.0,
    "dfl_clip_start": 10.0,
    "dfl_clip_end": 5.0,
    "tal_topk": 13,
    "tal_alpha": 0.7,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}

# =============================================================================
# ARCH 1: P3 Attention — simplest change, enable attention at P3 head
# 
# Current: P3 head uses A2C2f with attention=False
# Changed: P3 head uses A2C2f with attention=True, 4 heads
# 
# Why: Small objects are detected at P3. Without attention, the model
# treats all spatial positions equally. Attention lets it focus on
# regions that look like small weapons (knife tips, pistol grips).
# =============================================================================
ARCH_P3_ATTN = """# YOLOv12s + P3 Head Attention
nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 (unchanged)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, True, 4]]          # 14 — P3 with ATTENTION (was False)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# ARCH 2: LD-Redistribute — move capacity from P5 to P3
#
# Backbone P3: C3k2 reps 2→3 (more small object features)
# Backbone P5: A2C2f reps 4→2 (less large object features — they're easy)
# Head P3: A2C2f reps 2→3 (more P3 head processing)
#
# Why: Large objects (86.32% mAP50) are already easy. Small objects
# (63.25%) need more capacity. Redistribute compute where it matters.
# =============================================================================
ARCH_LD = """# YOLOv12s + LD-Redistribute (P5→P3 capacity shift)
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2,  [512, False, 0.25]]      # 4 — P3: 2→3 reps (+capacity)
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — P4: unchanged
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # 8 — P5: 4→2 reps (-capacity)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # 14 — P3 head: 2→3 reps (+capacity)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# ARCH 3: P3 Attention + LD — combine both proven ideas
#
# LD redistribution + attention at P3 head.
# More P3 capacity AND attention to focus on weapon features.
# =============================================================================
ARCH_P3_ATTN_LD = """# YOLOv12s + P3 Attention + LD-Redistribute
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2,  [512, False, 0.25]]      # 4 — P3: 2→3 reps
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — P4: unchanged
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # 8 — P5: 4→2 reps

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, True, 4]]          # 14 — P3: 3 reps + ATTENTION

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# ARCH 4: Deep P3 + Bidirectional P4
#
# Deep P3: 3x A2C2f blocks for more small object processing
# Bidi P4: P4 gets refined from both top-down AND bottom-up, then merged
# This gives P4 context from both P5 (semantic) and P3 (detail).
#
# Most complex, highest risk, highest potential.
# =============================================================================
ARCH_DEEP_P3_BIDI = """# YOLOv12s + Deep P3 + Bidirectional P4
nc: 4
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
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 top-down

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # 14 — Deep P3 (3 reps)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up

  # Bidirectional P4 merge: top-down + bottom-up
  - [[11, 17], 1, Concat, [1]]              # 18 — concat both P4 refinements
  - [-1, 1, Conv, [512, 1, 1]]              # 19 — project back to 512ch

  - [17, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 22

  - [[14, 19, 22], 1, Detect, [nc]]         # Detect: P3, P4-bidi, P5
"""


# =============================================================================
# EPOCH SYNC CALLBACK
# =============================================================================
def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


# =============================================================================
# RUNS
# =============================================================================
RUNS = [
    {
        "name": "arch_p3_attn_70",
        "desc": "P3 attention enabled (simplest change)",
        "yaml_content": ARCH_P3_ATTN,
        "batch": 58,
    },
    {
        "name": "arch_ld_70",
        "desc": "LD-Redistribute: P5→P3 capacity shift",
        "yaml_content": ARCH_LD,
        "batch": 58,
    },
    {
        "name": "arch_p3_attn_ld_70",
        "desc": "P3 attention + LD-Redistribute combined",
        "yaml_content": ARCH_P3_ATTN_LD,
        "batch": 56,
    },
    {
        "name": "arch_deep_p3_bidi_70",
        "desc": "Deep P3 + Bidirectional P4 (most complex)",
        "yaml_content": ARCH_DEEP_P3_BIDI,
        "batch": 54,
    },
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def main():
    os.makedirs(YAML_DIR, exist_ok=True)

    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — 4 ARCHITECTURE PROPOSALS")
    print(f"  Target to beat: 80.02% mAP50 (v5_tal07_70 on ablation)")
    print(f"{'=' * 70}")
    print(f"  Training:  topk=13, alpha=0.7, beta=4.0 (best config)")
    print(f"  Epochs:    {EPOCHS}    ImgSize: {IMG_SIZE}")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<35} {run['desc']}")

    print(f"{'=' * 70}\n")

    results = []
    for i, run in enumerate(RUNS):
        yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
        save_yaml(run["yaml_content"], yaml_path)

        print(f"\n{'#' * 70}")
        print(f"# [{i+1}/{len(RUNS)}] {run['name']}")
        print(f"# {run['desc']}")
        print(f"# Batch: {run['batch']}")
        print(f"{'#' * 70}\n")

        start_time = time.time()

        try:
            model = YOLO(yaml_path)
            model.load("yolov12s.pt")
            model.add_callback('on_train_epoch_start', on_train_epoch_start)

            train_kwargs = {
                "data": DATA_YAML,
                "epochs": EPOCHS,
                "imgsz": IMG_SIZE,
                "batch": run["batch"],
                "device": DEVICE,
                "workers": WORKERS,
                "project": PROJECT_DIR,
                "name": run["name"],
                "patience": 100,
                "close_mosaic": 10,
                "seed": 0,
                "deterministic": True,
            }
            train_kwargs.update(TRAIN_PARAMS)

            model.train(**train_kwargs)
            elapsed = (time.time() - start_time) / 3600
            results.append({"name": run["name"], "status": "OK", "time": elapsed})
            print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")

        except Exception as e:
            elapsed = (time.time() - start_time) / 3600
            results.append({"name": run["name"], "status": f"FAILED: {e}", "time": elapsed})
            print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<35} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
