#!/usr/bin/env python3
"""
70% Ablation — 3 Architecture proposals targeting overall improvement.

KEY INSIGHT:
  - 68% of objects are large (COCO threshold) → P5 matters most
  - P5 head is the ONLY head using C3k2 instead of A2C2f
  - Every other path uses A2C2f area-attention
  - P5 at 20x20 = 400 positions → attention is extremely cheap
  - This is a consistency fix, not a speculative change

PROPOSALS:
  1. P5 A2C2f — replace C3k2 at P5 head with A2C2f + attention
  2. SKA — LSKA at P3 head (large kernel context, proven on luggage)
  3. SKA + P5 — combine both
  4. Deeper P3 — backbone P3 2→3 reps (no P5 reduction)
  5. P5 A2C2f + Deeper P3 — both combined

All use DEFAULT training params to isolate architecture effect.
Target to beat: baseline 78.45% mAP50

Usage:
  python run_arch_p5_ablation.py
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
IMG_SIZE = 640

# =============================================================================
# ARCH 1: P5 head A2C2f with attention (replace C3k2)
# Only change: layer 20 C3k2 → A2C2f True, 1 head
# 20x20 = 400 positions — cheapest attention possible
# =============================================================================
ARCH_P5_ATTN = """# YOLOv12s + P5 Head A2C2f Attention (replace C3k2)
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # 20 — A2C2f with ATTENTION (was C3k2)

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# ARCH 2: Deeper backbone P3 (2→3 reps, no P5 reduction)
# Only change: backbone layer 4 C3k2 reps 2→3
# Unlike LD, we do NOT reduce P5 — just add capacity at P3
# =============================================================================
ARCH_DEEPER_P3 = """# YOLOv12s + Deeper Backbone P3 (no P5 reduction)
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
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # P5 unchanged (4 reps)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# ARCH 3: Both — P5 A2C2f + Deeper P3
# =============================================================================
ARCH_P5_ATTN_DEEPER_P3 = """# YOLOv12s + P5 A2C2f Attention + Deeper P3
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
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # P5 unchanged

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # 20 — A2C2f with ATTENTION (was C3k2)

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_SKA = """# YOLOv12s + SKA (LSKA at P3 head)
# LSKA = Large Separable Kernel Attention — 7x7 decomposed kernel
# Captures wider spatial context at P3 for small/medium objects
# Proven on luggage dataset — never tested on weapons
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA, [256, False]]          # 14 — LSKA at P3 (large kernel context)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_SKA_P5 = """# YOLOv12s + SKA at P3 + P5 A2C2f Attention
# Combines both: LSKA at P3 for medium objects + A2C2f attention at P5 for large
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA, [256, False]]          # 14 — LSKA at P3

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # 20 — A2C2f with ATTENTION (was C3k2)

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_RUNS = [
    {
        "name": "arch_p5_attn_70",
        "desc": "P5 head: C3k2 → A2C2f attention (consistency fix)",
        "yaml_content": ARCH_P5_ATTN,
        "batch": 58,
    },
    {
        "name": "arch_ska_70",
        "desc": "SKA: LSKA at P3 head (large kernel context)",
        "yaml_content": ARCH_SKA,
        "batch": 56,
    },
    {
        "name": "arch_ska_p5_70",
        "desc": "SKA at P3 + P5 A2C2f attention (both improvements)",
        "yaml_content": ARCH_SKA_P5,
        "batch": 54,
    },
    {
        "name": "arch_deeper_p3_70",
        "desc": "Backbone P3: 2→3 reps (no P5 reduction)",
        "yaml_content": ARCH_DEEPER_P3,
        "batch": 56,
    },
    {
        "name": "arch_p5_attn_deeper_p3_70",
        "desc": "Both: P5 A2C2f + deeper P3",
        "yaml_content": ARCH_P5_ATTN_DEEPER_P3,
        "batch": 54,
    },
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: ARCHITECTURE (default params — no custom loss)")
    print(f"# Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")

        model.train(
            data=DATA_YAML,
            epochs=80,
            imgsz=IMG_SIZE,
            batch=run["batch"],
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            # --- DEFAULT TAL params ---
            tal_topk=10,
            tal_alpha=0.5,
            tal_beta=6.0,
            # --- DISABLE SWA ---
            alpha_start=0.0,
            alpha_end=0.0,
            alpha_min=0.0,
            alpha_max=0.0,
            # --- DISABLE clipping ---
            iou_clip_start=999.0,
            iou_clip_end=999.0,
            dfl_clip_start=999.0,
            dfl_clip_end=999.0,
            # --- DISABLE custom features ---
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
    print(f"  70% ABLATION — 3 ARCHITECTURES (OVERALL IMPROVEMENT)")
    print(f"  Target to beat: 78.45% mAP50 (baseline)")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}")
    print(f"  68% large objects → P5 matters most")
    print(f"{'=' * 70}")

    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i+1}] {run['name']:<35} batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Run {i+1}/{len(ARCH_RUNS)}: {run['name']}")
        result = run_arch_experiment(run)
        results.append(result)

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
