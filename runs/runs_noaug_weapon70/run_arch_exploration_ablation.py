#!/usr/bin/env python3
"""
70% Ablation — 6 Novel Architecture Explorations

Going beyond incremental tweaks. These are bold architectural ideas
targeting small object detection weakness (63.25% vs 83.12% overall).

CONFIGS:
  1. Deeper P3 Backbone (4 reps) — more small object feature capacity
  2. SE at P3 Head — channel attention for small object filtering
  3. SPP at P3 Head — multi-scale context without large kernels
  4. Dense Skip P3↔P4 — bidirectional feature flow between scales
  5. Dual-Path P3 Head — lightweight + heavy branches merged
  6. ASPP at P3 Head — atrous spatial pyramid (multi-rate dilated conv)

All use DEFAULT training params to isolate architecture effect.
Target to beat: baseline 78.45% mAP50

Usage:
  python run_arch_exploration_ablation.py
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
# Config 1: Deeper P3 Backbone (2→4 reps)
# Small objects need more feature extraction depth before reaching head.
# =============================================================================
ARCH_DEEPER_P3_BACKBONE = """# Deeper P3 Backbone — 4 reps (2x depth)
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 4, C3k2,  [512, False, 0.25]]      # 4 — P3: 2→4 reps (DOUBLED)
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
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Config 2: SE (Squeeze-Excitation) at P3 Head
# Channel attention to filter out noise and emphasize weapon-relevant features.
# SE = Global Pool → FC → ReLU → FC → Sigmoid → Channel reweighting
# Note: Requires SE module implementation in Ultralytics
# =============================================================================
ARCH_SE_P3 = """# SE at P3 Head — channel attention for small objects
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
  - [-1, 2, C2fSE, [256, False]]            # 14 — SE at P3 head

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Config 3: SPP (Spatial Pyramid Pooling) at P3 Head
# Multi-scale context aggregation: 1x1, 2x2, 3x3 grid pooling → concat
# Captures context at multiple scales without large kernel "washing out"
# Note: Use SPPF (SPP-Fast) from Ultralytics if available, else standard SPP
# =============================================================================
ARCH_SPP_P3 = """# SPP at P3 Head — multi-scale context for small objects
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
  - [-1, 1, SPPF, [256, 5]]                 # 14 — SPP at P3 head (k=5)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Config 4: Dense Skip P3↔P4
# Add direct bidirectional skip connections between P3 and P4.
# P3 bottom-up path gets enriched by P4 top-down early.
# Creates mesh-like feature flow instead of linear FPN chain.
# =============================================================================
ARCH_DENSE_P3P4 = """# Dense Skip P3↔P4 — bidirectional feature flow
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4 — P3 backbone
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — P4 backbone
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  # --- Top-down ---
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 top-down

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head

  # --- Bottom-up with DENSE skip from P3 ---
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11, 4], 1, Concat, [1]]           # 16 — concat P4_td + P4_bu + P3_backbone (DENSE)
  - [-1, 1, Conv, [512, 1, 1]]              # 17 — project back to 512ch
  - [-1, 2, A2C2f, [512, False, -1]]        # 18 — P4 bottom-up

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 21 — P5

  - [[14, 18, 21], 1, Detect, [nc]]
"""

# =============================================================================
# Config 5: Dual-Path P3 Head
# Split P3 into two parallel processing branches:
#   - Path A: lightweight (1 rep, fast, preserves detail)
#   - Path B: heavy (3 reps, rich features)
# Merge before detection. Ensemble-like effect at head level.
# =============================================================================
ARCH_DUAL_P3 = """# Dual-Path P3 Head — lightweight + heavy branches
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
  - [[-1, 4], 1, Concat, [1]]               # 13 — concat for P3

  # Dual path split
  - [-1, 1, A2C2f, [128, False, 1]]         # 14 — Path A: lightweight (1 rep, 128ch)
  - [13, 3, A2C2f, [256, False, 3]]         # 15 — Path B: heavy (3 reps, 256ch)
  
  # Merge paths
  - [[14, 15], 1, Concat, [1]]              # 16 — concat both paths (128+256=384ch)
  - [-1, 1, Conv, [256, 1, 1]]              # 17 — project to standard 256ch

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 20

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 23

  - [[17, 20, 23], 1, Detect, [nc]]
"""

# =============================================================================
# Config 6: ASPP (Atrous Spatial Pyramid Pooling) at P3 Head
# Multi-rate dilated convolutions capture multi-scale context:
#   - 1x1 conv (point-wise)
#   - 3x3 dilated rate=6 (captures medium range)
#   - 3x3 dilated rate=12 (captures large range)
#   - Global avg pool (image-level context)
# All concatenated and projected back.
# Similar to DeepLabV3. More sophisticated than SPP.
# Note: Requires ASPP module implementation
# =============================================================================
ARCH_ASPP_P3 = """# ASPP at P3 Head — atrous spatial pyramid for multi-scale
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
  - [-1, 2, C2fASPP, [256, False]]          # 14 — ASPP at P3 head

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_RUNS = [
    # Reordered by confidence (highest → lowest probability of improvement)
    {
        "name": "arch_spp_p3_70",
        "desc": "[1/6] SPP at P3 head — multi-scale pooling (HIGHEST CONFIDENCE)",
        "yaml_content": ARCH_SPP_P3,
        "batch": 56,
        "rationale": "Proven in YOLOv5, safe, lightweight, no risk"
    },
    {
        "name": "arch_deeper_p3_backbone_70",
        "desc": "[2/6] Deeper P3 backbone (4 reps) — more feature capacity",
        "yaml_content": ARCH_DEEPER_P3_BACKBONE,
        "batch": 54,
        "rationale": "Straightforward capacity increase, depth always helps"
    },
    {
        "name": "arch_se_p3_70",
        "desc": "[3/6] SE at P3 head — channel attention filtering",
        "yaml_content": ARCH_SE_P3,
        "batch": 56,
        "rationale": "Proven in ResNet/EfficientNet, ~1% params, safe bet"
    },
    {
        "name": "arch_aspp_p3_70",
        "desc": "[4/6] ASPP at P3 head — atrous pyramid",
        "yaml_content": ARCH_ASPP_P3,
        "batch": 54,
        "rationale": "Sophisticated (DeepLabV3), higher potential but more complex"
    },
    {
        "name": "arch_dense_p3p4_70",
        "desc": "[5/6] Dense skip P3↔P4 — bidirectional flow",
        "yaml_content": ARCH_DENSE_P3P4,
        "batch": 52,
        "rationale": "Rethinks FPN structure, medium risk, may help feature flow"
    },
    {
        "name": "arch_dual_p3_70",
        "desc": "[6/6] Dual-path P3 — lightweight + heavy merged (HIGHEST RISK)",
        "yaml_content": ARCH_DUAL_P3,
        "batch": 50,
        "rationale": "Most complex, ensemble-like, highest risk of overfitting"
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
    print(f"# Type: ARCHITECTURE (default params)")
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
    print(f"  70% ABLATION — 6 NOVEL ARCHITECTURES")
    print(f"  Target to beat: 78.45% mAP50 (baseline)")
    print(f"  Focus: small object detection (current 63.25%)")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}")
    print(f"{'=' * 70}")

    for i, run in enumerate(ARCH_RUNS):
        print(f"  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
