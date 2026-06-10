#!/usr/bin/env python3
"""
70% Ablation — P2 Auxiliary architecture (Arch-6, marked BEST from luggage).

5-head design with dedicated auxiliary detection branch at P2.
Aux branch uses C3k2+Conv and feeds its own head alongside main P2 output.
Detect heads: aux_P2, main_P2, P3, P4, P5

Default training params to isolate architecture effect.
Target to beat: baseline 78.45% mAP50

Usage:
  python run_arch_p2_aux_ablation.py
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
# Arch-6: P2 Auxiliary (BEST from luggage experiments)
# =============================================================================
ARCH_P2_AUX = """# Arch-6: P2 Auxiliary — 5 det heads, dedicated aux at P2
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def main():
    os.makedirs(YAML_DIR, exist_ok=True)

    yaml_path = os.path.join(YAML_DIR, "arch_p2_aux_70.yaml")
    save_yaml(ARCH_P2_AUX, yaml_path)

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — P2 AUXILIARY (ARCH-6)")
    print(f"  Target to beat: 78.45% mAP50 (baseline)")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}    Batch: 38")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")

        model.train(
            data=DATA_YAML,
            epochs=80,
            imgsz=IMG_SIZE,
            batch=38,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name="arch_p2_aux_70",
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
        print(f"\n  DONE: arch_p2_aux_70 ({elapsed:.2f}h)")

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: arch_p2_aux_70 ({elapsed:.2f}h) -- {e}")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
