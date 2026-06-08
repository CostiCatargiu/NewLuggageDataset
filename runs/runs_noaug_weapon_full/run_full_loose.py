#!/usr/bin/env python3
"""
Full Dataset — Best TAL + loose clips.

EVIDENCE:
  - At ablation, loose clips (79.72%) BEAT tight clips (79.33%)
  - Loose clips never tested on full dataset
  - Current best v5_tal07_full (83.12%) uses tight clips
  - If loose clips give the same advantage on full → potential new best

Config: topk=13, alpha=0.7, beta=4.0, loose clips (50/20, 25/10)

Usage:
  python run_full_loose.py
"""

import time
import gc
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_noaug_weapon"

EPOCHS = 90
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASE = v5_tal07 with LOOSE clips
# =============================================================================
BASE = {
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
    "iou_clip_start": 50.0,
    "iou_clip_end": 20.0,
    "dfl_clip_start": 25.0,
    "dfl_clip_end": 10.0,
    "tal_topk": 13,
    "tal_alpha": 0.7,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}


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


def main():
    print(f"\n{'=' * 70}")
    print(f"  FULL DATASET — LOOSE CLIPS + BEST TAL")
    print(f"  Target to beat: 83.12% mAP50 (v5_tal07_full)")
    print(f"  Config: topk=13, alpha=0.7, beta=4.0, loose clips (50/20, 25/10)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    train_kwargs = {
        "data": DATA_YAML,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": PROJECT_DIR,
        "name": "v5_tal07_loose_full",
        "patience": 100,
        "close_mosaic": 10,
        "seed": 0,
        "deterministic": True,
    }
    train_kwargs.update(BASE)

    model.train(**train_kwargs)

    elapsed = (time.time() - start_time) / 3600
    print(f"\n  DONE: v5_tal07_loose_full ({elapsed:.2f}h)")


if __name__ == "__main__":
    main()
