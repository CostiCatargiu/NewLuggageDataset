#!/usr/bin/env python3
"""
Full Dataset Training — wt_boost15 config (best ablation: 73.42% mAP50)

Config (proven winner across 70+ ablation experiments):
  alpha: 0.7->0.3, alpha_min=0.2, alpha_max=0.8
  small_obj_px=40, boost=1.5
  TAL: topk=13, alpha=0.5, beta=4.0
  DIoU, cls=1.0
  clipping: iou 50->20, dfl 25->10
  center loss: OFF
  VFL: OFF (BCE)

Usage:
  python run_full_dataset.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"
EXPERIMENT_NAME = "full_wt_boost15"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# EPOCH SYNC CALLBACK
# =============================================================================
def on_train_epoch_start(trainer):
    """Sync epoch to loss function for dynamic alpha scheduling."""
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
# MAIN
# =============================================================================
def main():
    print(f"\n{'=' * 70}")
    print(f"  FULL DATASET TRAINING — wt_boost15 config")
    print(f"{'=' * 70}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Model:       {MODEL_WEIGHTS}")
    print(f"  Dataset:     {DATA_YAML}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch:       {BATCH}")
    print(f"  Image size:  {IMG_SIZE}")
    print(f"{'=' * 70}")
    print(f"  Config (proven best from 70+ ablation experiments):")
    print(f"    [A] alpha 0.7->0.3, boost=1.5, px=40")
    print(f"    [B] center loss: OFF")
    print(f"    [C] clip: iou 50->20, dfl 25->10")
    print(f"    [D] TAL: topk=13, alpha=0.5, beta=4.0")
    print(f"    IoU: DIoU, cls=1.0")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        patience=100,
        close_mosaic=10,
        seed=0,
        deterministic=True,

        # ── Loss gains ──
        cls=1.0,

        # ── Section A: Size-aware weighting ──
        alpha_start=0.7,
        alpha_end=0.3,
        alpha_min=0.2,
        alpha_max=0.8,
        small_obj_px=40,
        small_obj_boost=1.5,

        # ── Section B: Center loss (OFF) ──
        center_loss_weight_init=0.0,
        center_loss_weight_min=0.0,
        center_loss_decay_epochs=35,

        # ── Section C: Clipping ──
        iou_clip_start=50.0,
        iou_clip_end=20.0,
        dfl_clip_start=25.0,
        dfl_clip_end=10.0,

        # ── Section D: TAL ──
        tal_topk=13,
        tal_alpha=0.5,
        tal_beta=4.0,

        # ── IoU type ──
        iou_type="DIoU",

        # ── Classification loss ──
        use_vfl=False,
    )

    training_time = (time.time() - start_time) / 3600

    print(f"\n{'=' * 70}")
    print(f"  DONE: {EXPERIMENT_NAME} ({training_time:.2f}h)")
    print(f"{'=' * 70}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
