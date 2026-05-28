#!/usr/bin/env python3
"""
Weapon-Tuned on Full Dataset Training
======================================

Train the best config (weapon_tuned_all72) on the FULL dataset
instead of DS2 only. More training data should:
  - Reduce overfitting from aggressive settings (boost=2.5, cls=1.0)
  - Potentially push mAP50 beyond 73.13%
  - Better generalization across test sets

Config (same as weapon_tuned_all72):
  [1] Alpha 0.7→0.3
  [2] small_obj_px=40, boost=2.5
  [3] TAL topk=13, beta=4.0
  [4] DIoU instead of CIoU
  [5] Clipping: iou 50→20, dfl 25→10
  [6] cls=1.0

Usage:
  python run_weapon_tuned_full.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram/data.yaml"  # FULL DATASET
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"
EXPERIMENT_NAME = "weapon_tuned_full"

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
            if epoch % 10 == 0:
                print(f"[Epoch Sync] Epoch {epoch}")
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
    print(f"  WEAPON-TUNED on FULL DATASET")
    print(f"{'=' * 70}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Model:       {MODEL_WEIGHTS}")
    print(f"  Dataset:     {DATA_YAML} (FULL)")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch:       {BATCH}")
    print(f"{'=' * 70}")
    print(f"  Config (same as weapon_tuned_all72):")
    print(f"    [A] Alpha:    0.7 → 0.3")
    print(f"    [A] Boost:    small_obj_px=40, boost=2.5")
    print(f"    [C] Clipping: iou 50→20, dfl 25→10")
    print(f"    [D] TAL:      topk=13, alpha=0.5, beta=4.0")
    print(f"    [+] DIoU, cls=1.0")
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

        # ── Classification gain ──
        cls=1.0,

        # ── Section A: Alpha schedule ──
        alpha_start=0.7,
        alpha_end=0.3,
        alpha_min=0.2,
        alpha_max=0.8,

        # ── Section A: Small object targeting ──
        small_obj_px=40,
        small_obj_boost=2.5,

        # ── Section B: Center loss (disabled) ──
        center_loss_weight_init=0.0,
        center_loss_weight_min=0.0,
        center_loss_decay_epochs=35,

        # ── Section C: Loosened clipping ──
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

        # ── VFL disabled ──
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
