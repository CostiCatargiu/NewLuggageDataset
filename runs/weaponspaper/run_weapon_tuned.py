#!/usr/bin/env python3
"""
Weapon-Tuned Loss — Single Training Run.

Uses loss_weapon_tuned.py with all 7 changes tuned for the weapon dataset:
  [1] VarifocalLoss instead of BCE   (1.3 obj/img = massive neg imbalance)
  [2] small_obj_px=40, boost=2.5     (real small objects are <32px, only 1.4%)
  [3] TAL topk=13, beta=4.0          (reduce IoU noise for small weapons)
  [4] Alpha 0.7 -> 0.3               (84% large — don't over-weight area)
  [5] DIoU instead of CIoU            (no aspect ratio penalty for knives)
  [6] Loosened clipping               (let hard examples contribute gradients)
  [7] cls=1.0                         (double cls gain for 4-class discrimination)

Usage:
  python run_weapon_tuned.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage2/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"
EXPERIMENT_NAME = "weapon_tuned_all7"

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
    print(f"  WEAPON-TUNED LOSS TRAINING")
    print(f"{'=' * 70}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Model:       {MODEL_WEIGHTS}")
    print(f"  Dataset:     {DATA_YAML}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch:       {BATCH}")
    print(f"  Image size:  {IMG_SIZE}")
    print(f"{'=' * 70}")
    print(f"  Loss changes:")
    print(f"    [1] VFL instead of BCE (use_vfl=True)")
    print(f"    [2] small_obj_px=40, boost=2.5")
    print(f"    [3] TAL topk=13, beta=4.0")
    print(f"    [4] alpha: 0.7 -> 0.3")
    print(f"    [5] DIoU instead of CIoU")
    print(f"    [6] Loosened clipping (50->20, 25->10)")
    print(f"    [7] cls=1.0 (doubled)")
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

        # ── [CHANGE 7] Classification gain ──
        cls=1.0,                         # was 0.5

        # ── [CHANGE 4] Section A: Alpha schedule ──
        alpha_start=0.7,                 # was 0.9
        alpha_end=0.3,                   # was 0.5
        alpha_min=0.2,                   # was 0.3
        alpha_max=0.8,                   # was 0.9

        # ── [CHANGE 2] Section A: Small object targeting ──
        small_obj_px=40,                 # was 70
        small_obj_boost=2.5,             # was 1.5

        # ── Section B: Center loss (disabled) ──
        center_loss_weight_init=0.0,
        center_loss_weight_min=0.0,
        center_loss_decay_epochs=35,

        # ── [CHANGE 6] Section C: Loosened clipping ──
        iou_clip_start=50.0,             # was 20.0
        iou_clip_end=20.0,               # was 10.0
        dfl_clip_start=25.0,             # was 10.0
        dfl_clip_end=10.0,               # was 5.0

        # ── [CHANGE 3] Section D: TAL ──
        tal_topk=13,                     # was 10
        tal_alpha=0.5,
        tal_beta=4.0,                    # was 6.0

        # ── [CHANGE 5] IoU type ──
        iou_type="DIoU",                 # was CIoU

        # ── [CHANGE 1] VarifocalLoss ──
        use_vfl=True,
        vfl_alpha=0.75,
        vfl_gamma=2.0,
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
