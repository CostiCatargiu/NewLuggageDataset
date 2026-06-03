#!/usr/bin/env python3
"""
Ablation — 800px resolution test.

v5_tal_alpha08 (ablation winner: 74.20% at 640px) at 800px.
If this beats 74.20%, run v5_cls12_tight_clip (full winner) at 800px on full.

Usage:
  python run_ablation_800px.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_noaug_weapon"
EXPERIMENT_NAME = "v5_tal_alpha08_800px"

EPOCHS = 80
IMG_SIZE = 800
BATCH = 36
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
    print(f"  800px RESOLUTION TEST — ABLATION")
    print(f"{'=' * 70}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Model:       {MODEL_WEIGHTS}")
    print(f"  Dataset:     {DATA_YAML}")
    print(f"  Image size:  {IMG_SIZE}  (was 640)")
    print(f"  Batch:       {BATCH}  (was 58)")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Target:      beat 74.20% (v5_tal_alpha08 at 640px)")
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

        # v5_tal_alpha08 config (ablation winner)
        cls=1.2,
        alpha_start=0.7,
        alpha_end=0.3,
        alpha_min=0.2,
        alpha_max=0.8,
        small_obj_px=40,
        small_obj_boost=2.5,
        center_loss_weight_init=0.0,
        center_loss_weight_min=0.0,
        center_loss_decay_epochs=35,
        iou_clip_start=20.0,
        iou_clip_end=10.0,
        dfl_clip_start=10.0,
        dfl_clip_end=5.0,
        tal_topk=13,
        tal_alpha=0.8,
        tal_beta=4.0,
        iou_type="DIoU",
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
