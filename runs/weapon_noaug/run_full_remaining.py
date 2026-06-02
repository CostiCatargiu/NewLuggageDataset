#!/usr/bin/env python3
"""
Full Dataset (No Aug) — 2 remaining configs never tested on full.

Already tested on full:
  original_loss:    81.99%
  wt_tal_alpha08:   82.84%

Now testing:
  1. wt_boost15           — best recall + small objects on ablation (73.26%)
  2. v5_cls12_tight_clip  — highest on old augmented ablation (73.60%)

Usage:
  python run_full_remaining.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_noaug_weapon"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "wt_boost15_full",
        "description": "Best recall + small objects on ablation",
        "params": {
            "cls": 1.0,
            "alpha_start": 0.7,
            "alpha_end": 0.3,
            "alpha_min": 0.2,
            "alpha_max": 0.8,
            "small_obj_px": 40,
            "small_obj_boost": 1.5,
            "center_loss_weight_init": 0.0,
            "center_loss_weight_min": 0.0,
            "center_loss_decay_epochs": 35,
            "iou_clip_start": 50.0,
            "iou_clip_end": 20.0,
            "dfl_clip_start": 25.0,
            "dfl_clip_end": 10.0,
            "tal_topk": 13,
            "tal_alpha": 0.5,
            "tal_beta": 4.0,
            "iou_type": "DIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "v5_cls12_tight_clip_full",
        "description": "Highest on old augmented ablation",
        "params": {
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
            "tal_alpha": 0.5,
            "tal_beta": 4.0,
            "iou_type": "DIoU",
            "use_vfl": False,
        },
    },
]


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
# TRAINING
# =============================================================================
def run_experiment(exp):
    name = exp["name"]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"{'#' * 70}\n")

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
        "name": name,
        "patience": 100,
        "close_mosaic": 10,
        "seed": 0,
        "deterministic": True,
    }
    train_kwargs.update(exp["params"])

    try:
        model.train(**train_kwargs)
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {name} ({elapsed:.2f}h)")
        return {"name": name, "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {name} ({elapsed:.2f}h) -- {e}")
        return {"name": name, "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  2 REMAINING CONFIGS ON FULL (NO AUG)")
    print(f"  Current best: wt_tal_alpha08 = 82.84%")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")
    print(f"  [1] wt_boost15_full          tal_alpha=0.5, boost=1.5, loose clip")
    print(f"  [2] v5_cls12_tight_clip_full cls=1.2, boost=2.5, tight clip")
    print(f"{'=' * 70}\n")

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n>>> Run {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        result = run_experiment(exp)
        results.append(result)

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<30} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
