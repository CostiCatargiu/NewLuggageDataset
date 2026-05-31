#!/usr/bin/env python3
"""
ABLATION3 Test — 4 runs on the new cleaner subset.

Does the better ablation change the rankings?
  1. original_loss_ds3       — original loss, no modifications (baseline)
  2. wt_boost15_ds3          — proven winner from ABLATION1/2
  3. v5_cls12_tight_clip_ds3 — close second on ABLATION1/2
  4. wt_tal_alpha08_ds3      — tied on ABLATION1/2, untested on cleaner data

If rankings hold: ablation quality doesn't matter, go with full_wt_boost15.
If rankings change: ABLATION3 reveals a better config to try on full.

Usage:
  python run_ablation3_test.py
"""

import time
import gc
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage3/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# EXPERIMENT CONFIGS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "original_loss_ds3",
        "description": "Original loss, no modifications — baseline for ABLATION3",
        "params": {
            "cls": 0.5,
            "alpha_start": 0.5,
            "alpha_end": 0.5,
            "alpha_min": 0.3,
            "alpha_max": 0.9,
            "small_obj_px": 0,
            "small_obj_boost": 1.0,
            "center_loss_weight_init": 0.0,
            "center_loss_weight_min": 0.0,
            "center_loss_decay_epochs": 35,
            "iou_clip_start": 999.0,
            "iou_clip_end": 999.0,
            "dfl_clip_start": 999.0,
            "dfl_clip_end": 999.0,
            "tal_topk": 10,
            "tal_alpha": 0.5,
            "tal_beta": 6.0,
            "iou_type": "CIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "wt_boost15_ds3",
        "description": "Proven winner from ABLATION1/2 (73.42% mAP50)",
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
        "name": "v5_cls12_tight_clip_ds3",
        "description": "Close second on ABLATION1/2 (73.60% mAP50)",
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
    {
        "name": "wt_tal_alpha08_ds3",
        "description": "Tied on ABLATION1/2 (73.33% mAP50), lowest val/cls loss",
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
            "tal_alpha": 0.8,
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
    params = exp["params"]

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
    train_kwargs.update(params)

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
    print(f"  ABLATION3 TEST — 4 configs on cleaner subset")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML} (ABLATION3 / V2)")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        print(f"  [{i+1}] {exp['name']:<30} {exp['description']}")

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
