#!/usr/bin/env python3
"""
70% Ablation — 3 configs to see scaling behavior.

Fills the gap between 42% ablation and 100% full:
  42% (8K)  → original: 71.34%, best: 74.20% (+4.01%)
  70% (13K) → ???
  100% (19K) → original: 81.99%, best: 83.03% (+1.27%)

Configs:
  1. original_loss_70      — baseline
  2. v5_tal_alpha08_70     — ablation winner (74.20% at 42%)
  3. v5_cls12_tight_clip_70 — full winner (83.03% at 100%)

Usage:
  python run_ablation70_test.py
"""

import time
import gc
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_noaug_weapon"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "original_loss_70",
        "description": "Baseline — default params",
        "params": {
            "cls": 0.5,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
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
        "name": "v5_tal_alpha08_70",
        "description": "Ablation winner (74.20% at 42%)",
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
            "tal_alpha": 0.8,
            "tal_beta": 4.0,
            "iou_type": "DIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "v5_cls12_tight_clip_70",
        "description": "Full winner (83.03% at 100%)",
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
    print(f"  70% ABLATION — SCALING BEHAVIOR TEST")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")
    print(f"  [1] original_loss_70          baseline")
    print(f"  [2] v5_tal_alpha08_70         ablation winner (tal_a=0.8)")
    print(f"  [3] v5_cls12_tight_clip_70    full winner (tal_a=0.5)")
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
