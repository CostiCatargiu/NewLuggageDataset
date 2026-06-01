#!/usr/bin/env python3
"""
Shape-Aware TAL — test on full dataset with wt_boost15 config.

Adds aspect ratio similarity to the TAL alignment metric:
  Standard:  align = score^alpha * iou^beta
  Shape-TAL: align = score^alpha * iou^beta * shape_sim^gamma

3 runs to find optimal gamma:
  1. gamma=0.5 (mild shape preference)
  2. gamma=1.0 (moderate shape preference)
  3. gamma=2.0 (strong shape preference)

All use wt_boost15 base config (82.79% mAP50 on full).

Usage:
  python run_shape_tal.py
"""

import time
import gc
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage2/data.yaml"  # ablation first
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

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
        "name": "shape_tal_gamma05",
        "description": "Shape-TAL gamma=0.5 (mild)",
        "gamma": 0.5,
        "shape_min": 0.3,
    },
    {
        "name": "shape_tal_gamma10",
        "description": "Shape-TAL gamma=1.0 (moderate)",
        "gamma": 1.0,
        "shape_min": 0.3,
    },
    {
        "name": "shape_tal_gamma20",
        "description": "Shape-TAL gamma=2.0 (strong)",
        "gamma": 2.0,
        "shape_min": 0.3,
    },
]

# wt_boost15 base params
BASE_PARAMS = {
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
}


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
    print(f"# gamma={exp['gamma']}, shape_min={exp['shape_min']}")
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
        # Shape-TAL params
        "use_shape_tal": True,
        "shape_gamma": exp["gamma"],
        "shape_min": exp["shape_min"],
    }
    train_kwargs.update(BASE_PARAMS)

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
    print(f"  SHAPE-AWARE TAL — 3 gamma values")
    print(f"  Base: wt_boost15 config")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")
    print(f"  align = score^alpha * iou^beta * shape_sim^gamma")
    print(f"  [1] gamma=0.5 (mild)")
    print(f"  [2] gamma=1.0 (moderate)")
    print(f"  [3] gamma=2.0 (strong)")
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
        print(f"  [{tag}] {r['name']:<25} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
