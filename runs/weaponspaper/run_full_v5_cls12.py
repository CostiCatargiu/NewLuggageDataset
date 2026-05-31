#!/usr/bin/env python3
"""
Full Dataset — 2 final runs.

Run 1: full_wt_tal_alpha08
  Same as full_wt_boost15 (82.79%) with tal_alpha 0.5->0.8
  Assigner trusts classification more.

Run 2: full_v5_cls12_tight_clip
  Highest ablation mAP50 (73.60%), never tested on full.
  cls=1.2, boost=2.5, tight clipping.

Usage:
  python run_full_v5_cls12.py
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
        "name": "full_wt_tal_alpha08",
        "description": "wt_boost15 + tal_alpha 0.5->0.8",
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
            "tal_alpha": 0.8,              # <-- 0.5 -> 0.8
            "tal_beta": 4.0,
            "iou_type": "DIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "full_v5_cls12_tight_clip",
        "description": "Ablation #1 config: cls=1.2, boost=2.5, tight clip",
        "params": {
            "cls": 1.2,                      # was 1.0
            "alpha_start": 0.7,
            "alpha_end": 0.3,
            "alpha_min": 0.2,
            "alpha_max": 0.8,
            "small_obj_px": 40,
            "small_obj_boost": 2.5,          # was 1.5
            "center_loss_weight_init": 0.0,
            "center_loss_weight_min": 0.0,
            "center_loss_decay_epochs": 35,
            "iou_clip_start": 20.0,          # was 50.0
            "iou_clip_end": 10.0,            # was 20.0
            "dfl_clip_start": 10.0,          # was 25.0
            "dfl_clip_end": 5.0,             # was 10.0
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
    print(f"  2 FINAL FULL DATASET RUNS")
    print(f"  Target to beat: 82.79% (full_wt_boost15)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")
    print(f"  [1] full_wt_tal_alpha08        tal_alpha 0.5->0.8")
    print(f"  [2] full_v5_cls12_tight_clip   cls=1.2, boost=2.5, tight clip")
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
