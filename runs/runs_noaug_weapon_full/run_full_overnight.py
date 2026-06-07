#!/usr/bin/env python3
"""
Full Dataset — 3 overnight configs exploring beyond pure TAL tuning.

CURRENT BEST: v5_tal07_full = 83.12% mAP50, 53.38% mAP50-95 (topk=13, a=0.7, b=4.0)

WHAT WE KNOW:
  - topk=13, beta=4.0 is the winning combo on full
  - alpha=0.7 is the sweet spot (beats 0.5 and 0.8)
  - TAL tuning alone is near ceiling (~83.1%)
  - boost=1.5 helped at ablation but never combined with alpha=0.7
  - Higher beta (5.0) never tested with topk=13 + alpha=0.7
  - topk=13, alpha=0.7 with DIoU confirmed — CIoU never tested

STRATEGY — mix TAL sweet spot with training-time changes:
  1. Best TAL + boost=1.5 — proven at ablation, never with alpha=0.7
  2. Best TAL + beta=5.0 — push localization harder, may recover AR metrics
  3. Best TAL + CIoU — different IoU loss, could help localization

Usage:
  python run_full_overnight.py
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
# BASE = v5_tal07 (new best)
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
    "iou_clip_start": 20.0,
    "iou_clip_end": 10.0,
    "dfl_clip_start": 10.0,
    "dfl_clip_end": 5.0,
    "tal_topk": 13,
    "tal_alpha": 0.7,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}


def make_exp(name, desc, **overrides):
    params = copy.deepcopy(BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Run 1: Best TAL + boost=1.5
    # boost=1.5 consistently beat 2.5 at ablation. Never combined with
    # alpha=0.7. Could help small objects without hurting large.
    make_exp(
        "v5_tal07_boost15_full",
        "best TAL + boost=1.5 — proven at ablation, new combo",
        small_obj_boost=1.5,
    ),

    # Run 2: Best TAL + beta=5.0
    # beta=4.0 is proven, but baseline's beta=6.0 had best AR/localization.
    # beta=5.0 is the midpoint — might recover AR metrics while keeping
    # the alpha=0.7 + topk=13 detection gains.
    make_exp(
        "v5_tal07_beta5_full",
        "best TAL + beta=5.0 — recover localization quality",
        tal_beta=5.0,
    ),

    # Run 3: Best TAL + CIoU instead of DIoU
    # All runs used DIoU. CIoU adds aspect ratio penalty which could
    # improve bounding box quality (mAP50-95, AR metrics).
    make_exp(
        "v5_tal07_ciou_full",
        "best TAL + CIoU — different IoU loss for better boxes",
        iou_type="CIoU",
    ),
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

    diffs = [f"{k}: {BASE[k]} -> {v}" for k, v in params.items() if BASE.get(k) != v]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Changes from v5_tal07_full (current best 83.12%):")
    for d in diffs:
        print(f"#   {d}")
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
    print(f"  FULL DATASET — 3 OVERNIGHT CONFIGS")
    print(f"  Target to beat: 83.12% mAP50 (v5_tal07_full)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"  Base:     topk=13, alpha=0.7, beta=4.0 (new best)")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        diffs = [f"{k}={exp['params'][k]}" for k in exp['params'] if BASE.get(k) != exp['params'][k]]
        print(f"  [{i+1}] {exp['name']:<40} {', '.join(diffs) if diffs else 'base config'}")

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
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
