#!/usr/bin/env python3
"""
Full Dataset — 3 configs promoted from ablation analysis.

ABLATION FINDINGS (20 runs):
  - topk=15 consistently best
  - alpha=0.5 best mAP50, alpha=0.6-0.7 best mAP50-95
  - beta=3.0-3.5 best at ablation, but full rewards higher beta (4.0)
  - TAL tuning ceiling ~80.45% at ablation

PROMOTING:
  1. topk=15, a=0.7, b=3.5 — ablation best balanced (80.28%, 50.05% mAP50-95)
  2. topk=15, a=0.6, b=3.5 — ablation best mAP50-95 ever (50.15%)
  3. topk=15, a=0.5, b=4.0 — topk=15 with full-proven beta=4.0 (gap fill)

Current full best mAP50:     v5_cls12_tight_clip_full = 83.03% (topk=13, a=0.5, b=4.0)
Current full best all-round: v5_topk15_beta35_full    = 82.93% (topk=15, a=0.5, b=3.5)

Usage:
  python run_full_next3.py
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

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASE = v5_cls12_tight_clip (proven full baseline)
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
    "tal_alpha": 0.5,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}


def make_exp(name, desc, **overrides):
    params = copy.deepcopy(BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Run 1: topk=15 + alpha=0.7 + beta=3.5
    # Ablation: 80.28% mAP50, 50.05% mAP50-95 — best balanced
    make_exp(
        "v5_topk15_tal07_beta35_full",
        "ablation best balanced — topk=15, alpha=0.7, beta=3.5",
        tal_topk=15,
        tal_alpha=0.7,
        tal_beta=3.5,
    ),

    # Run 2: topk=15 + alpha=0.6 + beta=3.5
    # Ablation: 79.90% mAP50, 50.15% mAP50-95 — best mAP50-95 ever
    make_exp(
        "v5_topk15_tal06_beta35_full",
        "ablation best mAP50-95 — topk=15, alpha=0.6, beta=3.5",
        tal_topk=15,
        tal_alpha=0.6,
        tal_beta=3.5,
    ),

    # Run 3: topk=15 + alpha=0.5 + beta=4.0
    # Ablation: 79.57% mAP50 — modest, but full dataset rewarded
    # topk=13+beta=4.0 as #1 (83.03%). topk=15 upgrade never tested on full.
    make_exp(
        "v5_topk15_beta4_full",
        "gap fill — topk=15 with full-proven beta=4.0",
        tal_topk=15,
        tal_beta=4.0,
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
    print(f"# Changes from v5_cls12_tight_clip (current full best 83.03%):")
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
    print(f"  FULL DATASET — 3 PROMOTED FROM ABLATION")
    print(f"  Target to beat: 83.03% mAP50 (v5_cls12_tight_clip_full)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
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
