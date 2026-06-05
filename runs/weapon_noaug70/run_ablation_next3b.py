#!/usr/bin/env python3
"""
70% Ablation — 3 configs focused on alpha=0.8 exploration at topk=15.

KEY FINDINGS:
  - alpha=0.8 + topk=13 scored 79.90% at 70% (#5 overall)
  - alpha=0.7 + topk=15 + beta=3.5 scored 80.28% (#2, best mAP50-95)
  - alpha=0.8 was NEVER tested with topk=15
  - topk=15 + beta=3.0 is current best mAP50 (80.45%)
  - topk=15 + beta=3.5 is current best mAP50-95 (50.05%)

STRATEGY:
  1. alpha=0.8 + topk=15 + beta=3.5 — highest confidence, two proven signals
  2. alpha=0.8 + topk=15 + beta=3.0 — alpha=0.8 with mAP50-winning beta
  3. alpha=0.6 + topk=15 + beta=3.5 — unexplored midpoint alpha

Current 70% best mAP50:     v5_topk15_beta3_70     = 80.45% (topk=15, a=0.5, b=3.0)
Current 70% best mAP50-95:  v5_topk15_tal07_beta35  = 50.05% (topk=15, a=0.7, b=3.5)

Usage:
  python run_ablation_next3b.py
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
PROJECT_DIR = "runs_noaug_weapon70"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASE = v5_cls12_tight_clip
# =============================================================================
V5_BASE = {
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
    params = copy.deepcopy(V5_BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Run 1 (HIGH confidence): alpha=0.8 + topk=15 + beta=3.5
    # alpha=0.8 proven at topk=13 (79.90%). alpha=0.7 + topk=15 + beta=3.5
    # scored 80.28% with best mAP50-95. Pushing alpha 0.7->0.8 is the
    # logical next step.
    make_exp(
        "v5_topk15_tal08_beta35_70",
        "alpha=0.8 + topk=15 + beta=3.5 — highest confidence combo",
        tal_topk=15,
        tal_alpha=0.8,
        tal_beta=3.5,
    ),

    # Run 2 (MEDIUM-HIGH confidence): alpha=0.8 + topk=15 + beta=3.0
    # Same alpha=0.8 but with the mAP50-winning beta=3.0.
    # Tests alpha=0.8 at two betas to isolate the effect.
    make_exp(
        "v5_topk15_tal08_beta3_70",
        "alpha=0.8 + topk=15 + beta=3.0 — alpha=0.8 with mAP50-best beta",
        tal_topk=15,
        tal_alpha=0.8,
        tal_beta=3.0,
    ),

    # Run 3 (MEDIUM confidence): alpha=0.6 + topk=15 + beta=3.5
    # alpha=0.6 completely untested. Midpoint between alpha=0.5 (best mAP50)
    # and alpha=0.7 (best mAP50-95). Paired with best all-rounder beta.
    make_exp(
        "v5_topk15_tal06_beta35_70",
        "alpha=0.6 + topk=15 + beta=3.5 — unexplored alpha midpoint",
        tal_topk=15,
        tal_alpha=0.6,
        tal_beta=3.5,
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

    diffs = [f"{k}: {V5_BASE[k]} -> {v}" for k, v in params.items() if V5_BASE.get(k) != v]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Changes from v5_cls12_tight_clip:")
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
    print(f"  70% ABLATION — ALPHA EXPLORATION AT TOPK=15")
    print(f"  Target to beat: 80.45% mAP50 / 50.05% mAP50-95")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        diffs = [f"{k}={exp['params'][k]}" for k in exp['params'] if V5_BASE.get(k) != exp['params'][k]]
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
