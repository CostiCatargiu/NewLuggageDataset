#!/usr/bin/env python3
"""
70% Ablation — SWA exploration with best TAL config.

EVIDENCE:
  - v5_alpha09_04_full used SWA 0.9/0.4 with old TAL (a=0.5) → 82.30% on full
  - Best TAL (a=0.7) uses default SWA 0.7/0.3 → 83.12% on full
  - SWA 0.9/0.4 and 0.9/0.5 NEVER combined with tal_alpha=0.7
  - Stronger SWA = more averaging = potentially better generalization

TESTING:
  1. Best TAL + SWA 0.9/0.4 — same SWA as v5_alpha09_04
  2. Best TAL + SWA 0.9/0.5 — slightly less aggressive

Current 70% best: v5_topk15_beta3_70 = 80.45% (but v5_tal07_70 = 80.02% is most consistent)

Usage:
  python run_ablation_swa.py
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
# BASE = v5_tal07 (best TAL, default SWA 0.7/0.3)
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
    "tal_alpha": 0.7,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}


def make_exp(name, desc, **overrides):
    params = copy.deepcopy(V5_BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Run 1: SWA 0.9/0.4 — same as v5_alpha09_04 but with best TAL
    make_exp(
        "v5_tal07_swa09_04_70",
        "best TAL + SWA 0.9/0.4 — stronger averaging",
        alpha_start=0.9,
        alpha_end=0.4,
        alpha_min=0.3,
        alpha_max=0.9,
    ),

    # Run 2: SWA 0.9/0.5 — slightly less aggressive
    make_exp(
        "v5_tal07_swa09_05_70",
        "best TAL + SWA 0.9/0.5 — moderate averaging",
        alpha_start=0.9,
        alpha_end=0.5,
        alpha_min=0.4,
        alpha_max=0.9,
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
    print(f"# Changes from v5_tal07 (SWA 0.7/0.3):")
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
    print(f"  70% ABLATION — SWA EXPLORATION WITH BEST TAL")
    print(f"  Target to beat: 80.02% mAP50 (v5_tal07_70)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"  Base TAL: topk=13, alpha=0.7, beta=4.0")
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
