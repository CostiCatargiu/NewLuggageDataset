#!/usr/bin/env python3
"""
4 Proposals — based on wt_boost15 (73.42% mAP50, current best)

wt_boost15 config (the actual winner):
  alpha: 0.7->0.3, alpha_min=0.2, alpha_max=0.8
  small_obj_px=40, boost=1.5
  TAL: topk=13, alpha=0.5, beta=4.0
  DIoU, cls=1.0
  clipping: iou 50->20, dfl 25->10
  center loss: OFF
  VFL: OFF (BCE)

Each model changes 1-2 params from wt_boost15:
  1. wt_beta3_topk15       — beta 4->3, topk 13->15
  2. wt_cls15_boost30      — cls 1.0->1.5, boost 1.5->3.0
  3. wt_box5_dfl1          — box 7.5->5.0, dfl 1.5->1.0 (shift gradient to cls)
  4. wt_tal_alpha08        — tal_alpha 0.5->0.8 (assigner trusts cls more)

Usage:
  python run_4_proposals.py
"""

import time
import gc
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage2/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASE = wt_boost15 (the actual winner, 73.42% mAP50)
# =============================================================================
BASE = {
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


def make_exp(name, desc, **overrides):
    params = copy.deepcopy(BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Model 1: push TAL further
    make_exp(
        "wt_beta3_topk15",
        "Push TAL: beta 4->3, topk 13->15",
        tal_beta=3.0,
        tal_topk=15,
    ),

    # Model 2: more cls pressure + stronger boost for "other" class
    make_exp(
        "wt_cls15_boost30",
        "Break 'other' ceiling: cls 1.0->1.5, boost 1.5->3.0",
        cls=1.5,
        small_obj_boost=3.0,
    ),

    # Model 3: shift gradient balance toward cls via loss gains (never tested)
    make_exp(
        "wt_box5_dfl1",
        "Shift gradient to cls: box 7.5->5.0, dfl 1.5->1.0",
        box=5.0,
        dfl=1.0,
    ),

    # Model 4: assigner trusts classification more (tal_alpha never tested at 0.8)
    make_exp(
        "wt_tal_alpha08",
        "Assigner trusts cls more: tal_alpha 0.5->0.8",
        tal_alpha=0.8,
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

    # Show what changed from BASE
    diffs = []
    for k, v in params.items():
        if BASE.get(k) != v:
            diffs.append(f"{k}: {BASE.get(k)} -> {v}")

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Changes from wt_boost15:")
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
    print(f"  4 PROPOSALS from wt_boost15 baseline (73.42%)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")
    print(f"  wt_boost15 base: topk=13, b=4.0, cls=1.0, a=0.7->0.3,")
    print(f"                   boost=1.5, DIoU, clip=50->20, center=OFF")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        diffs = [f"{k}={exp['params'][k]}" for k in exp['params'] if BASE.get(k) != exp['params'][k]]
        print(f"  [{i+1}] {exp['name']:<25} diff: {', '.join(diffs)}")

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
