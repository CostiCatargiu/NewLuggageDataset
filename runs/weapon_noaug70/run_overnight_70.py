#!/usr/bin/env python3
"""
Overnight — 6 untested configs on 70% ablation.

Current 70% best: v5_tal_alpha08_70 = 79.90%
Ordered by chance of beating it.

Usage:
  python run_overnight_70.py
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
# v5_cls12_tight_clip base (full winner)
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


# Ordered by chance of beating 79.90%
EXPERIMENTS = [
    # 40% — boost=1.5 consistently beat 2.5, never tested with cls=1.2
    make_exp(
        "v5_boost15_cls12_70",
        "boost 2.5->1.5 with cls=1.2 (40%)",
        small_obj_boost=1.5,
    ),

    # 35% — three proven winners combined: boost=1.5 + cls=1.2 + tal_alpha=0.8
    make_exp(
        "v5_boost15_cls12_tal08_70",
        "boost=1.5 + cls=1.2 + tal_alpha=0.8 (35%)",
        small_obj_boost=1.5,
        tal_alpha=0.8,
    ),

    # 30% — tal_alpha=0.7, midpoint between full winner (0.5) and ablation winner (0.8)
    make_exp(
        "v5_tal07_70",
        "tal_alpha=0.7 midpoint (30%)",
        tal_alpha=0.7,
    ),

    # 30% — full winner config but loose clip, might fix mAP50-95 drop
    make_exp(
        "v5_cls12_loose_70",
        "v5 config + loose clip 50->20 (30%)",
        iou_clip_start=50.0,
        iou_clip_end=20.0,
        dfl_clip_start=25.0,
        dfl_clip_end=10.0,
    ),

    # 25% — beta=3.0 with v5 config, isolate beta effect
    make_exp(
        "v5_cls12_beta3_70",
        "beta 4.0->3.0 with v5 config (25%)",
        tal_beta=3.0,
    ),

    # 25% — topk=15 + beta=3.0 with cls=1.2
    make_exp(
        "v5_topk15_beta3_70",
        "topk=15 + beta=3.0 with cls=1.2 (25%)",
        tal_topk=15,
        tal_beta=3.0,
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
    print(f"  OVERNIGHT — 6 CONFIGS ON 70% ABLATION")
    print(f"  Target to beat: 79.90% (v5_tal_alpha08_70)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        diffs = [f"{k}={exp['params'][k]}" for k in exp['params'] if V5_BASE.get(k) != exp['params'][k]]
        print(f"  [{i+1}] {exp['name']:<30} {', '.join(diffs) if diffs else 'base config'}")

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
