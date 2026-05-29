#!/usr/bin/env python3
"""
Overnight Combos V4 — Informed by training curve analysis
===========================================================

Key insights from training CSVs:
  - wt_no_clip_no_boost1: BEST validation (73.90%), still rising at ep80, no overfitting
  - wt_boost15: BEST test (73.42%), moderate boost helps test distribution
  - wt_alpha_08_03: still rising at ep80, α=0.8 learns slower but steadier
  - Clipping causes earlier peaking and more overfitting
  - No-clip models generalize better (val box_loss LOWER than train)
  - cls=1.0 is important (biggest ablation drop -0.98%)

Strategy: combine the best generalizing configs with the best test configs.
The sweet spot is likely between no_clip_no_boost1 (best val) and boost15 (best test).

6 experiments:
  1. No clip + boost=1.5 + α=0.8 (3 best insights combined)
  2. No clip + boost=1.5 (test winner without clipping — simplest strong combo)
  3. No clip + boost=1.2 (gentler boost, no clip for stability)
  4. No clip + α=0.8 (validation winners combined, no boost)
  5. No clip + boost=1.5 + cls=0.8 (trade mAP50 for mAP50-95, no clip)
  6. No clip + boost=2.0 (between 1.5 and 2.5, no clip)

All WITHOUT clipping — training curves show it causes overfitting.
All trained on DS2, 80 epochs.
Estimated time: ~12 hours (6 × ~2h).
"""

import time
import gc
from pathlib import Path

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
# BASE CONFIG (weapon_tuned but NO clipping — based on training curve insights)
# =============================================================================
BASE_PARAMS = {
    # Phase A
    "alpha_start": 0.7,
    "alpha_end": 0.3,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "small_obj_px": 40,
    "small_obj_boost": 1.5,
    # Phase B: disabled
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    # Phase C: NO clipping (training curves show it causes overfitting)
    "iou_clip_start": 100.0,
    "iou_clip_end": 100.0,
    "dfl_clip_start": 100.0,
    "dfl_clip_end": 100.0,
    # Phase D: TAL
    "tal_topk": 13,
    "tal_alpha": 0.5,
    "tal_beta": 4.0,
    # IoU type
    "iou_type": "DIoU",
    # cls gain
    "cls": 1.0,
}

# =============================================================================
# 6 EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "v4_noclip_b15_a08",
        "desc": "No clip + boost=1.5 + α=0.8→0.3 (3 best insights combined)",
        "overrides": {
            "alpha_start": 0.8,
            "small_obj_boost": 1.5,
        },
    },
    {
        "name": "v4_noclip_b15",
        "desc": "No clip + boost=1.5 (test winner config without clipping)",
        "overrides": {
            "small_obj_boost": 1.5,
        },
    },
    {
        "name": "v4_noclip_b12",
        "desc": "No clip + boost=1.2 (gentler boost for stability)",
        "overrides": {
            "small_obj_boost": 1.2,
        },
    },
    {
        "name": "v4_noclip_a08",
        "desc": "No clip + α=0.8→0.3 + no boost (validation winners combined)",
        "overrides": {
            "alpha_start": 0.8,
            "small_obj_boost": 1.0,
            "small_obj_px": 48,
        },
    },
    {
        "name": "v4_noclip_b15_cls08",
        "desc": "No clip + boost=1.5 + cls=0.8 (trade mAP50 for mAP50-95)",
        "overrides": {
            "small_obj_boost": 1.5,
            "cls": 0.8,
        },
    },
    {
        "name": "v4_noclip_b20",
        "desc": "No clip + boost=2.0 (between 1.5 and 2.5, no clip for stability)",
        "overrides": {
            "small_obj_boost": 2.0,
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
            if epoch % 10 == 0:
                print(f"[Epoch Sync] Epoch {epoch}")
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


# =============================================================================
# TRAINING
# =============================================================================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_experiment(exp: dict) -> float:
    name = exp["name"]
    params = {**BASE_PARAMS, **exp["overrides"]}

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  Full config:")
    print(f"    alpha:     {params['alpha_start']} → {params['alpha_end']}")
    print(f"    boost:     {params['small_obj_boost']} (px={params['small_obj_px']})")
    print(f"    topk:      {params['tal_topk']}, beta: {params['tal_beta']}")
    print(f"    cls:       {params['cls']}")
    print(f"    iou_type:  {params['iou_type']}")
    print(f"    clipping:  DISABLED")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=name,
        patience=100,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        cls=params["cls"],
        alpha_start=params["alpha_start"],
        alpha_end=params["alpha_end"],
        alpha_min=params["alpha_min"],
        alpha_max=params["alpha_max"],
        small_obj_px=params["small_obj_px"],
        small_obj_boost=params["small_obj_boost"],
        center_loss_weight_init=params["center_loss_weight_init"],
        center_loss_weight_min=params["center_loss_weight_min"],
        center_loss_decay_epochs=params["center_loss_decay_epochs"],
        iou_clip_start=params["iou_clip_start"],
        iou_clip_end=params["iou_clip_end"],
        dfl_clip_start=params["dfl_clip_start"],
        dfl_clip_end=params["dfl_clip_end"],
        tal_topk=params["tal_topk"],
        tal_alpha=params["tal_alpha"],
        tal_beta=params["tal_beta"],
        iou_type=params["iou_type"],
        use_vfl=False,
    )

    training_time = (time.time() - start_time) / 3600
    print(f"\nCOMPLETED: {name} ({training_time:.2f}h)\n")
    cleanup()
    return training_time


def main():
    print(f"\n{'#' * 70}")
    print(f"# OVERNIGHT COMBOS V4 — No-clip combos (training curve informed)")
    print(f"# Insight: clipping causes overfitting, no-clip models generalize better")
    print(f"# Base: weapon_tuned config WITHOUT clipping")
    print(f"# Dataset: {DATA_YAML}")
    print(f"{'#' * 70}")

    project_path = Path(PROJECT_DIR)
    to_run = []
    for exp in EXPERIMENTS:
        run_dir = project_path / exp["name"]
        weights = run_dir / "weights" / "best.pt"
        if weights.exists():
            print(f"  [SKIP] {exp['name']} — already trained")
        else:
            to_run.append(exp)
            print(f"  [TODO] {exp['name']} — {exp['desc']}")

    if not to_run:
        print("\nAll experiments completed!")
        return

    print(f"\n{len(to_run)} experiments (~{len(to_run) * 2:.1f}h)")
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_time = 0
    for i, exp in enumerate(to_run, 1):
        print(f"\n>>> [{i}/{len(to_run)}] {exp['name']}")
        t = train_experiment(exp)
        total_time += t
        remaining = (len(to_run) - i) * (total_time / i)
        print(f"    Elapsed: {total_time:.2f}h | Remaining: {remaining:.2f}h")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {len(to_run)} experiments in {total_time:.2f}h")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
