#!/usr/bin/env python3
"""
Ablation Study for weapon_tuned_all72 (best: 73.13% mAP50)
===========================================================

Remove ONE change at a time to measure each contribution.

weapon_tuned_all72 has 6 changes vs the original swa_09_04_ds2:
  [1] Alpha 0.7→0.3 (was 0.9→0.4)
  [2] small_obj_px=40, boost=2.5 (was 48, 1.0)
  [3] TAL topk=13, beta=4.0 (was topk=10, beta=6.0)
  [4] DIoU instead of CIoU
  [5] Clipping iou 50→20, dfl 25→10 (was disabled at 100)
  [6] cls=1.0 (was 0.5)

Each ablation removes ONE change (reverts to old value), keeps all others.
If removing a change HURTS performance → that change was important.
If removing a change HELPS or is neutral → that change was unnecessary.

6 ablation runs + compare to weapon_tuned_all72 (already have results).
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage2/data.yaml"  # DS2
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# FULL weapon_tuned_all72 CONFIG (the winner)
# =============================================================================
WINNER_PARAMS = {
    # Phase A: Size-aware weighting
    "alpha_start": 0.7,
    "alpha_end": 0.3,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "small_obj_px": 40,
    "small_obj_boost": 2.5,
    # Phase B: disabled
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    # Phase C: Loosened clipping
    "iou_clip_start": 50.0,
    "iou_clip_end": 20.0,
    "dfl_clip_start": 25.0,
    "dfl_clip_end": 10.0,
    # Phase D: TAL
    "tal_topk": 13,
    "tal_alpha": 0.5,
    "tal_beta": 4.0,
    # IoU type
    "iou_type": "DIoU",
    # Classification gain
    "cls": 1.0,
}

# =============================================================================
# 6 ABLATIONS — remove one change at a time
# =============================================================================
EXPERIMENTS = [
    {
        "name": "ablation_no_alpha",
        "desc": "Remove [1] Alpha change → revert to 0.9→0.4 (original SWA schedule)",
        "overrides": {
            "alpha_start": 0.9,
            "alpha_end": 0.4,
            "alpha_min": 0.3,
            "alpha_max": 1.0,
        },
    },
    {
        "name": "ablation_no_boost",
        "desc": "Remove [2] Small obj boost → revert to px=48, boost=1.0 (neutral)",
        "overrides": {
            "small_obj_px": 48,
            "small_obj_boost": 1.0,
        },
    },
    {
        "name": "ablation_no_tal",
        "desc": "Remove [3] TAL change → revert to topk=10, beta=6.0 (original)",
        "overrides": {
            "tal_topk": 10,
            "tal_beta": 6.0,
        },
    },
    {
        "name": "ablation_no_diou",
        "desc": "Remove [4] DIoU → revert to CIoU (original)",
        "overrides": {
            "iou_type": "CIoU",
        },
    },
    {
        "name": "ablation_no_clip",
        "desc": "Remove [5] Clipping → revert to disabled (100.0)",
        "overrides": {
            "iou_clip_start": 100.0,
            "iou_clip_end": 100.0,
            "dfl_clip_start": 100.0,
            "dfl_clip_end": 100.0,
        },
    },
    {
        "name": "ablation_no_cls",
        "desc": "Remove [6] cls=1.0 → revert to cls=0.5 (original)",
        "overrides": {
            "cls": 0.5,
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
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_experiment(exp: dict) -> float:
    """Run a single training experiment. Returns training time in hours."""

    name = exp["name"]

    # Start from winner, apply overrides (revert one change)
    params = {**WINNER_PARAMS, **exp["overrides"]}

    print(f"\n{'=' * 70}")
    print(f"ABLATION: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  Reverted params:")
    for k, v in exp["overrides"].items():
        orig = WINNER_PARAMS[k]
        print(f"    {k}: {orig} → {v}")
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
        # Classification gain
        cls=params["cls"],
        # Phase A: Size-aware weighting
        alpha_start=params["alpha_start"],
        alpha_end=params["alpha_end"],
        alpha_min=params["alpha_min"],
        alpha_max=params["alpha_max"],
        small_obj_px=params["small_obj_px"],
        small_obj_boost=params["small_obj_boost"],
        # Phase B: Center loss (disabled)
        center_loss_weight_init=params["center_loss_weight_init"],
        center_loss_weight_min=params["center_loss_weight_min"],
        center_loss_decay_epochs=params["center_loss_decay_epochs"],
        # Phase C: Clipping
        iou_clip_start=params["iou_clip_start"],
        iou_clip_end=params["iou_clip_end"],
        dfl_clip_start=params["dfl_clip_start"],
        dfl_clip_end=params["dfl_clip_end"],
        # Phase D: TAL
        tal_topk=params["tal_topk"],
        tal_alpha=params["tal_alpha"],
        tal_beta=params["tal_beta"],
        # IoU type
        iou_type=params["iou_type"],
        # VFL disabled
        use_vfl=False,
    )

    training_time = (time.time() - start_time) / 3600

    print(f"\n{'=' * 50}")
    print(f"COMPLETED: {name} ({training_time:.2f}h)")
    print(f"{'=' * 50}\n")

    cleanup()
    return training_time


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'#' * 70}")
    print(f"# ABLATION STUDY — weapon_tuned_all72")
    print(f"# Remove one change at a time to measure contribution")
    print(f"# Base: 73.13% mAP50 (weapon_tuned_all72)")
    print(f"# Dataset: {DATA_YAML}")
    print(f"# Output:  {PROJECT_DIR}")
    print(f"# Epochs:  {EPOCHS}, Batch: {BATCH}, ImgSize: {IMG_SIZE}")
    print(f"{'#' * 70}")

    # Check for already completed runs
    project_path = Path(PROJECT_DIR)
    to_run = []
    for exp in EXPERIMENTS:
        run_dir = project_path / exp["name"]
        weights = run_dir / "weights" / "best.pt"
        if weights.exists():
            print(f"  [SKIP] {exp['name']} — already trained (best.pt exists)")
        else:
            to_run.append(exp)
            print(f"  [TODO] {exp['name']} — {exp['desc']}")

    if not to_run:
        print("\nAll ablations already completed!")
        return

    print(f"\n{len(to_run)} ablations to run (~{len(to_run) * 2:.1f} hours)")
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_time = 0
    for i, exp in enumerate(to_run, 1):
        print(f"\n>>> [{i}/{len(to_run)}] Starting {exp['name']}")
        t = train_experiment(exp)
        total_time += t
        remaining = (len(to_run) - i) * (total_time / i)
        print(f"    Elapsed: {total_time:.2f}h | Est. remaining: {remaining:.2f}h")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {len(to_run)} ablations in {total_time:.2f}h")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: Run CocoEvalAllFolders_weapons.py to evaluate")
    print(f"")
    print(f"Expected results table:")
    print(f"  weapon_tuned_all72    73.13%  (all changes)")
    print(f"  ablation_no_alpha     ???%    (remove alpha change)")
    print(f"  ablation_no_boost     ???%    (remove small obj boost)")
    print(f"  ablation_no_tal       ???%    (remove TAL change)")
    print(f"  ablation_no_diou      ???%    (remove DIoU)")
    print(f"  ablation_no_clip      ???%    (remove clipping)")
    print(f"  ablation_no_cls       ???%    (remove cls=1.0)")
    print(f"")
    print(f"  Drop = how much that change contributed")
    print(f"  Biggest drop = most important change")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
