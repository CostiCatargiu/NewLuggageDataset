#!/usr/bin/env python3
"""
Overnight Runs — 6 variations around weapon_tuned_all72 (best: 73.13% mAP50)
=============================================================================

Base config (weapon_tuned_all72):
  alpha: 0.7→0.3, small_obj_px=40, boost=2.5
  TAL: topk=13, alpha=0.5, beta=4.0
  DIoU, cls=1.0, clipping: iou 50→20, dfl 25→10

Variations explore:
  1. Higher topk (15) — more anchor candidates
  2. Lower topk (11) — between old best (10) and current (13)
  3. Alpha 0.8→0.3 — more area focus early (was 0.7→0.3)
  4. Alpha 0.7→0.4 — less score focus late (was 0.7→0.3)
  5. Beta 3.0 — even less IoU weight in TAL assignment
  6. Beta 5.0 — between old (6.0) and current (4.0)

All trained on DS2. Uses loss_weapon_tuned.py (DIoU + all changes).
Estimated time: ~12-14 hours (6 × ~2h). Skips already-completed runs.
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
# BASE CONFIG (from weapon_tuned_all72 winner)
# =============================================================================
BASE_PARAMS = {
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
# 6 EXPERIMENTS — variations around the winner
# =============================================================================
EXPERIMENTS = [
    {
        "name": "wt_topk15",
        "desc": "topk=15 — more anchor candidates (was 13)",
        "overrides": {"tal_topk": 15},
    },
    {
        "name": "wt_topk11",
        "desc": "topk=11 — between old best (10) and current (13)",
        "overrides": {"tal_topk": 11},
    },
    {
        "name": "wt_cls08",
        "desc": "cls=0.8 — between old (0.5) and current (1.0)",
        "overrides": {"cls": 0.8},
    },
    {
        "name": "wt_ciou",
        "desc": "CIoU instead of DIoU — test if DIoU actually matters",
        "overrides": {"iou_type": "CIoU"},
    },
    {
        "name": "wt_beta3",
        "desc": "Beta=3.0 — even less IoU weight in TAL (was 4.0)",
        "overrides": {"tal_beta": 3.0},
    },
    {
        "name": "wt_beta5",
        "desc": "Beta=5.0 — between old (6.0) and current (4.0)",
        "overrides": {"tal_beta": 5.0},
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
    
    # Merge base params with experiment overrides
    params = {**BASE_PARAMS, **exp["overrides"]}
    
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  alpha:     {params['alpha_start']} → {params['alpha_end']}")
    print(f"  tal_topk:  {params['tal_topk']}")
    print(f"  tal_beta:  {params['tal_beta']}")
    print(f"  boost:     {params['small_obj_boost']}")
    print(f"  cls:       {params['cls']}")
    print(f"  iou_type:  {params['iou_type']}")
    print(f"  clipping:  iou {params['iou_clip_start']}→{params['iou_clip_end']}, "
          f"dfl {params['dfl_clip_start']}→{params['dfl_clip_end']}")
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
    print(f"# OVERNIGHT WEAPON-TUNED VARIATIONS — 6 experiments on DS2")
    print(f"# Base: weapon_tuned_all72 (73.13% mAP50)")
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
        print("\nAll experiments already completed!")
        return

    print(f"\n{len(to_run)} experiments to run (~{len(to_run) * 2:.1f} hours)")
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_time = 0
    for i, exp in enumerate(to_run, 1):
        print(f"\n>>> [{i}/{len(to_run)}] Starting {exp['name']}")
        t = train_experiment(exp)
        total_time += t
        remaining = (len(to_run) - i) * (total_time / i)
        print(f"    Elapsed: {total_time:.2f}h | Est. remaining: {remaining:.2f}h")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {len(to_run)} experiments in {total_time:.2f}h")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: Run CocoEvalAllFolders_weapons.py to evaluate on DS1/DS2/Full")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
