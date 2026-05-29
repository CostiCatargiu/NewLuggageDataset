#!/usr/bin/env python3
"""
Overnight Combos V5 — Fine-tuning around wt_boost15 (best: 73.42% mAP50)
==========================================================================

Everything we learned:
  - boost=1.5 with clipping is the best (73.42%)
  - Clipping HELPS test generalization (no-clip peaks on val but drops on test)
  - cls=1.0 is the most important single change
  - topk=11-13 range is optimal
  - α=0.7→0.3 is best, α=0.8 steadier but lower
  - β=4.0 confirmed

Strategy: fine-tune around wt_boost15. Explore:
  1. Tighter clipping (current 50→20, try 30→10 — more regularization)
  2. cls=1.2 (cls was most important, push it higher)
  3. cls=1.5 (even more aggressive cls)  
  4. topk=14 + boost=1.5 (topk14 was bad alone, but might work with boost=1.5)
  5. β=3.5 (fine-tune between 3.0 and 4.0)
  6. boost=1.5 + α=0.75→0.3 (between 0.7 and 0.8, splitting the difference)

All with clipping enabled. All on DS2.
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
# BASE CONFIG (wt_boost15 — current best)
# =============================================================================
BEST_PARAMS = {
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
    "cls": 1.0,
}

# =============================================================================
# 6 EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    # ── HIGH CONFIDENCE ──────────────────────────────────────────────────
    {
        "name": "v5_cls12",
        "desc": "[HIGH] cls=1.2 — cls was #1 ablation contributor, wt_cls08 was #2 on DS1",
        "overrides": {
            "cls": 1.2,
        },
    },
    {
        "name": "v5_tight_clip",
        "desc": "[HIGH] Tighter clipping: iou 30→10, dfl 15→5 — clipping helps all test sets",
        "overrides": {
            "iou_clip_start": 30.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 15.0,
            "dfl_clip_end": 5.0,
        },
    },
    {
        "name": "v5_alpha075",
        "desc": "[HIGH] α=0.75→0.3 — split between 0.7 (best mAP50) and 0.8 (steadiest)",
        "overrides": {
            "alpha_start": 0.75,
        },
    },
    # ── MEDIUM CONFIDENCE ────────────────────────────────────────────────
    {
        "name": "v5_cls12_alpha075",
        "desc": "[MED] cls=1.2 + α=0.75 — combine two high-confidence changes",
        "overrides": {
            "cls": 1.2,
            "alpha_start": 0.75,
        },
    },
    {
        "name": "v5_cls12_tight_clip",
        "desc": "[MED] cls=1.2 + tighter clipping — combine two high-confidence changes",
        "overrides": {
            "cls": 1.2,
            "iou_clip_start": 30.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 15.0,
            "dfl_clip_end": 5.0,
        },
    },
    # ── EXPLORATORY ──────────────────────────────────────────────────────
    {
        "name": "v5_cls12_alpha075_tight",
        "desc": "[EXPL] cls=1.2 + α=0.75 + tight clip — all 3 high-confidence combined",
        "overrides": {
            "cls": 1.2,
            "alpha_start": 0.75,
            "iou_clip_start": 30.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 15.0,
            "dfl_clip_end": 5.0,
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
    params = {**BEST_PARAMS, **exp["overrides"]}

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  Changes from wt_boost15:")
    for k, v in exp["overrides"].items():
        orig = BEST_PARAMS[k]
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
    print(f"# OVERNIGHT COMBOS V5 — Fine-tuning wt_boost15 (73.42%)")
    print(f"# Strategy: keep clipping, explore cls/topk/beta/alpha fine-tuning")
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
