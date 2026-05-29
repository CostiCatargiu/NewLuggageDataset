#!/usr/bin/env python3
"""
Overnight Combos V3 — Smart combinations around wt_boost15 (best: 73.42% mAP50)
=================================================================================

wt_boost15 config (current winner):
  alpha: 0.7→0.3, small_obj_px=40, boost=1.5
  TAL: topk=13, alpha=0.5, beta=4.0
  DIoU, cls=1.0, clipping: iou 50→20, dfl 25→10

Insights from all test sets + validation:
  - wt_boost15: #1 on DS1, DS2, Full test
  - wt_alpha_08_03: #2 on validation, good DS1
  - wt_no_clip_no_boost1: #1 on validation, smallest test-valid gap
  - topk11 was close to topk13 in earlier experiments

3 smart combos combining wt_boost15 with best validation insights:
  1. wt_boost15 + alpha 0.8→0.3 (test winner + valid winner)
  2. wt_boost15 + no clipping (simpler, clipping was least important)
  3. wt_boost15 + topk11 (explore topk sweet spot with boost=1.5)

All trained on DS2. Estimated time: ~6 hours (3 × ~2h).
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
# BASE CONFIG (wt_boost15 — current winner)
# =============================================================================
WINNER_PARAMS = {
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
# 3 EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "wt_b15_alpha08",
        "desc": "boost=1.5 (test winner) + alpha 0.8→0.3 (valid winner)",
        "overrides": {
            "alpha_start": 0.8,
        },
    },
    {
        "name": "wt_b15_no_clip",
        "desc": "boost=1.5 + no clipping (simpler config, clipping was least important)",
        "overrides": {
            "iou_clip_start": 100.0,
            "iou_clip_end": 100.0,
            "dfl_clip_start": 100.0,
            "dfl_clip_end": 100.0,
        },
    },
    {
        "name": "wt_b15_topk11",
        "desc": "boost=1.5 + topk=11 (topk11 was close to topk13 in earlier tests)",
        "overrides": {
            "tal_topk": 11,
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
    params = {**WINNER_PARAMS, **exp["overrides"]}

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  Changes from wt_boost15:")
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
    print(f"# OVERNIGHT COMBOS V3 — Around wt_boost15 (73.42%)")
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
