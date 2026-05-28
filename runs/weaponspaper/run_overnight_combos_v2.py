#!/usr/bin/env python3
"""
Overnight Combos V2 — Smart combinations based on ablation insights
====================================================================

Ablation showed:
  - no_alpha: BEST small objects (56.47%) and recall (68.36%)
  - no_cls: BEST mAP50-95 (43.29%)
  - no_clip: almost same mAP50 (73.00%) + better mAP50-95
  - no_boost: BEST precision (80.43%)

Idea: combine the "winning" ablation insights into new configs.
Also: try weapon_tuned on the old alpha schedule (which was proven best for mAP50)
      but keep the new changes that helped (DIoU, cls, TAL, clipping).

6 experiments:
  1. weapon_tuned WITHOUT clipping AND without boost (two least important removed)
  2. weapon_tuned with old alpha (0.9→0.4) but keep everything else (DIoU, cls=1.0, TAL, clip, boost)
  3. weapon_tuned with cls=0.75 (between 0.5 and 1.0 — balance mAP50 and mAP50-95)
  4. weapon_tuned with topk=12, beta=4.0 (between topk11 and topk13)
  5. weapon_tuned with alpha 0.8→0.3 (more area focus early, same end)
  6. weapon_tuned with boost=1.5 (between 1.0 and 2.5)

All trained on DS2. Uses loss_weapon_tuned.py.
Estimated time: ~12 hours (6 × ~2h). Skips already-completed runs.
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
# BASE CONFIG (weapon_tuned_all72 winner)
# =============================================================================
WINNER_PARAMS = {
    "alpha_start": 0.7,
    "alpha_end": 0.3,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "small_obj_px": 40,
    "small_obj_boost": 2.5,
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
    {
        "name": "wt_old_alpha_new_rest",
        "desc": "Old best alpha (0.9→0.4) + all new changes (DIoU, cls=1.0, TAL, clip, boost)",
        "overrides": {
            "alpha_start": 0.9,
            "alpha_end": 0.4,
            "alpha_min": 0.3,
            "alpha_max": 1.0,
        },
    },
    {
        "name": "wt_no_clip_no_boost",
        "desc": "Remove 2 least important: no clipping + no boost (simplest config)",
        "overrides": {
            "iou_clip_start": 100.0,
            "iou_clip_end": 100.0,
            "dfl_clip_start": 100.0,
            "dfl_clip_end": 100.0,
            "small_obj_boost": 1.0,
            "small_obj_px": 48,
        },
    },
    {
        "name": "wt_cls075",
        "desc": "cls=0.75 — balance between mAP50 (cls=1.0) and mAP50-95 (cls=0.5)",
        "overrides": {
            "cls": 0.75,
        },
    },
    {
        "name": "wt_topk12",
        "desc": "topk=12 — between topk11 (73.15%) and topk13 (73.13%)",
        "overrides": {
            "tal_topk": 12,
        },
    },
    {
        "name": "wt_alpha_08_03",
        "desc": "Alpha 0.8→0.3 — more area focus early, same end as winner",
        "overrides": {
            "alpha_start": 0.8,
        },
    },
    {
        "name": "wt_boost15",
        "desc": "boost=1.5 — between neutral (1.0) and aggressive (2.5)",
        "overrides": {
            "small_obj_boost": 1.5,
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
    print(f"  Changes from winner:")
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
    print(f"# OVERNIGHT COMBOS V2 — Smart combinations from ablation")
    print(f"# Base: weapon_tuned_all72 (73.13% mAP50)")
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
