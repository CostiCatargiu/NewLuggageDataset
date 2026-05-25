#!/usr/bin/env python3
"""
Overnight Ablation Runs — 5 experiments on DS1
===============================================

1. swa_10_05          — SWA 1.0→0.5 (extreme start, upper bound)
2. swa_09_06          — SWA 0.9→0.6 (not enough shift, lower bound of end)
3. swa_08_05_topk6    — SWA 0.8→0.5 with topk=6 (extreme low topk)
4. swa_08_05_topk8    — SWA 0.8→0.5 with topk=8 (clean topk comparison)
5. swa_08_05_topk12   — SWA 0.8→0.5 with topk=12 (clean topk comparison on DS1)
6. swa_08_06          — SWA 0.8→0.6 (not enough shift for start=0.8)
7. swa_08_05_phasec2  — SWA 0.8→0.5 + clipping (iou 50→20, dfl 25→10)
8. swa_09_04_phasec2  — SWA 0.9→0.4 + clipping (iou 50→20, dfl 25→10)

All trained on DS1 with TAL 0.5/6.0 (default).
No COCO evaluation — just training. Use CocoEvalAllFolders_weapons.py after.

Estimated time: ~12 hours (8 × ~1.5h per run). Skips already-completed runs.
"""

import time
import gc
from pathlib import Path

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

EPOCHS = 1
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# FIXED PARAMETERS (disabled/neutral)
# =============================================================================
FIXED_PARAMS = {
    # Phase A: fixed
    "alpha_min": 0.3,
    "alpha_max": 1.0,
    "small_obj_px": 48,
    "small_obj_boost": 1.0,
    # Phase B: disabled
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    # Phase C: disabled
    "iou_clip_start": 100.0,
    "iou_clip_end": 100.0,
    "dfl_clip_start": 100.0,
    "dfl_clip_end": 100.0,
}

# =============================================================================
# EXPERIMENTS TO RUN
# =============================================================================
EXPERIMENTS = [
    {
        "name": "swa_08_5_tal_07_42",
        "desc": "SWA 1.0→0.5 — extreme start (upper bound)",
        "alpha_start": 0.8,
        "alpha_end": 0.5,
        "tal_topk": 10,
        "tal_alpha": 0.7,
        "tal_beta": 4.0,
    },
    # {
    #     "name": "swa_09_06",
    #     "desc": "SWA 0.9→0.6 — not enough shift (end point boundary)",
    #     "alpha_start": 0.9,
    #     "alpha_end": 0.6,
    #     "tal_topk": 10,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    # },
    # {
    #     "name": "swa_08_05_topk6",
    #     "desc": "SWA 0.8→0.5, topk=6 — extreme low topk",
    #     "alpha_start": 0.8,
    #     "alpha_end": 0.5,
    #     "tal_topk": 6,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    # },
    # {
    #     "name": "swa_08_05_topk8",
    #     "desc": "SWA 0.8→0.5, topk=8 — low topk",
    #     "alpha_start": 0.8,
    #     "alpha_end": 0.5,
    #     "tal_topk": 8,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    # },
    # {
    #     "name": "swa_08_05_topk12",
    #     "desc": "SWA 0.8→0.5, topk=12 — high topk (DS1 version)",
    #     "alpha_start": 0.8,
    #     "alpha_end": 0.5,
    #     "tal_topk": 12,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    # },
    # {
    #     "name": "swa_08_06",
    #     "desc": "SWA 0.8→0.6 — not enough shift (end point boundary for start=0.8)",
    #     "alpha_start": 0.8,
    #     "alpha_end": 0.6,
    #     "tal_topk": 10,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    # },
    # {
    #     "name": "swa_08_05_phasec2",
    #     "desc": "SWA 0.8→0.5 + clipping (iou 50→20, dfl 25→10)",
    #     "alpha_start": 0.8,
    #     "alpha_end": 0.5,
    #     "tal_topk": 10,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    #     "iou_clip_start": 50.0,
    #     "iou_clip_end": 20.0,
    #     "dfl_clip_start": 25.0,
    #     "dfl_clip_end": 10.0,
    # },
    # {
    #     "name": "swa_09_04_phasec2",
    #     "desc": "SWA 0.9→0.4 + clipping (iou 50→20, dfl 25→10)",
    #     "alpha_start": 0.9,
    #     "alpha_end": 0.4,
    #     "tal_topk": 10,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    #     "iou_clip_start": 50.0,
    #     "iou_clip_end": 20.0,
    #     "dfl_clip_start": 25.0,
    #     "dfl_clip_end": 10.0,
    # },
    # {
    #     "name": "original_loss_phasec2",
    #     "desc": "No SWA (alpha=1.0→1.0) + clipping (iou 50→20, dfl 25→10)",
    #     "alpha_start": 1.0,
    #     "alpha_end": 1.0,
    #     "tal_topk": 10,
    #     "tal_alpha": 0.5,
    #     "tal_beta": 6.0,
    #     "iou_clip_start": 50.0,
    #     "iou_clip_end": 20.0,
    #     "dfl_clip_start": 25.0,
    #     "dfl_clip_end": 10.0,
    # },
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
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    # Get clipping values (per-experiment override or default disabled)
    iou_clip_start = exp.get("iou_clip_start", FIXED_PARAMS["iou_clip_start"])
    iou_clip_end = exp.get("iou_clip_end", FIXED_PARAMS["iou_clip_end"])
    dfl_clip_start = exp.get("dfl_clip_start", FIXED_PARAMS["dfl_clip_start"])
    dfl_clip_end = exp.get("dfl_clip_end", FIXED_PARAMS["dfl_clip_end"])
    has_clipping = iou_clip_start < 100.0

    print(f"  alpha_start:    {exp['alpha_start']}")
    print(f"  alpha_end:      {exp['alpha_end']}")
    print(f"  tal_topk:       {exp['tal_topk']}")
    print(f"  tal_alpha:      {exp['tal_alpha']}")
    print(f"  tal_beta:       {exp['tal_beta']}")
    if has_clipping:
        print(f"  iou_clip:       {iou_clip_start}→{iou_clip_end}")
        print(f"  dfl_clip:       {dfl_clip_start}→{dfl_clip_end}")
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
        # SWA schedule
        alpha_start=exp["alpha_start"],
        alpha_end=exp["alpha_end"],
        alpha_min=FIXED_PARAMS["alpha_min"],
        alpha_max=FIXED_PARAMS["alpha_max"],
        small_obj_px=FIXED_PARAMS["small_obj_px"],
        small_obj_boost=FIXED_PARAMS["small_obj_boost"],
        # Phase B: disabled
        center_loss_weight_init=FIXED_PARAMS["center_loss_weight_init"],
        center_loss_weight_min=FIXED_PARAMS["center_loss_weight_min"],
        center_loss_decay_epochs=FIXED_PARAMS["center_loss_decay_epochs"],
        # Phase C: clipping (per-experiment or disabled)
        iou_clip_start=iou_clip_start,
        iou_clip_end=iou_clip_end,
        dfl_clip_start=dfl_clip_start,
        dfl_clip_end=dfl_clip_end,
        # TAL parameters
        tal_topk=exp["tal_topk"],
        tal_alpha=exp["tal_alpha"],
        tal_beta=exp["tal_beta"],
        # Reproducibility
        seed=0,
        deterministic=True,
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
    print(f"# OVERNIGHT ABLATION — 5 experiments on DS1")
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

    print(f"\n{len(to_run)} experiments to run (~{len(to_run) * 1.3:.1f} hours)")
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
