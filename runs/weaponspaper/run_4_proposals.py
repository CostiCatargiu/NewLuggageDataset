#!/usr/bin/env python3
"""
4 Incremental Proposals — based on what actually worked in previous results.

Current best ablation models (73.0-73.6% mAP50) all share:
  - topk=13, beta=4.0, cls=1.0, alpha 0.9->0.5, CIoU, BCE

What these 4 test:
  1. wt_beta3_topk15       — push TAL further (beta 4->3, topk 13->15)
  2. wt_cls15_boost30      — break "other" class ceiling (cls=1.5, boost=3.0)
  3. wt_alpha10_06_topk13  — max area-weight early (alpha 1.0->0.6)
  4. wt_beta4_phaseb       — center loss ON with proven TAL config

Target to beat: 73.6% mAP50 (v5_cls12_tight_clip)

Usage:
  python run_4_proposals.py
"""

import time
import gc
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
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "wt_beta3_topk15",
        "description": "Push TAL further: beta 4->3, topk 13->15",
        "params": {
            "cls": 1.0,
            "alpha_start": 0.9,
            "alpha_end": 0.5,
            "alpha_min": 0.3,
            "alpha_max": 0.9,
            "small_obj_px": 40,
            "small_obj_boost": 2.5,
            "center_loss_weight_init": 0.0,
            "center_loss_weight_min": 0.0,
            "center_loss_decay_epochs": 35,
            "iou_clip_start": 20.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 10.0,
            "dfl_clip_end": 5.0,
            "tal_topk": 15,
            "tal_alpha": 0.5,
            "tal_beta": 3.0,
            "iou_type": "CIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "wt_cls15_boost30",
        "description": "Break 'other' class ceiling: cls=1.5, boost=3.0",
        "params": {
            "cls": 1.5,
            "alpha_start": 0.9,
            "alpha_end": 0.5,
            "alpha_min": 0.3,
            "alpha_max": 0.9,
            "small_obj_px": 40,
            "small_obj_boost": 3.0,
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
            "iou_type": "CIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "wt_alpha10_06_topk13",
        "description": "Max area-weight early: alpha 1.0->0.6",
        "params": {
            "cls": 1.0,
            "alpha_start": 1.0,
            "alpha_end": 0.6,
            "alpha_min": 0.4,
            "alpha_max": 1.0,
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
            "iou_type": "CIoU",
            "use_vfl": False,
        },
    },
    {
        "name": "wt_beta4_phaseb",
        "description": "Phase B center loss ON with proven TAL config",
        "params": {
            "cls": 1.0,
            "alpha_start": 0.9,
            "alpha_end": 0.5,
            "alpha_min": 0.3,
            "alpha_max": 0.9,
            "small_obj_px": 40,
            "small_obj_boost": 2.5,
            "center_loss_weight_init": 0.5,
            "center_loss_weight_min": 0.05,
            "center_loss_decay_epochs": 40,
            "iou_clip_start": 20.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 10.0,
            "dfl_clip_end": 5.0,
            "tal_topk": 13,
            "tal_alpha": 0.5,
            "tal_beta": 4.0,
            "iou_type": "CIoU",
            "use_vfl": False,
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
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


# =============================================================================
# TRAINING LOOP
# =============================================================================
def run_experiment(exp):
    name = exp["name"]
    params = exp["params"]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Key: topk={params['tal_topk']}, beta={params['tal_beta']}, "
          f"cls={params['cls']}, alpha={params['alpha_start']}->{params['alpha_end']}, "
          f"boost={params['small_obj_boost']}, "
          f"center={'ON' if params['center_loss_weight_init'] > 0 else 'OFF'}")
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
        print(f"\n  FAILED: {name} ({elapsed:.2f}h) — {e}")
        return {"name": name, "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  4 PROPOSALS — INCREMENTAL EXPERIMENTS")
    print(f"  Target to beat: 73.6% mAP50 (v5_cls12_tight_clip)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Batch:    {BATCH}")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        p = exp["params"]
        print(f"  [{i+1}] {exp['name']:<30} "
              f"topk={p['tal_topk']:<3} b={p['tal_beta']:<4} "
              f"cls={p['cls']:<4} a={p['alpha_start']}->{p['alpha_end']} "
              f"boost={p['small_obj_boost']:<4} "
              f"center={'ON' if p['center_loss_weight_init'] > 0 else 'OFF'}")

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
