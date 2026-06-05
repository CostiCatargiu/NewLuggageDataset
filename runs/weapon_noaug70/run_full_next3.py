#!/usr/bin/env python3
"""
Full Dataset — 3 configs based on complete analysis of 14 ablation + 8 full runs.

KEY FINDINGS FROM DATA:
  - Full dataset rewards higher beta (stricter localization) vs ablation
  - topk=15 consistently beats topk=13 on mAP50/recall at both scales
  - alpha=0.7 was #4 at 70% (0.8002) but NEVER tested on full
  - topk=15 + beta=4.0 never tested on full (only beta=3.0 and 3.5)
  - The full best (topk=13, beta=4.0) = 83.03%, topk=15 best = 82.93%

STRATEGY:
  1. topk=15 + beta=4.0 — fill the obvious gap: topk=15 with the full-proven beta
  2. topk=15 + alpha=0.7 + beta=4.0 — alpha=0.7 was strong at 70%, combine with topk=15
  3. topk=15 + alpha=0.7 + beta=3.5 — same but with beta35 (best all-rounder beta)

Current full best mAP50:     v5_cls12_tight_clip_full = 83.03% (topk=13, a=0.5, b=4.0)
Current full best all-round: v5_topk15_beta35_full    = 82.93% (topk=15, a=0.5, b=3.5)

Usage:
  python run_full_next3.py
"""

import time
import gc
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_noaug_weapon"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASE = v5_cls12_tight_clip (proven full baseline)
# =============================================================================
BASE = {
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
    params = copy.deepcopy(BASE)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


EXPERIMENTS = [
    # Run 1: topk=15 + beta=4.0
    # RATIONALE: topk=15 is proven (best mAP50 at 70%, best recall on full).
    # beta=4.0 is the full-dataset winner's beta. This exact combo was NEVER
    # tested on full. At 70% it scored 79.57% — modest, but full dataset
    # consistently boosts topk=15 configs more than ablation predicts.
    # This is the most obvious gap in our search space.
    make_exp(
        "v5_topk15_beta4_full",
        "topk=15 + beta=4.0 — obvious untested combo, fill the gap",
        tal_topk=15,
        tal_beta=4.0,
    ),

    # Run 2: topk=15 + alpha=0.7 + beta=4.0
    # RATIONALE: alpha=0.7 was #4 at 70% (80.02%) with topk=13 — strong.
    # It was NEVER tested on full dataset. Combining it with topk=15 and
    # the full-proven beta=4.0 merges two independently strong signals.
    # At 70%, topk=15+alpha=0.7+beta=4.0 scored 79.62%, but alpha=0.7
    # with topk=13 scored 80.02%, suggesting alpha=0.7 works better with
    # moderate topk — worth testing if full dataset changes this dynamic.
    make_exp(
        "v5_topk15_tal07_beta4_full",
        "topk=15 + alpha=0.7 + beta=4.0 — alpha=0.7 never tested on full",
        tal_topk=15,
        tal_alpha=0.7,
        tal_beta=4.0,
    ),

    # Run 3: topk=15 + alpha=0.7 + beta=3.5
    # RATIONALE: Same alpha=0.7 exploration, but with beta=3.5 which produced
    # the best all-rounder on full (82.93%). If alpha=0.7 helps mAP50 at
    # topk=15, pairing it with beta=3.5 should maximize recall + small objects.
    # This tests alpha=0.7 at two beta values to isolate its effect.
    make_exp(
        "v5_topk15_tal07_beta35_full",
        "topk=15 + alpha=0.7 + beta=3.5 — alpha=0.7 + best all-rounder beta",
        tal_topk=15,
        tal_alpha=0.7,
        tal_beta=3.5,
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

    diffs = [f"{k}: {BASE[k]} -> {v}" for k, v in params.items() if BASE.get(k) != v]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Changes from v5_cls12_tight_clip (current full best 83.03%):")
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
    print(f"  FULL DATASET — 3 DATA-DRIVEN CONFIGS")
    print(f"  Target to beat: 83.03% mAP50 (v5_cls12_tight_clip_full)")
    print(f"{'=' * 70}")
    print(f"  Model:    {MODEL_WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}    Batch: {BATCH}")
    print(f"{'=' * 70}")

    for i, exp in enumerate(EXPERIMENTS):
        diffs = [f"{k}={exp['params'][k]}" for k in exp['params'] if BASE.get(k) != exp['params'][k]]
        print(f"  [{i+1}] {exp['name']:<40} {', '.join(diffs) if diffs else 'base config'}")

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
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
