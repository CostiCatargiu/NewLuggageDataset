#!/usr/bin/env python3
"""
Phase B (Center Loss) Test — Add center loss to the 2 best configs
===================================================================

Testing center loss (Phase B) with the two best models:
  1. wt_boost15 (73.42% mAP50) — overall best, most balanced
  2. v5_cls12_tight_clip (73.57% mAP50) — best test mAP50, best recall

Center loss: L1 penalty on center coordinates for small objects.
  - weight_init=0.3 (gentle)
  - weight_min=0.05
  - decay_epochs=40 (strong first 40 epochs, minimal after)

Estimated time: ~4 hours (2 × ~2h).
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
# 2 EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "wt_boost15_phaseB",
        "desc": "Best overall (wt_boost15) + center loss",
        "params": {
            # Phase A
            "alpha_start": 0.7,
            "alpha_end": 0.3,
            "alpha_min": 0.2,
            "alpha_max": 0.8,
            "small_obj_px": 40,
            "small_obj_boost": 1.5,
            # Phase B: ENABLED
            "center_loss_weight_init": 0.3,
            "center_loss_weight_min": 0.05,
            "center_loss_decay_epochs": 40,
            # Phase C: standard clipping
            "iou_clip_start": 50.0,
            "iou_clip_end": 20.0,
            "dfl_clip_start": 25.0,
            "dfl_clip_end": 10.0,
            # Phase D: TAL
            "tal_topk": 13,
            "tal_alpha": 0.5,
            "tal_beta": 4.0,
            # Other
            "iou_type": "DIoU",
            "cls": 1.0,
        },
    },
    {
        "name": "v5_cls12_tight_clip_phaseB",
        "desc": "Best test mAP50 (v5_cls12_tight_clip) + center loss",
        "params": {
            # Phase A
            "alpha_start": 0.7,
            "alpha_end": 0.3,
            "alpha_min": 0.2,
            "alpha_max": 0.8,
            "small_obj_px": 40,
            "small_obj_boost": 1.5,
            # Phase B: ENABLED
            "center_loss_weight_init": 0.3,
            "center_loss_weight_min": 0.05,
            "center_loss_decay_epochs": 40,
            # Phase C: TIGHT clipping
            "iou_clip_start": 30.0,
            "iou_clip_end": 10.0,
            "dfl_clip_start": 15.0,
            "dfl_clip_end": 5.0,
            # Phase D: TAL
            "tal_topk": 13,
            "tal_alpha": 0.5,
            "tal_beta": 4.0,
            # Other
            "iou_type": "DIoU",
            "cls": 1.2,
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
    params = exp["params"]

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"{'=' * 70}")
    print(f"  Phase A: alpha {params['alpha_start']}→{params['alpha_end']}, boost={params['small_obj_boost']}")
    print(f"  Phase B: center_loss={params['center_loss_weight_init']}→{params['center_loss_weight_min']} (decay {params['center_loss_decay_epochs']} ep)")
    print(f"  Phase C: iou_clip {params['iou_clip_start']}→{params['iou_clip_end']}, dfl_clip {params['dfl_clip_start']}→{params['dfl_clip_end']}")
    print(f"  Phase D: topk={params['tal_topk']}, beta={params['tal_beta']}")
    print(f"  cls={params['cls']}, iou_type={params['iou_type']}")
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
    print(f"# PHASE B TEST — Center Loss on 2 best configs")
    print(f"# center_loss: 0.3→0.05 over 40 epochs")
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
