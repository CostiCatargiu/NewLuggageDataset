#!/usr/bin/env python3
"""
TAL / loss-component PAIRWISE ablation on the FULL revised dataset.

Companion to run_full_tal_ablation.py (which isolates each component alone).
This script tests the two-way COMBINATIONS with TAL-tune, so the full ladder is:

    default                         (baseline, from the other script)
    + SWA+boost   only              (other script)
    + clip        only              (other script)
    + TAL-tune    only              (other script)
    + SWA+boost + TAL-tune          <-- THIS script, run 1
    + clip      + TAL-tune          <-- THIS script, run 2
    + best-TAL  (all together)      (run_full_*; the full combo)

With these you can see each part alone, each pair, and the full combo, and judge
whether the components are additive or overlap.

All on STOCK YOLOv12s (no architecture changes), full revised data, 90 ep, batch 48,
seed 0 (single seed, matching the component study so the deltas are comparable).

Terminology: "SWA" here = the assignment alpha schedule (alpha_start/end/min/max),
per the project's naming, NOT weight averaging.

IMPORTANT (dataset): DATA_YAML below MUST match whatever the rest of the ablation
used so the deltas are comparable. It defaults to the same split as the sibling
script. For the leakage-free study, set it to the regrouped split AND re-run the
sibling individual-component runs on the same split.
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION  (kept identical to run_full_tal_ablation.py for comparability)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # <-- match the sibling script
# For the leakage-free study instead use:
# DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/regrouped_split/data_regrouped.yaml"
PROJECT_DIR = "runs_noaug_weapon_full"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
BATCH = 48
EPOCHS = 90
SEED = 0

# ---- base vanilla loss (identical to the sibling script's LOSS_DEFAULT) ----
LOSS_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, use_vfl=False,
)

# individual component deltas (same definitions as the sibling script) --------
SWA_BOOST = dict(alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
                 small_obj_px=40, small_obj_boost=2.5)
CLIP      = dict(iou_clip_start=50.0, iou_clip_end=20.0,
                 dfl_clip_start=25.0, dfl_clip_end=10.0)
TALTUNE   = dict(cls=1.2, tal_topk=13, tal_alpha=0.7, tal_beta=4.0, iou_type="DIoU")

# ---- the two PAIRWISE combinations requested ----
LOSS_SWA_TAL  = dict(LOSS_DEFAULT, **SWA_BOOST, **TALTUNE)   # SWA + boost + TAL-tune
LOSS_CLIP_TAL = dict(LOSS_DEFAULT, **CLIP,      **TALTUNE)   # clip       + TAL-tune

RUNS = [
    {"name": "stock_talabl_swa_tal",  "loss": LOSS_SWA_TAL,  "desc": "[1/2] stock default + SWA + boost + TAL-tune"},
    {"name": "stock_talabl_clip_tal", "loss": LOSS_CLIP_TAL, "desc": "[2/2] stock default + clip + TAL-tune"},
]


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if getattr(trainer, "criterion", None) is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, "_sync_bbox_loss_state"):
                trainer.criterion._sync_bbox_loss_state()
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
    except Exception:
        pass


def run_experiment(run):
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n"
          f"# stock YOLOv12s · FULL data · Batch {BATCH} · Epochs {EPOCHS} · seed {SEED}\n{'#' * 80}\n")
    start = time.time()
    try:
        model = YOLO(PRETRAINED)   # stock arch + full pretrained transfer (no arch changes)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}  (expect 'Detect')")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=SEED, deterministic=True)
        kw.update(run["loss"])
        model.train(**kw)
        el = (time.time() - start) / 3600
        print(f"\n  DONE: {run['name']} ({el:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": el}
    except Exception as e:
        el = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({el:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time": el}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    t0 = time.time()
    print(f"\n{'=' * 80}\n  PAIRWISE TAL/LOSS ABLATION on STOCK YOLOv12s · FULL revised data")
    print(f"  data = {DATA_YAML}")
    print(f"  runs = SWA+boost+TAL-tune , clip+TAL-tune  (compare vs default and vs each part alone)")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<34} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
