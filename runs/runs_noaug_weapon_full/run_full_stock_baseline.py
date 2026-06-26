#!/usr/bin/env python3
"""
FULL revised dataset -- the two MISSING baseline cells (stock YOLOv12s).

Completes the clean full-dataset 2x2 (stock vs globalctx) x (default vs best TAL).
You already ran the globalctx half:
  globalctx_full_default  84.51 / 55.54 / small 67.73   (arch, vanilla loss)
  globalctx_full_besttal  85.18 / 55.34 / small 70.84   (arch + best TAL)
This adds the stock half so the comparison is on IDENTICAL data/config (no
70%-vs-full or epoch confound):
  Cell 1 (stock_full_default) : stock YOLOv12s + VANILLA default loss  = TRUE baseline
  Cell 2 (stock_full_besttal) : stock YOLOv12s + BEST TAL              = loss-only effect

Stock = plain yolov12s.pt (standard Detect head, full pretrained transfer, no remap).
Same FULL revised data / 90 ep / batch 48 / seed 0 as the globalctx runs.

After this you get the full attribution, all same-config:
  arch effect    = globalctx_full_default vs stock_full_default
  loss effect    = stock_full_besttal vs stock_full_default
  combined       = globalctx_full_besttal vs stock_full_default  (headline)
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION (identical to run_full_globalctx_final.py)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # FULL revised dataset
PROJECT_DIR = "runs_noaug_weapon_full"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"     # stock arch + full pretrained weights
BATCH = 48
EPOCHS = 90

# TRUE vanilla default -- matches globalctx_full_default's loss exactly
DEFAULT_TAL = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    use_vfl=False,
)

# Best-TAL recipe -- matches globalctx_full_besttal's loss exactly
TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0,
    dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0,
    iou_type="DIoU", use_vfl=False,
)

RUNS = [
    {"name": "stock_full_default", "loss": DEFAULT_TAL,
     "desc": "[1/2] stock YOLOv12s + VANILLA default loss  = TRUE baseline"},
    {"name": "stock_full_besttal", "loss": TAL_BEST_LOOSE,
     "desc": "[2/2] stock YOLOv12s + BEST TAL              = loss-only effect"},
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
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# FULL revised data  Batch {BATCH}  Epochs {EPOCHS}  seed 0\n{'#' * 80}\n")
    start = time.time()
    try:
        model = YOLO(PRETRAINED)   # stock arch + full pretrained transfer (no remap)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, strides = {model.model.stride.tolist()}")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=0, deterministic=True)
        kw.update(run["loss"])
        model.train(**kw)
        elapsed = (time.time() - start) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    t0 = time.time()
    print(f"\n{'=' * 80}\n  FULL revised dataset -- stock YOLOv12s x {{default, best TAL}}")
    print(f"  completes the 2x2 vs globalctx_full_{{default,besttal}}")
    print(f"  data: {DATA_YAML}")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<22} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
