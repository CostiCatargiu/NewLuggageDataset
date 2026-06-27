#!/usr/bin/env python3
"""
Cumulative TAL / loss-component ablation on the FULL revised dataset.

On the STOCK YOLOv12s baseline (no architecture changes), each run adds ONE loss
component on top of the vanilla default loss, INDIVIDUALLY (not cumulative), so each
component's standalone effect is isolated. Three runs:

  1. swa_boost  : default + alpha schedule ("SWA-alpha") + small-object boost   (only)
  2. clip       : default + IoU / DFL clip schedules                            (only)
  3. tal_tune   : default + tuned TAL assignment (topk/alpha/beta) + cls + DIoU (only)

Reference: stock + vanilla default loss is the baseline each run is compared against
(the component delta = run - default). The full best-TAL run (all components together)
is available separately, so default + the 3 isolated parts + the full combo lets you
see both each part alone AND whether they add up.

Terminology note: "SWA" here means the assignment alpha schedule
(alpha_start/end/min/max), per the project's naming — not weight averaging.

All on full revised data, stock YOLOv12s, 90 ep, batch 48, seed 0 (single seed for
the component study).
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # FULL revised dataset
PROJECT_DIR = "runs_noaug_weapon_full"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48
EPOCHS = 90
AUX_W = 0.5
SEED = 0

ARCH_GLOBALCTX = f"""nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23
  - [22, 1, ZGGlobalContext, [256]]                  # 24
  - [21, 1, ZGGlobalContext, [512]]                  # 25
  - [23, 1, ZGGlobalContext, [1024]]                 # 26
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

# ---- INDIVIDUAL loss components: each = stock vanilla default + ONE component ----
# (NOT cumulative — every run isolates a single component on top of the default loss)
LOSS_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, use_vfl=False,
)
# default + ONLY alpha schedule ("SWA-alpha") + small-object boost
LOSS_SWA_BOOST = dict(LOSS_DEFAULT,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5)
# default + ONLY the IoU/DFL clip schedules
LOSS_CLIP = dict(LOSS_DEFAULT,
    iou_clip_start=50.0, iou_clip_end=20.0, dfl_clip_start=25.0, dfl_clip_end=10.0)
# default + ONLY the tuned TAL assignment (topk/alpha/beta) + cls weight + DIoU
LOSS_TALTUNE = dict(LOSS_DEFAULT,
    cls=1.2, tal_topk=13, tal_alpha=0.7, tal_beta=4.0, iou_type="DIoU")

RUNS = [
    {"name": "stock_talabl_swaboost", "loss": LOSS_SWA_BOOST, "desc": "[1/3] stock default + SWA + boost ONLY"},
    {"name": "stock_talabl_clip",     "loss": LOSS_CLIP,      "desc": "[2/3] stock default + clip ONLY"},
    {"name": "stock_talabl_taltune",  "loss": LOSS_TALTUNE,   "desc": "[3/3] stock default + TAL-tune ONLY"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: {len(matched)}/{len(remapped)} keys")
    return model


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
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# stock YOLOv12s · FULL data · Batch {BATCH} · Epochs {EPOCHS} · seed {SEED}\n{'#' * 80}\n")
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
    os.makedirs(YAML_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 80}\n  ISOLATED TAL/LOSS COMPONENT ABLATION on STOCK YOLOv12s · FULL revised data")
    print(f"  each run = stock default + ONE component (swa+boost | clip | tal-tune), not cumulative")
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
