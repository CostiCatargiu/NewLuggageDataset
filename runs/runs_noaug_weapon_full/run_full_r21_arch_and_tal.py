#!/usr/bin/env python3
"""
FINAL CHECK on the REVISED (corrected-label) full dataset — a clean 2x2 factorial:

        |  default TAL            |  best TAL (v5_tal07_loose)
  ------+-------------------------+----------------------------------
  stock |  [2] stock + default    |  [3] stock + best TAL  (loss only)
  r21   |  [1] r21 + default      |  [4] r21 + best TAL    (combined)

  [2] stock + default TAL      = the baseline (architecture original, default loss)
  [1] r21  + default TAL       = pure architecture effect
  [3] stock + best TAL         = pure loss effect
  [4] r21  + best TAL          = combined

All four on the SAME corrected dataset, SAME fixed batch / seed / epochs, so the
four cells are directly comparable (no batch confound). Evaluate all four on the
corrected test for the final table.

  stock arch = plain yolov12s.pt (Detect head; honors TAL args directly).
  r21 arch   = ZGLSKAWideFuse[512,11,23] @ P4 + DetectAux@0.5 (detect-remap loader).
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # REVISED/corrected full dataset
PROJECT_DIR = "runs_noaug_weapon_full"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 90
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48          # fixed across ALL four cells for a clean comparison
AUX_W = 0.5

# -----------------------------------------------------------------------------
# Shared YOLOv12s backbone + neck (layers 0-20). This is the ORIGINAL yolov12s
# structure; the stock cell appends a plain Detect, r21 appends widefuse + aux.
# Both archs therefore share these 21 layers VERBATIM -> the only difference
# between stock and r21 is the widefuse+aux. (BASE_0_20 reproduces yolov12s: the
# pretrained yolov12s.pt has Detect at index 21, which is why DETECT_SRC_IDX=21.)
# -----------------------------------------------------------------------------
BASE_0_20 = """nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head
"""

# ORIGINAL arch: stock yolov12s = BASE_0_20 + plain Detect at index 21 (= the
# detection structure baked into yolov12s.pt; full pretrained transfer, no remap).
ARCH_STOCK = BASE_0_20 + "  - [[14, 17, 20], 1, Detect, [nc]]              # 21 — stock Detect\n"

# r21 arch: same BASE_0_20 + gated wide-fuse @ P4 + train-only aux.
ARCH_R21 = BASE_0_20 + f"""  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]       # 21 — gated wide-fuse @ P4
  - [[14, 21, 20], 1, DetectAux, [nc, {AUX_W}]]  # 22 — train-only aux
"""

# -----------------------------------------------------------------------------
# Loss recipes
# -----------------------------------------------------------------------------
TAL_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    use_vfl=False,
)

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

# -----------------------------------------------------------------------------
# The four cells
# -----------------------------------------------------------------------------
RUNS = [
    {"name": "rev_r21_arch_default",   "arch": "r21",   "loss": TAL_DEFAULT,
     "desc": "[1/4] r21 best arch + DEFAULT TAL"},
    {"name": "rev_stock_default",      "arch": "stock", "loss": TAL_DEFAULT,
     "desc": "[2/4] stock arch (original) + DEFAULT TAL  = baseline"},
    {"name": "rev_stock_tal",          "arch": "stock", "loss": TAL_BEST_LOOSE,
     "desc": "[3/4] stock arch + BEST TAL  = loss only"},
    {"name": "rev_r21_tal",            "arch": "r21",   "loss": TAL_BEST_LOOSE,
     "desc": "[4/4] r21 best arch + BEST TAL  = combined"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """For r21: transfer backbone+neck+box head; aux towers train fresh."""
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
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} keys transferred")
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


def build_model(arch, name):
    if arch == "r21":
        yaml_path = os.path.join(YAML_DIR, f"{name}.yaml")
        save_yaml(ARCH_R21, yaml_path)
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
    else:  # stock yolov12s
        model = YOLO(PRETRAINED)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    return model


def run_experiment(run):
    print(f"\n{'#' * 70}")
    print(f"# {run['name']}  ({run['arch']})")
    print(f"# {run['desc']}")
    print(f"# Batch: {BATCH}   Epochs: {EPOCHS}   Seed: 0")
    print(f"{'#' * 70}\n")

    start_time = time.time()
    try:
        model = build_model(run["arch"], run["name"])
        train_kwargs = dict(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=run["name"],
            patience=100, close_mosaic=10, seed=0, deterministic=True,
        )
        train_kwargs.update(run["loss"])
        model.train(**train_kwargs)

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()
    print(f"\n{'=' * 70}")
    print(f"  FINAL 2x2 on REVISED dataset: {{stock, r21}} x {{default, best TAL}}")
    print(f"  data: {DATA_YAML}")
    print(f"{'=' * 70}")
    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<22} {run['desc']}")
    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(RUNS):
        print(f"\n>>> Run {i+1}/{len(RUNS)}: {run['name']}")
        results.append(run_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<22} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
