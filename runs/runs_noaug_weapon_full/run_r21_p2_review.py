#!/usr/bin/env python3
"""
r21 + P2 (4-level P2–P5) on the REVISED dataset.

Adds a proper P2 (stride-4) detection level to the r21 backbone — built as a real
top-down FPN branch, not a bare extra head:
   P3-head (14) --upsample--> + backbone-P2 (2) --> A2C2f --> P2 head (stride 4)
then detects from {P2, P3, P4(widefuse), P5}. Keeps r21's widefuse @ P4 + aux.

Why this might do better than round-18's P2 (which raised recall but AP flat /
overall regressed): (a) the corrected labels removed the small-"other" annotation
noise, so a finer head is now graded fairly on small; (b) P2 is fused from the
FPN (context-rich) rather than a raw stride-4 tap; (c) best-TAL's small_obj_boost
+ assignment already help small, so the extra level has a cleaner target.

Two cells on the revised data (compare to rev_r21_tal = 0.8237 / 50-95 0.5223 / small 0.664):
  1. rev_r21p2_default : r21+P2 + DEFAULT TAL  (isolate the P2 architecture effect)
  2. rev_r21p2_tal     : r21+P2 + BEST TAL     (the candidate model)

NOTE: P2 (stride-4) ~doubles activation memory -> BATCH lowered to 32. That breaks
batch-parity with the batch-48 2x2 cells, so treat the comparison as indicative
until you rerun the finalists at a common batch. Smoke-test the build first
(4-level DetectAux is unusual): confirm it instantiates and strides are [4,8,16,32].
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # REVISED dataset (same as the 2x2)
PROJECT_DIR = "run_weapon_70_review"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 90
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 32          # P2 is memory-heavy; raise to 48 if it fits (note batch-parity caveat)
AUX_W = 0.5

BASE_0_20 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2  — backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4  — backbone P3
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6  — backbone P4
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8  — backbone P5

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head (stride 8)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up (stride 16)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head (stride 32)
"""

# r21 + P2: top-down P2 branch + widefuse @ P4 + 4-level DetectAux
ARCH_R21_P2 = BASE_0_20 + f"""  - [14, 1, nn.Upsample, [None, 2, "nearest"]]   # 21 — P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                     # 22 — fuse backbone P2 (layer 2)
  - [-1, 2, A2C2f, [128, False, -1]]              # 23 — P2 head (stride 4, NEW small level)
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]        # 24 — widefuse @ P4 (= r21)
  - [[23, 14, 24, 20], 1, DetectAux, [nc, {AUX_W}]]  # 25 — Detect P2,P3,P4(widefuse),P5 + aux
"""

TAL_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, use_vfl=False,
)
TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0, dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0, iou_type="DIoU", use_vfl=False,
)

RUNS = [
    {"name": "rev_r21p2_default", "loss": TAL_DEFAULT,    "desc": "[1/2] r21+P2 + DEFAULT TAL (arch effect of P2)"},
    {"name": "rev_r21p2_tal",     "loss": TAL_BEST_LOOSE, "desc": "[2/2] r21+P2 + BEST TAL (candidate model)"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Transfer backbone+neck (layers 0-20) via model.load; Detect/P2/aux train fresh."""
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
    print(f"  [detect-remap] {len(matched)}/{len(remapped)} Detect keys transferred (P2 level fresh)")
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
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(ARCH_R21_P2, yaml_path)
    print(f"\n{'#' * 70}\n# {run['name']}\n# {run['desc']}\n# Batch: {BATCH}  Epochs: {EPOCHS}\n{'#' * 70}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")
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
    os.makedirs(YAML_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 70}\n  r21 + P2 (4-level) on REVISED data — {DATA_YAML}\n{'=' * 70}")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 70}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<20} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
