#!/usr/bin/env python3
"""
THE HEADLINE RUN — best architecture x best training recipe (never combined).

  arch:   zg_p4_k11 (gated LSKA k=11 @ P4)  79.19% with default TAL (+0.74)
          (round-6 winner; beats baseline on BOTH mAP50 and mAP50-95)
  recipe: TAL topk15/beta3                  80.45% with stock arch  (+2.00)
  bar:    80.45% — anything clearly above means arch + recipe stack.

Usage:
  python run_zgp4_tal_combo.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
RUN_NAME = "zgp4k11_tal_70"

ARCH_ZG_P4 = """nc: 4
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
  - [17, 1, ZGLSKA, [512, 11]]              # 21 — gated LSKA @ P4, k=11 (round-6 winner)
  - [[14, 21, 20], 1, Detect, [nc]]         # 22
"""


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
    remapped = {pfx_dst + k[len(pfx_src):]: v
                for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} keys transferred on top")
    return model


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    yaml_path = os.path.join(YAML_DIR, f"{RUN_NAME}.yaml")
    with open(yaml_path, "w") as f:
        f.write(ARCH_ZG_P4)

    print(f"\n{'=' * 70}")
    print(f"  {RUN_NAME}: zg_p4 (LSKA k11) + TAL topk15/beta3 — bar: 80.45%")
    print(f"{'=' * 70}\n")

    start = time.time()
    model = YOLO(yaml_path)
    load_pretrained_with_detect_remap(model)

    model.train(
        data=DATA_YAML,
        epochs=80,
        imgsz=IMG_SIZE,
        batch=54,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=100,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        # --- BEST TAL recipe (v5_topk15_beta3_70 = 80.45%) ---
        tal_topk=15,
        tal_alpha=0.5,
        tal_beta=3.0,
        # --- DISABLE SWA ---
        alpha_start=0.0,
        alpha_end=0.0,
        alpha_min=0.0,
        alpha_max=0.0,
        # --- DISABLE clipping ---
        iou_clip_start=999.0,
        iou_clip_end=999.0,
        dfl_clip_start=999.0,
        dfl_clip_end=999.0,
        # --- DISABLE custom features ---
        small_obj_boost=1.0,
        small_obj_px=0,
        center_loss_weight_init=0.0,
        center_loss_weight_min=0.0,
        use_vfl=False,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n  DONE: {RUN_NAME} ({(time.time() - start) / 3600:.2f}h)")


if __name__ == "__main__":
    main()
