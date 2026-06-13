#!/usr/bin/env python3
"""
70% Ablation — Round 8: NEW MECHANISM FAMILIES @ P4 (non-SKA).

EVERY ZG block tried so far (LSKA k7/k11/k15, GC, SE, MHSA, Strip, DCN) is
some variant of spatial attention / large-kernel context -- the SKA family.
Round 8 tests two mechanisms with NO spatial-attention map at all:

  1. r8_star_p4_70   — ZGStar: StarNet-style multiplicative feature mixing.
                       x + gamma * proj_out(act(proj1(z)) * proj2(z)),
                       z = BN(DWConv7x7(x)). Element-wise product of two 1x1
                       projections implicitly realizes a high-dimensional
                       polynomial feature expansion in low-dim space (Ma et
                       al., StarNet 2024) -- multiplicative interaction, not
                       additive attention. Cheap, proven family. HIGHER
                       CONFIDENCE (well-validated mechanism, simple/stable).

  2. r8_dsconv_p4_70 — ZGDSConv: Dynamic Snake Convolution (Qi et al. 2023).
                       Two 1D kernels (along x and along y) with CUMULATIVE
                       per-tap offsets that "snake" along elongated
                       structures -- originally for tubular vessel
                       segmentation. long_gun/knife are intrinsically
                       elongated/thin: directly encodes a shape prior no
                       prior experiment has tested. Implemented via
                       F.grid_sample (pure PyTorch, sidesteps the
                       torchvision deform_conv2d crash from ZGDCN). LOWER
                       CONFIDENCE (novel custom op, heavier compute -> k=7,
                       smaller batch).

Both: pure arch, default TAL, gated identity-at-init.
Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best architecture).

Usage:
  python run_round8_new_mechanisms.py
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21

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

# 1: StarNet-style multiplicative mixing @ P4
ARCH_STAR = BASE_0_20 + """
  - [17, 1, ZGStar, [512, 4]]               # 21 — gated star-mixing @ P4 (hidden=4x)
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: Dynamic Snake Convolution @ P4 (elongated-shape prior)
ARCH_DSCONV = BASE_0_20 + """
  - [17, 1, ZGDSConv, [512, 7]]             # 21 — gated dynamic snake conv @ P4, k=7
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r8_star_p4_70",
     "desc": "[1/2] ZGStar: multiplicative feature mixing (StarNet) @ P4 — non-attention",
     "yaml_content": ARCH_STAR, "batch": 54},
    {"name": "r8_dsconv_p4_70",
     "desc": "[2/2] ZGDSConv: dynamic snake conv @ P4 — elongated-shape prior",
     "yaml_content": ARCH_DSCONV, "batch": 40},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted."""
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
          f"{len(matched)}/{len(remapped)} Detect keys transferred on top")
    return model


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: DEFAULT   Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=80,
            imgsz=IMG_SIZE,
            batch=run["batch"],
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            # --- DEFAULT TAL (pure architecture effect) ---
            **TAL_DEFAULT,
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
    print(f"  70% ABLATION — ROUND 8: NEW MECHANISM FAMILIES @ P4 (non-SKA)")
    print(f"  Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<16} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<16} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
