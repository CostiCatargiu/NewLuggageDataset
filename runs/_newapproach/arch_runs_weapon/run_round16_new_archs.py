#!/usr/bin/env python3
"""
70% Ablation — Round 16: two NEW, independent architecture ideas targeting
the "other"-class small-object AP50 gap, each tested in isolation.

Context / evidence base (cross-round synthesis, rounds 6-15):

  - baseline (original_loss_70) = 78.45% mAP50, mAP50_small=61.79% (highest
    of any variant), "other"-class AP50_small=38.57% (also highest).
  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4-BU, layer 17->21) =
    79.40% mAP50 (current best), but mAP50_small=56.65%, "other" AP50_small
    =23.59% -- a ~15pp ABSOLUTE drop on "other"-small vs baseline.
  - Per-class breakdown (round-15 analysis) showed this small-object
    degradation is overwhelmingly concentrated in the "other" class
    (-14.98pp AP50_small) vs the weapon classes (-1.47 to -2.34pp). This
    sharpened diagnosis from "general small-object trade-off" to
    "other-class cls-discriminability trade-off".
  - Round 13's DetectLKACls (k=11 ZGLKA isolated to the cls branch only)
    FAILED on two backbones (-0.38, -0.86 mAP50): a k=11 receptive field is
    too coarse/smoothing for the per-anchor cls decision on small,
    ambiguous "other" objects.
  - Round 15's ZGLSKAWideFuse3 (additive 3rd small-RF branch inside
    WideFuse, strict superset, gamma=0 at init) addresses the SHARED-feature
    side of the problem.

This round tests TWO further, independent ideas -- NOT additive supersets,
so each carries real downside risk (could regress below 79.40), but each
targets the diagnosis from a different angle:

  1. r16_widefuse_smallcls_70 (~30-35%, highest-ranked) -- NEW DetectSmallCls
     head. Keeps WideFuse@P4-BU EXACTLY as in r11 (unchanged, proven 79.40
     backbone), and adds a per-scale, zero-gated SMALL-kernel (k=3,
     dilation=1, depthwise+GroupNorm+SiLU) refinement on the CLS branch
     ONLY -- box branch (cv2) untouched. This is the OPPOSITE regime from
     DetectLKACls's failed k=11 (large-RF, smoothing) approach: pure local
     fine detail for the per-anchor classifier. Directly targets the
     "other"-class cls-discriminability loss without re-touching the shared
     P4 features that drive r11's overall mAP50 gain. gamma=0 at init ->
     exact r11_widefuse_70 at epoch 0; risk is the new branch hurts cls
     during training (regression below 79.40), not a structural break.

  2. r16_compactfuse_70 (~20-25%) -- NEW ZGLSKACompactFuse @ P4-BU (SAME slot
     as r11_widefuse_70, REPLACES it rather than extending it). Same
     2-branch WideFuse shape (pw1: c1->2c1, pw2: 2c1->c1, one gamma) and
     keeps the proven k=11 ZGLKA branch, but REPLACES the strip-23 LSKA
     branch with a compact multi-scale SMALL-kernel branch (k=3/dilation=1 +
     k=5/dilation=2 depthwise convs, summed, GroupNorm+SiLU). Tests whether
     the strip-23 branch itself (not just "a second large-RF branch") is the
     binding constraint on "other"-small. Higher risk: this is a DIFFERENT
     2-branch architecture, not a strict superset of r11_widefuse_70 -- could
     plausibly land anywhere relative to 79.40.

Both: pure arch, default TAL, gated identity-at-init, append-only (standard
Detect-remap loader -- backbone/FPN layers 0-20 unchanged, only the Detect
index shifts).

Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best)
      | 79.40 r15_widefuse3_70 pending (round 15, additive superset of r11).

Usage:
  python run_round16_new_archs.py
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
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

# 1: r11_widefuse_70 backbone (UNCHANGED) + NEW DetectSmallCls head
#    (zero-gated k3 small-detail refinement on the cls branch only)
ARCH_WIDEFUSE_SMALLCLS = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — proven WideFuse @ P4 (= r11_widefuse_70, unchanged)
  - [[14, 21, 20], 1, DetectSmallCls, [nc, 3]]  # NEW: zero-gated k3 small-kernel cls-branch refinement
"""

# 2: NEW ZGLSKACompactFuse @ P4-BU (REPLACES WideFuse's strip-23 branch
#    with a compact k3/dilation1 + k5/dilation2 multi-scale branch)
ARCH_COMPACTFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKACompactFuse, [512, 11, 3, 1, 5, 2]]  # 21 — NEW: k11 LKA + compact k3/k5 multiscale fuse @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r16_widefuse_smallcls_70",
     "desc": "[1/2] r11_widefuse_70 backbone UNCHANGED + NEW DetectSmallCls head: zero-gated k3 small-detail refinement on cls branch only (opposite regime from round-13's failed k11 DetectLKACls) -- ~30-35% confidence",
     "yaml_content": ARCH_WIDEFUSE_SMALLCLS, "batch": 56, "seed": 0},
    {"name": "r16_compactfuse_70",
     "desc": "[2/2] NEW ZGLSKACompactFuse @ P4-BU: k11 ZGLKA + compact k3/dilation1+k5/dilation2 multiscale branch, REPLACES WideFuse's strip-23 branch -- tests whether strip-23 itself is the binding constraint on 'other'-small -- ~20-25% confidence",
     "yaml_content": ARCH_COMPACTFUSE, "batch": 56, "seed": 0},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted.

    Use for architectures that APPEND new layer(s) after the original
    layer 20 (layers 0-20 unchanged, only Detect's index shifts)."""
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
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Seed: {run['seed']}")
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
            seed=run["seed"],
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
    print(f"  70% ABLATION — ROUND 16: 2 new independent arch ideas")
    print(f"  Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={run['batch']} seed={run['seed']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<24} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
