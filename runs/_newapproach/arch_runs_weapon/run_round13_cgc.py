#!/usr/bin/env python3
"""
70% Ablation — Round 13: HEAD-LEVEL CLS-BRANCH GATING.

Context / evidence base:
  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4 bottom-up) = 79.40% mAP50,
    current best pure architecture.
  - r6_zgp4_k11_70 (ZGLSKA[512,11] @ P4 bottom-up) = 79.19% mAP50, second best.
  - Per-class analysis: "other" is the dominant lever (AP50 ~50-55 vs
    ~86-90 for the other 3 classes). Every prior architecture change has
    applied its gated branch to the SHARED feature feeding both cv2 (box)
    and cv3 (cls) -- any classification gain on "other" has to compete with
    the box-regression objective on the same tensor, which likely caps it.
  - DetectCGC (pre-existing, registered, never used): isolates a change to
    the CLS branch only (cv2/box input stays raw) behind a zero-init gate --
    safe, low-risk, no box/cls tradeoff. But its mechanism is GLOBAL P5-pooled
    context, which is a poor match for small, visually-diverse "other"
    objects (P5 is the coarsest/worst map for small-object detail).

New module this round: DetectLKACls (head.py) -- combines the two validated
ideas. Per scale, applies the PROVEN k=11 ZGLKA receptive field (the round-7
dose-response peak, used in r6/r11_widefuse) ISOLATED to the cls-branch
input only, behind its own per-channel zero-init gate:
    cls_in_i = x_i + gamma_i * ZGLKA(k=11)(x_i)
    x_i = cat([cv2_i(x_i), cv3_i(cls_in_i)], 1)
gamma=0 at init -> exact stock Detect at epoch 0, full pretrained transfer.

Three runs:
  1. r13_widefuse_lkacls_70 (~35-40%) -- r11_widefuse_70 backbone (UNCHANGED,
     79.40 winner) + DetectLKACls[nc, 11] head.

  2. r13_r6_lkacls_70 (~35-40%) -- r6's k11@P4-BU backbone (UNCHANGED, 79.19)
     + DetectLKACls[nc, 11] head. SAME head mod as run 1, DIFFERENT backbone
     -- this is the CONSISTENCY check: if both runs show a similar bump over
     their respective backbone baselines, that's a genuinely consistent
     architectural lever (not a one-off interaction with widefuse).

  3. r13_widefuse_cgc_70 (~20-25%) -- r11_widefuse_70 backbone (UNCHANGED)
     + DetectCGC[nc] head (global P5 context -> cls branch). MECHANISM
     comparison vs run 1 (local k=11 LKA-on-cls vs global-context-on-cls)
     on the same backbone.

All three: pure arch (head-level only), default TAL, gated identity-at-init,
append-only (standard Detect-remap loader -- backbone/FPN layers 0-21
unchanged, only the Detect index shifts from 21 -> 22).

Bars: 78.45 baseline | 79.19 r6_zgp4_k11 | 79.40 r11_widefuse_70 (current best).

Usage:
  python run_round13_cgc.py
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

# 1: r11_widefuse_70 backbone UNCHANGED (= 79.40 winner) + DetectLKACls head
ARCH_WIDEFUSE_LKACLS = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]     # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner, unchanged)
  - [[14, 21, 20], 1, DetectLKACls, [nc, 11]]  # 22 — NEW: per-scale gated k11 LKA on cls branch only
"""

# 2: r6's k11@P4-BU backbone UNCHANGED (= 79.19) + DetectLKACls head
#    SAME head mod as run 1, DIFFERENT backbone -- consistency check.
ARCH_R6_LKACLS = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 11]]                 # 21 — gated k11 @ P4 bottom-up (= r6, unchanged)
  - [[14, 21, 20], 1, DetectLKACls, [nc, 11]]  # 22 — NEW: per-scale gated k11 LKA on cls branch only
"""

# 3: r11_widefuse_70 backbone UNCHANGED + DetectCGC head (global P5 context)
#    Mechanism comparison vs run 1 on the same backbone.
ARCH_WIDEFUSE_CGC = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner, unchanged)
  - [[14, 21, 20], 1, DetectCGC, [nc]]      # 22 — gated global-context (P5-pooled) cls branch
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r13_widefuse_lkacls_70",
     "desc": "[1/3] r11_widefuse_70 (79.40, unchanged) + DetectLKACls[k=11] head -- ~35-40% confidence",
     "yaml_content": ARCH_WIDEFUSE_LKACLS, "batch": 56},
    {"name": "r13_r6_lkacls_70",
     "desc": "[2/3] r6_zgp4_k11_70 (79.19, unchanged) + DetectLKACls[k=11] head -- consistency check vs run 1 -- ~35-40% confidence",
     "yaml_content": ARCH_R6_LKACLS, "batch": 56},
    {"name": "r13_widefuse_cgc_70",
     "desc": "[3/3] r11_widefuse_70 (79.40, unchanged) + DetectCGC head -- mechanism comparison vs run 1 -- ~20-25% confidence",
     "yaml_content": ARCH_WIDEFUSE_CGC, "batch": 56},
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
    print(f"  70% ABLATION — ROUND 13: HEAD-LEVEL CLS-BRANCH GATING")
    print(f"  Bars: 78.45 baseline | 79.19 r6_zgp4_k11 | 79.40 r11_widefuse_70 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={run['batch']}  {run['desc']}")

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
