#!/usr/bin/env python3
"""
70% Ablation — Round 15: WideFuse3 (3-branch fuse) + r11 seed-check.

Context / evidence base — cross-round synthesis of ALL 17 prior variants
(rounds 6-14 + arch_zg_p4/p45), not just round 14:

  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4-BU, layer 17) = 79.40%
    mAP50, current best architecture. mAP50_small=56.65%, "other" AP50_all=
    54.10%, "other" AP50_small=23.59%.
  - baseline (original_loss_70) = 78.45% mAP50, mAP50_small=61.79% (the
    HIGHEST of any variant ever tried), "other" AP50_small=38.57% (also the
    highest).
  - Across ALL 17 architecture variants (rounds 6-14, arch_zg_p4/p45),
    mAP50_small never came within 2.15pp of baseline's 61.79% (best
    alternative: r7_k11_gc4_70 at 59.64%, itself a weak overall performer at
    78.66%). "other"-class AP50_small fell as low as 23.48-23.59% in the
    variants closest to r11's structure (r11_refine=23.48, r11_widefuse=
    23.59) -- roughly a 15pp ABSOLUTE / ~40% RELATIVE drop vs baseline.
  - This pattern is INDEPENDENT of where the added module sits: P4-BU
    (layer 17, the whole r6/r7/r11 family), P4-TD (layer 11, r10_k11_p4td),
    P2->P3 detail injection (r14_widefuse_p2fuse), dual P3+P4 (r11_dual_p3p4,
    r12_dualwidefuse) -- ALL reproduce the same trade-off. None of the
    "add a small-object module elsewhere" attempts (round 12, round 14
    p2fuse) recovered it; several made it worse.

  DIAGNOSIS: ZGLSKAWideFuse's two branches (k=11 square ZGLKA + strip-23
  LSKA) are BOTH large-receptive-field operators. Neither preserves fine,
  small-scale detail at P4-BU -- exactly the capacity "other"-small needs.
  Every variant tried either reshapes/relocates large-RF capacity or adds a
  SEPARATE small-object branch elsewhere (competing gate, different failure
  mode) -- none added a small-RF branch INSIDE the proven WideFuse fusion
  itself.

This round, 2 runs:

  1. r15_widefuse3_70 (~30-35%) -- NEW module ZGLSKAWideFuse3[512,11,23,3]
     @ P4-BU (SAME slot as r11_widefuse_70). Adds a THIRD, genuinely
     small-receptive-field branch (k=3 depthwise + GroupNorm + SiLU, pure
     fine-detail pass) in PARALLEL with WideFuse's two proven branches,
     under ONE gamma (pw1 expands c1->3c1 so each branch gets a full
     c1-width stream, avoiding StripFuse's channel-starvation failure mode).
     This is a STRICT GENERALIZATION of r11_widefuse_70: gamma=0 at init is
     identical to r11's checkpoint, and the small-RF branch's contribution
     can shrink toward ~0 during training if unhelpful, collapsing back to
     ~WideFuse behavior. Downside risk is "ties 79.40", not "regresses below
     it". If "other"-small AP50 recovers materially (>30%) without giving
     back the overall mAP50 gain, this directly confirms the
     missing-fine-detail-branch diagnosis.

  2. r11_widefuse_70_seed1 (~N/A, variance probe) -- EXACT r11_widefuse_70
     architecture (ZGLSKAWideFuse[512,11,23] @ P4-BU), re-run with seed=1
     instead of seed=0. Training-curve analysis (prior session) showed
     r11_widefuse_70 has the LARGEST test-vs-val generalization gap
     (+1.30pp) of any round-11-14 run, concentrated in "other" -- suggesting
     r11's 79.40 test-set lead may be partly seed/variance-driven. This run
     gives a second sample at the SAME architecture, letting us read
     r15_widefuse3_70's result against the spread of r11 itself rather than
     a single point estimate.

All: pure arch, default TAL, gated identity-at-init, append-only (standard
Detect-remap loader -- backbone/FPN layers 0-20 unchanged, only the Detect
index shifts).

Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best).

Usage:
  python run_round15_widefuse3.py
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

# 1: WideFuse + a NEW small-RF (k=3) fine-detail branch, 3-way fuse, ONE gamma
ARCH_WIDEFUSE3 = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse3, [512, 11, 23, 3]]  # 21 — NEW: 3-branch gated fuse (k11 LKA + strip23 + k3 detail) @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: r11_widefuse_70 EXACT architecture, re-run with a different seed (variance probe)
ARCH_WIDEFUSE_R11 = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11_widefuse_70, unchanged)
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r15_widefuse3_70",
     "desc": "[1/2] NEW ZGLSKAWideFuse3: WideFuse (k11 LKA + strip23) + NEW k3 small-RF detail branch, 3-way fuse, ONE gamma @ P4-BU -- strict generalization of r11_widefuse_70 -- ~30-35% confidence",
     "yaml_content": ARCH_WIDEFUSE3, "batch": 36, "seed": 0},
    {"name": "r11_widefuse_70_seed1",
     "desc": "[2/2] r11_widefuse_70 EXACT architecture (ZGLSKAWideFuse[512,11,23]@P4-BU), seed=1 -- variance probe (r11 had the largest test-vs-val gap of rounds 11-14, concentrated in 'other')",
     "yaml_content": ARCH_WIDEFUSE_R11, "batch": 38, "seed": 1},
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
    print(f"  70% ABLATION — ROUND 15: WideFuse3 + r11 seed-check (2 runs)")
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
