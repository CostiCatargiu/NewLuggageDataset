#!/usr/bin/env python3
"""
70% Ablation — Round 17: SPATIALLY-ROUTED receptive-field fusion (the one
degree of freedom every prior fusion lacked), tested as a CONTROLLED ablation
against its own static-fusion twin.

Context / evidence base (cross-round synthesis, rounds 6-16):

  - baseline (original_loss_70) = 78.45% mAP50, mAP50_small=61.79% (highest of
    any variant), "other"-class AP50_small=38.57% (also highest).
  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4-BU) = 79.40% mAP50 (current
    best architecture), but mAP50_small=56.65%, "other" AP50_small=23.59% -- a
    ~15pp ABSOLUTE drop on small-"other" vs baseline.
  - The trade-off is INVARIANT across ~20 P4/head modules (LKA, strip, GC, star,
    multi-dil, P2/P3/P4 fusions, cls-branch context). Diagnosis sharpened over
    rounds 13-16: the loss is concentrated in the "other" class small objects.

  ROOT CAUSE (new): every gated fusion so far -- WideFuse (concat k11+strip23),
  WideFuse3 (additive k3, one gamma), CompactFuse (k3/k5), GCFuse (global ctx)
  -- combines its branches with a SINGLE GLOBAL mixing rule that is identical at
  every spatial location. So the small-RF branch fires on large objects (noise)
  and the large-RF branch fires on small objects (smoothing). The network can
  only learn ONE global small-vs-large compromise, and the compromise that
  maximises overall mAP is exactly the one that sacrifices small-"other".
  Searching WHICH branches to fuse cannot escape this; the binding constraint is
  that the fuse is STATIC.

This round, 2 runs forming a clean controlled ablation:

  1. r17_selectfuse_70 (~25-30%, the innovation) -- NEW module ZGLSKASelectFuse
     [512,11,23,3] @ P4-BU. SAME three branches as WideFuse3 (square ZGLKA k=11,
     strip LSKA k=23, small depthwise k=3) but combined by a lightweight spatial
     router -> per-LOCATION softmax over the 3 branches, instead of a static
     concat+projection. The receptive field thus adapts to local object scale:
     small objects route to the k=3 detail branch (preserving fine structure),
     large objects route to k=11/strip-23 context -- simultaneously, in
     different regions of the same P4 map. Falsifiable prediction: small-"other"
     AP50 recovers (router sends those pixels to k=3) WITHOUT giving back
     large-"other" (router keeps k11/strip23 elsewhere) -- the trade-off no
     static fusion could escape. gamma=0 at init -> exact r11-style identity;
     router bias warm-started toward the square-LKA branch.

  2. r17_widefuse3_70 (the CONTROL) -- ZGLSKAWideFuse3[512,11,23,3] @ P4-BU.
     IDENTICAL three branches and IDENTICAL YAML args as run 1; the ONLY
     difference is static concat+pw2 (this) vs spatial-softmax routing (run 1).
     This isolates "spatially-adaptive receptive field" as the mechanism. (This
     is the same architecture as the pending r15_widefuse3_70 -- run here in the
     SAME environment as selectfuse so the head-to-head is on identical
     conditions rather than across sessions.)

Reference bars (already trained): 78.45 baseline (mAP50_small=61.79) |
79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best, mAP50_small=56.65).

NOTE ON SEEDS: r11_widefuse_70 had the largest test-vs-val gap of rounds 11-14,
so a single-seed 79.5 from selectfuse would be inside the noise. If run 1 wins
run 2 here, re-run BOTH at seeds 1,2 before claiming the mechanism -- the point
of this round is the selectfuse-vs-widefuse3 DELTA, read against seed spread.

All: pure arch, default TAL, gated identity-at-init, append-only (standard
Detect-remap loader -- backbone/FPN layers 0-20 unchanged, only the Detect
index shifts 20 -> 21 -> 22).

Usage:
  python run_round17_selectfuse.py
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

# 1: THE INNOVATION — spatially-routed fusion of the same 3 branches as WideFuse3
ARCH_SELECTFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKASelectFuse, [512, 11, 23, 3]]  # 21 — per-location router over k11 LKA + strip23 + k3 detail @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: THE CONTROL — identical branches/args, static concat fusion (= r15_widefuse3)
ARCH_WIDEFUSE3 = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse3, [512, 11, 23, 3]]   # 21 — static 3-branch concat fuse (k11 LKA + strip23 + k3) @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r17_selectfuse_70",
     "desc": "[1/2] INNOVATION ZGLSKASelectFuse: per-location softmax router over k11 LKA + strip23 + k3 (same branches as WideFuse3, spatial routing instead of static concat) @ P4-BU -- ~25-30% confidence",
     "yaml_content": ARCH_SELECTFUSE, "batch": 60, "seed": 0},
    {"name": "r17_widefuse3_70",
     "desc": "[2/2] CONTROL ZGLSKAWideFuse3: IDENTICAL branches/args, STATIC concat fusion -- isolates spatial-routing as the mechanism (= r15_widefuse3, same env)",
     "yaml_content": ARCH_WIDEFUSE3, "batch": 56, "seed": 0},
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
    print(f"  70% ABLATION — ROUND 17: spatially-routed fusion vs static-fusion control")
    print(f"  Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<22} batch={run['batch']} seed={run['seed']}  {run['desc']}")

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
