#!/usr/bin/env python3
"""
70% Ablation — Round 10: THREE VARIATIONS ON THE k11 WINNER.

r6_zgp4_k11_702 (ZGLSKA k=11 @ P4 bottom-up, default TAL) = 79.19% mAP50,
the best pure-architecture result so far. Round 9's spatial-gate idea was
judged low-confidence (LKA already provides spatial selectivity internally).
Round 10 instead tries three DIFFERENT modifications, ranked by confidence
of beating 79.19:

  1. r10_k11_p4td_70    (~40%) — RELOCATE the *same validated* ZGLSKA(k=11)
                          branch from P4 bottom-up (layer 17, post-fusion)
                          to P4 top-down (layer 11, pre-fusion). Same module,
                          same k, zero new failure modes -- only the
                          injection point changes. Layer 11's output also
                          feeds the P3 head, so this could help small
                          objects too. NOTE: inserting mid-network shifts
                          every subsequent layer index by +1, so this run
                          uses a GENERALIZED shift-remap loader (see below)
                          instead of the usual Detect-only remap.

  2. r10_lskastripfuse_70 (~30%) — FUSE k11 (square, dilated, 79.19%) and
                          strip23 (1x23+23x1, 79.07%) -- the two best
                          single-branch shapes -- inside ONE gated branch
                          via channel-split (half the channels through each
                          shape, concat, 1x1 fuse, ONE gamma). Round 7
                          showed stacking TWO SEPARATE gates hurts
                          (k11+GC@P4=78.66); this is a different failure
                          mode (one gate, two RF shapes sharing capacity).

  3. r10_lskamultidil_70 (~25%) — Single-branch MULTI-SCALE LKA: parallel
                          k7/dilation2 (RF~17) and k11/dilation3 (RF~35,
                          the dose-response peak) dilated depthwise convs,
                          summed before the final pointwise, ONE gamma.
                          A single-branch "ensemble" of the two best points
                          on the k7/k11/k15 dose-response curve instead of
                          picking one.

All three: pure arch, default TAL, gated identity-at-init, P4 placement.
Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best architecture).

Usage:
  python run_round10_k11_variants.py
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

# 1: k11 RELOCATED to P4 top-down (layer 11 -> insert as new layer 12).
# Every layer from the old "12" onward shifts to "+1" (13..21), and the
# P4-bottom-up concat (old "[[-1, 11], ...]") now points at the GATED
# layer 12 instead of the raw layer 11. At init gamma=0 so layer 12's
# output == layer 11's output exactly -> the whole net is identity-at-init
# and architecturally equivalent to the baseline (just re-indexed).
ARCH_K11_P4TD = """nc: 4
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
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]       # 9
  - [[-1, 6], 1, Concat, [1]]                         # 10
  - [-1, 2, A2C2f, [512, False, -1]]                  # 11 — P4 top-down
  - [-1, 1, ZGLSKA, [512, 11]]                        # 12 — gated k11 @ P4 top-down (relocated)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 13
  - [[-1, 4], 1, Concat, [1]]                         # 14
  - [-1, 2, A2C2f, [256, False, -1]]                  # 15 — P3 head
  - [-1, 1, Conv, [256, 3, 2]]                        # 16
  - [[-1, 12], 1, Concat, [1]]                        # 17 — concat with GATED layer 12 (was 11)
  - [-1, 2, A2C2f, [512, False, -1]]                  # 18 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]                        # 19
  - [[-1, 8], 1, Concat, [1]]                         # 20
  - [-1, 2, C3k2, [1024, True]]                       # 21 — P5 head
  - [[15, 18, 21], 1, Detect, [nc]]                   # 22 — Detect (P3, P4, P5)
"""

# 2: fuse k11 (square) + strip23 in one branch, channel-split, @ P4
ARCH_LSKASTRIPFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAStripFuse, [512, 11, 23]]  # 21 — gated k11+strip23 fusion @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 3: single-branch multi-scale LKA (k7/dil2 + k11/dil3), @ P4
ARCH_LSKAMULTIDIL = BASE_0_20 + """
  - [17, 1, ZGLSKAMultiDil, [512, 7, 2, 11, 3]]  # 21 — gated multi-dilation LKA @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r10_k11_p4td_70",
     "desc": "[1/3] k11 relocated to P4 top-down (layer 11, pre-fusion) -- ~40% confidence",
     "yaml_content": ARCH_K11_P4TD, "batch": 54, "loader": "shift"},
    {"name": "r10_lskastripfuse_70",
     "desc": "[2/3] ZGLSKAStripFuse: k11+strip23 channel-split fusion @ P4 -- ~30% confidence",
     "yaml_content": ARCH_LSKASTRIPFUSE, "batch": 48, "loader": "detect"},
    {"name": "r10_lskamultidil_70",
     "desc": "[3/3] ZGLSKAMultiDil: k7/dil2 + k11/dil3 fused @ P4 -- ~25% confidence",
     "yaml_content": ARCH_LSKAMULTIDIL, "batch": 48, "loader": "detect"},
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


def load_pretrained_with_shift_remap(model, weights=PRETRAINED, shift_from=12, shift_by=1):
    """model.load() first (sets model.ckpt, transfers all UNSHIFTED matching
    layers -- here layers 0..shift_from-1, which are identical to the
    pretrained YAML), then remap every checkpoint key 'model.{i}.*' with
    i >= shift_from to 'model.{i+shift_by}.*' and load on top.

    Use for architectures that INSERT a new layer in the middle of the
    YAML (everything from that point on is re-indexed by +shift_by, but
    is otherwise architecturally identical to the pretrained net)."""
    model.load(weights)
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    remapped = {}
    for k, v in csd.items():
        if not k.startswith("model."):
            continue
        parts = k.split(".")
        idx = int(parts[1])
        if idx >= shift_from:
            parts[1] = str(idx + shift_by)
            remapped[".".join(parts)] = v
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [shift-remap] model.{shift_from}+ -> model.{shift_from + shift_by}+: "
          f"{len(matched)}/{len(remapped)} keys transferred on top")
    return model


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Loader: {run['loader']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        if run["loader"] == "shift":
            load_pretrained_with_shift_remap(model)
        else:
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
    print(f"  70% ABLATION — ROUND 10: THREE VARIATIONS ON THE k11 WINNER")
    print(f"  Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<20} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<20} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
