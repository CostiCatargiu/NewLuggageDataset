#!/usr/bin/env python3
"""
70% Ablation — Round 19: NECK TOPOLOGY (the last untouched frontier at
yolov12s/640). Learnable weighted multi-scale fusion (BiFPN), not another
single-module insert.

WHY THIS ROUND
--------------
Rounds 1-18 (~35 variants) all varied a single block INSIDE a fixed-topology
PAN, on yolov12s @ 640. Every structural lever inside that box -- receptive
field, spatial routing, classifier capacity, detection scale -- was negative
vs loss tuning, and input resolution was separately found not to help. The
diagnosis (per-class): "other" has high recall (~0.84) but low precision/AP
(~0.51) -> the features feeding the head are not discriminative enough to
score "other" confidently. Resolution failing means the detail IS present at
640; the limit is FEATURE FUSION, not visibility or head capacity.

The one thing never changed: the NECK TOPOLOGY itself. The stock neck fuses
scales with hard Concat -- every branch contributes equally and
unconditionally. EfficientDet's BiFPN makes the fusion weights LEARNABLE
(per-branch, non-negative) and adds an extra cross-scale edge so each node
fuses more context. This round tests that, staying at yolov12s/640.

New module: WeightedConcat (block.py) -- out = cat([relu(w_i) * x_i]); at init
all w_i=1 -> EXACTLY standard Concat, so the pretrained neck transfers
unchanged and the model only *learns* to reweight branches (identity-at-init,
same discipline as the ZG family).

Three runs:
  1. r19_bifpn_70          BiFPN-lite neck: all neck Concats -> WeightedConcat,
                           plus the BiFPN extra edge (3-way fusion at P4-BU:
                           P3-down + P4-td + backbone-P4). Learnable fusion +
                           richer P4 context, same s backbone, same channels.
  2. r19_bifpn_widefuse_70 BiFPN neck + r11_widefuse @ P4-BU -- does the proven
                           best module stack on the new fusion topology?
  3. r19_doubleneck_70     CONTROL: iterate the STOCK PAN twice (plain Concat,
                           no learnable weights). Isolates "more fusion depth"
                           from "learnable weighting" -- if bifpn > doubleneck,
                           the gain is the weighting, not just extra capacity.

Honest expectation: after ruling out RF/routing/head-capacity/scale/resolution,
the residual signal still points at ranking/labels, so probability any neck
redesign cracks the "other" ceiling is modest -- BiFPN is the best-motivated
remaining structural bet, and a clean ablation either way. READ on per-class
"other" AP50 (all + small), not mAP50_all. Confirm any winner at seeds 1,2:
split variance here is ~+/-1pp, as large as the whole arch signal.

Bars: 78.45 baseline | 79.40 r11_widefuse (best arch) | 80.45 v5_topk15_beta3
(best overall, loss tuning only).

Usage:
  python run_round19_bifpn.py
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

BACKBONE = """nc: 4
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
"""

# 1: BiFPN-lite -- learnable weighted fusion everywhere + extra 3-way edge @ P4-BU
ARCH_BIFPN = BACKBONE + """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 9
  - [[-1, 6], 1, WeightedConcat, [1, 2]]            # 10 — weighted cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 11 — P4 top-down
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 12
  - [[-1, 4], 1, WeightedConcat, [1, 2]]            # 13 — weighted cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]                # 14 — P3 head (detect)
  - [-1, 1, Conv, [256, 3, 2]]                      # 15
  - [[-1, 11, 6], 1, WeightedConcat, [1, 3]]        # 16 — BiFPN 3-way: P3-down + P4-td + backbone-P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 17 — P4 bottom-up (detect)
  - [-1, 1, Conv, [512, 3, 2]]                      # 18
  - [[-1, 8], 1, WeightedConcat, [1, 2]]            # 19 — weighted cat backbone P5
  - [-1, 2, C3k2, [1024, True]]                     # 20 — P5 head (detect)
  - [[14, 17, 20], 1, Detect, [nc]]                 # 21
"""

# 2: BiFPN neck + r11_widefuse @ P4-BU (stack proven module on new topology)
ARCH_BIFPN_WIDEFUSE = BACKBONE + """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 9
  - [[-1, 6], 1, WeightedConcat, [1, 2]]            # 10
  - [-1, 2, A2C2f, [512, False, -1]]                # 11 — P4 top-down
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 12
  - [[-1, 4], 1, WeightedConcat, [1, 2]]            # 13
  - [-1, 2, A2C2f, [256, False, -1]]                # 14 — P3 head (detect)
  - [-1, 1, Conv, [256, 3, 2]]                      # 15
  - [[-1, 11, 6], 1, WeightedConcat, [1, 3]]        # 16 — BiFPN 3-way @ P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 17 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]                      # 18
  - [[-1, 8], 1, WeightedConcat, [1, 2]]            # 19
  - [-1, 2, C3k2, [1024, True]]                     # 20 — P5 head (detect)
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]          # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, Detect, [nc]]                 # 22
"""

# 3: CONTROL -- iterate the STOCK PAN twice (plain Concat, no learnable weights)
ARCH_DOUBLENECK = BACKBONE + """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 9
  - [[-1, 6], 1, Concat, [1]]                       # 10
  - [-1, 2, A2C2f, [512, False, -1]]                # 11 — P4 top-down (pass 1)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 12
  - [[-1, 4], 1, Concat, [1]]                       # 13
  - [-1, 2, A2C2f, [256, False, -1]]                # 14 — P3 (pass 1)
  - [-1, 1, Conv, [256, 3, 2]]                      # 15
  - [[-1, 11], 1, Concat, [1]]                      # 16
  - [-1, 2, A2C2f, [512, False, -1]]                # 17 — P4 bottom-up (pass 1)
  - [-1, 1, Conv, [512, 3, 2]]                      # 18
  - [[-1, 8], 1, Concat, [1]]                       # 19
  - [-1, 2, C3k2, [1024, True]]                     # 20 — P5 (pass 1)
  # ---- second fusion pass ----
  - [20, 1, nn.Upsample, [None, 2, "nearest"]]      # 21
  - [[-1, 17], 1, Concat, [1]]                      # 22
  - [-1, 2, A2C2f, [512, False, -1]]                # 23 — P4 top-down (pass 2)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 24
  - [[-1, 14], 1, Concat, [1]]                      # 25
  - [-1, 2, A2C2f, [256, False, -1]]                # 26 — P3 (pass 2, detect)
  - [-1, 1, Conv, [256, 3, 2]]                      # 27
  - [[-1, 23], 1, Concat, [1]]                      # 28
  - [-1, 2, A2C2f, [512, False, -1]]                # 29 — P4 (pass 2, detect)
  - [-1, 1, Conv, [512, 3, 2]]                      # 30
  - [[-1, 20], 1, Concat, [1]]                      # 31
  - [-1, 2, C3k2, [1024, True]]                     # 32 — P5 (pass 2, detect)
  - [[26, 29, 32], 1, Detect, [nc]]                 # 33
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r19_bifpn_70",
     "desc": "[1/3] BiFPN-lite neck: WeightedConcat fusion + 3-way edge @ P4 -- learnable multi-scale fusion, yolov12s/640",
     "yaml_content": ARCH_BIFPN, "batch": 56, "seed": 0, "epochs": 80},
    # {"name": "r19_bifpn_widefuse_70",
    #  "desc": "[2/3] BiFPN neck + r11_widefuse @ P4 -- does the best module stack on the new fusion topology?",
    #  "yaml_content": ARCH_BIFPN_WIDEFUSE, "batch": 54, "seed": 0, "epochs": 80},
    # {"name": "r19_doubleneck_70",
    #  "desc": "[3/3] CONTROL: stock PAN iterated twice (plain Concat) -- isolates 'more fusion depth' from 'learnable weighting' (second pass trains fresh -> 120 ep)",
    #  "yaml_content": ARCH_DOUBLENECK, "batch": 40, "seed": 0, "epochs": 120},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first, then remap Detect keys model.21.* -> model.N.* if the
    index shifted. Neck layers transfer where shape matches; WeightedConcat is
    identity-at-init so the 2-way fusion nodes transfer exactly, and only the
    layer(s) downstream of a changed (3-way / second-pass) node train fresh."""
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
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Seed: {run['seed']}   Epochs: {run.get('epochs', 80)}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=run.get("epochs", 80),
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
    print(f"  70% ABLATION — ROUND 19: NECK TOPOLOGY (BiFPN weighted fusion)")
    print(f"  Bars: 78.45 baseline | 79.40 r11_widefuse | 80.45 v5_topk15_beta3 (loss-only)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={run['batch']} ep={run.get('epochs',80)}  {run['desc']}")

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
