#!/usr/bin/env python3
"""
70% Ablation -- Round 34: LEARNED AUX P3 PROJECTION

Builds on R32B (DetectAuxDual) — the best architecture so far:
  - R32B: mAP50=82.58, mAP50-95=52.41, "other" AP50=65.05

R33 showed that ZGSmallDetail on aux P3 helps "other" small (+3.70pp)
but hurts knife small (-5.70pp) and "other" large (-3.72pp). The spatial
detail kernels (k=3+k=5) were too aggressive and class-blind.

FIX: Replace the fixed spatial detail block with a LEARNED projection.
Instead of forcing spatial detail extraction, let the network learn WHICH
channels to emphasize for the aux head's P3 input. This is class-adaptive
— different channels can serve different classes.

Two variants:
  1. Conv 1x1: pure channel re-weighting (no spatial ops)
  2. Conv 3x3: small spatial context + channel re-weighting

Both use standard Conv (Conv2d + BN + SiLU) — no new modules needed.

Architecture:
  Main head: [P3=14, P4_fused=21, P5=20]      (unchanged from R32B)
  Aux head:  [P3_proj=22, P4_prefuse=17, P5=20] (22 = learned projection of P3)

SIX RUNS (2 architectures x 3 TAL configs):

  r34_proj1x1_arch_only_70       Conv 1x1 + default TAL
  r34_proj1x1_besttal70_70       Conv 1x1 + 70% best TAL
  r34_proj1x1_reviewtal_70       Conv 1x1 + review TAL
  r34_proj3x3_arch_only_70       Conv 3x3 + default TAL
  r34_proj3x3_besttal70_70       Conv 3x3 + 70% best TAL
  r34_proj3x3_reviewtal_70       Conv 3x3 + review TAL

Usage:
  python run_round34_auxdual_proj.py
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
BATCH = 48
EPOCHS = 80

# =============================================================================
# SHARED BACKBONE + HEAD (layers 0-21)
# =============================================================================
BASE_0_21 = """nc: 4
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (PRE-widefuse)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
"""

# Variant 1: Conv 1x1 projection — pure channel re-weighting
ARCH_PROJ_1x1 = BASE_0_21 + """  - [14, 1, Conv, [256, 1]]                                    # 22 -- learned 1x1 aux P3 projection
  - [[14, 21, 20, 22, 17, 20], 1, DetectAuxDual, [nc, 0.5]]   # 23
"""

# Variant 2: Conv 3x3 projection — spatial + channel
ARCH_PROJ_3x3 = BASE_0_21 + """  - [14, 1, Conv, [256, 3]]                                    # 22 -- learned 3x3 aux P3 projection
  - [[14, 21, 20, 22, 17, 20], 1, DetectAuxDual, [nc, 0.5]]   # 23
"""

# =============================================================================
# TRAINING PARAMETER CONFIGURATIONS
# =============================================================================
DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
BEST_TAL_70 = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)
REVIEW_TAL = dict(tal_topk=13, tal_alpha=0.7, tal_beta=4.0)

RUNS = [
    {
        "name": "r34_proj1x1_arch_only_70",
        "desc": "[1/2] Conv 1x1 aux P3 + default TAL (highest confidence)",
        "yaml_content": ARCH_PROJ_1x1,
        "training_params": DEFAULT_TAL,
    },
    {
        "name": "r34_proj3x3_arch_only_70",
        "desc": "[2/2] Conv 3x3 aux P3 + default TAL",
        "yaml_content": ARCH_PROJ_3x3,
        "training_params": DEFAULT_TAL,
    },
]

# =============================================================================
# HELPERS
# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights and remap Detect keys if index shifted."""
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
    run_name = run["name"]
    yaml_path = os.path.join(YAML_DIR, f"{run_name}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 80}")
    print(f"# {run_name}")
    print(f"# {run['desc']}")
    print(f"# TAL: {run['training_params']}")
    print(f"{'#' * 80}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run_name,
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            **run["training_params"],
        )

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run_name} ({elapsed:.2f}h)")
        return {"name": run_name, "status": "OK", "time": elapsed}

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run_name} ({elapsed:.2f}h) -- {e}")
        import traceback
        traceback.print_exc()
        return {"name": run_name, "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"  70% ABLATION -- ROUND 34: LEARNED AUX P3 PROJECTION (2 runs)")
    print(f"  Target: Recover 'other' AP50_small without R33's regressions")
    print(f"  Build on: R32B (mAP50=82.58, best arch so far)")
    print(f"{'=' * 80}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<36} TAL={run['training_params']}")
    print(f"\n{'=' * 80}\n")

    results = []
    for i, run in enumerate(RUNS):
        print(f"\n>>> Run {i + 1}/{len(RUNS)}: {run['name']}")
        results.append(run_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 80}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<36} {r['time']:.2f}h")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
