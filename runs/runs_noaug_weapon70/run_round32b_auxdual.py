#!/usr/bin/env python3
"""
70% Ablation -- Round 32B: DUAL-PATH AUX SUPERVISION (detail teacher)

This is the highest-confidence architectural fix for the small-object
precision collapse in r21. The key insight:

PROBLEM: DetectAux mirrors the main head -- both see the SAME fused features.
So the aux gradient cannot teach the backbone anything different. The aux head
is just extra loss on the same signal, not a corrective force.

FIX: DetectAuxDual routes DIFFERENT features to main vs aux towers:
  - Main towers see: [P3, P4_widefuse, P5]  (context-rich, good for med/large)
  - Aux towers see:  [P3, P4_pre_widefuse, P5] (detail-rich, good for small)

This forces the backbone to satisfy BOTH objectives simultaneously:
  - Main head rewards CONTEXT (large-RF widefuse features)
  - Aux head rewards DETAIL PRESERVATION (pre-widefuse features)

The backbone MUST maintain fine-grained features because the aux head
directly supervises them. This addresses gradient contamination at the
source -- the exact root cause of the "other" class collapse.

At inference: aux towers are dropped. ZERO cost. Identical to stock Detect.

ARCHITECTURE:
  layers 0-20: standard YOLOv12s backbone + head
  layer 21: ZGLSKAWideFuseV2[512, 11, 23, 3, 5]  @ P4 (= r31)
  layer 22: DetectAuxDual[nc, 0.5] from:
            main=[P3=14, P4_fused=21, P5=20]
            aux =[P3=14, P4_prefuse=17, P5=20]

TWO-STEP ABLATION:

1. r32b_auxdual_arch_only_70
   - DetectAuxDual + WideFuseV2
   - Default TAL (topk=10, beta=6.0)
   - Purpose: prove dual-path supervision is better than mirror-aux

2. r32b_auxdual_besttal_70
   - DetectAuxDual + WideFuseV2
   - Best TAL (topk=15, beta=3.0)
   - Purpose: combine best arch with best loss for new SOTA

Bars: 78.45 baseline | 79.57 r21_widefuse_aux_w50 | 80.45 v5_topk15_beta3

Usage:
  python run_round32b_auxdual.py
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
# ARCHITECTURE: WideFuseV2 + DetectAuxDual
# =============================================================================
# Key difference from r21/r31:
#   DetectAuxDual takes 6 inputs [main_p3, main_p4, main_p5, aux_p3, aux_p4, aux_p5]
#   Main sees post-widefuse P4 (layer 21)
#   Aux sees pre-widefuse P4 (layer 17) -- forces detail preservation
ARCH_YAML_CONTENT = """nc: 4
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
  - [[14, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 22
"""
# DetectAuxDual inputs:
#   main: P3=14(256ch), P4_fused=21(512ch), P5=20(1024ch)
#   aux:  P3=14(256ch), P4_prefuse=17(512ch), P5=20(1024ch)

# =============================================================================
# TRAINING PARAMETER CONFIGURATIONS
# =============================================================================
DEFAULT_TAL_PARAMS = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
BEST_TAL_70 = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)
REVIEW_TAL = dict(tal_topk=13, tal_alpha=0.7, tal_beta=4.0)

RUNS = [
    {
        "name": "r32b_auxdual_arch_only_70",
        "desc": "[1/3] Arch-only: WideFuseV2 + DetectAuxDual with DEFAULT TAL",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": DEFAULT_TAL_PARAMS,
    },
    {
        "name": "r32b_auxdual_besttal70_70",
        "desc": "[2/3] Best-of-both: WideFuseV2 + DetectAuxDual + 70% best TAL (topk15/b3)",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": BEST_TAL_70,
    },
    {
        "name": "r32b_auxdual_reviewtal_70",
        "desc": "[3/3] Best-of-both: WideFuseV2 + DetectAuxDual + review TAL (topk13/a07/b4)",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": REVIEW_TAL,
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
    print(f"  70% ABLATION -- ROUND 32B: DUAL-PATH AUX SUPERVISION (3 runs)")
    print(f"  Target: Force backbone to preserve detail via dual-path aux supervision")
    print(f"  Key change: Aux head sees PRE-widefuse P4, main head sees POST-widefuse P4")
    print(f"{'=' * 80}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<32} TAL={run['training_params']}")
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
        print(f"  [{tag}] {r['name']:<32} {r['time']:.2f}h")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
