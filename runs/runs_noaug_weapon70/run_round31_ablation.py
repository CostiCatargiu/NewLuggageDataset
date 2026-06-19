#!/usr/bin/env python3
"""
70% Ablation -- Round 31: WIDEFUSE-V2 ABLATION

This script performs a clean ablation to first validate the new ZGLSKAWideFuseV2
architecture against the previous best (r21) using default training parameters,
and then combines the new architecture with the best known training parameters.
This isolates the effect of the architectural change before attempting to find
a new state-of-the-art.

NEW ARCHITECTURE: ZGLSKAWideFuseV2
  - A direct, surgical upgrade to ZGLSKAWideFuse.
  - The second branch is now a HYBRID of large-RF (strip-23) and small-RF
    (k=3 + k=5) operators, allowing it to model features at all scales.
  - This is designed to fix the small-object precision drop in r21.

TWO-STEP EXPERIMENT:

1. Run 1: Pure Architecture Comparison
   - Name: r31_widefuse_v2_arch_only_70
   - Arch: ZGLSKAWideFuseV2 + DetectAux(0.5)
   - TAL: Default (topk=10, beta=6.0)
   - Purpose: Apples-to-apples comparison vs. r21_widefuse_aux_w50_703.
     This will prove if the architecture itself is superior.

2. Run 2: Best of Both Worlds
   - Name: r31_widefuse_v2_besttal_70
   - Arch: ZGLSKAWideFuseV2 + DetectAux(0.5)
   - TAL: Best (topk=15, beta=3.0)
   - Purpose: Combine the new, validated architecture with the best training
     configuration to achieve a new state-of-the-art.

Usage:
  python run_round31_ablation.py
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

# --- The new WideFuseV2 architecture ---
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]] # 21 -- NEW Hybrid-Branch WideFuseV2 @ P4
  - [[14, 21, 20], 1, DetectAux, [nc, 0.5]]         # 22 -- Detect from P3, widefuse-v2-P4, P5
"""

# --- Training Parameter Configurations ---
DEFAULT_TAL_PARAMS = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
BEST_TAL_PARAMS = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)

RUNS = [
    {
        "name": "r31_widefuse_v2_arch_only_70",
        "desc": "[1/2] Arch-only test: WideFuseV2 with DEFAULT TAL params",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": DEFAULT_TAL_PARAMS,
    },
    {
        "name": "r31_widefuse_v2_besttal_70",
        "desc": "[2/2] Best-of-both: WideFuseV2 with BEST TAL params",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": BEST_TAL_PARAMS,
    },
]

def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: {len(matched)}/{len(remapped)} Detect keys transferred on top")
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

def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"  70% ABLATION -- ROUND 31: WIDEFUSE-V2 ABLATION (2 runs)")
    print(f"  Target: 1. Prove V2 arch is better. 2. Combine with best TAL.")
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
