#!/usr/bin/env python3
"""
70% Ablation -- Round 31: WIDEFUSE-V2 (THE HYBRID BRANCH) + BEST TAL

This is the most ambitious and data-driven experiment yet. It combines the
strongest architectural idea (a surgical fix for r21's small-object weakness)
with the strongest training parameters (the v5 TAL config).

NEW ARCHITECTURE: ZGLSKAWideFuseV2
  - A direct upgrade to the winning ZGLSKAWideFuse from r11/r21.
  - It keeps the proven two-branch, expand-then-fuse structure.
  - Branch 1 (Unchanged): The proven square k=11 LKA for general context.
  - Branch 2 (NEW Hybrid): This branch is now a HYBRID of large and small RF ops:
      - Large-RF Path: The original strip-23 LKA for elongated objects.
      - Small-RF Path: The ZGSmallDetail logic (k=3 + k=5) for fine detail.
  - The two sub-paths are ADDED together. The result is a single "Hybrid
    Branch" effective at all scales, preventing large-RF operators from
    destroying small-object features at the source.

TRAINING PARAMS: v5_topk15_beta3_70
  - tal_topk=15, tal_alpha=0.5, tal_beta=3.0
  - Proven best training loss parameters. The lower beta weights classification
    more heavily, directly addressing the precision weakness of the r21 arch.

One run:
  1. r31_widefuse_v2_besttal_70
     - ZGLSKAWideFuseV2 @ P4 + DetectAux(0.5) + Best TAL params

This single run tests the most promising architecture with the most promising
training setup. If this does not produce a new state-of-the-art model, the
bottleneck is likely the dataset itself.

Usage:
  python run_round31_widefuse_v2.py
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

# --- The winning TAL parameters from Round 5 ---
BEST_TAL_PARAMS = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)

RUNS = [
    {
        "name": "r31_widefuse_v2_besttal_70",
        "desc": "[1/1] ZGLSKAWideFuseV2 arch + v5_topk15_beta3 TAL params",
        "yaml_content": ARCH_YAML_CONTENT,
        "batch": BATCH,
        "seed": 0,
        "epochs": EPOCHS,
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
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 80}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: {run['training_params']}")
    print(f"{'#' * 80}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=run["epochs"],
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
            **run["training_params"],
        )

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback
        traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"  70% ABLATION -- ROUND 31: WIDEFUSE-V2 + BEST TAL (1 run)")
    print(f"  Target: Combine best arch (WideFuseV2) with best TAL to fix small objects")
    print(f"{'=' * 80}")

    run = RUNS[0]
    print(f"  [1] {run['name']:<32} batch={run['batch']}  {run['desc']}")
    print(f"\n{'=' * 80}\n")
    
    result = run_experiment(run)

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 80}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    tag = "OK" if result["status"] == "OK" else "FAIL"
    print(f"  [{tag}] {result['name']:<32} {result['time']:.2f}h")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
