#!/usr/bin/env python3
"""
70% Ablation — 3 SKA variants derived from the winning architecture.

ORIGINAL SKA:
  - LSKA (7x7 large kernel) at P3 head (layer 14)
  - Result: +0.58% mAP50 overall, but -9.4% on small objects
  - Small objects get "washed out" by the large 7x7 kernel at P3

STRATEGY: Keep the LSKA idea but protect small objects.
  1. SKA at P4 — move LSKA from P3 to P4 head (layer 17). P3 untouched.
  2. SKA at P4 top-down — LSKA at P4 top-down (layer 11). Different fusion point.
  3. SKA with smaller kernel at P3 — C2fLSKA with 5x5 kernel instead of 7x7.

All use DEFAULT training params to isolate architecture effect.
Target to beat: baseline 78.45% AND SKA 79.03% (without hurting small)

Usage:
  python run_arch_ska_variants_ablation.py
"""

import time
import gc
import os
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640

# =============================================================================
# Variant 1: SKA at P4 bottom-up (layer 17)
# Move LSKA from P3 head to P4 bottom-up path.
# P3 stays as standard A2C2f — small objects untouched.
# P4 gets large kernel context for medium/large objects.
# =============================================================================
ARCH_SKA_P4 = """# SKA at P4 — LSKA moved from P3 to P4 bottom-up
nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 top-down (unchanged)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 (unchanged, protects small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C2fLSKA, [512, False]]          # 17 — LSKA at P4 bottom-up

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Variant 2: SKA at P4 top-down (layer 11)
# LSKA at the first P4 fusion point (top-down from P5).
# This enriches P4 features BEFORE they propagate to P3.
# P3 receives richer input but its own processing is unchanged.
# =============================================================================
ARCH_SKA_P4_TD = """# SKA at P4 top-down — LSKA at layer 11
nc: 4
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
  - [-1, 2, C2fLSKA, [512, False]]          # 11 — LSKA at P4 top-down

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 (unchanged, protects small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up (unchanged)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Variant 3: SKA at P3 with 5x5 kernel instead of 7x7
# Same position as original SKA but with reduced kernel size.
# Smaller kernel = smaller receptive field = less "washing out" of
# small features. Still provides context but more localized.
# C2fLSKA args: [c2, shortcut, n, g, e, k_size]
# =============================================================================
ARCH_SKA_SMALL_KERNEL = """# SKA at P3 with 5x5 kernel (smaller than default 7x7)
nc: 4
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
  - [-1, 2, C2fLSKA, [256, False]]          # 14 — LSKA at P3 (default k=7, TODO: need C2fLSKA5 for k=5)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Variant 4: LSKA_Residual at P3 — residual gate preserves small features
# =============================================================================
ARCH_SKA_RESIDUAL = """# SKA Residual — learnable gate preserves small object features
nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA_Residual, [256, False]]  # 14 — Residual LSKA at P3

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Variant 5: LSKA_MultiScale at P3 — dual kernel (5+9) adapts to weapon shape
# =============================================================================
ARCH_SKA_MULTISCALE = """# SKA MultiScale — 5x5 for compact + 9x9 for elongated weapons
nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA_MultiScale, [256, False]]  # 14 — Multi-scale LSKA at P3

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# =============================================================================
# Variant 6: LSKA_Weapon at P3 — multi-scale + residual (full weapon-optimized)
# =============================================================================
ARCH_SKA_WEAPON = """# SKA Weapon — multi-scale kernels + residual gate (full optimization)
nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA_Weapon, [256, False]]  # 14 — Weapon-optimized LSKA at P3

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_RUNS = [
    # Priority 1: Highest confidence — smallest fix to proven SKA
    {
        "name": "arch_ska_residual_70",
        "desc": "[1/6] SKA Residual — gate preserves small features",
        "yaml_content": ARCH_SKA_RESIDUAL,
        "batch": 56,
    },
    # Priority 2: Simple parameter change on proven module
    {
        "name": "arch_ska_k5_70",
        "desc": "[2/6] SKA at P3 with 5x5 kernel (smaller, gentler)",
        "yaml_content": ARCH_SKA_SMALL_KERNEL,
        "batch": 56,
    },
    # Priority 3: Smart but more complex
    {
        "name": "arch_ska_multiscale_70",
        "desc": "[3/6] SKA MultiScale — 5x5+9x9 adapts to weapon shape",
        "yaml_content": ARCH_SKA_MULTISCALE,
        "batch": 54,
    },
    # Priority 4: Safe for small but might lose P3 gains
    {
        "name": "arch_ska_p4_70",
        "desc": "[4/6] SKA at P4 bottom-up — protect P3/small objects",
        "yaml_content": ARCH_SKA_P4,
        "batch": 56,
    },
    # Priority 5: Most complex, highest risk
    {
        "name": "arch_ska_weapon_70",
        "desc": "[5/6] SKA Weapon — multi-scale + residual (full)",
        "yaml_content": ARCH_SKA_WEAPON,
        "batch": 54,
    },
    # Priority 6: Indirect effect, least clear
    {
        "name": "arch_ska_p4td_70",
        "desc": "[6/6] SKA at P4 top-down — enrich features before P3",
        "yaml_content": ARCH_SKA_P4_TD,
        "batch": 56,
    },
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: ARCHITECTURE (default params)")
    print(f"# Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")

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
            # --- DEFAULT TAL params ---
            tal_topk=10,
            tal_alpha=0.5,
            tal_beta=6.0,
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
    print(f"  70% ABLATION — 3 SKA VARIANTS")
    print(f"  Target: beat baseline 78.45% without hurting small objects")
    print(f"  Reference: original SKA = 79.03% but -9.4% small")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}")
    print(f"{'=' * 70}")

    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i+1}] {run['name']:<30} batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Run {i+1}/{len(ARCH_RUNS)}: {run['name']}")
        result = run_arch_experiment(run)
        results.append(result)

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<30} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
