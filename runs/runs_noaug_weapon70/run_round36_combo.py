#!/usr/bin/env python3
"""
70% Ablation -- Round 36: BEST COMBINATIONS (2 runs)

Combines the proven winners from R32B → R35 into two final architectures.

DATA-DRIVEN DESIGN RULES (confirmed across 12+ experiments):
  1. Enhancement on MAIN, raw on AUX (R34 > R33)
  2. WideFuseV2@P4 is essential (every arch without it is worse)
  3. ZGSmallDetail@P3 on main helps small objects (R34: +1.34 mAP50_small)
  4. ZGLSKAWideFuse@P5 on main helps mAP50-95 (R35: 52.82 = new best)
  5. Higher aux weight hurts (R35 aux075: "other" small collapsed)

ARCH 1: R34main + P5 context — "Full Symmetric Dual-Path" (80% confidence)
  Combines R34 (best small-obj) + R35 p5context (best mAP50-95).
  ALL three scales enhanced on main, ALL three raw on aux.
    Main: [P3_detail, P4_widefuse, P5_context]
    Aux:  [P3_raw,    P4_raw,     P5_raw]

ARCH 2: R32B + P5 context — "Minimal P5 Upgrade" (70% confidence)
  R32B is the mAP50 king (82.58). Just add P5 context to boost mAP50-95
  without touching the winning P3/P4 formula.
    Main: [P3_raw,  P4_widefuse, P5_context]
    Aux:  [P3_raw,  P4_raw,     P5_raw]

Baselines:
  R32B:       82.58 mAP50 / 52.41 mAP50-95 / 64.73 mAP50_small
  R34main:    82.36 mAP50 / 52.28 mAP50-95 / 66.07 mAP50_small
  R35 p5ctx:  82.38 mAP50 / 52.82 mAP50-95 / 65.16 mAP50_small

Usage:
  python run_round36_combo.py
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
# SHARED BASE (layers 0-20)
# =============================================================================
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (raw)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (raw)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head (raw)
"""

# =============================================================================
# ARCH 1: Full Symmetric Dual-Path (R34 + P5 context) — 80% confidence
# =============================================================================
# All three scales enhanced on main, all three raw on aux.
ARCH_1_FULL_SYMMETRIC = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- P4 widefuse (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 24
"""
# Main: [P3_detail=22(256), P4_widefuse=21(512), P5_context=23(1024)]
# Aux:  [P3_raw=14(256),    P4_raw=17(512),      P5_raw=20(1024)]

# =============================================================================
# ARCH 2: R32B + P5 context — 70% confidence
# =============================================================================
# Keep R32B's winning P3/P4 formula, just add P5 enhancement.
ARCH_2_R32B_P5 = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- P4 widefuse (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 22 -- P5 context (main)
  - [[14, 21, 22, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 23
"""
# Main: [P3_raw=14(256),  P4_widefuse=21(512), P5_context=22(1024)]
# Aux:  [P3_raw=14(256),  P4_raw=17(512),      P5_raw=20(1024)]

# =============================================================================
# TRAINING — DEFAULT TAL ONLY (arch comparison)
# =============================================================================
DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {
        "name": "r36_full_symmetric_arch_only_70",
        "desc": "[1/2] Full symmetric: detail@P3 + widefuse@P4 + context@P5 — 80% confidence",
        "yaml_content": ARCH_1_FULL_SYMMETRIC,
        "training_params": DEFAULT_TAL,
    },
    {
        "name": "r36_r32b_p5ctx_arch_only_70",
        "desc": "[2/2] R32B + P5 context only — 70% confidence",
        "yaml_content": ARCH_2_R32B_P5,
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
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, "
              f"strides = {model.model.stride.tolist()}")

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
    print(f"  70% ABLATION -- ROUND 36: BEST COMBINATIONS (2 runs)")
    print(f"  Baselines: R32B=82.58 | R34=82.36/66.07small | R35p5=82.38/52.82m5095")
    print(f"{'=' * 80}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<40} {run['desc']}")
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
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
