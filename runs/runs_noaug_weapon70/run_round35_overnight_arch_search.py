#!/usr/bin/env python3
"""
70% Ablation -- Round 35: OVERNIGHT ARCHITECTURE SEARCH (4 runs)

Based on ALL prior experiments (R21 → R34), we now have clear design rules:

  RULE 1: Enhancement on MAIN, raw on AUX (R34 > R33)
  RULE 2: WideFuseV2 is the best feature enhancer (R32B >> R21 >> baseline)
  RULE 3: ZGSmallDetail helps small objects but hurts "other" large (R33, R34)
  RULE 4: Dual-path supervision is essential (R32B >> R21)

Current best architectures:
  R32B: mAP50=82.58, mAP50_small=64.73, other_small=45.79 (best overall)
  R34:  mAP50=82.36, mAP50_small=66.07, other_small=47.23 (best small-obj)

Four new architectures, all default TAL, all following the proven rules.

ARCH 1 — WideFuseV2 at P3 instead of ZGSmallDetail (75% confidence)
  R34 used ZGSmallDetail (detail-only) at P3. This replaces it with
  WideFuseV2 (context+detail hybrid) at P3 — the same mechanism that
  proved itself at P4. Smaller kernels (k_sq=7, k_strip=15) to match
  P3's higher resolution. Should give P3 BOTH context and detail.

ARCH 2 — R34main + higher aux weight 0.75 (65% confidence)
  R34's "other" large dropped vs R32B because the aux signal wasn't
  strong enough to prevent the backbone from shifting toward detail.
  Increasing aux_weight from 0.5 to 0.75 pushes harder on raw-feature
  preservation. Same architecture as R34, different supervision balance.

ARCH 3 — Full dual enhancement: WideFuseV2@P3 + WideFuseV2@P4 (70% confidence)
  Both P3 and P4 get WideFuseV2 (different kernel sizes). Aux anchors
  both raw P3 and raw P4. The "enhance everything on main, anchor
  everything on aux" philosophy taken to its logical conclusion.
  Risk: might be too much capacity / over-parameterized.

ARCH 4 — R32B + stronger aux weight 0.75 (60% confidence)
  The simplest possible improvement: take the best overall arch (R32B)
  and just increase aux_weight. If R32B's weakness is that aux doesn't
  push hard enough on detail preservation, this fixes it without any
  architectural change. Control experiment.

Usage:
  python run_round35_overnight.py
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
# SHARED BASE (layers 0-20, identical for all)
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
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

# =============================================================================
# ARCH 1: WideFuseV2 at P3 (replaces ZGSmallDetail) — 75% confidence
# =============================================================================
# P3 gets WideFuseV2 with smaller kernels (k_sq=7, k_strip=15) to match
# its higher resolution. Both context AND detail in one block.
ARCH_1_WFV2_P3 = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
  - [14, 1, ZGLSKAWideFuseV2, [256, 7, 15, 3, 5]]   # 22 -- WideFuseV2 @ P3 (smaller kernels)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 23
"""
# Main: [P3_fused=22, P4_fused=21, P5=20]
# Aux:  [P3_raw=14,   P4_raw=17,   P5=20]

# =============================================================================
# ARCH 2: R34main + higher aux weight (0.75) — 65% confidence
# =============================================================================
# Exact same as R34 but aux_weight 0.5 -> 0.75. Stronger preservation signal.
ARCH_2_R34_AUX075 = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail P3 (MAIN)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.75]]  # 23 -- aux_weight=0.75
"""
# Main: [P3_detail=22, P4_fused=21, P5=20]
# Aux:  [P3_raw=14,    P4_raw=17,   P5=20]

# =============================================================================
# ARCH 3: R34main + ZGSmallDetail@P5 — 70% confidence
# =============================================================================
# R34 enhances P3 (detail) and P4 (widefuse). P5 is untouched.
# Adding a ZGSmallDetail at P5 for main lets the main head also get
# enhanced P5 features, while aux anchors raw P5.
# Tests: does enhancing ALL three scales help?
ARCH_3_R34_P5DETAIL = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail P3 (MAIN)
  - [20, 1, ZGSmallDetail, [1024, 3, 5]]             # 23 -- detail P5 (MAIN)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 24
"""
# Main: [P3_detail=22, P4_fused=21, P5_detail=23]
# Aux:  [P3_raw=14,    P4_raw=17,   P5_raw=20]

# =============================================================================
# ARCH 4: R32B + higher aux weight (0.75) — 60% confidence
# =============================================================================
# Simplest control: R32B architecture, just increase aux_weight.
# Tests whether supervision strength alone fixes the small-obj gap.
ARCH_4_R32B_AUX075 = BASE_0_20 + """\
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
  - [[14, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.75]]  # 22 -- aux_weight=0.75
"""
# Main: [P3=14,     P4_fused=21, P5=20]
# Aux:  [P3_raw=14, P4_raw=17,   P5=20]

# =============================================================================
# TRAINING — DEFAULT TAL ONLY (arch comparison)
# =============================================================================
DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {
        "name": "r35_wfv2_p3_arch_only_70",
        "desc": "[1/4] WideFuseV2@P3 (k7/k15) + WideFuseV2@P4 — 75% confidence",
        "yaml_content": ARCH_1_WFV2_P3,
        "training_params": DEFAULT_TAL,
    },
    {
        "name": "r35_r34_p5detail_arch_only_70",
        "desc": "[2/4] R34main + ZGSmallDetail@P5 (enhance all 3 scales) — 70% confidence",
        "yaml_content": ARCH_3_R34_P5DETAIL,
        "training_params": DEFAULT_TAL,
    },
    {
        "name": "r35_r34_aux075_arch_only_70",
        "desc": "[3/4] R34main + aux_weight=0.75 — 65% confidence",
        "yaml_content": ARCH_2_R34_AUX075,
        "training_params": DEFAULT_TAL,
    },
    {
        "name": "r35_r32b_aux075_arch_only_70",
        "desc": "[4/4] R32B + aux_weight=0.75 (control) — 60% confidence",
        "yaml_content": ARCH_4_R32B_AUX075,
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
    print(f"  70% ABLATION -- ROUND 35: OVERNIGHT ARCHITECTURE SEARCH (4 runs)")
    print(f"  Baselines: R32B=82.58 mAP50 | R34=82.36 mAP50, 66.07 mAP50_small")
    print(f"{'=' * 80}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<36} {run['desc']}")
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
