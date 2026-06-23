#!/usr/bin/env python3
"""
70% Ablation -- Round 33: R32B + DETAIL-ENHANCED AUX P3

Builds on R32B (DetectAuxDual), which proved dual-path supervision works:
  - R32B: mAP50=82.58 (+1.12 vs R21, +1.41 vs baseline)
  - Best "other" AP50 of all runs (65.05)
  - Best mAP50-95 of all runs (52.41)

REMAINING WEAKNESS: "other" AP50_small drops -3.09pp vs baseline.

ROOT CAUSE: In R32B, both main and aux heads see the SAME P3 features
(layer 14). The aux head forces detail preservation at P4 (pre-widefuse),
but P3 goes through untouched. There is no dual-path supervision at the
P3 level — the scale most critical for small objects.

FIX: Pass P3 through ZGSmallDetail (k=3 + k=5 depthwise detail block)
and feed the detail-enhanced P3 to the aux head. This forces the backbone
to preserve fine-grained detail at the P3 level too.

  Main head sees: [P3=14,          P4_fused=21,    P5=20]  (context-rich)
  Aux head sees:  [P3_detail=22,   P4_prefuse=17,  P5=20]  (detail-rich)

ZGSmallDetail is zero-gated (gamma=0 at init) — safe, non-destructive.
At inference: aux is dropped, ZGSmallDetail is unused. Zero cost.

THREE RUNS (architecture x TAL):
  1. r33_auxdual_p3d_arch_only_70   -- default TAL (arch comparison)
  2. r33_auxdual_p3d_besttal70_70   -- 70% best TAL (topk15/b3)
  3. r33_auxdual_p3d_reviewtal_70   -- review TAL (topk13/a07/b4)

Usage:
  python run_round33_auxdual_p3detail.py
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
# ARCHITECTURE: R33 = WideFuseV2 + ZGSmallDetail@P3 + DetectAuxDual
# =============================================================================
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (standard)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (PRE-widefuse)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- WideFuseV2 @ P4
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail-enhanced P3 for aux
  - [[14, 21, 20, 22, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 23
"""
# DetectAuxDual inputs (6 total):
#   Main: P3=14(256ch), P4_fused=21(512ch), P5=20(1024ch)
#   Aux:  P3_detail=22(256ch), P4_prefuse=17(512ch), P5=20(1024ch)

# =============================================================================
# TRAINING PARAMETER CONFIGURATIONS
# =============================================================================
DEFAULT_TAL_PARAMS = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
BEST_TAL_70 = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)
REVIEW_TAL = dict(tal_topk=13, tal_alpha=0.7, tal_beta=4.0)

RUNS = [
    {
        "name": "r33_auxdual_p3d_arch_only_70",
        "desc": "[1/3] Arch-only: R33 (AuxDual + P3 detail) with DEFAULT TAL",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": DEFAULT_TAL_PARAMS,
    },
    {
        "name": "r33_auxdual_p3d_besttal70_70",
        "desc": "[2/3] R33 + 70% best TAL (topk15/b3)",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": BEST_TAL_70,
    },
    {
        "name": "r33_auxdual_p3d_reviewtal_70",
        "desc": "[3/3] R33 + review TAL (topk13/a07/b4)",
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
    print(f"  70% ABLATION -- ROUND 33: AUXDUAL + P3 DETAIL (3 runs)")
    print(f"  Target: Fix 'other' AP50_small by adding detail-enhanced P3 to aux head")
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
