#!/usr/bin/env python3
"""
70% Ablation -- Round 30: P3 DETAIL GUARD (small-object recovery for r21).

r21_widefuse_aux_w50_703 is the best overall model (mAP50-95=50.33, mAP50=
79.57) but has a persistent small-object weakness: mAP50_small=57.95 vs
baseline's 61.79 (-3.8pp), and "other" AP50_small collapses from 38.57 to
26.49 (-12pp). Crucially, its RECALL on small objects is actually the BEST
of all 90 runs (AR50_small=88.69) -- it FINDS small objects but misclassifies
them, especially the "other" class.

DIAGNOSIS: ZGLSKAWideFuse @ P4 has two large-RF branches (k=11 LKA + strip-23
LSKA). During training, gradients from these large-RF branches shift shared
backbone representations toward medium/large features, degrading the fine-
grained detail P3 needs for small objects. P3 features (layer 14) pass through
to Detect untouched, but their upstream backbone features are polluted.

EVIDENCE: r16_CompactFuse (replaced strip-23 with small k3+k5 kernels AT P4)
recovered "other" AP50_small from 23.59 to 32.87 (+9.3pp), proving small-
kernel operators DO recover fine detail. But it REPLACED the strip-23 at P4,
losing overall mAP50 (79.40->78.25). This round keeps widefuse intact and
adds the detail guard SEPARATELY at P3.

NEW MODULE: ZGSmallDetail[c2, k_fine, k_mid] -- zero-gated small-kernel block:
  y = x + gamma * pw2(act(GN(dw_k_fine(z) + dw_k_mid(z)))), z = act(pw1(x))
  gamma=0 at init -> exact identity, append-only. Two parallel depthwise convs
  (k=3 + k=5, dilation=1) capture fine local detail at two micro-scales.
  NO large-RF operator. Placed after P3 head (layer 14), before Detect.

Architecture for this round (widefuse @ P4 + detail guard @ P3 + DetectAux):
  layers 0-20: standard YOLOv12s backbone + head
  layer 21: ZGLSKAWideFuse[512, 11, 23]  @ P4 bottom-up output  (= r11/r21)
  layer 22: ZGSmallDetail[256, 3, 5]     @ P3 head output        (NEW)
  layer 23: DetectAux[nc, 0.5] from [P3_guarded=22, P4_widefuse=21, P5=20]

Three runs (all yolov12s/640, FIXED batch, default TAL for clean arch comparison):

  1. r30_p3detail_70          widefuse@P4 + P3-detail-guard + DetectAux(0.5)
                              -- the full stack, directly comparable to r21_widefuse_aux_w50

  2. r30_p3detail_noaux_70    widefuse@P4 + P3-detail-guard + stock Detect
                              -- isolate the P3-guard effect without the aux head

  3. r30_p3detail_only_70     stock P4 (no widefuse) + P3-detail-guard + stock Detect
                              -- isolate the P3-guard effect without widefuse

READ: mAP50_small, per-class "other" AP50_small, plus overall mAP50/mAP50-95.
Target: recover 3-5pp of small-object mAP50 without losing the overall mAP50 gains.

Bars: 78.45 baseline | 79.40 r11_widefuse | 79.57 r21_widefuse_aux_w50 (current best)
      | 80.45 v5_topk15_beta3 (loss-only, no arch).

Usage:
  python run_round30_p3detail.py
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
DETECT_SRC_IDX = 21  # Detect index in the pretrained yolov12s.pt
BATCH = 52  # fixed across all runs for a clean comparison
EPOCHS = 80

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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

# Run 1: widefuse @ P4 + P3 detail guard + DetectAux (the full stack)
ARCH_WIDEFUSE_P3DETAIL_AUX = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]    # 21 -- gated wide-fuse @ P4 (= r11/r21)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]        # 22 -- P3 detail guard (NEW)
  - [[22, 21, 20], 1, DetectAux, [nc, 0.5]]   # 23 -- detect from guarded-P3, widefuse-P4, P5
"""

# Run 2: widefuse @ P4 + P3 detail guard + stock Detect (isolate P3 guard without aux)
ARCH_WIDEFUSE_P3DETAIL = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]    # 21 -- gated wide-fuse @ P4 (= r11)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]        # 22 -- P3 detail guard (NEW)
  - [[22, 21, 20], 1, Detect, [nc]]            # 23 -- stock Detect
"""

# Run 3: stock P4 + P3 detail guard + stock Detect (isolate P3 guard alone)
ARCH_P3DETAIL_ONLY = BASE_0_20 + """
  - [14, 1, ZGSmallDetail, [256, 3, 5]]        # 21 -- P3 detail guard only (NEW)
  - [[21, 17, 20], 1, Detect, [nc]]            # 22 -- Detect from guarded-P3, stock-P4, P5
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {
        "name": "r30_p3detail_70",
        "desc": "[1/3] widefuse@P4 + ZGSmallDetail@P3 + DetectAux(0.5) -- full stack, direct r21 comparison",
        "yaml_content": ARCH_WIDEFUSE_P3DETAIL_AUX,
        "batch": BATCH,
        "seed": 0,
        "epochs": EPOCHS,
    },
    {
        "name": "r30_p3detail_noaux_70",
        "desc": "[2/3] widefuse@P4 + ZGSmallDetail@P3 + stock Detect -- P3-guard effect without aux head",
        "yaml_content": ARCH_WIDEFUSE_P3DETAIL,
        "batch": BATCH,
        "seed": 0,
        "epochs": EPOCHS,
    },
    {
        "name": "r30_p3detail_only_70",
        "desc": "[3/3] stock P4 + ZGSmallDetail@P3 + stock Detect -- P3-guard effect alone (no widefuse)",
        "yaml_content": ARCH_P3DETAIL_ONLY,
        "batch": BATCH,
        "seed": 0,
        "epochs": EPOCHS,
    },
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first, then remap Detect keys if the index shifted. Main head
    + backbone + neck transfer; aux towers (cv2a/cv3a) are fresh."""
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
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Seed: {run['seed']}   Epochs: {run['epochs']}")
    print(f"{'#' * 70}\n")

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

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION -- ROUND 30: P3 DETAIL GUARD (3 runs)")
    print(f"  Target: recover small-obj mAP50 without losing r21's overall gains")
    print(f"  Bars: 78.45 baseline | 79.40 r11_widefuse | 79.57 r21_widefuse_aux_w50")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<28} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<28} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
