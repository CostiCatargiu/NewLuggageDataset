#!/usr/bin/env python3
"""
70% Ablation — Round 12: SMALL-OBJECT RECOVERY on top of the new best
architecture (r11_widefuse_70 = 79.40% mAP50).

Context:
  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4 bottom-up) = 79.40% mAP50,
    the new best -- but like every gated P4-BU variant, it trades away
    small-object performance: mAP50_small = 56.65 vs baseline's 61.79
    (-5.14pp). widefuse's two branches (k_sq=11, k_strip=23) are BOTH
    large-receptive-field shapes (RF~35 and RF~45+), so they're redundant
    in scale -- neither targets small objects.
  - r11_dual_p3p4_70 (plain ZGLSKA[512,11]@P4-BU + ZGLSKA[256,7]@P3 head,
    two independent gates) got mAP50_small up to 59.00 (best gated result)
    but cost -1.88pp on "other" and -0.63pp overall vs r6 -- a full
    spatial-attention branch at P3 apparently competes for "other"-class
    capacity.

Three ideas, each targeting the small-object gap via a DIFFERENT mechanism,
ALL reusing EXISTING registered modules (ZGLSKAWideFuse, ZGGC) -- no new
module code this round:

  1. r12_widefuse_ksmall_70 (~30-35%) -- near-zero-risk single-arg change.
     ZGLSKAWideFuse[512, 11, 3] @ P4-BU (exact same slot/structure as the
     79.40 winner). Swaps the second branch's kernel from strip-23 (RF~45,
     redundant with k11's RF~35) to strip-3 (RF~11) so the fused branch
     covers genuinely different scales instead of two overlapping
     large receptive fields.

  2. r12_widefuse_p3gc_70 (~25-30%) -- different mechanism: global context,
     not spatial attention. Keeps widefuse@P4 EXACTLY as in the 79.40 run,
     and adds ZGGC[256, 8] (gated global-context: pooled context vector ->
     bottleneck -> broadcast-add, single gamma) at the P3 head. Much
     cheaper / less spatially invasive than dual_p3p4's full k=7 LKA
     branch -- recalibrates P3 channels for small objects without the
     large-kernel spatial remap that hurt "other" last time.

  3. r12_dualwidefuse_70 (~20-25%) -- most ambitious: apply the WINNING
     wide-fuse structure at BOTH scales. widefuse[512,11,23]@P4-BU
     (unchanged, = 79.40 winner) + a second ZGLSKAWideFuse[256, 7, 3] @
     P3 head, with kernels scaled down for P3's higher resolution (k_sq=7,
     k_strip=3 vs P4's 11/23). Two independent gates (the riskiest part,
     cf. dual_p3p4), but both now use the proven fusion structure with
     P3-appropriate (smaller) kernels rather than dual_p3p4's bare k=7-only
     branch.

All three: pure arch, default TAL, gated identity-at-init, append-only
(standard Detect-remap loader, det shifts by +1 or +2 depending on # of
new layers). Bars: 78.45 baseline (mAP50_small=61.79) | 79.19 zg_p4_k11 |
79.40 r11_widefuse_70 (current best, mAP50_small=56.65).

Usage:
  python run_round12_small_obj.py
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head
"""

# 1: widefuse @ P4-BU, but second branch is small-RF (strip-3) instead of
#    strip-23, so the fused branch covers two genuinely different scales
ARCH_WIDEFUSE_KSMALL = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 3]]  # 21 — gated wide-fuse k11+strip3 @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: widefuse @ P4-BU UNCHANGED (= 79.40 winner) + cheap global-context gate @ P3
ARCH_WIDEFUSE_P3GC = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner)
  - [14, 1, ZGGC, [256, 8]]                 # 22 — NEW: gated global-context @ P3 head
  - [[22, 21, 20], 1, Detect, [nc]]
"""

# 3: widefuse @ P4-BU UNCHANGED (= 79.40 winner) + second wide-fuse @ P3,
#    with smaller, P3-appropriate kernels
ARCH_DUALWIDEFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner)
  - [14, 1, ZGLSKAWideFuse, [256, 7, 3]]    # 22 — NEW: gated wide-fuse k7+strip3 @ P3 head
  - [[22, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r12_widefuse_ksmall_70",
     "desc": "[1/3] widefuse @ P4-BU, strip23 -> strip3 (multi-scale fusion) -- ~30-35% confidence",
     "yaml_content": ARCH_WIDEFUSE_KSMALL, "batch": 60},
    {"name": "r12_widefuse_p3gc_70",
     "desc": "[2/3] widefuse@P4 (unchanged) + ZGGC global-context @ P3 head -- ~25-30% confidence",
     "yaml_content": ARCH_WIDEFUSE_P3GC, "batch": 60},
    {"name": "r12_dualwidefuse_70",
     "desc": "[3/3] widefuse@P4 (unchanged) + widefuse[256,7,3] @ P3 head -- ~20-25% confidence",
     "yaml_content": ARCH_DUALWIDEFUSE, "batch": 56},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted.

    Use for architectures that APPEND new layer(s) after the original
    layer 20 (layers 0-20 unchanged, only Detect's index shifts)."""
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
    print(f"# TAL: DEFAULT   Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

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
    print(f"  70% ABLATION — ROUND 12: SMALL-OBJECT RECOVERY")
    print(f"  Bars: 78.45 baseline (small=61.79) | 79.19 zg_p4_k11 | "
          f"79.40 r11_widefuse_70 (small=56.65, current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={run['batch']}  {run['desc']}")

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
