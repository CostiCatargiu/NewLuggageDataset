#!/usr/bin/env python3
"""
70% Ablation — Round 11: FOUR NEW ARCHITECTURE IDEAS, each targeting a
specific diagnosis from rounds 6-10.

Recap of the evidence base:
  - r6_zgp4_k11_702 (ZGLSKA k=11 @ P4 bottom-up/post-fusion, full 512ch,
    default TAL) = 79.19% mAP50 -- best pure architecture so far.
  - Round 7 dose-response (single ZGLKA @ P4-BU): k7=79.05, k11=79.19
    (peak), k15=79.03 -- fairly FLAT near the peak. ZGStrip(k23)=79.07.
    k11+GC stacked (two SEPARATE gates, same scale, full channels each)
    = 78.66 (-0.53 vs k11).
  - Round 10: relocating k11 to P4 top-down (pre-fusion) = 78.83 (-0.36);
    ZGLSKAStripFuse (k11+strip23 fused via CHANNEL-SPLIT, one gate, only
    c1/2 width each) = 78.27 (-0.92, WORSE than the two-gate stack above);
    ZGLSKAMultiDil (k7/dil2+k11/dil3 summed, full width, one gate) = 79.06
    (-0.13, but notably HIGHER mAP50_95/recall/mAP50_large than r6).
  - Per-class analysis: "other" is the dominant lever (lowest AP50 ~50-55
    vs ~86-90 for the rest -- a 1pp swing there moves mAP50_all by 0.25pp).
    r6's win was driven mostly by "other" (+1.00) and knife (+1.10). Every
    round-10 failure hit "other" hardest (stripfuse: -3.58pp vs r6).
  - Channel width matters more than shape: stripfuse's channel-split
    (c1/2 per branch) was worse than round 7's full-width two-gate stack.
    P4 bottom-up (post-fusion) is structurally special -- the SAME k=11
    branch moved pre-fusion (P4 top-down) lost -0.36pp.

Four ideas, each keeping r6's proven k11@P4-BU branch UNTOUCHED where
possible (every "replace it" attempt in round 10 failed):

  1. r11_dual_p3p4_70 (~35-40%) -- ADD, don't replace. Keep r6's
     ZGLSKA[512,11] @ P4-BU exactly as-is, and ADD a second, INDEPENDENT
     ZGLSKA[256,7] @ P3 head (different feature map, own gamma). Round 7's
     same-scale stacking only cost -0.53pp; cross-scale stacking (different
     gradient paths) hasn't been tested and could be closer to additive.
     r6 trades away small-object mAP50 (56.34 vs baseline 61.79) for its
     P4 gains -- P3 is the natural place to recover that, and "other"
     (likely full of small/varied objects) might benefit too.

  2. r11_widefuse_70 (~30-35%) -- Direct fix to stripfuse's diagnosed
     failure cause. ZGLSKAWideFuse[512,11,23]: EXPAND first (pw1: c1->2c1)
     so k11 and strip23 each get a FULL c1-width stream (same width either
     gets operating alone), concat, pw2: 2c1->c1, one gamma. Same "combine
     the two best shapes" idea as stripfuse, without the channel-starvation
     that made it the worst round-10 result.

  3. r11_refine_70 (~25-30%) -- ZGLSKARefine[512,11,3]: k11's LKA branch is
     kept BYTE-FOR-BYTE as in r6, with one cheap depthwise k=3 "local
     refinement" pass appended after the attention output, before pw2 --
     same gamma, no parallel competition, no channel split.

  4. r11_expand_70 (~25%) -- ZGLSKAExpand[512,11,2]: same k=11 (the
     dose-response peak, which was fairly flat -- RF may not be the
     bottleneck), but pw1 projects c1->2c1 and the LKA operates on 2c1
     channels before pw2 projects back. Tests whether k11 is
     capacity-limited rather than RF-limited.

All four: pure arch, default TAL, gated identity-at-init, append-only
(standard Detect-remap loader). Bars: 78.45 baseline | 79.19 zg_p4_k11
(current best) | 79.06 r10_lskamultidil (best mAP50_95/recall/large).

Usage:
  python run_round11_arch_ideas.py
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
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

# 1: keep r6's k11 @ P4-BU exactly as-is, ADD an independent k7 @ P3 head
ARCH_DUAL_P3P4 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 11]]     # 21 — gated k11 @ P4 bottom-up (= r6, unchanged)
  - [14, 1, ZGLSKA, [256, 7]]      # 22 — NEW: independent gated k7 @ P3 head
  - [[22, 21, 20], 1, Detect, [nc]]
"""

# 2: wide fusion -- k11 + strip23, each on a FULL c1-width stream (fixes
#    round 10 ZGLSKAStripFuse's channel-starvation)
ARCH_WIDEFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 3: r6's k11 branch + one cheap depthwise k=3 local-refinement pass, same gamma
ARCH_REFINE = BASE_0_20 + """
  - [17, 1, ZGLSKARefine, [512, 11, 3]]  # 21 — gated k11 + local refine @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 4: same k=11, channel-expanded (c1 -> 2*c1 -> LKA(k=11) -> c1)
ARCH_EXPAND = BASE_0_20 + """
  - [17, 1, ZGLSKAExpand, [512, 11, 2]]  # 21 — gated capacity-expanded k11 @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r11_dual_p3p4_70",
     "desc": "[1/4] keep r6 k11@P4-BU + ADD independent k7 @ P3 head -- ~35-40% confidence",
     "yaml_content": ARCH_DUAL_P3P4, "batch": 48},
    {"name": "r11_widefuse_70",
     "desc": "[2/4] ZGLSKAWideFuse: k11+strip23, full-width streams (fixes stripfuse) -- ~30-35% confidence",
     "yaml_content": ARCH_WIDEFUSE, "batch": 40},
    {"name": "r11_refine_70",
     "desc": "[3/4] ZGLSKARefine: k11 + depthwise k3 local refinement, same gamma -- ~25-30% confidence",
     "yaml_content": ARCH_REFINE, "batch": 50},
    {"name": "r11_expand_70",
     "desc": "[4/4] ZGLSKAExpand: k11 with 2x channel-expanded branch -- ~25% confidence",
     "yaml_content": ARCH_EXPAND, "batch": 36},
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
    print(f"  70% ABLATION — ROUND 11: FOUR NEW ARCHITECTURE IDEAS")
    print(f"  Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best) | 79.06 multidil")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<20} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<20} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
