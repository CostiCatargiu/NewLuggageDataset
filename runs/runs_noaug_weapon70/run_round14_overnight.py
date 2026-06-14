#!/usr/bin/env python3
"""
70% Ablation — Round 14: OVERNIGHT BATCH (4 runs).

Context / evidence base:
  - r11_widefuse_70 (ZGLSKAWideFuse[512,11,23] @ P4 bottom-up) = 79.40% mAP50,
    current best pure architecture, mAP50_small=56.65, "other" AP50~54.10.
  - r6_zgp4_k11_70 (ZGLSKA[512,11] @ P4-BU) = 79.19%, "other" AP50~53.78.
  - Round 12 (small-object recovery, P3/backbone-level): r12_widefuse_ksmall_70
    = 78.71 (-0.69 vs widefuse), r12_widefuse_p3gc_70 = 78.29 (-1.11) -- both
    nudged mAP50_small up ~2pp over widefuse but cost overall mAP50 AND hurt
    "other" further. r12_dualwidefuse_70 (3rd idea, never run) is still
    pending -- included here as run 4.
  - Round 13 (head-level cls-branch gating, DetectLKACls = per-scale gated
    k=11 ZGLKA isolated to the cls branch only): r13_r6_lkacls_70 = 78.81
    (-0.38 vs r6), r13_widefuse_lkacls_70 = 78.54 (-0.86 vs widefuse) -- BOTH
    backbones got CONSISTENTLY WORSE (overall mAP50 down, mAP50_small down,
    "other" AP50 down). Likely cause: k=11's RF (~34, dilated) is too
    coarse/smoothing for the per-anchor cls decision -- diffuses exactly the
    local detail the classifier needs. r13_widefuse_cgc_70 (global P5
    context, cls-branch-only) was cancelled before running.

Four ideas this round, each targeting a DIFFERENT, previously-untested axis
(not a rehash of round 12's backbone-small-object or round 13's
cls-isolated-large-RF, both of which failed):

  1. r14_widefuse_gcfuse_70 (~25-30%) -- NEW module ZGLSKAGCFuse[512,11,8]
     @ P4-BU. WideFuse's two branches (k11 LKA + strip23 LSKA) are BOTH
     large-local-RF shapes -- never tested against a QUALITATIVELY different
     branch. This replaces the strip23 branch with a GCNet-style global
     context branch (z2 + gc_transform(ctx(z2))), full c1-width stream (same
     proven WideFuse structure), SHARED feature (affects both box and cls --
     unlike round 13's failed cls-only LKA injection). Tests: does global
     context help when it's part of the shared, proven fusion structure?

  2. r13_widefuse_cgc_70 (~20-25%, CARRIED OVER from round 13, cancelled
     before running) -- widefuse@P4-BU UNCHANGED (= 79.40) + DetectCGC
     (global P5-pooled context injected into the cls branch ONLY, gated).
     Paired directly with run 1: SAME underlying mechanism (global pooled
     context), two different injection points -- shared feature (run 1) vs
     cls-branch-only (this run). Round 13's cls-only LKA failed; this tests
     whether cls-only injection fails specifically because of LKA's local RF,
     or whether cls-only injection itself is the problem.

  3. r14_widefuse_p2fuse_70 (~30-35%) -- reuses ZGP2Fuse (implemented &
     registered early in the project, NEVER used in any run). Brings in the
     backbone's P2 feature (stride-4, the finest resolution available,
     128ch) as a zero-gated detail-injection into the Detect P3 branch:
     p3_det = P3_head + gamma * refine(down(P2)). widefuse@P4-BU stays
     UNCHANGED (= 79.40 winner). Append-only (P2/P3 already computed by
     layer 14/2; P4-BU downsample path still reads the original layer 14,
     so this purely enriches the P3 *prediction* with detail no other run
     has used -- a genuinely new information source for small "other"
     objects, not just an RF/context reshuffle of existing P3/P4 features).

  4. r12_dualwidefuse_70 (~20-25%, CARRIED OVER from round 12, never run) --
     widefuse@P4-BU UNCHANGED (= 79.40) + second ZGLSKAWideFuse[256,7,3] @ P3
     head (smaller, P3-appropriate kernels). Two independent gates -- the
     riskiest part, but both now use the proven fusion structure rather than
     round 11's bare-k7 dual_p3p4 (which cost "other" -1.88pp).

All four: pure arch, default TAL, gated identity-at-init, append-only
(standard Detect-remap loader -- backbone/FPN layers 0-20 unchanged, only
the Detect index shifts).

Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best).

Usage:
  python run_round14_overnight.py
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

# 1: WideFuse's strip23 branch -> GCNet-style global-context branch (NEW module)
ARCH_WIDEFUSE_GCFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAGCFuse, [512, 11, 8]]  # 21 — NEW: gated fuse k11 LKA + global-context @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: widefuse@P4-BU UNCHANGED (= 79.40 winner) + ZGP2Fuse (P2 detail -> Detect P3 branch)
ARCH_WIDEFUSE_P2FUSE = BASE_0_20 + """
  - [[14, 2], 1, ZGP2Fuse, []]              # 21 — NEW: gated P2(stride4) detail -> P3 Detect input
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 22 — gated wide-fuse k11+strip23 @ P4 (= r11 winner, unchanged)
  - [[21, 22, 20], 1, Detect, [nc]]
"""

# 2 (carried over from round 13): widefuse@P4-BU UNCHANGED + DetectCGC
#    (global P5 context, cls-branch-only) -- paired with run 1 (gcfuse)
ARCH_WIDEFUSE_CGC = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner, unchanged)
  - [[14, 21, 20], 1, DetectCGC, [nc]]      # 22 — gated global-context (P5-pooled) cls branch
"""

# 4: CARRIED OVER from round 12 -- widefuse@P4 (unchanged) + second wide-fuse @ P3
ARCH_DUALWIDEFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse k11+strip23 @ P4 (= r11 winner, unchanged)
  - [14, 1, ZGLSKAWideFuse, [256, 7, 3]]    # 22 — gated wide-fuse k7+strip3 @ P3 head
  - [[22, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r14_widefuse_gcfuse_70",
     "desc": "[1/4] NEW ZGLSKAGCFuse: k11 LKA + global-context (SHARED feature) @ P4-BU -- ~25-30% confidence",
     "yaml_content": ARCH_WIDEFUSE_GCFUSE, "batch": 38},
    {"name": "r13_widefuse_cgc_70",
     "desc": "[2/4] CARRIED OVER from round 13: widefuse@P4 (unchanged) + DetectCGC (global context, CLS-ONLY) -- paired with run 1 -- ~20-25% confidence",
     "yaml_content": ARCH_WIDEFUSE_CGC, "batch": 38},
    {"name": "r14_widefuse_p2fuse_70",
     "desc": "[3/4] widefuse@P4 (unchanged) + ZGP2Fuse (P2 detail -> Detect P3) -- ~30-35% confidence",
     "yaml_content": ARCH_WIDEFUSE_P2FUSE, "batch": 36},
    {"name": "r12_dualwidefuse_70",
     "desc": "[4/4] CARRIED OVER from round 12: widefuse@P4 (unchanged) + widefuse[256,7,3]@P3 -- ~20-25% confidence",
     "yaml_content": ARCH_DUALWIDEFUSE, "batch": 32},
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
    print(f"  70% ABLATION — ROUND 14: OVERNIGHT BATCH (4 runs)")
    print(f"  Bars: 78.45 baseline | 79.19 r6_zgp4_k11_70 | 79.40 r11_widefuse_70 (current best)")
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
