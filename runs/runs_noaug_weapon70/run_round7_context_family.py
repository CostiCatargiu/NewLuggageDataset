#!/usr/bin/env python3
"""
70% Ablation — Round 7: CONTEXT FAMILY @ P4 (extending the proven winner).

EVIDENCE BASE (all pure-arch, default TAL, test_full_dataset):
  gated LSKA@P4 k7   79.05 (+0.60)
  gated LSKA@P4 k11  79.19 (+0.74)  <- current best; +ALL classes; +50-95
  replace LSKA@P4    78.84 (+0.39)  -> gating worth ~+0.35
  gated GC@P5        78.89 (+0.44)
  Everything else (SE, MHSA, P2-fuse, CGC head, depth, P3 anything): <= ~0.
  Dose-response so far: more context at P4 = more gain.

FOUR RUNS, each testing a different axis of the winning mechanism
(spatial context @ P4). PURE ARCH — default TAL.
Bars: 78.45 baseline | 79.19 zg_p4_k11. Ordered by confidence.

  1. r7_dcn_70      — ZGDCN: zero-gated DEFORMABLE conv. Adaptive context:
                      the kernel learns WHERE to look per object (bends
                      along a diagonal rifle) instead of a fixed view.
                      Generalizes the whole kernel-size/shape search.
  2. r7_k15_70      — kernel dose-response completion (k7->k11->k15).
                      At P4's 40x40 grid, k15's RF (~47) is effectively
                      global: finds the optimum or the saturation point.
  3. r7_k11_gc4_70  — local (LSKA k11) + global (GC) context stacked at the
                      SAME scale. Previous combo failures were cross-scale.
  4. r7_strip23_70  — separable strip kernels 1x23+23x1 (ZGStrip, wraps the
                      proven conv.LSKA primitive). Shape hypothesis: strips
                      match elongated weapons better than square dilated.

REQUIRES: ZGStrip + ZGDCN registered in the fork (done) — copy block.py,
          modules/__init__.py, tasks.py again. ZGDCN uses torchvision's
          deform_conv2d when available (your DeformableConv2d handles the
          fallback automatically).

Usage:
  python run_round7_context_family.py
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

# 1: dose-response completion — k15 (effectively global at P4)
ARCH_K15 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 15]]              # 21 — gated LSKA @ P4, k=15
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: local + global context at the SAME scale (P4)
ARCH_K11_GC4 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 11]]              # 21 — gated LSKA @ P4, k=11
  - [-1, 1, ZGGC,   [512, 8]]               # 22 — gated global context @ P4
  - [[14, 22, 20], 1, Detect, [nc]]
"""

# 3: separable strip kernels — elongated-shape hypothesis
ARCH_STRIP23 = BASE_0_20 + """
  - [17, 1, ZGStrip, [512, 23]]             # 21 — gated strip LSKA 1x23+23x1 @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 4 -> 1: adaptive context — zero-gated deformable conv at P4
ARCH_DCN = BASE_0_20 + """
  - [17, 1, ZGDCN, [512, 3]]                # 21 — gated deformable conv @ P4
  - [[14, 21, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r7_dcn_70",
     "desc": "[1/4] gated deformable conv @ P4 — ADAPTIVE context (best odds)",
     "yaml_content": ARCH_DCN, "batch": 52},
    {"name": "r7_k15_70",
     "desc": "[2/4] gated LSKA @ P4, k=15 — dose-response completion",
     "yaml_content": ARCH_K15, "batch": 54},
    {"name": "r7_k11_gc4_70",
     "desc": "[3/4] LSKA k11 + GC stacked @ P4 — local+global, same scale",
     "yaml_content": ARCH_K11_GC4, "batch": 52},
    {"name": "r7_strip23_70",
     "desc": "[4/4] strip kernels 1x23+23x1 @ P4 — elongated-shape hypothesis",
     "yaml_content": ARCH_STRIP23, "batch": 54},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted."""
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
    print(f"  70% ABLATION — ROUND 7: CONTEXT FAMILY @ P4 (PURE ARCH)")
    print(f"  Bars: 78.45 baseline | 79.19 zg_p4_k11 (current best)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<14} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<14} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
