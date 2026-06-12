#!/usr/bin/env python3
"""
70% Ablation — Round 3: SMALL OBJECTS (mAP50_small recovery).

WHY ROUND 2 "HURT" SMALL — three findings from the test JSON:
  F1. mAP50_small is computed on ~82-113 test boxes -> ~5 missed dets = 4-5pts.
      Even pure TAL runs (zero arch change) spread 58.4% - 62.6% small.
  F2. CONFOUND: in all ZG runs Detect moved from index 21, so the PRETRAINED
      BOX-REGRESSION branch (cv2 + DFL) never transferred. Small objects are
      the most regression-sensitive -> this round fixes it with a key-remapped
      loader (Detect 21 -> N), so ZG runs are truly baseline-at-init.
  F3. Small at 640 means ~3px of P3 features. The strongest small-object
      lever is RESOLUTION, not blocks: at 960, a 25px object becomes ~37px.
      No new labels needed (374 small train instances can't feed new heads).

RUNS (all use load_pretrained_with_detect_remap, all DEFAULT TAL):
  1. base_960_70       — baseline arch @ 960. The resolution lever alone.
  2. zg_p4_960_70      — best round-2 arch (79.05%) @ 960 + detect fix.
  3. zg_p4_fixdet_70   — zg_p4 @ 640 with detect fix ONLY. If small returns
                         to ~61%, F2 confound is confirmed (key diagnostic).
  4. zg_p2fuse_70      — NEW: zero-gated P2->P3 detail fusion @ 640.
                         High-res backbone detail into P3 head, no P2 head.

REQUIRES: ZGLSKA, ZGGC, ZGP2Fuse registered in the fork (done).

Bars:
  640 runs : baseline 78.45% all / 61.79% small
  960 runs : compare ONLY against base_960_70 (resolution changes everything)

Usage:
  python run_arch_small_obj_round3.py
"""

import time
import gc
import os
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
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21  # Detect layer index in stock yolov12s

# =============================================================================
# Shared base = UNMODIFIED YOLOv12s layers 0-20
# =============================================================================
BASE_0_20 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2 — P2 backbone (160x160)
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

ARCH_BASELINE = BASE_0_20 + """
  - [[14, 17, 20], 1, Detect, [nc]]         # 21 — stock YOLOv12s
"""

# Round-2 winner: gated LSKA at P4 (79.05% @ 640)
ARCH_ZG_P4 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 7]]               # 21 — gated LSKA on P4
  - [[14, 21, 20], 1, Detect, [nc]]         # 22
"""

# NEW: zero-gated P2 detail fusion into P3 head (no P2 detection head)
ARCH_ZG_P2FUSE = BASE_0_20 + """
  - [[14, 2], 1, ZGP2Fuse, []]              # 21 — P2 (160x160) detail -> P3, gated
  - [[21, 17, 20], 1, Detect, [nc]]         # 22
"""

ARCH_RUNS = [
    # 1: resolution lever alone — the expected biggest small-object jump
    {"name": "base_960_70",
     "desc": "[1/4] Baseline @ 960 — resolution is the small-object lever",
     "yaml_content": ARCH_BASELINE, "imgsz": 960, "batch": 24},
    # 2: best arch + resolution + detect fix
    {"name": "zg_p4_960_70",
     "desc": "[2/4] ZG LSKA@P4 @ 960 + detect-remap",
     "yaml_content": ARCH_ZG_P4, "imgsz": 960, "batch": 22},
    # 3: diagnostic — isolates the Detect-transfer confound at 640
    {"name": "zg_p4_fixdet_70",
     "desc": "[3/4] ZG LSKA@P4 @ 640 + detect-remap — confound check",
     "yaml_content": ARCH_ZG_P4, "imgsz": 640, "batch": 54},
    # 4: new small-object architecture — P2 detail without a P2 head
    {"name": "zg_p2fuse_70",
     "desc": "[4/4] ZG P2->P3 fusion @ 640 + detect-remap",
     "yaml_content": ARCH_ZG_P2FUSE, "imgsz": 640, "batch": 50},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping Detect keys (model.21.* -> model.N.*)
    so the box branch (cv2) + DFL transfer even when Detect's index shifted.
    (cls branch cv3 won't transfer anyway: nc=80 vs nc=4 shape mismatch.)
    """
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    det_dst = len(model.model.model) - 1  # Detect is always the last layer
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    csd = {(pfx_dst + k[len(pfx_src):]) if k.startswith(pfx_src) else k: v
           for k, v in csd.items()}
    sd = model.model.state_dict()
    matched = intersect_dicts(csd, sd)
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Transferred {len(matched)}/{len(sd)} items "
          f"(Detect {DETECT_SRC_IDX} -> {det_dst})")
    return model


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# ImgSize: {run['imgsz']}   Batch: {run['batch']}   TAL: DEFAULT")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=80,
            imgsz=run["imgsz"],
            batch=run["batch"],
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            # --- DEFAULT TAL params (pure architecture/resolution effect) ---
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
    print(f"  70% ABLATION — ROUND 3: SMALL OBJECTS")
    print(f"  640 bars: 78.45% all / 61.79% small | 960 runs: compare to base_960_70")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80")
    print(f"{'=' * 70}")

    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i+1}] {run['name']:<18} imgsz={run['imgsz']}  batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Run {i+1}/{len(ARCH_RUNS)}: {run['name']}")
        results.append(run_arch_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<18} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
