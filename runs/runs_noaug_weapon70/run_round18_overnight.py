#!/usr/bin/env python3
"""
70% Ablation — Round 18: OVERNIGHT BATCH (5 archs) aimed at the DIAGNOSED
bottleneck, not the exhausted P4 receptive-field space.

WHY THIS ROUND IS DIFFERENT
---------------------------
Rounds 1-17 (~30 variants) searched feature-side receptive field at P4/P3/P5
(LKA, strip, GC, multi-dil, routing, cls-input gating). NONE beat plain loss
tuning (best arch r11_widefuse=79.40 test, vs v5_topk15_beta3=80.45 with no
arch change), and validation showed even those leads carry ~+1pp test-vs-val
gaps (i.e. partly lucky splits).

Per-class analysis says WHY: the entire dataset ceiling is the "other" class,
and its failure mode is CLASSIFICATION, not localization --
    "other":  AR50 ~= 0.84 (recall)  vs  AP50 ~= 0.51 (precision)   gap ~0.33
    weapons:  AR50 ~= 0.96           vs  AP50 ~= 0.87               gap ~0.09
The detector FINDS "other" objects and mis-RANKS/mis-scores them. Every prior
module added LOCALIZATION context to a head that already localizes "other"
fine -- a category error repeated 30 times.

This round attacks the two bottlenecks the data actually points at:
  (A) CLS CAPACITY  -- deepen / widen the classifier tower (cv3) so it can
      separate the heterogeneous "other" class. Box branch (cv2) untouched.
  (B) SMALL-OBJECT SCALE -- add a real P2 (stride-4) detection head, the
      resolution where small-"other" detail still exists (every prior attempt
      reconstructed detail at stride-16 P4 and failed). Append-only: P3/P4/P5
      detect inputs (14/17/20) are UNCHANGED, only a P2 branch + 4-scale Detect
      are appended, so the pretrained FPN transfers in full.

This round tests the two levers and their interaction, P2-scale runs FIRST:

  1. r18_p2head_70            stock + P2 (stride-4) head, plain Detect, C3k2 @ P2
                              -- scale-only control
  2. r18_p2deepcls_70         stock + P2 head + DetectDeepCls -- THE smart one:
                              small-object RESOLUTION + cls CAPACITY together
                              (small-"other" fails on both: AR50_s~0.86, AP50_s~0.29)
  3. r18_deepcls_70           stock + DetectDeepCls (cv3: 2 -> 4 blocks)
  4. r18_widecls_70           stock + DetectWideCls (cv3: 2x width)
  5. r18_widefuse_deepcls_70  r11_widefuse backbone + DetectDeepCls

  Factorial decode: runs 1 (P2 only), 3 (deepcls only), 2 (P2 + deepcls) + the
  existing stock baseline attribute any gain to scale / cls-capacity / their
  interaction. Run 4 = depth-vs-width control; run 5 = cls capacity on the best
  backbone.

NOTE: unlike the gated ZGLSKA family these are NOT identity-at-init -- the new
cls tower / P2 branch train fresh (box + backbone + FPN still transfer from
pretrained). All: default TAL (pure architecture effect), append-only Detect
remap where the index shifts.

READ THE RESULT on the per-class "other" AP50 (all + small), not just mAP50_all
-- the weapon classes are already ~87% and dominate the overall metric, so even
a real "other" gain shows small overall. And run any winner at seeds 1,2 before
believing it: split variance here is ~+/-1pp, as large as the whole arch signal.

Bars: 78.45 baseline (other AP50 ~52, other-small ~38.6) | 79.40 r11_widefuse.

Usage:
  python run_round18_overnight.py
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

# 1: deeper cls tower on stock backbone (Detect stays at index 21)
ARCH_DEEPCLS = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectDeepCls, [nc]]  # 21 — cls tower 2 -> 4 blocks
"""

# 2: wider cls tower on stock backbone
ARCH_WIDECLS = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectWideCls, [nc]]  # 21 — cls tower 2x width
"""

# 3: r11_widefuse backbone + deeper cls tower (combine best backbone + cls capacity)
ARCH_WIDEFUSE_DEEPCLS = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]  # 21 — gated wide-fuse @ P4 (= r11, unchanged)
  - [[14, 21, 20], 1, DetectDeepCls, [nc]]  # 22 — deeper cls tower on top
"""

# P2: append-only P2 (stride-4) detection head on stock backbone.
#     P3/P4/P5 detect inputs (14/17/20) UNCHANGED -> full FPN transfer.
#     Lightweight C3k2 @ P2 (no area-attention at 160x160 -- cheaper and better
#     suited to the finest scale, where local detail beats global attention).
ARCH_P2HEAD = BASE_0_20 + """
  - [14, 1, nn.Upsample, [None, 2, "nearest"]]  # 21 — P3 -> stride 4
  - [[-1, 2], 1, Concat, [1]]                    # 22 — cat backbone P2 (layer 2)
  - [-1, 2, C3k2, [128, False]]                  # 23 — P2 detect feature (stride 4, lightweight)
  - [[23, 14, 17, 20], 1, Detect, [nc]]          # 24 — Detect P2,P3,P4,P5
"""

# P2 + deeper cls tower: small-object RESOLUTION (P2 scale) AND cls CAPACITY
# (deeper cv3) at once -- aimed at small-"other", which fails on BOTH axes
# (AR50_small~0.86 vs AP50_small~0.29: found but mis-ranked, at small scale too).
ARCH_P2DEEPCLS = BASE_0_20 + """
  - [14, 1, nn.Upsample, [None, 2, "nearest"]]  # 21 — P3 -> stride 4
  - [[-1, 2], 1, Concat, [1]]                    # 22 — cat backbone P2 (layer 2)
  - [-1, 2, C3k2, [128, False]]                  # 23 — P2 detect feature (stride 4, lightweight)
  - [[23, 14, 17, 20], 1, DetectDeepCls, [nc]]   # 24 — 4-scale Detect, deeper cls tower
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # P2 runs add a brand-new stride-4 scale + fresh 4-scale Detect head (the
    # 3-scale pretrained head cannot remap to 4 scales), so they retrain the most
    # fresh structure and need a LONGER horizon -- and the cosine LR schedule must
    # SPAN those epochs to anneal the new scale properly (not resume after 80).
    # The cls-tower runs only swap cv3 on a fully-transferred model -> 80 is plenty.
    {"name": "r18_p2head_70",
     "desc": "[1/5] stock + P2 (stride-4) head, plain Detect, lightweight C3k2 @ P2 -- scale-only control: does small-object resolution alone help?",
     "yaml_content": ARCH_P2HEAD, "batch": 28, "seed": 0, "epochs": 150},
    {"name": "r18_p2deepcls_70",
     "desc": "[2/5] stock + P2 head + DetectDeepCls -- THE smart one: small-object RESOLUTION + cls CAPACITY together (small-'other' fails on both: AR0.86/AP0.29)",
     "yaml_content": ARCH_P2DEEPCLS, "batch": 24, "seed": 0, "epochs": 150},
    {"name": "r18_deepcls_70",
     "desc": "[3/5] stock + DetectDeepCls (cls tower 2->4 blocks) -- cls capacity at normal scales; decomposes run 2 (scale vs cls vs interaction)",
     "yaml_content": ARCH_DEEPCLS, "batch": 48, "seed": 0, "epochs": 80},
    {"name": "r18_widecls_70",
     "desc": "[4/5] stock + DetectWideCls (cls tower 2x width) -- depth-vs-width ablation vs run 3",
     "yaml_content": ARCH_WIDECLS, "batch": 46, "seed": 0, "epochs": 80},
    {"name": "r18_widefuse_deepcls_70",
     "desc": "[5/5] r11_widefuse backbone + DetectDeepCls -- does cls capacity stack on the best backbone?",
     "yaml_content": ARCH_WIDEFUSE_DEEPCLS, "batch": 34, "seed": 0, "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted.

    For append-only archs (layers 0-20 unchanged, only Detect's index shifts).
    For the 4-scale P2 heads the pretrained 3-scale Detect mostly does not match
    by shape, so the Detect head trains fresh while backbone+FPN still transfer
    -- that is expected and standard for P2 variants."""
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
    print(f"# TAL: DEFAULT   Batch: {run['batch']}   Seed: {run['seed']}   Epochs: {run.get('epochs', 80)}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=run.get("epochs", 80),
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
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — ROUND 18: OVERNIGHT (5 archs, cls-capacity + P2 scale)")
    print(f"  Bars: 78.45 baseline | 79.40 r11_widefuse (best arch)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<26} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<26} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
