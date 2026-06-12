#!/usr/bin/env python3
"""
70% Ablation — Round 4: THE 'other' CLASS (the real bottleneck).

FINDING (from runs_noaug_weapon70__test_full_dataset.json, every run):
  pistol ~0.89 | long_gun ~0.88 | knife ~0.86 | other ~0.53-0.56  AP50
  'other' alone costs ~8 points of mAP50_all. Lifting it 0.53 -> 0.70
  is worth +4 points — ~7x the entire architecture ceiling (+0.6).
  NO previous run targeted it.

This round: stock architecture, attack via loss/recipe only.
All runs start from the proven best recipe (TAL topk15/beta3 = 80.45%)
so gains stack on the best known baseline, and the bar is 80.45%.

RUNS:
  1. r4_cls_weight_70   — stronger classification loss (cls 0.5 -> 1.5).
                          'other' errors are largely classification
                          confusion; box branches are already strong.
  2. r4_vfl_70          — VariFocal loss (use_vfl, in your fork, never
                          tested in this sweep). Down-weights easy
                          negatives -> hard/ambiguous 'other' samples
                          get relatively more gradient.
  3. r4_long_cos_70     — same recipe, 140 epochs + cosine LR. Rare
                          classes/hard examples benefit most from
                          longer schedules.
  4. r4_satal_70        — your own size-adaptive TAL (use_satal=True,
                          also never tested): relaxed assignment for
                          small boxes; 'other' has the worst small AP
                          (~0.30 vs ~0.70 for pistol).

BEFORE TRAINING — 30 minutes that may be worth more than all 4 runs:
  Open ~30 test images where 'other' fails (lowest-conf or missed GT).
  If 'other' is a grab-bag of inconsistent objects, the fix is taxonomy
  (split/clean the class), and no loss function will recover it.

Bar: 80.45% mAP50_all — but WATCH per-class 'other' AP50 (target: >0.60).

Usage:
  python run_round4_other_class.py
"""

import time
import gc
import os
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640

# Stock YOLOv12s (Detect at 21 -> full pretrained transfer incl. box branch)
ARCH_BASELINE = """nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[14, 17, 20], 1, Detect, [nc]]
"""

# Best proven recipe — base for every run this round (80.45% reference)
TAL_BEST = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)

RUNS = [
    # 1: classification capacity — 'other' is a cls problem, not a box problem
    {"name": "r4_cls15_70",
     "desc": "[1/4] cls loss weight 0.5 -> 1.5 on best TAL recipe",
     "extra": dict(cls=1.5), "epochs": 80},
    # 2: VariFocal — hard-example-aware classification (untested fork knob)
    {"name": "r4_vfl_70",
     "desc": "[2/4] VariFocal loss on best TAL recipe",
     "extra": dict(use_vfl=True), "epochs": 80},
    # 3: longer schedule + cosine — rare/hard classes gain most
    {"name": "r4_long_cos_70",
     "desc": "[3/4] 140 epochs + cosine LR on best TAL recipe",
     "extra": dict(cos_lr=True), "epochs": 140},
    # 4: your size-adaptive TAL — never tested; 'other' small AP is worst
    {"name": "r4_satal_70",
     "desc": "[4/4] SATAL (size-adaptive assignment) on best TAL recipe",
     "extra": dict(use_satal=True), "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, "r4_baseline.yaml")
    save_yaml(ARCH_BASELINE, yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Epochs: {run['epochs']}   Extra: {run['extra']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")  # stock arch -> standard load, full transfer

        model.train(
            data=DATA_YAML,
            epochs=run["epochs"],
            imgsz=IMG_SIZE,
            batch=56,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            # --- BEST TAL recipe (reference 80.45%) ---
            **TAL_BEST,
            # --- per-run lever ---
            **run["extra"],
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
            # --- DISABLE other custom features (unless the run enables one) ---
            small_obj_boost=1.0,
            small_obj_px=0,
            center_loss_weight_init=0.0,
            center_loss_weight_min=0.0,
            **({} if "use_vfl" in run["extra"] else dict(use_vfl=False)),
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
    print(f"  70% ABLATION — ROUND 4: 'other' CLASS (AP50 ~0.53 vs ~0.88 rest)")
    print(f"  Base recipe: TAL topk15/beta3 | Bar: 80.45% all, 'other' > 0.60")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<16} epochs={run['epochs']:<4} {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<16} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
