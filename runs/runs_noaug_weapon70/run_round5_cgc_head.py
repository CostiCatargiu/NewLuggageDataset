#!/usr/bin/env python3
"""
70% Ablation — Round 5: HEAD ARCHITECTURE (DetectCGC).

THE ARGUMENT FOR A HEAD ARCH (from all previous rounds):
  - Backbone/neck capacity is saturated: 20 designs, ceiling +0.6% mAP50.
  - The measured weakness is CLASSIFICATION: 'other' AP50 ~0.53 vs ~0.88
    for the named classes, while box recall is ~0.95 everywhere.
  - The Detect head is the ONLY stock component never modified.

DetectCGC = Detect + Context-Gated Classification:
  global P5 context (softmax attention pooling) -> MLP -> per-scale
  projection -> ZERO-GATED injection into the cls-branch input only.
  Box branch untouched. gamma=0 at init -> exact stock Detect at epoch 0,
  pretrained box-branch weights transfer normally with the stock body.

Thesis narrative: Zero-Gated adaptation in the NECK (ZGLSKA@P4, +0.6 pure
arch) + Zero-Gated context in the HEAD (DetectCGC) — one framework, with
gate magnitudes as interpretability.

REQUIRES: DetectCGC registered in the fork (nn/modules/head.py,
          nn/modules/__init__.py, nn/tasks.py) — done.

PURE ARCHITECTURE ROUND — ALL runs use DEFAULT TAL params, so every delta
vs 78.45% is attributable to the architecture alone. TAL combination comes
in a follow-up round ONLY for whichever architecture wins here.

RUNS:
  1. r5_cgc_70       — stock body + DetectCGC.
                       Bars: 78.45% (baseline), 79.05% (zg_p4, arch to beat).
  2. r5_zgp4_cgc_70  — ZGLSKA@P4 + DetectCGC (best neck + new head;
                       uses detect-remap loader since Detect index = 22).

WATCH per run: mAP50_all AND per-class 'other' AP50 (success = clearly >0.56),
plus ctx_gamma magnitudes in best.pt (did the head use the context?).

Usage:
  python run_round5_cgc_head.py
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

# Stock body + CGC head (Detect stays at index 21 -> standard transfer)
ARCH_CGC = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectCGC, [nc]]      # 21 — Context-Gated Cls head
"""

# Full system: ZG LSKA@P4 (best neck, +0.6) + CGC head (Detect at 22 -> remap)
ARCH_ZGP4_CGC = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 7]]               # 21 — gated LSKA on P4
  - [[14, 21, 20], 1, DetectCGC, [nc]]      # 22 — Context-Gated Cls head
"""

# PURE ARCH ROUND: default TAL everywhere. (Best-TAL combination is a
# follow-up round, winners only.)
TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # 1: pure head-arch effect — must beat 78.45 (and ideally 79.05 = zg_p4)
    {"name": "r5_cgc_70",
     "desc": "[1/2] DetectCGC, default TAL — pure head-arch effect",
     "yaml_content": ARCH_CGC, "tal": TAL_DEFAULT, "extra": {}, "batch": 54},
    # 2: arch combo — best neck (+0.6) + new head, still default TAL
    {"name": "r5_zgp4_cgc_70",
     "desc": "[2/2] ZGLSKA@P4 + DetectCGC, default TAL — arch combo",
     "yaml_content": ARCH_ZGP4_CGC, "tal": TAL_DEFAULT, "extra": {}, "batch": 52},
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
    print(f"# TAL: {run['tal']}   Extra: {run['extra']}   Batch: {run['batch']}")
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
            **run["tal"],
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
    print(f"  70% ABLATION — ROUND 5: DetectCGC HEAD (PURE ARCH, default TAL)")
    print(f"  Bars: 78.45 baseline | 79.05 zg_p4 | 'other' AP50 > 0.56")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<22} batch={run['batch']}  {run['desc']}")

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
        print(f"  [{tag}] {r['name']:<22} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
