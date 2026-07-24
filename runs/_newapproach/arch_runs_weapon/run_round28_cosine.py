#!/usr/bin/env python3
"""
70% Ablation — Round 28: COSINE / PROTOTYPE classifier (change the scoring).

New angle: every prior head changed the FEATURES feeding the classifier or
suppressed outputs. But "other" is FOUND (recall ~0.84) and mis-SCORED (AP ~0.51)
-- the failure is how the linear classifier turns features into class scores.
This replaces the linear cls layer with a COSINE/prototype classifier:
    logit_c = scale * cos(feature, prototype_c) + bias_c
i.e. angular similarity to learnable class prototypes instead of a dot product.
Cosine scoring is better-calibrated and better-ranked for hard, visually-diverse
classes -- aimed directly at the "found but mis-scored" symptom. Architecture
only (cls layer), standard BCE loss.

Three runs (yolov12s/640, default TAL, batch 48):
  1. r28_cosine_stock_70        stock + DetectCosine (isolate the scoring change)
  2. r28_cosine_widefuse_70     widefuse + DetectCosine (best backbone)
  3. r28_decoupled_cosine_70    widefuse + DetectDecoupledCosine -- the combo:
     decoupled cls features (your one consistent signal) + cosine scoring.

Read precision + "other" AP50. Cosine heads are finicky -- if a run trains worse
than its linear counterpart, the scale (default 16) may need tuning; check the
first-epoch cls_loss is sane (not exploding). Seed-check any winner.

No new loss (cosine only changes the cls layer). SMOKE-TEST FIRST.

Bars: 79.40 r11_widefuse | r25_decoupled val 78.98 | r26_decoupledobj val 78.75.

Usage:
  python run_round28_cosine.py
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
BATCH = 50

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

# 1: stock + cosine cls head
ARCH_COSINE_STOCK = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectCosine, [nc]]    # 21 — cosine/prototype cls
"""

# 2: widefuse + cosine cls head
ARCH_COSINE_WIDEFUSE = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]   # 21 — widefuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectCosine, [nc]]     # 22 — cosine/prototype cls
"""

# 3: widefuse + decoupled cls pathway + cosine scoring (the combo)
#    box = [14, 21(widefuse), 20]; cls = [22,23,24]
ARCH_DECOUPLED_COSINE = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]   # 21 — widefuse @ P4 (box P4)
  - [14, 1, C3k2, [256, False]]               # 22 — cls_p3 (dedicated)
  - [21, 1, C3k2, [512, False]]               # 23 — cls_p4
  - [20, 1, C3k2, [1024, True]]               # 24 — cls_p5
  - [[14, 21, 20, 22, 23, 24], 1, DetectDecoupledCosine, [nc]]  # 25
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # {"name": "r28_cosine_stock_70",
    #  "desc": "[1/3] stock + DetectCosine (cosine/prototype cls -- isolate the scoring change)",
    #  "yaml_content": ARCH_COSINE_STOCK, "batch": BATCH, "seed": 0, "epochs": 80},
    # {"name": "r28_cosine_widefuse_70",
    #  "desc": "[2/3] widefuse + DetectCosine (best backbone + cosine scoring)",
    #  "yaml_content": ARCH_COSINE_WIDEFUSE, "batch": BATCH, "seed": 0, "epochs": 80},
    {"name": "r28_decoupled_cosine_70",
     "desc": "[3/3] widefuse + DetectDecoupledCosine -- COMBO: decoupled cls features + cosine scoring",
     "yaml_content": ARCH_DECOUPLED_COSINE, "batch": BATCH, "seed": 0, "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() then remap Detect keys if the index shifted. Backbone + neck
    + box branch transfer; the cosine cls prototypes train fresh."""
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
            **TAL_DEFAULT,
            alpha_start=0.0,
            alpha_end=0.0,
            alpha_min=0.0,
            alpha_max=0.0,
            iou_clip_start=999.0,
            iou_clip_end=999.0,
            dfl_clip_start=999.0,
            dfl_clip_end=999.0,
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
    print(f"  70% ABLATION — ROUND 28: COSINE/PROTOTYPE classifier")
    print(f"  Bars: 79.40 r11_widefuse | r25_decoupled val 78.98 | r26_decoupledobj 78.75")
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
