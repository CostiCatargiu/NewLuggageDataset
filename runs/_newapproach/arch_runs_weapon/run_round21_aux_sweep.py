#!/usr/bin/env python3
"""
70% Ablation — Round 21: EXPLOIT THE AUX HEAD (4 runs overnight).

r20_aux (DetectAux, train-only auxiliary head, aux_weight=0.25) is the first
architecture to land NOMINALLY ABOVE r11_widefuse on the fair (validation)
comparison: val mAP50 78.77 vs r11 78.38 vs baseline 77.84, and it generalizes
honestly (test 78.52 ~= val 78.77, vs r11's +1.0 lucky test-split gap). The edge
(+0.39) is still inside the ~+/-1pp noise floor, so this round pushes the one
lever that showed life, two ways:

  (A) aux_weight sweep -- if the aux effect is real, more aux supervision should
      strengthen it and reveal where it tops out; if it's noise, it scatters.
      r20 already covers 0.25; this adds 0.50 and 1.00.
  (B) stack on the best backbone -- the widefuse backbone (feature fusion) and
      the aux head (training signal) act on different things and have never been
      combined; they could be additive.

Four runs (all yolov12s/640, default TAL, FIXED batch for clean comparison):
  1. r21_aux_w50_70           stock neck + DetectAux, aux_weight=0.50
  2. r21_aux_w100_70          stock neck + DetectAux, aux_weight=1.00
  3. r21_widefuse_aux_70      widefuse @ P4 + DetectAux, aux_weight=0.25
  4. r21_widefuse_aux_w50_70  widefuse @ P4 + DetectAux, aux_weight=0.50

  With r20_aux (stock, 0.25) this gives the stock weight sweep {0.25,0.50,1.00}
  and the widefuse+aux stack {0.25,0.50}.

aux_weight is now a YAML arg on DetectAux: [nc, aux_weight]. The aux head is
dropped at inference (zero deploy cost); main head + backbone + neck transfer
from yolov12s.pt, aux towers fresh.

READ on per-class "other" AP50 + the VALIDATION mAP50 (test is split-noisy).
CONFIRM any winner at seeds 1,2 vs r11 at the same batch/split -- and confirm
the training split matches your other runs before crediting the architecture.

Bars: 78.45 baseline | 79.40 r11_widefuse | 80.45 v5_topk15_beta3 (loss-only)
      | r20_aux: val 78.77 / test 78.52.

Usage:
  python run_round21_aux_sweep.py
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
BATCH = 48  # fixed across all 4 for a clean comparison

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

# 1 & 2: stock neck + DetectAux at higher aux weights (Detect stays at index 21)
ARCH_AUX_W50 = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectAux, [nc, 0.5]]   # 21 — aux_weight=0.5
"""
ARCH_AUX_W100 = BASE_0_20 + """
  - [[14, 17, 20], 1, DetectAux, [nc, 1.0]]   # 21 — aux_weight=1.0
"""

# 3 & 4: widefuse @ P4-BU + DetectAux (aux at index 22)
ARCH_WIDEFUSE_AUX = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]    # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectAux, [nc, 0.25]]  # 22 — aux_weight=0.25
"""
ARCH_WIDEFUSE_AUX_W50 = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]    # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectAux, [nc, 0.5]]   # 22 — aux_weight=0.5
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # {"name": "r21_aux_w50_70",
    #  "desc": "[1/4] stock + DetectAux aux_weight=0.50 -- weight sweep point",
    #  "yaml_content": ARCH_AUX_W50, "batch": 50, "seed": 0, "epochs": 80},
    # {"name": "r21_aux_w100_70",
    #  "desc": "[2/4] stock + DetectAux aux_weight=1.00 -- weight sweep point (max)",
    #  "yaml_content": ARCH_AUX_W100, "batch": 50, "seed": 0, "epochs": 80},
    # {"name": "r21_widefuse_aux_70",
    #  "desc": "[3/4] widefuse @ P4 + DetectAux aux_weight=0.25 -- stack best backbone + aux",
    #  "yaml_content": ARCH_WIDEFUSE_AUX, "batch": 50, "seed": 0, "epochs": 80},
    {"name": "r21_widefuse_aux_w50_70_aug",
     "desc": "[4/4] widefuse @ P4 + DetectAux aux_weight=0.50 -- stack + stronger aux",
     "yaml_content": ARCH_WIDEFUSE_AUX_W50, "batch": 50, "seed": 0, "epochs": 80},
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
    print(f"  70% ABLATION — ROUND 21: AUX HEAD SWEEP + WIDEFUSE STACK (4 runs)")
    print(f"  Bars: 78.45 baseline | 79.40 r11_widefuse | r20_aux val 78.77")
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
