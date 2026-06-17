#!/usr/bin/env python3
"""
70% Ablation — Round 23: BEST ARCH x BEST TAL (the combination never tried).

The whole search separated two levers:
  - Loss/TAL tuning (arch frozen): best = v5_topk15_beta3, test 80.45 / val 79.40
    and v5_tal07, test 80.02 / val 79.54 -- the ONLY lever that ever beat baseline.
  - Architecture (TAL frozen at DEFAULT): best = widefuse + aux@0.5 (round 21/22),
    nominally the best arch, but only ever on default TAL (topk10/a0.5/b6.0).

They have NEVER been multiplied. This round runs the best architecture
(widefuse + DetectAux@0.5) under the two best TAL configs. If the arch's
contribution stacks even partially on the loss-tuning gain, this is the run that
clears 80.45.

Two runs (widefuse + aux@0.5, yolov12s/640, fixed batch 48):
  1. r23_waux_topk15b3_70   TAL topk=15, alpha=0.5, beta=3.0  + cls=1.2  (= v5_topk15_beta3 TAL)
  2. r23_waux_tal07_70      TAL topk=13, alpha=0.7, beta=4.0  + cls=1.2  (= v5_tal07 TAL)

NOTE: applies the TAL params + cls weighting (the core of the best config).
The epoch-scheduled features (iou/dfl clips, alpha schedule, small-obj boost)
are kept OFF for clean integration with the aux-head loss wrapper -- if these
land below the TAL-only baselines, the scheduled features may be needed and we
add them next.

Bars: 80.45 v5_topk15_beta3 | 80.02 v5_tal07 | 79.57 widefuse_aux@0.5 (default TAL).

Usage:
  python run_round23_aux_besttal.py
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
BATCH = 48

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

# best arch: widefuse @ P4 + DetectAux @ 0.5
ARCH = BASE_0_20 + """
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]     # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectAux, [nc, 0.5]]    # 22 — aux_weight=0.5
"""

RUNS = [
    {"name": "r23_waux_topk15b3_70",
     "desc": "[1/2] widefuse+aux@0.5  x  TAL topk15/a0.5/b3.0 + cls1.2 (= v5_topk15_beta3 TAL, best 80.45)",
     "tal": dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0), "cls": 1.2,
     "batch": BATCH, "seed": 0, "epochs": 80},
    {"name": "r23_waux_tal07_70",
     "desc": "[2/2] widefuse+aux@0.5  x  TAL topk13/a0.7/b4.0 + cls1.2 (= v5_tal07 TAL, robust 80.02)",
     "tal": dict(tal_topk=13, tal_alpha=0.7, tal_beta=4.0), "cls": 1.2,
     "batch": BATCH, "seed": 0, "epochs": 80},
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
    save_yaml(ARCH, yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: {run['tal']}  cls={run['cls']}  Batch: {run['batch']}  Seed: {run['seed']}  Epochs: {run.get('epochs', 80)}")
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
            # --- BEST TAL (per run) + cls weighting ---
            **run["tal"],
            cls=run["cls"],
            # --- epoch-scheduled features OFF (clean integration with aux loss) ---
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
    print(f"  70% ABLATION — ROUND 23: BEST ARCH x BEST TAL (widefuse+aux@0.5)")
    print(f"  Bars: 80.45 v5_topk15_beta3 | 80.02 v5_tal07 | 79.57 widefuse_aux@0.5")
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
