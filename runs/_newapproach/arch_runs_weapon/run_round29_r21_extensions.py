#!/usr/bin/env python3
"""
70% Ablation — Round 29: extend the project-best architecture (r21).

Best arch so far = r21_widefuse_aux_w50 (ZGLSKAWideFuse[512,11,23] @ P4-BU +
train-only DetectAux @ 0.5): test mAP50 79.57 / mAP50-95 50.33. Its one clear
weakness is small-"other" (AP50 0.265, recall 0.889 -> a ranking problem, not a
findability one). This round extends r21 along two orthogonal, never-combined axes
and tests whether they stack:

  B (kernel fix):  swap the widefuse second branch strip-23 -> strip-3 so the
                   fused P4 branch covers two GENUINELY different scales (k11
                   RF~35 + strip-3 RF~11) instead of two redundant large RFs.
                   Round-12 finding, but it was only ever tried on a plain Detect
                   backbone -- never on the AUX backbone that actually won.

  A (decoupled+aux): give classification its OWN feature pathway (DetectDecoupled,
                   round 24 -- the one reproducible precision effect) WHILE keeping
                   r21's train-only aux head. widefuse has been paired with aux
                   (r21), with decoupled (r25), and with decoupled+obj (r26) -- but
                   never aux AND decoupled together (no head class did both).
                   New head: DetectDecoupledAux (subclass of DetectAux -> uses
                   DetectAuxLoss; aux dropped at inference, zero deploy cost).

Four cells, fixed batch 48 (= r21/r25 for comparability), default TAL, aux 0.5,
seed 0, 80 epochs:

  1. r29_r21_control          = r21 reproduced (control on THIS batch/seed)
  2. r29_kernelfix            = B
  3. r29_decoupled_aux        = A
  4. r29_kernelfix_decoupled  = A + B

Read mAP50-95 (more split-stable than mAP50) and the small-"other" AP. Confirm any
winner at 2-3 seeds before believing it -- the architecture deltas in this project
sit inside the ~+-1pt run-to-run noise band.

Usage:
  python run_round29_r21_extensions.py
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
DETECT_SRC_IDX = 21          # Detect index in stock yolov12s
BATCH = 50                   # same as r21/r25 for a clean comparison
AUX_W = 0.5                  # r21's winning aux weight

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

# -----------------------------------------------------------------------------
# 1) CONTROL — r21 reproduced: widefuse[11,23] @ P4 + DetectAux @ 0.5
# -----------------------------------------------------------------------------
ARCH_CONTROL = BASE_0_20 + f"""
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]       # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectAux, [nc, {AUX_W}]]  # 22 — train-only aux (= r21)
"""

# -----------------------------------------------------------------------------
# 2) B — kernel fix: widefuse second branch strip-23 -> strip-3 (multi-scale)
# -----------------------------------------------------------------------------
ARCH_KERNELFIX = BASE_0_20 + f"""
  - [17, 1, ZGLSKAWideFuse, [512, 11, 3]]        # 21 — k11 + strip-3 (genuinely multi-scale)
  - [[14, 21, 20], 1, DetectAux, [nc, {AUX_W}]]  # 22 — train-only aux
"""

# -----------------------------------------------------------------------------
# 3) A — decoupled cls pathway + aux: widefuse[11,23] @ P4 + DetectDecoupledAux
#    box = [14, 21(widefuse P4), 20]; cls = [22, 23, 24] (dedicated C3k2)
# -----------------------------------------------------------------------------
ARCH_DECOUPLED_AUX = BASE_0_20 + f"""
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]   # 21 — gated wide-fuse @ P4 (box P4)
  - [14, 1, C3k2, [256, False]]               # 22 — cls_p3 (dedicated)
  - [21, 1, C3k2, [512, False]]               # 23 — cls_p4 (from widefuse output)
  - [20, 1, C3k2, [1024, True]]               # 24 — cls_p5 (dedicated)
  - [[14, 21, 20, 22, 23, 24], 1, DetectDecoupledAux, [nc, {AUX_W}]]  # 25
"""

# -----------------------------------------------------------------------------
# 4) A + B — kernel fix AND decoupled cls pathway + aux
# -----------------------------------------------------------------------------
ARCH_KERNELFIX_DECOUPLED = BASE_0_20 + f"""
  - [17, 1, ZGLSKAWideFuse, [512, 11, 3]]    # 21 — k11 + strip-3 (box P4, kernel fix)
  - [14, 1, C3k2, [256, False]]               # 22 — cls_p3 (dedicated)
  - [21, 1, C3k2, [512, False]]               # 23 — cls_p4 (from widefuse output)
  - [20, 1, C3k2, [1024, True]]               # 24 — cls_p5 (dedicated)
  - [[14, 21, 20, 22, 23, 24], 1, DetectDecoupledAux, [nc, {AUX_W}]]  # 25
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # {"name": "r29_r21_control",
    #  "desc": "[1/4] CONTROL: widefuse[11,23] + DetectAux@0.5 (= r21, this batch/seed)",
    #  "yaml_content": ARCH_CONTROL, "batch": BATCH, "seed": 0, "epochs": 80},
    {"name": "r29_kernelfix",
     "desc": "[2/4] B: widefuse[11,3] (multi-scale fix) + DetectAux@0.5",
     "yaml_content": ARCH_KERNELFIX, "batch": 50, "seed": 0, "epochs": 80},
    # {"name": "r29_decoupled_aux",
    #  "desc": "[3/4] A: widefuse[11,23] + DetectDecoupledAux@0.5 (decoupled cls + aux)",
    #  "yaml_content": ARCH_DECOUPLED_AUX, "batch": 48, "seed": 0, "epochs": 80},
    # {"name": "r29_kernelfix_decoupled",
    #  "desc": "[4/4] A+B: widefuse[11,3] + DetectDecoupledAux@0.5",
    #  "yaml_content": ARCH_KERNELFIX_DECOUPLED, "batch": BATCH, "seed": 0, "epochs": 80},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first, then remap Detect keys model.21.* -> model.N.* if the
    index shifted. Transfers backbone + neck + the box branch (matching shapes via
    intersect_dicts); the dedicated cls pathway and the aux towers train fresh.
    Generic: det_dst is computed from the built model, so it handles both the
    +1-layer (control/B, Detect at 22) and +5-layer (A/A+B, Detect at 25) cases."""
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
            # --- DEFAULT TAL (isolate the architecture effect) ---
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
            # --- DISABLE custom loss features (pure arch) ---
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
    print(f"  70% ABLATION — ROUND 29: r21 EXTENSIONS (kernel fix x decoupled+aux)")
    print(f"  Bar to beat: r21_widefuse_aux_w50 = test 79.57 / mAP50-95 50.33")
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
