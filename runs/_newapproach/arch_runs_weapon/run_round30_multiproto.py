#!/usr/bin/env python3
"""
Full Dataset — Round 30: MIXTURE (multi-prototype) classification head.

PROPOSAL / RATIONALE:
  The project ceiling is the catch-all "other" class: high recall (~0.84-0.89),
  low precision/AP -> found but mis-RANKED. Every prior change touched features
  (RF, routing, neck, scale, capacity) or supervision (aux, objectness); none
  touched the classifier's DECISION GEOMETRY. "other" is multimodal (it lumps
  many unrelated sub-types), but the stock head learns ONE hyperplane per class --
  which cannot rank a multimodal class cleanly. Round-28 cosine (ONE prototype)
  made it WORSE, i.e. the single-mode assumption IS the problem.

  Fix: MultiProtoHead -- K learnable sub-prototypes per class, combined by
  logsumexp (soft-OR: "matches ANY sub-prototype"). Lets "other" occupy several
  regions of feature space; unimodal weapon classes leave extra prototypes
  redundant. K=1 == stock head. Append-only, ~0 inference cost.

DESIGN: same widefuse backbone (= r11/r21) for all cells, DEFAULT TAL, loss
extras OFF -> isolate the HEAD effect. The control is the SAME backbone with a
plain Detect head (no aux), so control-vs-multiproto = pure cls-geometry effect.

  1. r30_widefuse_head_control — widefuse + plain Detect            (matched control)
  2. r30_multiproto_k4         — widefuse + DetectMultiProto K=4
  3. r30_multiproto_k3         — widefuse + DetectMultiProto K=3    (K sensitivity)

Read the "other" AP (esp. small-"other") and precision -- that is where, if the
mechanism is real, the gain must show. Full dataset, 90 ep, seed 0, FIXED batch.
Confirm any winner at 2-3 seeds; arch deltas live near the run-to-run noise band.

NOTE: the ceiling may be label-bound (not model-bound). If multiproto doesn't move
"other", that is itself evidence the residual gap is the annotations, not the head.

Usage:
  python run_round30_multiproto.py
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"   # FULL dataset
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 90
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 60          # same as the r21 full runs, shared across cells for clean comparison

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

WIDEFUSE = "  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]   # 21 — gated wide-fuse @ P4 (= r11)\n"

ARCH_CONTROL    = BASE_0_20 + WIDEFUSE + "  - [[14, 21, 20], 1, Detect, [nc]]                 # 22 — plain head (control)\n"
ARCH_MULTIPROTO_K4 = BASE_0_20 + WIDEFUSE + "  - [[14, 21, 20], 1, DetectMultiProto, [nc, 4]]    # 22 — mixture cls head, K=4\n"
ARCH_MULTIPROTO_K3 = BASE_0_20 + WIDEFUSE + "  - [[14, 21, 20], 1, DetectMultiProto, [nc, 3]]    # 22 — mixture cls head, K=3\n"

# DEFAULT TAL + every custom loss feature OFF -> isolate the architecture/head effect.
TAL_ARCH_ONLY = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    use_vfl=False,
)

RUNS = [
    # {"name": "r30_widefuse_head_control",
    #  "desc": "[1/3] widefuse + plain Detect (matched head control)",
    #  "yaml_content": ARCH_CONTROL},
    {"name": "r30_multiproto_k4",
     "desc": "[2/3] widefuse + DetectMultiProto K=4 (mixture classifier)",
     "yaml_content": ARCH_MULTIPROTO_K4},
    {"name": "r30_multiproto_k3",
     "desc": "[3/3] widefuse + DetectMultiProto K=3 (K sensitivity)",
     "yaml_content": ARCH_MULTIPROTO_K3},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() then remap Detect keys 21.* -> N.* if the index shifted
    (widefuse appends 1 layer, Detect 21 -> 22). Backbone + neck + box branch
    transfer via intersect_dicts; the mixture cls conv (nc*K channels) shape-
    mismatches and so trains fresh, exactly as intended."""
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
    print(f"# TAL: DEFAULT   Batch: {BATCH}   Epochs: {EPOCHS}   Seed: 0")
    print(f"{'#' * 70}\n")

    start_time = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            **TAL_ARCH_ONLY,
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
    print(f"  FULL DATASET — ROUND 30: MIXTURE (MULTI-PROTOTYPE) CLS HEAD")
    print(f"  Bar: r21_arch_full mAP50-95 = 53.67 (best mAP50-95 on full)")
    print(f"  Watch: 'other' AP and small-'other' AP -- where the gain must appear")
    print(f"{'=' * 70}")
    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<28} batch={BATCH}  {run['desc']}")
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
        print(f"  [{tag}] {r['name']:<28} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
