#!/usr/bin/env python3
"""
70% Ablation -- Round 34: R32B + DETAIL-ENHANCED P3 ON THE *MAIN* HEAD

Single architecture run (default TAL only) — the controlled inversion of R33.

WHAT R32B / R33 ESTABLISHED:
  - r32b WINS by dual-path supervision where the MAIN head consumes the
    *enhanced* P4 (widefuse, layer 21) and the AUX head anchors the *raw* P4
    (layer 17). +1.12 mAP50, +2.93 "other" AP50 vs r21 at the same loss.
  - R33 added a detail block at P3 but fed it to the AUX head (dropped at
    inference). Result was a WASH: it redistributed "other" toward small
    (+3.69 other-small) but lost other-overall and overall mAP50, because the
    enhancement it forced was never available to the inference (main) head.

THE RULE THE DATA POINTS TO:
  Enhancement belongs on the MAIN (inference-available) path; the RAW feature
  is the AUX (training-only) anchor. r32b followed this at P4. R33 violated it
  at P3. R34 fixes it: put the SAME ZGSmallDetail P3 block on the MAIN head,
  and let the aux anchor the raw P3.

ARCHITECTURE (only the head tail differs from r33 — the DetectAuxDual order is
swapped so the detail-enhanced P3 goes to MAIN, raw P3 goes to AUX):
  layer 21: ZGLSKAWideFuseV2[512,11,23,3,5]  @ P4 (= r32b)
  layer 22: ZGSmallDetail[256,3,5]           @ P3 (detail-enhanced, zero-gated)
  layer 23: DetectAuxDual[nc,0.5]
            main = [22 enhanced-P3, 21 widefuse-P4, 20 P5]   <- inference path
            aux  = [14 raw-P3,      17 raw-P4,      20 P5]    <- training anchor

  vs R33:   main = [14 raw-P3, 21 P4, 20 P5] ; aux = [22 detail-P3, 17 P4, 20 P5]
  -> r34 is exactly R33 with the P3 detail moved from aux to main. Single-variable
     ablation of "which side should the P3 enhancement live on".

ZGSmallDetail is zero-gated (gamma=0 at init) -> identity at epoch 0, clean
pretrained transfer. At inference the aux towers drop; the main P3 detail block
stays (it's on the inference path) -> small extra cost only at P3.

PREDICTION: keep R33's small / other-small gain (now inference-available) WITHOUT
R33's loss of other-overall / overall mAP50 — i.e. >= r32b's 82.58 mAP50 while
recovering other-small.

SMOKE-TEST FIRST:
  python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/r34_auxdual_p3main_arch_only_70.yaml'); \
             print(type(m.model.model[-1]).__name__, m.model.stride.tolist())"
  -> expect  DetectAuxDual [8.0, 16.0, 32.0]

Usage:
  python run_round34_auxdual_p3main.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION  (identical to r32b/r33 so the arch comparison is clean)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
PROJECT_DIR = "runs_noaug_weapon_70_review"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 52
EPOCHS = 80

# =============================================================================
# ARCHITECTURE: WideFuseV2 @ P4 + ZGSmallDetail @ P3 (on MAIN) + DetectAuxDual
# =============================================================================
ARCH_YAML_CONTENT = """nc: 4
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (RAW; aux anchor)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (RAW; aux anchor)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]  # 21 -- widefuse-enhanced P4 (MAIN)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail-enhanced P3 (MAIN, zero-gated)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 23
"""
# DetectAuxDual inputs (6 total):
#   Main (inference): P3_detail=22(256ch), P4_fused=21(512ch), P5=20(1024ch)
#   Aux  (train-only): P3_raw=14(256ch),   P4_prefuse=17(512ch), P5=20(1024ch)

# =============================================================================
# TRAINING PARAMETERS -- DEFAULT TAL ONLY (arch comparison, no loss tuning)
# =============================================================================
DEFAULT_TAL_PARAMS = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {
        "name": "r34_auxdual_p3main_arch_only_70",
        "desc": "R34 (AuxDual + P3 detail on MAIN) with DEFAULT TAL — arch only",
        "yaml_content": ARCH_YAML_CONTENT,
        "training_params": DEFAULT_TAL_PARAMS,
    },
]


# =============================================================================
# HELPERS
# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights and remap Detect keys if index shifted."""
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
    run_name = run["name"]
    yaml_path = os.path.join(YAML_DIR, f"{run_name}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 80}")
    print(f"# {run_name}")
    print(f"# {run['desc']}")
    print(f"# TAL: {run['training_params']}")
    print(f"{'#' * 80}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run_name,
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
            **run["training_params"],
        )

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run_name} ({elapsed:.2f}h)")
        return {"name": run_name, "status": "OK", "time": elapsed}

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run_name} ({elapsed:.2f}h) -- {e}")
        import traceback
        traceback.print_exc()
        return {"name": run_name, "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"  70% ABLATION -- ROUND 34: AUXDUAL + P3 DETAIL ON MAIN (arch only)")
    print(f"  Inversion of R33: P3 detail moved from AUX -> MAIN (inference path)")
    print(f"  Compare to: r32b (82.58 mAP50) and r33 (P3 detail on aux)")
    print(f"{'=' * 80}\n")

    results = [run_experiment(r) for r in RUNS]

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 80}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
