#!/usr/bin/env python3
"""
70% Ablation -- Round 35 BEST-4: the strongest cells merged from both proposals.

Built on R34 (best small-obj arch: mAP50 82.36 / 50-95 52.28 / small 66.07).
Compare also to R32B (best overall: 82.58 / 52.41 / small 64.73).

Design rules confirmed R21->R34:
  - Enhancement on MAIN, raw feature anchors the AUX (R34 > R33).
  - WideFuseV2 is the best enhancer (R32B >> R21).
  - ZGSmallDetail helps small but hurts large -> use a CONTEXT block at P5, not detail.
  - Dual-path supervision is essential (R32B >> R21).

THE 4 CELLS (one per distinct hypothesis, no redundancy), ordered by confidence:

  1. r35_wfv2_p3     -- P3 enhancement = WideFuseV2[256,7,15,3,5] (context+detail
       hybrid, smaller kernels for P3's resolution) instead of detail-only.
       Fixes the other-large dip that the plain detail block caused. HIGHEST conf.
  2. r35_p5context   -- complete the symmetric dual-path: add ZGLSKAWideFuse@P5
       (a CONTEXT block, correct for the large-object level), aux anchors raw P5.
       Recovers R34's lost other-medium/large.
  3. r35_multiproto  -- R34 neck + DetectMultiProto[nc,3] main head (K=3 mixture
       classifier, no aux). The ONLY cell attacking "other" overall (the 25-pt
       gap). High upside / higher variance (drops the dual-path aux).
  4. r35_r34_aux075  -- exact R34, aux_weight 0.5 -> 0.75. Cheap control: does a
       stronger raw anchor recover other-breadth with no structural change?

All: default TAL / batch 48 / 80 ep / seed 0 / same data -> drop straight into the
r32b-r33-r34 comparison. Each run is wrapped in try/except (one failure != abort).
All modules verified registered (ZGLSKAWideFuse/V2, ZGSmallDetail, DetectAuxDual,
DetectMultiProto are imported + handled in parse_model).

SMOKE-TEST first:
  for n in r35_wfv2_p3_70 r35_p5context_70 r35_multiproto_70 r35_r34_aux075_70; do
    python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/${n}.yaml'); \
      print('$n', type(m.model.model[-1]).__name__, m.model.stride.tolist())"; done

Usage:
  python run_round35_best4.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION (identical to r32b/r33/r34 so the arch comparison is clean)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
PROJECT_DIR = "runs_noaug_weapon_70_review"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48
EPOCHS = 80

# Shared YOLOv12s backbone + head (layers 0-20). 14=P3 head, 17=P4 bottom-up, 20=P5.
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (raw; aux anchor)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (raw; aux anchor)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

# 1. WideFuseV2 @ P3 (context+detail hybrid) + WideFuseV2 @ P4
ARCH_WFV2_P3 = BASE_0_20 + """  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- WideFuseV2 @ P4 (main)
  - [14, 1, ZGLSKAWideFuseV2, [256, 7, 15, 3, 5]]    # 22 -- WideFuseV2 @ P3 (main; small kernels)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 23
"""

# 2. r34 + CONTEXT block @ P5 (complete symmetric dual-path)
ARCH_P5CONTEXT = BASE_0_20 + """  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- WideFuseV2 @ P4 (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail @ P3 (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- context @ P5 (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]  # 24
"""

# 3. r34 neck + mixture-classifier main head (attack "other"; no aux)
ARCH_MULTIPROTO = BASE_0_20 + """  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- WideFuseV2 @ P4 (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail @ P3 (main)
  - [[22, 21, 20], 1, DetectMultiProto, [nc, 3]]     # 23 -- mixture cls head, K=3
"""

# 4. exact r34, aux_weight 0.75 (control)
ARCH_R34_AUX075 = BASE_0_20 + """  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- WideFuseV2 @ P4 (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- detail @ P3 (main)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.75]]  # 23 -- aux_weight 0.75
"""

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# Ordered by confidence (most likely to build AND beat r34 first):
RUNS = [
    # {"name": "r35_wfv2_p3_70",    "yaml": ARCH_WFV2_P3,
    #  "desc": "[1/4] WideFuseV2 @ P3 (context+detail) + WideFuseV2 @ P4  -- highest conf"},
    # {"name": "r35_p5context_70",  "yaml": ARCH_P5CONTEXT,
    #  "desc": "[2/4] complete symmetric dual-path: + ZGLSKAWideFuse(context) @ P5"},
    # {"name": "r35_multiproto_70", "yaml": ARCH_MULTIPROTO,
    #  "desc": "[3/4] DetectMultiProto[nc,3] main head -- attack 'other' (high upside)"},
    {"name": "r35_r34_aux075_70", "yaml": ARCH_R34_AUX075,
     "desc": "[4/4] exact r34, aux_weight 0.75 -- cheap control"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Transfer backbone+neck (0-20); new blocks + head train fresh; remap Detect index."""
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} keys transferred")
    return model


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if getattr(trainer, "criterion", None) is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, "_sync_bbox_loss_state"):
                trainer.criterion._sync_bbox_loss_state()
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
    except Exception:
        pass


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml"], yaml_path)
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# Batch {BATCH}  Epochs {EPOCHS}  default TAL  seed 0\n{'#' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=0, deterministic=True)
        kw.update(DEFAULT_TAL)
        model.train(**kw)
        elapsed = (time.time() - start) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 80}\n  ROUND 35 BEST-4 ON R34  (default TAL, arch-only)")
    print(f"  vs r34 (82.36 / 52.28 / small 66.07) and r32b (82.58 / 52.41 / 64.73)")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")

    results = [run_experiment(r) for r in RUNS]

    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<20} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
