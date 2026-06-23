#!/usr/bin/env python3
"""
70% Ablation -- Round 35: FOUR OVERNIGHT ARCHS, all built on R34.

R34 = best small-object arch so far (small mAP50 66.07, +3.23% vs baseline) =
  WideFuseV2 @ P4 (main) + ZGSmallDetail @ P3 (main) + DetectAuxDual
  (main=[P3-detail, P4-widefuse, P5], aux=[P3-raw, P4-raw, P5-raw]).

Confirmed pattern across r32b/r33/r34: dual-path supervision helps ONLY when the
enhancement is on the MAIN (inference) path and the RAW feature anchors the AUX.

These 4 cells each test ONE distinct, data-grounded hypothesis on top of r34.
All default TAL / batch 48 / 80 ep / seed 0 / same data -> directly comparable to
r34 (mAP50 82.36 / 50-95 52.28 / small 66.07) and r32b (82.58 / 52.41 / 64.73).

  A. r35_p3widefuse  : P3 enhancement ZGSmallDetail -> ZGLSKAWideFuse[256,7,15]
       (detail + elongated-strip context; targets knife/long_gun small).
  B. r35_dysample    : r34 + DySample replacing the two FPN nearest-upsamples
       (content-aware upsampling -> sharper P3; compounds r34's P3 detail).
       NOTE: DySample is new/untested here -> isolated in try/except.
  C. r35_multiproto  : r34 neck + DetectMultiProto[nc,3] main head (NO aux)
       (mixture classifier for the heterogeneous "other" class, the biggest gap).
  D. r35_p5context   : complete the symmetric dual-path -- add ZGLSKAWideFuse@P5
       main, aux anchors raw P5 (recover r34's lost other-medium/large).

Each run is wrapped in try/except, so one failing cell does NOT abort the rest.

SMOKE-TEST each yaml first (sandbox was down here):
  for n in r35_p3widefuse r35_dysample r35_multiproto r35_p5context; do
    python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/${n}_70.yaml'); \
      print('$n', type(m.model.model[-1]).__name__, m.model.stride.tolist())"; done

Usage:
  python run_round35_overnight.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION  (identical to r32b/r33/r34 so the arch comparison is clean)
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
EPOCHS = 80
AUX_W = 0.5

# -----------------------------------------------------------------------------
# Standard YOLOv12s backbone + head (layers 0-20). 14=P3 head, 17=P4 bottom-up,
# 20=P5 head. (Same 0-20 as r32b/r33/r34.)
# -----------------------------------------------------------------------------
BACKBONE = """nc: 4
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
"""

HEAD_NEAREST = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (PRE-widefuse)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

# Same head but with DySample (content-aware) replacing the two nearest upsamples.
HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                  # 9  content-aware upsample
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, DySample, [2]]                  # 12 content-aware upsample (-> P3)
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (PRE-widefuse)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

BASE = BACKBONE + HEAD_NEAREST
BASE_DYS = BACKBONE + HEAD_DYSAMPLE

# -----------------------------------------------------------------------------
# A. r35_p3widefuse -- richer P3 enhancement (widefuse instead of small-detail)
# -----------------------------------------------------------------------------
ARCH_P3WIDEFUSE = BASE + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 widefuse (main)
  - [14, 1, ZGLSKAWideFuse, [256, 7, 15]]            # 22 -- P3 widefuse (main; detail+strip)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 23
"""

# -----------------------------------------------------------------------------
# B. r35_dysample -- r34 + content-aware upsampling in the FPN
# -----------------------------------------------------------------------------
ARCH_DYSAMPLE = BASE_DYS + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 widefuse (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [[22, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 23
"""

# -----------------------------------------------------------------------------
# C. r35_multiproto -- r34 neck + mixture classifier main head (no aux)
# -----------------------------------------------------------------------------
ARCH_MULTIPROTO = BASE + """  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 widefuse (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [[22, 21, 20], 1, DetectMultiProto, [nc, 3]]     # 23 -- mixture cls head, K=3
"""

# -----------------------------------------------------------------------------
# D. r35_p5context -- complete the symmetric dual-path: enhance P5 too
# -----------------------------------------------------------------------------
ARCH_P5CONTEXT = BASE + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 widefuse (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# Ordered by confidence (most likely to build cleanly AND beat r34 first), so a
# truncated night still yields the best bets:
#   1 (D) p5context  -- proven dual-path pattern, safe args, targets r34's weakness
#   2 (A) p3widefuse -- safe module, targets knife/long_gun small (r34 soft spot)
#   3 (C) multiproto -- high upside on 'other' gap, but drops the confirmed aux
#   4 (B) dysample   -- highest upside-variance; UNTESTED module, may not build -> last
RUNS = [
    {"name": "r35_p5context_70",  "yaml": ARCH_P5CONTEXT,
     "desc": "D [1/4]. complete symmetric dual-path: + ZGLSKAWideFuse @ P5 (recover other med/large)"},
    {"name": "r35_p3widefuse_70", "yaml": ARCH_P3WIDEFUSE,
     "desc": "A [2/4]. P3 enhancement = ZGLSKAWideFuse[256,7,15] (detail+strip; knife/long_gun small)"},
    {"name": "r35_multiproto_70", "yaml": ARCH_MULTIPROTO,
     "desc": "C [3/4]. r34 neck + DetectMultiProto[nc,3] main head (attack 'other' gap; drops aux)"},
    {"name": "r35_dysample_70",   "yaml": ARCH_DYSAMPLE,
     "desc": "B [4/4]. r34 + DySample content-aware upsampling (UNTESTED module; last on purpose)"},
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
    print(f"\n{'=' * 80}\n  ROUND 35 -- 4 OVERNIGHT ARCHS ON R34  (default TAL, arch-only)")
    print(f"  Compare to r34 (82.36 / 52.28 / small 66.07) and r32b (82.58 / 52.41 / 64.73)")
    print(f"{'=' * 80}")
    for i, r in enumerate(RUNS):
        print(f"  [{i+1}] {r['name']:<22} {r['desc']}")
    print(f"{'=' * 80}\n")

    results = [run_experiment(r) for r in RUNS]

    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<22} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
