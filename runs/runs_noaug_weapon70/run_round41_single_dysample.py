#!/usr/bin/env python3
"""
70% Ablation -- Round 41: SINGLE DySample (P3-feeding only)

This is the FINAL architecture experiment, targeting the last remaining
inefficiency discovered across 22 experiments.

DATA-DRIVEN DESIGN:

R39 (globalctx + DySample on BOTH upsamples) revealed a clear signal:
  - other_small:  54.14% -- best EVER by +3.6pp (massive improvement)
  - knife_small:  60.51% -- worst of all runs (-7.05pp vs globalctx)
  - ALL medium classes dropped (knife -2.1pp, long_gun -1.1pp, other -1.4pp)

R39 replaced BOTH FPN upsamples with DySample:
  Layer 9:  DySample (P5 -> P4)  -- HURTS medium objects (oversharpens P4 fusion)
  Layer 12: DySample (P4 -> P3)  -- HELPS small objects (preserves P3 detail)

FIX: Keep DySample ONLY on the P4->P3 upsample. Use standard nearest-neighbor
for P5->P4. This isolates the beneficial effect (sharper P3 features for small
objects) while removing the harmful one (corrupted P4 fusion for medium objects).

Architecture:
  Layer 9:   nn.Upsample (nearest) -- P5->P4 (standard, safe)
  Layer 12:  DySample [2]          -- P4->P3 (content-aware, helps small)
  Layers 21-26: globalctx neck     -- proven best for small objects
  Layer 27:  DetectAuxDual          -- dual-path supervision

Expected result:
  - Preserve R39's other_small gain (54.14% -> ~52-54%)
  - Recover R39's knife_small loss (60.51% -> ~67%)
  - Recover R39's medium-object losses
  - mAP50 between globalctx (82.52) and r36 (82.65)

TWO CELLS:
  Cell 1: Single DySample + globalctx neck (highest confidence)
  Cell 2: Single DySample + p5context neck (for mAP50 comparison)

Compare to (all default TAL):
  r36_p5ctx   82.65 / 52.93 / small 64.49 / other_S 48.95  (best mAP50)
  r38_globalctx 82.52 / 52.69 / small 66.18 / other_S 50.54 (best small)
  r39_combo   82.10 / 52.54 / small 65.59 / other_S 54.14 (best other_S)

Usage:
  python run_round41_single_dysample.py
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
EPOCHS = 80
AUX_W = 0.5

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

# KEY CHANGE: nearest for P5->P4, DySample for P4->P3 ONLY
HEAD_SINGLE_DYSAMPLE = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9  P5->P4: standard (safe for medium)
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, DySample, [2]]                        # 12 P4->P3: content-aware (helps small)
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]              # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]              # 17 -- P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]                   # 20 -- P5 head
"""

# --- Cell 1: Single DySample + globalctx neck (Detect @ 27) ---
TAIL_GLOBALCTX = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext, [256]]                  # 24 -- P3 + global context
  - [21, 1, ZGGlobalContext, [512]]                  # 25 -- P4 + global context
  - [23, 1, ZGGlobalContext, [1024]]                 # 26 -- P5 + global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

# --- Cell 2: Single DySample + p5context neck (Detect @ 24) ---
TAIL_P5CTX = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

ARCH_SINGLE_DY_GLOBALCTX = BACKBONE + HEAD_SINGLE_DYSAMPLE + TAIL_GLOBALCTX
ARCH_SINGLE_DY_P5CTX     = BACKBONE + HEAD_SINGLE_DYSAMPLE + TAIL_P5CTX

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r41_singledy_globalctx_70", "yaml": ARCH_SINGLE_DY_GLOBALCTX, "seed": 0,
     "desc": "[1/2] Single DySample (P3 only) + globalctx neck -- 80% confidence"},
    {"name": "r41_singledy_p5ctx_70",     "yaml": ARCH_SINGLE_DY_P5CTX,     "seed": 0,
     "desc": "[2/2] Single DySample (P3 only) + p5context neck -- 70% confidence"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
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
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: {len(matched)}/{len(remapped)} keys")
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
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# Batch {BATCH}  Epochs {EPOCHS}  default TAL  seed {run['seed']}\n{'#' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")

        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=run["seed"], deterministic=True)
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
    print(f"\n{'=' * 80}")
    print(f"  ROUND 41 -- SINGLE DySample (P3-feeding only) + globalctx/p5ctx")
    print(f"  Target: keep R39's other_small=54.14 without the medium-object regression")
    print(f"  Method: DySample on P4->P3 ONLY; nearest on P5->P4")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<35} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
