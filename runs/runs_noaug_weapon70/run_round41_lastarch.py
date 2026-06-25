#!/usr/bin/env python3
"""
70% Ablation -- Round 41: THE LAST ARCH TRY (avg+max global context).

Context: the architecture is saturated. globalctx (p5context + per-level gentle
global context) is the seed-region champion (clean valid+test mean 82.29 / 52.57 /
small 66.91). Everything ADDED to it has failed: r39 (globalctx+dysample) flat-to-
worse, r40 (deeper P3 head) hurt small, and earlier gather / wfv2_p3 / bifpn /
multiproto / P2 / aux-weight all underperformed.

So this last try is NOT a new axis (those keep failing) -- it's a principled
ENRICHMENT of the one mechanism that works: globalctx's gated global context.

  globalctx uses AVG-pool only (whole-scene context). ZGGlobalContext2 adds a
  MAX-pool branch (avg+max -> MLP -> gated add): avg = context, max = the single
  most SALIENT activation, which small/rare weapon cues spike on but that averages
  away in avg-pool (the CBAM/BAM channel-attention insight). It enriches the
  WINNING module itself (not stacking a 2nd module, which sank r39), stays
  gentle/gated/identity-init (the property that made globalctx generalize).

  Cell: r41_globalctx2 = globalctx neck with ZGGlobalContext2 (avg+max) per level.

Honest expectation: arch is saturated -> most likely a fraction-of-a-point gain or
a wash; judge on the CLEAN valid+test mean (the noisy 70%-val curve misleads).
After this, the real levers are globalctx + best-TAL and seed validation, NOT arch.

Compare to globalctx (clean mean) 82.29 / 52.57 / small 66.91.
Default TAL / batch 48 / 80 ep / seed 0.

SMOKE-TEST first:
  python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/r41_globalctx2_70.yaml'); \
    print(type(m.model.model[-1]).__name__, m.model.stride.tolist())"
  -> expect DetectAuxDual [8.0, 16.0, 32.0] (Detect @ 27).
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

# globalctx neck, but ZGGlobalContext2 (avg+max) instead of ZGGlobalContext.
ARCH_GLOBALCTX2 = f"""nc: 4
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
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head (raw; aux anchor)
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext2, [256]]                 # 24 -- P3 + avg+max global context
  - [21, 1, ZGGlobalContext2, [512]]                 # 25 -- P4 + avg+max global context
  - [23, 1, ZGGlobalContext2, [1024]]                # 26 -- P5 + avg+max global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
RUN_NAME = "r41_globalctx2_70"


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


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    yaml_path = os.path.join(YAML_DIR, f"{RUN_NAME}.yaml")
    save_yaml(ARCH_GLOBALCTX2, yaml_path)
    print(f"\n{'=' * 80}\n  ROUND 41 -- LAST ARCH TRY: globalctx + avg+max global context")
    print(f"  vs globalctx (clean mean) 82.29 / 52.57 / small 66.91")
    print(f"{'=' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=RUN_NAME, patience=100,
                  close_mosaic=10, seed=0, deterministic=True)
        kw.update(DEFAULT_TAL)
        model.train(**kw)
        print(f"\n  DONE: {RUN_NAME} ({(time.time()-start)/3600:.2f}h)")
    except Exception as e:
        print(f"\n  FAILED: {RUN_NAME} ({(time.time()-start)/3600:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
