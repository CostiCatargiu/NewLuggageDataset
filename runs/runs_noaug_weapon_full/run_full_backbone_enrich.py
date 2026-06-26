#!/usr/bin/env python3
"""
FULL revised dataset -- the final arch attempt: BACKBONE ENRICHMENT.

Rationale (from all prior data): at full data the neck/head changes are neutral
(globalctx ~ stock), because neck enhancements are data-substitute PRIORS that more
data makes redundant. The ONE untouched axis is the BACKBONE -- representational
CAPABILITY, which more data CAN exploit. So this enriches the backbone P4 and P5
stages with gated, identity-safe large-kernel context blocks (ZGLSKAWideFuse),
then keeps the converged globalctx neck on top.

CRITICAL -- transfer preserved despite insertion: inserting blocks shifts all later
layer indices, which normally breaks the name-based pretrained load (the neck would
train from scratch -> unfair + worse). The loader below REMAPS stock indices to the
new shifted indices, so the original backbone+neck still transfer from yolov12s.pt;
only the 2 new context blocks + the globalctx tail train fresh.

Layer map (new <- stock):
  0..6 <- 0..6        (unchanged)
  7    =  NEW ZGLSKAWideFuse @ backbone-P4   (fresh)
  8    <- 7  (Conv down)     9 <- 8 (A2C2f P5)
  10   =  NEW ZGLSKAWideFuse @ backbone-P5   (fresh)
  11..22 <- 9..20    (neck, shift +2)
  29   <- 21 (Detect, det-remap)

Cells (full data, 90 ep, batch 48, seed 0):
  1. bbenrich_full_default : backbone-enrich + globalctx + VANILLA loss
       -> clean arch comparison vs stock_full_default (84.80) and globalctx_full_default (84.51).
  2. bbenrich_full_besttal : + BEST TAL (headline candidate).

Honest EV: low (~25-30%). yolov12's backbone already has attention (A2C2f), and the
baseline is strong (84.80). This is the most mechanistically-defensible last arch
swing; if it's neutral, the architecture question is closed.

SMOKE-TEST FIRST (untested here -- sandbox down):
  python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/bbenrich_full_default.yaml'); \
    print(type(m.model.model[-1]).__name__, len(m.model.model), m.model.stride.tolist())"
  -> expect DetectAuxDual, 30 layers, [8.0,16.0,32.0]. Then check the loader prints
     a HIGH transfer count (most of backbone+neck), not ~115.
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # FULL revised dataset
PROJECT_DIR = "runs_noaug_weapon_full"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
BATCH = 48
EPOCHS = 90
AUX_W = 0.5

# Backbone with ZGLSKAWideFuse inserted after the P4 (layer 6) and P5 (layer 9)
# stages, then the standard neck (re-indexed), then the globalctx tail. Detect @ 29.
ARCH_BBENRICH = f"""nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]                       # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]                # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]               # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]                # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]               # 4  backbone P3
  - [-1, 1, Conv,  [512, 3, 2]]                      # 5
  - [-1, 4, A2C2f, [512, True, 4]]                   # 6  backbone P4
  - [-1, 1, ZGLSKAWideFuse, [512, 11, 23]]           # 7  *** NEW: enrich backbone P4
  - [-1, 1, Conv,  [1024, 3, 2]]                     # 8  (down from enriched P4)
  - [-1, 4, A2C2f, [1024, True, 1]]                  # 9  backbone P5
  - [-1, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 10 *** NEW: enrich backbone P5

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]       # 11  (from enriched P5)
  - [[-1, 7], 1, Concat, [1]]                        # 12  concat enriched P4
  - [-1, 2, A2C2f, [512, False, -1]]                 # 13  P4 td
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]       # 14
  - [[-1, 4], 1, Concat, [1]]                        # 15  concat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]                 # 16  P3 head (raw; aux anchor)
  - [-1, 1, Conv, [256, 3, 2]]                       # 17
  - [[-1, 13], 1, Concat, [1]]                       # 18  concat P4 td
  - [-1, 2, A2C2f, [512, False, -1]]                 # 19  P4 bottom-up (raw; aux anchor)
  - [-1, 1, Conv, [512, 3, 2]]                       # 20
  - [[-1, 10], 1, Concat, [1]]                       # 21  concat enriched P5
  - [-1, 2, C3k2, [1024, True]]                      # 22  P5 head (raw; aux anchor)
  - [19, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 23  P4 hybrid (main)
  - [16, 1, ZGSmallDetail, [256, 3, 5]]              # 24  P3 detail (main)
  - [22, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 25  P5 context (main)
  - [24, 1, ZGGlobalContext, [256]]                  # 26  P3 + global context
  - [23, 1, ZGGlobalContext, [512]]                  # 27  P4 + global context
  - [25, 1, ZGGlobalContext, [1024]]                 # 28  P5 + global context
  - [[26, 27, 28, 16, 19, 22], 1, DetectAuxDual, [nc, {AUX_W}]]  # 29
"""

DEFAULT_TAL = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, use_vfl=False,
)
TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0, dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0, iou_type="DIoU", use_vfl=False,
)

RUNS = [
    {"name": "bbenrich_full_default", "loss": DEFAULT_TAL,
     "desc": "[1/2] backbone-enrich + globalctx + VANILLA loss (clean arch vs stock 84.80)"},
    {"name": "bbenrich_full_besttal", "loss": TAL_BEST_LOOSE,
     "desc": "[2/2] backbone-enrich + globalctx + BEST TAL (headline candidate)"},
]


def remap_index(old):
    """stock layer index -> new (shifted) index, accounting for inserts at 7 and 10."""
    if old <= 6:
        return old
    if old == 7:
        return 8
    if old == 8:
        return 9
    if 9 <= old <= 20:
        return old + 2
    if old == 21:           # Detect
        return 29
    return None


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_shifted(model, weights=PRETRAINED):
    """Transfer stock backbone+neck+Detect into the index-shifted model via remap_index."""
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    remapped = {}
    for k, v in csd.items():
        if not k.startswith("model."):
            continue
        parts = k.split(".", 2)            # ["model", "<idx>", "<rest>"]
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        new_i = remap_index(int(parts[1]))
        if new_i is None:
            continue
        remapped[f"model.{new_i}.{parts[2]}"] = v
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [shifted-remap] transferred {len(matched)}/{len(model.model.state_dict())} tensors "
          f"(new backbone/neck + Detect; 2 context blocks + globalctx tail train fresh)")
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
    save_yaml(ARCH_BBENRICH, yaml_path)
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# FULL data  Batch {BATCH}  Epochs {EPOCHS}  seed 0\n{'#' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_shifted(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, layers = {len(model.model.model)}, "
              f"strides = {model.model.stride.tolist()}")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=0, deterministic=True)
        kw.update(run["loss"])
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
    print(f"\n{'=' * 80}\n  FULL data -- BACKBONE ENRICHMENT (final arch attempt)")
    print(f"  vs stock_full_default 84.80 | globalctx_full_default 84.51")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<24} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
