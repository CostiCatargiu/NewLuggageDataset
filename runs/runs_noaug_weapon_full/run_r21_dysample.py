#!/usr/bin/env python3
"""
r21 + DySample (content-aware upsampling) on the REVISED dataset — a small-object
architecture experiment.

IDEA (small-object specific, NOT resolution): the YOLOv12 FPN builds the P3
(stride-8, small-object) map by NEAREST-NEIGHBOUR upsampling the coarser P4/P5
features and fusing them with the backbone P3 skip. Nearest upsampling smears a
single low-res cell across a 2x2 block — it cannot place a boundary between two
neighbouring grid cells, which is exactly the detail a small object lives in. So
the small level is fed blurred top-down context regardless of how good the
backbone features are.

FIX: replace the two top-down nn.Upsample ops with DySample (ICCV'23) — a
learned, content-aware dynamic upsampler that predicts per-location sampling
offsets and gathers with grid_sample. It sharpens object boundaries when
upsampling, so the P3 fusion receives crisp top-down features. Channel-preserving,
~tens of k params, near-zero inference cost. The offset conv is near-zero-init so
the net starts ~bilinear (clean pretrained transfer via the r21 detect-remap).

This is orthogonal to everything tried in the ~60-arch search: every prior block
operated on EXISTING feature maps (attention / fusion / extra heads); none touched
HOW features are upsampled. It is also the standard, citable lever for small
objects (CARAFE/DySample/FADE line of work) — a clean paper contribution.

Built ON TOP of r21 (widefuse @ P4 + train-only aux), so it is a strict superset:
the only change vs rev_r21_* is nearest -> DySample on head layers 9 and 12.

Two cells on the revised data (compare to rev_r21_tal = 0.8237 / 50-95 0.5223 / small 0.664):
  1. rev_r21dys_default : r21+DySample + DEFAULT TAL  (isolate the upsampler effect)
  2. rev_r21dys_tal     : r21+DySample + BEST TAL     (the candidate model)

SMOKE-TEST FIRST (sandbox was down; this build is untested here):
  python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/rev_r21dys_default.yaml'); \
             print(type(m.model.model[-1]).__name__, m.model.stride.tolist())"
  -> expect DetectAux and strides [8.,16.,32.]. Confirm DySample layers parsed
  (channels at layer 9 = 512, layer 12 = 256; both divisible by groups=4).
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # REVISED dataset (same as the 2x2)
PROJECT_DIR = "runs_noaug_weapon_full"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 90
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48          # same as the 2x2 cells (DySample is channel-preserving, ~free) -> batch-parity preserved
AUX_W = 0.5

# Shared YOLOv12s backbone + neck (layers 0-20), IDENTICAL to run_full_r21_arch_and_tal.py
# EXCEPT the two top-down nn.Upsample ops (head layers 9, 12) are DySample.
BASE_0_20_DYS = """nc: 4
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
  - [-1, 1, DySample, [2]]                  # 9  content-aware upsample (P5 -> P4 level)
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, DySample, [2]]                  # 12 content-aware upsample (P4 -> P3 level) <- the small-object one
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head
"""

# r21 + DySample: same widefuse @ P4 + train-only aux as r21.
ARCH_R21_DYS = BASE_0_20_DYS + f"""  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]       # 21 — gated wide-fuse @ P4 (= r21)
  - [[14, 21, 20], 1, DetectAux, [nc, {AUX_W}]]  # 22 — train-only aux
"""

# -----------------------------------------------------------------------------
# Loss recipes (identical to the 2x2)
# -----------------------------------------------------------------------------
TAL_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    use_vfl=False,
)

TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0,
    dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0,
    iou_type="DIoU", use_vfl=False,
)

RUNS = [
    {"name": "rev_r21dys_default", "loss": TAL_DEFAULT,    "desc": "[1/2] r21+DySample + DEFAULT TAL (isolate upsampler effect)"},
    {"name": "rev_r21dys_tal",     "loss": TAL_BEST_LOOSE, "desc": "[2/2] r21+DySample + BEST TAL (candidate model)"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Transfer backbone+neck (0-20) via model.load; DySample offset convs + aux train fresh."""
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
          f"{len(matched)}/{len(remapped)} keys transferred (DySample/aux fresh)")
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
    save_yaml(ARCH_R21_DYS, yaml_path)
    print(f"\n{'#' * 70}\n# {run['name']}\n# {run['desc']}\n# Batch: {BATCH}  Epochs: {EPOCHS}  Seed: 0\n{'#' * 70}\n")

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
    print(f"\n{'=' * 70}\n  r21 + DySample (content-aware upsampling) on REVISED data\n  data: {DATA_YAML}\n{'=' * 70}")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 70}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<20} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
