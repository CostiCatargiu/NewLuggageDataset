#!/usr/bin/env python3
"""
Full Dataset — train the BEST ARCHITECTURE, two ways:

  1. r21_arch_full        — best arch (r21), DEFAULT TAL, loss extras OFF.
                            Isolates the pure architecture effect on full.
  2. r21_tal07_loose_full — best arch (r21) + BEST TAL (v5_tal07_loose recipe).
                            The combination: does the project-best loss config
                            stack on the project-best architecture?

Best architecture  = r21_widefuse_aux_w50:
    ZGLSKAWideFuse[512,11,23] @ P4 bottom-up  +  train-only DetectAux @ 0.5
    (test mAP50 79.57 / mAP50-95 50.33 on the 70% ablation; best arch in project).

Best TAL (full)    = v5_tal07_loose_full (test mAP50 83.35 / mAP50-95 53.41 on full,
    the best loss config; tops mAP50, mAP50-95, recall AND small):
    topk=13, alpha=0.7, beta=4.0, loose clips (iou 50/20, dfl 25/10),
    small_obj_boost=2.5 @ <40px, DIoU, cls=1.2, SWA-alpha schedule 0.7->0.3.

Both runs: yolov12s / 640, full dataset, 90 epochs, seed 0, FIXED batch (same as
each other for a clean arch-vs-arch+TAL comparison). r21 appends layers after the
stock layer 20, so we transfer pretrained backbone+neck+box head via Detect-remap;
the aux towers train fresh.

NOTE / VERIFY before trusting run 2:
  r21's head is DetectAux -> DetectionModel.init_criterion returns DetectAuxLoss.
  In this repo snapshot DetectAuxLoss does `v8DetectionLoss(model, tal_topk=10)`
  and v8DetectionLoss builds the assigner with hardcoded alpha=0.5/beta=6.0, so the
  aux path may IGNORE tal_topk=13/alpha=0.7/beta=4.0. Confirm on your training code
  that DetectAuxLoss forwards the TAL args (see patch note at the bottom of this file).
  The clip / small_obj / SWA-alpha terms are applied by the trainer, not the assigner,
  so they are unaffected.

Usage:
  python run_full_r21_arch_and_tal.py
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
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # FULL dataset
PROJECT_DIR = "runs_noaug_weapon_full"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 90
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21          # Detect index in stock yolov12s
# r21 (widefuse + aux towers) is heavier than stock yolov12s. The full "loose" run
# used batch 58 on plain yolov12s; r21 needs less. 48 = r21's established ablation
# batch. BOTH r21 runs share this batch so the arch-vs-arch+TAL comparison is clean.
BATCH = 48
AUX_W = 0.5                  # r21's winning aux weight

# -----------------------------------------------------------------------------
# Best architecture: r21 = ZGLSKAWideFuse[512,11,23] @ P4-BU + DetectAux @ 0.5
# -----------------------------------------------------------------------------
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

ARCH_R21 = BASE_0_20 + f"""
  - [17, 1, ZGLSKAWideFuse, [512, 11, 23]]       # 21 — gated wide-fuse @ P4 (= r11)
  - [[14, 21, 20], 1, DetectAux, [nc, {AUX_W}]]  # 22 — train-only aux (= r21)
"""

# -----------------------------------------------------------------------------
# Loss recipes
# -----------------------------------------------------------------------------
# Run 1 — pure architecture: DEFAULT TAL, every custom loss feature OFF.
TAL_ARCH_ONLY = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    use_vfl=False,
)

# Run 2 — best TAL: the exact v5_tal07_loose recipe (= run_full_loose.py BASE,
# matches the v5_tal07_loose_full args dump).
TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0,
    dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0,
    iou_type="DIoU",
    use_vfl=False,
)

RUNS = [
    {"name": "r21_arch_full",
     "desc": "[1/2] best arch (r21 widefuse+aux@0.5), DEFAULT TAL — pure architecture on full",
     "loss": TAL_ARCH_ONLY},
    {"name": "r21_tal07_loose_full",
     "desc": "[2/2] best arch (r21) + best TAL (v5_tal07_loose recipe) on full",
     "loss": TAL_BEST_LOOSE},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() then remap Detect keys model.21.* -> model.N.* if the index
    shifted (r21 appends 1 layer, Detect 21 -> 22). Transfers backbone + neck +
    box head via intersect_dicts; the aux towers train fresh."""
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


def on_train_epoch_start(trainer):
    """Sync the SWA-alpha / clip schedule with the current epoch (= run_full_loose)."""
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
    save_yaml(ARCH_R21, yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch: {BATCH}   Epochs: {EPOCHS}   Seed: 0")
    print(f"{'#' * 70}\n")

    start_time = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)

        train_kwargs = dict(
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
        )
        train_kwargs.update(run["loss"])

        model.train(**train_kwargs)

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
    print(f"  FULL DATASET — BEST ARCH (r21)  +/-  BEST TAL (v5_tal07_loose)")
    print(f"  Bars on full: stock+best-TAL = 83.35 mAP50 / 53.41 mAP50-95")
    print(f"{'=' * 70}")
    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={BATCH}  {run['desc']}")
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


# =============================================================================
# PATCH NOTE — make DetectAuxLoss honor the TAL config (needed for run 2)
# =============================================================================
# If your DetectAuxLoss hardcodes tal_topk=10 / the assigner hardcodes alpha,beta,
# the aux head will ignore tal_topk=13/alpha=0.7/beta=4.0. Forward them from args:
#
#   # ultralytics/utils/loss.py  -- class DetectAuxLoss.__init__
#   def __init__(self, model, aux_weight=0.25):
#       h = model.args
#       self.det = v8DetectionLoss(model, tal_topk=int(getattr(h, "tal_topk", 10)))
#       # if your v8DetectionLoss reads alpha/beta from args, this is enough;
#       # otherwise also set them on the assigner explicitly:
#       self.det.assigner.alpha = float(getattr(h, "tal_alpha", 0.5))
#       self.det.assigner.beta  = float(getattr(h, "tal_beta", 6.0))
#       self.aux_weight = getattr(model.model[-1], "aux_weight", aux_weight)
#
# Verify the same TAL-arg plumbing exists in your main v8DetectionLoss before the
# run, so the assigner actually uses topk=13/alpha=0.7/beta=4.0.
