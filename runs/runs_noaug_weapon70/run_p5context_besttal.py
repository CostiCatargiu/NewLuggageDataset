#!/usr/bin/env python3
"""
p5context + BEST-TAL -- the candidate final model.

p5context (round-35 winner, default TAL) = scale-matched dual-path enhancement:
  P3 detail / P4 hybrid / P5 context, all enhanced-on-main + raw-on-aux.
  -> best mAP50-95 of the campaign (52.82), best P/R, small 65.16 (just under r34 66.07).

This run stacks the proven best-TAL recipe (TAL_BEST_LOOSE, the SAME one that gave
rev_r21_tal small 66.36 / overall 82.37) on top. The key ingredient is
small_obj_boost=2.5 + small_obj_px=40 -- the lever that lifted r21 small by ~+3pt.
Stacked on the best-50-95 neck, this is the most likely path to a clean new champion.

Compare to:
  rev_r21_tal   82.37 / 52.23 / small 66.36   (current small champion; r21 + this TAL)
  rev_stock_tal 82.61 / 51.99 / small 65.37
  p5context     82.38 / 52.82 / small 65.16   (this arch, DEFAULT TAL)

NOTE on parity: TAL_BEST_LOOSE explicitly overrides the fork loss defaults
(use_vfl=False, DIoU, alpha 0.7->0.3, clips on, boost 2.5) so this matches the
rev_*_tal champions' loss exactly. Batch 48 to match rev_r21_tal (best4 ran arch
cells at 50; the ~2 diff is within noise but 48 = exact parity with the TAL champ).
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

# p5context architecture (round-35 winner)
ARCH_P5CONTEXT = f"""nc: 4
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
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# Best-TAL recipe (identical to rev_r21_tal / rev_stock_tal): the small_obj_boost
# is the key lever here.
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

RUN_NAME = "p5context_besttal_70"


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
    save_yaml(ARCH_P5CONTEXT, yaml_path)

    print(f"\n{'=' * 80}\n  p5context + BEST-TAL  (candidate final model)")
    print(f"  small_obj_boost=2.5 stacked on the best-50-95 neck")
    print(f"  vs rev_r21_tal 82.37 / 52.23 / small 66.36")
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
        kw.update(TAL_BEST_LOOSE)
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
