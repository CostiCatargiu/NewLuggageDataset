#!/usr/bin/env python3
"""
Train the YOLOv12 luggage-detection models reported in the paper.

Requires the patched YOLOv12 code base (see README, "Environment A"):
sunsmarterjie/yolov12 @ 2db2bae + ultralytics_patches/v12.

    python train/train_yolov12.py --data /path/to/LuggageDataset.v6i/data.yaml --model yolov12s_default
    python train/train_yolov12.py --data ... --model yolov12s_lsshift_sqrt
    python train/train_yolov12.py --data ... --model yolov12x_lsshift

Every setting below is copied from the args.yaml of the run that produced the
published weights (results/train_args/). Loss keys are passed explicitly because
the patched default.yaml does NOT ship neutral values for them.

Checkpoint note: the runs behind the published v12 weights saved the FINAL-epoch
(epoch 70) checkpoint and that is the file that was evaluated. With this code base
the final-epoch file is weights/last.pt -- evaluate last.pt to match the paper.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel

ROOT = Path(__file__).resolve().parents[1]

# Common recipe, identical for every run.
COMMON = dict(epochs=70, imgsz=640, seed=0, deterministic=True, close_mosaic=10, patience=100, workers=8)

# Custom-loss switches with every term OFF (stock YOLOv12 loss).
STOCK_LOSS = dict(
    small_obj_px=0, small_obj_boost=1.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    satal_alpha_large=1.0, satal_small_area=0.0025, satal_large_area=0.0225,
    nwd_weight=0.0, artal_ar_thresh=2.0, artal_ar_scale=2.0, snatal_kmin=1,
    ardfl_w_weight=1.5, ardfl_h_weight=0.75,
)

# Scale-weighted alignment ("sqrt 0.7->0.3"): alpha schedule 0.7 -> 0.3 over training
# (clamped to [alpha_min=0.3 from default.yaml, alpha_max=0.7]), sqrt-area box weighting,
# x2.0 boost for objects below 48 px. Exactly the keys that run set; the rest stay at
# default.yaml values (their mechanisms are switched off there).
SQRT_LOSS = dict(
    small_obj_px=32, small_obj_boost=2.0,
    alpha_start=0.7, alpha_end=0.3, alpha_max=0.7,
    area_weight_mode="sqrt",
    center_loss_weight_min=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    nwd_weight=0.0,
)

MODELS = {
    # stock YOLOv12-s, stock loss
    "yolov12s_default": dict(model="yolov12s.pt", pretrained=None, batch=54, loss=STOCK_LOSS),
    # ls_shift graph (P2-P5, DySample, gctx2 @P2, ZGDSConv k9 @P3-P5) + sqrt loss
    "yolov12s_lsshift_sqrt": dict(model=str(ROOT / "configs/yolov12/yolov12s-lsshift.yaml"),
                                  pretrained="yolov12s.pt", batch=28, loss=SQRT_LOSS),
    # ls_shift graph at x scale. The published weights were trained with the STOCK loss
    # (see README, "Known issues"); set loss=SQRT_LOSS to train the arch+loss variant.
    "yolov12x_lsshift": dict(model=str(ROOT / "configs/yolov12/yolov12x-lsshift.yaml"),
                             pretrained="yolov12x.pt", batch=8, loss=STOCK_LOSS),
}


def sync_epoch(trainer):
    """The alpha schedule in the patched loss reads criterion.epoch; keep it in sync."""
    m = de_parallel(trainer.model)
    m.current_epoch = trainer.epoch
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            crit.epoch = trainer.epoch
            if hasattr(crit, "_sync_bbox_loss_state"):
                crit._sync_bbox_loss_state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data", required=True, help="path to the dataset data.yaml")
    ap.add_argument("--project", default="runs_luggage")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=None, help="override (changes results)")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    model = YOLO(cfg["model"])
    if cfg["pretrained"]:
        model.load(cfg["pretrained"])  # transfers every layer whose name/shape matches
    model.add_callback("on_train_epoch_start", sync_epoch)
    model.train(data=a.data, project=a.project, name=a.model, device=a.device,
                batch=a.batch or cfg["batch"], **COMMON, **cfg["loss"])


if __name__ == "__main__":
    main()
