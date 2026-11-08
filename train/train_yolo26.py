#!/usr/bin/env python3
"""
Train the YOLO26 luggage-detection models reported in the paper.

Requires the patched YOLO26 code base (see README, "Environment B"):
ultralytics @ 12d61b806 (8.4.117) + ultralytics_patches/v26.

    python train/train_yolo26.py --data /path/to/LuggageDataset.v6i/data.yaml --model yolo26s_default
    python train/train_yolo26.py --data ... --model yolo26s_p234rich_beta1
    python train/train_yolo26.py --data ... --model yolo26x_p234rich

Every setting below is copied from the args.yaml of the run that produced the
published weights (results/train_args/). The published YOLO26 weights are the
standard best.pt (best validation fitness).

p234rich graphs (P2-P4 detect levels, DySample upsampling, deeper heads) are built
from their YAML, then:
  1. model.load(<official checkpoint>) transfers every layer whose name/shape matches;
  2. the stock P3-P5 head rows 17-19 of the checkpoint are copied into rows 23-25 of the
     p234rich graph (the extra P2 branch shifts the PAN rows by 6), so those rows also
     start from pretrained weights instead of random init.
"""

import argparse
import re
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

COMMON = dict(epochs=70, imgsz=640, seed=0, deterministic=True, close_mosaic=10, patience=100, workers=8)

REMAP_ROWS = range(17, 20)  # stock PAN/head rows in the official checkpoint
REMAP_SHIFT = 6             # their index in the p234rich graph is +6

MODELS = {
    # stock YOLO26-s, stock loss
    "yolo26s_default": dict(model="yolo26s.pt", pretrained=None, batch=82, extra={}),
    # p234rich graph + TAL beta = 1.0 (stock beta is 6.0); everything else stock
    "yolo26s_p234rich_beta1": dict(model=str(ROOT / "configs/yolo26/yolo26-p2dys-p234rich.yaml"),
                                   pretrained="yolo26s.pt", batch=48, extra=dict(tal_beta=1.0)),
    # p234rich graph at x scale, stock loss
    "yolo26x_p234rich": dict(model=str(ROOT / "configs/yolo26/yolo26x-p2dys-p234rich.yaml"),
                             pretrained="yolo26x.pt", batch=12, extra={}),
}


def remap_pan(model, weights):
    """Copy checkpoint rows 17-19 into rows 23-25 of the p234rich graph (exact shapes only)."""
    ckpt = YOLO(weights).model.float().state_dict()
    sd = model.model.state_dict()
    moved = {}
    for k, v in ckpt.items():
        m = re.match(r"model\.(\d+)\.(.+)", k)
        if m and int(m.group(1)) in REMAP_ROWS:
            k2 = f"model.{int(m.group(1)) + REMAP_SHIFT}.{m.group(2)}"
            if k2 not in sd or sd[k2].shape != v.shape:
                raise RuntimeError(f"remap target {k2} missing or shape mismatch")
            moved[k2] = v
    if not moved:
        raise RuntimeError("remap matched nothing - wrong graph?")
    model.model.load_state_dict(moved, strict=False)
    print(f"[remap] {len(moved)} tensors copied from {weights} rows 17-19 -> 23-25")


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
        model.load(cfg["pretrained"])        # 1. name/shape transfer (must run first)
        remap_pan(model, cfg["pretrained"])  # 2. shifted PAN rows
    model.train(data=a.data, project=a.project, name=a.model, device=a.device,
                batch=a.batch or cfg["batch"], **COMMON, **cfg["extra"])


if __name__ == "__main__":
    main()
