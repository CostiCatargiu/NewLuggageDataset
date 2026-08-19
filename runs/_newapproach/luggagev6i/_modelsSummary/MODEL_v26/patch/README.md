# YOLO26 patch files — DO NOT MIX WITH MODEL_v12/patch/

`loss.py` and `tal.py` exist in BOTH patch folders with the SAME NAME and
DIFFERENT CONTENT. Installing the wrong one gives a model that trains without
error and optimises the wrong objective.

    MODEL_v26/patch/loss.py   94571 bytes   md5 df6419d8...   <- THIS ONE, for yolo26s
    MODEL_v12/patch/loss.py   91637 bytes   md5 067a61c0...   <- YOLOv12, do not use here

Source of truth: `ultralytics26/ultralytics/utils/` (verified byte-identical).

## Contents
    loss.py        SWA, SNL1, SBB, E2ELoss wiring, NWD/iou_type on BboxLoss
    tal.py         SCB, SNT, TSH, SBAL, LevelBalancedTaskAlignedAssigner
    metrics.py     bbox_iou with EIoU + NWD, IOU_FLAGS  (was stock until round 11)
    default.yaml   all custom keys; every mechanism defaults to OFF

## Install
    python ../training/verify_patch_v6i.py --ref <ultralytics pkg dir> --install --runtime

The runtime check asserts `bbox_iou` at defaults is BITWISE identical to stock
CIoU. metrics.py was restructured from early-return to fall-through so the NWD
block could see the computed variant — if that file is stale while loss.py/tal.py
are patched, CIoU silently returns early: no crash, wrong arithmetic.

Re-run after ANY `pip install -e .` — a reinstall reverts every patched file.
That is how rounds 4-6 produced ten identically-configured runs under ten names.
