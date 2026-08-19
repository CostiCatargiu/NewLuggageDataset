# YOLOv12 patch files — DO NOT MIX WITH MODEL_v26/patch/

`loss.py` and `tal.py` exist in BOTH patch folders with the SAME NAME and
DIFFERENT CONTENT. Installing the wrong one gives a model that trains without
error and optimises the wrong objective.

    MODEL_v12/patch/loss.py   91637 bytes   md5 067a61c0...   <- THIS ONE, for yolov12s
    MODEL_v26/patch/loss.py   94571 bytes   md5 df6419d8...   <- YOLO26, do not use here

`loss.py` / `tal.py` here came from `archAblation/`.

## Contents
    loss.py, tal.py            the v12 SWA / LB-TAL assigner
    lossv2updated*.py          earlier loss revisions
    loss_custom_v3_fixed.py
    satal.py                   size-aware TAL
    zg_modules_v6i.py          ZGGlobalContext2, ZGDSConv, DySample
    patch_ultralytics_modules.py, verify_port.py

## Why the two differ structurally
YOLOv12: reg_max=16 (DFL bins), NMS, ONE assignment branch, topk=10.
YOLO26 : reg_max=1 (DFL-free), NMS-free end2end head, TWO branches, topk2=1.

The v26 file has no DFL branch and carries E2ELoss; the v12 file has no
one2one/one2many split. They are not interchangeable in either direction.
