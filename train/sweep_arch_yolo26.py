#!/usr/bin/env python3
"""
YOLO26-s ARCHITECTURE SWEEP — 11 graphs, stock loss, LuggageDataset v6i
Environment B (ultralytics 8.4.117 + ultralytics_patches/v26).

Every architecture evaluated for YOLO26-s is written out below, with what it changes and
why it was tried. Only the graph changes between runs: the loss is stock (every custom term
switched off), the recipe is the same and the batch is the same (48) so all runs take the
same number of optimizer steps. Each graph starts from yolo26s.pt (all layers whose shape
matches are copied). Graphs with an inserted P2 level shift the PAN/head rows by six, so
checkpoint rows 17-22 are also copied to rows 23-28 (where the shapes match).

   #  architecture                       status
   1  yolo26-p2dys-p234rich              FINAL (paper)
   2  yolo26-stock                       BASELINE
   3  yolo26-p2                          TRIED
   4  yolo26-p2dys                       TRIED
   5  yolo26-p2dys-wide                  TRIED
   6  yolo26-p2dys-gctx                  TRIED (transfer from YOLOv12s)
   7  yolo26-lsshift                     TRIED (transfer from YOLOv12s)
   8  yolo26-p2dys-p234                  TRIED
   9  yolo26-p2dys-p234-d4-2             TRIED
  10  yolo26-p2dys-p234-d6-2             TRIED
  11  yolo26-p2dys-p234-d6-6             TRIED

The same configs, with the same descriptions, are in configs/yolo26/yolo26s_architectures.yaml.

Usage:
  python sweep_arch_yolo26.py             # all eleven, in order

To run a subset, comment entries out of ARCH_RUNS at the bottom of the config section.
Results: one folder per run in PROJECT_DIR, and PROJECT_DIR_results.json with the test-split
mAP50 / mAP50-95 of every run (written after every run, so finished runs survive a crash).
The paper's per-size numbers (mAP@50 small) come from the COCO evaluator in
original_scripts/CocoEvalAllFolders_luggage.py.
"""

import gc
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = str(ROOT / "data" / "luggage_v6i.yaml")  # edit: path to your LuggageDataset v6i data.yaml
PROJECT_DIR = "runs_arch_yolo26"
YAML_DIR = "arch_yamls_yolo26"
WEIGHTS = "yolo26s.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 70
BATCH = 48  # one batch for every graph, so step counts match
SEED = 0

# Stock YOLO26 assigner (the patched default.yaml is already neutral; written out for clarity).
STOCK_LOSS = dict(tal_alpha=0.5, tal_beta=6.0)
REMAP_ROWS = range(17, 23)  # stock PAN/head rows in yolo26s.pt
REMAP_SHIFT = 6             # their index in graphs with an inserted P2 level

# =============================================================================
# ARCH 1: yolo26-p2dys-p234rich  --  FINAL (paper)
#   What:
#     P2 level inserted into the PAN (bottom-up path rebuilt through it); DySample for P3->P2; stride-32
#     detection branch removed (stride-32 backbone kept, detect on 4/8/16); P2 and P3 blocks deepened
#     from 2 to 4 repeats.
#   Why:
#     Level reallocation. The anchor-supply diagnostic shows stride 32 is nearly idle on this data (0.01
#     positives per small ground truth, 0.00 per medium; large objects take most positives from stride
#     16), so its detection branch is deleted while the stride-32 backbone keeps feeding context into
#     the top-down path. Dropping P5 also shortens the upsampling chain into P2 from three steps to two.
#     The freed parameters are reinvested as depth (2 -> 4 repeats) at P2 and P3, where 75% of the
#     instances are. Depth rather than width, because widening had already been tried and did not help.
#   Parameters, unfused (3 classes, scale s): 7.06M, strides 4/8/16
# =============================================================================
ARCH_YOLO26_P2DYS_P234RICH = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10 <- still feeds the top-down path at row 11
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 4, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 4, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# ARCH 2: yolo26-stock  --  BASELINE
#   What:
#     Stock YOLO26 (ultralytics 8.4.117), three detection levels (8/16/32).
#   Why:
#     Reference graph (official YOLO26, three detection levels).
#   Parameters, unfused (3 classes, scale s): 9.95M, strides 8/16/32
# =============================================================================
ARCH_YOLO26_STOCK = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:  # model compound scaling constants, i.e. 'model=yolo26n.yaml' will call yolo26.yaml with scale 'n'
  s: [0.50, 0.50, 1024]  # summary: 260 layers, 10,009,784 parameters, 10,009,784 gradients, 22.8 GFLOPs
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P4
  - [-1, 2, C3k2, [512, True]]  # 19 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]  # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 22 (P5/32-large)
  - [[16, 19, 22], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""


# =============================================================================
# ARCH 3: yolo26-p2  --  TRIED
#   What:
#     Stock four-level YOLO26 P2 graph (strides 4/8/16/32), nearest upsampling.
#   Why:
#     Ultralytics ships a native four-level YOLO26 P2 graph, so the stride-4 level that had to be hand-
#     built for YOLOv12s comes for free here. Tests the anchor-supply argument directly on YOLO26: does
#     adding the stride-4 level alone help?
#   Parameters, unfused (3 classes, scale s): 9.66M, strides 4/8/16/32
# =============================================================================
ARCH_YOLO26_P2 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:  # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  s: [0.50, 0.50, 1024]  # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P2
  - [-1, 2, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]  # cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]  # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28 (P5/32-large)
  - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""


# =============================================================================
# ARCH 4: yolo26-p2dys  --  TRIED
#   What:
#     P2 graph with DySample for P3->P2 (P2 + DySample alone).
#   Why:
#     DySample had been the largest single architectural effect on YOLOv12s. This pairs it with the
#     native P2 graph: content-aware P3 -> P2 upsampling, everything else identical, so it is a one-
#     variable test of DySample on YOLO26. This 'P2 + DySample' graph is the base all later YOLO26
#     variants build on.
#   Parameters, unfused (3 classes, scale s): 9.67M, strides 4/8/16/32
# =============================================================================
ARCH_YOLO26_P2DYS = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 2, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]  # 26
  - [[-1, 10], 1, Concat, [1]]  # 27 cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28 (P5/32-large)
  - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""


# =============================================================================
# ARCH 5: yolo26-p2dys-wide  --  TRIED
#   What:
#     P2 + DySample with the P2 branch widened from 128 to 256 channels (width instead of depth).
#   Why:
#     Capacity probe at the finest level by width: the P2 branch goes from 128 to 256 channels. If P2
#     was capacity-starved, a wider branch should help.
#   Outcome:
#     Widening did not beat the matched control; it is the reason the reallocation later spends capacity
#     on depth instead.
#   Parameters, unfused (3 classes, scale s): 10.47M, strides 4/8/16/32
# =============================================================================
ARCH_YOLO26_P2DYS_WIDE = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True
reg_max: 1
scales:  # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  s: [0.50, 0.50, 1024]  # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P2
  - [-1, 2, C3k2, [256, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [256, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]  # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28 (P5/32-large)
  - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""


# =============================================================================
# ARCH 6: yolo26-p2dys-gctx  --  TRIED (transfer from YOLOv12s)
#   What:
#     P2 + DySample + ZGGlobalContext2 on P2 (YOLOv12s global-context block on the P2/DySample graph).
#   Why:
#     Transfer of the YOLOv12s global-context block onto the P2 + DySample graph (one of the two
#     placements in Appendix A.2), without any snake. Asks whether the block that helps YOLOv12s small
#     objects adds anything on top of P2 + DySample.
#   Outcome:
#     Null at the run-to-run noise level and below P2 + DySample alone (Appendix A.2).
#   Parameters, unfused (3 classes, scale s): 9.67M, strides 4/8/16/32
# =============================================================================
ARCH_YOLO26_P2DYS_GCTX = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True
reg_max: 1
scales:  # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  s: [0.50, 0.50, 1024]  # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P2
  - [-1, 2, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]  # cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]  # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28 (P5/32-large)
  - [19, 1, ZGGlobalContext2, [128]]  # 29 P2 + gated global context
  - [[29, 22, 25, 28], 1, Detect, [nc]]
"""


# =============================================================================
# ARCH 7: yolo26-lsshift  --  TRIED (transfer from YOLOv12s)
#   What:
#     YOLOv12s level-specific head ported to YOLO26: P2 + DySample, ZGGlobalContext2 on P2, snake k=9 on
#     P3-P5.
#   Why:
#     Full transfer of the YOLOv12s level-specific head to YOLO26: same three additions as on YOLOv12s
#     (DySample, global context at P2, snake k = 9 at P3-P5), built on YOLO26's native P2 graph. Tests
#     whether the YOLOv12s architecture carries over to the other detector.
#   Outcome:
#     Negligible effect on small-object mAP@50 on YOLO26s, against a clear gain on YOLOv12s (Table 9).
#   Parameters, unfused (3 classes, scale s): 12.09M, strides 4/8/16/32
# =============================================================================
ARCH_YOLO26_LSSHIFT = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # YOLO26 one2one head — NMS-free
reg_max: 1  # DFL-free; the third loss term is L1 on ltrb
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 1, C2PSA, [1024, 1]]  # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 2, C3k2, [128, True]]  # 19 (P2/4)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]]  # 26
  - [[-1, 10], 1, Concat, [1]]  # 27
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28 (P5/32)
  - [19, 1, ZGGlobalContext2, [128]]  # 29 P2 + gated avg/max global context
  - [22, 1, ZGDSConv, [256, 9]]  # 30 P3 + snake k=9
  - [25, 1, ZGDSConv, [512, 9]]  # 31 P4 + snake k=9
  - [28, 1, ZGDSConv, [1024, 9]]  # 32 P5 + snake k=9
  - [[29, 30, 31, 32], 1, Detect, [nc]]  # 33 Detect(P2, P3, P4, P5)
"""


# =============================================================================
# ARCH 8: yolo26-p2dys-p234  --  TRIED
#   What:
#     P2 + DySample with the stride-32 detection branch removed (detect on 4/8/16), depth 2 at P2 and
#     P3.
#   Why:
#     One-variable test of the level-reallocation premise: identical to yolo26-p2dys except that the
#     stride-32 detection branch is removed (stride-32 backbone kept). If stride 32 is really vestigial,
#     removing it should cost little, especially on large objects.
#   Parameters, unfused (3 classes, scale s): 6.99M, strides 4/8/16
# =============================================================================
ARCH_YOLO26_P2DYS_P234 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10 <- still feeds the top-down path at row 11
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 2, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# ARCH 9: yolo26-p2dys-p234-d4-2  --  TRIED
#   What:
#     As yolo26-p2dys-p234 with depth 4 at P2 and 2 at P3.
#   Why:
#     Attribution run. The final graph deepens P2 and P3 together, so a gain could come from either.
#     Deepening only P2 (4, 2) separates the two: (4,2) - (2,2) is the P2 contribution and (4,4) - (4,2)
#     the P3 contribution.
#   Parameters, unfused (3 classes, scale s): 7.01M, strides 4/8/16
# =============================================================================
ARCH_YOLO26_P2DYS_P234_D4_2 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10 <- still feeds the top-down path at row 11
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 4, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small) <- held at 2, unlike p234rich
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# ARCH 10: yolo26-p2dys-p234-d6-2  --  TRIED
#   What:
#     As yolo26-p2dys-p234 with depth 6 at P2 and 2 at P3.
#   Why:
#     Pushes the P2-only depth further (6, 2). In the attribution step, P2 depth improved localization
#     and the small-object recall ceiling, whereas P3 depth bought small-object mAP@50 at a recall cost.
#     This tests whether the P2 half keeps improving without paying the P3 cost.
#   Parameters, unfused (3 classes, scale s): 7.02M, strides 4/8/16
# =============================================================================
ARCH_YOLO26_P2DYS_P234_D6_2 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10 <- still feeds the top-down path at row 11
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 6, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 2, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# ARCH 11: yolo26-p2dys-p234-d6-6  --  TRIED
#   What:
#     As yolo26-p2dys-p234 with depth 6 at P2 and P3.
#   Why:
#     Third point on the depth axis (both levels at 6) to tell a monotone trend from a peak: two points
#     make a line, three show whether it turns over.
#   Outcome:
#     Inferior to 4 repeats on every metric, so 4 is a bracketed optimum (Section 4.2.4, Table 6).
#   Parameters, unfused (3 classes, scale s): 7.13M, strides 4/8/16
# =============================================================================
ARCH_YOLO26_P2DYS_P234_D6_6 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
end2end: True  # whether to use end-to-end mode
reg_max: 1  # DFL bins
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]  # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 9
  - [-1, 2, C2PSA, [1024]]  # 10 <- still feeds the top-down path at row 11
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
  - [[-1, 6], 1, Concat, [1]]  # 12 cat backbone P4
  - [-1, 2, C3k2, [512, True]]  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
  - [[-1, 4], 1, Concat, [1]]  # 15 cat backbone P3
  - [-1, 2, C3k2, [256, True]]  # 16 (P3/8-small)
  - [-1, 1, DySample, [2]]  # 17 content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]]  # 18 cat backbone P2
  - [-1, 6, C3k2, [128, True]]  # 19 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]  # 20
  - [[-1, 16], 1, Concat, [1]]  # 21 cat head P3
  - [-1, 6, C3k2, [256, True]]  # 22 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]  # 23
  - [[-1, 13], 1, Concat, [1]]  # 24 cat head P4
  - [-1, 2, C3k2, [512, True]]  # 25 (P4/16-medium)
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# THE RUNS — in order; comment an entry out to skip it
# =============================================================================
ARCH_RUNS = [
    {"name": "yolo26-p2dys-p234rich", "desc": "FINAL: P2 + DySample, stride-32 detection removed, P2/P3 depth 4",
     "yaml_content": ARCH_YOLO26_P2DYS_P234RICH, "batch": BATCH},
    {"name": "yolo26-stock", "desc": "BASELINE: stock YOLO26-s graph",
     "yaml_content": ARCH_YOLO26_STOCK, "batch": BATCH},
    {"name": "yolo26-p2", "desc": "stock four-level P2 graph, nearest upsampling",
     "yaml_content": ARCH_YOLO26_P2, "batch": BATCH},
    {"name": "yolo26-p2dys", "desc": "P2 + DySample",
     "yaml_content": ARCH_YOLO26_P2DYS, "batch": BATCH},
    {"name": "yolo26-p2dys-wide", "desc": "P2 + DySample, P2 branch widened 128 -> 256",
     "yaml_content": ARCH_YOLO26_P2DYS_WIDE, "batch": BATCH},
    {"name": "yolo26-p2dys-gctx", "desc": "P2 + DySample + global context on P2 (from YOLOv12s)",
     "yaml_content": ARCH_YOLO26_P2DYS_GCTX, "batch": BATCH},
    {"name": "yolo26-lsshift", "desc": "YOLOv12s level-specific head ported to YOLO26",
     "yaml_content": ARCH_YOLO26_LSSHIFT, "batch": BATCH},
    {"name": "yolo26-p2dys-p234", "desc": "P2 + DySample, stride-32 detection removed, depth 2",
     "yaml_content": ARCH_YOLO26_P2DYS_P234, "batch": BATCH},
    {"name": "yolo26-p2dys-p234-d4-2", "desc": "p234 with depth 4 at P2, 2 at P3",
     "yaml_content": ARCH_YOLO26_P2DYS_P234_D4_2, "batch": BATCH},
    {"name": "yolo26-p2dys-p234-d6-2", "desc": "p234 with depth 6 at P2, 2 at P3",
     "yaml_content": ARCH_YOLO26_P2DYS_P234_D6_2, "batch": BATCH},
    {"name": "yolo26-p2dys-p234-d6-6", "desc": "p234 with depth 6 at P2 and P3",
     "yaml_content": ARCH_YOLO26_P2DYS_P234_D6_6, "batch": BATCH},
]


# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def remap_pan(model):
    """Graphs with an inserted P2 level: copy checkpoint rows 17-22 to rows 23-28."""
    ref = YOLO(WEIGHTS).model.float()
    if len(model.model.model) <= len(ref.model):
        return  # three-level graph: nothing is shifted
    sd, moved = model.model.state_dict(), {}
    for k, v in ref.state_dict().items():
        parts = k.split(".", 2)  # model.<row>.<rest>
        if parts[0] == "model" and parts[1].isdigit() and int(parts[1]) in REMAP_ROWS:
            k2 = f"model.{int(parts[1]) + REMAP_SHIFT}.{parts[2]}"
            if k2 in sd and sd[k2].shape == v.shape:
                moved[k2] = v
    model.model.load_state_dict(moved, strict=False)
    print(f"  [remap] {len(moved)} tensors copied from checkpoint rows 17-22 to 23-28")


def init_from_pretrained(model):
    """Copy every layer of WEIGHTS whose shape matches; report how much of the model that is."""
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.load(WEIGHTS)
    remap_pan(model)
    after = model.model.state_dict()
    moved = sum(after[k].numel() for k in before if not torch.equal(before[k].float(), after[k].float()))
    print(f"  [init] {moved / sum(v.numel() for v in after.values()):.1%} of parameters from {WEIGHTS}")


def test_eval(save_dir, batch):
    w = Path(save_dir) / "weights" / "best.pt"
    r = YOLO(str(w)).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch, device=DEVICE,
                         workers=WORKERS, plots=False, verbose=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


def run_arch_experiment(run):
    # the file name must contain 'yolo26s' so ultralytics picks scale s
    yaml_path = os.path.join(YAML_DIR, f"{run['name'].replace('yolo26-', 'yolo26s-')}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 78}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: ARCHITECTURE (stock loss)   Batch: {run['batch']}")
    print(f"{'#' * 78}\n")
    start_time = time.time()
    out = {"name": run["name"], "desc": run["desc"], "batch": run["batch"], "status": "OK",
           "test_mAP50": None, "test_mAP50-95": None}
    try:
        model = YOLO(yaml_path)
        init_from_pretrained(model)
        model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=run["batch"], device=DEVICE,
            workers=WORKERS, project=PROJECT_DIR, name=run["name"], exist_ok=True,
            patience=100, close_mosaic=10, seed=SEED, deterministic=True,
            **STOCK_LOSS,
        )
        out["test_mAP50"], out["test_mAP50-95"] = test_eval(model.trainer.save_dir, run["batch"])
    except Exception as e:
        out["status"] = f"FAILED: {e}"
        print(f"\n  FAILED: {run['name']} -- {e}")
    finally:
        out["hours"] = round((time.time() - start_time) / 3600, 2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main():
    total_start = time.time()
    print(f"\n{'=' * 78}")
    print(f"  YOLO26-s ARCHITECTURE SWEEP — {len(ARCH_RUNS)} graphs, stock loss")
    print(f"{'=' * 78}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs: {EPOCHS}   ImgSize: {IMG_SIZE}   Weights: {WEIGHTS}")
    print(f"{'=' * 78}")
    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i + 1:>2}] {run['name']:<36} batch={run['batch']}  {run['desc']}")
    print(f"{'=' * 78}\n")

    results, out_path = [], f"{PROJECT_DIR}_results.json"
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Run {i + 1}/{len(ARCH_RUNS)}: {run['name']}")
        results.append(run_arch_experiment(run))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 78}")
    print(f"  ALL DONE ({(time.time() - total_start) / 3600:.2f}h)   test split")
    print(f"{'=' * 78}")
    print(f"  {'architecture':<36}{'mAP50':>8}{'mAP50-95':>10}{'hours':>7}")
    for r in results:
        if r["status"] == "OK":
            print(f"  {r['name']:<36}{r['test_mAP50']:>8.2f}{r['test_mAP50-95']:>10.2f}{r['hours']:>7.2f}")
        else:
            print(f"  {r['name']:<36}  {r['status']}")
    print(f"\n  results -> {out_path}\n")


if __name__ == "__main__":
    main()
