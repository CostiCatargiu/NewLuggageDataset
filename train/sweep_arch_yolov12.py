#!/usr/bin/env python3
"""
YOLOv12-s ARCHITECTURE SWEEP — 11 graphs, stock loss, LuggageDataset v6i
Environment A (YOLOv12 fork + ultralytics_patches/v12).

Every architecture evaluated for YOLOv12-s is written out below, with what it changes and
why it was tried. Only the graph changes between runs: the loss is stock (every custom term
switched off), the recipe is the same and the batch is the same (32) so all runs take the
same number of optimizer steps. Each graph starts from yolov12s.pt (all layers whose shape
matches are copied).

   #  architecture                       status
   1  yolov12s-lsshift                   FINAL (paper)
   2  yolov12s-stock                     BASELINE
   3  yolov12s-p2-gctx-all               TRIED
   4  yolov12s-p2dys-gctx-all            TRIED
   5  yolov12s-levelspec-k9              TRIED
   6  yolov12s-levelspec-k5              TRIED
   7  yolov12s-gctx-snake-stacked-p2p3   TRIED
   8  yolov12s-gctx-snake-p3-k11         TRIED
   9  yolov12s-lsshift-v6blocks          TRIED
  10  yolov12s-lsshift-p234              TRIED (transfer from YOLO26s)
  11  yolov12s-lsshift-p234rich          TRIED (transfer from YOLO26s)

The same configs, with the same descriptions, are in configs/yolov12/yolov12s_architectures.yaml.

Usage:
  python sweep_arch_yolov12.py            # all eleven, in order

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
PROJECT_DIR = "runs_arch_yolov12"
YAML_DIR = "arch_yamls_yolov12"
WEIGHTS = "yolov12s.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 70
BATCH = 32  # one batch for every graph, so step counts match
SEED = 0

# Every custom loss term OFF (= stock YOLOv12 loss). Passed explicitly because the
# patched default.yaml ships the paper's loss settings, not neutral values.
STOCK_LOSS = dict(
    small_obj_px=0, small_obj_boost=1.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
)

# =============================================================================
# ARCH 1: yolov12s-lsshift  --  FINAL (paper)
#   What:
#     Stride-4 (P2) level appended from row 14; DySample replaces nearest upsampling in the top-down
#     path and for P3->P2; ZGGlobalContext2 on P2; ZGDSConv snake k=9 on P3, P4, P5. Four detection
#     levels (strides 4/8/16/32).
#   Why:
#     Small luggage is supply-limited: a small box holds only ~12.90 anchor centres on the stock 3-level
#     head, so no assigner or loss can create positives that do not exist. A stride-4 level quadruples
#     that pool. DySample replaces nearest upsampling because blurring on the path into the finest level
#     destroys the detail small objects depend on. Blocks are then matched to what each level sees:
#     global context at P2, where a 12-px blob is identifiable only from its surroundings; a directional
#     snake kernel at P3-P5, where boxes are large enough for shape to be legible and are mostly tall
#     (70.6%). Pushing the snake down to P3 (instead of stopping at P4) gave the best overall result of
#     the level-split family.
#   Parameters, unfused (3 classes, scale s): 9.87M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_LSSHIFT = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]  # 25 P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]  # 26 P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]  # 27 P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 2: yolov12s-stock  --  BASELINE
#   What:
#     Stock YOLOv12-s graph written out explicitly (identical to the official model). Three detection
#     levels (8/16/32).
#   Why:
#     Reference graph. Written out explicitly so the custom graphs can be diffed against it line by
#     line, and used as a matched-batch control: the 4-level heads do not fit at the stock batch size,
#     so a stock run at the same reduced batch separates the effect of the topology from the effect of a
#     different number of optimizer steps.
#   Parameters, unfused (3 classes, scale s): 9.10M, strides 8/16/32
# =============================================================================
ARCH_YOLOV12S_STOCK = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [[14, 17, 20], 1, Detect, [nc]]  # 21 stock 3-level head
"""


# =============================================================================
# ARCH 3: yolov12s-p2-gctx-all  --  TRIED
#   What:
#     P2 level with nearest upsampling (no DySample); ZGGlobalContext2 on all four levels.
#   Why:
#     First 4-level design: add the stride-4 level and give every level the same global-context block.
#     Kept with plain nearest upsampling so that, paired with yolov12s-p2dys-gctx-all (identical except
#     DySample), it isolates the contribution of the upsampler.
#   Parameters, unfused (3 classes, scale s): 9.46M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_P2_GCTX_ALL = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, nn.Upsample, [None, 2, "nearest"]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]  # 25 P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]  # 26 P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]  # 27 P5 + avg+max global ctx
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 4: yolov12s-p2dys-gctx-all  --  TRIED
#   What:
#     P2 level + DySample; ZGGlobalContext2 on all four levels (global context everywhere instead of the
#     level split).
#   Why:
#     The DySample twin of yolov12s-p2-gctx-all: identical tail, content-aware upsampler. Also the 'one
#     block everywhere' control for the level-specific idea - if global context helped equally at every
#     level, a level split would be unnecessary.
#   Outcome:
#     Applying either block at every level was worse than the level split (paper, Section 4.2.3).
#   Parameters, unfused (3 classes, scale s): 9.49M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_P2DYS_GCTX_ALL = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]  # 25 P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]  # 26 P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]  # 27 P5 + avg+max global ctx
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 5: yolov12s-levelspec-k9  --  TRIED
#   What:
#     P2 + DySample; ZGGlobalContext2 on P2 and P3; snake k=9 on P4 and P5 (alternative level split).
#   Why:
#     First level-specific split: the two finest levels (P2, P3), where small objects are, get global
#     context; P4 and P5, where medium and large objects are, get the snake kernel. Tests the hypothesis
#     that small objects need scene context and larger ones need a shape prior. Its sibling
#     yolov12s-lsshift moves the boundary one level down (snake also at P3).
#   Parameters, unfused (3 classes, scale s): 9.84M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_LEVELSPEC_K9 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + gctx2 (small)
  - [14, 1, ZGGlobalContext2, [256]]  # 25 P3 + gctx2 (small)
  - [17, 1, ZGDSConv, [512, 9]]  # 26 P4 + snake conv (medium)
  - [20, 1, ZGDSConv, [1024, 9]]  # 27 P5 + snake conv (large)
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 6: yolov12s-levelspec-k5  --  TRIED
#   What:
#     As yolov12s-levelspec-k9 with snake kernel k=5 on P4 and P5.
#   Why:
#     Kernel-size probe on the level-specific split: snake k = 9 -> 5. Derived from the mean box (39x55
#     px), for which k = 9 spans several times the object at P4/P5, so a smaller, size-matched kernel
#     was the natural hypothesis.
#   Outcome:
#     k = 5 was below k = 9; kernel size behaved monotonically upward on this data (Table 6 selects k =
#     9).
#   Parameters, unfused (3 classes, scale s): 9.78M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_LEVELSPEC_K5 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]  # 25 P3 + gctx2
  - [17, 1, ZGDSConv, [512, 5]]  # 26 P4 + snake k=5
  - [20, 1, ZGDSConv, [1024, 5]]  # 27 P5 + snake k=5
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 7: yolov12s-gctx-snake-stacked-p2p3  --  TRIED
#   What:
#     P2 + DySample; ZGGlobalContext2 followed by snake k=9 (stacked) on both P2 and P3; snake k=9 on P4
#     and P5.
#   Why:
#     Instead of choosing between context and shape at a level, stack both (global context, then snake).
#     Stacking at P3 was the only 'both blocks' variant that did not lose, and snake coverage had been
#     monotonic in the number of levels, so this extends the stack to P2 as well, giving a snake at all
#     four levels.
#   Parameters, unfused (3 classes, scale s): 9.92M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_GCTX_SNAKE_STACKED_P2P3 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + gctx2
  - [24, 1, ZGDSConv, [256, 9]]  # 25 P2 + snake k=9 (STACKED)
  - [14, 1, ZGGlobalContext2, [256]]  # 26 P3 + gctx2
  - [26, 1, ZGDSConv, [256, 9]]  # 27 P3 + snake k=9 (STACKED)
  - [17, 1, ZGDSConv, [512, 9]]  # 28 P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]  # 29 P5 + snake k=9
  - [[25, 27, 28, 29], 1, Detect, [nc]]  # 30
"""


# =============================================================================
# ARCH 8: yolov12s-gctx-snake-p3-k11  --  TRIED
#   What:
#     P2 + DySample; ZGGlobalContext2 on P2; ZGGlobalContext2 + snake k=9 stacked on P3; snake k=11 on
#     P4 and P5.
#   Why:
#     Keeps the stacked P3 block unchanged and only enlarges the coarse snake (k = 9 -> 11 at P4/P5).
#     The stacked variants gained on small objects but lost on large ones; this targets the large-object
#     deficit where large objects live, without touching what produced the small-object gain.
#   Parameters, unfused (3 classes, scale s): 9.91M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_GCTX_SNAKE_P3_K11 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]  # 24 P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]  # 25 P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]  # 26 P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 11]]  # 27 P4 + snake k=11
  - [20, 1, ZGDSConv, [1024, 11]]  # 28 P5 + snake k=11
  - [[24, 26, 27, 28], 1, Detect, [nc]]  # 29
"""


# =============================================================================
# ARCH 9: yolov12s-lsshift-v6blocks  --  TRIED
#   What:
#     Final graph with alternative block versions: attention-pooled global context (ZGGlobalContext2V6)
#     on P2 and snake k=5 with 3-tap cross branch (ZGDSConvV6) on P3-P5.
#   Why:
#     Re-derivation of both blocks for this dataset: attention-weighted pooling in the context block
#     instead of plain average/max, and a snake kernel size-matched to the luggage boxes (k = 5 along
#     the axis, 3 across). Tests whether blocks tuned to the box statistics beat the generic ones on the
#     same graph.
#   Parameters, unfused (3 classes, scale s): 9.79M, strides 4/8/16/32
# =============================================================================
ARCH_YOLOV12S_LSSHIFT_V6BLOCKS = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [-1, 1, Conv, [512, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, C3k2, [1024, True]]  # 20 P5 head
  - [14, 1, DySample, [2]]  # 21 P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]  # 22 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 23 P2 head (stride 4)
  - [23, 1, ZGGlobalContext2V6, [256]]  # 24 P2 + attn-pooled ctx
  - [14, 1, ZGDSConvV6, [256, 5]]  # 25 P3 + snake k=5, k_x=3
  - [17, 1, ZGDSConvV6, [512, 5]]  # 26 P4 + snake k=5, k_x=3
  - [20, 1, ZGDSConvV6, [1024, 5]]  # 27 P5 + snake k=5, k_x=3
  - [[24, 25, 26, 27], 1, Detect, [nc]]  # 28
"""


# =============================================================================
# ARCH 10: yolov12s-lsshift-p234  --  TRIED (transfer from YOLO26s)
#   What:
#     Final YOLOv12s head with the stride-32 detection branch removed (detect on strides 4/8/16);
#     stride-32 backbone kept; depth 2.
#   Why:
#     Cross-detector transfer (Section 5.2.5). On YOLO26s the stride-32 detection branch draws almost no
#     positives (0.01 per small and 0.00 per medium ground truth; large objects take 7.96 of their 9.79
#     positives from stride 16), so removing it was free there. This applies the same deletion to the
#     YOLOv12s head, keeping the stride-32 backbone.
#   Outcome:
#     Costs YOLOv12s large-object accuracy (stride-32 head still carries large luggage; Appendix A.1).
#   Parameters, unfused (3 classes, scale s): 6.99M, strides 4/8/16
# =============================================================================
ARCH_YOLOV12S_LSSHIFT_P234 = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 2, A2C2f, [256, False, -1]]  # 14 P3 head <- depth 2
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [14, 1, DySample, [2]]  # 18 P3 head -> stride 4
  - [[18, 2], 1, Concat, [1]]  # 19 + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]  # 20 P2 head <- depth 2
  - [20, 1, ZGGlobalContext2, [256]]  # 21 P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]  # 22 P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]  # 23 P4 + snake k=9
  - [[21, 22, 23], 1, Detect, [nc]]  # 24 Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# ARCH 11: yolov12s-lsshift-p234rich  --  TRIED (transfer from YOLO26s)
#   What:
#     As yolov12s-lsshift-p234 with the P2 and P3 blocks deepened from 2 to 4 repeats (the YOLO26s level
#     reallocation applied to YOLOv12s).
#   Why:
#     Completes the transfer: the YOLO26s level reallocation in full - delete the stride-32 detection
#     branch and spend the freed capacity on depth at P2 and P3 (2 -> 4 repeats).
#   Outcome:
#     Negligible effect on small-object mAP@50 and a large-object cost (Table 9, Appendix A.1).
#   Parameters, unfused (3 classes, scale s): 7.10M, strides 4/8/16
# =============================================================================
ARCH_YOLOV12S_LSSHIFT_P234RICH = """\
nc: 3  # luggage classes (backpack, bag, trolley); data.yaml overrides it
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]  # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]  # 2 backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]  # 4 backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]  # 5
  - [-1, 4, A2C2f, [512, True, 4]]  # 6 backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7
  - [-1, 4, A2C2f, [1024, True, 1]]  # 8 backbone P5 (stride 32)
head:
  - [-1, 1, DySample, [2]]  # 9
  - [[-1, 6], 1, Concat, [1]]  # 10
  - [-1, 2, A2C2f, [512, False, -1]]  # 11
  - [-1, 1, DySample, [2]]  # 12
  - [[-1, 4], 1, Concat, [1]]  # 13
  - [-1, 4, A2C2f, [256, False, -1]]  # 14 P3 head <- depth 4
  - [-1, 1, Conv, [256, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, A2C2f, [512, False, -1]]  # 17 P4 head
  - [14, 1, DySample, [2]]  # 18 P3 head -> stride 4
  - [[18, 2], 1, Concat, [1]]  # 19 + backbone P2
  - [-1, 4, A2C2f, [256, False, -1]]  # 20 P2 head <- depth 4
  - [20, 1, ZGGlobalContext2, [256]]  # 21 P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]  # 22 P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]  # 23 P4 + snake k=9
  - [[21, 22, 23], 1, Detect, [nc]]  # 24 Detect(P2, P3, P4) — strides 4, 8, 16
"""


# =============================================================================
# THE RUNS — in order; comment an entry out to skip it
# =============================================================================
ARCH_RUNS = [
    {"name": "yolov12s-lsshift", "desc": "FINAL: P2 + DySample, global context on P2, snake k9 on P3-P5",
     "yaml_content": ARCH_YOLOV12S_LSSHIFT, "batch": BATCH},
    {"name": "yolov12s-stock", "desc": "BASELINE: stock YOLOv12-s graph",
     "yaml_content": ARCH_YOLOV12S_STOCK, "batch": BATCH},
    {"name": "yolov12s-p2-gctx-all", "desc": "P2 (nearest upsampling), global context on all four levels",
     "yaml_content": ARCH_YOLOV12S_P2_GCTX_ALL, "batch": BATCH},
    {"name": "yolov12s-p2dys-gctx-all", "desc": "P2 + DySample, global context on all four levels",
     "yaml_content": ARCH_YOLOV12S_P2DYS_GCTX_ALL, "batch": BATCH},
    {"name": "yolov12s-levelspec-k9", "desc": "P2 + DySample, global context P2+P3, snake k9 on P4+P5",
     "yaml_content": ARCH_YOLOV12S_LEVELSPEC_K9, "batch": BATCH},
    {"name": "yolov12s-levelspec-k5", "desc": "as levelspec-k9 with snake k5 on P4+P5",
     "yaml_content": ARCH_YOLOV12S_LEVELSPEC_K5, "batch": BATCH},
    {"name": "yolov12s-gctx-snake-stacked-p2p3", "desc": "global context + snake k9 stacked on P2 and P3",
     "yaml_content": ARCH_YOLOV12S_GCTX_SNAKE_STACKED_P2P3, "batch": BATCH},
    {"name": "yolov12s-gctx-snake-p3-k11", "desc": "context on P2, context+snake k9 on P3, snake k11 on P4+P5",
     "yaml_content": ARCH_YOLOV12S_GCTX_SNAKE_P3_K11, "batch": BATCH},
    {"name": "yolov12s-lsshift-v6blocks", "desc": "final graph with the V6 block versions",
     "yaml_content": ARCH_YOLOV12S_LSSHIFT_V6BLOCKS, "batch": BATCH},
    {"name": "yolov12s-lsshift-p234", "desc": "final head, stride-32 detection removed (4/8/16), depth 2",
     "yaml_content": ARCH_YOLOV12S_LSSHIFT_P234, "batch": BATCH},
    {"name": "yolov12s-lsshift-p234rich", "desc": "as lsshift-p234 with P2/P3 depth 4 (YOLO26 reallocation)",
     "yaml_content": ARCH_YOLOV12S_LSSHIFT_P234RICH, "batch": BATCH},
]


# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def init_from_pretrained(model):
    """Copy every layer of WEIGHTS whose shape matches; report how much of the model that is."""
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.load(WEIGHTS)
    after = model.model.state_dict()
    moved = sum(after[k].numel() for k in before if not torch.equal(before[k].float(), after[k].float()))
    print(f"  [init] {moved / sum(v.numel() for v in after.values()):.1%} of parameters from {WEIGHTS}")


def test_eval(save_dir, batch):
    w = Path(save_dir) / "weights" / "best.pt"
    r = YOLO(str(w)).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch, device=DEVICE,
                         workers=WORKERS, plots=False, verbose=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")  # name contains 'yolov12s' -> scale s
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
    print(f"  YOLOv12-s ARCHITECTURE SWEEP — {len(ARCH_RUNS)} graphs, stock loss")
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
