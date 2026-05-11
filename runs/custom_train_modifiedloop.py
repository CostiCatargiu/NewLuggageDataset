# custom_train_top3_final.py
"""
Final training runs for the Top-3 architectures.
Fixed training parameters, optimized for small object detection.

Top 3 architectures (from 25 experiments):
  1. arch_A_p2_head:     mAP50=0.5989, small50=0.3542, small95=0.1668
  2. p2_v5_auxiliary:    mAP50=0.5949, small50=0.3535, small95=0.1656
  3. p2_b1_deeper_aux3:  mAP50=0.5949, small50=0.3327, small95=0.1675

Training config (improved over previous runs):
  - Resolution: 800 (was 640) → small objects get +56% more pixels
  - Epochs: 300 (was 200-250) → longer convergence for P2 features
  - Copy-paste: 0.3 → explicitly augments small objects
  - Scale: 0.7 → more multi-scale variation
  - Close mosaic: 30 → longer clean fine-tuning at end
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path


# ============================================================
# CONSTANTS
# ============================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubsetNEW/data.yaml"
PROJECT_DIR = "runs_top3_final_luggage_newdataset"
YAML_DIR = "ultralytics/cfg/models/v12"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8

# --- Fixed training parameters for all 3 runs ---
EPOCHS = 80
IMGSZ = 640
PATIENCE = 20
CLOSE_MOSAIC = 10
COPY_PASTE = 0.3
MIXUP = 0.1
SCALE = 0.7
TRANSLATE = 0.2
DEGREES = 5.0


# ============================================================
# ARCHITECTURES
# ============================================================

ARCH_LUG2 = """# Original YOLOv12s (3 detection heads)
# YOLOv12-LSKA-P3: Single LSKA replacement at P3 detection feature
nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fLSKA, [256, False]]          # 14 — LSKA at P3 detection feature

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_LUG3= """#  LD-Redistribute
nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 3, C3k2,  [512, False, 0.25]]      # 4 — 2→3 (more P3 capacity)
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — keep 4 reps (P4 unchanged)
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 2, A2C2f, [1024, True, 1]]         # 8 — 4→2 (moderate P5 reduction)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # 14 — 2→3 (more P3 head capacity)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — keep 2 reps

  - [[14, 17, 20], 1, Detect, [nc]]
  """


ARCH_LUG4= """#  CBAM


# Parameters
nc: 3  # number of classes: backpack, bag, trolley
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.50, 1024]  # nano - fast experiments

# YOLOv12 backbone with strategic LuggageCBAM integration
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 2, C2fCBAM,  [512, False]]         # 4 - LuggageCBAM for P3 (small objects 27%)
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 - keep A2C2f area-attention for P4
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8 - keep A2C2f for P5

# YOLOv12 head with LuggageCBAM at critical fusion points
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]               # cat backbone P4
  - [-1, 2, C2fCBAM, [512, False]]          # 11 - LuggageCBAM for P4 fusion (medium objects 57%)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]               # cat backbone P3
  - [-1, 2, C2fCBAM, [256, False]]          # 14 - LuggageCBAM for P3 output (small objects)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]              # cat head P4
  - [-1, 2, C2fCBAM, [512, False]]          # 17 - LuggageCBAM for P4 output (medium objects)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]               # cat head P5
  - [-1, 2, C3k2, [1024, True]]             # 20 (P5/32-large) - keep C3k2 for large objects

  - [[14, 17, 20], 1, Detect, [nc]]         # Detect(P3, P4, P5)ct(P3, P4, P5)         # Detect(P3, P4, P5)
"""

ARCH_LUG5 = """#cbam2

# Dataset: backpack (27%), bag (22%), trolley (51%)
# Sizes: Small 27% | Medium 57% | Large 16%

nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, LuggageCBAMv2, [8]]             # 12 - CBAMv2 at P4 (reduction=8)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 15
  - [-1, 1, LuggageCBAMv2, [8]]             # 16 - CBAMv2 at P3 (reduction=8)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]              # cat with P4 after CBAM
  - [-1, 2, A2C2f, [512, False, -1]]        # 19

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 22

  - [[16, 19, 22], 1, Detect, [nc]]         # Detect(P3, P4, P5)
"""

ARCH_LUG5= """#  LD-Redistribute +SKA
# YOLOv12 LD + SKA Cascade
# 
# LD's redistribution (backbone P3 +1 rep, backbone P5 -2 reps, head P3 +1 rep)
# + SKA's large-kernel context refinement at head P3 detection feature
#
# Previous isolated results:
# - LD alone: +0.3% mAP50, +1.5% recall, +4.9% large
# - SKA alone: +0.05% mAP50, +0.8% small, -2.4% large
# - Goal: keep LD's recall/large gain + add SKA's small-object refinement

nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 3, C3k2,  [512, False, 0.25]]      # 4 — LD: 2→3
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 — unchanged
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 2, A2C2f, [1024, True, 1]]         # 8 — LD: 4→2

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # 14 — LD: 2→3
  - [-1, 1, C2fLSKA, [256, False]]          # 15 — SKA refinement

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 18

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 21

  - [[15, 18, 21], 1, Detect, [nc]]
"""

ARCH_LUG = """# Original YOLOv12s (3 detection heads)
# Parameters
nc: 3  # number of classes: backpack, bag, trolley
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.50, 1024]  # nano

# YOLOv12 backbone - standard
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]      # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]       # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4
  - [-1, 1, Conv,  [512, 3, 2]]             # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]          # 6 - A2C2f area-attention
  - [-1, 1, Conv,  [1024, 3, 2]]            # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8 - A2C2f for P5

# YOLOv12 head with ChannelSE after P3/P4 fusion
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]               # cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 - A2C2f for P4 fusion
  - [-1, 1, ChannelSE, []]                  # 12 - SE: reweight P4 channels

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]               # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]        # 15 - A2C2f for P3 fusion
  - [-1, 1, ChannelSE, []]                  # 16 - SE: reweight P3 channels (small objects)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]              # cat head P4 (after SE)
  - [-1, 2, A2C2f, [512, False, -1]]        # 19 - A2C2f for P4 output
  - [-1, 1, ChannelSE, []]                  # 20 - SE: reweight P4 output (medium objects)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]               # cat head P5
  - [-1, 2, C3k2, [1024, True]]             # 23 (P5/32-large)

  - [[16, 20, 23], 1, Detect, [nc]]         # Detect(P3, P4, P5)
"""

# ============================================================
# ARCHITECTURES
# ============================================================
ARCH_ORIGINAL = """# Original YOLOv12s (3 detection heads)
# YOLOv12-turbo + FSUS
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

# YOLO12-turbo head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]] # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P4
  - [-1, 2, A2C2f, [512, False, -1]] # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""

ARCH_BIDI_P4 = """# Bidirectional P4: double refinement at the dominant object scale
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]     # P4 backbone features  [layer 6]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]    # P5 backbone  [layer 8]

head:
  # top-down pass
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]   # first P4 refinement  [layer 11]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]   # P3 features  [layer 14]

  # bottom-up pass
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 3, A2C2f, [512, False, -1]]   # second P4 refinement: top-down + bottom-up  [layer 17]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]        # P5  [layer 20]

  # P4 gets a third fusion: merge first and second refinements
  - [[11, 17], 1, Concat, [1]]         # [layer 21] — bidirectional P4 merge
  - [-1, 1, Conv, [512, 1, 1]]         # [layer 22] — project back to 512

  - [[14, 22, 20], 1, Detect, [nc]]    # P3, P4-bidi, P5
"""

ARCH_CROSS_SCALE = """# Cross-scale fusion: P3 and P4 exchange context before detection
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  # FPN top-down
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]   # P4: 256ch, 16x16  [layer 11]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [128, False, -1]]   # P3: 128ch, 32x32  [layer 14]

  # P4→P3: P4 is 16x16, P3 is 32x32 → upsample P4 to 32x32, project 256→128
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]  # [layer 15] 16x16→32x32
  - [-1, 1, Conv, [128, 1, 1]]                  # [layer 16] 256→128ch
  - [[14, 16], 1, Concat, [1]]                  # [layer 17] 32x32, 256ch
  - [-1, 2, C3k2, [128, False, 0.5]]            # [layer 18] fused P3, 128ch, 32x32

  # P3→P4: P3 is 32x32, P4 is 16x16 → stride-2 conv, project 128→256
  - [14, 1, Conv, [256, 3, 2]]                  # [layer 19] 32x32→16x16, 256ch
  - [[11, 19], 1, Concat, [1]]                  # [layer 20] 16x16, 512ch
  - [-1, 2, A2C2f, [256, False, -1]]            # [layer 21] fused P4, 256ch, 16x16

  # PAN bottom-up using fused features
  - [18, 1, Conv, [256, 3, 2]]                  # [layer 22] 32x32→16x16
  - [[-1, 21], 1, Concat, [1]]                  # [layer 23] 16x16, 512ch
  - [-1, 2, A2C2f, [256, False, -1]]            # [layer 24] 256ch, 16x16

  - [-1, 1, Conv, [256, 3, 2]]                  # [layer 25] 16x16→8x8
  - [[-1, 8], 1, Concat, [1]]                   # [layer 26] 8x8
  - [-1, 2, C3k2, [512, True]]                  # [layer 27] 512ch, 8x8

  - [[18, 24, 27], 1, Detect, [nc]]
"""

ARCH_BIDI_LATERAL = """# Bidi-Lateral: bidirectional P4 refinement + P3/P4 cross-scale context
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]    # [layer 8] P5: 512ch 8x8

head:
  # ── FPN top-down ──────────────────────────────────────────────────
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]   # [layer 11] P4 top-down: 256ch 16x16

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [128, False, -1]]   # [layer 14] P3 top-down: 128ch 32x32

  # ── cross-scale lateral: P4 → P3 ─────────────────────────────────
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]  # [layer 15] P4 16x16→32x32
  - [15, 1, Conv, [128, 1, 1]]                  # [layer 16] 256→128ch
  - [[14, 16], 1, Concat, [1]]                  # [layer 17] 256ch 32x32
  - [-1, 2, C3k2, [128, False, 0.5]]            # [layer 18] enriched P3: 128ch 32x32

  # ── PAN bottom-up ─────────────────────────────────────────────────
  - [18, 1, Conv, [128, 3, 2]]                  # [layer 19] 32x32→16x16
  - [[-1, 11], 1, Concat, [1]]                  # [layer 20] 384ch 16x16
  - [-1, 2, A2C2f, [256, False, -1]]            # [layer 21] P4 bottom-up: 256ch 16x16

  # ── bidirectional P4 merge ────────────────────────────────────────
  # layer 11 = P4 top-down, layer 21 = P4 bottom-up (now also has P3 context via layer 18)
  - [[11, 21], 1, Concat, [1]]                  # [layer 22] 512ch 16x16
  - [-1, 1, Conv, [256, 1, 1]]                  # [layer 23] project→256ch 16x16

  # ── continue to P5 ────────────────────────────────────────────────
  - [21, 1, Conv, [256, 3, 2]]                  # [layer 24] 16x16→8x8
  - [[-1, 8], 1, Concat, [1]]                   # [layer 25] 768ch 8x8
  - [-1, 2, C3k2, [512, True]]                  # [layer 26] P5: 512ch 8x8

  - [[18, 23, 26], 1, Detect, [nc]]             # P3(128ch 32x32), P4-bidi(256ch 16x16), P5(512ch 8x8)
"""

ARCH_BIDI_LATERAL = """# Bidi-Lateral: bidirectional P4 + P3/P4 cross-scale context
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]   # [layer 11] P4 top-down: 256ch 16x16

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [128, False, -1]]   # [layer 14] P3 top-down: 128ch 32x32

  # P4→P3 lateral
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]  # [layer 15] 16x16→32x32
  - [15, 1, Conv, [128, 1, 1]]                  # [layer 16] 256→128ch
  - [[14, 16], 1, Concat, [1]]                  # [layer 17] 256ch 32x32
  - [-1, 2, C3k2, [128, False, 0.5]]            # [layer 18] enriched P3: 128ch 32x32

  # PAN bottom-up
  - [18, 1, Conv, [128, 3, 2]]                  # [layer 19] 32x32→16x16
  - [[-1, 11], 1, Concat, [1]]                  # [layer 20] 384ch 16x16
  - [-1, 2, A2C2f, [256, False, -1]]            # [layer 21] P4 bottom-up: 256ch 16x16

  # bidirectional P4 merge
  - [[11, 21], 1, Concat, [1]]                  # [layer 22] 512ch 16x16
  - [-1, 1, Conv, [256, 1, 1]]                  # [layer 23] 256ch 16x16

  - [21, 1, Conv, [256, 3, 2]]                  # [layer 24] 16x16→8x8
  - [[-1, 8], 1, Concat, [1]]                   # [layer 25] 768ch 8x8
  - [-1, 2, C3k2, [512, True]]                  # [layer 26] P5: 512ch 8x8

  - [[18, 23, 26], 1, Detect, [nc]]
"""


ARCH_DEEP_P3_BIDI = """# Deep P3 + Bidi P4: 3x A2C2f at P3, bidirectional merge at P4
nc: 3
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]   # [layer 11] P4 top-down: 256ch 16x16

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [128, False, -1]]   # [layer 14] P3: 3x blocks, 128ch 32x32

  # PAN bottom-up
  - [14, 1, Conv, [128, 3, 2]]         # [layer 15] 32x32→16x16
  - [[-1, 11], 1, Concat, [1]]         # [layer 16] 384ch 16x16
  - [-1, 2, A2C2f, [256, False, -1]]   # [layer 17] P4 bottom-up: 256ch 16x16

  # bidirectional P4 merge
  - [[11, 17], 1, Concat, [1]]         # [layer 18] 512ch 16x16
  - [-1, 1, Conv, [256, 1, 1]]         # [layer 19] 256ch 16x16

  - [17, 1, Conv, [256, 3, 2]]         # [layer 20] 16x16→8x8
  - [[-1, 8], 1, Concat, [1]]          # [layer 21] 768ch 8x8
  - [-1, 2, C3k2, [512, True]]         # [layer 22] P5: 512ch 8x8

  - [[14, 19, 22], 1, Detect, [nc]]
"""

# ============================================================
# 3 RUNS
# ============================================================

def debug_arch(yaml_content, name):
    import torch
    import tempfile

    yaml_path = f"/tmp/debug_{name}.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # patch tasks.py _predict_once to print shapes
    from ultralytics.nn import tasks as t
    original = t.DetectionModel._predict_once

    def patched(self, x, profile=False, visualize=False, embed=None):
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            try:
                x = m(x)
                shape = x.shape if isinstance(x, torch.Tensor) else [xi.shape for xi in x]
                print(f"  layer {i:2d} | {type(m).__name__:<15} | f={str(m.f):<12} | {shape}")
            except Exception as e:
                print(f"  layer {i:2d} | {type(m).__name__:<15} | f={str(m.f):<12} | FAILED: {e}")
                raise
            y.append(x if m.i in self.save else None)
        return x

    t.DetectionModel._predict_once = patched

    from ultralytics import YOLO
    try:
        model = YOLO(yaml_path)
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        t.DetectionModel._predict_once = original

debug_arch(ARCH_ORIGINAL, "original")


def check_repconv():
    from ultralytics.nn import modules
    import inspect

    # check if RepConv exists and what args it takes
    if hasattr(modules, 'RepConv'):
        print("RepConv found")
        print(inspect.signature(modules.RepConv.__init__))
    else:
        print("RepConv NOT found in ultralytics.nn.modules")

    # list all available modules
    print("\nAvailable modules containing 'rep' or 'Rep':")
    for name in dir(modules):
        if 'rep' in name.lower():
            print(f"  {name}")


check_repconv()

RUNS = [
    # {
    #     "name": "top0_ARCH_SKA",
    #     "yaml_content": ARCH_LUG2,
    #     "batch": 58,
    # },
    # {
    #     "name": "top0_ARCH_lug_LD",
    #     "yaml_content": ARCH_LUG3,
    #     "batch": 58,
    # },
    # {
    #     "name": "top0_ARCH_lug_CBAM",
    #     "yaml_content": ARCH_LUG4,
    #     "batch": 54,
    # },
    {
        "name": "top0_ARCH_lug_ORIGINAL",
        "yaml_content": ARCH_ORIGINAL,
        "batch": 58,
    },

    # {
    #     "name": "new3_bidi_lateral",
    #     "yaml_content": ARCH_BIDI_LATERAL,
    #     "batch": 58,
    # },
    # {
    #     "name": "new4_deep_p3_bidi",
    #     "yaml_content": ARCH_DEEP_P3_BIDI,
    #     "batch": 62,
    # }
]

# ============================================================
# MAIN
# ============================================================

def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def main():
    os.makedirs(YAML_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  TOP-3 FINAL TRAINING RUNS")
    print("=" * 70)
    print(f"  Resolution:   {IMGSZ}px")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Patience:     {PATIENCE}")
    print(f"  Close mosaic: last {CLOSE_MOSAIC} epochs")
    print(f"  Copy-paste:   {COPY_PASTE}")
    print(f"  Mixup:        {MIXUP}")
    print(f"  Scale:        {SCALE}")
    print(f"  Translate:    {TRANSLATE}")
    print(f"  Degrees:      {DEGREES}")
    print("=" * 70)

    results = {}

    for i, run in enumerate(RUNS):
        yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
        save_yaml(run["yaml_content"], yaml_path)

        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/3] 🚀 {run['name']}")
        print(f"  Batch: {run['batch']}")
        print(f"{'=' * 70}\n")

        try:
            model = YOLO(yaml_path)
            model.load("yolov12s.pt")
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=run["batch"],
                device=DEVICE,
                workers=WORKERS,
                project=PROJECT_DIR,
                name=run["name"],
                patience=PATIENCE,
                close_mosaic=CLOSE_MOSAIC,
                # copy_paste=COPY_PASTE,
                # mixup=MIXUP,
                # scale=SCALE,
                # translate=TRANSLATE,
                # degrees=DEGREES,
            )

            results[run["name"]] = "✅ Success"

        except Exception as e:
            results[run["name"]] = f"❌ Failed: {e}"
            print(f"❌ Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
