#!/usr/bin/env python3
"""
ARCHITECTURE PORT — the 5 best v5i @640 topologies, re-run on v6i.

=============================================================================
WHAT THIS IS AND WHY
=============================================================================
The arch campaign (53 runs) was on LuggageDataset.v5i with the STOCK loss —
no SWA, no LB-TAL, no NWD, nothing. So the architecture axis is completely
orthogonal to the 28-run v6i loss campaign, and neither result contaminates
the other. This script re-measures the five best v5i @640 topologies on v6i,
still with the stock loss, so the comparison is architecture-only.

v5i @640 results (anchor v12s_default2 = 0.5763):

    run                       v5i mAP50-95   d_anchor   levels  batch
    arch_levelspec                0.5802      +0.39       4       32
    arch_dysample_p2_gctx2        0.5791      +0.28       4       36
    arch_ls_shift                 0.5786      +0.23       4       32
    arch_gctx22                   0.5784      +0.21       4       28
    arch_ls_k5                    0.5779      +0.16       4       32

Plus two v6i-ADAPTED twins with no v5i number, testing the module re-derivation:
    arch_levelspec_v6   ZGDSConvV6(k=5,k_x=3) + ZGGlobalContext2V6, b32
    arch_ls_shift_v6    same two module swaps on the ls_shift tail, b32

READ THOSE DELTAS HONESTLY. The best is +0.39 pp and the rest are +0.16 to
+0.28. Measured seed noise on v6i is 0.12 pp. Four of these five are inside
2x that, so this sweep is asking "does the architecture family transfer at
all", not "which of the five is best" — the latter is not answerable at n=1
per config on either dataset.

For scale: the 28-run v6i LOSS campaign moved 0.5477 -> 0.5564 (+0.87 pp), and
the four 640->896 resolution pairs moved +1.25 to +1.64 pp. Architecture at
640 was the weakest of the three axes on v5i.

=============================================================================
WHAT ALL FIVE HAVE IN COMMON — and why it matters here
=============================================================================
EVERY ONE is a 4-LEVEL head, strides [4, 8, 16, 32]. They all add a P2 head.

That is not incidental. The v6i footprint diagnostic found small GTs are
SUPPLY-limited: 12.90 candidate anchors total (9.82 at P3, 2.46 at P4, 0.62
at P5), and no assigner can create candidates that do not exist — which is
why 28 loss runs could not move the small-object localisation ratio off 0.65.
A stride-4 level roughly quadruples that pool. These architectures attack the
constraint the loss campaign kept hitting, which is the strongest reason to
expect any transfer at all.

The differences between the five are small and specific:
    levelspec   gctx2 @P2,P3 (small)  +  ZGDSConv k=9 @P4,P5 (medium/large)
    dys_p2_gctx2 gctx2 at ALL four levels
    ls_shift    gctx2 @P2 only  +  ZGDSConv k=9 @P3,P4,P5
    gctx22      gctx2 at all four levels, but nn.Upsample instead of DySample
    ls_k5       gctx2 @P2,P3  +  ZGDSConv k=5 @P4,P5 (levelspec with k 9->5)

gctx22 vs dysample_p2_gctx2 is the clean DySample-vs-Upsample contrast:
identical tails, different upsampler. On v5i that was 0.5784 vs 0.5791.

=============================================================================
CONFOUND YOU MUST NOT IGNORE
=============================================================================
The five ran at DIFFERENT batch sizes on v5i (28/32/36) because the 4-level
heads have different VRAM footprints. Different batch = different step count
per epoch = a real confound on top of a 0.2-0.4 pp effect. The v6i loss
campaign ran at batch 54 throughout, so the existing v6i anchor (0.5477) is
NOT a valid reference for these runs.

A matched-batch reference (v12s_stock_b32) WAS in this file and has been
removed, because the loss ablation already provides a stock number: 0.5477.
That number ran at BATCH 54. At 9138 train images that is 170 optimiser steps
per epoch vs 286 at b32 — 68% more updates over 70 epochs, against an
architecture effect the v5i data puts at 0.16-0.39 pp. So comparisons against
0.5477 are topology + step count, entangled, and this sweep cannot separate
them. Matching b54 is not possible: a P2 head at 640 does not fit.

WHAT REMAINS CLEAN: the INTERNAL comparisons. arch_levelspec, arch_ls_k5,
arch_levelspec_v6, arch_ls_shift and arch_ls_shift_v6 all run at b32, so the
2x2 that isolates kernel size from the module re-derivation is unaffected.
Only dysample_p2_gctx2 (b36) and gctx22 (b28) carry an extra offset.

The v12s_stock_b32 block is preserved, commented, at the top of RUNS. Restore
it if any result lands within ~0.5 pp of 0.5477.

=============================================================================
LOSS CONFIG
=============================================================================
_ALL_OFF is passed EXPLICITLY on every run rather than relying on
default.yaml. default.yaml's shipped values are not neutral (small_obj_px 36,
alpha_start 0.9, ...), so an implicit run would silently carry SWA weighting
and the comparison against the v5i numbers would be meaningless.

NOTE FOR LATER: if you ever combine these with LB-TAL, the budget dicts are
keyed {8, 16, 32} and both budget paths do `.get(stride, min_level_k)`. With a
4-level head, STRIDE 4 SILENTLY FALLS BACK TO min_level_k = 1 — the level with
the most small-object candidates would get one slot. Any LB-TAL + P2 run needs
a `4:` entry, and the footprint re-run to know what to put in it.

Usage:
    python run_arch_port_v6i.py                 # all 7
    python run_arch_port_v6i.py v12s_stock_b32 arch_levelspec
"""

import sys
import time
import gc
import copy
import json
import os
import hashlib

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_arch_v6i"
YAML_DIR = "arch_yamls_v6i"
EPOCHS = 70
IMG_SIZE = 640
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100

# v6i references. The 0.5477 anchor ran at BATCH 54 — see the confound note.
V6I_ANCHOR_B54 = 0.5477
V6I_BEST_B54 = 0.5560          # cmb_p4wide, loss campaign, batch 54

OVERWRITE_EXISTING = False

# Stock loss, everything off. Passed explicitly — default.yaml is NOT neutral.
_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
    use_lbtal=False,
)

# =============================================================================
# TOPOLOGIES — copied verbatim from the v5i scripts
# =============================================================================
BACKBONE = """nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]               # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]        # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]       # 2  backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]        # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]       # 4  backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]              # 5
  - [-1, 4, A2C2f, [512, True, 4]]           # 6  backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]             # 7
  - [-1, 4, A2C2f, [1024, True, 1]]          # 8  backbone P5 (stride 32)
"""

HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                        # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, DySample, [2]]                        # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
"""

HEAD_STOCK = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
"""

TAIL_STOCK_3LVL = """  - [[14, 17, 20], 1, Detect, [nc]]                 # 21  stock 3-level head
"""

TAIL_LEVELSPEC = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2      (small)
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2      (small)
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake conv (medium)
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake conv (large)
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

TAIL_DYS_P2_GCTX2 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 + avg+max global ctx
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

TAIL_P2_GCTX2 = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]      # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 + avg+max global ctx
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

TAIL_LS_K5 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGDSConv, [512, 5]]                     # 26  P4 + snake k=5
  - [20, 1, ZGDSConv, [1024, 5]]                    # 27  P5 + snake k=5
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""


# =============================================================================
# v6i-ADAPTED TAILS — same topology, modules re-derived from v6i geometry
# =============================================================================
# ZGGlobalContext2 -> ZGGlobalContext2V6 : adds an attention-pooled descriptor
#   (zero-init score conv, so it STARTS exactly equal to the avg pool it
#   augments) because 640x360 letterboxes to 43.8% grey padding and mosaic
#   varies that fraction per sample.
# ZGDSConv k=9    -> ZGDSConvV6 k=5, k_x=3 : the snake steps one feature cell
#   per tap, and v6i's mean box is 39x55 px, i.e. 2.4x3.4 cells at P4 and
#   1.2x1.7 at P5 — k=9 traced 2.6-5.3x the object. k_x=round(5/1.55)=3
#   because v6i is 70.6% tall and only 6.0% wide.
#
# ATTRIBUTION THIS BUYS. arch_levelspec_v6 completes a 2x2 with two runs that
# already exist, so the two changes separate cleanly:
#
#                       ORIGINAL modules        V6 modules
#     k = 9             arch_levelspec          --
#     k = 5             arch_ls_k5              arch_levelspec_v6
#
#   arch_levelspec   vs arch_ls_k5        -> the kernel-size change alone
#   arch_ls_k5       vs arch_levelspec_v6 -> the module changes alone, at
#                                            MATCHED k=5
# arch_ls_shift_v6 is the second architecture, to check the module changes are
# not specific to one tail.

TAIL_LEVELSPEC_V6 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2V6, [256]]              # 24  P2 + attn-pooled ctx
  - [14, 1, ZGGlobalContext2V6, [256]]              # 25  P3 + attn-pooled ctx
  - [17, 1, ZGDSConvV6, [512, 5]]                   # 26  P4 + snake k=5, k_x=3
  - [20, 1, ZGDSConvV6, [1024, 5]]                  # 27  P5 + snake k=5, k_x=3
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

TAIL_LS_SHIFT_V6 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2V6, [256]]              # 24  P2 + attn-pooled ctx
  - [14, 1, ZGDSConvV6, [256, 5]]                   # 25  P3 + snake k=5, k_x=3
  - [17, 1, ZGDSConvV6, [512, 5]]                   # 26  P4 + snake k=5, k_x=3
  - [20, 1, ZGDSConvV6, [1024, 5]]                  # 27  P5 + snake k=5, k_x=3
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""


RUNS = [
    # -------------------------------------------------------------------------
    # REMOVED: v12s_stock_b32 (stock arch, stock loss, batch 32).
    #
    # Dropped because the loss ablation already has a stock number:
    # yolov12s_default = 0.5477. Accept the following when reading results.
    #
    # THE EXISTING ANCHOR RAN AT BATCH 54; these run at 28-36. At 9138 train
    # images that is 170 optimiser steps/epoch vs 286 at b32 — 68% MORE UPDATES
    # per epoch, for 70 epochs, against an architecture effect the v5i data puts
    # at 0.16-0.39 pp. The batch change is not a nuisance term here; it is
    # plausibly larger than the thing being measured.
    #
    # So "arch_X beat 0.5477 by +0.4" is NOT evidence the topology helped. It is
    # topology + step count, entangled, and this sweep cannot separate them.
    #
    # Matching batch to the anchor instead is not an option: the 4-level heads
    # were run at 28-36 on v5i because a P2 head at 640 will not fit at b54.
    #
    # IF a result lands within ~0.5 pp of 0.5477, run v12s_stock_b32 before
    # concluding anything — the block is preserved below, uncomment to restore.
    #
    # {"name": "v12s_stock_b32", "batch": 32, "levels": 3, "v5i": None,
    #  "yaml": BACKBONE + HEAD_STOCK + TAIL_STOCK_3LVL,
    #  "label": "stock yolov12s, stock loss, batch 32 — matched-batch reference",
    #  "why": "The only reference for arch_levelspec / ls_shift / ls_k5 / the "
    #         "two V6 twins, which all run at b32."},
    # -------------------------------------------------------------------------

    {"name": "arch_levelspec", "batch": 32, "levels": 4, "v5i": 0.5802,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LEVELSPEC,
     "label": "P2 head + gctx2@P2P3 + ZGDSConv k=9 @P4P5 — best v5i @640 (+0.39)",
     "why": "Top v5i 640 result. Per-level specialisation: global context where "
            "small objects live, snake conv where medium/large do. Batch matches "
            "v12s_stock_b32 exactly, so this is the cleanest single comparison "
            "in the file."},

    {"name": "arch_dysample_p2_gctx2", "batch": 36, "levels": 4, "v5i": 0.5791,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYS_P2_GCTX2,
     "label": "DySample + P2 head + gctx2 at all four levels (v5i +0.28)",
     "why": "gctx2 uniformly rather than per-level. Pairs with arch_gctx22 as "
            "the DySample-vs-Upsample contrast: identical tails, only the "
            "upsampler differs (v5i 0.5791 vs 0.5784). CONFOUND: b36 vs the "
            "b32 reference."},

    {"name": "arch_ls_shift", "batch": 32, "levels": 4, "v5i": 0.5786,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT,
     "label": "gctx2 @P2 only + ZGDSConv k=9 @P3P4P5 (v5i +0.23)",
     "why": "Pushes snake conv down to P3 and keeps global context only at the "
            "finest level. With levelspec and ls_k5 this makes a 3-point sweep "
            "of where the gctx2/snake boundary sits. Batch matches the reference."},

    {"name": "arch_gctx22", "batch": 28, "levels": 4, "v5i": 0.5784,
     "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX2,
     "label": "P2 head + gctx2 at all four levels, plain Upsample (v5i +0.21)",
     "why": "The v5i notes call gctx2 the only TWICE-replicated lever (+0.17 and "
            "+0.28). Both replications are inside v6i's 0.12 pp seed noise, so "
            "treat it as a hypothesis, not a known win. CONFOUND: b28 vs b32."},

    {"name": "arch_ls_k5", "batch": 32, "levels": 4, "v5i": 0.5779,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_K5,
     "label": "levelspec with snake kernel k=9 -> 5 (v5i +0.16)",
     "why": "Single-variable delta from arch_levelspec: kernel size only. On v5i "
            "the difference was 0.5802 vs 0.5779 = 0.23 pp, i.e. NOT resolved at "
            "n=1. Run it for the kernel-size reading, not because 5 or 9 is "
            "known to be right. Batch matches the reference. ALSO the ORIGINAL-"
            "module half of the 2x2 against arch_levelspec_v6."},

    # ---- v6i-adapted modules ------------------------------------------------
    {"name": "arch_levelspec_v6", "batch": 32, "levels": 4, "v5i": None,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LEVELSPEC_V6,
     "label": "levelspec with ZGDSConvV6 (k=5, k_x=3) + ZGGlobalContext2V6",
     "why": "The headline test of the module re-derivation. Identical topology "
            "to arch_levelspec; only the two modules change. Because it runs at "
            "k=5, comparing it with arch_ls_k5 (same k, ORIGINAL modules) "
            "isolates the module changes with kernel size held fixed, while "
            "arch_levelspec vs arch_ls_k5 isolates the kernel. Batch 32 matches "
            "both, so no step-count confound anywhere in that 2x2. "
            "PREDICTION IF THE DERIVATION IS RIGHT: gains concentrate in the "
            "SMALL and MEDIUM buckets (the snake now traces objects rather than "
            "background) and P50 should not fall — the attention pool is a "
            "strict superset of the average pool it replaces, zero-init."},

    {"name": "arch_ls_shift_v6", "batch": 32, "levels": 4, "v5i": None,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT_V6,
     "label": "ls_shift with the V6 modules — second tail, same two changes",
     "why": "Checks the module changes are not specific to one topology. This "
            "tail puts the snake at P3 as well, where v6i objects are 4.9x6.9 "
            "cells and k=9 was closest to matched (1.3x) — so if the k=9->5 "
            "argument is right, the gain here should be SMALLER than in "
            "arch_levelspec_v6, whose snakes sit at P4/P5 where k=9 was 2.6-5.3x "
            "oversized. A gain of the same size in both would mean the "
            "attention-pool change is doing the work, not the kernel."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def env_provenance():
    """Record the loss AND the custom modules these topologies depend on."""
    info = {"loss_path": None, "loss_md5": None, "modules": {}}
    try:
        import ultralytics.utils.loss as _lm
        p = getattr(_lm, "__file__", None)
        info["loss_path"] = p
        if p and os.path.exists(p):
            info["loss_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    except Exception as e:
        info["loss_error"] = str(e)
    # The real failure mode for this sweep is a missing custom module, not the
    # loss: these YAMLs reference blocks that live only in the fork.
    try:
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        for name in ("DySample", "ZGGlobalContext2", "ZGDSConv",
                     "ZGGlobalContext2V6", "ZGDSConvV6"):
            info["modules"][name] = bool(
                hasattr(M, name) or name in getattr(T, "__dict__", {}) or
                any(hasattr(getattr(M, sub, None), name) for sub in dir(M)))
    except Exception as e:
        info["modules_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  loss.py: {ENV.get('loss_path')}  md5={ENV.get('loss_md5')}")
    print(f"  custom modules: {ENV.get('modules')}")
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if missing or ENV.get("modules_error"):
        print(f"\n  [ABORT] custom modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        print("  These YAMLs reference blocks that exist only in your ultralytics")
        print("  fork. parse_model would fail at build time on every run. Check")
        print("  that the fork with DySample / ZGGlobalContext2 / ZGDSConv is the")
        print("  installed one before spending any hours here.")
        return False

    if not any(r["name"] == "v12s_stock_b32" for r in todo) and len(todo) > 1:
        print("\n  [warn] v12s_stock_b32 is NOT in this selection. The existing")
        print("         v6i anchor ran at batch 54; these run at 28-36, so with")
        print("         no matched-batch reference a +0.3 pp result cannot be")
        print("         separated from the batch change.")

    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))]
    if clash and OVERWRITE_EXISTING:
        print(f"\n  [warn] OVERWRITE_EXISTING=True — reusing: {', '.join(clash)}")
        clash = []
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        print("  Delete them, or set OVERWRITE_EXISTING=True at the top.")
        return False
    return True


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            try:
                crit.epoch = epoch
                if hasattr(crit, "_sync_bbox_loss_state"):
                    crit._sync_bbox_loss_state()
            except Exception:
                pass


def run_one(rc):
    name, batch = rc["name"], rc["batch"]
    yaml_path = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{name}.yaml"))
    print(f"\n{'=' * 76}\n  RUN {name}  ({rc['levels']} levels, batch {batch})\n"
          f"  {rc['label']}\n"
          f"  yaml={yaml_path}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    # Build from YAML, then load the pretrained backbone weights that match.
    model = YOLO(yaml_path)
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] could not transfer {MODEL_WEIGHTS}: {e}")
        print("         training from scratch — NOT comparable to the v5i runs,")
        print("         which all started from pretrained weights.")
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(_ALL_OFF))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")

    out = {"name": name, "batch": batch, "levels": rc["levels"], "seed": SEED,
           "v5i_map5095": rc.get("v5i"), "hours": hours,
           "save_dir": save_dir, "weights": weights, "yaml": yaml_path,
           "loss": "STOCK (_ALL_OFF passed explicitly)", "env": ENV,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_params.json"), "w") as f:
            json.dump({k: v for k, v in out.items() if k != "env"} |
                      {"yaml_text": rc["yaml"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                               batch=batch, device=DEVICE, workers=WORKERS,
                               project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    ref = next((r["test_map5095"] for r in res if r["name"] == "v12s_stock_b32"), None)
    matched = ref is not None
    if not matched:
        ref = V6I_ANCHOR_B54          # b54 anchor — CONFOUNDED, see the note below
    print(f"\n{'=' * 84}\n  ARCHITECTURE PORT — v6i @640, stock loss\n{'=' * 84}")
    print(f"{'run':<24}{'b':>4}{'lvl':>5}{'mAP50':>9}{'mAP50-95':>11}"
          f"{'vs b54anc' if not matched else 'vs b32ref':>11}{'v5i':>9}{'h':>6}")
    print("-" * 84)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        vr = "%+11.2f" % ((r["test_map5095"] - ref) * 100)
        v5 = (f"{r['v5i_map5095']:9.4f}" if r.get("v5i_map5095") else f"{'—':>9}")
        print(f"{r['name']:<24}{r['batch']:>4}{r['levels']:>5}"
              f"{r['test_map50']*100:>9.2f}{r['test_map5095']*100:>11.2f}{vr}{v5}{r['hours']:>6.1f}")

    print(f"\n  v6i loss-campaign references (BATCH 54):")
    print(f"    anchor {V6I_ANCHOR_B54*100:.2f} | best cmb_p4wide {V6I_BEST_B54*100:.2f}")
    if not matched:
        print("\n  [!!] THE 'vs' COLUMN IS CONFOUNDED. It compares b28-36 runs against")
        print("       a BATCH-54 anchor. At 9138 images that is 286 optimiser steps")
        print("       per epoch (b32) vs 170 (b54) — 68% more updates, for 70 epochs,")
        print("       against an architecture effect the v5i data puts at 0.16-0.39 pp.")
        print("       The batch change is plausibly LARGER than what is being measured.")
        print("       A positive number here is topology + step count, not topology.")
        print("       Anything landing within ~0.5 pp of the anchor: run")
        print("       v12s_stock_b32 (commented in RUNS) before concluding.")
        print("\n       What IS clean regardless: the internal comparisons, since")
        print("       arch_levelspec / arch_ls_k5 / arch_levelspec_v6 / arch_ls_shift")
        print("       / arch_ls_shift_v6 all run at b32.")
    print("\n  READ THIS BEFORE RANKING ANYTHING:")
    print("    seed noise on v6i is 0.12 pp. The v5i spread across these five was")
    print("    0.5779-0.5802 = 0.23 pp. Four of the five sit inside 2x noise, so")
    print("    this sweep answers 'does the P2-head family transfer', not 'which")
    print("    of the five wins'. If the family transfers, re-run the top one at")
    print("    2-3 seeds before believing an ordering.")
    print("\n    Note b36/b28 runs carry a step-count offset against the b32 ref.")
    print("\n  Per-size metrics: run CocoEvalAllFolders_luggage.py on:")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<24} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 84}\n  ARCHITECTURE PORT — {len(todo)} runs @640 on v6i, STOCK loss")
    print(f"  runs: {', '.join(r['name'] for r in todo)}")
    print(f"{'=' * 84}\n")
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "batch": r["batch"],
                        "levels": r["levels"], "seed": SEED,
                        "v5i_map5095": r.get("v5i"), "hours": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
