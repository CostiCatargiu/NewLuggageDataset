#!/usr/bin/env python3
"""
ARCH FINAL — the 3 most promising topologies, merged from refine2 + round2.

=============================================================================
WHERE THINGS STAND (49 runs, v6i, 640, all b32 unless noted)
=============================================================================
                        overall    small    large   loss
  ls_shift_gctxP3        0.5602   0.5158   0.5614   stock   <- best overall+small
  arch_ls_shift          0.5598   0.5122   0.6009   stock   <- best balanced
  ls_shift_wiou          0.5591   0.5151   0.5859   wiou
  ls_shift_k5            0.5587   0.5115   0.5932   stock
  ls_shift_sqrt          0.5561   0.5121   0.5942   sqrt0703
  ls_shift_gctxP3P4      0.5549   0.5100   0.5742   stock
  ls_shift_lbtalB        0.5514   0.5086   0.5562   SWA+LB-TAL
  yolov12s_sqrt0703      0.5564   0.5063   0.5999   (3-level, b54)

gctxP3's LARGE deficit is the thing to fix: 0.5614 vs ls_shift's 0.6009 is
-3.95 pp ABSOLUTE / -6.6% relative. (The round2 file called this "-2.75%" and
quoted the small gain as "+3.20%" against a DIFFERENT baseline -- small vs the
anchor, large vs ls_shift. Stated consistently against ls_shift, the trade is
small +0.36 pp for large -3.95 pp.)

FIVE MEASURED FACTS. All three configs come from them:

  (1) STACKING WON, REPLACING LOST -- and they are not the same knob.
      gctxP3   STACKS gctx2 -> snake at P3           -> +0.04 (best overall)
      gctxP3P4 REPLACES the snake with gctx2 @P3,P4  -> -0.49
      Only stacking has a win. Extending it is the highest-value move.

  (2) WIoU IS THE BETTER DEAL THAN gctxP3, and it is easy to miss.
      gctxP3 0.5602 / small 0.5158 / large 0.5614
      wiou   0.5591 / small 0.5151 / large 0.5859
      Overall differs by 0.11 pp (noise), small is a TIE, and WIoU keeps
      +2.45 pp more LARGE -- the small gain without the trade.

  (3) THE LOSS AXIS COLLAPSES ON THIS TOPOLOGY -- except WIoU.
      sqrt0703 was +0.86 pp on the stock 3-level model; here it is -0.37.
      lbtalB -0.84. WIoU -0.07 (neutral). Reading: SWA's small-object loss
      boost was COMPENSATING for what the stride-4 head now supplies directly.
      => WIoU is the ONLY loss lever still compatible.

  (4) KERNEL IS MONOTONIC UP: k=5 lost 0.22 pp on levelspec, 0.11 pp on
      ls_shift. k=11 untested. This contradicts the k=5 "size-matched"
      derivation I argued from the 39x55 px mean box, falsified on both tails.
      Following the measurement.

  (5) SNAKE COVERAGE is monotonic (0 -> 2 -> 3 levels: 0.5578 -> 0.5590 ->
      0.5598, replicated at k=5). P2 is the one level never given a snake.

=============================================================================
THE THREE
=============================================================================
  1. gctxp3_snake4     STACK gctx+snake at P2 as well as P3 (snake at all 4).
                       Combines (1) and (5), the two monotonic levers. Best
                       shot at beating gctxP3 on SMALL.
                       [from round2. NOTE that file contained this topology
                        TWICE under two names -- gctxp3_snake4 and
                        gctx_stack_p2p3 are byte-identical once comments are
                        stripped. Kept once.]

  2. gctxp3_k11coarse  gctxP3's P3 stack kept EXACTLY; only the coarse snake
                       goes k9 -> k11 at P4,P5. Targets the large deficit
                       where large objects live, without touching what
                       produced the small gain. The only config here that can
                       be a STRICT improvement over gctxP3.
                       [from round2. Preferred over refine2's gctxP3_k11,
                        which raised k at P3 too and so confounds the fix with
                        the thing being protected.]

  3. gctxp3_wiou       gctxP3 + WIoU v3, per (2) and (3).
                       [from refine2.]

DROPPED: gctx_stack_p2p3 (duplicate of #1); stackP3P4 (stacks at P4, where the
large problem is better addressed by #2 or by the box metric); gctxP3_k11
(k=11 everywhere confounds with #2's targeted version).

=============================================================================
EXPECTATION
=============================================================================
Round 2 (6 runs) bought +0.04 pp over the previous best. Across 49 runs the
usable band is 0.5477-0.5602 = 1.25 pp. These three plausibly buy 0.05-0.10 pp
and a "win" may sit inside the 0.12 pp seed floor. gctxp3_k11coarse is the one
with a shot at a MEANINGFUL result -- a better small/large trade rather than a
higher small number. Seed-confirm any winner before believing an ordering.

STILL UNTOUCHED: the 896 axis. Four controlled v5i 640->896 pairs on four
different architectures: +1.25 / +1.36 / +1.58 / +1.64 pp. An order of
magnitude more than anything left at 640, and one run. If these three land
flat, that is the next move -- not a fourth round of 640 tails.

STILL UNMOVED after 49 runs: the small-object localisation ratio. Loss,
assignment, architecture and the box metric have all been varied; it sits at
0.635-0.659 throughout.

All b32, 640, 70 ep, stock loss unless noted, pretrained backbone transfer,
zero-gated modules (identity at epoch 0). Directly comparable with every b32
run in rounds 1-2.

REQUIRES the fork with DySample / ZGGlobalContext2 / ZGDSConv installed as the
active ultralytics. preflight() checks and aborts otherwise.

Usage:
    python run_arch_final_v6i.py
    python run_arch_final_v6i.py gctxp3_k11coarse
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
PROJECT_DIR = "runs_arch_final_v6i"
YAML_DIR = "arch_yamls_final_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                    # matches ls_shift / levelspec / ls_k5 exactly
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# b32 references from the port sweep (all comparable — same batch).
REF = {"arch_ls_shift": 0.5598, "arch_levelspec": 0.5590, "arch_ls_k5": 0.5568}

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

# Best LB-TAL from the loss campaign was p4wide {8:4,16:7,32:1} on the 3-LEVEL
# head (cmb_p4wide, small 51.15). These two runs put it on the ls_shift 4-LEVEL
# topology: the P2 head CREATES the stride-4 small-object candidates, LB-TAL then
# ALLOCATES the budget across all four levels. CRITICAL (port-script warning):
# the budget is keyed by stride and iterates the head's strides {4,8,16,32};
# WITHOUT a `4:` entry stride-4 silently falls back to min_level_k=1 — the level
# with the MOST small candidates would get one slot, defeating the P2 head. Both
# budgets below include an explicit `4:`.
#   _LBTAL_A extends p4wide's small-emphasis to the new finest level (P2=P3=4),
#     keeps P4=7 (large's measured supply), P5=1. Sum 16 > tal_topk 10 -> the
#     per-GT cap re-ranks by metric (re-allocation, not inflation).
#   _LBTAL_B pushes MORE budget to P2 (now the richest small-object level).
_LBTAL_A = {4: 4, 8: 4, 16: 7, 32: 1}
_LBTAL_B = {4: 5, 8: 4, 16: 6, 32: 1}

# SWA (sqrt0703) box-weighting — the other half of cmb_p4wide.
_SWA0703 = dict(
    area_weight_mode="sqrt",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)


# WIoU v3 — box_loss_type has been 'ciou' in ALL 33 loss runs AND all 7 arch
# runs. It is the only major loss component never varied once, and it governs
# the mAP50->mAP50-95 ratio, which is the number NEITHER campaign moved:
# 0.635-0.658 across the loss runs, 0.653-0.659 across the arch runs, while
# overall mAP spanned 8.4 pp. Either a dead end, or the reason it is stuck.
_WIOU = dict(_ALL_OFF, box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0,
             wiou_momentum=0.02)


def _ls_shift_lbtal(level_topk, swa=False):
    """ls_shift topology (via YAML) + LB-TAL fixed budget (+ optional SWA)."""
    cfg = dict(_ALL_OFF, use_lbtal=True, lbtal_mode="fixed",
               lbtal_level_topk=level_topk, lbtal_min_level_k=1)
    if swa:
        cfg.update(_SWA0703)
    return cfg

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

# 1) ls_shift with snake k=9 -> k=5 (size-matched at P4/P5)
TAIL_LS_SHIFT_K5 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGDSConv, [256, 5]]                     # 25  P3 + snake k=5
  - [17, 1, ZGDSConv, [512, 5]]                     # 26  P4 + snake k=5
  - [20, 1, ZGDSConv, [1024, 5]]                    # 27  P5 + snake k=5
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 2) P3 gets BOTH gctx2 AND snake (context + shape prior), snake k=9
TAIL_LS_SHIFT_GCTXP3 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (stacked on gctx)
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""

# 3) global context pushed DEEPER: gctx2 @P2,P3,P4, snake only @P5
TAIL_LS_SHIFT_GCTXP3P4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + gctx2
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 4) snake at the finest level too: P2 gets gctx2 + snake k=5, rest = ls_shift
TAIL_LS_SHIFT_SNAKEP2 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [24, 1, ZGDSConv, [256, 5]]                     # 25  P2 + snake k=5 (stacked on gctx)
  - [14, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[25, 26, 27, 28], 1, Detect, [nc]]             # 29
"""

# The ORIGINAL ls_shift tail (unmodified) — used by the LB-TAL combo runs so the
# ONLY change vs the best arch is the assigner, not the topology.
TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""


# gctxP3 (the new best) — reference tail, unchanged.
TAIL_GCTXP3 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (STACKED)
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""

# STACK at P4 as well. NOTE this is stacking (gctx2 -> snake in series), which
# is what WON at P3. gctxP3P4 replaced the snake and lost 0.49 pp; that is a
# different operation and its result does not predict this one.
TAIL_STACKP3P4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (STACKED)
  - [17, 1, ZGGlobalContext2, [512]]                # 27  P4 + gctx2
  - [27, 1, ZGDSConv, [512, 9]]                     # 28  P4 + snake k=9 (STACKED)
  - [20, 1, ZGDSConv, [1024, 9]]                    # 29  P5 + snake k=9
  - [[24, 26, 28, 29], 1, Detect, [nc]]             # 30
"""

# gctxP3 with the kernel one step past the measured trend.
TAIL_GCTXP3_K11 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 11]]                    # 26  P3 + snake k=11 (STACKED)
  - [17, 1, ZGDSConv, [512, 11]]                    # 27  P4 + snake k=11
  - [20, 1, ZGDSConv, [1024, 11]]                   # 28  P5 + snake k=11
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""


# gctxP3 + snake ALSO at P2 -> gctx+snake STACKED at both fine levels, snake at
# all four. Combines the two monotonic levers: stacking (fact 1), coverage (5).
TAIL_GCTXP3_SNAKE4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [24, 1, ZGDSConv, [256, 9]]                     # 25  P2 + snake k=9 (STACKED)
  - [14, 1, ZGGlobalContext2, [256]]                # 26  P3 + gctx2
  - [26, 1, ZGDSConv, [256, 9]]                     # 27  P3 + snake k=9 (STACKED)
  - [17, 1, ZGDSConv, [512, 9]]                     # 28  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 29  P5 + snake k=9
  - [[25, 27, 28, 29], 1, Detect, [nc]]             # 30
"""

# gctxP3 EXACTLY, with the coarse snake upsized k9 -> k11 at P4,P5 ONLY. P3
# keeps k=9 so the small-object gain is untouched and only the large path moves.
TAIL_GCTXP3_K11COARSE = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (STACKED, unchanged)
  - [17, 1, ZGDSConv, [512, 11]]                    # 27  P4 + snake k=11
  - [20, 1, ZGDSConv, [1024, 11]]                   # 28  P5 + snake k=11
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""


RUNS = [
    {"name": "gctxp3_snake4", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3_SNAKE4,
     "label": "gctx+snake STACKED at P2 and P3, snake at all four levels",
     "why": "Combines the only two levers with monotonic evidence: stacking "
            "(gctxP3 +0.04, the sole win from that operation) and snake "
            "coverage (0 -> 2 -> 3 levels gave 0.5578 -> 0.5590 -> 0.5598, "
            "replicated at k=5). P2 is the one level never given a snake and "
            "is where the smallest objects resolve. Best shot at beating "
            "gctxP3 on SMALL. Watch large: gctxP3 already gave up 3.95 pp "
            "there and this does nothing to protect it."},

    {"name": "gctxp3_k11coarse", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3_K11COARSE,
     "label": "gctxP3 unchanged at P3; coarse snake k9 -> k11 at P4,P5 only",
     "why": "THE ONLY CONFIG HERE THAT CAN BE A STRICT IMPROVEMENT. gctxP3's "
            "sole deficit is large (0.5614 vs ls_shift 0.6009). This leaves "
            "the P3 stack that produced the small gain completely untouched "
            "and moves only the two levels where large objects live. The "
            "kernel direction is measured, not argued: k=5 lost 0.22 pp on "
            "levelspec and 0.11 pp on ls_shift, so bigger has won twice; k=11 "
            "is the next point and has never been run. Note this contradicts "
            "my k=5 'size-matched' derivation, falsified on both tails."},

    {"name": "gctxp3_wiou", "batch": BATCH, "levels": 4,
     "params": _WIOU,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3,
     "label": "gctxP3 + WIoU v3 — the only loss lever compatible with this topology",
     "why": "gctxP3 and WIoU each reach small ~0.515 by different routes "
            "(0.5158 vs 0.5151), and WIoU is the ONLY loss change that did not "
            "collapse on the P2 architecture: -0.07, against sqrt0703 -0.37 "
            "and lbtalB -0.84. One acts on backbone features, the other on the "
            "box metric, so the compatibility is measured rather than assumed. "
            "Watch: does small clear 0.5158, and does large recover from "
            "0.5614 toward WIoU's 0.5859."},

]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def env_provenance():
    info = {"loss_md5": None, "modules": {}}
    try:
        import ultralytics.utils.loss as _lm
        p = getattr(_lm, "__file__", None)
        if p and os.path.exists(p):
            info["loss_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    except Exception as e:
        info["loss_error"] = str(e)
    try:
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        for name in ("DySample", "ZGGlobalContext2", "ZGDSConv", "ZGDSConvV6"):
            info["modules"][name] = bool(
                hasattr(M, name) or name in getattr(T, "__dict__", {}) or
                any(hasattr(getattr(M, sub, None), name) for sub in dir(M)))
    except Exception as e:
        info["modules_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  loss md5={ENV.get('loss_md5')}")
    print(f"  custom modules: {ENV.get('modules')}")
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if missing or ENV.get("modules_error"):
        print(f"\n  [ABORT] custom modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        print("  These YAMLs reference fork-only blocks; parse_model would fail.")
        return False
    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"])) and not OVERWRITE_EXISTING]
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        print("  Delete them, or set OVERWRITE_EXISTING=True.")
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
          f"  {rc['label']}\n  yaml={yaml_path}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(yaml_path)
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] backbone transfer failed: {e} — training from scratch")
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    # Use the run's own loss params if provided (the LB-TAL combos), else stock.
    kw.update(copy.deepcopy(rc.get("params", _ALL_OFF)))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "batch": batch, "levels": rc["levels"], "seed": SEED,
           "hours": hours, "weights": weights, "yaml": yaml_path,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_params.json"), "w") as f:
            json.dump({**{k: v for k, v in out.items()},
                       "yaml_text": rc["yaml"], "why": rc["why"], "env": ENV}, f, indent=2)
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
    ref = REF["arch_ls_shift"]
    print(f"\n{'=' * 80}\n  ARCH REFINE — v6i @640, stock loss, b32 (comparable to port b32 runs)\n{'=' * 80}")
    print(f"{'run':<22}{'lvl':>4}{'mAP50':>9}{'mAP50-95':>11}{'vs ls_shift':>13}{'h':>6}")
    print("-" * 80)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        vs = "%+13.2f" % ((r["test_map5095"] - ref) * 100)
        print(f"{r['name']:<22}{r['levels']:>4}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{vs}{r['hours']:>6.1f}")
    print(f"\n  b32 refs: ls_shift {REF['arch_ls_shift']*100:.2f} | "
          f"levelspec {REF['arch_levelspec']*100:.2f} | ls_k5 {REF['arch_ls_k5']*100:.2f}")
    print("  vs ls_shift > 0 -> a refinement beat the best topology. Seed noise 0.12 pp.")
    print("  READ per-size: CocoEvalAllFolders_luggage.py on best.pt (ls_shift small 51.22 / large 60.09).")
    print("  A real win beats small WITHOUT dropping large.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 80}\n  ARCH REFINE — {len(todo)} configs around arch_ls_shift (b32, stock loss)")
    print(f"  runs: {', '.join(r['name'] for r in todo)}  (~{1.6*len(todo):.0f} GPU-h)")
    print(f"{'=' * 80}\n")
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
            res.append({"name": r["name"], "batch": r["batch"], "levels": r["levels"],
                        "seed": SEED, "hours": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
