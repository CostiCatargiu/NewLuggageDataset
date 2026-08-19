#!/usr/bin/env python3
"""
YOLO26 SBB OVERNIGHT — 7 runs. New mechanism + the controls that make it readable.

=============================================================================
THE MECHANISM
=============================================================================
E2ELoss blends two branches with a gain that depends ONLY on the epoch:

    one2many   topk=10          ~10 positives per GT
    one2one    topk=7, topk2=1   1 positive per GT, produces EVERY prediction
    blend      o2m 0.8 -> 0.1 over training, a global scalar

Nothing makes that blend depend on the object. But the branches are not equally
reliable per object. one2one's single positive is an argmax of
score^alpha * IoU^beta: for an 8 px box a one-pixel shift reorders that ranking,
for a 200 px box it does not. one2one inherits that noise in full; one2many's
ten positives average it out. The noise is size-dependent, the blend is not.

SBB gives the two branches OPPOSITE area weightings so the EFFECTIVE blend
becomes size-dependent:

    one2many  sign -1  -> leans on SMALL   (ten positives absorb the noisy pick)
    one2one   sign +1  -> leans on LARGE   (single pick is reliable there)

    w = (sqrt(area_px) / sbb_ref_px) ** (sign * q),  renormalised to mean 1

    q=0.50, ref=64px       8px    16px    32px    64px   128px   256px
      one2many (small)    2.01    1.42    1.00    0.71    0.50    0.36
      one2one  (large)    0.36    0.50    0.71    1.00    1.42    2.01

q = 0 is bit-identical to stock.

WHY THIS IS NOT SWA. SWA applies ONE area weighting to BOTH branches in the SAME
direction and asks "do small objects need more regression gradient?" That was
measured: +0.35, at the hyperparameter floor. SBB applies OPPOSITE weightings and
asks "should each branch specialise by object size?" - a question only a
dual-branch head can pose. v12 has one branch and cannot express it.

WHY IT MIGHT WORK. It extends the reliability argument behind SCB, the only
intervention in 21 loss configs that produced signal (+0.42, and a two-axis
sweep showed a peak at its setting rather than a plateau).

=============================================================================
WHY 7 RUNS AND NOT 3
=============================================================================
21 loss configs have landed within ~0.4 pp of baseline, and run-to-run noise
looks like 0.08-0.10 pp (two pairs of DIFFERENT configs came back identical).
At that resolution a single point proves nothing - which is how y26_scb_b3's
+0.42 survived four messages before its own sweep showed its four neighbours
sitting BELOW baseline.

So this round is built as an axis with a direction control, not a hunt:

    q = 0.25 / 0.50 / 0.75         three points -> trend vs scatter
    INVERTED at the best q         direction control: if flipping the signs
                                   helps equally, the effect is "any per-branch
                                   area weighting" and SBB's specific story
                                   (small->o2m, large->o2o) is wrong
    ref_px 32 and 128 at q=0.50    where the crossover sits
    stock repeat                   the baseline is n=1 and 7 unrelated configs
                                   all landed 0.13-0.42 ABOVE it, which is more
                                   consistent with a low draw than with seven
                                   independent wins

=============================================================================
REFERENCE POINTS (stock yolo26s, b82/640/seed0)
=============================================================================
    yolo26_custom-9   55.24   baseline, n=1
    y26_scb_b3        55.66   best of 21 loss configs (+0.42)
    y26_alpha075      55.59
    hyperparam floor  55.37   dfl gain and global beta, two trivial nudges
    scb's neighbours  55.16   mean of 4 perturbations - BELOW baseline

    apparent noise    0.08-0.10 pp

    >= 55.75 (+0.51)  clears the best config -> worth a seed repeat
    55.40 - 55.70     inside the band everything else occupies
    <= 55.20          the direction is wrong

Read PRECISION and SMALL per-size. SCB's signal was precision (+2.02) and SBB
targets the same selection-quality pathway from the other side.

Stock yolo26s.pt, b82/640/seed 0 — matches every loss run in this project.
REQUIRES the patched loss.py (SBB) + default.yaml.

DEFAULT IS THREE RUNS, NOT SEVEN
================================
Two of the three pay off whatever SBB does, and the third is the bet:

  y26_base_rep    validates or invalidates 21 EXISTING loss results. Seven
                  unrelated configs sit 0.13-0.42 above a single-sample baseline,
                  clustered within 0.11 pp. That is better explained by a low
                  draw than by seven independent wins, and nothing else you can
                  run answers it.
  y26_sbb_q50     the mechanism. ~30%.
  y26_sbb_inv50   makes q50 readable. Without it a good result is round 5's
                  starve/rich situation - a number you cannot interpret.

The other four (q=0.25, q=0.75, ref=32, ref=128) characterise an axis that may
not exist. They run only with --all, and the summary tells you whether they have
earned it: q50 must clear the band AND beat its mirror by >0.20.

TWO ARMS IN ONE SCRIPT
======================
  LOSS ARM   stock yolo26s.pt, b82, SBB on/off      -> reference 55.24 (n=1)
  ARCH ARM   P2+DySample yaml, b32, STOCK loss      -> reference 56.08 +- 0.19 (n=10)

They are different models at different batch sizes. Each is read against its own
reference; the two columns are NOT comparable to each other. The arch arm is
included because it is a sensitivity study on the module that actually carried
YOLO26's gain (+1.05 over the same graph without it) and it needs no new code —
`groups` has been at its library default of 4 in every run ever done.

Usage:
    python run_yolo26_sbb_overnight_v6i.py            # CORE: 3 loss + 3 arch, ~10.8 GPU-h
    python run_yolo26_sbb_overnight_v6i.py --all      # all 10, ~18 GPU-h
    python run_yolo26_sbb_overnight_v6i.py y26_base_rep y26_dys_g8
"""

import gc
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_sbb_overnight_v6i"
YAML_DIR = "arch_yamls_y26_sbb"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# ---- ARCH ARM (batch 32 — a DIFFERENT regime from the loss arm's b82) ------
BATCH_ARCH = 32
ARCH_CENTRE = 56.08    # y26_p2_dysample, groups=4, n=10, b32
ARCH_CENTRE_SD = 0.19
ARCH_NO_DYS = 55.03    # y26_p2_b32, same batch, n=1

BASELINE = 55.24
BEST_LOSS = 55.66      # y26_scb_b3
FLOOR = 55.37          # dfl gain / global beta
SCB_NEIGHBOURS = 55.16 # mean of scb's 4 perturbations
NOISE = 0.10
REAL = BASELINE + 0.51 # clears the best existing config

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


CORE = ["y26_base_rep", "y26_sbb_q50", "y26_sbb_inv50",
        "y26_dys_g8", "y26_dys_2of3", "y26_dys_g2"]  # the default set: 3 loss + 3 arch

YAML_G8 = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2, 8]] # 17  P3 -> P2, groups=8
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

YAML_G2 = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2, 2]] # 17  P3 -> P2, groups=2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

YAML_2OF3 = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, DySample, [2]] # 14  content-aware P4 -> P3
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

RUNS = [
    {"name": "y26_base_rep", "arm": "loss", "expect": {"sbb": False}, "params": cfg(),
     "label": "STOCK REPEAT — everything off, same settings as the baseline",
     "why": "Runs first because it can invalidate the whole table. yolo26_custom-9 "
            "(55.24) is n=1, and SEVEN unrelated configs - including two trivial "
            "hyperparameter nudges - all landed 0.13 to 0.42 above it, clustered "
            "within 0.11 pp of each other. Seven independent changes cannot all "
            "help by the same small amount; a low baseline draw explains it in one "
            "stroke. If this comes back near 55.5, every 'gain' in the loss "
            "campaign including scb_b3's +0.42 is inside noise and the axis is a "
            "clean null. If it comes back near 55.24, the deltas are real and the "
            "rest of tonight is readable against a confirmed reference."},

    {"name": "y26_sbb_q50", "arm": "loss", "expect": {"sbb": True, "q": 0.5, "invert": False},
     "params": cfg(sbb_q=0.5, sbb_ref_px=64.0),
     "label": "SBB q=0.50 ref=64px — the centre of the mechanism",
     "why": "The main run. At q=0.50 one2many weights an 8 px box 2.01x and a "
            "256 px box 0.36x, one2one the mirror image - a 5.6x swing between the "
            "branches across the size range, large enough to matter and small "
            "enough that neither branch is starved. Chosen as the centre because "
            "SNL1's useful range was p=0.25-0.5 on a comparable redistribution, "
            "and because 0.25/0.50/0.75 then brackets it symmetrically."},

    {"name": "y26_sbb_inv50", "arm": "loss", "expect": {"sbb": True, "q": 0.5, "invert": True},
     "params": cfg(sbb_q=0.5, sbb_ref_px=64.0, sbb_invert=True),
     "label": "DIRECTION CONTROL — q=0.50 with BOTH signs flipped",
     "why": "The run that makes q50 interpretable, and the one most likely to be "
            "skipped by mistake. It applies the same magnitude of per-branch area "
            "weighting in the OPPOSITE direction: one2many leans large, one2one "
            "leans small. If this helps as much as q50, the effect is 'any "
            "per-branch area weighting perturbs training usefully' and SBB's "
            "specific claim - that small objects belong to the ten-positive branch "
            "and large ones to the single-positive branch - is unsupported. That "
            "is exactly the confound that killed the P2-starve story in round 5, "
            "where a mirrored pair turned out to differ in three places at once."},

    {"name": "y26_sbb_q25", "arm": "loss", "expect": {"sbb": True, "q": 0.25, "invert": False},
     "params": cfg(sbb_q=0.25, sbb_ref_px=64.0),
     "label": "SBB q=0.25 ref=64px — gentle",
     "why": "The conservative end (1.48x / 0.62x across the range). Needed to turn "
            "q into a curve: with only q=0.50 a good result cannot be told from "
            "'any perturbation helps'. Also the version most likely to preserve "
            "large-object AP, which every small-object intervention on this "
            "dataset has cost so far."},

    {"name": "y26_sbb_q75", "arm": "loss", "expect": {"sbb": True, "q": 0.75, "invert": False},
     "params": cfg(sbb_q=0.75, sbb_ref_px=64.0),
     "label": "SBB q=0.75 ref=64px — aggressive",
     "why": "The far end (2.54x / 0.19x). At this strength one2one gives a 8 px "
            "box a fifth of its neutral weight, so if the mechanism's premise is "
            "right this should be where it either peaks or breaks. A monotone rise "
            "through 0.25 -> 0.50 -> 0.75 means the optimum is beyond the range; a "
            "peak at 0.50 means the axis is characterised in one night."},

    {"name": "y26_sbb_r32", "arm": "loss", "expect": {"sbb": True, "q": 0.5, "invert": False},
     "params": cfg(sbb_q=0.5, sbb_ref_px=32.0),
     "label": "SBB q=0.50 ref=32px — crossover at the small/medium boundary",
     "why": "ref_px sets where the weight is neutral and the branches swap "
            "preference. At 64 the crossover sits mid-medium; at 32 it sits at the "
            "COCO small/medium boundary, so the whole medium and large range goes "
            "to one2one and only true small objects lean on one2many. SCB's own "
            "sweep found ref_px mattered as much as the magnitude, so leaving it "
            "at one value would repeat that gap."},

    {"name": "y26_sbb_r128", "arm": "loss", "expect": {"sbb": True, "q": 0.5, "invert": False},
     "params": cfg(sbb_q=0.5, sbb_ref_px=128.0),
     "label": "SBB q=0.50 ref=128px — crossover deep in the large range",
     "why": "The other side: almost everything counts as 'small' and leans on "
            "one2many, with only genuinely large trolleys going to one2one. Makes "
            "ref_px three points. It also indirectly tests whether the gain (if "
            "any) comes from the SIZE CONDITIONING or simply from shifting weight "
            "toward one2many overall - at ref=128 the two look similar, so a win "
            "here but not at 32 points at the latter."},
    {"name": "y26_dys_g8", "arm": "arch", "yaml": YAML_G8,
     "expect": {'n_dys': 1, 'groups': 8},
     "params": cfg(),   # EXPLICIT _ALL_OFF — every loss mechanism off
     "label": "groups 4 -> 8  (offset conv 32 -> 64 ch, 4,128 -> 8,256 params)",
     "why": "The first half of the only knob this module has. More groups means more independent offset fields per location, so the upsampler can warp different channel subsets differently - the plausible direction if the P3->P2 gain is limited by offset expressiveness rather than by the idea itself. Paired with g2 below it makes groups a curve; alone it would only show a direction. "
            "ARCH ARM: batch {} on the DySample P2 graph, read against the n=10 "
            "control 56.08 +- 0.19 — NOT against the loss arm above, which is a "
            "different model at a different batch.".format(BATCH_ARCH)},

    {"name": "y26_dys_2of3", "arm": "arch", "yaml": YAML_2OF3,
     "expect": {'n_dys': 2, 'groups': 4},
     "params": cfg(),   # EXPLICIT _ALL_OFF — every loss mechanism off
     "label": "DySample at P4->P3 AND P3->P2 (layers 14 and 17), groups=4",
     "why": "The missing middle of the count axis. One DySample gives 56.08, three gives 54.49 - so more is worse, but TWO has never been measured and the useful count is somewhere in 1-2. This adds it at the P4->P3 upsample, the one feeding the level where COCO-small objects mostly live, while leaving P5->P4 (which serves large) alone. If two matches one, the mechanism is specifically about the stride-4 path and should be described that way. "
            "ARCH ARM: batch {} on the DySample P2 graph, read against the n=10 "
            "control 56.08 +- 0.19 — NOT against the loss arm above, which is a "
            "different model at a different batch.".format(BATCH_ARCH)},

    {"name": "y26_dys_g2", "arm": "arch", "yaml": YAML_G2,
     "expect": {'n_dys': 1, 'groups': 2},
     "params": cfg(),   # EXPLICIT _ALL_OFF — every loss mechanism off
     "label": "groups 4 -> 2  (offset conv 32 -> 16 ch, 4,128 -> 2,064 params)",
     "why": "The other side, and the more useful null. If HALVING the offset capacity costs nothing, the mechanism does not depend on offset expressiveness at all and the claim becomes 'a minimal content-aware upsampler suffices' - cheaper to deploy and more portable than a tuned value. With g8 and the default at 4 this gives three points. "
            "ARCH ARM: batch {} on the DySample P2 graph, read against the n=10 "
            "control 56.08 +- 0.19 — NOT against the loss arm above, which is a "
            "different model at a different batch.".format(BATCH_ARCH)},
]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        from ultralytics.utils.loss import BboxLoss, E2ELoss
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    checks = {
        "BboxLoss.sbb_enabled": hasattr(BboxLoss, "sbb_enabled"),
        "BboxLoss.sbb_weight": hasattr(BboxLoss, "sbb_weight"),
        "E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
        "E2ELoss sets opposite signs": "sign" in inspect.getsource(E2ELoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<32}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] the SBB patch is not installed.")
        print("  Copy ultralytics26/ultralytics/{utils/loss.py,cfg/default.yaml} then:")
        print("  python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False
    if any(r.get("arm") == "arch" for r in todo):
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        src_t = open(T.__file__, encoding="utf-8").read()
        ok = hasattr(M, "DySample") and "elif m is DySample:" in src_t and "args = [c1, *args]" in src_t
        print(f"  DySample importable + registered + args forwarded   {ok}")
        if not ok:
            print("  [ABORT] the arch arm needs the nn module port with arg forwarding.")
            return False
    for r in todo:
        if r.get("arm") == "arch" or not r["expect"].get("sbb"):
            continue
        bl = BboxLoss(1)
        bl.sbb_q = r["params"]["sbb_q"]
        bl.sbb_sign = 1.0
        if not bl.sbb_enabled():
            print(f"  [ABORT] {r['name']}: sbb_enabled() False at q={bl.sbb_q}")
            return False
    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock)  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  baseline {BASELINE:.2f} (n=1)   best loss config {BEST_LOSS:.2f}   floor {FLOOR:.2f}")
    print(f"  apparent noise ~{NOISE:.2f} pp  ->  >= {REAL:.2f} clears everything measured")
    print(f"  y26_base_rep runs FIRST and can invalidate the whole comparison.")
    print(f"  Read PRECISION and SMALL per-size, not overall alone.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print()
        print("  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_callbacks(model, rc):
    """Assert at epoch 1 that SBB is live with the requested q and OPPOSITE signs."""
    state = {"verified": False}
    exp = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        o2m = getattr(crit, "one2many", None)
        o2o = getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — SBB needs both branches")
        a, b = o2m.bbox_loss, o2o.bbox_loss
        if not exp["sbb"]:
            if a.sbb_enabled() or b.sbb_enabled():
                raise RuntimeError(f"{rc['name']}: expected SBB OFF but it is live")
            print(f"  [guard] SBB off on both branches (stock)")
            state["verified"] = True
            return
        for tag, bl in (("one2many", a), ("one2one", b)):
            if not bl.sbb_enabled():
                raise RuntimeError(
                    f"{rc['name']}: sbb_q={rc['params']['sbb_q']} requested but "
                    f"sbb_enabled() is False on {tag} (q={bl.sbb_q} sign={bl.sbb_sign}). "
                    f"E2ELoss is not wiring SBB — aborting rather than producing a number.")
            if abs(float(bl.sbb_q) - float(exp["q"])) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag}.sbb_q is {bl.sbb_q}, expected {exp['q']}")
        if a.sbb_sign * b.sbb_sign >= 0:
            raise RuntimeError(
                f"{rc['name']}: branch signs are {a.sbb_sign:+.0f}/{b.sbb_sign:+.0f} — they must be "
                f"OPPOSITE or this is just SWA applied twice.")
        want_o2m = +1.0 if exp["invert"] else -1.0
        if a.sbb_sign != want_o2m:
            raise RuntimeError(
                f"{rc['name']}: one2many sign is {a.sbb_sign:+.0f}, expected {want_o2m:+.0f} "
                f"(invert={exp['invert']})")
        print(f"  [guard] SBB live  q={a.sbb_q} ref={a.sbb_ref_px}px  "
              f"one2many {a.sbb_sign:+.0f} / one2one {b.sbb_sign:+.0f}"
              f"{'  [INVERTED control]' if exp['invert'] else ''}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def check_arch_graph(model, rc):
    """Verify the BUILT graph has the requested DySample count and groups."""
    mods = [m for m in model.model.modules() if type(m).__name__ == "DySample"]
    want = rc["expect"]
    if len(mods) != want["n_dys"]:
        raise RuntimeError(f"{rc['name']}: graph has {len(mods)} DySample, yaml declares {want['n_dys']}")
    for m in mods:
        g = int(getattr(m, "groups", -1))
        if g != want["groups"]:
            raise RuntimeError(
                f"{rc['name']}: DySample built with groups={g}, expected {want['groups']}. "
                f"parse_model is not forwarding the yaml arg — this run would be a "
                f"replicate of the default.")
    print(f"  [guard] DySample x{len(mods)}  groups={want['groups']}  "
          f"in_ch={[int(m.offset.in_channels) for m in mods]}")


def attach_stock_loss_guard(model, rc):
    """ARCH ARM: assert at epoch 1 that EVERY loss mechanism is inert.

    The arch arm exists to measure the DySample module, so any live loss
    mechanism would confound it. The runs pass an explicit _ALL_OFF dict, but
    default.yaml is the fallback for anything not listed and a stale value there
    would leak in silently — which is exactly how rounds 4-6 produced ten
    identically-configured runs under ten different names.
    """
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        for tag in ("one2many", "one2one"):
            br = getattr(crit, tag, crit)
            bl = getattr(br, "bbox_loss", None)
            a = getattr(br, "assigner", None)
            if bl is None or a is None:
                raise RuntimeError(f"{rc['name']}: no bbox_loss/assigner on {tag}")
            live = []
            if hasattr(bl, "sbb_enabled") and bl.sbb_enabled():
                live.append(f"SBB(q={bl.sbb_q})")
            if hasattr(bl, "snl1_enabled") and bl.snl1_enabled():
                live.append(f"SNL1(p={bl.l1_scale_p})")
            if hasattr(bl, "swa_enabled") and bl.swa_enabled():
                live.append(f"SWA(a0={bl.alpha_start})")
            if hasattr(a, "scb_enabled") and a.scb_enabled():
                live.append(f"SCB(bs={a.beta_small})")
            if type(a).__name__ == "LevelBalancedTaskAlignedAssigner":
                live.append("LB-TAL")
            if live:
                raise RuntimeError(
                    f"{rc['name']}: ARCH ARM must run STOCK loss, but {tag} has "
                    f"{', '.join(live)} live. This would confound the DySample "
                    f"measurement — check default.yaml. Aborting.")
            for attr, want in (("alpha", 0.5), ("beta", 6.0)):
                got = float(getattr(a, attr))
                if abs(got - want) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag}.{attr} is {got}, expected stock {want}")
        hyp = getattr(crit, "one2one", crit).hyp
        for k, want in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):
            got = float(getattr(hyp, k, want))
            if abs(got - want) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {k} gain is {got}, expected stock {want}")
        print(f"  [guard] stock loss confirmed on both branches "
              f"(no SBB/SNL1/SWA/SCB/LB-TAL, alpha=0.5 beta=6.0, gains 7.5/0.5/1.5)")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name, arm = rc["name"], rc.get("arm", "loss")
    batch = BATCH if arm == "loss" else BATCH_ARCH
    print()
    print("=" * 78)
    print(f"  RUN {name}   [{arm.upper()} ARM]")
    print(f"  {rc['label']}")
    print("=" * 78)
    if arm == "arch":
        src = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{name}.yaml"))
        print(f"  cfg={src} (P2 + DySample)  imgsz={IMG_SIZE}  batch={batch}  "
              f"epochs={EPOCHS}  seed={SEED}")
        print(f"  stock loss — read against the arch control {ARCH_CENTRE:.2f} +- {ARCH_CENTRE_SD:.2f}")
    else:
        src = MODEL_WEIGHTS
        print(f"  model={src} (stock)  imgsz={IMG_SIZE}  batch={batch}  "
              f"epochs={EPOCHS}  seed={SEED}")
        diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  differs from _ALL_OFF: {diff or '(nothing — stock repeat)'}")
    print()
    t0 = time.time()

    model = YOLO(src)
    if arm == "arch":
        check_arch_graph(model, rc)
        try:
            model.load(MODEL_WEIGHTS)
        except Exception as e:
            print(f"  [warn] weight transfer failed: {e}")
        state = attach_stock_loss_guard(model, rc)
    else:
        state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": arm, "params": rc["params"], "expect": rc["expect"], "seed": SEED,
           "model": src, "imgsz": IMG_SIZE, "batch": batch, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "sbb_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    by = {r["name"]: r["test_map5095"] * 100 for r in ok}
    ref = by.get("y26_base_rep", BASELINE)
    reflbl = "y26_base_rep (this run)" if "y26_base_rep" in by else "yolo26_custom-9 (n=1)"
    print()
    print("=" * 88)
    print(f"  SBB OVERNIGHT — stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 88)
    print(f"  reference: {ref:.2f}  ({reflbl})")
    print()
    loss_runs = [r for r in ok if r.get("arm", "loss") == "loss"]
    arch_runs = [r for r in ok if r.get("arm") == "arch"]

    print(f"  LOSS ARM — stock yolo26s, b{BATCH}")
    print(f"{'run':<16}{'q':>6}{'ref_px':>8}{'inv':>5}{'mAP50-95':>10}{'vs ref':>9}   verdict")
    print("-" * 88)
    for r in sorted(loss_runs, key=lambda x: -x["test_map5095"]):
        p, e = r["params"], r["expect"]
        v = r["test_map5095"] * 100
        d = v - ref
        vd = ("REAL (clears every config)" if v >= REAL else
              "above the band" if d > 0.30 else
              "wrong direction" if d < -0.30 else "inside the band")
        print(f"{r['name']:<16}{p['sbb_q']:>6.2f}{p['sbb_ref_px']:>8.0f}"
              f"{('Y' if p['sbb_invert'] else '-'):>5}{v:>10.2f}{d:>+9.2f}   {vd}")
    print("-" * 88)
    if arch_runs:
        print()
        print(f"  ARCH ARM — P2 + DySample, b{BATCH_ARCH}   (control {ARCH_CENTRE:.2f} +- {ARCH_CENTRE_SD:.2f}, n=10)")
        print(f"{'run':<16}{'n_dys':>7}{'groups':>8}{'mAP50-95':>10}{'vs ctrl':>9}   verdict")
        print("-" * 88)
        for r in sorted(arch_runs, key=lambda x: -x["test_map5095"]):
            v = r["test_map5095"] * 100
            d = v - ARCH_CENTRE
            vd = ("BETTER" if d > 0.40 else "wrong direction" if v < 55.70 else "flat")
            print(f"{r['name']:<16}{r['expect']['n_dys']:>7}{r['expect']['groups']:>8}"
                  f"{v:>10.2f}{d:>+9.2f}   {vd}")
        print("-" * 88)
        print(f"  {'y26_p2_dysample':<16}{1:>7}{4:>8}{ARCH_CENTRE:>10.2f}   <- control, n=10")
        print(f"  {'y26_p2_b32':<16}{0:>7}{'-':>8}{ARCH_NO_DYS:>10.2f}   <- no DySample")
        g = [r["test_map5095"] * 100 for r in arch_runs if r["expect"]["n_dys"] == 1]
        if len(g) >= 2 and max(g + [ARCH_CENTRE]) - min(g + [ARCH_CENTRE]) < 0.40:
            print("  groups is FLAT — report the gain as insensitive to it, which is a")
            print("  stronger claim than a tuned constant.")
        print()
    for lbl, v in (("y26_scb_b3 (best)", BEST_LOSS), ("hyperparam floor", FLOOR),
                   ("scb's neighbours", SCB_NEIGHBOURS), ("baseline n=1", BASELINE)):
        print(f"  {lbl:<22}{v:>7.2f}")

    if "y26_base_rep" in by:
        gap = by["y26_base_rep"] - BASELINE
        print(f"\n  BASELINE CHECK: repeat {by['y26_base_rep']:.2f} vs published {BASELINE:.2f} -> {gap:+.2f}")
        if abs(gap) > 0.30:
            print("    The published baseline was a LOW DRAW. Every delta in the loss")
            print("    campaign — including scb_b3's +0.42 — shrinks by that amount, and")
            print("    the axis is a clean null. Report it as one; it is a complete")
            print("    negative across 21 configs with code-level reasons.")
        else:
            print("    Baseline confirmed. The campaign's deltas stand as measured.")

    q = sorted((r["params"]["sbb_q"], by[r["name"]]) for r in ok
               if r["expect"]["sbb"] and not r["params"]["sbb_invert"]
               and r["params"]["sbb_ref_px"] == 64.0)
    if q:
        print("\n  q axis (ref_px = 64):")
        print(f"    q=0.00  {ref:.2f}   (stock)")
        for qq, v in q:
            print(f"    q={qq:.2f}  {v:.2f}   {v - ref:+.2f}")
    r_ = sorted((r["params"]["sbb_ref_px"], by[r["name"]]) for r in ok
                if r["expect"]["sbb"] and not r["params"]["sbb_invert"]
                and r["params"]["sbb_q"] == 0.5)
    if r_:
        print("\n  ref_px axis (q = 0.50):")
        for rr, v in r_:
            print(f"    ref={rr:>5.0f}  {v:.2f}   {v - ref:+.2f}")

    if "y26_sbb_q50" in by and "y26_sbb_inv50" in by:
        a, b = by["y26_sbb_q50"], by["y26_sbb_inv50"]
        print(f"\n  DIRECTION CONTROL")
        print(f"    q50  (small->o2m, large->o2o)   {a:.2f}   {a - ref:+.2f}")
        print(f"    inv  (large->o2m, small->o2o)   {b:.2f}   {b - ref:+.2f}")
        print(f"    difference                      {a - b:+.2f}")
        if abs(a - b) < 0.20:
            print("    Same within noise -> the DIRECTION does not matter, so any gain is")
            print("    'per-branch area weighting perturbs training', not SBB's premise.")
            print("    Do not claim the small->one2many / large->one2one story.")
        elif a > b:
            print("    q50 beats its mirror -> the direction matters and the premise holds.")
        else:
            print("    The MIRROR wins -> the premise is backwards. one2one should carry")
            print("    small objects. Worth understanding before discarding.")

    print(f"\n  Per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print(f"  SCB's signal was PRECISION (+2.02); SBB targets the same selection-")
    print(f"  quality pathway, so check precision and small before overall.")
    print(f"  Do NOT read LARGE at n=1 — sd 2.11 pp on this dataset.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<16} {r['weights']}")
    core_done = {n for n in ("y26_base_rep", "y26_sbb_q50", "y26_sbb_inv50") if n in by}
    if core_done == {"y26_base_rep", "y26_sbb_q50", "y26_sbb_inv50"}:
        a, b = by["y26_sbb_q50"], by["y26_sbb_inv50"]
        earned = a >= REAL and (a - b) >= 0.20
        print()
        print("  " + "=" * 74)
        print("  DOES PHASE 2 EARN ITS 7 HOURS?")
        print(f"    q50 clears {REAL:.2f}          : {a >= REAL}   ({a:.2f})")
        print(f"    q50 beats its mirror by >0.20 : {(a - b) >= 0.20}   ({a - b:+.2f})")
        if earned:
            print("    YES — the mechanism moved AND the direction matters. Characterise it:")
            print("      python " + os.path.basename(__file__) + " --all")
        else:
            print("    NO — stop here. Either the mechanism did not move or the direction")
            print("    does not matter, and four more points on a flat axis add nothing.")
            print("    The negative is already complete: 21 loss configs plus SBB, all")
            print("    inside the band, with code-level reasons why.")
        print("  " + "=" * 74)
    if any(r.get("arm") == "arch" for r in ok):
        print()
        print("  NOTE: the two arms are DIFFERENT models at DIFFERENT batch sizes.")
        print(f"  Loss arm (b{BATCH}, stock 3-level) and arch arm (b{BATCH_ARCH}, P2+DySample)")
        print("  are not comparable to each other — only each to its own reference.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    run_all = "--all" in sys.argv[1:]
    only = set(args)
    if only:
        todo = [r for r in RUNS if r["name"] in only]
    elif run_all:
        todo = list(RUNS)
    else:
        todo = [r for r in RUNS if r["name"] in CORE]   # DEFAULT: the informative 3
    print()
    print("=" * 88)
    print(f"  YOLO26 SBB — {len(todo)} runs, ~{1.8 * len(todo):.1f} GPU-h"
          f"{'  [CORE set]' if not only and not run_all else ''}")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 88)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "params": r["params"], "expect": r["expect"],
                        "seed": SEED, "hours": float("nan"), "error": str(e),
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
