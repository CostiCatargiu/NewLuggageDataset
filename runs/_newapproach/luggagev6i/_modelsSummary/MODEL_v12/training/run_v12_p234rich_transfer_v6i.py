#!/usr/bin/env python3
r"""
V26 -> V12 TRANSFER — does the YOLO26 result reproduce on YOLOv12?

Six runs, seed 0, no seed repeats. The 2x2 (runs 1-4) is BATCH 32 on every
cell; run 5 is a droppable attribution run at 32; run 6 is deliberately at 54
and sits OUTSIDE the 2x2 — see the batch section below. No patch required:
tal_beta is already a live train kwarg on this fork (46/46 existing runs record
it), and every YAML in this file is built inline from the constants that
run_arch_port_v6i.py already uses.

    WHAT V26 ESTABLISHED, and what is being carried over
        y26_identity        n=3   mAP50_small 77.62   <- baseline
        y26_b1              n=2   78.94   (+1.70%)    <- LOSS: tal_beta 6 -> 1
        y26_p2dys_p234rich  n=1   78.72   (+1.41%)    <- ARCH: drop P5 detect,
                                                         depth 4 on both fine levels
        y26_p234rich_b1     n=3   79.47   (+2.38%)    <- BOTH. the headline.
        decomposition: ARCH +1.10, LOSS +1.32, sum +2.42, observed +1.85
                       interaction -0.57  ->  76% additive, sub-additive

    V12 REFERENCES ON v6i, from results/*.json
        yolov12s_default   b54   sm50 76.65  mAP50 79.75  m5095 54.77
                                 P 80.37  R 72.16  AR50_95_small 67.03
        arch_ls_shift      b32   sm50 78.34  mAP50 81.70  m5095 55.98
                                 P 81.39  R 73.56  AR50_95_small 68.22
        ls_shift_sqrt      b32   sm50 78.73  <- best of all 46 v12 runs


=============================================================================
TWO THINGS ABOUT THE EXISTING V12 EVIDENCE THAT THIS ROUND FIXES
=============================================================================
1. tal_beta HAS NEVER BEEN MOVED ON V12. All 46 runs in results/ record
   tal_beta = 6.0. Not one exception. The best single mechanism found on v26 —
   worth +1.70% on mAP50_small — has simply never been tried here. It is a
   one-key change and it is the cheapest untested thing in the project.

   (Note the stock pair differs between the two models: v12's v8DetectionLoss
   assigner is alpha=0.5, beta=6.0; v26's E2E branches are alpha=1.0. beta=6.0
   is stock on BOTH, so "beta 6 -> 1" is the same intervention either way.)

2. EVERY V12 ARCHITECTURE NUMBER IS CONFOUNDED WITH BATCH SIZE, and the
   control that would fix it was written and then commented out.

       yolov12s_default     batch 54     <- the anchor everything is compared to
       arch_levelspec       batch 32
       arch_ls_shift        batch 32
       arch_gctx22          batch 28
       arch_dysample_p2...  batch 36
       ls_shift_* (six)     batch 32
       v12s_stock_b32       NEVER RUN    <- run_arch_port_v6i.py line 331,
                                            preserved as a comment

   So "arch_ls_shift beat the anchor by +1.69 on mAP50_small" is topology PLUS
   1.7x the optimiser steps, entangled, and thirteen architecture runs sit on
   top of that. This is the identical failure that cost v26 a round, where the
   fix was renaming y26_stock_b32 to y26_stock_b48 and re-deriving 42 deltas.

   RUN 1 CLOSES IT. It is the highest-value hour in this file and it is
   completely independent of whether the transfer works.


=============================================================================
WHY EVERY CELL OF THE 2x2 RUNS AT BATCH 32
=============================================================================
v12 has already run the two axes at DIFFERENT batch sizes, and that is exactly
what makes the existing evidence hard to combine:

    axis    scripts                                       runs   batch
    LOSS    newluggage_ablation, overnight_tune,            31     54
            sizecond_configs, lbtal_v2, lbtal_isolated
    ARCH    arch_port_v6i                                    7   28/32/36
            arch_refine_v6i, arch_round2, arch_final,        6     32
            arch_loss_combo

The split was forced, not chosen: a P2 head at 640 will not fit at batch 54 on
this GPU, which is why the 4-level graphs were run at 28-36 in the first place.

THE 2x2 CANNOT INHERIT THAT SPLIT. If the ARCH cells ran at 32 and the LOSS
cells at 54, batch would be perfectly confounded with the arch factor:

    measured ARCH effect  =  arch  +  batch
    measured interaction  =  arch x loss  +  batch x loss     <- uninterpretable

and the interaction term is the entire point of the decomposition. On v26 all
four cells ran at 48, which is the only reason ARCH +1.10 / LOSS +1.32 /
interaction -0.57 could be stated at all. Runs 1-5 therefore all run at 32,
the batch the P2 graphs can actually hold.

THE COST, STATED PLAINLY. v12_b1 at batch 32 is NOT comparable to the 31
existing loss runs at 54. It is comparable to v12_stock_b32, which is what the
2x2 needs. Run 6 buys back the other comparison for one hour:

    v12_b1_b54    beta=1 on the stock 3-level graph at BATCH 54

which reads directly against yolov12s_default (76.65) and drops straight into
the existing loss table with no new control required. It also gives a SECOND,
independent estimate of the batch effect:

    estimate A   v12_stock_b32 - yolov12s_default     batch effect at beta=6
    estimate B   v12_b1        - v12_b1_b54           batch effect at beta=1

If A and B agree, the batch correction is a constant and can be subtracted from
all thirteen existing architecture deltas with a clear conscience. If they
disagree, batch INTERACTS with the loss setting, and the existing v12 arch/loss
comparisons cannot be repaired by any single offset — which would itself be the
most important methodological result in the v12 chapter.


=============================================================================
HOW THE V26 GRAPH MAPS ONTO V12 — this is NOT the same model
=============================================================================
The two P2 topologies are structurally different and the writeup must say so.

    v26 p234:      P2 is INSERTED INTO the PAN. Bottom-up is rebuilt:
                   19 (P2) -> Conv -> cat 16 -> 22 (P3) -> Conv -> cat 13 -> 25 (P4)
                   Detect [19, 22, 25]

    v12 ls_shift:  P2 is APPENDED. The stock PAN outputs are reused as-is:
                   14 (P3) -> DySample -> cat backbone P2 -> 23 (P2 head)
                   Detect [24, 25, 26] = gctx(23), snake(14), snake(17)

So this file does not port a model. It ports the two INTERVENTIONS onto v12's
own best architecture (ls_shift, 78.34, the strongest 4-level graph here):

    v26 change                          v12 equivalent
    -------------------------------     ---------------------------------------
    drop the P5 detection branch        drop rows 18-20 (P5 head) and detect
                                        from [P2, P3, P4] only
    depth 2 -> 4 on the two FINE        row 14 (P3 head) and row 20 (P2 head,
    detection-feeding rows (19, 22)     renumbered from 23) go 2 -> 4 repeats

ROW BOOKKEEPING, because getting this wrong is the exact bug that wasted a
night on the v26 side. Deleting the P5 head shortens the head by three rows,
so the tail renumbers — but ONLY the tail:

    rows  0-8    backbone            IDENTICAL to every existing v12 yaml
    rows  9-17   head, top-down+P4   IDENTICAL (row 14 P3 head, row 17 P4 head)
    rows 18-20   stock P5 head       DELETED
    rows 18-24   the P2 tail         renumbered from 21-27

Rows 0-17 keep stock numbering, so model.load() transfers the backbone and the
whole top-down path by name with no remapping. No PAN remap function is needed
on v12 at all — unlike v26, where the P2 insert shifted every head row by 6.

WHAT WILL NOT TRANSFER, and that is expected:
  - the P5 head weights (stock rows 18-20): intentionally discarded, that is
    the whole point of the intervention
  - the extra repeats in rows 14/20 on the *rich* configs: 2 pretrained blocks
    into a 4-block module, so the last two initialise fresh
  - ZGGlobalContext2 / ZGDSConv: fork-only modules, no pretrained equivalent
    exists in any of these graphs, including the ones already run

RUN 3/4/5 THEREFORE TRANSFER LESS THAN RUN 1/2 BY CONSTRUCTION. That is fine.
What is NOT fine is a silent partial copy, so load_pretrained() below prints
the fraction and aborts under MIN_PRETRAINED_FRAC. run_arch_port_v6i.py's bare
try/except only catches total failure and would not have caught this.


=============================================================================
THE FIVE
=============================================================================
1  v12_stock_b32            THE MISSING CONTROL. Stock 3-level graph, stock
                            loss, batch 32. Uncommented from
                            run_arch_port_v6i.py line 331 where it has been
                            sitting unrun. Makes all thirteen existing arch
                            runs interpretable and is the correct denominator
                            for runs 2-5 in this file. Run it first.

                            If it lands near 76.65, the batch confound is
                            benign and the existing arch deltas stand. If it
                            lands near 77.5, roughly half of arch_ls_shift's
                            +1.69 was step count and the architecture chapter
                            of the v12 story needs rewriting.

2  v12_b1                   LOSS ALONE. Stock 3-level graph, tal_beta 1.0.
                            The single cheapest untested change on v12. Read
                            against run 1, NOT against yolov12s_default.
                            v26 prior: +1.70% on mAP50_small.

3  v12_lsshift_p234rich     ARCH ALONE. ls_shift minus the P5 detection
                            branch, depth 4 on rows 14 and 20. Read against
                            arch_ls_shift (78.34), which is the same graph
                            WITH P5 and at depth 2, same batch.

                            This is the direct test of "P5 is vestigial". On
                            v26 that claim rested on a footprint diagnostic
                            measured on YOLOv12 weights — and was then only
                            ever tested on v26. Run 3 finally tests it on the
                            model it was measured on.

4  v12_lsshift_p234rich_b1  BOTH. The transferred configuration and the reason
                            this file exists. With runs 1-3 it completes the
                            same 2x2 that produced the v26 decomposition, so
                            the interaction term is directly comparable across
                            the two detectors.

5  v12_lsshift_p234         DEPTH ATTRIBUTION, droppable if the queue is tight.
                            Same as run 3 at depth 2 instead of 4. On v26 the
                            depth axis had a measured optimum at 4 and depth 2
                            was the WORST cell of six (78.10 vs 78.72). If that
                            ordering reproduces here, the depth finding is
                            architecture-independent; if it inverts, it was a
                            property of v26's bottom-up P2 and should be
                            reported as such.


=============================================================================
HOW TO SCORE IT — WRITTEN BEFORE THE RUNS
=============================================================================
V12 HAS NO MEASURED NOISE FLOOR. There is exactly one seed group in 46 runs
(lb_uniform_seed1). Everything below is n=1 vs n=1. Borrowing v26's pooled sd
on mAP50_small (0.428, df=12) as a working estimate gives a run-to-run SE of
~0.605, so TREAT ANYTHING UNDER ~1.2 AS UNRESOLVED. This round maps; it does
not establish. Seeds come after, on whichever cells survive.

THE PRIMARY QUESTION IS NOT "IS IT HIGHER". It is whether the SIGNS agree:

    v26 finding                                  does v12 agree?
    ------------------------------------------   ------------------------
    LOSS (beta=1) is a PRECISION mechanism        run 2 vs run 1:
      +3.63% P, -0.88% mAP50-95, -4.44% AR        P up, AR down?
    ARCH (p234rich) is a RECALL mechanism         run 3 vs arch_ls_shift:
      +2.51% R, +1.45% mAP50-95, AR held          R up, m5095 up?
    the two are SUB-ADDITIVE (76% of the sum)     run 4 vs runs 2+3

Two detectors agreeing on the sign of a mechanism is a much stronger claim
than either model's mAP number, and it survives the missing noise floor.
Two detectors DISAGREEING is also publishable, and is already half-expected:
SWA is v12's best loss mechanism and was a measured null on v26.

WATCH AR50_95_small ON ALL FIVE. v12 baseline 67.03. On v26 every
high-precision configuration bought small-object precision and paid for it in
the recall ceiling, without exception. For an unattended-luggage alarm a missed
bag is a missed alarm, so a config that gains mAP50 while dropping AR is worth
less than the headline suggests. If v12 shows the same trade on a structurally
different P2 topology, the trade is a property of the ASSIGNMENT, not of any
one graph — which is the strongest mechanism claim available from this data.

DO NOT ADD A SIXTH CONFIGURATION. The beta surface was confirmed flat five
independent ways on v26 and the depth axis has a bracketed peak. There is
nothing to sweep here; this round transfers two settled findings and closes one
open confound.

    Usage:
        python run_v12_p234rich_transfer_v6i.py                     # all six
        python run_v12_p234rich_transfer_v6i.py --preflight         # build only
        python run_v12_p234rich_transfer_v6i.py --core              # the 2x2 only
        python run_v12_p234rich_transfer_v6i.py v12_b1              # one by name
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import time

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
PROJECT_DIR = "runs_v12_p234rich_transfer_v6i"
YAML_DIR = "transfer_yamls_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                     # matches arch_ls_shift / levelspec / ls_k5 exactly
BATCH_LOSS_ERA = 54            # what all 31 existing LOSS runs used — run 6 only
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# Abort rather than train a model that quietly failed to load its backbone.
# The 3-level graphs should come in near 1.0; the P2 graphs discard the P5 head
# and add fork-only blocks, so they land lower by construction.
MIN_PRETRAINED_FRAC = 0.30

# v6i references, all mAP50_small x100, from MODEL_v12/results/*.json
REF_ANCHOR_B54 = 76.65         # yolov12s_default, BATCH 54 — confounded
REF_LS_SHIFT_B32 = 78.34       # arch_ls_shift, BATCH 32 — the matched control
REF_BEST_ANY = 78.73           # ls_shift_sqrt, best of all 46

# =============================================================================
# WHERE THE EXISTING BEST RUNS LIVE ON DISK
# =============================================================================
# Set RUNS_ROOT to whatever directory you launched the old scripts from — every
# PROJECT_DIR in this project is relative, so the run trees are siblings of the
# scripts' working directory, not of the scripts themselves.
#
#   python run_v12_p234rich_transfer_v6i.py --refs
#
# prints each path, whether it exists, and the training args recorded in its
# args.yaml, with the keys that DIFFER across runs called out. That is the fast
# way to confirm what batch / lr / optimiser / augmentation each of these
# actually used, rather than trusting the script constants — the scripts have
# been edited since, and the arch results/*.json files carry an EMPTY run_meta,
# so args.yaml on disk is the only surviving record of those 13 runs.
RUNS_ROOT = os.environ.get("V12_RUNS_ROOT", ".")

REFERENCE_RUNS = [
    # --- the anchor ---------------------------------------------------------
    {"project": "runs_newl_luggagev6i", "name": "yolov12s_default",
     "era": "baseline", "batch": 54,
     "note": "sm50 76.65  m5095 54.77  P 80.37  R 72.16  ARs 67.03 — "
             "the anchor every existing delta is measured against"},

    # --- best LOSS, batch 54, stock 3-level graph ---------------------------
    # None of the five loss-era scripts contain a 'backbone:' block, so all 33
    # of those runs are on the stock yolov12s graph. cmb_p4wide is a LOSS
    # config despite the name — 'p4wide' is an LB-TAL level budget, not a
    # topology change.
    {"project": "runs_lbtal_v2", "name": "cmb_p4wide",
     "era": "loss", "batch": 54,
     "note": "BEST LOSS on mAP50_small: 78.19 (+1.54 over anchor)  "
             "m5095 55.60  P 80.67  R 73.54  ARs 67.49"},
    {"project": "runs_newl_luggagev6i", "name": "yolov12s_sqrt0703",
     "era": "loss", "batch": 54,
     "note": "BEST LOSS on mAP50-95: 55.64 (+0.87)  sm50 77.37  ARs 66.96 — "
             "the SWA/sqrt area-weighting result, v12's signature mechanism "
             "and a measured NULL on v26"},

    # --- best ARCH, batch 28-36 --------------------------------------------
    {"project": "runs_arch_v6i", "name": "arch_ls_shift",
     "era": "arch", "batch": 32,
     "note": "PURE ARCH (stock loss, _ALL_OFF): sm50 78.34  m5095 55.98  "
             "P 81.39  R 73.56  ARs 68.22 — the graph runs 3-5 modify, and "
             "the correct control for them"},
    {"project": "runs_arch_refine_v6i", "name": "ls_shift_sqrt",
     "era": "arch+loss", "batch": 32,
     "note": "BEST OF ALL 46 on mAP50_small: 78.73  m5095 55.61  R 74.22 — "
             "but this is ls_shift PLUS sqrt weighting, i.e. already an "
             "arch+loss combo, not a pure arch result"},
    {"project": "runs_arch_refine_v6i", "name": "ls_shift_gctxP3",
     "era": "arch", "batch": 32,
     "note": "BEST ARCH on mAP50-95: 56.02  sm50 78.37  ARs 68.12 — the "
             "highest mAP50-95 anywhere in the v12 campaign"},
]


def reference_paths(r):
    d = os.path.join(RUNS_ROOT, r["project"], r["name"])
    return {"dir": d,
            "args": os.path.join(d, "args.yaml"),
            "weights": os.path.join(d, "weights", "best.pt"),
            "results_csv": os.path.join(d, "results.csv")}


def show_references():
    """Print each reference run's path and its recorded training args."""
    print(f"\n{'=' * 76}\n  EXISTING BEST RUNS — paths and training setup\n"
          f"  RUNS_ROOT = {os.path.abspath(RUNS_ROOT)}\n"
          f"  (override with  V12_RUNS_ROOT=/path/to/runs  or edit RUNS_ROOT)\n"
          f"{'=' * 76}")

    loaded = {}
    for r in REFERENCE_RUNS:
        p = reference_paths(r)
        ok = os.path.isdir(p["dir"])
        mark = "ok  " if ok else "MISS"
        print(f"\n  [{mark}] {r['name']}   ({r['era']}, script batch {r['batch']})")
        print(f"         {p['dir']}")
        print(f"         args.yaml   {'yes' if os.path.exists(p['args']) else 'NOT FOUND'}"
              f"    best.pt {'yes' if os.path.exists(p['weights']) else 'NOT FOUND'}"
              f"    results.csv {'yes' if os.path.exists(p['results_csv']) else 'NOT FOUND'}")
        print(f"         {r['note']}")
        if os.path.exists(p["args"]):
            try:
                import yaml as _y
                loaded[r["name"]] = _y.safe_load(open(p["args"])) or {}
            except Exception as e:
                print(f"         [warn] could not parse args.yaml: {e}")

    if len(loaded) < 2:
        print(f"\n  Fewer than two args.yaml files found — nothing to diff.\n"
              f"  Set V12_RUNS_ROOT to the directory you launched the old "
              f"scripts from.\n{'=' * 76}\n")
        return

    # The point of this is the DIFF: shared keys are the project defaults, the
    # differing ones are what actually separates these runs.
    keys = set().union(*(set(v) for v in loaded.values()))
    names = list(loaded)
    differing = sorted(k for k in keys
                       if len({repr(loaded[n].get(k)) for n in names}) > 1)
    same = sorted(k for k in keys if k not in differing)

    print(f"\n{'-' * 76}\n  KEYS THAT DIFFER ACROSS THESE RUNS  "
          f"({len(differing)} of {len(keys)})\n{'-' * 76}")
    w = max((len(n) for n in names), default=10)
    print(f"  {'key':28s} " + " ".join(f"{n[:14]:>14s}" for n in names))
    for k in differing:
        vals = []
        for n in names:
            v = loaded[n].get(k, "-")
            s = f"{v:.5g}" if isinstance(v, float) else str(v)
            vals.append(f"{s[:14]:>14s}")
        print(f"  {k[:28]:28s} " + " ".join(vals))

    print(f"\n  identical across all {len(names)} runs ({len(same)} keys): "
          f"{', '.join(same[:24])}{' ...' if len(same) > 24 else ''}")
    print(f"\n  READ THE 'batch' ROW FIRST. If it does not match the script "
          f"constants\n  above, the script was edited after the run and "
          f"args.yaml is the truth.\n{'=' * 76}\n")

# Stock loss, everything off. Passed EXPLICITLY on every run: this fork's
# default.yaml is not neutral (small_obj_px 36, alpha_start 0.9, ...), so an
# implicit run would silently carry SWA weighting and nothing here would be
# comparable to anything.
_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
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
# YAML fragments. BACKBONE / HEAD_* / TAIL_STOCK_3LVL / TAIL_LS_SHIFT are
# byte-identical to run_arch_port_v6i.py so the graphs are provably the same
# ones already measured. Only the *_NOP5 / *_RICH variants are new.
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

# --- 3-level control head, verbatim from run_arch_port_v6i.py ---------------
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

# --- P234 head: rows 9-17 IDENTICAL to HEAD_DYSAMPLE, rows 18-20 DELETED ----
# Row 14 is the P3 head and row 17 the P4 head, exactly as in every other v12
# yaml. Nothing before row 18 moves, so model.load() matches by name cleanly.
HEAD_DYSAMPLE_NOP5 = """
head:
  - [-1, 1, DySample, [2]]                        # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, DySample, [2]]                        # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head   <- depth 2
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
"""

# rich: P3 head 2 -> 4 repeats. The v26 analogue is row 22.
HEAD_DYSAMPLE_NOP5_RICH = HEAD_DYSAMPLE_NOP5.replace(
    "- [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head   <- depth 2",
    "- [-1, 4, A2C2f, [256, False, -1]]              # 14  P3 head   <- depth 4 (RICH)",
)
assert "# 14  P3 head   <- depth 4 (RICH)" in HEAD_DYSAMPLE_NOP5_RICH, \
    "P3 depth substitution failed — the fragment text drifted"

# --- P2 tail, renumbered 21-27 -> 18-24 because the P5 head is gone ---------
# Compare against TAIL_LS_SHIFT in run_arch_port_v6i.py: same modules, same
# sources, minus the P5 snake branch, minus 3 on every index above 17.
TAIL_LS_SHIFT_P234 = """  - [14, 1, DySample, [2]]                          # 18  P3 head -> stride 4
  - [[18, 2], 1, Concat, [1]]                       # 19  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 20  P2 head   <- depth 2
  - [20, 1, ZGGlobalContext2, [256]]                # 21  P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]                     # 22  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 23  P4 + snake k=9
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24  Detect(P2, P3, P4) — strides 4, 8, 16
"""

# rich: P2 head 2 -> 4 repeats. The v26 analogue is row 19.
TAIL_LS_SHIFT_P234_RICH = TAIL_LS_SHIFT_P234.replace(
    "- [-1, 2, A2C2f, [256, False, -1]]                # 20  P2 head   <- depth 2",
    "- [-1, 4, A2C2f, [256, False, -1]]                # 20  P2 head   <- depth 4 (RICH)",
)
assert "# 20  P2 head   <- depth 4 (RICH)" in TAIL_LS_SHIFT_P234_RICH, \
    "P2 depth substitution failed — the fragment text drifted"

YAML_3LVL = BACKBONE + HEAD_STOCK + TAIL_STOCK_3LVL
YAML_P234 = BACKBONE + HEAD_DYSAMPLE_NOP5 + TAIL_LS_SHIFT_P234
YAML_P234RICH = BACKBONE + HEAD_DYSAMPLE_NOP5_RICH + TAIL_LS_SHIFT_P234_RICH

# =============================================================================
RUNS = [
    {"name": "v12_stock_b32", "levels": 3, "yaml": YAML_3LVL,
     "batch": BATCH, "beta": 6.0, "min_frac": 0.90,
     "ref": ("yolov12s_default b54", REF_ANCHOR_B54),
     "label": "stock 3-level, stock loss, BATCH 32 — the missing control",
     "why": "run_arch_port_v6i.py line 331, written and never run. Thirteen "
            "architecture runs are compared to a batch-54 anchor. Until this "
            "exists, every one of those deltas is topology plus step count."},

    {"name": "v12_b1", "levels": 3, "yaml": YAML_3LVL,
     "batch": BATCH, "beta": 1.0, "min_frac": 0.90,
     "ref": ("v12_stock_b32 (run 1)", None),
     "label": "stock 3-level, tal_beta 6.0 -> 1.0 — LOSS alone",
     "why": "tal_beta is 6.0 in all 46 existing v12 runs. On v26 this single "
            "key was worth +1.70% on mAP50_small and is the best loss-only "
            "result in that campaign. One key, never tried here."},

    {"name": "v12_lsshift_p234rich", "levels": 3, "yaml": YAML_P234RICH,
     "batch": BATCH, "beta": 6.0, "min_frac": MIN_PRETRAINED_FRAC,
     "ref": ("arch_ls_shift b32", REF_LS_SHIFT_B32),
     "label": "ls_shift minus P5 detect, depth 4 on P2+P3 — ARCH alone",
     "why": "The v26 architecture result, ported onto v12's own best graph. "
            "The footprint diagnostic that motivated dropping P5 was measured "
            "on YOLOv12 weights and then only ever tested on v26; this is the "
            "first test on the model it came from."},

    {"name": "v12_lsshift_p234rich_b1", "levels": 3, "yaml": YAML_P234RICH,
     "batch": BATCH, "beta": 1.0, "min_frac": MIN_PRETRAINED_FRAC,
     "ref": ("arch_ls_shift b32", REF_LS_SHIFT_B32),
     "label": "p234rich + tal_beta 1.0 — BOTH, the transferred configuration",
     "why": "v26's headline (79.47 at n=3, +2.38%). With runs 1-3 this closes "
            "the same 2x2 that gave the v26 decomposition, so the interaction "
            "term can be compared across the two detectors directly."},

    {"name": "v12_lsshift_p234", "levels": 3, "yaml": YAML_P234,
     "batch": BATCH, "beta": 6.0, "min_frac": MIN_PRETRAINED_FRAC,
     "ref": ("v12_lsshift_p234rich (run 3)", None),
     "label": "same graph at depth 2 — depth attribution, droppable",
     "why": "On v26 depth 4 beat depth 2 by +0.62 and depth 2 was the worst "
            "cell of six. If that ordering reproduces on a structurally "
            "different P2 topology the depth finding generalises; if it "
            "inverts it belonged to v26's bottom-up P2."},

    # THE ONLY RUN IN THIS FILE THAT IS NOT AT BATCH 32. It is deliberately
    # OUTSIDE the 2x2 and must never be substituted into it.
    {"name": "v12_b1_b54", "levels": 3, "yaml": YAML_3LVL,
     "batch": BATCH_LOSS_ERA, "beta": 1.0, "min_frac": 0.90,
     "ref": ("yolov12s_default b54", REF_ANCHOR_B54),
     "label": "beta=1 at BATCH 54 — joins the existing loss table, outside the 2x2",
     "why": "All 31 existing loss runs are at batch 54, so v12_b1 (batch 32) "
            "cannot be tabulated with them. This run can, against the anchor "
            "directly and with no new control. It also gives a SECOND estimate "
            "of the batch effect (v12_b1 - v12_b1_b54) to cross-check the "
            "first (v12_stock_b32 - anchor): if the two agree the correction "
            "is a constant and the thirteen existing arch deltas can be "
            "repaired by subtraction; if they disagree, batch interacts with "
            "the loss setting and no single offset can repair them."},
]


# =============================================================================
def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def env_provenance():
    """The real failure mode here is a missing fork module, not the loss."""
    info = {"loss_path": None, "loss_md5": None, "modules": {}}
    try:
        import ultralytics.utils.loss as _lm
        p = getattr(_lm, "__file__", None)
        info["loss_path"] = p
        if p and os.path.exists(p):
            info["loss_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    except Exception as e:
        info["loss_error"] = str(e)
    try:
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        for name in ("DySample", "ZGGlobalContext2", "ZGDSConv"):
            info["modules"][name] = bool(
                hasattr(M, name) or name in getattr(T, "__dict__", {}) or
                any(hasattr(getattr(M, sub, None), name) for sub in dir(M)))
    except Exception as e:
        info["modules_error"] = str(e)
    return info


ENV = env_provenance()


def check_tal_beta_is_live():
    """tal_beta must reach the assigner. If the fork does not read it, runs 2
    and 4 are silent duplicates of runs 1 and 3 and the round is worthless."""
    try:
        from ultralytics.cfg import get_cfg
        cfg = get_cfg(overrides={"tal_beta": 1.0})
        if abs(float(getattr(cfg, "tal_beta", 6.0)) - 1.0) > 1e-9:
            return False, "get_cfg did not carry tal_beta=1.0 through"
    except Exception as e:
        return False, f"get_cfg rejected tal_beta: {e}"
    try:
        import inspect
        import ultralytics.utils.loss as _lm
        src = inspect.getsource(_lm)
        if "tal_beta" not in src:
            return False, "loss.py never reads tal_beta — the key is inert"
    except Exception as e:
        return True, f"(could not inspect loss.py: {e})"
    return True, "tal_beta reaches loss.py"


def load_pretrained(model, min_frac):
    """Report the transfer fraction and refuse to train a random model.

    run_arch_port_v6i.py wraps model.load() in a bare try/except, which catches
    a total failure but NOT a partial one. A silent partial copy is what
    produced a round of 0.742 mAP50 results on the v26 side: the log had no
    'Transferred X/Y items' line, epoch 1 mAP50 was 0.0152, and nothing failed.
    """
    sd_before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        raise RuntimeError(
            f"model.load({MODEL_WEIGHTS}) raised {e}. Training from scratch is "
            f"not comparable to any existing run in results/.") from e
    sd_after = model.model.state_dict()
    total = sum(v.numel() for v in sd_after.values())
    moved = sum(sd_after[k].numel() for k in sd_before
                if k in sd_after and sd_before[k].shape == sd_after[k].shape
                and not torch.equal(sd_before[k].float(), sd_after[k].float()))
    frac = moved / total if total else 0.0
    print(f"  [load ] {moved / 1e6:.2f}M of {total / 1e6:.2f}M params transferred "
          f"from {MODEL_WEIGHTS}  ({100 * frac:.1f}%)")
    if frac < min_frac:
        raise RuntimeError(
            f"only {100 * frac:.1f}% of parameters moved, expected >= "
            f"{100 * min_frac:.0f}%. Either the row numbering drifted or the "
            f"weights file is wrong. Do not train this.")
    return moved, frac


def preflight(todo):
    print(f"  loss.py: {ENV.get('loss_path')}  md5={ENV.get('loss_md5')}")
    print(f"  fork modules: {ENV.get('modules')}")
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if missing or ENV.get("modules_error"):
        print(f"\n  [ABORT] fork modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        print("  Runs 3-5 reference blocks that exist only in your fork.")
        return False

    ok, msg = check_tal_beta_is_live()
    print(f"  tal_beta: {msg}")
    if not ok:
        print("\n  [ABORT] tal_beta is not live on this install. Runs 2 and 4 "
              "would be\n          silent duplicates of runs 1 and 3.")
        return False

    print()
    for rc in todo:
        path = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{rc['name']}.yaml"))
        try:
            m = YOLO(path)
            n = sum(p.numel() for p in m.model.parameters())
            strides = [int(s) for s in getattr(m.model, "stride", [])] or "?"
            _, frac = load_pretrained(m, rc["min_frac"])
            print(f"  [ok   ] {rc['name']:26s} {n / 1e6:6.3f}M params  "
                  f"strides={strides}")
            del m
            gc.collect()
        except Exception as e:
            print(f"  [FAIL ] {rc['name']:26s} {e}")
            return False

    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))]
    if clash and OVERWRITE_EXISTING:
        print(f"\n  [warn] OVERWRITE_EXISTING=True — reusing: {', '.join(clash)}")
        clash = []
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        return False

    if not any(r["name"] == "v12_stock_b32" for r in todo) and len(todo) > 1:
        print("\n  [warn] v12_stock_b32 is NOT in this selection. Runs 2-5 then "
              "have no\n         matched-batch denominator and inherit the exact "
              "confound this\n         file exists to close.")

    # The 2x2 must be single-batch or the interaction term is meaningless.
    quad = {"v12_stock_b32", "v12_b1", "v12_lsshift_p234rich",
            "v12_lsshift_p234rich_b1"}
    sel = {r["name"] for r in todo}
    if quad <= sel:
        bs = {r["batch"] for r in todo if r["name"] in quad}
        if len(bs) != 1:
            print(f"\n  [ABORT] the four 2x2 cells are at mixed batches {sorted(bs)}. "
                  f"Batch would be\n          confounded with the arch factor and "
                  f"the interaction term would be\n          uninterpretable. "
                  f"Fix the 'batch' keys in RUNS.")
            return False
        print(f"\n  [ok   ] 2x2 is single-batch ({bs.pop()}) — decomposition "
              f"is valid")
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
    print(f"\n{'=' * 76}\n  RUN {name}  ({rc['levels']} levels, batch {batch}, "
          f"tal_beta {rc['beta']})\n  {rc['label']}\n"
          f"  yaml={yaml_path}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    if batch != BATCH:
        print(f"  [note ] batch {batch} != {BATCH}. This run is OUTSIDE the 2x2 "
              f"by design.\n         Do not substitute it into the "
              f"decomposition.\n")
    t0 = time.time()

    model = YOLO(yaml_path)
    moved, frac = load_pretrained(model, rc["min_frac"])
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(_ALL_OFF))
    kw["tal_beta"] = rc["beta"]          # the ONLY loss key that ever differs
    results = model.train(**kw)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")

    out = {"name": name, "batch": batch, "levels": rc["levels"], "seed": SEED,
           "tal_beta": rc["beta"], "pretrained_frac": frac, "hours": hours,
           "save_dir": save_dir, "weights": weights, "yaml": yaml_path,
           "loss": f"STOCK + tal_beta={rc['beta']} (_ALL_OFF passed explicitly)",
           "env": ENV, "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "transfer_params.json"), "w") as f:
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


def summarise(res):
    print(f"\n{'=' * 76}\n  SUMMARY — validation-set mAP50-95 only.\n"
          f"{'=' * 76}")
    print("  These are NOT the numbers to report. The per-size metrics that\n"
          "  every comparison above is written against (mAP50_small,\n"
          "  AR50_95_small, P50_small) come from the full-dataset test\n"
          "  evaluator, not from model.val(). Run that, then compare.\n")
    print(f"  {'run':28s} {'bs':>3s} {'beta':>5s} {'load%':>6s} "
          f"{'test mAP50':>11s} {'test m5095':>11s} {'h':>5s}")
    for r in res:
        print(f"  {r['name'][:28]:28s} {r.get('batch', 0):3d} "
              f"{r['tal_beta']:5.1f} {100 * r.get('pretrained_frac', 0):6.1f} "
              f"{r['test_map50']:11.4f} {r['test_map5095']:11.4f} "
              f"{r.get('hours', 0):5.2f}")

    def m(name):
        r = next((x for x in res if x["name"] == name), None)
        v = r and r.get("test_map5095")
        return v if v == v else None          # NaN-safe

    # --- the two independent estimates of the batch effect -------------------
    a = m("v12_stock_b32")
    b32, b54 = m("v12_b1"), m("v12_b1_b54")
    est_a = 100 * (a - 0.5477) if a is not None else None
    est_b = 100 * (b32 - b54) if (b32 is not None and b54 is not None) else None

    if est_a is not None or est_b is not None:
        print(f"\n  BATCH EFFECT (32 vs 54), mAP50-95, two independent estimates")
        if est_a is not None:
            print(f"    A  beta=6   v12_stock_b32 - yolov12s_default  "
                  f"{est_a:+.2f} pp")
        if est_b is not None:
            print(f"    B  beta=1   v12_b1        - v12_b1_b54        "
                  f"{est_b:+.2f} pp")
        if est_a is not None and est_b is not None:
            gap = abs(est_a - est_b)
            print(f"    |A - B| = {gap:.2f} pp")
            if gap < 0.5:
                print("    -> consistent. Treat the batch effect as a constant "
                      "and subtract it\n       from the thirteen existing "
                      "architecture deltas.")
            else:
                print("    -> BATCH INTERACTS WITH THE LOSS SETTING. No single "
                      "offset repairs the\n       existing v12 arch/loss "
                      "comparisons. Report this; it is a stronger\n       "
                      "methodological finding than any mAP number in this file.")

    mixed = {r.get("batch") for r in res}
    if len(mixed) > 1:
        print(f"\n  [reminder] batches present: {sorted(mixed)}. The 2x2 is the "
              f"FOUR\n             batch-{BATCH} cells only. v12_b1_b54 sits "
              f"outside it.")
    print(f"\n{'=' * 76}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--preflight", action="store_true",
                    help="build every graph, load weights, run nothing")
    ap.add_argument("--core", action="store_true",
                    help="runs 1-4 only, skip the depth attribution")
    a = ap.parse_args()

    only = set(a.names)
    todo = [r for r in RUNS if not only or r["name"] in only]
    if a.core and not only:
        todo = [r for r in todo
                if r["name"] not in ("v12_lsshift_p234", "v12_b1_b54")]
    if only and len(todo) != len(only):
        raise SystemExit(f"unknown run(s): {only - {r['name'] for r in todo}}")

    bs = sorted({r["batch"] for r in todo})
    print(f"\n{'=' * 76}\n  V26 -> V12 TRANSFER\n"
          f"  {len(todo)} runs, batch {bs if len(bs) > 1 else bs[0]}, "
          f"seed {SEED}, {EPOCHS} epochs, imgsz {IMG_SIZE}\n{'=' * 76}\n")

    if not preflight(todo):
        raise SystemExit(1)
    if a.preflight:
        print("\n  preflight only — nothing trained.\n")
        return

    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as e:
            print(f"\n  [FAILED] {rc['name']}: {e}\n")
            res.append({"name": rc["name"], "batch": rc["batch"],
                        "levels": rc["levels"], "seed": SEED,
                        "tal_beta": rc["beta"], "error": str(e),
                        "test_map50": float("nan"),
                        "test_map5095": float("nan")})
        with open(f"{PROJECT_DIR}_partial.json", "w") as f:
            json.dump(res, f, indent=2)

    summarise(res)


if __name__ == "__main__":
    main()
