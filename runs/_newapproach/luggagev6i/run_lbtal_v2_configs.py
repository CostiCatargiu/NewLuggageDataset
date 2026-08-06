#!/usr/bin/env python3
"""
v3 sweep — three axes designed to BEAT sqrt0703 (55.64) and lb_uniform (55.57).

Configs are ordered by CONFIDENCE OF IMPROVEMENT, highest first; each carries a
'why' field with its reasoning. Default run = tiers 1-2 (first 5). Use --all for
everything, or name configs explicitly.

  ASSIGNMENT   (lb_*)      per-level top-k + quality gate — the original v2 line
  CLASSIFIER   (posboost,  attacks the ~25 pp AR50_small vs R50_small ranking gap;
                qfl, clsw) untouched in all 18 prior runs
  LOCALISATION (nwd_*)     attacks the 0.65-vs-0.75 small/medium mAP50->50-95
                           ratio; use_nwd has been False in all 18 prior runs

=============================================================================
WHY THESE (grounded in the isolated + overnight results, NOT random tuning)
=============================================================================
MEASURED SEED NOISE (lb_uniform vs lb_uniform_seed1 — same config, seed 0 vs 1):
    overall mAP50-95  0.12 pp     large mAP50-95  2.06 pp
Every claim below is stated against those numbers. Read the table with them.

  SIGNAL A [SUPPORTED] - lb_coarse_244 has the HIGHEST recall ceiling
             (AR50_small 0.9658 vs uniform 0.9595) but LOWER mAP (55.34 vs
             55.57). AR is a ceiling metric and less jittery than mAP, so this
             is the solid signal: coarse levels FIND more small objects but
             some coarse positives are LOW QUALITY and don't convert.
             FIX: quality-gate the per-level picks (drop weak coarse positives).

  SIGNAL B [WEAK — do not over-trust] - the SWA+LB-TAL combo scored 55.10.
             The original rationale was "over-boosting small collapsed LARGE",
             but the data does not show that:
               small mAP50-95  cmb 0.5047  vs uniform 0.5085  -> small went DOWN
               large mAP50-95  cmb 0.5437  vs uniform 0.5775  -> -3.4 pp
             There was no small-boost to trade against large, and the 3.4 pp
             large drop is only ~1.6x the 2.06 pp seed noise on that metric,
             from a single pair. lb_balcap therefore tests a hypothesis that is
             not yet established. Keep it, but rank it below the qg configs.

NOT a premise: "uniform == uniform_mk2 proves tuning has peaked." Those two are
the same config under deterministic=True/seed=0 — identical to 4 decimals on
all 30 metrics. That demonstrates reproducibility, not a plateau.

=============================================================================
THE CONFIGS — see the RUNS list for the full reasoning on each
=============================================================================
  1. qfl              sqrt0703 + Quality Focal Loss — the ranking gap
  2. cmb_p4wide       sqrt0703 + LB-TAL {8:4,16:7,32:1} — cmb repaired
  3. cmb_p4wide_qg50  ... + gate 0.50 — adds the size-conditional filter
  4. nwd_small        sqrt0703 + NWD on small only — the localisation gap
  5. clsw_sqrt        sqrt0703 + class weighting — bag is 15 pp behind trolley
  6. lb_p4wide        {8:4,16:7,32:1} alone — attribution for #2
  7. lb_p3_3          {8:3,16:4,32:4} — the one gap in the P3 curve
  8. posboost_stock   STOCK + pos_boost — NEEDS loss_custom_v3_fixed.py
  9. nwd_blend25      NWD blend fallback for #4

The original six (lb_uniform_qg50, lb_uniform_qg70, lb_coarse_qg50, lb_balcap,
lb_balcap_qg50, lb_uniform_tk13_qg50) are gone — see the BLOCKED and REMOVED
blocks under RUNS for why each one went. Two were bit-for-bit duplicates of
lb_coarse_244, which had already been run.

Baselines to beat: lb_uniform 55.57 (+0.87 small, AR50_small 0.960),
                   SWA sqrt0703 55.64 (+0.65 small), anchor 54.77.
READ small mAP + AR50_small (CocoEvalAllFolders_luggage.py), not just overall.
This script CANNOT produce per-size metrics — ultralytics val has no
small/medium/large breakdown. It records the weights path for each run so the
coco eval can be pointed at them afterwards. Overall mAP alone will not settle
this sweep: the configs are expected to differ by ~0.2-0.5 pp.

GATE SEMANTICS (changed): quality_gate is measured against the GT's best metric
ACROSS ALL LEVELS, not within each level. The per-level version was near-inert
— torch.topk returns sorted, so each level's top-1 always passed and a
uniformly-weak coarse level survived intact, which is the opposite of the
intent. 4 of the 6 configs below depend on the gate doing real work.

VERIFY FIRST: python selftest_lbtal.py  (all PASS) — the new modes/gate are
covered if you re-run it; at minimum confirm no import/shape errors.
This script also preflights the installed loss.py (path/md5/features) and
aborts rather than silently training stock TAL for 9 GPU-h.

TWO LOSS FILES, DISJOINT FEATURE SETS — READ THIS BEFORE RUNNING.
lossv2updated.py and loss_custom_v3_fixed.py read 71 and 31 hyperparameters
respectively, and the intersection is EXACTLY ZERO:

  lossv2updated.py         lbtal_*, nwd_*, cls_mode/qfl_beta, satal/snatal/
                           artal, area_weight_mode, alpha_*, small_obj_*,
                           use_class_weighting, use_bag_penalty, ...
  loss_custom_v3_fixed.py  use_pos_boost + pos_boost_*, use_freq_weight,
                           use_lba + lba_*, use_ardfl + ardfl_*, use_peu +
                           peu_*, and the *_report() telemetry

Consequences:
  * configs 1-7 and 9 need lossv2updated.py installed as ultralytics/utils/loss.py
  * config 8 (posboost_stock) needs loss_custom_v3_fixed.py instead
  * sqrt0703 and pos_boost CANNOT be combined — no installed loss reads both,
    which is why posboost_stock measures against the 0.5477 anchor rather than
    sqrt0703's 0.5564. Porting Section P into lossv2updated.py is the change
    that would make those two comparable.

default.yaml whitelists the union of both, so the cfg checker accepts keys the
installed loss will silently ignore. preflight() therefore scans the installed
loss SOURCE for every key a selected config activates and aborts on a no-op,
and refuses a selection that mixes the two files.

Usage:
    python run_lbtal_v2_configs.py                 # tiers 1-2 (first 5)
    python run_lbtal_v2_configs.py --all
    python run_lbtal_v2_configs.py qfl cmb_p4wide
    python run_lbtal_v2_configs.py posboost_stock  # after swapping loss.py
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
PROJECT_DIR = "runs_lbtal_v2"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
BASELINE_TEST_MAP5095 = 0.5477

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


# sqrt0703 — the single best run to date (55.64). Base for every non-LB-TAL
# config below. NOT combined with LB-TAL: cmb_lbU_swa0703 differs from
# sqrt0703 by exactly the three lbtal params and scored 55.10, i.e. worse than
# sqrt0703 (55.64) AND worse than lb_uniform (55.57). The two axes are
# antagonistic — both push small objects and together they over-correct.
_SQRT0703 = dict(
    _ALL_OFF,
    area_weight_mode="sqrt",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)


def _lb(mode, level_topk=None, min_level_k=1, quality_gate=0.0, topk=10):
    """LB-TAL config on the all-off base (assignment axis)."""
    return dict(_ALL_OFF, use_lbtal=True, lbtal_mode=mode,
                lbtal_level_topk=level_topk, lbtal_min_level_k=min_level_k,
                lbtal_quality_gate=quality_gate, tal_topk=topk)


def _sq(**kw):
    """Config on the sqrt0703 base (cls / localisation axes)."""
    return dict(_SQRT0703, **kw)


def _po(**kw):
    """pos_boost on the sqrt0703 base — Section P is now PORTED into lossv2updated.py.

    Before the port the two loss files read DISJOINT hyperparameter sets (71
    keys vs 31, intersection exactly zero): pos_boost/freq_weight/lba/ardfl/peu
    lived only in loss_custom_v3_fixed.py, while area_weight_mode, alpha_*,
    small_obj_*, lbtal_*, nwd_* and cls_mode lived only in lossv2updated.py.
    sqrt0703 + pos_boost was therefore unreachable — no installed loss.py read
    both — and use_pos_boost=True against lossv2updated would have passed the
    default.yaml whitelist, been silently ignored, and trained a bit-identical
    copy of the base config.

    Section P is now implemented in lossv2updated.py, so this runs on the
    strongest known base and is directly comparable with everything else here.
    """
    return dict(_SQRT0703, use_pos_boost=True, **kw)


def _sq_lb(mode, level_topk=None, min_level_k=1, quality_gate=0.0, topk=10, **kw):
    """sqrt0703 base WITH LB-TAL — the 'cmb' combination, gated.

    cmb_lbU_swa0703 (this combination, ungated) scored 55.10 vs 55.64 for
    sqrt0703 alone. The footprint diagnostic explains why: under LB-TAL uniform
    a LARGE GT is forced to take 2.91 positives from stride 8 (stock took 0.16),
    and its positive-set quality collapses — mean IoU 0.876 -> 0.720, p10 IoU
    0.707 -> 0.387, %IoU<0.3 from 0.1% to 5.6%. On its own the network absorbs
    that (large mAP was flat, 0.5775 vs 0.5773). Combined with sqrt weighting,
    which additionally lowers large's loss weight, it cannot recover.
    The quality gate prunes exactly those forced P3 picks, because a large GT's
    best metric sits on P4 so its P3 picks fall below gate*global_best.
    """
    return dict(_SQRT0703, use_lbtal=True, lbtal_mode=mode,
                lbtal_level_topk=level_topk, lbtal_min_level_k=min_level_k,
                lbtal_quality_gate=quality_gate, tal_topk=topk, **kw)


# =============================================================================
# RUNS — ORDERED BY CONFIDENCE THAT WE SEE AN IMPROVEMENT (highest first)
# =============================================================================
# Confidence is a judgement about P(real gain), not about gain size. It is
# based on: (a) does the run attack a failure mode the data actually shows,
# (b) has the mechanism been validated anywhere, (c) how much of the config is
# guesswork. Every entry carries its reasoning so the order can be argued with.
#
# THE TWO GAPS THE DATA SHOWS (consistent across all 18 prior runs):
#   RANKING     AR50_small ~0.95 but R50_small ~0.70 -> ~25 pp of small objects
#               are detected but score below the F1-optimal threshold.
#               Nothing in 18 runs touched the cls/confidence path.
#   LOCALISATION mAP50->mAP50-95 ratio: small 0.65, medium 0.75. Small boxes
#               lose 35% of AP when IoU tightens. The ratio moved 0.649->0.657
#               across all 18 runs, i.e. nothing tried has affected it.
# Caveat: part of the ranking gap is structural (every detector has one). How
# much is recoverable is unknown — that is why tier 1 includes telemetry.
#
# -----------------------------------------------------------------------------
# THE LB-TAL AXIS IS ONE NUMBER: THE P3 BUDGET
# -----------------------------------------------------------------------------
# Footprint diagnostic, same weights, three schemes (diag_fp_out_v6i/*.json).
# s16 selection is IDENTICAL between uniform and coarse244 for every size
# bucket — small 2.08, medium 3.87, large 3.92 — because both give P4 four
# slots and P4 picks outrank everything else, so uniform's per-GT cap never
# touches them. s32 barely moves. Every LB-TAL mode therefore differs ONLY in
# how many stride-8 positives it allows.
#
#   P3 budget   config                              mAP50-95
#       8       lb_prop -> {8:8,16:2,32:1}           0.5482
#      ~6.6     stock (global top-10)                0.5477
#       5       lb_uniform_tk13 -> ceil(13/3)        0.5542
#       4       lb_uniform                           0.5557   <- peak
#       3       NOT TESTED                              ?
#       2       lb_coarse_244                        0.5534
#
# Single-peaked at 4. Most of the P3 sweep is already done; only P3=3 is
# missing, which is why this file has ONE P3 probe and not five.
#
# CAUTION — THE P3 STORY IS NOT THE WHOLE AXIS. The "s16 is identical" finding
# compared uniform (4/4/4) with coarse244 (2/4/4). BOTH HAVE P4=4. Concluding
# from that pair that P4 does not matter is circular: P3 was varied and P4 was
# held fixed. Every LB-TAL run that performed well had P4=4, and P4 has never
# been raised above it. The cross-tab says that is where the damage is:
#              s16 cand/GT   sel stock   sel LB-TAL   limited by
#   small          2.46        1.08        2.08       SUPPLY
#   medium        10.63        3.39        3.87       budget
#   large        125.28        7.96        3.92       budget
# Hence cmb_p4wide / lb_p4wide below. Small is supply-capped at 2.46, so the
# P4 budget cannot affect it — the repair is free of risk to the one bucket
# LB-TAL actually helps.
#
# ALSO SETTLED: balanced_capped resolves to {8:2,16:4,32:4} at topk=10 — bit
# for bit lb_coarse_244, already run. lb_balcap / lb_balcap_qg50 were removed
# as duplicates rather than re-measured.
#
# NOT IMPLEMENTED, but the principled fix: the level budgets are GLOBAL, while
# the cross-tab shows small / medium / large each want a different allocation
# (small is P3+P4 only, large is P4-dominant). A budget conditioned on GT size
# would dominate any single global 3-tuple. That is a loss.py change, not a
# config, so it is out of scope here — but it is the obvious next mechanism.
# =============================================================================

RUNS = [
    # ---------------------------------------------------------------- TIER 1
    {"name": "posboost", "batch": BATCH, "confidence": "HIGH",
     "label": "sqrt0703 + pos_boost (bag 1.5, small 1.5) — Section P, newly ported",
     "why": "Attacks the ~25 pp AR50_small-vs-R50_small ranking gap: the "
            "detections exist, they score below the F1-optimal threshold. "
            "Nothing in 18 prior runs touched the confidence path. Section P's "
            "defaults are hand-tuned for exactly this — default.yaml's own "
            "comment reads 'bag positives (recall cell 0.68 - the worst cell)' "
            "— so they are a considered choice, not a guess. Crucially it "
            "reports its own premise: posboost_report() gives mean predicted "
            "score at fg anchors PER CLASS, which measures the targeted "
            "quantity directly instead of inferring it from a mAP delta that "
            "will sit near the 0.12 pp seed noise. "
            "NOTE this only became runnable after porting Section P from "
            "loss_custom_v3_fixed.py into lossv2updated.py — the two files read "
            "disjoint hyperparameters, so sqrt0703 + pos_boost was previously "
            "unreachable and would have silently trained as plain sqrt0703. "
            "Read the [P] line in the startup banner to confirm it is live.",
     "params": _po(pos_boost_backpack=1.0, pos_boost_bag=1.5, pos_boost_trolley=1.0,
                   pos_boost_small=1.5, pos_boost_small_px=60.0,
                   pos_boost_clip=3.0, pos_boost_log=True)},

    {"name": "qfl", "batch": BATCH, "confidence": "HIGH",
     "label": "sqrt0703 + Quality Focal Loss (beta 2.0) — tie confidence to box quality",
     "why": "Now the top classifier config, because pos_boost turned out not to "
            "exist in loss.py (see BLOCKED below). QFL scales BCE by "
            "|target - pred|^beta, aligning the confidence a detection gets with "
            "how well it is localised — a textbook fix for the ~25 pp "
            "AR50_small-vs-R50_small ranking gap, and a published, widely "
            "replicated mechanism rather than a local invention. VERIFIED "
            "implemented: cls_mode=='qfl' is handled in the cls loss, and "
            "qfl_beta is read. cls_mode has been 'bce' in all 18 prior runs, so "
            "the entire confidence path is untouched territory.",
     "params": _sq(cls_mode="qfl", qfl_beta=2.0)},

    {"name": "cmb_p4wide", "batch": BATCH, "confidence": "MEDIUM-HIGH",
     "label": "sqrt0703 + LB-TAL fixed {8:4,16:7,32:1} — cmb repaired at the source",
     "why": "The best-motivated assignment run in this file, and the one that "
            "could make the two axes ADDITIVE. Measured s16 supply vs selection: "
            "small has only 2.46 candidates (SUPPLY-limited, takes 2.08), while "
            "large has 125.28 and takes just 3.92 — BUDGET-limited, down from "
            "7.96 under stock. Large lost half its natural P4 supply purely "
            "because the budget is 4, and that is what killed cmb (large IoU "
            "0.876->0.720, p10 0.707->0.387; fine alone, fatal once sqrt also "
            "lowers large's loss weight). Raising P4 to 7 restores large and "
            "medium and CANNOT touch small, which is supply-capped at 2.46. P5 "
            "goes to 1 because it is unfillable — small has 0.62 candidates "
            "there and selects 0.01. P3 stays at 4, the measured peak. "
            "Predicted footprint: small row IDENTICAL to uniform (3.64/2.08/0.01), "
            "large back to ~2/7/1 against stock's 0.16/7.96/1.68. Preferred over "
            "the gate because it removes the problem at the source rather than "
            "pruning after the fact, with no risk to small's picks. "
            "VERIFY FIRST (5 min): diag_anchor_footprint.py, ASSIGNER='p4wide'.",
     "params": _sq_lb("fixed", level_topk={8: 4, 16: 7, 32: 1})},

    {"name": "cmb_p4wide_qg50", "batch": BATCH, "confidence": "MEDIUM-HIGH",
     "label": "sqrt0703 + fixed {8:4,16:7,32:1} + gate 0.50 — all three size buckets addressed",
     "why": "The compound. MEASURED p4wide result: medium is essentially "
            "repaired (%IoU<0.3 4.6% -> 1.0% vs stock 0.0%, P4 3.87 -> 5.80), "
            "small is unchanged-to-better (P4 2.08 -> 2.27), but LARGE is only "
            "half fixed — mean IoU 0.720 -> 0.761 against stock's 0.876, and "
            "%<0.3 actually rose 5.6% -> 6.1%. The residue is large's 2.17 "
            "stride-8 positives (stock takes 0.16), forced in by the P3 budget "
            "of 4. Dropping P3 would fix large but 4 is small's measured peak, "
            "so ONE GLOBAL BUDGET CANNOT SERVE BOTH. The gate is the missing "
            "size-conditional filter and works through the metric rather than "
            "the budget: a large GT's best metric is on P4, so its P3 picks "
            "fall below gate*global_best and are pruned; a small GT's best IS "
            "on P3, so its picks survive. p4wide has already removed the P4 "
            "starvation the gate could not address, so the two are complements, "
            "not alternatives. "
            "RISK: the gate may also prune part of small's P4 (2.27/GT, now at "
            "92% of its available pool, so the marginal ones are low-metric). "
            "Confirm with the 5-min pass ASSIGNER='p4wide' + QUALITY_GATE=0.5 "
            "-> large s8 should collapse from 2.17 toward 0 while small s16 "
            "holds near 2.27. Needs the GLOBAL-reference gate installed.",
     "params": _sq_lb("fixed", level_topk={8: 4, 16: 7, 32: 1}, quality_gate=0.50)},

    # ---------------------------------------------------------------- TIER 2
    {"name": "nwd_small", "batch": BATCH, "confidence": "MEDIUM",
     "label": "sqrt0703 + NWD on small objects only (thr 36) — the untouched localisation gap",
     "why": "The only run here aimed at the 0.65-vs-0.75 localisation ratio, which "
            "nothing in 18 runs has moved. NWD is designed for small-box IoU "
            "sensitivity and use_nwd has been False every single time. 'small_only' "
            "swaps CIoU->NWD strictly below the area threshold, so large objects "
            "cannot collapse the way they did in cmb (-3.4 pp large). Threshold 36 "
            "= (small_obj_px 48 / stride 8)^2, aligning it with the Section A small "
            "criterion instead of the arbitrary 32 default. MEDIUM not higher "
            "because the threshold's coordinate units are the main uncertainty.",
     "params": _sq(use_nwd=True, nwd_mode="small_only", nwd_small_threshold=36.0, nwd_C=4.0)},

    {"name": "clsw_sqrt", "batch": BATCH, "confidence": "MEDIUM",
     "label": "sqrt0703 + class weighting (sqrt) — bag is 15 pp behind trolley",
     "why": "bag AP50-95 0.4666 vs trolley 0.6185, and bag has the worst "
            "find-vs-score gap of the three (AR50_s 0.9352 -> R50_s 0.62). "
            "class_weight_mode='sqrt' is already set in every run but inert because "
            "use_class_weighting=False. Cheap to test. MEDIUM because class "
            "weighting is a blunt instrument: it reweights everything for that "
            "class rather than targeting the under-scored cells, which is what "
            "pos_boost does more precisely.",
     "params": _sq(use_class_weighting=True, class_weight_mode="sqrt")},

    # ---------------------------------------------------------------- TIER 3
    {"name": "lb_p4wide", "batch": BATCH, "confidence": "MEDIUM",
     "label": "LB-TAL fixed {8:4,16:7,32:1} on the all-off base — attribution for cmb_p4wide",
     "why": "Isolates the assignment change from the sqrt combination. Without "
            "it, a cmb_p4wide win cannot be attributed: better allocation, or "
            "the two axes finally cooperating? Also interesting on its own — if "
            "it beats lb_uniform (0.5557) then 4/7/1 is simply a better LB-TAL "
            "than anything in the 18 prior runs, all of which held P4 at 4 or "
            "below. Run it AFTER cmb_p4wide; if that one fails, this explains "
            "whether the allocation or the combination was at fault.",
     "params": _lb("fixed", level_topk={8: 4, 16: 7, 32: 1})},

    {"name": "lb_p3_3", "batch": BATCH, "confidence": "MEDIUM-LOW",
     "label": "fixed {8:3,16:4,32:4} — the one missing point on the P3 curve",
     "why": "The P3-budget sweep is already 5/6 complete (see the table above): "
            "8 -> 0.5482, ~6.6 -> 0.5477, 5 -> 0.5542, 4 -> 0.5557, 2 -> 0.5534. "
            "Only P3=3 is untested, and it sits between two measured points that "
            "differ by 0.23 pp, so the expected gain is small — it fills in a "
            "curve rather than chasing a win. Worth 1.5 h only if you want the "
            "budget response documented for the thesis.",
     "params": _lb("fixed", level_topk={8: 3, 16: 4, 32: 4})},

    # ---------------------------------------------------------------- TIER 4
    {"name": "nwd_blend25", "batch": BATCH, "confidence": "LOW",
     "label": "sqrt0703 + NWD blend 0.25 — global fallback if the hard switch is too abrupt",
     "why": "Fallback for nwd_small only. 'blend' applies NWD to every box "
            "including large ones, which is the failure mode cmb already "
            "demonstrated. Run only if nwd_small looks promising but unstable.",
     "params": _sq(use_nwd=True, nwd_mode="blend", nwd_weight=0.25, nwd_C=4.0)},
]

# -----------------------------------------------------------------------------
# BLOCKED — NOT IMPLEMENTED IN loss.py. Do not re-add without writing the code.
#
#   posboost / posboost_bag2   use_pos_boost, pos_boost_bag, pos_boost_small, ...
#
# default.yaml ships a complete 'Section P' for these — hand-tuned defaults, a
# comment reading "bag positives (recall cell 0.68 - the worst cell)", even a
# pos_boost_log flag "for posboost_report() telemetry". lossv2updated.py
# contains ZERO references to any of it, and no posboost_report() exists.
#
# So use_pos_boost=True passes the cfg whitelist, is silently ignored by the
# loss, and trains a bit-identical copy of sqrt0703 — 1.5 GPU-h to reproduce
# 0.5564. This was ranked HIGH confidence here until preflight caught it.
#
# Same status: use_freq_weight, use_lba, use_ardfl, use_peu. All whitelisted,
# none read. NOTE this kills the earlier suggestion to try use_lba against the
# s32 metric bias (SEL BIAS 0.34 in the stock footprint) — there is nothing to
# turn on. Section P remains the best-targeted idea for the ranking gap; it
# just has to be written first.
#
# preflight() now checks every activated key against the installed loss source
# and aborts, so this class of mistake cannot cost GPU time again.
# -----------------------------------------------------------------------------
# REMOVED, and why — so nobody re-adds them:
#   lb_balcap            balanced_capped resolves to {8:2,16:4,32:4} at topk=10,
#                        bit for bit lb_coarse_244. ALREADY RUN: 0.5534.
#   lb_balcap_qg50       same duplication of lb_coarse_qg50.
#   lb_uniform_tk13_qg50 tk13 -> ceil(13/3) = P3 budget 5, already measured
#                        ungated at 0.5542 — on the wrong side of the peak.
#   lb_coarse_qg50       P3=2 is off-peak (0.5534 vs 0.5557 at P3=4); gating a
#                        weaker base has to recover that deficit first.
#   lb_uniform_qg50      uniform ALONE does not damage medium/large mAP (large
#                        0.5775 vs baseline 0.5773), so there is nothing for the
#                        gate to fix there. The gate earns its keep only in the
#                        sqrt combination.
#   lb_uniform_qg70      no point picking a gate strength before qg50 reports.
#   cmb_uniform_qg50     superseded by cmb_p4wide_qg50. The footprint pass shows
#                        p4wide dominates uniform on every bucket — same-or-
#                        better small (P4 2.08->2.27), medium %<0.3 4.6%->1.0%,
#                        large P4 3.92->6.79 — so gating uniform starts from a
#                        strictly worse allocation.
# -----------------------------------------------------------------------------

# Suggested stop line: run TIER 1 + TIER 2 (5 runs, ~7.5 GPU-h) and re-read the
# per-size metrics before committing to tiers 3-4. If posboost or qfl lands,
# the base config changes and most of tiers 3-4 should be re-derived anyway.
STOP_AFTER = 5


def loss_provenance():
    """Record which loss.py is actually installed.

    The whole sweep is a no-op if ultralytics is running stock TAL, and the
    unknown params would be silently ignored rather than raising. Cheap
    insurance: log path + md5 + a marker for the features this sweep needs.
    """
    info = {"path": None, "md5": None, "has_lbtal": False,
            "has_balanced_capped": False, "has_quality_gate": False, "_body": ""}
    try:
        import ultralytics.utils.loss as _lm
        path = getattr(_lm, "__file__", None)
        info["path"] = path
        if path and os.path.exists(path):
            src = open(path, "rb").read()
            info["md5"] = hashlib.md5(src).hexdigest()[:12]
            txt = src.decode("utf-8", "ignore")
            # Comments stripped: default.yaml documents params that the loss
            # only MENTIONS in a docstring, or does not mention at all. A key
            # appearing solely in prose is not an implementation.
            info["_body"] = "\n".join(
                l for l in txt.split("\n") if not l.strip().startswith("#"))
            info["has_lbtal"] = "use_lbtal" in info["_body"]
            info["has_balanced_capped"] = "balanced_capped" in info["_body"]
            info["has_quality_gate"] = "quality_gate" in info["_body"]
    except Exception as e:
        info["error"] = str(e)
    return info


def unimplemented_params(params):
    """Keys this config tries to ACTIVATE that the installed loss never reads.

    Whitelisting in default.yaml is not implementation. default.yaml ships a
    full 'Section P' for pos_boost — use_pos_boost, pos_boost_bag, the lot —
    and lossv2updated.py contains ZERO references to any of it. Passing
    use_pos_boost=True is accepted by the cfg checker, silently ignored by the
    loss, and trains an exact copy of the base config. Same for use_freq_weight,
    use_lba, use_ardfl and use_peu.

    Only keys whose value DIFFERS from the all-off default are checked — a
    config leaving use_lba=False does not care whether it is implemented.
    """
    body = LOSS_INFO.get("_body") or ""
    if not body:
        return []
    dead = []
    for k, v in params.items():
        if k in _ALL_OFF and _ALL_OFF[k] == v:
            continue          # not being activated by this config
        if k not in body:
            dead.append(k)
    return sorted(dead)


LOSS_INFO = loss_provenance()


def preflight(todo):
    """Abort before burning GPU-hours on configs the loss cannot actually run."""
    print(f"  loss.py: {LOSS_INFO.get('path')}")
    print(f"  md5={LOSS_INFO.get('md5')}  lbtal={LOSS_INFO.get('has_lbtal')} "
          f"balanced_capped={LOSS_INFO.get('has_balanced_capped')} "
          f"quality_gate={LOSS_INFO.get('has_quality_gate')}")
    missing = [k for k in ("has_lbtal", "has_balanced_capped", "has_quality_gate")
               if not LOSS_INFO.get(k)]
    if missing:
        print(f"\n  [ABORT] installed loss.py is missing: {', '.join(missing)}")
        print("  Install lossv2updated.py as ultralytics/utils/loss.py first.")
        return False

    # --- mixed-loss-file check ----------------------------------------------
    wanted = {r.get("loss", "lossv2updated") for r in todo}
    if len(wanted) > 1:
        print(f"\n  [ABORT] selection mixes configs needing DIFFERENT loss.py files:")
        for w in sorted(wanted):
            print(f"            {w}: "
                  f"{', '.join(r['name'] for r in todo if r.get('loss','lossv2updated')==w)}")
        print("  The two loss files read disjoint hyperparameter sets (71 vs 31,")
        print("  intersection zero). Run them as separate sweeps with the right")
        print("  file installed, or the wrong half trains as its base config.")
        return False

    # --- per-config no-op check ---------------------------------------------
    bad = {r["name"]: unimplemented_params(r["params"]) for r in todo}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print("\n  [ABORT] these configs set params the installed loss NEVER READS.")
        print("          They would train an exact copy of their base config and")
        print("          waste 1.5 GPU-h each producing a number you already have:")
        for name, ks in bad.items():
            print(f"            {name:<20} {', '.join(ks)}")
        print("\n  default.yaml whitelists these keys, which is why the cfg checker")
        print("  accepts them — but whitelisting is not implementation. Either")
        print("  implement them in loss.py or drop the config.")
        return False

    # --- run-dir collision check (exist_ok=False raises mid-sweep) -----------
    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))]
    if clash:
        print(f"\n  [ABORT] run dirs already exist (exist_ok=False): {', '.join(clash)}")
        print("  Delete or rename them, or those runs will fail after the")
        print("  earlier ones have already burned their hours.")
        return False
    return True


def collect_metrics(tm):
    """Pull everything the val object exposes, incl. per-class.

    NOTE: ultralytics' DetMetrics has no per-size (small/medium/large)
    breakdown — that comes from CocoEvalAllFolders_luggage.py. Small mAP and
    AR50_small are the stated target of this sweep, so run the coco eval on
    the `weights` path recorded below before drawing conclusions.
    """
    out = {}
    try:
        box = tm.box
        out["map50"] = float(box.map50)
        out["map5095"] = float(box.map)
        out["precision"] = float(box.mp)
        out["recall"] = float(box.mr)
        out["map75"] = float(getattr(box, "map75", float("nan")))
        names = getattr(tm, "names", {}) or {}
        per_class = {}
        ap50 = getattr(box, "ap50", None)
        ap = getattr(box, "maps", None)
        idxs = list(getattr(box, "ap_class_index", [])) or list(range(len(names)))
        for i, ci in enumerate(idxs):
            cname = names.get(ci, str(ci)) if isinstance(names, dict) else str(ci)
            rec = {}
            if ap50 is not None and i < len(ap50):
                rec["AP50"] = float(ap50[i])
            if ap is not None and ci < len(ap):
                rec["AP50_95"] = float(ap[ci])
            if rec:
                per_class[cname] = rec
        if per_class:
            out["per_class"] = per_class
    except Exception as e:
        out["error"] = str(e)
    return out


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
    name, params, batch = rc["name"], rc["params"], rc["batch"]
    print(f"\n{'=' * 76}\n  RUN {name}   [confidence: {rc.get('confidence', '?')}]\n"
          f"  {rc['label']}\n"
          f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch, "seed": SEED,
                       "loss_file": LOSS_INFO}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")
    rd = getattr(results, "results_dict", {}) or {}
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "label": rc["label"], "batch": batch, "hours": hours,
           "confidence": rc.get("confidence"), "why": rc.get("why"),
           "seed": SEED, "epochs": EPOCHS, "imgsz": IMG_SIZE,
           "save_dir": save_dir, "weights": weights,
           "params": copy.deepcopy(params), "loss_file": LOSS_INFO,
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        tm = YOLO(weights).val(
            data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
        out["test_metrics"] = collect_metrics(tm)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    ref = BASELINE_TEST_MAP5095
    print(f"\n{'=' * 88}\n  v3 RESULTS (test) — assignment + cls + localisation axes\n{'=' * 88}")
    print(f"{'run':<22}{'conf':>18}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'vs_best':>10}{'h':>6}")
    print("-" * 88)
    for r in sorted(res, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = ("%+10.2f" % ((r["test_map5095"] - ref) * 100)) if ref else "-"
        du = ("%+10.2f" % ((r["test_map5095"] - 0.5564) * 100))
        print(f"{r['name']:<22}{str(r.get('confidence', '?')):>18}"
              f"{r['test_map50'] * 100:>9.2f}{r['test_map5095'] * 100:>11.2f}{d}{du}{r['hours']:>6.1f}")
    print(f"\n  anchor {ref*100:.2f} | lb_uniform 55.57 | sqrt0703 55.64 (vs_best is vs sqrt0703)")
    print("  vs_best > 0 -> this config beat the best run to date.")
    print("\n  [!] For 'posboost' runs, read posboost_report() per-class mean fg score.")
    print("      That telemetry tests the ranking diagnosis directly and is more")
    print("      informative than the mAP delta, which may sit inside seed noise.")
    print("\n  [!] Measured seed noise (lb_uniform vs lb_uniform_seed1, same config):")
    print("        overall mAP50-95  ~0.12 pp")
    print("        large   mAP50-95  ~2.06 pp")
    print("      Treat gaps below those as unresolved, not as a ranking.")
    print("\n  [!] Per-size metrics (small mAP, AR50_small) are the real target and")
    print("      are NOT produced by ultralytics val. Run CocoEvalAllFolders_luggage.py")
    print("      on the weights below:")
    for r in res:
        w = r.get("weights")
        if w:
            print(f"        {r['name']:<22} {w}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_all = "--all" in sys.argv[1:]
    only = set(args)
    if only:
        todo = [r for r in RUNS if r["name"] in only]
    elif run_all:
        todo = list(RUNS)
    else:
        todo = RUNS[:STOP_AFTER]   # tiers 1-2; pass --all to run everything

    print(f"\n{'=' * 88}\n  v3 sweep — ordered by confidence of improvement (highest first)")
    print(f"  {len(todo)} of {len(RUNS)} configs  (~{1.5*len(todo):.0f} GPU-h)")
    if not only and not run_all:
        print(f"  [default] tiers 1-2 only. Re-read per-size metrics before the rest.")
        print(f"  [hint]    --all runs all {len(RUNS)}; or name configs explicitly.")
    print(f"{'=' * 88}")
    for i, r in enumerate(todo, 1):
        print(f"  {i:2d}. {r['name']:<22} {str(r.get('confidence', '?')):<22} {r['label'][:38]}")
    print()
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
            res.append({"name": r["name"], "batch": r["batch"], "hours": float("nan"),
                        "val_map5095": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
