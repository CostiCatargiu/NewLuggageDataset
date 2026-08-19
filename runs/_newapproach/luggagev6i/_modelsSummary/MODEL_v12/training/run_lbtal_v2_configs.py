#!/usr/bin/env python3
"""
v4 sweep — Section S: the scale-normalised confidence target.

STATE OF PLAY after 28 runs (see the COMPLETED block under RUNS for all nine
v3 results). sqrt0703 = 0.5564 is still champion; cmb_p4wide = 0.5560 ties it
overall while winning 8/8 SMALL-object metrics and posting the best mAP50
(0.8105) of anything trained. Nothing has beaten sqrt0703.

WHY THE PREVIOUS AXES ARE CLOSED
  ASSIGNMENT   the P3-budget curve is flat inside noise (2:0.5534 3:0.5521
               4:0.5557 5:0.5542, seed noise 0.0012). p4wide is the best
               allocation found and it only helps in the sqrt combination.
               The quality gate is destructive (-4.01 pp).
  CLASSIFIER   pos_boost -0.58 pp and it made bag WORSE, the exact cell it was
               tuned for. QFL at beta=2.0 lost 8.39 pp.
  LOCALISATION NWD failed (-1.26 pp) and made the small localisation ratio
               worse. That ratio has stayed in 0.635-0.658 across all 28 runs
               while overall mAP50-95 spanned 8.39 pp — nothing has moved it.

WHY SECTION S IS DIFFERENT — it attacks a cause the others cannot reach.
TAL builds the classification target as
    norm_align_metric = align_metric * pos_overlaps / pos_align_metrics
so a GT is trained toward a confidence ceiling equal to its own best
achievable IoU. Measured peak target per GT (diag_anchor_footprint 5b):

    small 0.8365    medium 0.8933    large 0.9028      (-6.4% small vs medium)

The model is explicitly taught to be less confident about small luggage, which
is the AR50_small 0.95 -> R50_small 0.70 gap.

Two measurements make this orthogonal to everything already tried:
  * the ceiling is INVARIANT TO ALLOCATION — under LB-TAL p4wide it reads
    0.8366, identical to stock to four decimals. No assigner can move it, so
    all 28 runs were structurally incapable of touching it.
  * it is out of reach of LOSS WEIGHTING — pos_boost scaled the loss on
    positives and moved R50_small by 0.34 pp while costing 0.58 pp, because
    weighting cannot push a prediction past its own target.

SNT rescales by ema_global/ema_size so the target measures localisation
quality relative to what is achievable at that scale: small x1.026, medium
x0.962, large x0.952, all mapping to ~0.861.

THE RISK, stated up front: raising targets on inherently poorly-localised
boxes trains toward overconfidence. WATCH P50_small AS CLOSELY AS R50_small.
If precision falls further than recall rises, lower snt_strength (snt_half is
already queued for that) rather than declaring the mechanism dead.

Also note the trade every intervention has obeyed: across all 28 runs
AR50_small and R50_small correlate at r = -0.639. Only cmb_p4wide and
lb_p4wide raised both. QFL is the proof the ceilings are movable at all — it
won ALL SIX AR metrics including AR50_large = 1.0000, and failed on ranking,
not detection. beta 0.5/1.0 remains a live lead not queued here.

READ small mAP + AR50_small + P50_small (CocoEvalAllFolders_luggage.py). This
script cannot produce per-size metrics — ultralytics val has no small/medium/
large breakdown — so it records each run's weights path for the coco eval.

REQUIRES lossv2updated.py installed as ultralytics/utils/loss.py. Section P
(pos_boost) was ported into it from loss_custom_v3_fixed.py, so the two files
are no longer disjoint for these configs. default.yaml whitelists the union of
both, meaning the cfg checker accepts keys the installed loss may ignore;
preflight() therefore scans the installed loss SOURCE for every key a selected
config activates and aborts on a no-op rather than training a silent copy of
the base config. It also refuses a selection mixing the two loss files, and
aborts on run-directory collisions before burning any hours.

Usage:
    python run_lbtal_v2_configs.py            # snt + cmb_p4wide_snt (3 GPU-h)
    python run_lbtal_v2_configs.py --all      # adds snt_half
    python run_lbtal_v2_configs.py snt
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
    {"name": "snt", "batch": BATCH, "confidence": "HIGH",
     "label": "sqrt0703 + scale-normalised confidence target (Section S, new)",
     "why": "The only mechanism here that attacks a cause NOTHING ELSE CAN "
            "REACH. TAL trains each GT toward a confidence ceiling equal to its "
            "own best achievable IoU (pos_overlaps). Measured peak target per "
            "GT: small 0.8365 / medium 0.8933 / large 0.9028 — small objects "
            "are explicitly taught to be 6.4% less confident, which is the "
            "AR50_small 0.95 -> R50_small 0.70 gap. "
            "CRITICALLY, the same diagnostic under LB-TAL p4wide gives small "
            "0.8366 — identical to four decimals. The ceiling is INVARIANT to "
            "allocation, so all 28 prior runs were structurally incapable of "
            "moving it, and so is any future assigner. It is also out of reach "
            "of loss weighting: pos_boost scaled the loss on positives and "
            "moved R50_small by 0.34 pp while costing 0.58 pp overall, because "
            "weighting cannot push a prediction past its own target. "
            "SNT rescales by ema_global/ema_size so the target measures "
            "localisation quality RELATIVE TO WHAT IS ACHIEVABLE AT THAT "
            "SCALE — small +2.6%, medium -3.8%, large -4.8%, all mapping to "
            "~0.861. "
            "RISK: raising targets on inherently poorly-localised boxes trains "
            "toward overconfidence. WATCH P50_small AS CLOSELY AS R50_small — "
            "if precision falls further than recall rises, lower snt_strength "
            "rather than declaring the mechanism dead.",
     "params": _sq(use_snt=True, snt_strength=1.0, snt_momentum=0.02,
                   snt_max_scale=1.15, snt_warmup_epochs=3,
                   snt_small_px=48.0, snt_medium_px=96.0)},

    {"name": "cmb_p4wide_snt", "batch": BATCH, "confidence": "MEDIUM-HIGH",
     "label": "cmb_p4wide + scale-normalised target — the two orthogonal wins",
     "why": "Combines the best measured allocation (cmb_p4wide: 8/8 small-object "
            "metrics, best mAP50 of all 28 runs) with the one mechanism that "
            "acts on the confidence ceiling. They are provably orthogonal — the "
            "footprint pass shows p4wide leaves the peak target unchanged "
            "(0.8366 vs stock 0.8365), so SNT has exactly the same work to do "
            "under either allocation. "
            "One caution from the same measurement: p4wide collapses the MEAN "
            "per-anchor target for medium/large by ~30% (0.7188 -> 0.5002 and "
            "0.7498 -> 0.5153) because the wide P4 budget admits marginal "
            "anchors. That is the better explanation for cmb_p4wide's -4.4 pp "
            "on large mAP50-95 than the junk-positive story, and SNT does not "
            "address it. If large regresses again here, the answer is a "
            "narrower P4 budget, not more SNT.",
     "params": _sq_lb("fixed", level_topk={8: 4, 16: 7, 32: 1},
                      use_snt=True, snt_strength=1.0, snt_momentum=0.02,
                      snt_max_scale=1.15, snt_warmup_epochs=3,
                      snt_small_px=48.0, snt_medium_px=96.0)},

    {"name": "snt_half", "batch": BATCH, "confidence": "MEDIUM (conditional)",
     "label": "SNT at half strength — partial correction",
     "why": "Run only if snt moves recall but costs too much precision. "
            "strength=0.5 interpolates the multiplier halfway to full "
            "equalisation (small x1.013 instead of x1.026), so the trade can be "
            "tuned rather than accepted or abandoned wholesale.",
     "params": _sq(use_snt=True, snt_strength=0.5, snt_momentum=0.02,
                   snt_max_scale=1.15, snt_warmup_epochs=3,
                   snt_small_px=48.0, snt_medium_px=96.0)},

]

# -----------------------------------------------------------------------------
# COMPLETED — runs_lbtal_v2__test_full_dataset.json. Do NOT re-run; results are
# recorded here so the sweep does not silently repeat a known outcome.
# References: anchor 0.5477, sqrt0703 0.5564 (still champion), lb_uniform 0.5557.
# Seed noise 0.0012 overall, 0.0206 on large.
#
#   config             mAP50-95   vs sqrt0703   verdict
#   cmb_p4wide           0.5560       -0.04     TIE overall, but wins 8/8 SMALL
#                                              metrics and posts the best mAP50
#                                              (0.8105) and best small-object
#                                              aggregate of ALL 28 runs. Costs
#                                              -3.4 pp on large. The P4 fix
#                                              worked: old cmb was 0.5510.
#   clsw_sqrt            0.5528       -0.35     best precision (0.819), no gain
#   nwd_blend25          0.5525       -0.39     —
#   lb_p3_3              0.5521       -0.43     fills the P3 curve at 3; the
#                                              curve is FLAT/noisy, not the
#                                              clean peak claimed earlier
#                                              (2:0.5534 3:0.5521 4:0.5557
#                                               5:0.5542)
#   posboost             0.5506       -0.58     FAILED. Made bag WORSE (0.4680
#                                              vs 0.4734) — the exact cell
#                                              Section P was tuned for. Cause:
#                                              loss weight cannot beat the
#                                              target ceiling -> Section S.
#   lb_p4wide            0.5503       -0.61     p4wide is WORSE alone than
#                                              lb_uniform (0.5557); its value
#                                              is only in the sqrt combination
#   nwd_small            0.5438       -1.26     FAILED, and made the small
#                                              localisation ratio WORSE (0.646
#                                              vs 0.654)
#   cmb_p4wide_qg50      0.5162       -4.01     GATE IS DESTRUCTIVE. AR50_s
#                                              +2.21 but R50_s -3.67
#   qfl                  0.4725       -8.39     CATASTROPHIC at beta=2.0 —
#                                              precision 0.676 vs 0.809. BUT it
#                                              won ALL SIX AR metrics incl.
#                                              AR50_large = 1.0000. Not a
#                                              detection failure, a ranking
#                                              one. beta 0.5/1.0 is a live lead.
#
# THE PATTERN: across all 28 runs AR50_small and R50_small correlate at
# r = -0.639. Interventions trade the recall ceiling against the operating
# point instead of adding. Only cmb_p4wide and lb_p4wide raised both.
# -----------------------------------------------------------------------------

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
STOP_AFTER = 2


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
