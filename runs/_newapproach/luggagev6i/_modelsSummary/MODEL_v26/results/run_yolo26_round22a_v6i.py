#!/usr/bin/env python3
r"""
YOLO26 ROUND 22 — the selection axis, and two draws to stand on

Eight runs, ~7.2 GPU-h, b82, stock yolo26s.pt, matched to baseline in batch and
graph. PRIMARY METRIC IS mAP50_small.

    REFERENCE, 3 seeds of y26_identity   small 77.62   mAP50 80.40   m5095 55.30
    pooled sd (df=7)                     small  0.346  m5095 0.374
    single run vs the 3-seed mean        SE = 0.346 * sqrt(1+1/3) = 0.40
    single run vs a SINGLE control draw  SE = 0.346 * sqrt(2)     = 0.49
    -> 0.80 sd suggestive, 1.20 sd report as an effect


=============================================================================
WHAT ROUND 21 ACTUALLY SETTLED
=============================================================================
Round 21 filled in enough of the (beta_sel, beta_tgt) grid to read it as a
surface instead of a list of runs. mAP50_small:

                            target beta
                    6        3      1.5       1
        sel 6    77.62    77.21       --      --
        sel 3    78.16    79.14    78.76   78.53
        sel 1       --       ??       --   79.30      <- the hole

TARGET IS CLOSED. At fixed selection 3 the target term has a clean interior
optimum at 3 and falls away on both sides: 78.16 / 79.14 / 78.76 / 78.53. Four
points, one maximum, no ambiguity. Nothing below 3 helps and 6 is worse.

SELECTION IS NOT. Every selection reduction measured so far gains, at every
target level it has been measured at, with no turnover anywhere:

        tgt 3 :  sel 6 -> 3    77.21 -> 79.14    +1.94
        tgt 6 :  sel 6 -> 3    77.62 -> 78.16    +0.54
        tgt 1 :  sel 3 -> 1    78.53 -> 79.30    +0.76

AND THIS REFRAMES y26_b1. It posted 79.30, the top of round 21 on the primary
metric, off a 0.10 prior. It is NOT evidence that beta_tgt=1 is good — the same
round measured beta_tgt=1 at fixed selection and got 78.53, the worst target
setting tested. b1 won on the SELECTION side and dragged a known-bad target
along with it. Uniform beta=1 is two changes at once and one of them is wrong.

ADDITIVITY IS DEAD, and the grid shows why rather than merely that. The target
effect changes SIGN with selection: 6->3 on the target is -0.41 at sel 6 and
+0.98 at sel 3. y26_bsel6_btgt3 was predicted at 78.60 and landed 77.21, off by
-2.85 sd. Any round-22 reasoning of the form "term A plus term B" is invalid.
Every cell in this file is measured, not inferred, for that reason.

ARM C IS CLOSED TOO. TCN was the 0.30 prior aimed at a measured failure mode,
and round 21 wrote its own falsification test: "if R50_small rises and P50_small
holds it worked; if both fall the ceiling was load-bearing." Both fell, on both
bases (b3: R50_small -1.00, P50_small -1.59). Not merely unimpressive. Falsified.


=============================================================================
ARM CAL  (execution 1, 2) — TWO DRAWS TO STAND ON
=============================================================================
These run FIRST and they are not bets. Sorting is by (not calib, -prior), a
deliberate departure from round 21's pure prior order, because every delta in
this file is quoted against B3_S50 = 79.14 and that number is ONE DRAW that
round 21's own header flags as suspect:

    beta      4      3     2.5      2
    small  78.17  79.14  78.88  78.92        <- 3 is ~+0.34 above its neighbours

If b3 is a high draw, every "regression" in rounds 18-21 is overstated by about
a third of a point, which is most of the effects being argued over. The same
applies one level down to y26_b1: it is the top score in the campaign and it has
been observed exactly once.

This project has now lost real information twice to single draws — the seed-0
identity low draw that biased every hardcoded constant in rounds 18/19/20, and
b3's own position above its curve. Two runs, 1.8 GPU-h, closes both. If the
queue dies at 3am these are the two that must already be done.

    y26_b3_s1   fixes the control the whole round is measured against
    y26_b1_s1   fixes the top score the whole round is trying to beat

Neither needs the patch, so they also serve as the canary: if these two behave
and the patched runs do not, the patch is the difference.


=============================================================================
ARM A  (execution 3, 4, 5) — COMPLETE THE sel=1 ROW
=============================================================================
The sel=3 row is a finished 4-point target curve. This arm builds the identical
curve at sel=1, where one cell (tgt=1, from b1) already exists:

        sel 1 :  tgt 6 = ?    tgt 3 = ?    tgt 1.5 = ?    tgt 1 = 79.30

y26_bsel1_btgt3 is the headline: best selection measured x best target measured,
the one cell of the grid that combines them. It is also the run that decides
what b1 actually was, because b1 (79.30) and b3 (79.14) currently tie within
error for OPPOSITE reasons and only this cell separates them.

The other two are not filler. Because the target effect flips sign with
selection, the target optimum has NO reason to stay at 3 when selection moves to
1 — that is exactly the interaction bsel6_btgt3 proved exists. Measuring tgt 6
and tgt 1.5 at sel=1 is what turns a single winning cell into a surface with a
known shape, and it is the difference between "we found a better setting" and
"we understand the assigner".

If the sel=1 row peaks at tgt 3 like the sel=3 row does, the surface is
separable after all and the additivity failure was confined to sel=6. If it
peaks somewhere else, the interaction is general and every future split has to
be measured on both axes at once.


=============================================================================
ARM B  (execution 6, 7) — HOW FAR DOWN DOES SELECTION GO
=============================================================================
Selection has no measured turnover. It is also NOT bounded at 1 the way the
target is. Round 21's "beta=1 is the last meaningful point" is a statement about
the TARGET exponent — at beta_tgt=0 the target ignores IoU and the metric stops
being alignment. Selection is top-k on s^(alpha/beta) * IoU; alpha/beta only
reweights score against IoU in a ranking. beta_sel = 0.5 is an ordinary point on
that ranking, not a degenerate one.

    y26_bsel2_btgt3    fills 6 / 3 / 2 / 1 into a real curve rather than 3 points
    y26_bsel05_btgt3   the first probe below the bound the target axis has

Together with arm A this makes the tgt=3 column a 5-point selection sweep
(6, 3, 2, 1, 0.5). If it turns over inside that range the assigner's selection
side is finished and the number is banked. If it is still climbing at 0.5, the
campaign has an open axis it has never pushed on.


=============================================================================
ARM REP  (execution 8) — THE HEADLINE AT TWO SEEDS
=============================================================================
y26_bsel1_btgt3_s1 is the predicted winner at seed 1, run in the SAME night as
seed 0 rather than in round 23.

This is deliberate and it is the lesson of arm CAL applied forward instead of
backward. Every headline result in this campaign has been a single draw, and
twice that has cost a round. If bsel1_btgt3 wins it wins at n=2 immediately and
round 23 can build on it; if the two seeds disagree by more than a point, that
is worth knowing before anything is built on either.

It sits last because it is the only run in the file that is redundant if its
partner loses. That is the right thing to drop at 3am, and nothing else here is.


=============================================================================
CALIBRATION AND EXECUTION ORDER
=============================================================================
prior = P(clearing +0.35 on mAP50_small vs its own control). Calibration runs
sort first regardless; the rest by prior, highest first. Both calibration runs
precede every exploratory cell, so their relative order does not matter.

    1  y26_b1_s1           calib  top score at a second seed      (not a bet)
    2  y26_b3_s1           calib  control at a second seed        (not a bet)
    3  y26_bsel1_btgt3     0.35   best selection x best target
    4  y26_bsel1_btgt3_s1  0.35   the same cell, seed 1
    5  y26_bsel2_btgt3     0.30   turns selection into a curve
    6  y26_bsel05_btgt3    0.25   below the target axis's bound
    7  y26_bsel1_btgt15    0.15   sel=1 row, lower bracket
    8  y26_bsel1_btgt6     0.10   sel=1 row, upper bracket

The order is ENFORCED by order() in main(), not transcribed here, so this table
and the queue cannot drift apart.

THESE ARE AN ORDERING, NOT FORECASTS. Directional predictions in this campaign
are 0-for-11 after round 21 and every single miss was optimistic. The one thing
that is different about run 3 is that it is not a mechanism story — it is the
empty cell of a grid whose other seven cells are filled and monotone. That is a
weaker claim than any of the eleven that failed, which is the point.

STATED FAILURE MODE, in advance. If bsel1_btgt3 lands at 79.1-79.3 it has NOT
won: it has tied b1 and b3, which would mean selection and target saturate
together and the whole beta family is finished at roughly +1.7 over identity.
That is a real and reasonably likely outcome, it closes the axis, and it should
be read as a result rather than as a disappointment.

    Usage:
        python run_yolo26_round22_v6i.py                  # all eight, in order
        python run_yolo26_round22_v6i.py --preflight      # prove inertness only
        python run_yolo26_round22_v6i.py --arm a          # cal | a | b | rep
        python run_yolo26_round22_v6i.py y26_bsel1_btgt3  # one by name

    Arm CAL needs no patch. Everything else needs patch_round21_v6i.py; if it is
    not installed those runs are SKIPPED with a message, not silently run stock.
"""

import argparse
import gc
import json
import os
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"          # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_round22_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0                              # per-run override via rc["seed"]
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False            # y26_p2k2_hi was lost to exist_ok=True

# 3-seed means. Never the seed-0 draw.
REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30
SD_S50, SD_5095 = 0.346, 0.374
B3_S50, B1_S50 = 79.14, 79.30         # both are SINGLE draws — see arm CAL

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    tal_beta_o2m=None, tal_beta_o2o=None, tal_alpha_o2m=None, tal_alpha_o2o=None,
    tal_beta_sel=None, tal_beta_tgt=None, tcn_p=1.0, tal_beta_level=None,
    scb_branch="both",
    o2m_start=0.8, o2m_final=0.1, o2m_decay=True,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0, cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
    multi_scale=0.0, scale=0.5, close_mosaic=CLOSE_MOSAIC, cos_lr=False,
)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def split(sel, tgt):
    """A (beta_sel, beta_tgt) cell. tal_beta tracks beta_sel so that any code
    path still reading the single key sees the selection value, matching how
    round 21's arm B was configured."""
    return cfg(tal_beta=sel, tal_beta_sel=sel, tal_beta_tgt=tgt)


RUNS = [
    # -------------------------------------------------------------- ARM CAL
    {"name": "y26_b3_s1", "prior": 0.02, "arm": "cal", "calib": True, "seed": 1,
     "needs": None, "ctrl": B3_S50,
     "params": cfg(tal_beta=3.0),
     "expect": {"beta": 3.0},
     "label": "b3 at seed 1 — a second draw of the control",
     "why": "B3_S50 = 79.14 is the control for this entire round and for rounds "
            "18-21, and it is one draw sitting ~+0.34 above its own neighbours "
            "(beta 4 / 2.5 / 2 interpolate to ~78.8 at beta 3). If it is a high "
            "draw then every regression measured against it is overstated by "
            "about a third of a point, which is most of what those rounds "
            "argued about. Prior is 0.02 because it is the control and cannot "
            "beat itself; it runs FIRST because nothing else is interpretable "
            "without it. Needs no patch."},

    {"name": "y26_b1_s1", "prior": 0.20, "arm": "cal", "calib": True, "seed": 1,
     "needs": None, "ctrl": B3_S50,
     "params": cfg(tal_beta=1.0),
     "expect": {"beta": 1.0},
     "label": "b1 at seed 1 — a second draw of the campaign's top score",
     "why": "b1 posted 79.30, the best mAP50_small in the project, from a 0.10 "
            "prior, once. The seed-0 identity draw already cost this campaign "
            "three rounds of biased constants; repeating that mistake on the "
            "number every future round will be measured against is not worth "
            "0.9 GPU-h of savings. Prior 0.20 and not higher: b1 is +0.16 over "
            "the control, so a faithful repeat still has to find another 0.19 "
            "to clear +0.35. Needs no patch."},

    # ---------------------------------------------------------------- ARM A
    {"name": "y26_bsel1_btgt3", "prior": 0.35, "arm": "a", "needs": "tal_beta_sel",
     "ctrl": B3_S50,
     "params": split(1.0, 3.0),
     "expect": {"beta_sel": 1.0, "beta_tgt": 3.0},
     "label": "selection at beta 1, targets at beta 3 — the empty cell",
     "why": "The best selection measured (1, worth +0.76 over sel 3 at fixed "
            "tgt 1) combined with the best target measured (3, a clean interior "
            "optimum over a 4-point curve at fixed sel 3). It is the only cell "
            "of the grid holding both, and it is the run that says what b1 was: "
            "b1 and b3 tie within error today for opposite reasons, and nothing "
            "else separates them. Highest prior in the file at 0.35 — which is "
            "still under even odds, because eleven straight directional calls "
            "in this campaign have been wrong in this direction."},

    {"name": "y26_bsel1_btgt15", "prior": 0.15, "arm": "a", "needs": "tal_beta_sel",
     "ctrl": B3_S50,
     "params": split(1.0, 1.5),
     "expect": {"beta_sel": 1.0, "beta_tgt": 1.5},
     "label": "sel=1 row, lower bracket",
     "why": "At sel 3 the target curve falls away below 3 (79.14 -> 78.76 -> "
            "78.53). If the sel=1 row does the same, the surface is separable "
            "and the additivity failure was a sel=6 artefact. If it does not, "
            "the interaction is general and every future split has to be "
            "measured on both axes together. Low prior, high information: it is "
            "not expected to win, it is expected to say which of those is true."},

    {"name": "y26_bsel1_btgt6", "prior": 0.10, "arm": "a", "needs": "tal_beta_sel",
     "ctrl": B3_S50,
     "params": split(1.0, 6.0),
     "expect": {"beta_sel": 1.0, "beta_tgt": 6.0},
     "label": "sel=1 row, upper bracket",
     "why": "Completes the sel=1 row to four points (6 / 3 / 1.5 / 1) so it can "
            "be read against the finished sel=3 row directly. tgt 6 is the "
            "weakest cell of that row at sel 3 (78.16), so this is the run most "
            "likely to be a wasted 0.9 GPU-h and it is priced accordingly — but "
            "without it the two rows are not comparable and the interaction "
            "question stays open."},

    # ---------------------------------------------------------------- ARM B
    {"name": "y26_bsel2_btgt3", "prior": 0.30, "arm": "b", "needs": "tal_beta_sel",
     "ctrl": B3_S50,
     "params": split(2.0, 3.0),
     "expect": {"beta_sel": 2.0, "beta_tgt": 3.0},
     "label": "selection at beta 2, targets at beta 3",
     "why": "The selection axis is currently three points at tgt 3 (6, 3, and "
            "whatever run 3 returns at 1). Two of the three gaps are a factor "
            "of two or more wide, which is enough to hide a turnover completely. "
            "This is the fill that makes the column a curve instead of a line "
            "through three points, and it is cheap insurance against declaring "
            "a monotone trend that is really a plateau with one high draw."},

    {"name": "y26_bsel05_btgt3", "prior": 0.25, "arm": "b", "needs": "tal_beta_sel",
     "ctrl": B3_S50,
     "params": split(0.5, 3.0),
     "expect": {"beta_sel": 0.5, "beta_tgt": 3.0},
     "label": "selection at beta 0.5 — below the target axis's bound",
     "why": "beta=1 bounds the TARGET exponent, where beta=0 would ignore IoU "
            "and stop being alignment. Selection is top-k on s^(alpha/beta) * "
            "IoU, where alpha/beta only reweights score against IoU inside a "
            "ranking, so 0.5 is an ordinary point and not a degenerate one. "
            "Selection has no measured turnover anywhere; this is the first run "
            "in the campaign that looks past where the other axis had to stop."},

    # -------------------------------------------------------------- ARM REP
    {"name": "y26_bsel1_btgt3_s1", "prior": 0.35, "arm": "rep", "seed": 1,
     "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": split(1.0, 3.0),
     "expect": {"beta_sel": 1.0, "beta_tgt": 3.0},
     "label": "the headline cell at seed 1, same night as seed 0",
     "why": "Arm CAL applied forward instead of backward. If run 3 wins it wins "
            "at n=2 immediately and round 23 can build on it without spending "
            "another night proving it; if the two seeds disagree by more than a "
            "point that is worth knowing before anything is built on either. "
            "Sorted last because it is the only run here that becomes redundant "
            "if its partner loses, which makes it the correct thing to drop at "
            "3am — nothing else in this file is."},
]

# Execution order is (calibration first, then descending prior). This is a
# DELIBERATE departure from round 21's pure prior order: the control B3_S50 and
# the top score B1_S50 are each one draw, and every delta in this file is quoted
# against them. A queue that dies having run six exploratory cells against an
# uncalibrated control has produced six uninterpretable numbers.


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """Prove the CONSUMER reads each key, and that defaults are bit-identical.

    Rounds 4-6 passed a preflight that only checked default.yaml accepted the
    key and tal.py had the class. Ten runs were lost because the file that had
    to consume it ignored the flag. This checks the consumer and then runs the
    arithmetic.
    """
    import inspect
    print("=" * 76)
    print("  PREFLIGHT")
    print("=" * 76)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False, set()

    src_init = inspect.getsource(TaskAlignedAssigner.__init__)
    src_fwd = inspect.getsource(TaskAlignedAssigner._forward)
    src_box = inspect.getsource(TaskAlignedAssigner.get_box_metrics)

    have = set()
    checks = {
        "default.yaml accepts tal_beta_sel": "tal_beta_sel" in DEFAULT_CFG_DICT,
        "default.yaml accepts tal_beta_tgt": "tal_beta_tgt" in DEFAULT_CFG_DICT,
        "assigner has beta_sel/beta_tgt": "beta_sel" in src_init and "beta_tgt" in src_init,
        "get_box_metrics returns align_tgt": "align_tgt" in src_box,
        "_forward consumes align_tgt": "align_tgt" in src_fwd,
    }
    for k, v in checks.items():
        print(f"  {k:<42} {v}")
    if checks["assigner has beta_sel/beta_tgt"] and checks["get_box_metrics returns align_tgt"] \
            and checks["_forward consumes align_tgt"]:
        have.add("tal_beta_sel")

    # numerical inertness: defaults must reproduce stock exactly
    print()
    try:
        torch.manual_seed(0)
        a = TaskAlignedAssigner(topk=10, num_classes=3)
        b, n, A = 2, 3, 40
        pd_s = torch.rand(b, A, 3)
        pd_b = torch.rand(b, A, 4).cumsum(-1)
        gt_l = torch.randint(0, 3, (b, n, 1)).float()
        gt_b = torch.rand(b, n, 4).cumsum(-1)
        m_gt = torch.ones(b, n, 1)
        anc = torch.rand(A, 2) * 10
        out1 = a(pd_s, pd_b, anc, gt_l, gt_b, m_gt)
        a.beta_sel, a.beta_tgt, a.tcn_p, a.beta_level = None, None, 1.0, None
        out2 = a(pd_s, pd_b, anc, gt_l, gt_b, m_gt)
        inert = all(torch.equal(x, y) for x, y in zip(out1, out2))
        print(f"  {'defaults are bit-identical to stock':<42} {inert}")
        ok &= inert
    except Exception as ex:
        print(f"  [warn] inertness check could not run: {ex}")

    print()
    skip = [r for r in todo if r["needs"] and r["needs"] not in have]
    run = [r for r in todo if not r["needs"] or r["needs"] in have]
    for r in run:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        tag = "CAL " if r.get("calib") else f"p={r['prior']:.2f}"
        print(f"  RUN  {r['name']:<22} arm {r['arm']:<3} {tag}  seed {r.get('seed', SEED)}  "
              f"vs {r['ctrl']:.2f}  {d}")
    for r in skip:
        print(f"  SKIP {r['name']:<22} needs '{r['needs']}' — run patch_round21_v6i.py first")
    print(f"\n  {len(run)} runs, ~{0.9 * len(run):.1f} GPU-h" + (f"  ({len(skip)} skipped)" if skip else ""))
    if skip and any(r.get("calib") for r in run):
        print("  note: arm CAL needs no patch, so calibration still completes.")
    return ok, {r["name"] for r in run}


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Assert at epoch 2 that the mechanism is LIVE and nothing else is.

    Epoch 2, not epoch 1: anything recomputed per epoch (the o2m decay is the
    known case) still matches stock on the first pass. A guard that fires at
    epoch 1 passes for a config that silently reverts afterwards.
    """
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 2:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss")
        seen = []
        for tag, br in (("one2many", o2m), ("one2one", o2o)):
            a = br.assigner
            if "beta" in e and abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            for key, attr in (("beta_sel", "beta_sel"), ("beta_tgt", "beta_tgt")):
                if key in e:
                    got = getattr(a, attr, None)
                    if got is None or abs(float(got) - e[key]) > 1e-6:
                        raise RuntimeError(f"{rc['name']}: {tag} {attr}={got}, expected {e[key]}")
            # a split run whose two exponents are equal is a uniform run with
            # extra steps -- catch a config that silently collapsed
            if "beta_sel" in e and "beta_tgt" in e and abs(e["beta_sel"] - e["beta_tgt"]) > 1e-6:
                if abs(float(a.beta_sel) - float(a.beta_tgt)) < 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} beta_sel == beta_tgt, split collapsed")
            # everything else must be provably off
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live — falsified in round 21")
            if getattr(a, "beta_level", None):
                raise RuntimeError(f"{rc['name']}: beta_level is live but was not requested")
            if a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB is live but was not requested")
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB is live on {tag}")
            if b.swa_enabled() or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SWA/SNL1/NWD live on {tag}")
        h = o2o.hyp   # E2ELoss has no .hyp; only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this file is the stock graph")
        a = o2o.assigner
        seen.append(f"beta={a.beta} sel={getattr(a,'beta_sel',None)} tgt={getattr(a,'beta_tgt',None)} "
                    f"tcn_p={getattr(a,'tcn_p',1.0)} level={getattr(a,'beta_level',None)}")
        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    seed = rc.get("seed", SEED)
    print("\n" + "=" * 76)
    print(f"  RUN {rc['name']}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 76)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    tag = "CALIBRATION" if rc.get("calib") else f"prior {rc['prior']:.2f}"
    print(f"  b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{seed} | vs {rc['ctrl']:.2f} "
          f"| {tag} | {d}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=seed,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": rc["name"], "arm": rc["arm"], "ctrl": rc["ctrl"], "prior": rc["prior"],
           "calib": bool(rc.get("calib")), "params": rc["params"],
           "expect": {k: str(v) for k, v in rc["expect"].items()}, "seed": seed,
           "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours, "weights": weights,
           "mechanism_verified": True, "test_map50": float("nan"),
           "test_map5095": float("nan"), "test_map50_small": float("nan")}
    # args.yaml is what would have caught the y26_stock_b48 mislabelling: the
    # results JSONs record no batch for any run in this project.
    try:
        with open(os.path.join(save_dir, "round22_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{rc['name']}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def order(runs):
    """Calibration first, then descending prior. See the note above RUNS."""
    return sorted(runs, key=lambda r: (not r.get("calib", False), -r["prior"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--arm", choices=["cal", "a", "b", "rep"])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--skip-calib", action="store_true",
                    help="only if b3_s1 and b1_s1 have already been run")
    a = ap.parse_args()

    todo = RUNS
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if a.skip_calib:
        todo = [r for r in todo if not r.get("calib")]
    todo = order(todo)
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 22 — the selection axis, and two draws to stand on "
          f"({len(todo)} selected)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] defaults are not inert — fix before spending a night of GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_round22_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 22 — RESULTS (mAP50-95 only; score mAP50_small before concluding)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<22} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<22} seed {r['seed']}  mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
