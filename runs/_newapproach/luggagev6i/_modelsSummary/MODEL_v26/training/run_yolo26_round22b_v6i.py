#!/usr/bin/env python3
r"""
YOLO26 ROUND 22B — the (beta_sel, beta_tgt) grid, and the diagonal extended

Ten runs, ~9.0 GPU-h, seed 0 throughout, b82, stock yolo26s.pt, 640, 70 ep.
Matched to the baseline in batch and graph. PRIMARY METRIC IS mAP50_small.

    REFERENCE, 3 seeds of y26_identity   small 77.62   mAP50 80.40   m5095 55.30
    pooled sd (df=7)                     small  0.346  m5095 0.374
    single run vs the 3-seed mean        SE = 0.346*sqrt(1+1/3) = 0.400
    single run vs a SINGLE control draw  SE = 0.346*sqrt(2)     = 0.489

    vs the 3-seed reference:  2 SE = 0.80 suggestive | 3 SE = 1.20 an effect
    vs another single run  :  2 SE = 0.98 | 3 SE = 1.47

NO SEED RUNS THIS ROUND, by request. State the cost plainly: every control
below (B3_S50 = 79.14, B1_S50 = 79.30) is ONE DRAW, and b3 sits ~+0.34 above
what beta 2 / 2.5 / 4 interpolate to at beta 3. A cell landing at 78.9 prints
as -0.24 against b3 and may in fact be level. READ EVERY DELTA AGAINST THE
CURVE, NOT AGAINST THE POINT. This project has already lost information to
single draws twice — the seed-0 identity that biased every hardcoded constant
in rounds 18/19/20, and b3's own position above its curve.


=============================================================================
THE GRID, AS ROUND 21 LEFT IT
=============================================================================
Round 21's runs read as a (beta_sel, beta_tgt) surface, not a list. mAP50_small:

                            target beta
                    6        3      1.5       1
        sel 6    77.62    77.21       --      --
        sel 3    78.16    79.14    78.76   78.53
        sel 1       --       ??       --   79.30

Cells: sel6/tgt6 = identity. sel6/tgt3 = bsel6_btgt3. sel3/tgt6 = a1_b6 (ratio
0.167, the same SELECTION as b3, targets built at beta 6). sel3/tgt3 = b3.
sel3/tgt1.5 and sel3/tgt1 = the round-21 split runs. sel1/tgt1 = b1.

CAVEAT ON THE MAPPING. a1_b6 reaches its cell through alpha = 1.0, so its
targets are s^1 * u^6 rather than s^0.5 * u^6. It is the best reading of a run
that already exists, not an exact split cell. Every cell in THIS file is
reached with split() at alpha = 0.5, so the grid becomes exact from here on.

WHAT THAT REFRAMES. y26_b1 posted 79.30, the campaign maximum, off a 0.10
prior. It is NOT evidence that beta_tgt = 1 is good: the same round measured
beta_tgt = 1 at fixed selection and got 78.53, the WORST target setting tested.
b1 won on the SELECTION side and dragged a known-bad target along with it.
Uniform beta = 1 is two changes at once and one of them is wrong.


=============================================================================
THE DIAGONAL IS WINNING — the reason runs 4 and 7 exist
=============================================================================
In every row where a diagonal cell (beta_sel = beta_tgt, i.e. plain uniform
beta) and an off-diagonal cell are both measured, the DIAGONAL wins:

    sel 3    diagonal 79.14 (tgt 3)   off-diagonal 78.16 / 78.76 / 78.53
    sel 6    diagonal 77.62 (tgt 6)   off-diagonal 77.21 (tgt 3)
    sel 1    diagonal 79.30 (tgt 1)   off-diagonal: this round fills them

Two rows, two wins for the diagonal. If that holds, THE SPLIT BUYS NOTHING and
the productive direction is not sideways into the grid but further down the
diagonal, where round 21 stopped at beta = 1. Run 4 (uniform beta = 0.5) is
that extension; run 7 is the low corner where both optima would meet if they
drift downward together rather than tracking the diagonal exactly.

Runs 3 and 4 are a matched pair — same selection, off-diagonal vs diagonal
target — which is the sel = 0.5 row's version of the test round 21 ran at
sel = 3.


=============================================================================
WHAT IS DELIBERATELY ABSENT
=============================================================================
THE ALPHA PLANE (a1_b2, a1_b1, a075_b1, a025_b1). Superseded. With alpha fixed
at 0.5, beta_sel in {0.5, 1, 2, 3, 6} spans ratio 0.083 to 1.00 — the same span
those runs would have covered. a1_b1 and bsel05_* select identically. Before
the round-21 patch, moving alpha was the only way to change selection without
changing the target; now it reintroduces the confound the patch removed, since
alpha scales the target magnitude too.

UNIFORM beta = 0. Changes selection AND target at once. The target degenerates
at 0 (it stops depending on IoU); selection is a top-k ranking and does not.
bsel05_btgt3 is the clean version of that question.

STACKED MECHANISMS (SBB, SWA, o2m_final on a beta base). All three would stack
onto tgt = 1, which round 21 measured at fixed selection as 78.53 — the worst
of four. You cannot usefully stack onto a cell you have not measured. They
belong in round 23, on whichever cell wins here.

TCN. Round 21 wrote its own falsification test: "if R50_small rises and
P50_small holds it worked; if both fall the ceiling was load-bearing." Both
fell, on both bases (b3: R50_small -1.00, P50_small -1.59). Closed.


=============================================================================
STATED IN ADVANCE
=============================================================================
If run 1 (bsel1_btgt3) lands BELOW b1's 79.30 while run 4 (uniform beta = 0.5)
lands ABOVE it, the split is dead and the entire finding is ONE hyperparameter.
That is the cleanest possible outcome for the paper and it is currently the way
the evidence leans. It should be read as a result, not a disappointment.

If run 1 lands at 79.1-79.3 it has NOT won — it has tied b1 and b3, meaning
selection and target saturate together and the beta family is finished at
roughly +1.7 over identity.


=============================================================================
EXECUTION ORDER
=============================================================================
prior = P(clearing +0.35 on mAP50_small vs its own control). Enforced by
sorting on prior in main(), not transcribed, so this table cannot drift.

    1  y26_bsel1_btgt3    0.35  best measured selection x best measured target
    2  y26_bsel2_btgt3    0.30  turns the tgt=3 column into a 5-point sweep
    3  y26_bsel05_btgt3   0.25  below the bound the target axis has
    4  y26_b05            0.22  THE DIAGONAL EXTENDED — needs no patch
    5  y26_bsel05_btgt1   0.20  single-variable comparison against b1
    6  y26_bsel1_btgt2    0.18  sel=1 row, between the two best targets
    7  y26_bsel05_btgt2   0.18  the low corner
    8  y26_bsel1_btgt15   0.15  sel=1 row, lower bracket
    9  y26_bsel2_btgt1    0.15  second row, does the tgt curve keep its shape
   10  y26_bsel1_btgt6    0.10  sel=1 row, upper bracket

After this round the tgt=3 column is 5 points (sel 6/3/2/1/0.5) and the sel=1
row is 5 points (tgt 6/3/2/1.5/1), so both axes have a known shape rather than
three dots and an extrapolation.

THESE ARE AN ORDERING, NOT FORECASTS. Directional predictions in this campaign
are 0-for-12 and every miss was optimistic. What is different about run 1 is
that it is not a mechanism story — it is the empty cell of a grid whose other
cells are filled and monotone. That is a weaker claim than any of the twelve
that failed, which is the point.

    Usage:
        python run_yolo26_round22b_v6i.py                  # all ten, prior order
        python run_yolo26_round22b_v6i.py --preflight      # prove inertness only
        python run_yolo26_round22b_v6i.py --arm grid       # grid | diag
        python run_yolo26_round22b_v6i.py y26_bsel1_btgt3  # one by name

    Run 4 needs no patch (plain tal_beta). The other nine need
    patch_round21_v6i.py; if it is not installed they are SKIPPED with a
    message, not silently run as stock.
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
PROJECT_DIR = "runs_yolo26_round22b_v6i"
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
    # ------------------------------------------------------------- THE GRID
    {"name": "y26_bsel1_btgt3", "prior": 0.35, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(1.0, 3.0), "expect": {"beta": 1.0, "beta_sel": 1.0, "beta_tgt": 3.0},
     "label": "sel 1 x tgt 3 — best measured selection, best measured target",
     "why": "The empty cell. b1 (79.30) and b3 (79.14) tie within error for "
            "OPPOSITE reasons: b1 has the better selection and a target measured "
            "at 78.53, b3 has the better target and a weaker selection. Only this "
            "cell combines them, and only it separates the two."},

    {"name": "y26_bsel2_btgt3", "prior": 0.30, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": split(2.0, 3.0), "expect": {"beta": 2.0, "beta_sel": 2.0, "beta_tgt": 3.0},
     "label": "sel 2 x tgt 3 — the tgt=3 column becomes a curve",
     "why": "With runs 1 and 3 this makes the tgt=3 column a 5-point selection "
            "sweep (6, 3, 2, 1, 0.5). Selection has no measured turnover "
            "anywhere; either it appears inside this range and the axis is "
            "finished, or it does not and the campaign has an open axis."},

    {"name": "y26_bsel05_btgt3", "prior": 0.25, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(0.5, 3.0), "expect": {"beta": 0.5, "beta_sel": 0.5, "beta_tgt": 3.0},
     "label": "sel 0.5 x tgt 3 — below the bound the target axis has",
     "why": "beta_sel is NOT bounded at 1 the way beta_tgt is. At beta_tgt = 0 "
            "the target stops depending on IoU; selection is top-k on a ranking, "
            "so 0.5 is an ordinary point on it, not a degenerate one. Pairs with "
            "run 4 as off-diagonal vs diagonal at the same selection."},

    {"name": "y26_bsel05_btgt1", "prior": 0.20, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(0.5, 1.0), "expect": {"beta": 0.5, "beta_sel": 0.5, "beta_tgt": 1.0},
     "label": "sel 0.5 x tgt 1 — one step past b1 on selection alone",
     "why": "b1 IS sel1/tgt1 = 79.30. This changes ONE variable against the "
            "campaign maximum: selection 1 -> 0.5, target held. If the selection "
            "axis is monotone as every measurement so far suggests, this beats "
            "b1; if it does not, the selection axis has a floor between 1 and "
            "0.5 and run 3 will show where."},

    {"name": "y26_bsel1_btgt2", "prior": 0.18, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(1.0, 2.0), "expect": {"beta": 1.0, "beta_sel": 1.0, "beta_tgt": 2.0},
     "label": "sel 1 x tgt 2 — between the two best targets",
     "why": "The sel=1 row jumps tgt 1 -> 1.5 -> 3 -> 6. At sel=3 the target "
            "peaked at 3 and fell away on both sides; if that shape survives at "
            "sel=1 the peak sits between 1.5 and 3 and this is where it lands."},

    {"name": "y26_bsel05_btgt2", "prior": 0.18, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(0.5, 2.0), "expect": {"beta": 0.5, "beta_sel": 0.5, "beta_tgt": 2.0},
     "label": "sel 0.5 x tgt 2 — the low corner",
     "why": "If both optima drift downward together rather than tracking the "
            "diagonal exactly, this is where the maximum sits. Bracketed by runs "
            "3, 4 and 5, so it is interpolation rather than a shot in the dark."},

    {"name": "y26_bsel1_btgt15", "prior": 0.15, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(1.0, 1.5), "expect": {"beta": 1.0, "beta_sel": 1.0, "beta_tgt": 1.5},
     "label": "sel 1 x tgt 1.5 — sel=1 row, lower bracket",
     "why": "Because the target effect CHANGES SIGN with selection (6->3 on the "
            "target is -0.41 at sel 6 and +0.98 at sel 3), the target optimum has "
            "no reason to stay at 3 when selection moves to 1. Bracketing the "
            "sel=1 row is what turns a winning cell into a surface with a shape."},

    {"name": "y26_bsel2_btgt1", "prior": 0.15, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": split(2.0, 1.0), "expect": {"beta": 2.0, "beta_sel": 2.0, "beta_tgt": 1.0},
     "label": "sel 2 x tgt 1 — a second row, for shape",
     "why": "The sel=3 row peaks at tgt=3 and the sel=1 row is being measured. A "
            "third partial row says whether the target curve keeps its shape as "
            "selection moves or whether the interaction is confined to sel=6, "
            "which is what bsel6_btgt3 proved exists but did not characterise."},

    {"name": "y26_bsel1_btgt6", "prior": 0.10, "arm": "grid", "calib": False,
     "seed": 0, "needs": "tal_beta_sel", "ctrl": B1_S50,
     "params": split(1.0, 6.0), "expect": {"beta": 1.0, "beta_sel": 1.0, "beta_tgt": 6.0},
     "label": "sel 1 x tgt 6 — sel=1 row, upper bracket",
     "why": "The stock target at the best selection. Expected to lose, and the "
            "loss is the point: it measures how much of b1's win was the target "
            "rather than the selection, which is the decomposition round 21's "
            "arm B failed to obtain by inference."},

    # -------------------------------------------------------- THE DIAGONAL
    {"name": "y26_b05", "prior": 0.22, "arm": "diag", "calib": False,
     "seed": 0, "needs": None, "ctrl": B1_S50,
     "params": cfg(tal_beta=0.5), "expect": {"beta": 0.5},  # uniform: beta_sel/beta_tgt stay None
     "label": "uniform beta 0.5 — the diagonal extended, NO PATCH NEEDED",
     "why": "Every row where both are measured, the diagonal beats its own "
            "off-diagonal cells: sel3 79.14 vs 78.16/78.76/78.53, sel6 77.62 vs "
            "77.21. If that holds the split buys nothing and the productive "
            "direction is further down the diagonal, where round 21 stopped at "
            "beta = 1. This also needs no patch, so it doubles as the canary: if "
            "it behaves and the nine patched cells do not, the patch is the "
            "difference."},
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
        with open(os.path.join(save_dir, "round22b_params.json"), "w") as f:
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
    ap.add_argument("--arm", choices=["grid", "diag"])
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
    print(f"  YOLO26 ROUND 22B — the selection axis, and two draws to stand on "
          f"({len(todo)} selected)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] defaults are not inert — fix before spending a night of GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_round22b_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 22B — RESULTS (mAP50-95 only; score mAP50_small before concluding)")
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
