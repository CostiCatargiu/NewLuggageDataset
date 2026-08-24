#!/usr/bin/env python3
r"""
YOLO26 ROUND 24 — the OTHER exponent on the target term, and one mechanism audit

Six runs, ~5.4 GPU-h, b82, stock yolo26s.pt, 640, 70 ep. NO PATCH REQUIRED —
tal_alpha, tal_beta, tal_beta_small, sbb_q and sbb_invert are all already in
default.yaml, so this can start tonight.

PRIMARY METRIC IS mAP50_small.

    REFERENCE, 3 seeds of y26_identity   small 77.62   mAP50 80.40   m5095 55.30
    POOLED sd (df=11, UPDATED IN R23)    small  0.377  m5095 0.374

    THE FLOOR MOVED. Round 23's four repeat pairs contributed df=4 at sd 0.427,
    which pooled with the old df=7 at 0.346 gives 0.377. Every sigma quoted
    before round 23 was ~8% optimistic. Divisors below use the new number.

      one run vs the 3-seed mean   SE = 0.377*sqrt(1+1/3) = 0.435
      one run vs another one run   SE = 0.377*sqrt(2)     = 0.533
      n=2 mean vs the 3-seed mean  SE = 0.377*sqrt(1/2+1/3) = 0.345


=============================================================================
READ THIS BEFORE INTERPRETING ANY NUMBER IN THIS ROUND
=============================================================================
ROUND 23 MEASURED THE WINNER'S CURSE AND IT IS LARGE.

    config              seed 0   seed 1    mean    vs 77.62
    y26_b1               79.30    78.59   78.95      +1.33
    y26_bsel05_btgt2     79.13    78.51   78.82      +1.20
    y26_b3               79.14    78.40   78.77      +1.15
    y26_bsel1_btgt6      78.81    78.66   78.74      +1.12

FOUR FOR FOUR BELOW THEIR SEED-0 DRAW, mean regression -0.55. That is not bad
luck at 1-in-16 — it is selection. Each of those four was chosen BECAUSE it was
the maximum of its round, so repeating it regresses. The identity baseline,
which was never selected for anything, went the other way (seed 0 low by 0.32).

THE CONSEQUENCE FOR THIS FILE: every ctrl below is an n=1 or n=2 number, and
the n=1 ones are drawn from runs that were noticed. A run landing 0.3 under its
ctrl has probably tied it. Do not retire anything in this round on a single
draw, and do not promote anything either.

The corollary that actually matters: y26_b1's headline is +1.33, not +1.67, and
the four best configs now span 0.21 — indistinguishable. The paper reports the
FAMILY, not a winner. Nothing in this round changes that unless it clears
+1.33 by more than 0.7, which nothing in this campaign ever has.


=============================================================================
ARM A — ALPHA, THE UNTESTED HALF OF THE TERM THAT CARRIES EVERYTHING (4 runs)
=============================================================================
y26_b0 (beta = 0, round 23) scored 78.38 on small. Against the REGRESSED means
above (78.74-78.95) that is a gap of 0.36-0.57, i.e. 0.8-1.2 SE at the run-vs-
mean SE of 0.462. NOT DISTINGUISHABLE.

At beta = 0 the alignment metric is score^alpha alone: assignment IGNORES IoU
ENTIRELY. Removing IoU from selection costs nothing measurable.

That is corroborated mechanically by probe_topk_binding_v2: schedule-weighted
over the real 70-epoch run (60 mosaic + 10 clean), ~65% of small-object
assignments hold FEWER anchor centres than topk, so every candidate is already
positive and the ranking cannot change the assignment. Two independent routes,
one empirical and one geometric, to the same conclusion:

    TASK-ALIGNED *SELECTION* IS INOPERATIVE ON THIS DATA.
    100% OF THE BETA EFFECT RUNS THROUGH TARGET MAGNITUDES.

The target magnitude is score^alpha * IoU^beta, normalised by the GT's own best
(tal.py:224-226). You have mapped beta across [0, 6] in eleven configurations.
ALPHA HAS NEVER BEEN TESTED AT A GOOD BETA. Every alpha run in 150 sits at
beta = 6 or is tangled with SCB/SWA:

    a1_b6   a1_b6_o2o   alpha075   a025_b4s2   a075_scb3_sbb50   swa_a0*

beta softens the target with respect to IoU; alpha softens it with respect to
classification confidence. One axis of a two-axis term is mapped and the other
is empty at every useful operating point.

This is the ONLY remaining direction in the loss with a mechanism argument
behind it rather than a hyperparameter story. It is also cheap: no patch.

    ratio note, now demoted. Selection is top-k on score^alpha * IoU^beta and
    therefore depends only on alpha/beta. That WAS the reason to care about the
    ratio. b0 says selection does not matter, so ignore the ratio and read
    these as four points on the TARGET surface.


=============================================================================
ARM B — AUDIT THE ONE MECHANISM PATTERN THAT SURVIVED (2 runs)
=============================================================================
The beta family buys small-object mAP50 and PAYS FOR IT on mAP50-95. Seeded,
all four round-23 pairs:

    y26_b1  -0.49    y26_bsel05_btgt2  -0.59
    y26_b3  -0.26    y26_bsel1_btgt6   -0.44        mean -0.44

Three configs carrying NEW CODE do not show that cost, at n=1:

    y26_b4s2_sbb50    SBB          small +1.13   m5095 +0.28
    y26_bsel3_btgt15  beta split   small +1.14   m5095 +0.18
    y26_scb_b4s2      SCB          small +1.03   m5095 +0.19

Three independent mechanisms, same signature: most of the small gain, none of
the aggregate cost. If that holds it is a METHOD result — something beta cannot
do by itself — and it is the only such candidate left in the campaign.

WHAT THIS ARM CAN AND CANNOT ESTABLISH. +0.28 on mAP50-95 is 0.75 SE against a
0.374 floor and is NOT detectable at any n you can afford. The testable claim is
the CONTRAST: does SBB avoid the -0.44 that beta pays? That gap is 0.72. With
seed 0 already in hand, seeds 1 and 2 give n=3, and n=3 against beta's n=2 has
SE = 0.374*sqrt(1/3+1/2) = 0.342, so 0.72 is ~2.1 SE. Suggestive at best.
Stated in advance so nobody reads a 2-sigma contrast as a result.

Two seeds and not one, precisely because of this: at n=2 the same contrast is
1.9 SE and the round would answer nothing.

HONEST PRIOR. Round 19 put ten mechanisms on beta=3 and ALL TEN LOST. Combos
are 0-for-10 in this campaign. The reasons this differs — those were judged on
mAP50_small while this claim is about mAP50-95, and b4s2_sbb50 was mid-pack
rather than a round winner so it carries less winner's-curse exposure — are
arguments, not evidence.

    y26_b4s2_sbb50 = SCB(beta 4.0, beta_small 2.0, ref 64) + SBB(q 0.5, inverted)
    seed 0:  small 78.75   mAP50 81.58   m5095 55.58


=============================================================================
WHAT IS DELIBERATELY NOT HERE
=============================================================================
LB-TAL / per-level topk. CLOSED — it was already run on v26, six times, and
every one is BELOW the 77.62 baseline on small: lb_uniform 77.40, 3lvl_head64
77.26, lb_coarse244 76.89, lb_p3_3 76.79, lb_p4wide 76.72. The v12 P3=4 peak
(+0.80 mAP50-95) did not transfer, consistent with LB-TAL reaching only
one2many, whose blend weight decays 0.8 -> 0.1.

topk itself. Raising it cannot help small (they take 7.73 of a budget of 10);
lowering it is a medium/large intervention, since those sit pinned at
take~10.00. And b0 killed the interaction hypothesis that motivated it: there
is no selection term left to re-enable.

EIoU (one run, 76.63), quality_gate (lives inside LB-TAL), SWA and branch
balance on beta (each 0-for-many, and neither has a mechanism story that
survived b0).


=============================================================================
EXECUTION ORDER
=============================================================================
INTERLEAVED, so a queue that dies at 3am keeps both stories alive rather than
all of one and none of the other.

    1  y26_a1_b1           0.30   the new axis, best cell
    2  y26_b4s2_sbb50_s1   ----   arm B reaches n=2
    3  y26_a1_b2           0.28   second alpha point, different beta
    4  y26_b4s2_sbb50_s2   ----   arm B reaches n=3, which is the point
    5  y26_a075_b1         0.22   fills the alpha surface
    6  y26_a025_b1         0.15   the low end

Priors are P(clearing +0.35 on mAP50_small vs its own control). They are an
ORDERING, not forecasts: directional predictions in this campaign are 0-for-12
and every miss was optimistic.

    Usage:
        python run_yolo26_round24_v6i.py                 # all six, in order
        python run_yolo26_round24_v6i.py --preflight     # print the plan only
        python run_yolo26_round24_v6i.py --arm a         # a | b
        python run_yolo26_round24_v6i.py y26_a1_b1       # one by name
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
PROJECT_DIR = "runs_yolo26_round24_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False            # y26_p2k2_hi was lost to exist_ok=True

# 3-seed means. Never a seed-0 draw.
REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30
SD_S50, SD_5095 = 0.377, 0.374        # small floor UPDATED in round 23 (df=11)

# n=2 pair means from round 23. These are the honest controls.
B1_S50 = 78.95        # y26_b1,  n=2   (seed-0 draw was 79.30 — regressed 0.35)
B3_S50 = 78.77        # y26_b3,  n=2
# n=1, seed 0 only. Arm B exists to turn this into n=3.
B4S2_SBB_S50 = 78.75
B4S2_SBB_5095 = 55.58

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

_B4S2 = dict(tal_beta=4.0, tal_beta_small=2.0, tal_beta_ref_px=64.0)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def ab(alpha, beta):
    """A point on the TARGET surface score^alpha * IoU^beta.

    Not a point on a selection surface: y26_b0 showed selection is inoperative
    here, so the alpha/beta RATIO — which is all selection depends on — is not
    what these runs vary in any way that matters.
    """
    return cfg(tal_alpha=alpha, tal_beta=beta)


RUNS = [
    # ----------------------------------------------------- ARM A — the alpha axis
    {"name": "y26_a1_b1", "prior": 0.30, "arm": "a", "order": 1, "seed": 0,
     "repeat_of": None, "ctrl": B1_S50,
     "params": ab(1.0, 1.0), "expect": {"alpha": 1.0, "beta": 1.0},
     "label": "alpha 1.0 / beta 1.0 — double the target's confidence exponent at the best beta",
     "why": "b1 is the campaign maximum and sits at the STOCK alpha of 0.5. "
            "Raising alpha sharpens the target with respect to classification "
            "confidence, which is the axis of the target term that has never "
            "been moved at a beta anyone would use. If the target surface has "
            "any structure off the beta line, this is the cell that shows it."},

    {"name": "y26_b4s2_sbb50_s1", "prior": 0.00, "arm": "b", "order": 2, "seed": 1,
     "repeat_of": "y26_b4s2_sbb50", "ctrl": B4S2_SBB_S50,
     "params": cfg(**_B4S2, sbb_q=0.5, sbb_invert=True),
     "expect": {"beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.5},
     "label": "SCB b4/s2 + SBB 0.5 inv, seed 1 — audit the no-cost signature",
     "why": "The only mechanism pattern that survived round 23: +1.13 small "
            "with mAP50-95 at +0.28 while the whole beta family pays -0.44. "
            "n=1. Prior is 0.00 by construction — a seed repeat has no expected "
            "gain, and after round 23 the expectation is that it REGRESSES."},

    {"name": "y26_a1_b2", "prior": 0.28, "arm": "a", "order": 3, "seed": 0,
     "repeat_of": None, "ctrl": B1_S50,
     "params": ab(1.0, 2.0), "expect": {"alpha": 1.0, "beta": 2.0},
     "label": "alpha 1.0 / beta 2.0 — the same alpha one step along the beta plateau",
     "why": "Two cells at the same alpha and different beta say whether the "
            "alpha effect is CONSTANT across the beta plateau or interacts with "
            "it. One cell cannot distinguish 'alpha helps' from 'alpha helps at "
            "beta=1'. The plateau is flat in beta, so any interaction found "
            "here is information about the target term's shape."},

    {"name": "y26_b4s2_sbb50_s2", "prior": 0.00, "arm": "b", "order": 4, "seed": 2,
     "repeat_of": "y26_b4s2_sbb50", "ctrl": B4S2_SBB_S50,
     "params": cfg(**_B4S2, sbb_q=0.5, sbb_invert=True),
     "expect": {"beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.5},
     "label": "SCB b4/s2 + SBB 0.5 inv, seed 2 — n=3, which is the whole point",
     "why": "At n=2 the mAP50-95 contrast against the beta family is 1.9 SE and "
            "the arm answers nothing. n=3 takes it to ~2.1. That is still only "
            "suggestive, and it is the honest ceiling for a 0.72 effect on a "
            "metric with a 0.374 floor. Run it or drop arm B entirely; n=2 is "
            "the one option that wastes the GPU hour."},

    {"name": "y26_a075_b1", "prior": 0.22, "arm": "a", "order": 5, "seed": 0,
     "repeat_of": None, "ctrl": B1_S50,
     "params": ab(0.75, 1.0), "expect": {"alpha": 0.75, "beta": 1.0},
     "label": "alpha 0.75 / beta 1.0 — between stock and run 1",
     "why": "Brackets run 1 from below. If alpha is monotone over [0.5, 1.0] "
            "this interpolates and confirms; if run 1 wins and this does not, "
            "the alpha response is threshold-like exactly as beta's turned out "
            "to be, and that parallel is worth a sentence in the paper."},

    {"name": "y26_a025_b1", "prior": 0.15, "arm": "a", "order": 6, "seed": 0,
     "repeat_of": None, "ctrl": B1_S50,
     "params": ab(0.25, 1.0), "expect": {"alpha": 0.25, "beta": 1.0},
     "label": "alpha 0.25 / beta 1.0 — the low end, and the curve's other edge",
     "why": "Closes the alpha curve downward the way b0 closed beta's. At "
            "alpha -> 0 the target stops depending on classification confidence "
            "and becomes IoU^beta alone. Whichever way it lands, nobody can ask "
            "whether the other direction was tried."},
]


def order(runs):
    """Interleaved by the explicit 'order' key — see the header. Deliberately
    NOT descending prior: arm B has no prior, and burying it behind four alpha
    runs would mean a queue that dies early answers neither question."""
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """Prove the CONSUMER reads each key, and that defaults are bit-identical.

    Rounds 4-6 passed a preflight that only checked default.yaml accepted the
    key and tal.py had the class. Ten runs were lost because the file that had
    to consume it ignored the flag.
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
    checks = {
        "default.yaml accepts tal_alpha": "tal_alpha" in DEFAULT_CFG_DICT,
        "default.yaml accepts tal_beta": "tal_beta" in DEFAULT_CFG_DICT,
        "default.yaml accepts tal_beta_small": "tal_beta_small" in DEFAULT_CFG_DICT,
        "default.yaml accepts sbb_q": "sbb_q" in DEFAULT_CFG_DICT,
        "default.yaml accepts sbb_invert": "sbb_invert" in DEFAULT_CFG_DICT,
        "assigner stores alpha": "self.alpha" in src_init,
        "assigner has beta_small (SCB)": "beta_small" in src_init,
    }
    for k, v in checks.items():
        print(f"  {k:<42} {v}")
        ok &= v

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
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        tag = f"p={r['prior']:.2f}" if r["prior"] > 0 else "SEED"
        print(f"  RUN  {r['name']:<20} arm {r['arm']}  {tag:<6} seed {r['seed']}  "
              f"vs {r['ctrl']:.2f}  {d}")
    print(f"\n  {len(todo)} runs, ~{0.9 * len(todo):.1f} GPU-h   (no patch required)")
    return ok, {r["name"] for r in todo}


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Assert at epoch 2 that the requested mechanism is LIVE and nothing else is.

    Expectation-DRIVEN, unlike round 23's blanket 'nothing but beta'. Arm B
    legitimately runs SCB and SBB, so a blanket guard would have to be disabled
    for it — and a disabled guard is how rounds 4-6 lost ten runs. Here each key
    is either asserted to a value or asserted OFF, per run, with no exceptions.
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
            if abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            # alpha: asserted when requested, else must be the stock 0.5
            want_a = e.get("alpha", 0.5)
            if abs(float(a.alpha) - want_a) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected {want_a}")
            # SCB
            bs = getattr(a, "beta_small", None)
            if "beta_small" in e:
                want, ref = e["beta_small"]
                if bs is None or abs(float(bs) - want) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} beta_small={bs}, expected {want}")
                if abs(float(getattr(a, "beta_ref_px", 0.0)) - ref) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} beta_ref_px != {ref}")
            elif bs is not None:
                raise RuntimeError(f"{rc['name']}: {tag} SCB is live (beta_small={bs}) "
                                   f"but was not requested")
            # everything on the assigner that is not this run's mechanism
            for attr, off in (("beta_sel", None), ("beta_tgt", None),
                              ("beta_level", None), ("tcn_p", 1.0)):
                got = getattr(a, attr, off)
                if (got is not None) if off is None else (float(got) != off):
                    raise RuntimeError(f"{rc['name']}: {tag} {attr}={got} but was not requested")
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            # SBB
            b = br.bbox_loss
            want_q = e.get("sbb", 0.0)
            if abs(float(getattr(b, "sbb_q", 0.0)) - want_q) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {want_q}")
            if b.swa_enabled() or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SWA/SNL1/NWD live on {tag}")
        if e.get("sbb", 0.0) > 0.0:
            s_m, s_o = float(o2m.bbox_loss.sbb_sign), float(o2o.bbox_loss.sbb_sign)
            if s_m * s_o >= 0:
                raise RuntimeError(f"{rc['name']}: SBB signs o2m={s_m:+.0f} o2o={s_o:+.0f} "
                                   f"— they must be OPPOSITE")
            if s_o >= 0:
                raise RuntimeError(f"{rc['name']}: one2one sbb_sign={s_o:+.0f}; inverted SBB "
                                   f"requires one2one to lean SMALL (negative)")
            seen.append(f"SBB q={want_q} o2m={s_m:+.0f}(large) o2o={s_o:+.0f}(small)")
        h = o2o.hyp   # E2ELoss has no .hyp; only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set — LB-TAL is CLOSED on v26 "
                               f"(6 runs, all below baseline on small)")
        a = o2o.assigner
        seen.insert(0, f"alpha={a.alpha} beta={a.beta} beta_small={getattr(a,'beta_small',None)}")
        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        if rc["repeat_of"]:
            print(f"  [guard] certified as a seed-{rc['seed']} repeat of {rc['repeat_of']} "
                  f"(seed 0 small = {B4S2_SBB_S50:.2f}, m5095 = {B4S2_SBB_5095:.2f})")
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
    tag = f"prior {rc['prior']:.2f}" if rc["prior"] > 0 else "SEED REPEAT (no expected gain)"
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
           "repeat_of": rc["repeat_of"], "params": rc["params"],
           "expect": {k: str(v) for k, v in rc["expect"].items()}, "seed": seed,
           "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours, "weights": weights,
           "mechanism_verified": True, "test_map50": float("nan"),
           "test_map5095": float("nan"), "test_map50_small": float("nan")}
    # args.yaml is what would have caught the y26_stock_b48 mislabelling: the
    # results JSONs record no batch for any run in this project.
    try:
        with open(os.path.join(save_dir, "round24_params.json"), "w") as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--arm", choices=["a", "b"])
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    todo = RUNS
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    todo = order(todo)
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 24 — alpha on the target term, and one mechanism audit "
          f"({len(todo)} selected)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] preflight failed — fix before spending a night of GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_round24_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 24 — RESULTS (mAP50-95 only; score mAP50_small before concluding)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<20} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<20} seed {r['seed']}  mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print("  1. export the test JSON, then run analyze_runs_v6i.py")
    print("  2. the floor is 0.377 on small (df=11) — update SD_S50 there if it")
    print("     still says 0.346, or every sigma prints ~8% high")
    print("  3. arm A: read against B1_S50 = 78.95, the n=2 PAIR MEAN. Reading")
    print("     against b1's seed-0 draw of 79.30 would make every alpha run look")
    print("     0.35 worse than it is — that is the winner's curse, measured.")
    print("  4. arm B: the number to look at is mAP50-95, not small. The question")
    print("     is whether it avoids the beta family's -0.44, and the answer is")
    print("     suggestive at best by construction.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
