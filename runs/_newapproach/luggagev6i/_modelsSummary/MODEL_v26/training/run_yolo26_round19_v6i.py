#!/usr/bin/env python3
r"""
YOLO26 ROUND 19 — THE OVERNIGHT (10 runs, loss only, stock yolo26s, b82)

~9.3 GPU-h, seed 0. Supersedes the round 19/20 drafts.

BASE      y26_b3        tal_beta=3.0, everything else stock   81.31 / 79.14 small
BASELINE  y26_identity                                        80.18 / 77.30 small
sd on mAP50_small ~0.27 (identity and scb3_sbb50 seed pairs)


=============================================================================
WHAT ROUND 18 ESTABLISHED
=============================================================================
Sorting every run by the RATIO alpha/beta its SMALL objects receive gives one
unimodal curve. Selection ranks anchors by s^alpha * u^beta, which is a monotone
function of (s^(alpha/beta) * u), and both selection sites are pure top-k, so
WHICH anchors become positive depends only on that ratio -- never on the
magnitudes. (The one non-invariant path, the eps cutoff in
select_topk_candidates, is dead: get_pos_mask always passes topk_mask.)

    ratio   mAP50_small   run
    0.083      77.30      identity     alpha .5 / beta 6
    0.125      78.17      beta4        alpha .5 / beta 4
    0.167      79.14      b3           alpha .5 / beta 3     <-- PEAK, the base
    0.250      78.92      b2           alpha .5 / beta 2
    0.333      78.42      b3s15        (SCB, small sees 1.5)
    0.500      78.33      b4s1         (SCB, small sees 1.0)

SCB has no mechanism of its own -- it moved small objects toward the peak by a
roundabout route, helping from beta=6 (+0.48) and HURTING from beta=3 (-0.72).
One scalar replaced a family that took ~100 runs. beta=3 also reaches the
P2+DySample band (79.14 vs 79.16-79.49) with no extra parameter.

TWO NEW RESULTS FROM THE LAST TWO ROUND-18 RUNS
-----------------------------------------------
y26_a025_b4s2  79.76 / 77.24 -- the only sub-baseline run in the round. It is
scb_b4s2 with alpha halved, so every ratio halves and its selection matches
(alpha .5, beta 8, beta_small 4), off the left end of the curve. Direction
predicted. But its SMALL objects see ratio 0.25/2.0 = 0.125, identical to
beta4's, and it missed beta4's 78.17 by -0.93 (3.4 sd). The residual has a
mechanism: halving the exponents makes align/align_max its SQUARE ROOT, closer
to 1, so runner-up positives are trained to HIGHER scores and the
winner/runner-up gap NARROWS. In an NMS-free head that gap IS the duplicate
suppression -- exactly how SNT failed (-3.93 overall, -12.00 large). Same
signature here: large fell hardest, -3.09 vs scb_b4s2.

    => the COMMON SCALE of (alpha, beta) is a target-sharpness knob that leaves
       selection untouched. Runs 1 and 7 test it in the other direction.

y26_b4s2_sbb50  81.58 / 78.75 -- SBB against the identical config without it:
    small  +0.10 (0.4 sd, nothing)   medium +0.22   large +2.66 (80.95->83.61)
SBB is a LARGE-object mechanism, which is what invert=True asks for. Its
headline +0.87 lives in the 603-instance bucket at sd 2.06. Expect nothing from
the invert arm on mAP50_small; the FORWARD arm is the interesting one.


=============================================================================
THE TEN
=============================================================================
STAGE 1 -- ASSIGNMENT (tal.py): which anchors become positive
STAGE 2 -- WEIGHTING  (loss.py): how much each positive then counts
SCB died because it shared a stage AND a tensor with beta. Everything below is
either a different stage, a different branch, or selection-preserving.

  1  a1_b6        alpha 1.0 / beta 6.0        b3's exact positives, targets x2 sharper
  2  b3_sbb50     beta 3 + SBB q.5 invert     small mechanism + large mechanism
  3  b3_sbb50f    beta 3 + SBB q.5 forward    the untested arm, points SBB at small
  4  b3_swa       beta 3 + SWA 0.9->0.4       the two multi-run mechanisms, never combined
  5  b3_o2m       beta 3 o2m / 6 o2o          is beta an auxiliary-branch effect?
  6  b3_o2o       beta 6 o2m / 3 o2o          the complement; 5+6 bracket b3
  7  a1_b6_o2o    sharpen ONE2ONE only        same selection everywhere, gap widened
                                              only where duplicates are suppressed
  8  b25          beta 2.5 uniform            b3 and b2 differ by 0.22, inside noise
  9  b3_o2mf30    beta 3 + o2m 0.8 -> 0.3     keep the branch beta works in alive
 10  b2m_b4o      beta 2 o2m / 4 o2o          asymmetry uniform beta cannot express

Ordered most-informative first: if the queue dies at 3am you keep the runs that
decide something. 2+3 and 5+6 are contrasts and pay off whichever way they land.
1+7 test one new mechanism from two angles. 4, 8, 9, 10 are bets.

CALIBRATION: this campaign is 1-for-10 on combinations, and nine of nine
directional predictions have been falsified, every one optimistic. The
sharpness story rests on ONE confounded run (a025_b4s2 carries SCB and differs
at medium/large), so runs 1 and 7 are a hypothesis under test, not a plan.


=============================================================================
CODE THIS ROUND DEPENDS ON  (already applied in this tree)
=============================================================================
Runs 5, 6, 7, 10 need branch-scoped exponents. loss.py E2ELoss.__init__, after
the scb_branch block:

    for br, tag, keys in (
        (self.one2many, "one2many", (("alpha", "tal_alpha_o2m"), ("beta", "tal_beta_o2m"))),
        (self.one2one,  "one2one",  (("alpha", "tal_alpha_o2o"), ("beta", "tal_beta_o2o"))),
    ):
        for attr, key in keys:
            v = getattr(h, key, None)
            if v is not None:
                setattr(br.assigner, attr, float(v))

default.yaml gains tal_beta_o2m/o2o and tal_alpha_o2m/o2o, all null -> the loop
does not execute and the run is bit-identical to stock. tal.py reads self.alpha
and self.beta at call time in get_box_metrics, so no other file changes.

COPY loss.py AND default.yaml TO THE TRAINING BOX. Diff before overwriting -- the
archived training/ scripts are known to disagree with the live tree. Preflight
aborts if the keys exist but nothing reads them, which is the use_lbtal failure
that cost rounds 4-6 ten runs.

NUMERICS (checked, not assumed): overlaps carries gt_bboxes.dtype (fp32 out of
preprocess), so align_metric is fp32 even under AMP; only bbox_scores.pow(alpha)
runs in the prediction dtype and s**alpha cannot underflow for alpha <= 1.0. The
eps in the normalisation denominator is 1e-9 against an align_max of ~1e-4 at
(1.0, 6.0) -- five orders of margin.

    Usage:
        python run_yolo26_round19_v6i.py                 # all ten, in order
        python run_yolo26_round19_v6i.py y26_a1_b6       # one by name
        python run_yolo26_round19_v6i.py --from 5        # resume at run 5
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
MODEL_WEIGHTS = "yolo26s.pt"  # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_round19_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches b3/b2 and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

CTRL_B3_50 = 81.31   # y26_b3        THE BASE and the bar
CTRL_B3_S50 = 79.14  # y26_b3        mAP50_small
CTRL_ID_50 = 80.18   # y26_identity
CTRL_ID_S50 = 77.30
NOISE_S50 = 0.27  # sd on mAP50_small, from the identity and scb3_sbb50 seed pairs

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    tal_beta_o2m=None, tal_beta_o2o=None, tal_alpha_o2m=None, tal_alpha_o2o=None,
    scb_branch="both",
    o2m_start=0.8, o2m_final=0.1, o2m_decay=True,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0,
    cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
    multi_scale=0.0, scale=0.5, close_mosaic=CLOSE_MOSAIC, cos_lr=False,
)


def _swa(start, end, boost=2.0, px=48):
    """Verbatim from run_yolo26_sweep_v6i.py, which produced y26_swa_a09_04.

    alpha_min/alpha_max are the ENDPOINTS (end, start), not 0.0/1.0 as a later
    helper in the same file used.
    """
    return dict(alpha_start=start, alpha_end=end, alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost)


_B3 = dict(tal_beta=3.0)
# expect: per-branch (alpha, beta), then the weighting stage
_DEF = {"o2m": (0.5, 3.0), "o2o": (0.5, 3.0), "sbb": None, "swa": None, "o2m_final": 0.1}
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def exp(**over):
    d = dict(_DEF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_a1_b6", "ctrl": CTRL_B3_50,
     "params": cfg(tal_alpha=1.0, tal_beta=6.0),
     "expect": exp(o2m=(1.0, 6.0), o2o=(1.0, 6.0)),
     "label": "alpha 1.0 / beta 6.0 — b3's exact positives, targets twice as sharp",
     "why": "Ratio 0.167, identical to b3's 0.5/3.0, so this assigns the same anchors to "
            "the same GTs. What changes is align/align_max, which gets SQUARED, pushing "
            "runner-up positives DOWN and widening the winner/runner-up gap. a025_b4s2 "
            "moved that knob the other way and lost 0.93 on small with large collapsing "
            "3.09 — SNT's exact signature. This is the same axis in the direction that "
            "should help, and it needs no new parameter. Matching b3 exactly would instead "
            "prove beta is pure selection, which is equally worth knowing."},

    {"name": "y26_b3_sbb50", "ctrl": CTRL_B3_50,
     "params": cfg(**_B3, sbb_q=0.5, sbb_invert=True),
     "expect": exp(sbb=(0.5, True)),
     "label": "beta 3 + SBB q=0.5 invert — small mechanism meets large mechanism",
     "why": "b4s2_sbb50 showed SBB is worth +2.66 on LARGE and nothing on small, while "
            "beta=3 is worth +1.84 on small. They target disjoint buckets at different "
            "stages, so this is the most likely configuration in the campaign to be the "
            "best overall mAP50. If SBB was instead rebalancing SCB's over-correction in "
            "one2many, it lands flat and that retires its +0.87 the same way beta retired "
            "SCB."},

    {"name": "y26_b3_sbb50f", "ctrl": CTRL_B3_50,
     "params": cfg(**_B3, sbb_q=0.5, sbb_invert=False),
     "expect": exp(sbb=(0.5, False)),
     "label": "beta 3 + SBB q=0.5 forward — one2many leans SMALL, the untested arm",
     "why": "loss.py:1682 gives invert=False the opposite lean: one2many toward small, "
            "one2one toward large. It lost at beta=6, on a base that needed the large "
            "pull. Here one2many gets a small lean in BOTH the assignment and the "
            "weighting, which is what round 16's branch attribution implies, and it is "
            "the only way to point SBB's weighting at the metric of record. Paired with "
            "run 2 this is a contrast: one of the two has to move."},

    {"name": "y26_b3_swa", "ctrl": CTRL_B3_50,
     "params": cfg(**_B3, **_swa(0.9, 0.4)),
     "expect": exp(swa=(0.9, 0.4, 48.0, 2.0)),
     "label": "beta 3 + SWA 0.9->0.4 — the two multi-run mechanisms, never combined",
     "why": "SWA is 7-for-7 above baseline on mAP50_small (mean +0.52) and was written off "
            "on a b32 run judged by overall mAP50-95, the metric a sqrt(area) box "
            "weighting is designed to trade away. It is the weighting stage, beta is "
            "assignment, and unlike SBB it aims at small objects on BOTH branches. Of the "
            "combinations left this pair has the most independent support."},

    {"name": "y26_b3_o2m", "ctrl": CTRL_B3_50,
     "params": cfg(tal_beta_o2m=3.0, tal_beta_o2o=6.0),
     "expect": exp(o2m=(0.5, 3.0), o2o=(0.5, 6.0)),
     "label": "beta 3 one2many / 6 one2one — is beta an auxiliary-branch effect?",
     "why": "Round 16 scoped SCB per branch: one2many-only 55.79, both 55.66, one2one-only "
            "54.98 — the effect lives in the branch discarded at inference, contradicting "
            "the published single-anchor justification. If the global beta behaves the "
            "same this reproduces b3 with the predicting branch left at stock, and it "
            "tells us to scope SBB and SWA to one2many next; neither ever has been."},

    {"name": "y26_b3_o2o", "ctrl": CTRL_B3_50,
     "params": cfg(tal_beta_o2m=6.0, tal_beta_o2o=3.0),
     "expect": exp(o2m=(0.5, 6.0), o2o=(0.5, 3.0)),
     "label": "beta 6 one2many / 3 one2one — the complement",
     "why": "The arm the original SCB write-up predicted would win, since topk2=1 makes "
            "one2one's exponent look decisive. Round 16 says it lands near baseline. With "
            "run 5 it brackets b3: if neither branch carries the gain alone the effect is "
            "joint, per-branch beta is closed, and that is settled in one night instead "
            "of five."},

    {"name": "y26_a1_b6_o2o", "ctrl": CTRL_B3_50,
     "params": cfg(**_B3, tal_alpha_o2o=1.0, tal_beta_o2o=6.0),
     "expect": exp(o2m=(0.5, 3.0), o2o=(1.0, 6.0)),
     "label": "beta 3 everywhere, targets sharpened on ONE2ONE only",
     "why": "Both branches keep ratio 0.167, so selection is unchanged on both — only "
            "one2one's target magnitudes sharpen. If the sharpness effect is really about "
            "duplicate suppression it must live here: one2one has topk2=1 and emits every "
            "prediction, and SNT's collapse was caused by narrowing precisely this gap. "
            "Run 1 sharpens both branches; this isolates the half that should matter, and "
            "a null here with a win on run 1 would say the effect is auxiliary after all."},

    {"name": "y26_b25", "ctrl": CTRL_B3_50,
     "params": cfg(tal_beta=2.5),
     "expect": exp(o2m=(0.5, 2.5), o2o=(0.5, 2.5)),
     "label": "beta 2.5 uniform — resolve the plateau between b3 and b2",
     "why": "b3 leads mAP50_small by 0.22 over b2 (inside the 0.27 sd) while b2 leads "
            "overall mAP50 by 0.25, so neither is established as the optimum and the two "
            "metrics currently disagree. The midpoint is the single most likely config to "
            "be the number actually reported, and it is one parameter with no interaction "
            "to argue about."},

    {"name": "y26_b3_o2mf30", "ctrl": CTRL_B3_50,
     "params": cfg(**_B3, o2m_final=0.3),
     "expect": exp(o2m_final=0.3),
     "label": "beta 3 + o2m decays 0.8 -> 0.3 — keep the branch beta works in alive",
     "why": "The blend was hardcoded 0.8->0.1 for the whole campaign. Round 16 moved the "
            "endpoint once and got 80.63 against identity's 80.18, and never followed up. "
            "If beta does its work in one2many, holding that branch at three times the "
            "weight through the late epochs should compound with it — two positives at "
            "different stages, the only combination shape that has ever worked here."},

    {"name": "y26_b2m_b4o", "ctrl": CTRL_B3_50,
     "params": cfg(tal_beta_o2m=2.0, tal_beta_o2o=4.0),
     "expect": exp(o2m=(0.5, 2.0), o2o=(0.5, 4.0)),
     "label": "beta 2 one2many / 4 one2one — the asymmetry uniform beta cannot express",
     "why": "The branches have opposite jobs: one2many is auxiliary supervision and wants "
            "many usable positives (ratio past the uniform peak), one2one emits the final "
            "box and plausibly wants the sharper IoU-led pick it has at 6. No run in the "
            "campaign could express this, because tal_beta is assigned once and lands on "
            "both assigners. Last because it presumes runs 5 and 6 split the way round 16 "
            "predicts."},
]


def preflight(todo):
    """Probe the CONSUMER, not the config surface.

    Rounds 4-6 lost ten runs to use_lbtal: default.yaml accepted it, the run header
    printed it, and loss.py never read it. Three files, and only the third matters.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False

    ok = True
    probe = TaskAlignedAssigner(topk=10, num_classes=3, alpha=1.0, beta=3.0)
    hit = abs(float(probe.alpha) - 1.0) < 1e-6 and abs(float(probe.beta) - 3.0) < 1e-6
    print(f"  {'TaskAlignedAssigner stores alpha and beta':<52} {hit}")
    ok &= hit

    branch_keys = ("tal_beta_o2m", "tal_beta_o2o", "tal_alpha_o2m", "tal_alpha_o2o")
    if any(r["params"][k] is not None for r in todo for k in branch_keys):
        try:
            import inspect

            from ultralytics.cfg import get_cfg
            from ultralytics.utils.loss import E2ELoss
            c = get_cfg()
            for k in branch_keys:
                hit = hasattr(c, k)
                print(f"  {'default.yaml declares ' + k:<52} {hit}")
                ok &= hit
                if hit and getattr(c, k) is not None:
                    print(f"  [ABORT] {k} defaults to {getattr(c, k)!r}, must be null")
                    ok = False
            src = inspect.getsource(E2ELoss.__init__)
            hit = all(k in src for k in branch_keys)
            print(f"  {'E2ELoss.__init__ reads all four keys':<52} {hit}")
            ok &= hit
            if not hit:
                print("\n  [ABORT] the keys exist but nothing reads them — this is the")
                print("          use_lbtal failure exactly. Copy loss.py to this box.")
        except Exception as ex:
            print(f"  [ABORT] cannot verify the branch-exponent patch: {ex}")
            return False
    if not ok:
        return False

    print()
    for i, r in enumerate(todo, 1):
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        e = r["expect"]
        rat = f"ratio o2m {e['o2m'][0] / e['o2m'][1]:.3f} o2o {e['o2o'][0] / e['o2o'][1]:.3f}"
        print(f"  {i:>2}. {r['name']:<16} {rat:<28} |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.93 * len(todo):.1f} GPU-h")
    print(f"  base/bar: {CTRL_B3_50:.2f} mAP50 / {CTRL_B3_S50:.2f} mAP50_small (y26_b3)"
          f"   sd_small ~{NOISE_S50}")
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that every requested mechanism is LIVE, and nothing else is."""
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 1:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — not a yolo26 e2e model")
        a1, a2 = o2m.assigner, o2o.assigner
        b1, b2 = o2m.bbox_loss, o2o.bbox_loss
        seen = []

        # ---- per-branch alpha and beta; the RATIO is the selection, the SCALE the target
        for tag, a, want in (("one2many", a1, e["o2m"]), ("one2one", a2, e["o2o"])):
            if abs(float(a.alpha) - want[0]) > 1e-6 or abs(float(a.beta) - want[1]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} (alpha, beta)=({a.alpha}, {a.beta}), "
                                   f"expected {want}")
        if e["o2m"] != e["o2o"] and (abs(float(a1.alpha) - float(a2.alpha)) < 1e-6
                                     and abs(float(a1.beta) - float(a2.beta)) < 1e-6):
            raise RuntimeError(f"{rc['name']}: both branches ended on the same exponents; the "
                               f"branch override did not take effect")
        r1, r2 = e["o2m"][0] / e["o2m"][1], e["o2o"][0] / e["o2o"][1]
        seen.append(f"o2m a={a1.alpha} b={a1.beta} (ratio {r1:.3f})  "
                    f"o2o a={a2.alpha} b={a2.beta} (ratio {r2:.3f})")

        # ---- SBB, including which way each branch leans
        want_sbb = e["sbb"]
        q = 0.0 if want_sbb is None else want_sbb[0]
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if abs(float(b.sbb_q) - q) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {q}")
        if want_sbb is None:
            seen.append("SBB off")
        else:
            invert = want_sbb[1]
            # loss.py:1682 -- invert flips both; sign>0 leans LARGE, sign<0 leans SMALL.
            # The docstring at loss.py:163 states this backwards; the code is authoritative.
            want_o2m = +1.0 if invert else -1.0
            if float(b1.sbb_sign) != want_o2m or float(b2.sbb_sign) != -want_o2m:
                raise RuntimeError(
                    f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f}, "
                    f"expected o2m={want_o2m:+.0f} for invert={invert}")
            seen.append(f"SBB q={q} invert={invert} | one2many leans "
                        f"{'LARGE' if invert else 'SMALL'}")

        # ---- SWA: presence AND absence asserted, on both branches
        want_swa = e["swa"]
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if want_swa is None:
                if b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but not requested")
                continue
            if not b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA not live on {tag}")
            got = (float(b.alpha_start), float(b.alpha_end),
                   float(b.small_obj_px), float(b.small_obj_boost))
            if any(abs(g - w) > 1e-6 for g, w in zip(got, want_swa)):
                raise RuntimeError(f"{rc['name']}: {tag} SWA={got}, expected {want_swa}")
            if b.area_weight_mode != "sqrt":
                raise RuntimeError(f"{rc['name']}: {tag} area_weight_mode={b.area_weight_mode}, "
                                   f"expected sqrt")
            if abs(float(b.alpha_min) - want_swa[1]) > 1e-6 or \
                    abs(float(b.alpha_max) - want_swa[0]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha clamp=({b.alpha_min}, "
                                   f"{b.alpha_max}), expected ({want_swa[1]}, {want_swa[0]})")
        seen.append("SWA off" if want_swa is None else
                    f"SWA {want_swa[0]}->{want_swa[1]} sqrt/max px<{want_swa[2]:.0f} "
                    f"x{want_swa[3]}")

        # ---- blend endpoint
        if abs(float(crit.final_o2m) - e["o2m_final"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: o2m_final={crit.final_o2m}, "
                               f"expected {e['o2m_final']}")
        if abs(float(crit.o2m_copy) - 0.8) > 1e-6 or not crit.o2m_decay:
            raise RuntimeError(f"{rc['name']}: blend start={crit.o2m_copy} decay="
                               f"{crit.o2m_decay}, expected 0.8 and True")
        seen.append(f"blend 0.8 -> {crit.final_o2m}")

        # ---- everything else provably off
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB live on {tag}; round 18 showed it is "
                                   f"subsumed by beta and harmful at beta=3 (-0.72)")
            if a.snt_enabled() or a.tsh_enabled() or a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: SNT/TSH/SBAL live on {tag}, all closed")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SNL1/NWD live on {tag} but not requested")
        if a2.topk2 != 1:
            raise RuntimeError(f"{rc['name']}: one2one topk2={a2.topk2}, expected 1")
        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this is the stock 3-level graph")
        if abs(float(h.cls) - 0.5) > 1e-6 or abs(float(h.box) - 7.5) > 1e-6:
            raise RuntimeError(f"{rc['name']}: gains moved (box={h.box} cls={h.cls})")
        if abs(float(trainer.args.multi_scale)) > 1e-6:
            raise RuntimeError(f"{rc['name']}: multi_scale is live; it measured -1.03 (4.3 sd)")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] SCB/SNT/TSH/SBAL/SNL1/NWD off | box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc, idx, total):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {idx}/{total}  {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}  base={rc['ctrl']:.2f} mAP50")
    print(f"  differs from _ALL_OFF: {({k: v for k, v in rc['params'].items() if _ALL_OFF.get(k, '__') != v})}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(rc["params"])
    results = model.train(**kw)
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "ctrl": rc["ctrl"], "params": rc["params"], "seed": SEED,
           "expect": {k: list(v) if isinstance(v, tuple) else v for k, v in rc["expect"].items()},
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True, "epochs_requested": EPOCHS,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # *_params.json so the eval script's glob binds metrics to a CONFIG, not to
    # directory order — round 16 was mis-evaluated twice for exactly that.
    try:
        with open(os.path.join(save_dir, "round19_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res):
    ok = [r for r in res if r["test_map50"] == r["test_map50"]]
    if not ok:
        print("\nno completed runs.")
        return
    print("\n" + "=" * 78)
    print("  ROUND 19 — RESULTS  (mAP50; mAP50_small needs the COCO pass)")
    print("=" * 78)
    print(f"{'run':<18}{'mAP50':>9}{'vs b3':>9}{'vs stock':>10}{'mAP50-95':>10}{'hours':>7}")
    print("-" * 63)
    print(f"{'y26_identity':<18}{CTRL_ID_50:>9.2f}{CTRL_ID_50 - CTRL_B3_50:>+9.2f}"
          f"{0.0:>+10.2f}{55.24:>10.2f}{'-':>7}")
    print(f"{'y26_b3 (base)':<18}{CTRL_B3_50:>9.2f}{0.0:>+9.2f}"
          f"{CTRL_B3_50 - CTRL_ID_50:>+10.2f}{55.30:>10.2f}{'-':>7}")
    print("-" * 63)
    for r in sorted(ok, key=lambda x: -x["test_map50"]):
        v, v95 = r["test_map50"] * 100, r["test_map5095"] * 100
        print(f"{r['name']:<18}{v:>9.2f}{v - CTRL_B3_50:>+9.2f}"
              f"{v - CTRL_ID_50:>+10.2f}{v95:>10.2f}{r['hours']:>7.2f}")

    by = {r["name"]: r["test_map50"] * 100 for r in ok}
    print("\n  READ IT — mechanism first, combinations second")

    if "y26_a1_b6" in by:
        v = by["y26_a1_b6"]
        print(f"\n    SHARPNESS (selection held fixed at ratio 0.167)")
        print(f"      b3 {CTRL_B3_50:.2f}   a1_b6 {v:.2f}"
              + (f"   o2o-only {by['y26_a1_b6_o2o']:.2f}" if "y26_a1_b6_o2o" in by else ""))
        if v > CTRL_B3_50 + 0.3:
            print("    Same positives, better score: the winner/runner-up gap is a real lever")
            print("    and the assigner can sharpen targets with no new parameter. That is a")
            print("    mechanism, not a hyperparameter, and it explains a025_b4s2's collapse.")
        elif abs(v - CTRL_B3_50) < 0.3:
            print("    Flat: beta is pure SELECTION and alpha/beta are ONE degree of freedom.")
            print("    a025_b4s2's -0.93 was the medium/large confound. Clean, publishable.")
        else:
            print("    Worse: sharper targets HURT, so the gap was already wide enough and")
            print("    a025_b4s2 lost for a different reason. Report both directions.")

    if "y26_b3_sbb50" in by and "y26_b3_sbb50f" in by:
        inv, fwd = by["y26_b3_sbb50"], by["y26_b3_sbb50f"]
        print(f"\n    SBB ARMS   invert {inv:.2f}   forward {fwd:.2f}   base {CTRL_B3_50:.2f}")
        if inv > CTRL_B3_50 + 0.3:
            print("    SBB survives the corrected base — small from beta, large from SBB,")
            print("    disjoint buckets. Check mAP50_small is not the thing that moved.")
        elif fwd > inv + 0.3:
            print("    The arm FLIPPED. SBB's +0.87 at beta=6 was rebalancing SCB's")
            print("    over-correction in one2many, not an effect of its own.")
        else:
            print("    Neither arm clears the base: SBB does not survive beta=3, which kills")
            print("    the campaign's second-best loss result. That is the finding.")

    if "y26_b3_o2m" in by and "y26_b3_o2o" in by:
        m, o = by["y26_b3_o2m"], by["y26_b3_o2o"]
        print(f"\n    BRANCH   o2m-only {m:.2f}   o2o-only {o:.2f}   both {CTRL_B3_50:.2f}")
        if m > o + 0.3:
            print("    beta acts through the AUXILIARY branch, as round 16's SCB scoping said.")
            print("    Scope SBB and SWA to one2many next; neither has ever been scoped.")
        elif o > m + 0.3:
            print("    beta acts through the PREDICTING branch — this CONTRADICTS round 16.")
        else:
            print("    Neither branch carries it alone: the effect is joint and per-branch")
            print("    beta is closed apart from b2m_b4o.")

    for nm, txt in (("y26_b3_swa", "SWA"), ("y26_b25", "beta 2.5"),
                    ("y26_b3_o2mf30", "blend 0.3"), ("y26_b2m_b4o", "asym beta 2/4")):
        if nm in by:
            print(f"    {txt:<16}{by[nm]:>8.2f}  ({by[nm] - CTRL_B3_50:+.2f} vs b3)")

    print(f"\n    mAP50_small is the metric that separates these and is NOT above. Run the")
    print(f"    COCO pass and read against {CTRL_B3_S50:.2f} (b3) and {CTRL_ID_S50:.2f} "
          f"(identity); sd ~{NOISE_S50}.")
    print("    Fix the eval script first: glob '*_params.json' instead of the hardcoded")
    print("    names, and skip runs whose results.csv is short of 70 rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    ap.add_argument("--from", dest="start", type=int, default=1, help="resume at run N (1-based)")
    a = ap.parse_args()

    todo = RUNS[max(a.start - 1, 0):]
    if a.names:
        todo = [r for r in RUNS if r["name"] in a.names]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 19 — overnight, {len(todo)} runs on the beta=3 base (loss only)")
    print("  " + "  ".join(r["name"] for r in todo))
    print("=" * 84)
    if not preflight(todo):
        return

    res, out_path = [], f"{PROJECT_DIR}_results.json"
    for i, rc in enumerate(todo, 1):
        try:
            res.append(run_one(rc, i, len(todo)))
        except Exception as ex:
            print(f"\n  [FAILED] {rc['name']}: {ex}\n")
        # written after EVERY run: if the queue dies at 3am the finished ones survive
        try:
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
        except Exception as ex:
            print(f"  [warn] results not saved: {ex}")
    summarise(res)
    print(f"\n  results -> {out_path}")


if __name__ == "__main__":
    main()
