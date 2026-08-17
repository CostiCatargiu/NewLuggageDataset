#!/usr/bin/env python3
r"""
YOLO26 round 9 — MECHANISM COMBINATIONS + TARGET SHARPENING (stock yolo26s, b82)
===============================================================================

Two things this campaign has never done, on the v6i test split, batch-matched to
the baseline.

WHY THIS IS NOT MORE PARAMETER TUNING
-------------------------------------
Arm A fills a hole in the ablation table: every loss run so far has been ONE
mechanism at a time, so "do they compose?" is unanswered. That question needs
answering for the paper whichever way it comes out.

Arm B is a new target definition (TSH), derived from the campaign's largest
measured effect rather than from a guess. See below.


WHAT CHANGED IN HOW TO READ THESE NUMBERS
-----------------------------------------
`y26_base_rep` trained a full 70 epochs (3149 s, close_mosaic firing at 61) and
came back BIT-IDENTICAL to `yolo26_custom-9` across all 28 metrics and 90
per-class values. Training on this box is DETERMINISTIC.

That retires the "sd = 0.19 noise floor" this project has been using. If the
rounds 4-6 runs had truly been one config repeated, determinism says they would
have been identical; they spanned 55.89..56.46. So they were real configuration
differences, and that 0.19 was never a noise floor. Same for the "large sd 2.11"
that was used to wave off every large-object result.

Consequence for this script: deltas below are EXACT and reproducible. A +0.42 is
a real, repeatable +0.42. Do not apply a noise band to it.

The limitation that REPLACES it, and it is a real one: determinism makes the
measurement exact, not general. Every number here is seed 0. An exactly
reproducible gain can still be seed-specific, and a reviewer will ask. This
script cannot answer that; only seeds can.


ARM A — DO THE WORKING MECHANISMS COMPOSE?
------------------------------------------
Batch-matched singles, all b82, all vs 55.24:

    y26_scb_b3      55.66   +0.42     SCB   tal_beta_small=3.0
    y26_snl1_p25    55.49   +0.25     SNL1  l1_scale_p=0.25
    y26_snl1_p50    55.48   +0.24     SNL1  l1_scale_p=0.50
    y26_sbb_inv50   55.39   +0.15     SBB   q=0.5, invert=True
    y26_beta4       55.37   +0.13     (gain/exponent probe)
    y26_dfl3        55.37   +0.13     (gain probe)

Where each one acts — this is why SCB and SNL1 should compose and why SBB and
SNL1 might not:

    SCB    tal.py:321     beta exponent inside align_metric
                          -> changes WHICH anchors become positive
    SNL1   loss.py:382    divides the L1 residual by extent^p
                          -> changes HOW MUCH each positive contributes
    SBB    loss.py:357    multiplies the SAME weight tensor SNL1 divides
                          -> overlaps with SNL1 by construction

SCB and SNL1 touch different files, different tensors, different stages. They
are orthogonal, so if both effects are real the composition should land near
additive (+0.42 + 0.25 = +0.67). SBB and SNL1 both scale the regression term
multiplicatively, so run 4 is the one that can come out sub-additive.

Reading it:
    near-additive        both mechanisms are real and independent -> best config
                         for the paper, and the ablation table writes itself
    sub-additive         they are two routes to the same correction; report the
                         better single and say so
    below either single  they interfere; that is also a finding, and it is the
                         outcome that makes SCB's knife-edge look seed-specific


ARM B — TSH, TARGET SHARPENING (the new mechanism)
--------------------------------------------------
SNT was falsified hard and the SHAPE of that failure is informative:

    tau=0.25   -3.93 mAP      small -3.54   medium -3.64   large -10.58
    tau=0.50  -12.00 mAP      small -12.67  medium -8.64   large -16.60

Monotone in tau, and 10-60x anything determinism allows to be noise. But note
what happened to recall:

    AR50_95_small   71.68 (base) -> 76.00 (t25) -> 75.79 (t50)

Detector recall went UP while AP collapsed. AR up + AP down is a RANKING
failure, not a detection failure. The boxes are found; their scores are wrong.

The mechanism is specific to this architecture. YOLO26's head is end2end with no
NMS, and one2one uses topk2=1 — exactly ONE anchor per GT is positive. The thing
that suppresses duplicates is the confidence gap between that winner and its
well-overlapping neighbours. SNT raised the neighbours' targets, closing the only
gap doing suppression, so duplicates were emitted at near-equal confidence and
interleaved with true positives in the ranking. Large objects span the most
anchors and so emit the most duplicates: -16.60, the worst column. That is the
mechanism's fingerprint, and it is why the whole soft-target family (VFL, label
smoothing, quality-aware targets) should fail on this head.

TSH tests the inverse: WIDEN the gap SNT narrowed. After normalization at
tal.py:184 the winner's target is

    norm_align_metric = IoU_max * (align / align_max)     which is < 1

TSH raises it toward 1.0 with a power:

    target_scores <- target_scores ** sharp_rho           rho < 1 sharpens

Monotone and order-preserving, so it CANNOT reorder the assignment; it only
changes magnitudes. rho = 1.0 is the identity, so stock is untouched rather than
approximated. One2one only, for the same reason SNT was: one2one is the branch
that produces every prediction.

This is a target-DEFINITION change, not a gain. The cls gain multiplies the loss;
this changes what the model is asked to predict. That distinction is why the gain
probes are not a substitute — and the gain axis is answered anyway, `dfl3`
(gain doubled) and `beta4` both landed at +0.13.

FALSIFIABLE, and pre-registered here so it cannot be reinterpreted after the
fact: if the gap account is right, LARGE-object AP rises first and small barely
moves — the exact mirror of how SNT failed. If large falls again, the gap is
already at its optimum, the account is wrong in the useful direction, and the
loss axis is closed.


CALIBRATION NOTE — READ BEFORE TRUSTING ANY EXPECTATION ABOVE
-------------------------------------------------------------
Nine directional predictions have been made across this campaign. Nine were
falsified, every one optimistic, including SNT which had a mechanistic story at
least as tidy as the one above. Treat the TSH argument as a reason the
experiment is WORTH RUNNING, not as a reason to expect it to work. What has
actually earned trust here is cheap falsifiable probes and hard guards.

Arm A is not a prediction at all. It is a hole in the ablation table.


COST
----
Measured from y26_base_rep/results.csv: 3149 s for 70 epochs = 52 min, plus the
test eval. ~1.0 GPU-h per run, ~6 h for all six. One overnight.

Usage:
    python run_yolo26_combo_v6i.py                    # all six
    python run_yolo26_combo_v6i.py y26_sharp_r50      # a subset
    python run_yolo26_combo_v6i.py --arm b            # sharpening only
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
PROJECT_DIR = "runs_yolo26_combo_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82          # matches the baseline and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# Batch-matched references (all b82, all exact under determinism)
BASELINE = 55.24        # yolo26_custom-9 == y26_base_rep, bit-identical
BASE_SMALL = 51.00
BASE_MED = 65.98
BASE_LARGE = 60.87      # unbeaten in 54 configs
SCB_B3 = 55.66          # +0.42, best single loss config
SNL1_P25 = 55.49        # +0.25
SNL1_P50 = 55.48        # +0.24
SBB_INV50 = 55.39       # +0.15

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0,
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


# expect: what the epoch-1 guard must find LIVE in the constructed criterion.
#   scb  -> (beta_small, ref_px) on BOTH branches   (v8DetectionLoss-level)
#   snl1 -> l1_scale_p on BOTH branches             (v8DetectionLoss-level)
#   sbb  -> q, with OPPOSITE signs across branches  (E2ELoss-level)
#   tsh  -> sharp_rho on one2one ONLY               (E2ELoss-level)
RUNS = [
    # ---- ARM A: do the working mechanisms compose? --------------------------
    {"name": "y26_scb3_snl25", "arm": "a",
     "expect": {"scb": (3.0, 64.0), "snl1": 0.25},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, l1_scale_p=0.25),
     "label": "SCB 3.0 + SNL1 0.25 — the clean orthogonal pair",
     "why": "The two best singles (+0.42, +0.25) acting at different stages: SCB "
            "changes which anchors are selected (tal.py, assignment), SNL1 changes "
            "how much each selected anchor contributes (loss.py, regression). No "
            "shared tensor, no shared file. If both effects are real and "
            "independent this should land near +0.67. This is the single most "
            "informative run in the script: it is the only one that can produce a "
            "best-config number for the paper AND test whether the two mechanisms "
            "are measuring the same underlying correction twice."},

    {"name": "y26_scb3_snl50", "arm": "a",
     "expect": {"scb": (3.0, 64.0), "snl1": 0.50},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, l1_scale_p=0.50),
     "label": "SCB 3.0 + SNL1 0.50 — same pair, stronger SNL1",
     "why": "p=0.25 and p=0.50 were within 0.01 of each other as singles (+0.25 vs "
            "+0.24), which says the SNL1 axis is flat in that range. Running both "
            "under SCB tests whether that flatness survives composition. If the two "
            "combos also land within 0.01, SNL1's contribution is genuinely "
            "insensitive to p and the mechanism is robust. If they diverge, the "
            "flatness was a coincidence of the singles and p matters after all — "
            "worth knowing before anything goes in a paper as a recommended value."},

    {"name": "y26_scb3_sbb50", "arm": "a",
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True),
     "label": "SCB 3.0 + SBB inv 0.5 — assignment + branch-asymmetric weighting",
     "why": "Also orthogonal, by a different route: SBB acts at the E2ELoss level "
            "and gives the two branches OPPOSITE size preferences, which nothing "
            "SCB does can imitate. sbb_invert=True is used because that is the arm "
            "that worked (+0.15); the non-inverted arm was -0.08, and running the "
            "losing sign here would test nothing. Weaker single than SNL1, but it "
            "is the only mechanism in the campaign that exploits the two-branch "
            "structure, so it composes differently."},

    {"name": "y26_scb3_snl25_sbb", "arm": "a",
     "expect": {"scb": (3.0, 64.0), "snl1": 0.25, "sbb": 0.5},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, l1_scale_p=0.25,
                   sbb_q=0.5, sbb_invert=True),
     "label": "SCB 3.0 + SNL1 0.25 + SBB inv 0.5 — all three",
     "why": "The one run that can come out sub-additive for a reason known in "
            "advance: SBB multiplies the exact weight tensor SNL1 divides "
            "(loss.py:357 vs :382), so these two are partially redundant by "
            "construction while SCB stays independent of both. Compare against "
            "y26_scb3_snl25 and y26_scb3_sbb50: if the triple beats both pairs, the "
            "redundancy is not binding. If it lands between them, it is, and the "
            "paper reports the better pair instead of stacking everything."},

    # ---- ARM B: target sharpening, derived from the SNT failure -------------
    {"name": "y26_sharp_r75", "arm": "b",
     "expect": {"tsh": 0.75},
     "params": cfg(sharp_rho=0.75),
     "label": "TSH rho=0.75 — widen the winner/runner-up gap, gently",
     "why": "The conservative point, deliberately first. SNT proved this gap is "
            "extraordinarily load-bearing — moving it the wrong way cost 12 mAP — "
            "so the prior on step size should be small in either direction. At "
            "rho=0.75 a target of 0.50 becomes 0.59 and 0.80 becomes 0.85: the "
            "winner is pushed toward 1.0 without collapsing the distinction between "
            "a well-fitted and a marginal assignment. Order is preserved exactly, "
            "so the assignment itself cannot change; only the magnitudes do."},

    {"name": "y26_sharp_r50", "arm": "b",
     "expect": {"tsh": 0.50},
     "params": cfg(sharp_rho=0.50),
     "label": "TSH rho=0.50 — stronger (target -> sqrt(target))",
     "why": "Makes rho a DIRECTION rather than a single guess, which is what made "
            "the SNT result readable even though it was negative: two points moving "
            "the same way is a trend, one point is an anecdote. At rho=0.50 a "
            "target of 0.50 becomes 0.71. If r75 and r50 both move large-object AP "
            "the same way, that is the pre-registered signature. If they move in "
            "OPPOSITE directions, the gap has an interior optimum near stock and "
            "the account is wrong — which closes the loss axis cleanly rather than "
            "leaving it ambiguous."},
]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")

    A = TAL.TaskAlignedAssigner
    # forward() is only an OOM-fallback wrapper delegating to _forward(); the
    # assignment work lives in _forward. Checking forward() gives a FALSE NEGATIVE
    # — that mistake cost a cycle on the SNT run, so both are searched here.
    def in_assign_path(needle):
        return any(needle in inspect.getsource(getattr(A, m))
                   for m in ("_forward", "forward") if hasattr(A, m))

    checks = {
        "tal.py  TaskAlignedAssigner.scb_enabled": hasattr(A, "scb_enabled"),
        "tal.py  TaskAlignedAssigner.tsh_enabled": hasattr(A, "tsh_enabled"),
        "tal.py  assignment path applies sharp_rho": in_assign_path("sharp_rho"),
        "loss.py BboxLoss.l1_scale_denom (SNL1)": hasattr(BboxLoss, "l1_scale_denom"),
        "loss.py BboxLoss.sbb_weight (SBB)": hasattr(BboxLoss, "sbb_weight"),
        "loss.py v8DetectionLoss reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py v8DetectionLoss reads l1_scale_p": "l1_scale_p" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
        "loss.py E2ELoss reads sharp_rho": "sharp_rho" in inspect.getsource(E2ELoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<46}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] the patch is not fully installed on this machine.")
        print("  Copy ultralytics26/ultralytics/{utils/tal.py,utils/loss.py,cfg/default.yaml}")
        print("  then: python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False

    # rho = 1.0 must be a genuine no-op, not an approximate one. If tsh_enabled()
    # were True at 1.0 the "stock" arm would silently run a pow() and every delta
    # in the campaign would shift.
    probe = A(topk=7, topk2=1)
    if probe.tsh_enabled():
        print(f"  [ABORT] tsh_enabled() is True at the default sharp_rho="
              f"{probe.sharp_rho}. rho=1.0 must be inert.")
        return False
    print(f"  {'TSH inert at default rho=1.0':<46}True")

    print()
    for r in todo:
        p, e = r["params"], r["expect"]
        bits = []
        if "scb" in e:
            a = A(topk=7, topk2=1)
            a.beta_small, a.beta_ref_px = p["tal_beta_small"], p["tal_beta_ref_px"]
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: scb_enabled() False at beta_small={a.beta_small}")
                return False
            bits.append(f"SCB {a.beta_small}->{a.beta} @{a.beta_ref_px}px")
        if "snl1" in e:
            bits.append(f"SNL1 p={p['l1_scale_p']}")
        if "sbb" in e:
            bits.append(f"SBB q={p['sbb_q']} inv={p['sbb_invert']}")
        if "tsh" in e:
            a = A(topk=7, topk2=1)
            a.sharp_rho = p["sharp_rho"]
            if not a.tsh_enabled():
                print(f"  [ABORT] {r['name']}: tsh_enabled() False at rho={a.sharp_rho}")
                return False
            bits.append(f"TSH rho={a.sharp_rho}")
        print(f"  {r['name']:<22}{' + '.join(bits)}")

    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock, no yaml)  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  baseline {BASELINE:.2f} (b82, == y26_base_rep bit-identical)")
    print(f"  singles: SCB {SCB_B3:+.2f}  SNL1p25 {SNL1_P25 - BASELINE:+.2f}  "
          f"SBBinv {SBB_INV50 - BASELINE:+.2f}".replace(f"{SCB_B3:+.2f}", f"{SCB_B3 - BASELINE:+.2f}"))
    print()
    print("  TRAINING ON THIS BOX IS DETERMINISTIC (y26_base_rep == yolo26_custom-9)")
    print("  -> deltas below are EXACT. Do not apply a noise band to them.")
    print("  -> but every number is seed 0; exact does not mean general.")

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
    """Assert at epoch 1 that every requested mechanism is LIVE in the constructed
    criterion — not merely accepted by the config system. A config key can be
    accepted, printed in the header, and silently ignored; that is exactly how
    rounds 4-6 produced ten identically-configured runs under ten different names.
    """
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — this is not yolo26 e2e")
        a1, a2 = o2m.assigner, o2o.assigner
        b1, b2 = o2m.bbox_loss, o2o.bbox_loss
        seen = []

        if "scb" in e:
            want_b, want_r = e["scb"]
            for tag, a in (("one2many", a1), ("one2one", a2)):
                if not (hasattr(a, "scb_enabled") and a.scb_enabled()):
                    raise RuntimeError(
                        f"{rc['name']}: SCB requested but scb_enabled() is False on {tag} "
                        f"(beta_small={getattr(a, 'beta_small', None)}). Aborting rather "
                        f"than producing a number that does not measure SCB.")
                if abs(float(a.beta_small) - want_b) > 1e-6 or abs(float(a.beta_ref_px) - want_r) > 1e-6:
                    raise RuntimeError(
                        f"{rc['name']}: {tag} SCB is ({a.beta_small}, {a.beta_ref_px}), "
                        f"expected ({want_b}, {want_r})")
            seen.append(f"SCB {a2.beta_small}->{a2.beta} @{a2.beta_ref_px}px on BOTH branches")

        if "snl1" in e:
            for tag, b in (("one2many", b1), ("one2one", b2)):
                if not (hasattr(b, "snl1_enabled") and b.snl1_enabled()):
                    raise RuntimeError(
                        f"{rc['name']}: SNL1 requested but not live on {tag} "
                        f"(l1_scale_p={getattr(b, 'l1_scale_p', None)}, "
                        f"dfl_loss={getattr(b, 'dfl_loss', 'n/a')}). SNL1 needs reg_max=1.")
                if abs(float(b.l1_scale_p) - e["snl1"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} l1_scale_p={b.l1_scale_p}, expected {e['snl1']}")
            seen.append(f"SNL1 p={b2.l1_scale_p} on BOTH branches")

        if "sbb" in e:
            for tag, b in (("one2many", b1), ("one2one", b2)):
                if not (hasattr(b, "sbb_enabled") and b.sbb_enabled()):
                    raise RuntimeError(
                        f"{rc['name']}: SBB requested but not live on {tag} "
                        f"(q={getattr(b, 'sbb_q', None)}, sign={getattr(b, 'sbb_sign', None)})")
                if abs(float(b.sbb_q) - e["sbb"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {e['sbb']}")
            # The whole point of SBB is the branch ASYMMETRY. Same sign on both
            # would be a global size reweighting, i.e. a different mechanism.
            if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
                raise RuntimeError(
                    f"{rc['name']}: SBB signs are one2many={b1.sbb_sign:+.0f} "
                    f"one2one={b2.sbb_sign:+.0f} — they must be OPPOSITE, or this is "
                    f"not SBB at all.")
            seen.append(f"SBB q={b2.sbb_q} signs o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f}")

        if "tsh" in e:
            if not (hasattr(a2, "tsh_enabled") and a2.tsh_enabled()):
                raise RuntimeError(
                    f"{rc['name']}: TSH requested but tsh_enabled() is False on one2one "
                    f"(sharp_rho={getattr(a2, 'sharp_rho', None)}). E2ELoss is not wiring it.")
            if abs(float(a2.sharp_rho) - e["tsh"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: one2one sharp_rho={a2.sharp_rho}, expected {e['tsh']}")
            if hasattr(a1, "tsh_enabled") and a1.tsh_enabled():
                raise RuntimeError(
                    f"{rc['name']}: TSH is live on ONE2MANY (rho={a1.sharp_rho}). It must be "
                    f"one2one only — one2many is auxiliary and discarded at inference, so "
                    f"sharpening it would not touch the duplicate-suppression gap this "
                    f"mechanism is about.")
            if int(getattr(a2, "topk2", 0)) != 1:
                raise RuntimeError(
                    f"{rc['name']}: one2one topk2={a2.topk2}, expected 1. TSH's premise is "
                    f"the single-positive selection — without it there is no winner/runner-up "
                    f"gap to widen.")
            seen.append(f"TSH rho={a2.sharp_rho} on one2one ONLY (topk2={a2.topk2})")

        # Anything NOT requested must be provably off, so a combination run is a
        # combination of exactly the named mechanisms and nothing else.
        if "scb" not in e and any(hasattr(a, "scb_enabled") and a.scb_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: SCB is live but was not requested")
        if "snl1" not in e and any(hasattr(b, "snl1_enabled") and b.snl1_enabled() for b in (b1, b2)):
            raise RuntimeError(f"{rc['name']}: SNL1 is live but was not requested")
        if "sbb" not in e and any(hasattr(b, "sbb_enabled") and b.sbb_enabled() for b in (b1, b2)):
            raise RuntimeError(f"{rc['name']}: SBB is live but was not requested")
        if "tsh" not in e and any(hasattr(a, "tsh_enabled") and a.tsh_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
        if any(hasattr(a, "snt_enabled") and a.snt_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00 and is off in every run here.")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if hasattr(b, "swa_enabled") and b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={crit.hyp.box} cls={crit.hyp.cls} dfl={crit.hyp.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {name}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": rc["arm"], "params": rc["params"], "expect": rc["expect"],
           "seed": SEED, "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH,
           "hours": hours, "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "combo_params.json"), "w") as f:
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


def summarise(res, path):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    print()
    print("=" * 88)
    print(f"  ROUND 9 — combinations + target sharpening | stock {MODEL_WEIGHTS}, "
          f"b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 88)
    print(f"{'run':<24}{'arm':>5}{'mAP50':>9}{'mAP50-95':>10}{'vs base':>9}")
    print("-" * 88)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<24}{r['arm'].upper():>5}{r['test_map50'] * 100:>9.2f}"
              f"{v:>10.2f}{v - BASELINE:>+9.2f}")
    print("-" * 88)
    print(f"  {'baseline (b82)':<24}{'':>5}{'':>9}{BASELINE:>10.2f}")
    print(f"  {'y26_scb_b3 (single)':<24}{'':>5}{'':>9}{SCB_B3:>10.2f}{SCB_B3 - BASELINE:>+9.2f}")
    print(f"  {'y26_snl1_p25 (single)':<24}{'':>5}{'':>9}{SNL1_P25:>10.2f}{SNL1_P25 - BASELINE:>+9.2f}")
    print(f"  {'y26_sbb_inv50 (single)':<24}{'':>5}{'':>9}{SBB_INV50:>10.2f}{SBB_INV50 - BASELINE:>+9.2f}")
    print()
    print("  Deltas are EXACT — training on this box is deterministic. No noise band.")
    print()

    arm_a = [r for r in ok if r["arm"] == "a"]
    if arm_a:
        print("  ARM A — additivity")
        add = (SCB_B3 - BASELINE) + (SNL1_P25 - BASELINE)
        print(f"    SCB {SCB_B3 - BASELINE:+.2f} and SNL1p25 {SNL1_P25 - BASELINE:+.2f} "
              f"-> perfectly additive would be {add:+.2f} ({BASELINE + add:.2f})")
        for r in sorted(arm_a, key=lambda x: -x["test_map5095"]):
            d = r["test_map5095"] * 100 - BASELINE
            if r["name"] == "y26_scb3_snl25":
                frac = d / add if abs(add) > 1e-9 else float("nan")
                print(f"    {r['name']:<24}{d:+.2f}   = {frac * 100:.0f}% of additive")
        print("      >= ~90% of additive  -> independent mechanisms; report the combo")
        print("      ~50%                 -> same correction reached two ways")
        print("      < either single      -> they interfere; SCB's peak likely seed-specific")
        print()

    arm_b = [r for r in ok if r["arm"] == "b"]
    if arm_b:
        print("  ARM B — the pre-registered test, and it is NOT this table")
        print("  Run CocoEvalAllFolders_luggage.py on each best.pt and fill in:")
        print()
        print(f"    {'config':<24}{'small':>8}{'medium':>8}{'large':>8}")
        print(f"    {'baseline':<24}{BASE_SMALL:>8.2f}{BASE_MED:>8.2f}{BASE_LARGE:>8.2f}")
        print(f"    {'y26_snt_t25 (SNT)':<24}{47.46:>8.2f}{62.34:>8.2f}{50.29:>8.2f}   <- gap CLOSED")
        for r in sorted(arm_b, key=lambda x: -x["params"]["sharp_rho"]):
            print(f"    {r['name']:<24}{'____':>8}{'____':>8}{'____':>8}")
        print()
        print("    BOTH rho points raise LARGE, small ~flat")
        print("        -> the gap account holds; SNT's -16.60 and this share one mechanism,")
        print("           which makes the negative result and the positive one ONE finding")
        print("    both LOWER large                 -> stock is already at the optimum;")
        print("                                        loss axis closes, cleanly and citably")
        print("    r75 and r50 move OPPOSITE ways   -> interior optimum near rho=1;")
        print("                                        the account is wrong, report it")
        print()
        print("    Watch PRECISION too. SNT lost AP while AR ROSE (71.68 -> 76.00);")
        print("    sharpening should do the mirror — if AR falls while AP rises, the")
        print("    duplicate-suppression story is confirmed from both directions.")
    print()
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<24} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    arm = None
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1].lower()
        del args[i:i + 2]
    only = set(args)
    todo = [r for r in RUNS if (not only or r["name"] in only) and (not arm or r["arm"] == arm)]
    if not todo:
        sys.exit(f"no runs match {only or ''} {('arm ' + arm) if arm else ''}")

    print()
    print("=" * 88)
    print(f"  YOLO26 ROUND 9 — {len(todo)} runs, ~{1.0 * len(todo):.1f} GPU-h "
          f"(measured: 52 min/70 epochs + test eval)")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 88)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")

    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except KeyboardInterrupt:
            print("\n  interrupted by user")
            break
        except Exception as ex:
            print(f"\n  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump({"baseline": BASELINE, "batch": BATCH, "seed": SEED,
                       "deterministic": True, "results": res}, f, indent=2)

    summarise(res, out_path)
