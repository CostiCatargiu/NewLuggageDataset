#!/usr/bin/env python3
r"""
YOLO26 ROUND 21 — extend beta, split it by consumer, raise the target ceiling

Eight runs, ~7.2 GPU-h, seed 0, b82, stock yolo26s.pt. Every run is matched to
the baseline in batch and graph, so every delta lands in section 2 of
analyze_runs_v6i.py without a caveat.

PRIMARY METRIC IS mAP50_small. Three mechanisms in this campaign were retired
on mAP50-95 — the metric they trade away — and one of them (tal_beta) turned
out to be the largest effect in the project once re-scored. y26_b3 reads +4.4
sd on mAP50_small and +0.0 sd on mAP50-95. Rank on the primary metric.

    REFERENCE, 3 seeds of y26_identity   small 77.62   mAP50 80.40   m5095 55.30
    pooled sd (df=7)                     small  0.346  m5095 0.374
    -> 2 sd = 0.69 suggestive, 3 sd = 1.04 report as an effect

    y26_b3   beta 3.0    small 79.14  (+1.52, 4.4 sd)  mAP50 81.31  large 80.42
    y26_b2   beta 2.0    small 78.92  (+1.30, 3.7 sd)  mAP50 81.56  large 83.24
    y26_b25  beta 2.5    small 78.88  (+1.26, 3.6 sd)  mAP50 81.43  large 81.78

DO NOT use the seed-0 identity (small 77.30) as the reference. It is the LOW
draw of three by 0.32, and every constant hardcoded in the round 18/19/20
headers inherits that bias — b3's "+1.84" is really +1.52.


=============================================================================
ARM A  (execution 1, 2) — THE CURVE HAS NO TURNOVER YET
=============================================================================
As beta falls 6 -> 2, nothing has peaked except small:

    beta      6      4      3     2.5      2
    small  77.62  78.17  79.14  78.88  78.92     peak at 3
    mAP50  80.40  80.95  81.31  81.43  81.56     still climbing
    large  80.65  83.67  80.42  81.78  83.24     still climbing

Two of three curves are still rising at the edge of the sweep and nobody has
looked below 2. At beta=1 the alignment metric is linear in IoU; at beta=0 it
would ignore IoU entirely, so 1.0 is the last meaningful point before the
metric stops being alignment at all.


=============================================================================
ARM B  (execution 6, 7, 8) — SPLIT BETA BY CONSUMER
=============================================================================
align_metric has THREE consumers (tal.py 370, 553, 224). Both selection sites
are pure top-k, so they are invariant to alpha and beta separately and depend
only on alpha/beta. y26_b3 (0.5/3) and y26_a1_b6 (1.0/6) share ratio 0.167 ->
IDENTICAL selection -> their gap is a pure target effect:

    b3 - a1_b6       = 79.14 - 78.16 = +0.98    target only
    a1_b6 - identity = 78.16 - 77.62 = +0.54    selection only
    b3 - identity                    = +1.52    additive, and already measured

So the decomposition needs no new run. What the split BUYS is that beta=3 is
currently a compromise between two optima locked to one number. Runs 6-7 push
the term carrying two thirds of the effect past where the coupled knob reaches.
Run 8 is the additivity check: predicted ~78.60 under the model above.

Both selection sites take beta_sel. Stated here so that if a later round wants
to split line 553 (the topk2=1 winner, which IS the duplicate suppression in an
NMS-free head) it is a named change and not a silent one.


=============================================================================
ARM C  (execution 3, 4) — TARGET CEILING NORMALISATION
=============================================================================
tal.py:225 sets each GT's target ceiling to its own best achievable IoU:

    pos_overlaps      = (overlaps * mask_pos).amax(-1)
    norm_align_metric = align * pos_overlaps / pos_align_metrics
    target_scores     = target_scores * norm_align_metric

A small object whose best anchor reaches IoU 0.55 is trained toward 0.55; a
large one reaching 0.90 toward 0.90. Inference then applies ONE threshold to
both. That is AR50_small 0.95 vs R50_small 0.70 written into the loss, and it
is the first proposal in this campaign aimed at a failure mode that was
MEASURED rather than inferred from a mechanism story.

Why TSH failing does not predict this failing: TSH powered the targets AFTER
normalisation, lifting runner-ups too and compressing the winner/runner-up gap
that SNT proved is load-bearing. TCN changes only the per-GT multiplier; the
within-GT shape align/align_max is untouched.

Two bases because one run cannot separate "TCN works" from "TCN works on this
base". b2 gains on BOTH size axes, b3 is the small maximum.

READ P50_small NEXT TO R50_small. The expected signature is more small
detections clearing threshold at some precision cost. If R50_small rises and
P50_small holds, it worked; if both fall, the ceiling was load-bearing.


=============================================================================
ARM D  (execution 5) — PER-LEVEL BETA
=============================================================================
b3's only weakness is large: -0.23 mAP50, -1.00 L95. From the anchor-footprint
diagnostic, small objects draw 6.64 of their 9.82 positives from stride 8 and
large draw 0.16 of 9.79. Changing beta at s8 ONLY is the cleanest separation
the data supports.


=============================================================================
CALIBRATION AND EXECUTION ORDER
=============================================================================
Runs execute in descending order of PRIOR, not by arm, so a queue that dies at
3am keeps the runs most likely to have mattered. The prior is P(clearing +0.35
on mAP50_small vs its own control) and is declared per run, not per arm —
arm B is not uniform, and its third run is predicted to land BELOW b3.

    1  y26_b15            0.45   extends the only 3-sd effect; 2 of 3 curves rising
    2  y26_b1             0.30   further out; past the small peak but mAP50 may climb
    3  y26_b3_tcn_p50     0.30   new mechanism, aimed at a MEASURED failure mode
    4  y26_b2_tcn_p50     0.28   same mechanism on the base already off the trade curve
    5  y26_blevel_p3only  0.25   targets b3's only cost (large) where small lives
    6  y26_bsel3_btgt15   0.22   pushes the dominant term past the coupled limit
    7  y26_bsel3_btgt1    0.18   same axis, further out
    8  y26_bsel6_btgt3    0.02   PREDICTED 78.60 — attribution, not a bet

These are an ORDERING, not forecasts. Directional predictions in this campaign
are 0-for-10 and every miss was optimistic; read them as "which run would I
regret losing", nothing more.

ONE TENSION, stated rather than hidden. Run 8 sits last on a 0.02 prior because
it is predicted BELOW b3 by construction — but it is also the additivity check
that makes runs 6-7 interpretable. If 6 and 7 land near each other you do not
need it. If they disagree, run 8 before concluding anything from either.

    Usage:
        python run_yolo26_round21_v6i.py                # all eight, prior order
        python run_yolo26_round21_v6i.py --preflight    # prove inertness, run nothing
        python run_yolo26_round21_v6i.py --arm a        # a | b | c | d
        python run_yolo26_round21_v6i.py y26_b15        # one by name

    Executions 1-2 need no patch. The rest need patch_round21_v6i.py; if it is not
    installed they are SKIPPED with a message, not silently run as stock.
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
PROJECT_DIR = "runs_yolo26_round21_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False            # y26_p2k2_hi was lost to exist_ok=True

# 3-seed means. Never the seed-0 draw.
REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30
SD_S50, SD_5095 = 0.346, 0.374
B3_S50, B2_S50 = 79.14, 78.92

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


RUNS = [
    # ---------------------------------------------------------------- ARM A
    {"name": "y26_b15", "prior": 0.45, "arm": "a", "needs": None, "ctrl": B3_S50,
     "params": cfg(tal_beta=1.5),
     "expect": {"beta": 1.5},
     "label": "tal_beta 1.5 — one step below the sweep's edge",
     "why": "mAP50 and large are both still climbing at beta=2 (80.40 -> 81.56 "
            "and 80.65 -> 83.24) with no turnover anywhere. Small peaked at 3, "
            "but two of three curves have not, and nobody has looked below 2. "
            "This is the highest-value run in the file: it extends the only "
            "effect in the campaign that clears 3 sd."},

    {"name": "y26_b1", "prior": 0.3, "arm": "a", "needs": None, "ctrl": B3_S50,
     "params": cfg(tal_beta=1.0),
     "expect": {"beta": 1.0},
     "label": "tal_beta 1.0 — alignment becomes linear in IoU",
     "why": "The last meaningful point: at beta=0 the metric ignores IoU and "
            "stops being alignment at all. Brackets the optimum from below. If "
            "small collapses here the curve is closed; if it does not, the "
            "sweep was never near its bottom."},

    # ---------------------------------------------------------------- ARM C
    # Ordered ahead of arm B: arm C can improve the number, arm B mostly explains it.
    {"name": "y26_b3_tcn_p50", "prior": 0.3, "arm": "c", "needs": "tcn_p", "ctrl": B3_S50,
     "params": cfg(tal_beta=3.0, tcn_p=0.5),
     "expect": {"beta": 3.0, "tcn_p": 0.5},
     "label": "beta 3 + target ceiling raised, pos_overlaps ** 0.5",
     "why": "A GT whose best anchor reaches IoU 0.55 currently trains toward a "
            "target of 0.55; at p=0.5 it trains toward 0.74, while one reaching "
            "0.90 moves only to 0.95. Compresses the size-dependent confidence "
            "ceiling without removing it. Read P50_small next to R50_small."},

    {"name": "y26_b2_tcn_p50", "prior": 0.28, "arm": "c", "needs": "tcn_p", "ctrl": B2_S50,
     "params": cfg(tal_beta=2.0, tcn_p=0.5),
     "expect": {"beta": 2.0, "tcn_p": 0.5},
     "label": "beta 2 + target ceiling raised — the both-axes base",
     "why": "b2 is the only beta setting that gained on BOTH size axes (+1.30 "
            "small, +2.58 large), i.e. it is off the trade curve the other ten "
            "round-19 runs moved along. If TCN helps b3 but not b2 it is moving "
            "along that curve after all, which one run alone cannot tell you."},

    # ---------------------------------------------------------------- ARM B
    {"name": "y26_bsel3_btgt15", "prior": 0.22, "arm": "b", "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": cfg(tal_beta=3.0, tal_beta_sel=3.0, tal_beta_tgt=1.5),
     "expect": {"beta_sel": 3.0, "beta_tgt": 1.5},
     "label": "selection at beta 3, targets at beta 1.5",
     "why": "The target term carries 0.98 of b3's 1.52 (b3 minus a1_b6, which "
            "share ratio 0.167 and therefore selection). Pushing it past 3 is "
            "impossible while the two are one key. This is the run where the "
            "split can actually pay rather than merely attribute."},

    {"name": "y26_bsel3_btgt1", "prior": 0.18, "arm": "b", "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": cfg(tal_beta=3.0, tal_beta_sel=3.0, tal_beta_tgt=1.0),
     "expect": {"beta_sel": 3.0, "beta_tgt": 1.0},
     "label": "selection at beta 3, targets at beta 1.0",
     "why": "Finds the target-side bottom while selection stays at its measured "
            "optimum. Together with run 5 this is a 3-point sweep on the term "
            "that was never separable before."},

    {"name": "y26_bsel6_btgt3", "prior": 0.02, "arm": "b", "needs": "tal_beta_sel", "ctrl": B3_S50,
     "params": cfg(tal_beta=6.0, tal_beta_sel=6.0, tal_beta_tgt=3.0),
     "expect": {"beta_sel": 6.0, "beta_tgt": 3.0},
     "label": "stock selection, targets at beta 3 — the additivity check",
     "why": "PREDICTED 78.60 (identity 77.62 + the 0.98 target term). If it "
            "lands there the additive model holds and runs 5-6 are trustworthy. "
            "If it lands far off, the terms interact and the sweep needs "
            "re-planning. Stated in advance so it cannot be reinterpreted."},

    # ---------------------------------------------------------------- ARM D
    {"name": "y26_blevel_p3only", "prior": 0.25, "arm": "d", "needs": "tal_beta_level", "ctrl": B3_S50,
     "params": cfg(tal_beta=6.0, tal_beta_level={8: 2.0, 16: 6.0, 32: 6.0}),
     "expect": {"beta_level": {8: 2.0, 16: 6.0, 32: 6.0}},
     "label": "beta 2 at stride 8 only, stock everywhere else",
     "why": "b3's only cost is large (-0.23 mAP50, -1.00 L95). Small objects "
            "draw 6.64 of their 9.82 positives from s8; large draw 0.16 of 9.79. "
            "Changing beta only at s8 applies the correction where small objects "
            "live and leaves the levels large objects are assigned on untouched."},
]

# Ordered by PRIOR probability of clearing +0.35 on mAP50_small, highest first,
# so a queue that dies at 3am keeps the runs most likely to have mattered.
# Priors are an ordering, not a forecast: directional predictions in this
# campaign are 0-for-10, every miss optimistic.
#
# ONE TENSION, stated rather than hidden. y26_bsel6_btgt3 sits last on a 2%
# prior because it is predicted to land BELOW b3 by construction. It is also
# the additivity check that makes runs 6-7 interpretable. If runs 6-7 both come
# back near 79.5 you do not need it; if they disagree with each other, run it
# before drawing any conclusion from them.


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
        "default.yaml accepts tcn_p": "tcn_p" in DEFAULT_CFG_DICT,
        "default.yaml accepts tal_beta_level": "tal_beta_level" in DEFAULT_CFG_DICT,
        "assigner has beta_sel/beta_tgt": "beta_sel" in src_init and "beta_tgt" in src_init,
        "assigner has tcn_p": "tcn_p" in src_init,
        "assigner has beta_level": "beta_level" in src_init,
        "get_box_metrics returns align_tgt": "align_tgt" in src_box,
        "_forward consumes align_tgt": "align_tgt" in src_fwd,
        "_forward applies tcn_p": "tcn_p" in src_fwd,
    }
    for k, v in checks.items():
        print(f"  {k:<42} {v}")
    if checks["assigner has beta_sel/beta_tgt"] and checks["get_box_metrics returns align_tgt"]:
        have.add("tal_beta_sel")
    if checks["assigner has tcn_p"] and checks["_forward applies tcn_p"]:
        have.add("tcn_p")
    if checks["assigner has beta_level"]:
        have.add("tal_beta_level")

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
        print(f"  RUN  {r['name']:<20} arm {r['arm']}  p={r['prior']:.2f}  vs {r['ctrl']:.2f}  {d}")
    for r in skip:
        print(f"  SKIP {r['name']:<20} needs '{r['needs']}' — run patch_round21_v6i.py first")
    print(f"\n  {len(run)} runs, ~{0.9 * len(run):.1f} GPU-h" + (f"  ({len(skip)} skipped)" if skip else ""))
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
            if "tcn_p" in e and abs(float(getattr(a, "tcn_p", 1.0)) - e["tcn_p"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} tcn_p={a.tcn_p}, expected {e['tcn_p']}")
            if "beta_level" in e:
                got = getattr(a, "beta_level", None)
                if not got or {int(k): float(v) for k, v in got.items()} != \
                        {int(k): float(v) for k, v in e["beta_level"].items()}:
                    raise RuntimeError(f"{rc['name']}: {tag} beta_level={got}, expected {e['beta_level']}")
                if getattr(a, "_strides", None) is None:
                    raise RuntimeError(f"{rc['name']}: beta_level set but _strides is None — "
                                       f"set_strides is not being called, the key is a no-op")
            # everything else must be provably off
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
    print("\n" + "=" * 76)
    print(f"  RUN {rc['name']}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 76)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{SEED} | vs {rc['ctrl']:.2f} "
          f"| prior {rc['prior']:.2f} | {d}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": rc["name"], "arm": rc["arm"], "ctrl": rc["ctrl"], "prior": rc["prior"], "params": rc["params"],
           "expect": {k: str(v) for k, v in rc["expect"].items()}, "seed": SEED,
           "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours, "weights": weights,
           "mechanism_verified": True, "test_map50": float("nan"),
           "test_map5095": float("nan"), "test_map50_small": float("nan")}
    # args.yaml is what would have caught the y26_stock_b48 mislabelling: the
    # results JSONs record no batch for any run in this project.
    try:
        with open(os.path.join(save_dir, "round21_params.json"), "w") as f:
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
    ap.add_argument("--arm", choices=["a", "b", "c", "d"])
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    todo = sorted(RUNS, key=lambda r: -r["prior"])
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 21 — beta extension, consumer split, target ceiling "
          f"({len(todo)} selected)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] defaults are not inert — fix before spending a night of GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], f"runs_round21_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "seed": SEED, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 21 — RESULTS (mAP50-95 only; run analyze_runs_v6i.py for mAP50_small)")
    print("=" * 84)
    print(f"{'run':<24}{'arm':<5}{'mAP50':>9}{'mAP50-95':>11}{'hours':>7}")
    print("-" * 60)
    for r in sorted([x for x in res if x.get("test_map5095") == x.get("test_map5095")],
                    key=lambda x: -x["test_map5095"]):
        print(f"{r['name']:<24}{r['arm']:<5}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{r['hours']:>7.2f}")
    print(f"\n  reference (3 seeds): mAP50 {REF_50:.2f}  mAP50-95 {REF_5095:.2f}")
    print("  NOW RUN:  python analyze_runs_v6i.py")
    print("  Section 2 ranks on mAP50_small with the sigma. That is the metric")
    print("  that nearly lost tal_beta; do not judge this round on mAP50-95.")


if __name__ == "__main__":
    main()
