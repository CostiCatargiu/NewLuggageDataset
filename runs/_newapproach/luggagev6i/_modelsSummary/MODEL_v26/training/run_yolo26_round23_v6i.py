#!/usr/bin/env python3
r"""
YOLO26 ROUND 23 — one seed on each config the paper intends to report

Five runs, ~4.5 GPU-h, b82, stock yolo26s.pt, 640, 70 ep. Four are repeats at
seed 1 of configs already run at seed 0; the fifth closes the beta curve.

PRIMARY METRIC IS mAP50_small.

    REFERENCE, 3 seeds of y26_identity   small 77.62   mAP50 80.40   m5095 55.30
    pooled sd (df=7)                     small  0.346  m5095 0.374


=============================================================================
WHY THIS ROUND IS SEEDS AND NOT ANOTHER CONFIG
=============================================================================
Round 22b put ten (beta_sel, beta_tgt) cells on the plane:

    bsel05_btgt2 79.13   bsel05_btgt3 78.84   bsel1_btgt6  78.81
    bsel05_btgt1 78.74   bsel1_btgt3  78.71   bsel2_btgt1  78.70
    bsel1_btgt2  78.67   bsel2_btgt3  78.64   b05          78.57
    bsel1_btgt15 78.48

    sd of the ten = 0.176.  The noise floor is 0.346.

The whole design plane varies at HALF the measurement error. That is a
mechanism with one degree of freedom whose value has been found, and it is the
third plateau in a row: uniform beta in [1,3] (sd 0.202), the split plane
(sd 0.176), and branch balance (ten runs, all at or below their controls).
Another cell draws from the same distribution. The loss axis is closed.

What is NOT closed is that NOT ONE reported config has ever been repeated.
Every seed group in this project is a null:

    identity     3 seeds    77.30  77.75  77.81
    scb3_sbb50   3 seeds    77.92  77.61  77.67
    cls075       3 seeds    78.26  77.65  77.14   <- +0.65 collapsed to +0.09
    cls065       2 seeds    77.72  77.76

cls075 is the warning. It read +0.65 at seed 0 and survived exactly until it
was repeated. y26_b1 at 79.30 is a SINGLE DRAW and is the number the paper
wants to headline.


=============================================================================
WHAT ONE SEED BUYS, AND WHAT IT DOES NOT
=============================================================================
Stated plainly so nobody reads more out of this round than is in it.

IT ANSWERS:  was 79.30 a high draw?
    A config mean of n=2 against the identity mean of n=3 has
        SE = 0.346 * sqrt(1/2 + 1/3) = 0.316   ->  2 SE = 0.63,  3 SE = 0.95
    b1's +1.67 survives that unless seed 1 lands below ~78.4. If it does land
    there, the headline is not 79.30 and it is far better to learn it now.

IT DOES NOT ANSWER:  is b1 better than b3, or than bsel05_btgt2?
    Two configs at n=2 each differ with SE = 0.346 * sqrt(1/2 + 1/2) = 0.346,
    so separating them needs a gap of ~0.69. They currently differ by 0.16 and
    0.17. NO seed budget realistically available will separate this cluster,
    because the cluster is flat -- that is the round 22b finding, not a
    shortfall of this round.

The honest paper claim these five runs support is therefore about the FAMILY,
not the point: "beta reweighting yields +1.5 to +1.7 mAP50 on small objects,
insensitive to parameterisation across sixteen configurations." That is a
stronger and more defensible sentence than naming a winner that is one draw.

If seed 1 confirms the level, a third seed on b1 alone is ~0.9 GPU-h and turns
n=2 into a quotable sd. Do that AFTER seeing these, not before.


=============================================================================
THE FOUR REPEATS
=============================================================================
                        seed 0   why this one and not another
    y26_b1               79.30   campaign maximum on small AND on mAP50 (81.80)
                                 AND both-up on the size trade (+2.05 large).
                                 It is the config the paper wants to name.
    y26_bsel05_btgt2     79.13   round 22b's top cell. Ties b1 and b3 within
                                 0.17. If the tie is real all three move
                                 together; if 22b's winner was a high draw off
                                 a flat plane, this is where it shows.
    y26_b3               79.14   the longest-standing result in the campaign
                                 and the control that rounds 18-19 were quoted
                                 against. Ten combos were judged against this
                                 single number.
    y26_bsel1_btgt6      78.81   the one unexplained result in 22b: large50
                                 85.45, second-highest of all 145 runs, while
                                 also +1.19 on small. Best both-axes point ever
                                 measured here -- but the within-round sd on
                                 large is 2.15 against small's 0.176, so this
                                 may be nothing. Large has ~1/20th the instance
                                 count and is the noisiest metric in the set.

Seeds are 1, matching the _s1 convention of the existing groups.


=============================================================================
THE FIFTH RUN — y26_b0, CLOSING THE CURVE
=============================================================================
Planned in round 22, never executed. beta = 0 makes the alignment metric
score^alpha alone, so SELECTION ignores IoU entirely.

    plateau measured so far:  beta 0.5 .. 3 all within 0.65
    beta 4  +0.55        beta 6  nothing (identity)

    If b0 holds the plateau -> assignment's IoU term contributes nothing on
    60%-small imagery. A strong, citable negative result.
    If b0 collapses        -> the plateau has a measured floor and the curve is
    closed at both ends.

Both outcomes are publishable, which is not true of any performance run that
could take this slot. It is the only remaining run that adds an ARGUMENT rather
than a number.

ONE HONEST CAVEAT, because the claim above is easy to overstate. At beta = 0
selection is IoU-free, but the TARGET is not: tal.py:224-226 computes

    pos_overlaps      = (overlaps * mask_pos).amax(-1)
    norm_align_metric = align_tgt * pos_overlaps / pos_align_metrics

and pos_overlaps is the RAW IoU regardless of beta. So b0 removes IoU from
assignment, not from the loss. Write it up that way. (torch gives 0.0**0.0 = 1,
so zero-IoU anchors get metric 1.0 in the selection term -- which is the
intended semantics here, but it is a thing to know rather than discover.)


=============================================================================
EXECUTION ORDER
=============================================================================
Ordered by REGRET, not by expected gain -- a seed run has no expected gain by
construction, so the round-21/22 prior does not apply. The question is which
result would be worst to lose at 3am.

    1  y26_b1_s1            the headline. Nothing else matters if this moves.
    2  y26_bsel05_btgt2_s1  the tie. Two of three top numbers become n=2.
    3  y26_b3_s1            the control ten other runs were quoted against.
    4  y26_bsel1_btgt6_s1   the large-object anomaly, the only open question.
    5  y26_b0               exploratory; the argument survives losing it a day.

Runs 1, 3 and 5 need NO patch (plain tal_beta). Runs 2 and 4 need
patch_round21_v6i.py; if it is absent they are SKIPPED with a message, not
silently run as stock.

    Usage:
        python run_yolo26_round23_v6i.py                  # all five, in order
        python run_yolo26_round23_v6i.py --preflight      # print the plan only
        python run_yolo26_round23_v6i.py --arm seed       # seed | curve
        python run_yolo26_round23_v6i.py y26_b1_s1        # one by name

    AFTERWARDS: add the new pairs to SEED_GROUPS in analyze_runs_v6i.py, or
    section 1 will keep reporting a noise floor built only from null configs.
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
PROJECT_DIR = "runs_yolo26_round23_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 1                              # THIS ROUND IS SEED 1. per-run override below.
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False            # y26_p2k2_hi was lost to exist_ok=True

# 3-seed means. Never the seed-0 draw.
REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30
SD_S50, SD_5095 = 0.346, 0.374

# The seed-0 draws being repeated. Each is n=1 -- that is the whole point.
S0 = {
    "y26_b1":            {"small": 79.30, "map50": 81.80, "m5095": 54.84, "large": 82.71},
    "y26_bsel05_btgt2":  {"small": 79.13, "map50": 81.48, "m5095": 54.63, "large": 80.70},
    "y26_b3":            {"small": 79.14, "map50": 81.31, "m5095": 55.30, "large": 80.42},
    "y26_bsel1_btgt6":   {"small": 78.81, "map50": 81.42, "m5095": 54.73, "large": 85.45},
}

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
    rounds 21 and 22b were configured."""
    return cfg(tal_beta=sel, tal_beta_sel=sel, tal_beta_tgt=tgt)


RUNS = [
    # ------------------------------------------------------- REPEATS (seed 1)
    {"name": "y26_b1_s1", "regret": 1, "arm": "seed", "repeat_of": "y26_b1",
     "seed": 1, "needs": None,
     "params": cfg(tal_beta=1.0), "expect": {"beta": 1.0},
     "label": "beta 1.0, seed 1 — repeat of the campaign maximum",
     "why": "79.30 on small, 81.80 on mAP50, +2.05 on large: the best number "
            "this project has and it is one draw. cls075 read +0.65 at seed 0 "
            "and fell to +0.09 across three. Until this run exists the paper "
            "cannot name a configuration."},

    {"name": "y26_bsel05_btgt2_s1", "regret": 2, "arm": "seed",
     "repeat_of": "y26_bsel05_btgt2", "seed": 1, "needs": "tal_beta_sel",
     "params": split(0.5, 2.0), "expect": {"beta": 0.5, "beta_sel": 0.5, "beta_tgt": 2.0},
     "label": "sel 0.5 x tgt 2, seed 1 — repeat of round 22b's top cell",
     "why": "79.13, within 0.17 of b1 and 0.01 of b3. Either the top of the "
            "plane is genuinely flat at ~79.2, or 22b's winner is the high draw "
            "of ten samples from a distribution centred at 78.73 — which is "
            "exactly what a max over ten flat draws looks like."},

    {"name": "y26_b3_s1", "regret": 3, "arm": "seed", "repeat_of": "y26_b3",
     "seed": 1, "needs": None,
     "params": cfg(tal_beta=3.0), "expect": {"beta": 3.0},
     "label": "beta 3.0, seed 1 — repeat of the campaign's reference point",
     "why": "Rounds 18 and 19 quoted TEN combination runs against b3's single "
            "79.14, and all ten 'lost'. If that draw was high by 0.3 then some "
            "of those ten were level and the round-19 conclusion needs the "
            "qualifier. This is the cheapest audit of an already-drawn "
            "conclusion in the campaign."},

    {"name": "y26_bsel1_btgt6_s1", "regret": 4, "arm": "seed",
     "repeat_of": "y26_bsel1_btgt6", "seed": 1, "needs": "tal_beta_sel",
     "params": split(1.0, 6.0), "expect": {"beta": 1.0, "beta_sel": 1.0, "beta_tgt": 6.0},
     "label": "sel 1 x tgt 6, seed 1 — the large-object anomaly",
     "why": "large50 85.45 is second of 145 runs, and unlike the run above it "
            "(swa_a06_b15, +0.34 small) this one is +1.19 on small too — the "
            "best both-axes point measured. But round 22b's within-round sd on "
            "large is 2.15, so 85.45 sits under 2 sd from the plane mean. This "
            "run decides whether it is a finding or a wide metric."},

    # ------------------------------------------------------- CURVE (new run)
    {"name": "y26_b0", "regret": 5, "arm": "curve", "repeat_of": None,
     "seed": 0, "needs": None,
     "params": cfg(tal_beta=0.0), "expect": {"beta": 0.0},
     "label": "beta 0 — IoU removed from ASSIGNMENT entirely, NO PATCH NEEDED",
     "why": "The plateau spans 0.5..3 and nobody has looked at the endpoint. At "
            "beta=0 selection ranks on score^alpha alone. Holds the plateau -> "
            "the IoU term contributes nothing on this imagery; collapses -> the "
            "curve has a measured floor. Both are publishable. NOTE the target "
            "ceiling still uses raw IoU (tal.py:224-226), so this is IoU-free "
            "assignment, not an IoU-free loss. Seed 0 to match the curve."},
]


def order(runs):
    """By regret. A seed run has no expected gain, so the round-21/22 prior
    ordering is meaningless here; the question is what is worst to lose."""
    return sorted(runs, key=lambda r: r["regret"])


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
        base = S0.get(r["repeat_of"] or "", {}).get("small")
        vs = f"seed0 {base:.2f}" if base else "new run"
        print(f"  RUN  {r['name']:<22} arm {r['arm']:<5} regret {r['regret']}  "
              f"seed {r['seed']}  {vs:<12} {d}")
    for r in skip:
        print(f"  SKIP {r['name']:<22} needs '{r['needs']}' — run patch_round21_v6i.py first")
    print(f"\n  {len(run)} runs, ~{0.9 * len(run):.1f} GPU-h" + (f"  ({len(skip)} skipped)" if skip else ""))
    if skip:
        print("  note: y26_b1_s1, y26_b3_s1 and y26_b0 need no patch, so the "
              "headline repeat completes regardless.")
    return ok, {r["name"] for r in run}


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Assert at epoch 2 that the mechanism is LIVE and nothing else is.

    Epoch 2, not epoch 1: anything recomputed per epoch (the o2m decay is the
    known case) still matches stock on the first pass.
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
                else:
                    # a uniform run must NOT have a stale split left installed
                    got = getattr(a, attr, None)
                    if got is not None:
                        raise RuntimeError(f"{rc['name']}: {tag} {attr}={got} but this is a "
                                           f"UNIFORM run — a stale split would silently "
                                           f"change the config being seeded")
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
        # a seed repeat that trains a DIFFERENT config than the one it repeats is
        # the single worst failure available to this round -- it would be scored
        # as a seed of something it is not.
        if rc["repeat_of"]:
            print(f"  [guard] certified as a seed-{rc['seed']} repeat of {rc['repeat_of']} "
                  f"(seed 0 small = {S0[rc['repeat_of']]['small']:.2f})")
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
    base = S0.get(rc["repeat_of"] or "", {}).get("small")
    vs = f"seed 0 drew {base:.2f} on small" if base else "no prior draw — new point"
    print(f"  b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{seed} | {vs} | {d}\n")
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
    out = {"name": rc["name"], "arm": rc["arm"], "regret": rc["regret"],
           "repeat_of": rc["repeat_of"], "params": rc["params"],
           "expect": {k: str(v) for k, v in rc["expect"].items()}, "seed": seed,
           "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours, "weights": weights,
           "mechanism_verified": True, "test_map50": float("nan"),
           "test_map5095": float("nan"), "test_map50_small": float("nan")}
    # args.yaml is what would have caught the y26_stock_b48 mislabelling: the
    # results JSONs record no batch for any run in this project.
    try:
        with open(os.path.join(save_dir, "round23_params.json"), "w") as f:
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
    ap.add_argument("--arm", choices=["seed", "curve"])
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
    print(f"  YOLO26 ROUND 23 — one seed on every config the paper reports "
          f"({len(todo)} selected)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] defaults are not inert — fix before spending a night of GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_round23_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "seed_round": True, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 23 — RESULTS (mAP50-95 only; score mAP50_small before concluding)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<22} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<22} seed {r['seed']}  mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print("  1. export the test JSON, then run analyze_runs_v6i.py")
    print("  2. ADD THE NEW PAIRS TO SEED_GROUPS in analyze_runs_v6i.py, e.g.")
    print("       'b1': ['y26_b1', 'y26_b1_s1'],  'b3': ['y26_b3', 'y26_b3_s1'], ...")
    print("     otherwise the noise floor stays built from null configs only.")
    print("  3. read the PAIR MEAN, not either draw. Config mean (n=2) vs the")
    print("     identity mean (n=3) has SE 0.316 -> 2 SE = 0.63, 3 SE = 0.95.")
    print("  4. do NOT try to rank b1 vs b3 vs bsel05_btgt2 off n=2. Separating")
    print("     them needs a 0.69 gap; they differ by 0.16. The cluster is flat.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
