#!/usr/bin/env python3
r"""
YOLO26 ROUND 18 — THE tal_beta AXIS (stock yolo26s, b82)

Six runs, ~5.2 GPU-h. All b82 on the stock 3-level graph, batch matched to
y26_identity.

PRIMARY METRIC IS mAP50 AND mAP50_small, NOT mAP50-95. That is what this
project reports, and it changes which mechanisms look good — see below.


=============================================================================
THE HOLE: tal_beta WAS NEVER SWEPT
=============================================================================
align_metric = score^alpha * IoU^beta. Roughly 100 runs varied `tal_beta_small`
while holding `tal_beta = 6.0`, the stock value. Only two runs ever moved the
GLOBAL beta, and both sit at the top of the mAP50_small table:

    config        beta   beta_small   mAP50   mAP50_small
    scb_b4s2       4.0      2.0       81.03      78.65     <- best small
    beta4          4.0       -        80.95      78.17
    scb_b3         6.0      3.0       80.75      78.02
    scb_s2         6.0      2.0       80.70      77.53
    identity       6.0       -        80.18      77.30
    scb_s4         6.0      4.0       80.25      77.13

At beta=6 the beta_small sweep is non-monotone and inside noise
(77.13 / 77.53 / 78.02). At beta=4 BOTH points sit above every beta=6 config.
Two independent runs agreeing on direction is more than any other axis in this
campaign has produced.

Reading it mechanically: lowering beta makes the assignment trust IoU less
overall. On a dataset where 60% of instances are small — and where IoU is a
high-variance ranking signal on small boxes — that is the correction the
dataset analysis argued for, applied globally rather than only below a size
threshold. The campaign swept the conditioning and left the level alone.

Runs 1-2 extend the axis (beta 3.0, 2.0). Runs 3-4 re-apply the conditioning at
the new level. Run 5 pairs the leader with SBB, which it has never met. Run 6
tests the complementary knob.


=============================================================================
CALIBRATION
=============================================================================
Round 16/17 measured pooled within-config sd = 0.24 on mAP50-95 (n=3 x 3
configs). On mAP50 the two available pairs give ~0.20, which is provisional.

    identity      mAP50  80.18 / 79.79     range 0.39
    scb3_sbb50    mAP50  80.86 / 80.85     range 0.01

Treat anything under +0.4 on mAP50 as unresolved at n=1. The bar to beat is
scb_b4s2: 81.03 mAP50 / 78.65 mAP50_small, itself n=1.

Nine directional predictions have been made in this campaign and nine were
falsified, every one optimistic. Nothing above is a prediction. What makes this
round worth running is that the axis has TWO measured points that agree, which
no falsified proposal here ever had.


=============================================================================
SIZE BUCKETS COME FROM THE COCO PASS
=============================================================================
ultralytics `val()` returns box.map50 / box.map only. mAP50_small is produced
by CocoEvalAllFolders_luggage.py. This runner records the overall numbers
inline; run the COCO eval afterwards and read mAP50_small from its JSON before
drawing any conclusion — the whole point of this round is the small bucket.

Also add the truncation guard to that script first. y26_scb3_sbb50_scale75
entered the round-17 table as a 40-epoch model evaluated against 70-epoch ones.

    Usage:
        python run_yolo26_round18_v6i.py                # all six, in order
        python run_yolo26_round18_v6i.py --arm beta     # runs 1-4
        python run_yolo26_round18_v6i.py --arm combo    # runs 5-6
        python run_yolo26_round18_v6i.py y26_b3         # one by name
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
PROJECT_DIR = "runs_yolo26_round18_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches y26_base_rep and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

SD50 = 0.20  # provisional: 2 pairs on mAP50. mAP50-95 sd is 0.24 from n=3 x 3.
CTRL_STOCK_50 = 79.99  # y26_identity   2-seed mAP50 mean (80.18 / 79.79)
CTRL_STOCK_S50 = 77.30  # y26_identity   mAP50_small, seed 0
CTRL_B4S2_50 = 81.03  # y26_scb_b4s2   the bar,  n=1
CTRL_B4S2_S50 = 78.65  # y26_scb_b4s2   mAP50_small, n=1
CTRL_BETA4_50 = 80.95  # y26_beta4      beta=4 alone, n=1

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
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
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    # ------------------------------------------------------------ THE AXIS
    {"name": "y26_b3", "arm": "beta", "seed": 0, "ctrl": CTRL_BETA4_50,
     "params": cfg(tal_beta=3.0),
     "expect": {"alpha": 0.5, "beta": 3.0, "beta_small": None, "sbb": 0.0},
     "label": "tal_beta 6.0 -> 3.0, no size conditioning — the next point on the axis",
     "why": "beta has exactly two measured values, 6.0 (everywhere) and 4.0 (y26_beta4, "
            "80.95 mAP50 / 78.17 small), and the 4.0 point beats every beta=6 config on "
            "small. This is the cheapest test of whether that was a direction or a lucky "
            "draw, and it isolates the LEVEL from the conditioning that has been swept "
            "instead of it for a hundred runs."},

    {"name": "y26_b2", "arm": "beta", "seed": 0, "ctrl": CTRL_BETA4_50,
     "params": cfg(tal_beta=2.0),
     "expect": {"alpha": 0.5, "beta": 2.0, "beta_small": None, "sbb": 0.0},
     "label": "tal_beta 6.0 -> 2.0 — find where the axis turns over",
     "why": "Two points make a direction, three make a curve. If 3.0 improves on 4.0 and "
            "2.0 does not, the optimum is bracketed and the axis is finished in one "
            "night. If 2.0 is still climbing, IoU is being over-trusted far more than "
            "anyone assumed and that is the finding. A monotone axis is also the only "
            "case in this project where extrapolation was ever justified."},

    {"name": "y26_b4s1", "arm": "beta", "seed": 0, "ctrl": CTRL_B4S2_50,
     "params": cfg(tal_beta=4.0, tal_beta_small=1.0, tal_beta_ref_px=64.0),
     "expect": {"alpha": 0.5, "beta": 4.0, "beta_small": (1.0, 64.0), "sbb": 0.0},
     "label": "beta 4.0, beta_small 1.0 — stronger conditioning at the winning level",
     "why": "b4s2 (beta_small=2.0) is the campaign's best mAP50_small. beta_small=1.0 has "
            "never been run at any level. If the conditioning gain scales with the gap "
            "between beta and beta_small, this is the direction; if it collapses, the "
            "conditioning is a threshold effect rather than a slope, which is equally "
            "useful and retires half the SCB narrative."},

    {"name": "y26_b3s15", "arm": "beta", "seed": 0, "ctrl": CTRL_B4S2_50,
     "params": cfg(tal_beta=3.0, tal_beta_small=1.5, tal_beta_ref_px=64.0),
     "expect": {"alpha": 0.5, "beta": 3.0, "beta_small": (1.5, 64.0), "sbb": 0.0},
     "label": "beta 3.0, beta_small 1.5 — b4s2's ratio, shifted down the axis",
     "why": "Holds the beta/beta_small RATIO at 2.0 while moving the level, so together "
            "with run 1 it separates 'the level matters' from 'the ratio matters'. "
            "Neither has ever been isolated because the level was never varied."},

    # ------------------------------------------------------------ COMBINATIONS
    {"name": "y26_b4s2_sbb50", "arm": "combo", "seed": 0, "ctrl": CTRL_B4S2_50,
     "params": cfg(tal_beta=4.0, tal_beta_small=2.0, tal_beta_ref_px=64.0,
                   sbb_q=0.5, sbb_invert=True),
     "expect": {"alpha": 0.5, "beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.5},
     "label": "b4s2 + SBB 0.5 inv — the leader has never met the other working mechanism",
     "why": "SBB is worth +0.87 on mAP50 independently (scb3_sbb50, n=3) and every SBB "
            "pairing so far used beta=6.0. This is the one combination in the file with "
            "two independently positive ingredients. Note the campaign record on "
            "combining two positives is 1 for 4, so a null here is the modal outcome — "
            "but it is the modal outcome of the highest-value cell."},

    {"name": "y26_a025_b4s2", "arm": "combo", "seed": 0, "ctrl": CTRL_B4S2_50,
     "params": cfg(tal_alpha=0.25, tal_beta=4.0, tal_beta_small=2.0, tal_beta_ref_px=64.0),
     "expect": {"alpha": 0.25, "beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.0},
     "label": "tal_alpha 0.5 -> 0.25 on top of b4s2 — the complementary knob",
     "why": "alpha has only ever been 0.5 and 0.75 (y26_alpha075, +0.35). Lowering beta "
            "reduces the IoU exponent; lowering alpha reduces the score exponent, which "
            "raises IoU's RELATIVE weight — the opposite correction. If lowering beta "
            "works because the absolute exponents are too high, this should also help. "
            "If it works because the score/IoU BALANCE is wrong, this should hurt. The "
            "two outcomes are mechanically distinguishable, which is why it is here."},
]


def preflight(todo):
    """No new loss.py keys this round — every parameter already exists upstream.

    Still asserts the assigner exposes alpha/beta, because a silently ignored key is
    how rounds 4-6 lost ten runs: default.yaml accepted it, the header printed it,
    and the consumer never read it.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False
    probe = TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.25, beta=3.0)
    ok = abs(float(probe.alpha) - 0.25) < 1e-6 and abs(float(probe.beta) - 3.0) < 1e-6
    print(f"  {'TaskAlignedAssigner stores alpha/beta':<44} {ok}")
    if not ok:
        print("\n  [ABORT] the assigner is not reading alpha/beta — nothing here would run.")
        return False

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  {r['name']:<18} {r['arm']:<6} seed{r['seed']}  vs {r['ctrl']:.2f} mAP50  |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.87 * len(todo):.1f} GPU-h")
    print(f"  bar: {CTRL_B4S2_50:.2f} mAP50 / {CTRL_B4S2_S50:.2f} mAP50_small   "
          f"(y26_scb_b4s2, n=1)   sd50 ~{SD50}")
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that alpha, beta and beta_small are what was requested."""
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

        # ---- alpha and beta are the experiment this round
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if abs(float(a.alpha) - e["alpha"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected {e['alpha']}")
            if abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
        seen.append(f"alpha={e['alpha']} beta={e['beta']} on BOTH branches")

        # ---- beta_small: absence is asserted as strictly as presence
        bs = e["beta_small"]
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if bs is None:
                if a.scb_enabled():
                    raise RuntimeError(f"{rc['name']}: {tag} SCB live (beta_small={a.beta_small}) "
                                       f"but this run varies the LEVEL only")
            else:
                if not a.scb_enabled():
                    raise RuntimeError(f"{rc['name']}: {tag} SCB not live, expected {bs}")
                if abs(float(a.beta_small) - bs[0]) > 1e-6 or abs(float(a.beta_ref_px) - bs[1]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} SCB=({a.beta_small}, {a.beta_ref_px}), "
                                       f"expected {bs}")
        seen.append("beta_small off" if bs is None else f"beta_small={bs[0]}@{bs[1]}px")

        # ---- SBB
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if abs(float(b.sbb_q) - e["sbb"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {e['sbb']}")
        if e["sbb"] > 0.0:
            if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} "
                                   f"o2o={b2.sbb_sign:+.0f} — they must be OPPOSITE")
            if float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: one2one sbb_sign={b2.sbb_sign:+.0f}; the arm "
                                   f"that won is invert=True -> one2one leans SMALL (sign<0)")
            seen.append(f"SBB q={e['sbb']} o2m={b1.sbb_sign:+.0f}(large) o2o={b2.sbb_sign:+.0f}(small)")
        else:
            seen.append("SBB off")

        # ---- everything else must be provably off
        for a in (a1, a2):
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            if a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: SBAL is live but was not requested")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")
            if b.snl1_enabled():
                raise RuntimeError(f"{rc['name']}: SNL1 is live on {tag} but was not requested")
            if float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: NWD is live on {tag} but was not requested")
        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this file is the stock 3-level graph")
        if abs(float(h.cls) - 0.5) > 1e-6 or abs(float(h.box) - 7.5) > 1e-6:
            raise RuntimeError(f"{rc['name']}: gains moved (box={h.box} cls={h.cls}); "
                               f"this round varies the assignment metric only")
        if abs(float(trainer.args.multi_scale)) > 1e-6:
            raise RuntimeError(f"{rc['name']}: multi_scale is live; it measured -1.03 (4.3 sd)")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name, seed = rc["name"], rc["seed"]
    print()
    print("=" * 78)
    print(f"  RUN {name}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={seed}  control={rc['ctrl']:.2f} mAP50")
    print(f"  differs from _ALL_OFF: {({k: v for k, v in rc['params'].items() if _ALL_OFF.get(k, '__') != v})}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(rc["params"])
    results = model.train(**kw)
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": rc["arm"], "ctrl": rc["ctrl"], "params": rc["params"],
           "expect": {k: list(v) if isinstance(v, tuple) else v for k, v in rc["expect"].items()},
           "seed": seed, "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH,
           "hours": hours, "weights": weights, "mechanism_verified": True,
           "epochs_completed": EPOCHS,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # *_params.json so the eval script's glob binds metrics to a CONFIG, not to
    # directory order — round 16 was mis-evaluated for exactly this reason.
    try:
        with open(os.path.join(save_dir, "round18_params.json"), "w") as f:
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
    print("  ROUND 18 — RESULTS  (mAP50; small buckets need the COCO pass)")
    print("=" * 78)
    print(f"{'run':<20}{'beta':>6}{'b_small':>9}{'mAP50':>9}{'vs bar':>9}{'vs stock':>10}{'hours':>7}")
    print("-" * 70)
    print(f"{'y26_identity (n=2)':<20}{6.0:>6.1f}{'-':>9}{CTRL_STOCK_50:>9.2f}"
          f"{CTRL_STOCK_50 - CTRL_B4S2_50:>+9.2f}{0.0:>+10.2f}{'-':>7}")
    print(f"{'y26_beta4':<20}{4.0:>6.1f}{'-':>9}{CTRL_BETA4_50:>9.2f}"
          f"{CTRL_BETA4_50 - CTRL_B4S2_50:>+9.2f}{CTRL_BETA4_50 - CTRL_STOCK_50:>+10.2f}{'-':>7}")
    print(f"{'y26_scb_b4s2 (bar)':<20}{4.0:>6.1f}{2.0:>9.1f}{CTRL_B4S2_50:>9.2f}"
          f"{0.0:>+9.2f}{CTRL_B4S2_50 - CTRL_STOCK_50:>+10.2f}{'-':>7}")
    print("-" * 70)
    for r in sorted(ok, key=lambda x: -x["test_map50"]):
        v = r["test_map50"] * 100
        bs = r["params"]["tal_beta_small"]
        print(f"{r['name']:<20}{r['params']['tal_beta']:>6.1f}"
              f"{(f'{bs:.1f}' if bs else '-'):>9}{v:>9.2f}"
              f"{v - CTRL_B4S2_50:>+9.2f}{v - CTRL_STOCK_50:>+10.2f}{r['hours']:>7.2f}")

    print("\n  READ IT")
    got = {r["name"]: r["test_map50"] * 100 for r in ok}
    curve = [(6.0, CTRL_STOCK_50), (4.0, CTRL_BETA4_50)]
    for n, b in (("y26_b3", 3.0), ("y26_b2", 2.0)):
        if n in got:
            curve.append((b, got[n]))
    curve.sort(key=lambda x: -x[0])
    if len(curve) >= 3:
        print("    beta axis, no conditioning:  " +
              "   ".join(f"{b:.0f}->{v:.2f}" for b, v in curve))
        best = max(curve, key=lambda x: x[1])
        if best[0] in (3.0, 2.0) and best[1] - CTRL_BETA4_50 >= 2 * SD50:
            print(f"    beta={best[0]:.0f} beats 4.0 by more than 2 sd. The axis is real and the")
            print("    campaign swept the wrong parameter for a hundred runs. Seed it, then")
            print("    re-run the conditioning sweep at the new level.")
        elif best[0] == 6.0:
            print("    Stock beta is best. The two beta=4 results were draws, and the")
            print("    assignment metric is already tuned. That closes the axis.")
        else:
            print("    beta=4 remains the peak. The axis has an interior optimum near 4.0;")
            print("    report it as a tuned value and stop.")

    for n in ("y26_b4s1", "y26_b3s15", "y26_b4s2_sbb50", "y26_a025_b4s2"):
        if n in got:
            d = got[n] - CTRL_B4S2_50
            verdict = "beats the bar" if d >= 2 * SD50 else (
                "below the bar" if d <= -2 * SD50 else "inside the null band")
            print(f"    {n:<18} {got[n]:.2f}  {d:+.2f} vs b4s2   {verdict}")

    print("\n    Nothing here is reportable until the COCO pass gives mAP50_small and the")
    print(f"    top two configs have a second seed. The bar is {CTRL_B4S2_S50:.2f} small, and it")
    print("    is itself n=1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    ap.add_argument("--arm", choices=["beta", "combo"])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    todo = RUNS
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if a.seed != SEED:
        todo = [{**r, "seed": a.seed, "name": f"{r['name']}_s{a.seed}"} for r in todo]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 18 — the tal_beta axis ({len(todo)} runs)")
    print("  " + "  ".join(r["name"] for r in todo))
    print("=" * 84)
    if not preflight(todo):
        return

    res, out_path = [], f"{PROJECT_DIR}_results.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
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
