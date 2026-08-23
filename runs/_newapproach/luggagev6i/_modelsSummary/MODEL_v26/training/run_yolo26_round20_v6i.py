#!/usr/bin/env python3
r"""
YOLO26 ROUND 20 — NWD ON A CORRECTED BASE (3 runs, loss only, b82)

~2.8 GPU-h, seed 0. The last loss mechanism with an unfinished dose curve.

    BASE      y26_b3     tal_beta=3.0    81.31 / 79.14 small / 55.30
              y26_b2     tal_beta=2.0    81.56 / 78.92 small / 55.16
    BASELINE  y26_identity               80.18 / 77.30 small / 55.24
    sd on mAP50_small ~0.27


=============================================================================
WHY NWD, AND WHY IT WAS WRONGLY CLOSED
=============================================================================
                 mAP50   small    large   mAP50-95
    identity     80.18   77.30    81.75     55.24
    nwd25        80.37   77.94    78.48     55.05
    nwd50        80.62   78.19    78.13     54.78

+0.89 on mAP50_small is the second-largest single-mechanism gain in the whole
campaign, behind only beta. It was retired because mAP50-95 fell 0.46 -- the
same metric that buried SWA (y26_sqrt0703, a b32 run scored on mAP50-95) and
that nearly buried beta itself. Three mechanisms, one recurring error.

NWD is also the only untested mechanism that attacks beta's problem by a
DIFFERENT route. metrics.py:213 blends

    iou = (1 - NWD) * iou + NWD * exp(-wasserstein / nwd_c)

IoU is a RATIO, so a 3px error costs ~0.5 CIoU on a 15x25 box and ~0.05 on a
100px one; at beta=6 that is a ~47x swing in assignment weight from the same
absolute error. beta=3 DOWN-WEIGHTS that noisy signal. NWD REPLACES it with an
absolute distance that degrades identically at every scale. Different stage of
the same fix, so they are not obviously redundant -- but see below.


THE EVIDENCE AGAINST (it is already in the tree, and it is not weak)
--------------------------------------------------------------------
    SCB3 on a plain base      77.30 -> 78.02    +0.72
    SCB3 on an NWD base       78.05 -> 78.26    +0.21

NWD absorbs ~70% of a beta-style correction, which is what two fixes for the
same defect should do. Scaled onto beta=3 the expected residual is ~+0.26 --
inside the 0.27 sd. Honest prior on this round: ~25%, and this campaign's
directional predictions are 0-for-10, every miss optimistic.

It is worth 2.8 GPU-h anyway because the dose curve is MONOTONE and UNFINISHED
(0 -> 0.25 -> 0.5, still climbing) and nobody has run 0.75, on any base.


THE SIZE-TRADE FRAME (round 19's real lesson)
----------------------------------------------
Round 19 put ten mechanisms on the beta=3 base and all ten came in below it on
mAP50_small. The reason is that nearly everything in this loss is a TRADE:

    config            d small   d large
    b2                 +1.62     +1.49      <- gains on BOTH
    b3                 +1.84     -1.33
    b25                +1.58     +0.03
    b3_swa             +1.32     +0.54
    b3_sbb50           +1.04     +2.17
    nwd50              +0.89     -3.62      <- the steepest trade measured

Those ten runs were all moving ALONG one exchange curve that beta=3 already
sits near the small end of. Stacking two mechanisms that both move along it
lands on the same point or past it. NWD trades harder than anything else, so on
the beta=3 base it may simply overshoot -- which is exactly what SCB did at
beta=3 (-0.72).

b2 is the one config that gained on both axes, and it DOMINATES b25 outright
(78.92/83.24 vs 78.88/81.78). That is why run 3 puts NWD there instead: if the
trade overshoots from beta=3, it may still pay from beta=2.


  RUN 1  b3_nwd50    beta 3 + NWD 0.5     the ~+0.26 residual, measured directly
  RUN 2  b3_nwd75    beta 3 + NWD 0.75    the dose curve never turned over
  RUN 3  b2_nwd50    beta 2 + NWD 0.5     the base that gained on both axes

NWD 1.0 is deliberately NOT here. The same bbox_iou call feeds the assigner's
`overlaps` AND BboxLoss, so at NWD=1.0 box regression stops being an IoU loss
and becomes an exponential-distance loss. That is a different experiment, and
if it destabilises it says nothing about the dose curve.

CONFOUND, stated up front: one `nwd` key drives both the assignment metric and
the box loss, so a win here cannot be attributed to either. If any of these
clears +0.3, the follow-up is to split the key by consumer -- not to celebrate.

    Usage:
        python run_yolo26_round20_v6i.py                 # all three
        python run_yolo26_round20_v6i.py y26_b3_nwd50    # one by name
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
PROJECT_DIR = "runs_yolo26_round20_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

CTRL_B3_50, CTRL_B3_S50 = 81.31, 79.14    # y26_b3
CTRL_B2_50, CTRL_B2_S50 = 81.56, 78.92    # y26_b2
CTRL_ID_50, CTRL_ID_S50 = 80.18, 77.30    # y26_identity
CTRL_NWD_50, CTRL_NWD_S50 = 80.62, 78.19  # y26_nwd50, beta=6
NOISE_S50 = 0.27

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
NWD_C = 24.0  # unchanged from the round-11 runs, so the dose curve stays comparable
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_b3_nwd50", "ctrl": CTRL_B3_50, "ctrl_s": CTRL_B3_S50,
     "params": cfg(tal_beta=3.0, nwd=0.5, nwd_c=NWD_C),
     "expect": {"beta": 3.0, "nwd": 0.5},
     "label": "beta 3 + NWD 0.5 — the residual, measured directly",
     "why": "NWD is worth +0.89 on mAP50_small at beta=6 and was closed on mAP50-95. It "
            "fixes IoU's scale dependence by replacing the metric; beta fixes it by "
            "down-weighting the metric. The in-tree evidence says NWD absorbs ~70% of a "
            "beta-style correction (SCB3 falls from +0.72 to +0.21 on an NWD base), so "
            "the expected residual is ~+0.26 and this run is the direct test of it. A "
            "null closes the last loss mechanism with an open dose curve."},

    {"name": "y26_b3_nwd75", "ctrl": CTRL_B3_50, "ctrl_s": CTRL_B3_S50,
     "params": cfg(tal_beta=3.0, nwd=0.75, nwd_c=NWD_C),
     "expect": {"beta": 3.0, "nwd": 0.75},
     "label": "beta 3 + NWD 0.75 — the dose curve never turned over",
     "why": "0, 0.25, 0.5 gave 77.30, 77.94, 78.19 on mAP50_small: monotone and still "
            "climbing when the sweep stopped. No run at any base has gone past 0.5, so "
            "the optimum is asserted rather than measured. Paired with run 1 this is a "
            "dose contrast on the corrected base, so it reports a direction even if both "
            "points sit below b3."},

    {"name": "y26_b2_nwd50", "ctrl": CTRL_B2_50, "ctrl_s": CTRL_B2_S50,
     "params": cfg(tal_beta=2.0, nwd=0.5, nwd_c=NWD_C),
     "expect": {"beta": 2.0, "nwd": 0.5},
     "label": "beta 2 + NWD 0.5 — the only base that gained on both axes",
     "why": "b2 is the single config in the campaign that beats identity on small AND "
            "large (+1.62 / +1.49), it dominates b25 on both, and it holds the best "
            "overall mAP50 of any beta point at 81.56. NWD is the steepest size trade "
            "measured (-3.62 large), so from beta=3 it may overshoot the way SCB did; "
            "from beta=2, which is 0.22 short on small but 2.82 up on large, there is "
            "room to spend. This is the run where the trade can still pay."},
]


def preflight(todo):
    """Probe the CONSUMER, not the config surface.

    NWD is read in TWO places from ONE key -- the assigner's iou_calculation and
    BboxLoss -- so a patch that reaches only one of them would give a half-live run
    that still looks correct in the header.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    try:
        import torch as _t

        from ultralytics.utils.metrics import bbox_iou
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False

    # bbox_iou must actually blend. Two identical boxes: NWD pushes the similarity of a
    # SHIFTED pair up, because exp(-d/c) decays far slower than IoU does at small scale.
    a = _t.tensor([[100.0, 100.0, 120.0, 120.0]])
    b = _t.tensor([[103.0, 103.0, 123.0, 123.0]])  # 3px shift on a 20px box
    plain = float(bbox_iou(a, b, xywh=False, CIoU=True).squeeze())
    blend = float(bbox_iou(a, b, xywh=False, CIoU=True, NWD=0.5, nwd_c=NWD_C).squeeze())
    ok = blend > plain + 0.05
    print(f"  {'bbox_iou honours NWD (3px shift, 20px box)':<50} {ok}")
    print(f"  {'  CIoU':<50} {plain:.4f}")
    print(f"  {'  0.5 CIoU + 0.5 NWD':<50} {blend:.4f}")
    if not ok:
        print("\n  [ABORT] NWD is not reaching bbox_iou — copy metrics.py to this box.")
        return False

    print()
    for i, r in enumerate(todo, 1):
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  {i}. {r['name']:<16} vs {r['ctrl']:.2f}/{r['ctrl_s']:.2f}  |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.93 * len(todo):.1f} GPU-h")
    print(f"  bars: b3 {CTRL_B3_50:.2f}/{CTRL_B3_S50:.2f}   b2 {CTRL_B2_50:.2f}/"
          f"{CTRL_B2_S50:.2f}   sd_small ~{NOISE_S50}")
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that beta and NWD are live on BOTH consumers, nothing else is."""
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

        for tag, a in (("one2many", a1), ("one2one", a2)):
            if abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            if abs(float(a.alpha) - 0.5) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected the stock 0.5")

        # one key, two consumers: assignment metric AND box regression loss
        for tag, obj, where in (("one2many", a1, "assigner"), ("one2one", a2, "assigner"),
                                ("one2many", b1, "bbox_loss"), ("one2one", b2, "bbox_loss")):
            got = float(getattr(obj, "nwd", -1.0))
            if abs(got - e["nwd"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} {where}.nwd={got}, expected {e['nwd']}")
            c = float(getattr(obj, "nwd_c", -1.0))
            if abs(c - NWD_C) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} {where}.nwd_c={c}, expected {NWD_C}")

        # everything else provably off, so a win is attributable to beta x NWD alone
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB live on {tag}; subsumed by beta and "
                                   f"harmful at beta=3 (-0.72)")
            if a.snt_enabled() or a.tsh_enabled() or a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: SNT/TSH/SBAL live on {tag}, all closed")
            if getattr(a, "iou_kwargs", {}).get("CIoU") is not True:
                raise RuntimeError(f"{rc['name']}: iou_kwargs={a.iou_kwargs}, expected CIoU — "
                                   f"EIoU measured 79.31/76.63 and is closed")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.swa_enabled() or b.snl1_enabled() or float(b.sbb_q) != 0.0:
                raise RuntimeError(f"{rc['name']}: SWA/SNL1/SBB live on {tag} but not requested")
        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this is the stock 3-level graph")
        if abs(float(h.cls) - 0.5) > 1e-6 or abs(float(h.box) - 7.5) > 1e-6:
            raise RuntimeError(f"{rc['name']}: gains moved (box={h.box} cls={h.cls})")
        if abs(float(crit.final_o2m) - 0.1) > 1e-6 or not crit.o2m_decay:
            raise RuntimeError(f"{rc['name']}: blend moved from the stock 0.8 -> 0.1 decay")
        if abs(float(trainer.args.multi_scale)) > 1e-6:
            raise RuntimeError(f"{rc['name']}: multi_scale is live; it measured -1.03 (4.3 sd)")

        print(f"  [guard] beta={a1.beta} alpha={a1.alpha} on both branches")
        print(f"  [guard] NWD={e['nwd']} c={NWD_C} live on assigner AND bbox_loss, both branches")
        print(f"  [guard] SCB/SBB/SWA/SNT/TSH/SBAL/SNL1 off | CIoU base | box={h.box} cls={h.cls}")
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
          f"epochs={EPOCHS}  seed={SEED}  base={rc['ctrl']:.2f}/{rc['ctrl_s']:.2f}")
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
    out = {"name": name, "ctrl": rc["ctrl"], "ctrl_s": rc["ctrl_s"], "params": rc["params"],
           "seed": SEED, "expect": rc["expect"], "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE,
           "batch": BATCH, "hours": hours, "weights": weights, "mechanism_verified": True,
           "epochs_requested": EPOCHS,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # *_params.json so the eval script's glob binds metrics to a CONFIG, not to
    # directory order — round 16 was mis-evaluated twice for exactly that.
    try:
        with open(os.path.join(save_dir, "round20_params.json"), "w") as f:
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
    print("  ROUND 20 — RESULTS  (mAP50; mAP50_small needs the COCO pass)")
    print("=" * 78)
    print(f"{'run':<18}{'mAP50':>9}{'vs base':>9}{'vs stock':>10}{'mAP50-95':>10}{'hours':>7}")
    print("-" * 63)
    for nm, v, v95 in (("y26_identity", CTRL_ID_50, 55.24), ("y26_nwd50 (b=6)", CTRL_NWD_50, 54.78),
                       ("y26_b2", CTRL_B2_50, 55.16), ("y26_b3", CTRL_B3_50, 55.30)):
        print(f"{nm:<18}{v:>9.2f}{'-':>9}{v - CTRL_ID_50:>+10.2f}{v95:>10.2f}{'-':>7}")
    print("-" * 63)
    for r in sorted(ok, key=lambda x: -x["test_map50"]):
        v, v95 = r["test_map50"] * 100, r["test_map5095"] * 100
        print(f"{r['name']:<18}{v:>9.2f}{v - r['ctrl']:>+9.2f}"
              f"{v - CTRL_ID_50:>+10.2f}{v95:>10.2f}{r['hours']:>7.2f}")

    by = {r["name"]: (r["test_map50"] * 100, r["ctrl"]) for r in ok}
    print("\n  READ IT")
    best = max(ok, key=lambda x: x["test_map50"] * 100 - x["ctrl"])
    d = best["test_map50"] * 100 - best["ctrl"]
    print(f"    best vs its own base: {best['name']} {d:+.2f}")
    if d < 0.3:
        print("    Nothing clears the noise floor. NWD is absorbed by beta, exactly as the")
        print("    SCB-on-NWD evidence predicted, and the LOSS AXIS IS CLOSED. Every")
        print("    mechanism in the campaign is now either subsumed by tal_beta, a size")
        print("    trade that stops paying once beta is correct, or negative. Write it up.")
    else:
        print("    Clears the floor. Do NOT report it yet — one `nwd` key drives both the")
        print("    assignment metric and the box loss, so this is unattributed. Split the")
        print("    key by consumer before it goes in a table.")
    if "y26_b3_nwd50" in by and "y26_b3_nwd75" in by:
        lo, hi = by["y26_b3_nwd50"][0], by["y26_b3_nwd75"][0]
        print(f"\n    DOSE   0.50 {lo:.2f}   0.75 {hi:.2f}   (0 = b3 {CTRL_B3_50:.2f})")
        print("    Still rising at 0.75 means the beta=6 sweep stopped early and the"
              if hi > lo + 0.2 else
              "    Turned over: 0.5 was the optimum after all and the curve is now closed.")
        if hi > lo + 0.2:
            print("    curve needs one more point. Falling means beta already did NWD's job.")
    if "y26_b2_nwd50" in by:
        v, c = by["y26_b2_nwd50"]
        print(f"\n    ON b2   {v:.2f} ({v - c:+.2f}) — b2 is the only base with large-object")
        print("    headroom to spend, so a win here and a null on b3 would say NWD's trade")
        print("    is only affordable when large is already ahead.")
    print(f"\n    mAP50_small is the metric NWD was wrongly closed on. Run the COCO pass and")
    print(f"    read against b3 {CTRL_B3_S50:.2f}, b2 {CTRL_B2_S50:.2f}, nwd50 {CTRL_NWD_S50:.2f},"
          f" identity {CTRL_ID_S50:.2f}; sd ~{NOISE_S50}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    a = ap.parse_args()

    todo = [r for r in RUNS if not a.names or r["name"] in a.names]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 20 — NWD on a corrected base ({len(todo)} runs, loss only)")
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
