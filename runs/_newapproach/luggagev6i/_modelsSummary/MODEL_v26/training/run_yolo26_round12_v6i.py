#!/usr/bin/env python3
r"""
ROUND 12 — the last four PURE-CONFIG runs, aimed by the FP decomposition
========================================================================

Four runs, ~4 GPU-h, batch 82. No code changes: every key below already exists
in default.yaml and is read by the patched loss. If a run needs a patch, it is
not in this file.


WHY THESE FOUR, AND NOT ANOTHER SIZE-REWEIGHTING
-------------------------------------------------
`diag_fp_decomposition.py` decomposed everything that outranks the median true
positive, on three checkpoints:

    model              total   dup     cls     loc     bg
    y26_identity        192    4.2%   35.9%   16.7%   43.2%
    y26_scb3_sbb50      172    2.3%   37.2%   16.9%   43.6%
    y26_p2k2_hi         148    0.0%   32.4%   18.2%   49.3%

Two things follow, and they set this round's agenda.

1. DUPLICATES ARE NOT LEAKING. 4.2% -> 0.0%. Suppression in the NMS-free head
   works, so the one loss-reachable failure mode in that table is not occurring.
   That also settles SNT (-3.93, closed a gap that was not leaking) and TSH
   (+0.11, widened one that did not need it).

2. WRONG-CLASS IS THE SECOND LARGEST CATEGORY, at 32-37%. In absolute counts the
   P2 architecture won mostly THERE: cls FPs 69 -> 48 (-30%) while background
   barely moved (83 -> 73). The architecture result is largely a CLASSIFICATION
   result, which is not what "P2 helps small objects" predicts.

So the classification path is the one with measured evidence in front of it, and
`cls` has never been touched: `cls=0.5` is hardcoded in the `_ALL_OFF` block of
all eight loss scripts across 73 runs. Runs 1-2 close it. Run 3 attacks the same
target from the assignment side. Run 4 is a one-value control on the box metric.

Base config for all four is `y26_scb3_sbb50` (55.65, +0.41) — SCB 3.0 with the
SBB counterweight at `invert=True`. Note what that arm actually does: `E2ELoss`
sets one2many sign=-1 / one2one sign=+1, and `sbb_weight` uses
`w = (sqrt(area_px)/ref) ** (sign*q)`, so invert=True puts one2one on SMALL and
one2many on LARGE. That is the opposite of the intuition in the code comments,
and it is the arm that won.


CALIBRATION — read before hoping
---------------------------------
The bar is `y26_scb2_sbb50` at 55.70, i.e. +0.46 over baseline. In 73 runs
exactly one config has ever cleared it. Priors here are 20-30% each, and the
campaign's record on COMBINING two positives is 1 for 4 (SNL1+SCB -0.37,
arch+loss -0.19, NWD+SBB+SCB monotonically worse; only SCB+SBB stacked).

Also: with four runs on a single seed, the expected MAXIMUM of four draws sits
~0.25-0.35 above the mean by selection alone. Treat anything under +0.2 as noise
and do not promote it without a second seed.

    Usage:
        python run_yolo26_round12_v6i.py                    # all four
        python run_yolo26_round12_v6i.py y26_scb3_sbb50_cls075
        python run_yolo26_round12_v6i.py --arm cls          # just runs 1-2
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
MODEL_WEIGHTS = "yolo26s.pt"  # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_round12_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches the baseline and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24  # y26_base_rep
BEST_LOSS = 55.65  # y26_scb3_sbb50 — the base config for all four runs here
BEST_RAW = 55.70  # y26_scb2_sbb50 — the number to beat

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
    cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
)

# The base every run in this file sits on: y26_scb3_sbb50.
_BASE = dict(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True)
_BASE_EXPECT = {"scb": (3.0, 64.0), "sbb": 0.5}
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(_BASE)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_scb3_sbb50_cls075", "arm": "cls",
     "expect": {**_BASE_EXPECT, "cls": 0.75},
     "params": cfg(cls=0.75),
     "label": "cls gain 0.5 -> 0.75 on the best config",
     "why": "The ONLY untouched gain in the campaign: cls=0.5 is hardcoded in the "
            "_ALL_OFF block of all eight loss scripts, so none of the 73 runs varied it. "
            "The FP decomposition puts wrong-class at 32-37% of everything outranking the "
            "median true positive — second largest category — and in absolute counts that "
            "is where the P2 architecture actually won (69 -> 48). `loss[1] *= hyp.cls` is "
            "the only term this scales. Take the LOW point first: loss[1] sums BCE over "
            "every anchor while loss[0] sums over positives only, so the balance is "
            "already tilted and doubling straight to 1.0 could destabilise it. A null "
            "here is worth having: it forecloses the obvious reviewer question, given "
            "the project's own diagnostics name ranking as the binding constraint."},

    {"name": "y26_scb3_sbb50_cls10", "arm": "cls",
     "expect": {**_BASE_EXPECT, "cls": 1.0},
     "params": cfg(cls=1.0),
     "label": "cls gain 0.5 -> 1.0 on the best config",
     "why": "The second point is what makes cls a DIRECTION rather than a lone guess — "
            "the same reason nwd was readable at 0.25/0.50 and the single-point SCB spike "
            "was not. If 0.75 and 1.0 are both flat, the gain axis is closed and the "
            "classification deficit is confirmed as immovable from the loss, consistent "
            "with cls_pw making bag worse and misclassification never leaving 3.94-4.90% "
            "across 81 runs."},

    {"name": "y26_a075_scb3_sbb50", "arm": "exp",
     "expect": {**_BASE_EXPECT, "alpha": 0.75},
     "params": cfg(tal_alpha=0.75),
     "label": "tal_alpha 0.5 -> 0.75 on top of SCB + SBB",
     "why": "align_metric = score^alpha * IoU^beta. The two best single mechanisms in the "
            "campaign both live in that ONE expression and have never been run together: "
            "alpha075 (+0.35, and the summary dismisses it as 'not a mechanism') and "
            "beta_small=3.0 (+0.42). They push the same diagnosis from opposite sides — "
            "raise alpha to weight the classification score, lower beta on small boxes to "
            "distrust noisy IoU. Unlike SNL1+SCB (regression vs assignment, did not stack) "
            "these compose multiplicatively inside a single term. It also points the same "
            "way as runs 1-2: more weight on the classification signal, which is where the "
            "FP decomposition says the recoverable errors are."},

    {"name": "y26_scb3_sbb50_diou", "arm": "metric",
     "expect": {**_BASE_EXPECT, "iou_type": "diou"},
     "params": cfg(iou_type="diou"),
     "label": "DIoU — delete CIoU's aspect term instead of replacing it",
     "why": "EIoU REPLACED CIoU's aspect penalty with explicit w/h terms and cost 7.80 on "
            "large (-0.09 overall). DIoU simply DELETES it. CIoU's aspect gradients "
            "satisfy dv/dw = -(h/w) dv/dh, so they are always opposite in sign: it "
            "regresses aspect RATIO and can never scale w and h together. This dataset is "
            "70.6% tall boxes at class-stable ratios (trolley 1.68, backpack 1.47, bag "
            "1.33), so the quantity that term optimises is nearly constant and may be "
            "contributing gradient noise for a prior the model can simply learn. Cleanly "
            "interpretable and never run: giou/diou were available in IOU_FLAGS the whole "
            "campaign and only eiou was ever tried. ~= CIoU means the term is inert, "
            "better means harmful, worse means load-bearing."},
]


def preflight(todo):
    """Prove the patch is installed and every requested key is live BEFORE training.

    A config key can be accepted by the config system, echoed in the run header, and
    then silently ignored — that is exactly how rounds 4-6 produced ten identically
    configured runs under ten different names.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    try:
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
        from ultralytics.utils.metrics import IOU_FLAGS
        from ultralytics.utils.tal import TaskAlignedAssigner as A
    except ImportError as ex:
        print(f"  [ABORT] cannot import the patched modules: {ex}")
        return False

    import inspect
    checks = {
        "loss.py v8DetectionLoss reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py v8DetectionLoss reads iou_type": "iou_type" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py v8DetectionLoss reads tal_alpha": "tal_alpha" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
        "metrics.py IOU_FLAGS has diou": "diou" in IOU_FLAGS,
    }
    for k, v in checks.items():
        print(f"  {k:<52}{v}")
    if not all(checks.values()):
        print("\n  [ABORT] the patch is not fully installed on this machine.")
        return False

    probe = A(topk=7, topk2=1)
    if probe.tsh_enabled() or probe.snt_enabled() or probe.sbal_enabled():
        print("  [ABORT] a mechanism is live at its default value — deltas would be measured"
              " against a moving baseline.")
        return False
    print(f"  {'TSH / SNT / SBAL inert at defaults':<52}True")

    print()
    for r in todo:
        p, e = r["params"], r["expect"]
        a = A(topk=7, topk2=1)
        a.beta_small, a.beta_ref_px = p["tal_beta_small"], p["tal_beta_ref_px"]
        if not a.scb_enabled():
            print(f"  [ABORT] {r['name']}: scb_enabled() False at beta_small={a.beta_small}")
            return False
        bits = [f"SCB {a.beta_small}->{a.beta}@{a.beta_ref_px}px", f"SBB q={p['sbb_q']} inv={p['sbb_invert']}"]
        for key, tag in (("cls", "cls gain"), ("alpha", "tal_alpha"), ("iou_type", "iou_type")):
            if key in e:
                bits.append(f"{tag}={e[key]}")
        print(f"  {r['name']:<26}{'  +  '.join(bits)}")

    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock, no yaml)  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  base config = y26_scb3_sbb50 ({BEST_LOSS})   bar to beat = {BEST_RAW}")
    clash = [os.path.join(PROJECT_DIR, r["name"]) for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))] if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_callbacks(model, rc):
    """Assert at epoch 1 that every requested mechanism is LIVE in the constructed
    criterion, and that nothing else is.
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

        want_b, want_r = e["scb"]
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if not a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB not live on {tag} "
                                   f"(beta_small={getattr(a, 'beta_small', None)})")
            if abs(float(a.beta_small) - want_b) > 1e-6 or abs(float(a.beta_ref_px) - want_r) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} SCB is ({a.beta_small}, {a.beta_ref_px}), "
                                   f"expected ({want_b}, {want_r})")
        seen.append(f"SCB {a2.beta_small}->{a2.beta}@{a2.beta_ref_px}px on BOTH branches")

        for tag, b in (("one2many", b1), ("one2one", b2)):
            if not b.sbb_enabled():
                raise RuntimeError(f"{rc['name']}: SBB not live on {tag} (q={getattr(b, 'sbb_q', None)})")
            if abs(float(b.sbb_q) - e["sbb"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {e['sbb']}")
        # The whole point of SBB is the branch ASYMMETRY; same sign on both would be a
        # global size reweighting, i.e. a different mechanism.
        if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
            raise RuntimeError(f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f} "
                               f"— they must be OPPOSITE")
        # invert=True is the WINNING arm: sign<0 favours small, so one2one must be NEGATIVE.
        if float(b2.sbb_sign) >= 0:
            raise RuntimeError(f"{rc['name']}: one2one sbb_sign={b2.sbb_sign:+.0f}; the arm that won "
                               f"(+0.15) is invert=True -> one2one leans SMALL (sign<0)")
        seen.append(f"SBB q={b2.sbb_q} o2m={b1.sbb_sign:+.0f}(large) o2o={b2.sbb_sign:+.0f}(small)")

        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if "cls" in e and abs(float(h.cls) - e["cls"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: hyp.cls={h.cls}, expected {e['cls']}")
        if "alpha" in e:
            for tag, a in (("one2many", a1), ("one2one", a2)):
                if abs(float(a.alpha) - e["alpha"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} assigner alpha={a.alpha}, expected {e['alpha']}")
            seen.append(f"tal_alpha={a2.alpha} on BOTH branches")
        if "iou_type" in e:
            from ultralytics.utils.metrics import IOU_FLAGS

            want = IOU_FLAGS[e["iou_type"]]
            for tag, obj in (("assigner", a2), ("bbox_loss", b2)):
                if getattr(obj, "iou_kwargs", None) != want:
                    raise RuntimeError(f"{rc['name']}: one2one {tag} iou_kwargs={getattr(obj, 'iou_kwargs', None)}, "
                                       f"expected {want}. Loss and assignment must share one metric.")
            seen.append(f"iou_type={e['iou_type']} on assigner AND loss")

        # Anything not requested must be provably off.
        if any(a.snt_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00 and is off in every run here.")
        if any(a.tsh_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
        if any(a.sbal_enabled() for a in (a1, a2)):
            raise RuntimeError(f"{rc['name']}: SBAL is live; it also has no clamp on target_scores")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")
            if b.snl1_enabled():
                raise RuntimeError(f"{rc['name']}: SNL1 is live on {tag} but was not requested")
            if float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: NWD is live on {tag} (nwd={b.nwd}) but was not requested")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
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
        with open(os.path.join(save_dir, "round12_params.json"), "w") as f:
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    if not ok:
        print("\nno completed runs.")
        return
    print("\n" + "=" * 78)
    print("  ROUND 12 — RESULTS")
    print("=" * 78)
    print(f"{'run':<26}{'arm':<8}{'mAP50-95':>10}{'vs base':>10}{'vs best':>10}{'hours':>8}")
    print("-" * 72)
    print(f"{'y26_base_rep (ref)':<26}{'-':<8}{BASELINE:>10.2f}{0.0:>+10.2f}{BASELINE - BEST_RAW:>+10.2f}{'-':>8}")
    print(f"{'y26_scb3_sbb50 (base)':<26}{'-':<8}{BEST_LOSS:>10.2f}{BEST_LOSS - BASELINE:>+10.2f}"
          f"{BEST_LOSS - BEST_RAW:>+10.2f}{'-':>8}")
    print("-" * 72)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<26}{r['arm']:<8}{v:>10.2f}{v - BASELINE:>+10.2f}{v - BEST_RAW:>+10.2f}"
              f"{r['hours']:>8.2f}")

    print("\n  READ IT")
    cls_runs = sorted((r for r in ok if r["arm"] == "cls"), key=lambda x: x["params"]["cls"])
    if len(cls_runs) == 2:
        d = [r["test_map5095"] * 100 - BEST_LOSS for r in cls_runs]
        print(f"    cls 0.75 -> {d[0]:+.2f}   cls 1.00 -> {d[1]:+.2f}   (vs the {BEST_LOSS} base)")
        if max(d) < 0.2:
            print("    Both flat. The last untouched gain is closed, and the classification")
            print("    deficit is confirmed immovable from the loss — consistent with cls_pw")
            print("    and with misclassification never leaving 3.94-4.90% across 81 runs.")
        elif d[1] > d[0] > 0:
            print("    Monotone and positive — a real direction. A third point (1.5) is")
            print("    justified; a lone spike would not have been.")
        else:
            print("    Non-monotone. Single-seed spikes on a knife-edge base config are how")
            print("    SCB looked for two days. Do not promote without a second seed.")
    best = max(ok, key=lambda x: x["test_map5095"])["test_map5095"] * 100
    if best <= BEST_RAW:
        print(f"    Nothing beat {BEST_RAW}. With four single-seed draws that is the")
        print("    expected outcome, and it leaves the campaign's conclusions intact.")
    else:
        print(f"    Best is {best:.2f}. Note the expected MAXIMUM of four draws sits")
        print("    ~0.25-0.35 above the mean by selection alone — confirm on a second seed")
        print("    before this goes in a table.")
    print("=" * 78 + "\n")


def main():
    args = [a for a in sys.argv[1:]]
    arm = None
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1]
        del args[i:i + 2]
    todo = [r for r in RUNS if (not args or r["name"] in args) and (not arm or r["arm"] == arm)]
    if not todo:
        print(f"no runs matched. available: {[r['name'] for r in RUNS]}")
        return
    if not preflight(todo):
        return
    print(f"\n  {len(todo)} runs, ~{1.0 * len(todo):.0f} GPU-h\n")

    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"\n  [FAIL] {rc['name']}: {ex}\n")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex),
                        "hours": 0.0, "test_map50": float("nan"), "test_map5095": float("nan")})
    try:
        with open(f"{PROJECT_DIR}__runs.json", "w") as f:
            json.dump({"baseline": BASELINE, "base_config": BEST_LOSS, "bar": BEST_RAW,
                       "batch": BATCH, "seed": SEED, "results": res}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] results json not saved: {ex}")
    summarise(res)


if __name__ == "__main__":
    main()
