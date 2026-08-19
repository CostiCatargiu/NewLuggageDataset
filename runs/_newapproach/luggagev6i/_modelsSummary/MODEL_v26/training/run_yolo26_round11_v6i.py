#!/usr/bin/env python3
r"""
ROUND 11 — NWD / EIoU: the first mechanisms that act UPSTREAM of everything else
===============================================================================

Six runs, ~6 GPU-h. Ported from `small-object-loss-mods/`, which was written and
reviewed but NEVER EXECUTED. Read the identity section before trusting any number.


WHY THIS IS DIFFERENT FROM THE PREVIOUS 40 CONFIGS
--------------------------------------------------
`bbox_iou` is the single function that decides BOTH what gets assigned and what the
box loss is:

    tal.py  iou_calculation  -> bbox_iou     which anchor becomes positive
    loss.py BboxLoss.forward -> bbox_iou     the regression loss itself

Every mechanism in this project so far took that function's OUTPUT and reweighted
it. SCB lowers the exponent on it, SBB scales the loss built from it, SNL1
normalises a different term. None of them changed what it COMPUTES. Until now
metrics.py was completely untouched.

The case for changing it. IoU is a RATIO, so the same absolute error means
completely different things by size:

    15 x 25 px box, 3px offset   ->  CIoU ~0.50     ^6 = 0.016
    100 px box, same 3px offset  ->  CIoU ~0.95     ^6 = 0.735

That is a ~47x difference in assignment weight from the SAME absolute localisation
error, and 35.8% of this dataset is under 32px. NWD models each box as a Gaussian
and maps the 2nd Wasserstein distance to a similarity by exp(-d/c) — an ABSOLUTE
measure, so it degrades identically at every scale.

The strongest argument is that YOUR BEST RESULT ALREADY ASSUMES THIS IS TRUE. SCB
is "lower beta for small objects" — trust IoU less where it is noisy — and it gave
+0.42, the best single mechanism in the campaign. NWD attacks the same diagnosis
from the other side: don't discount the noisy metric, replace it. If SCB works
because IoU is unreliable on small boxes, fixing the metric should work at least as
well. If it doesn't, that is evidence SCB was doing something else, which is worth
knowing.

Side effect, free: exp(-d/c) is strictly positive, so align_metric never lands on
exactly 0. `iou_calculation` ends in `.clamp_(0)`, and with beta=6 every bad
candidate collapses to exactly zero — at which point `torch.topk` breaks ties by
INDEX ORDER. In the one2one branch topk2=1, so a GT whose candidates are all
clamped gets a spatially-arbitrary anchor. NWD defuses that without a separate fix.


WHERE THE PROJECT STANDS (b82, deterministic, batch-matched)
------------------------------------------------------------
    run                 mAP50-95   mAP50    small    med     large
    y26_base_rep          55.24    80.18    77.30   86.45   81.75
    y26_scb3_sbb50        55.65    80.86    77.92   87.25   83.36   <- best
    y26_scb2_sbb50        55.70    80.95    77.83   87.09   82.44

Closed out, do not revisit: SNL1 (== the folder's `dfl_obj_norm`), SNT, TSH,
cls_pw, SWA, LB-TAL, tal_beta sweep, sbb_q knife-edge.


RUN 1 IS THE ONLY ONE THAT MATTERS IF IT FAILS
-----------------------------------------------
`y26_identity` is stock: nwd=0.0, iou_type=ciou, every mechanism off. Training on
this box is DETERMINISTIC — y26_base_rep came back bit-identical to
yolo26_custom-9 across all 118 values. So this run MUST return 55.24 exactly.

    exactly 55.24    -> the port is arithmetically inert; runs 2-6 mean something
    anything else    -> the restructure changed the stock path. STOP. Every number
                        after it is measuring the bug, not the mechanism.

metrics.py was restructured from early-`return` to fall-through so the NWD block
could see the computed variant. That is the kind of edit that looks obviously safe
and silently isn't, which is exactly why this run exists.

The source folder states three defects were caught by REVIEW rather than by
running: an in-place clamp_ on an autograd-tracked tensor, an infinite sqrt
gradient at zero distance, and fp16 overflow under AMP. All three guards are in the
ported code. Others may remain.


CALIBRATION
-----------
Eleven directional predictions this campaign; ten falsified. The one that landed
(scb2 rescued by SBB) came from a measured asymmetry, not a story. NWD has a story
AND an aligned measurement, which is better than TSH had, but the honest prior is
still low. Run 1 is not a prediction at all — it is a correctness check.


Usage:
    python run_yolo26_round11_v6i.py                  # all six, ~6 GPU-h
    python run_yolo26_round11_v6i.py y26_identity     # do this FIRST if unsure
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
PROJECT_DIR = "runs_yolo26_round11_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24
BASE_S, BASE_M, BASE_L = 77.30, 86.45, 81.75   # mAP50 buckets
BEST = 55.65                                    # y26_scb3_sbb50
BEST_S, BEST_M, BEST_L = 77.92, 87.25, 83.36

# sqrt(39 x 55) ~= 46 — the dataset's mean object size, in pixels.
NWD_C = 46.0

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


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_identity", "arm": "check",
     "expect": {"stock": True},
     "params": cfg(),
     "label": "IDENTITY CHECK — stock through the ported metrics.py",
     "why": "Run this first and do not proceed until it passes. metrics.py was "
            "restructured from early-return to fall-through so the NWD block could "
            "blend with the computed variant; bbox_iou feeds both the assigner and "
            "the box loss, so an arithmetic slip there corrupts everything silently "
            "rather than crashing. Determinism means the answer is not 'close to "
            "55.24' — it is 55.24 exactly, and every per-class value should match "
            "y26_base_rep. Anything else and runs 2-6 are measuring a bug."},

    {"name": "y26_eiou", "arm": "a",
     "expect": {"iou_type": "eiou"},
     "params": cfg(iou_type="eiou"),
     "label": "EIoU — independent w/h penalties instead of CIoU's aspect term",
     "why": "Free once metrics.py is in: no new code path, one config value. CIoU's "
            "aspect term has dv/dw = -(h/w) * dv/dh, so its width and height "
            "gradients are ALWAYS opposite in sign — it can trade w against h but "
            "never scale both together. It regresses aspect RATIO, never aspect "
            "MAGNITUDE. This dataset is 70.6% tall boxes at class-stable ratios "
            "(trolley 1.68, backpack 1.47, bag 1.33), so the thing CIoU optimises is "
            "already nearly constant and the thing it cannot optimise is what "
            "varies. EIoU is 2021 work — a baseline row, not a contribution — but "
            "it is one run and the argument is specific to your data."},

    {"name": "y26_nwd25", "arm": "b",
     "expect": {"nwd": 0.25},
     "params": cfg(nwd=0.25, nwd_c=NWD_C),
     "label": "NWD 0.25 — gentle blend",
     "why": "The conservative point. NWD changes the metric that BOTH the assigner "
            "and the loss consume, so it is the highest-blast-radius change in the "
            "project — every other mechanism could only reweight a fixed quantity. "
            "0.25 keeps CIoU dominant while letting the Wasserstein term carry the "
            "small-box cases where CIoU is noisiest. Two points make nwd a direction "
            "rather than a guess, which is what made the SNT result readable even "
            "though it was negative."},

    {"name": "y26_nwd50", "arm": "b",
     "expect": {"nwd": 0.5},
     "params": cfg(nwd=0.5, nwd_c=NWD_C),
     "label": "NWD 0.50 — the paper's suggested setting",
     "why": "Equal blend, and the value the source folder recommends for this "
            "dataset. nwd_c=46 is sqrt(39 x 55), the measured mean object size — it "
            "is a dataset statistic, not a tuned knob, which matters if a reviewer "
            "asks. Read the SMALL bucket first: if NWD does what it claims, small "
            "moves before overall does. Watch for NaN in the first iterations — if "
            "box loss goes NaN immediately, suspect the fp16 branch."},

    {"name": "y26_nwd50_sbb", "arm": "b",
     "expect": {"nwd": 0.5, "sbb": 0.5},
     "params": cfg(nwd=0.5, nwd_c=NWD_C, sbb_q=0.5, sbb_invert=True),
     "label": "NWD 0.50 + SBB inv 0.5 — paired, per your own principle",
     "why": "Your data says every single-direction size push costs ~4 points of "
            "large and only OPPOSING pairs recover it — confirmed twice, with "
            "SCB+SBB (large 60.87 -> 60.82) and SNL1+SBB (-> 60.48), where both "
            "singles individually cost ~4. NWD helps small boxes, so by that "
            "principle it is a small-side push and should need the same "
            "counterweight. If NWD alone loses large and this pair recovers it, "
            "that is a THIRD independent instance of the principle — which would be "
            "worth more to the paper than the mAP number."},

    {"name": "y26_nwd50_scb3_sbb", "arm": "b",
     "expect": {"nwd": 0.5, "scb": (3.0, 64.0), "sbb": 0.5},
     "params": cfg(nwd=0.5, nwd_c=NWD_C, tal_beta_small=3.0, tal_beta_ref_px=64.0,
                   sbb_q=0.5, sbb_invert=True),
     "label": "NWD on top of the best config — do they compose or overlap?",
     "why": "The most informative run after the identity check. SCB and NWD address "
            "THE SAME DIAGNOSIS by different routes: SCB discounts a noisy metric "
            "(lower beta on small GTs), NWD replaces it with one that isn't noisy. "
            "If they compose, the diagnosis is right and they fix different parts of "
            "it. If this lands at or below y26_scb3_sbb50 (55.65), they are two "
            "routes to the same correction and the paper reports the better single. "
            "Either answer is publishable; 'we tried both and stacked them' is not."},
]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
        from ultralytics.utils.metrics import IOU_FLAGS, bbox_iou
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")

    sig = inspect.signature(bbox_iou).parameters
    A = TAL.TaskAlignedAssigner
    checks = {
        "metrics.py bbox_iou accepts EIoU": "EIoU" in sig,
        "metrics.py bbox_iou accepts NWD": "NWD" in sig,
        "metrics.py bbox_iou accepts nwd_c": "nwd_c" in sig,
        "metrics.py IOU_FLAGS has all 5 variants": set(IOU_FLAGS) == {"iou", "giou", "diou", "ciou", "eiou"},
        "tal.py assigner has instance_gain (SBAL)": hasattr(A, "instance_gain"),
        "tal.py iou_calculation passes NWD": "NWD" in inspect.getsource(A.iou_calculation),
        "loss.py BboxLoss.forward passes NWD": "NWD" in inspect.getsource(BboxLoss.forward),
        "loss.py v8DetectionLoss reads iou_type": "iou_type" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py v8DetectionLoss reads nwd": "nwd" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<46}{v}")
    if not all(checks.values()):
        print("\n  [ABORT] the NWD/EIoU port is not installed on this machine.")
        return False

    # Defaults must be inert. If bbox_iou at defaults differs from CIoU=True, the
    # restructure changed the stock path and y26_identity would silently drift.
    a, b = torch.tensor([[10.0, 10.0, 50.0, 60.0]]), torch.tensor([[12.0, 13.0, 48.0, 63.0]])
    stock = bbox_iou(a, b, xywh=False, CIoU=True)
    ported = bbox_iou(a, b, xywh=False, **IOU_FLAGS["ciou"], NWD=0.0, nwd_c=24.0)
    same = torch.equal(stock, ported)
    print(f"  {'bbox_iou @ defaults == CIoU=True (bitwise)':<46}{same}")
    if not same:
        print(f"  [ABORT] stock {stock.item():.10f} vs ported {ported.item():.10f}")
        return False
    # And NWD must actually change it, or the mechanism is a silent no-op.
    moved = not torch.equal(stock, bbox_iou(a, b, xywh=False, CIoU=True, NWD=0.5, nwd_c=NWD_C))
    print(f"  {'bbox_iou @ nwd=0.5 differs from stock':<46}{moved}")
    if not moved:
        print("  [ABORT] NWD=0.5 produced an identical value — the block is dead code.")
        return False

    print()
    for r in todo:
        p, e = r["params"], r["expect"]
        if e.get("stock"):
            live = [k for k, v in p.items() if _ALL_OFF.get(k, "__") != v]
            if live:
                print(f"  [ABORT] {r['name']} is the identity control but differs: {live}")
                return False
            print(f"  {r['name']:<22}STOCK — must return exactly {BASELINE:.2f}")
            continue
        bits = []
        if "nwd" in e:
            bits.append(f"NWD {p['nwd']} @ {p['nwd_c']}px")
        if e.get("iou_type"):
            bits.append(f"iou_type={p['iou_type']}")
        if "scb" in e:
            bits.append(f"SCB {p['tal_beta_small']}->6.0")
        if "sbb" in e:
            bits.append(f"SBB q={p['sbb_q']} inv")
        print(f"  {r['name']:<22}{' + '.join(bits)}")

    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock)  b{BATCH}  {IMG_SIZE}px  seed {SEED}")
    print(f"  baseline {BASELINE:.2f}   best so far {BEST:.2f} (y26_scb3_sbb50)")
    print(f"  mAP50 buckets — baseline S {BASE_S} M {BASE_M} L {BASE_L}")
    print("  DETERMINISTIC: y26_identity must return 55.24 EXACTLY, not approximately.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_callbacks(model, rc):
    """Assert at epoch 1 that exactly the named mechanisms are live — and that the
    assigner and the loss agree about what 'overlap' means. Upstream hardcoded
    CIoU=True in two separate places; if they ever disagree the model regresses one
    definition while being assigned by another, and nothing crashes.
    """
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return
        if state["verified"] or trainer.epoch < 1:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss")
        seen = []

        for tag, br in (("one2many", o2m), ("one2one", o2o)):
            a, bl = br.assigner, br.bbox_loss
            # THE critical invariant: assigner and loss must use the same metric.
            if a.iou_kwargs != bl.iou_kwargs or float(a.nwd) != float(bl.nwd) \
                    or float(a.nwd_c) != float(bl.nwd_c):
                raise RuntimeError(
                    f"{rc['name']}: {tag} assigner and loss DISAGREE on the overlap metric — "
                    f"assigner {a.iou_kwargs}/nwd={a.nwd}/c={a.nwd_c} vs "
                    f"loss {bl.iou_kwargs}/nwd={bl.nwd}/c={bl.nwd_c}. The model would regress "
                    f"one definition while being assigned by another.")
            want_nwd = float(e.get("nwd", 0.0))
            if abs(float(a.nwd) - want_nwd) > 1e-9:
                raise RuntimeError(f"{rc['name']}: {tag} nwd={a.nwd}, expected {want_nwd}")
            want_iou = e.get("iou_type", "ciou")
            from ultralytics.utils.metrics import IOU_FLAGS
            if a.iou_kwargs != IOU_FLAGS[want_iou]:
                raise RuntimeError(f"{rc['name']}: {tag} iou_kwargs={a.iou_kwargs}, expected {want_iou}")
            if "scb" in e:
                wb, wr = e["scb"]
                if not a.scb_enabled() or abs(float(a.beta_small) - wb) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: SCB not live/wrong on {tag}")
            elif a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB live on {tag}, not requested")
            if "sbb" in e:
                if not bl.sbb_enabled() or abs(float(bl.sbb_q) - e["sbb"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: SBB not live/wrong on {tag}")
            elif bl.sbb_enabled():
                raise RuntimeError(f"{rc['name']}: SBB live on {tag}, not requested")
            # Never requested anywhere in round 11.
            for meth, nm in (("snt_enabled", "SNT"), ("tsh_enabled", "TSH"), ("sbal_enabled", "SBAL")):
                if hasattr(a, meth) and getattr(a, meth)():
                    raise RuntimeError(f"{rc['name']}: {nm} live on {tag}")
            for meth, nm in (("snl1_enabled", "SNL1"), ("swa_enabled", "SWA")):
                if hasattr(bl, meth) and getattr(bl, meth)():
                    raise RuntimeError(f"{rc['name']}: {nm} live on {tag}")

        a2, b2 = o2o.assigner, o2o.bbox_loss
        seen.append(f"overlap: iou_kwargs={a2.iou_kwargs} nwd={a2.nwd} c={a2.nwd_c}px "
                    f"(assigner == loss on BOTH branches)")
        if "scb" in e:
            seen.append(f"SCB {a2.beta_small}->{a2.beta}")
        if "sbb" in e:
            seen.append(f"SBB q={b2.sbb_q} signs o2m={o2m.bbox_loss.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f}")
        if e.get("stock"):
            seen.append("STOCK verified — nothing live; this run must return 55.24")
        h = o2o.hyp
        if (float(h.box), float(h.cls), float(h.dfl)) != (7.5, 0.5, 1.5):
            raise RuntimeError(f"{rc['name']}: gains {h.box}/{h.cls}/{h.dfl} != 7.5/0.5/1.5")
        for s in seen:
            print(f"  [guard] {s}")
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
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  b{rc.get('batch', BATCH)}  seed {SEED}  differs: {diff or 'NOTHING (identity control)'}")
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
           "seed": SEED, "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan"), "per_class": {}}
    try:
        with open(os.path.join(save_dir, "round11_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
        try:
            names = tm.names if isinstance(tm.names, dict) else dict(enumerate(tm.names))
            out["per_class"] = {names[int(c)]: float(tm.box.maps[int(c)]) for c in range(len(tm.box.maps))}
        except Exception:
            pass
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rc["expect"].get("stock") and out["test_map5095"] == out["test_map5095"]:
        got = out["test_map5095"] * 100
        print()
        print("  " + "=" * 70)
        if abs(got - BASELINE) < 0.005:
            print(f"  IDENTITY CHECK PASSED — {got:.2f} == {BASELINE:.2f}. The port is inert.")
        else:
            print(f"  *** IDENTITY CHECK FAILED — got {got:.2f}, expected {BASELINE:.2f} ***")
            print("  The restructured metrics.py changed the stock path. Every run after")
            print("  this measures the bug, not the mechanism. Fix before continuing.")
        print("  " + "=" * 70)
    return out


def summarise(res, path):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    print()
    print("=" * 88)
    print(f"  ROUND 11 — NWD / EIoU | stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 88)
    print(f"{'run':<24}{'arm':>6}{'mAP50':>9}{'mAP50-95':>10}{'vs base':>9}{'vs best':>9}")
    print("-" * 88)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<24}{r['arm'].upper():>6}{r['test_map50']*100:>9.2f}"
              f"{v:>10.2f}{v-BASELINE:>+9.2f}{v-BEST:>+9.2f}")
    print("-" * 88)
    print(f"  {'baseline':<24}{'':>6}{80.18:>9.2f}{BASELINE:>10.2f}")
    print(f"  {'y26_scb3_sbb50 (best)':<24}{'':>6}{80.86:>9.2f}{BEST:>10.2f}{BEST-BASELINE:>+9.2f}")
    print()

    idc = [r for r in ok if r["expect"].get("stock")]
    if idc:
        got = idc[0]["test_map5095"] * 100
        verdict = "PASSED — port is inert" if abs(got - BASELINE) < 0.005 else "*** FAILED ***"
        print(f"  IDENTITY: {got:.2f} vs {BASELINE:.2f}  ->  {verdict}\n")

    print("  Read the SMALL bucket first — NWD's whole claim is about boxes under 32px,")
    print("  which are 35.8% of instances. Overall mAP can stay flat while the mechanism")
    print("  is working or not working, so it cannot settle this.")
    print()
    print(f"    {'config':<24}{'small':>8}{'med':>8}{'large':>8}   (mAP50)")
    print(f"    {'baseline':<24}{BASE_S:>8.2f}{BASE_M:>8.2f}{BASE_L:>8.2f}")
    print(f"    {'y26_scb3_sbb50':<24}{BEST_S:>8.2f}{BEST_M:>8.2f}{BEST_L:>8.2f}")
    for r in sorted(ok, key=lambda x: x["name"]):
        print(f"    {r['name']:<24}{'____':>8}{'____':>8}{'____':>8}")
    print()
    print("      small UP, large flat/down  -> NWD does what it claims; pair it (nwd50_sbb)")
    print("      nothing moves              -> the metric was not the binding constraint,")
    print("                                    and SCB's +0.42 came from something else")
    print("      nwd50_scb3_sbb <= 55.65    -> SCB and NWD are two routes to one correction")
    print("      nwd50_scb3_sbb >  55.65    -> they fix different parts; report the stack")
    print()
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<24} {r['weights']}")
    print(f"\n  Run CocoEvalAllFolders_luggage.py on each best.pt.\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    if not todo:
        sys.exit(f"no runs match {only}")
    print()
    print("=" * 88)
    print(f"  YOLO26 ROUND 11 — {len(todo)} runs, ~{1.0*len(todo):.1f} GPU-h")
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
                        "expect": rc["expect"],
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump({"baseline": BASELINE, "best": BEST, "deterministic": True,
                       "nwd_c": NWD_C, "results": res}, f, indent=2)

    summarise(res, out_path)
