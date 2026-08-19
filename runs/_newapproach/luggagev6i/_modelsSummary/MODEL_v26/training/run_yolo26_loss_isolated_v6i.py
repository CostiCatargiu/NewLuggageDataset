#!/usr/bin/env python3
"""
YOLO26 LOSS ISOLATION — SNL1 / SCB on the STOCK yolo26s, everything else OFF.

=============================================================================
DESIGN — same shape as run_lbtal_isolated.py on the v12 side
=============================================================================
Base = _ALL_OFF: stock CIoU + L1 + BCE, stock TAL (topk 10 / a 0.5 / b 6.0),
gains 7.5 / 0.5 / 1.5, no SWA, no LB-TAL. The model is the shipped
`yolo26s.pt` — NO custom yaml, NO P2 head, NO DySample. Only the loss changes,
so the delta versus the anchor is the loss mechanism's own effect.

THE ANCHOR RUNS IN THIS SCRIPT. yolo26_custom-9 (55.24) is a single run whose
batch is not recorded anywhere in this repo, and the existing 14-run loss
campaign is split across b32 (port_v6i) and b82 (sweep, phase2) — two of its
"best" configs are not even comparable to each other. Re-measuring the anchor
here at the same batch as everything else is the only way this study answers
"does the loss beat the baseline" without inheriting that mess.

BATCH 82 matches sweep_v6i and phase2_v6i (13 of the 14 existing loss runs), so
these results pool with those rather than starting a third batch regime.

=============================================================================
WHY THESE TWO MECHANISMS AND NOT THE PORTED ONES
=============================================================================
SWA redistributes regression weight AMONG the positives of a GT; LB-TAL
reallocates a top-k budget ACROSS levels. YOLO26 supplies neither precondition:
one2one uses topk2 = 1 (one positive per GT), carries ~90% of the loss by the
last epoch, and produces every prediction. Measured consequence, on the P2
architecture against a 10-replicate control:

    SWA alone                   -0.48  (-2.5 sd)
    LB-TAL uniform              -0.43  (-2.2 sd)
    LB-TAL {4:2,8:3,16:4,32:4}  -0.82  (-4.3 sd)
    SWA + LB-TAL                -0.82  (-4.3 sd)

All four below the band. The ported mechanisms are not merely ineffective here.

SNL1 and SCB are built from YOLO26's own structure instead:

  SNL1  YOLO26 is DFL-free (reg_max 1) and replaces DFL with an L1 on ltrb
        normalised by IMAGE size, so the target magnitude - and the gradient -
        is proportional to object size. At 640 px:

            GT side    8 px -> target 0.0063     1x gradient
            GT side  256 px -> target 0.2000    32x gradient

        For the SAME RELATIVE error a 256 px trolley contributes ~32x the
        regression gradient of an 8 px bag. YOLOv12 has no such bias: DFL is a
        cross-entropy over bins, independent of the target value. SNL1 divides
        the residual by the GT's own extent^p, renormalised to mean 1 so p
        redistributes without rescaling. p = 0 is bit-identical to stock.

  SCB   align_metric = score^alpha * IoU^beta selects positives, and with
        topk2 = 1 it picks THE single anchor per GT in the output branch. IoU is
        a high-variance ranking signal on small boxes and a stable one on large,
        so one global beta over-trusts IoU where it is least reliable. beta 6.0
        is a COCO default never varied in 29 recorded YOLO26 runs.

=============================================================================
BOUND YOUR EXPECTATIONS
=============================================================================
    term          gain   share   scale behaviour
    box  (CIoU)    7.5   78.9%   IoU is a ratio -> already scale-INVARIANT
    cls  (BCE)     0.5    5.3%   size-independent
    dfl  (L1)      1.5   15.8%   the biased term SNL1 corrects

The defect is real but sits in ~16% of the loss while the dominant regression
term is already scale-fair. Do not expect YOLOv12's +0.86.

y26_dfl3 EXISTS TO TELL YOU WHETHER TO BOTHER: it doubles that term's weight
with no code change. If it does not move, the model is not limited by the L1
term and correcting its internal scaling cannot help either.

=============================================================================
REFERENCE POINTS (v6i test_full_dataset)
=============================================================================
    yolo26_custom-9    55.24   stock baseline, n=1, batch UNRECORDED
    y26_swa_a06_03     55.59   best of the 14-run ported campaign (b82)
    y26_lb_uniform     55.52   (b32 - not comparable to the line above)
    14-run campaign     8/14 above baseline, mean -0.05, sign test p = 0.40

Read SMALL per-size. Overall resolves ~0.4 pp; large is unusable at n=1 (the
same architecture measured 53.72..60.66 across ten replicates, sd 2.11).

REQUIRES the patched loss.py / tal.py / default.yaml (SNL1 + SCB).

Usage:
    python run_yolo26_loss_isolated_v6i.py                 # all 5, ~9 GPU-h
    python run_yolo26_loss_isolated_v6i.py y26_anchor y26_dfl3
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
MODEL_WEIGHTS = "yolo26s.pt"          # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_loss_isolated_v6i"
EPOCHS = 70
IMG_SIZE = 640                        # eval MUST also be 640 (the 896 lesson)
BATCH = 82                            # matches sweep_v6i + phase2_v6i
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# published reference points; the ANCHOR run below supersedes the first of these
REF = {"yolo26_custom-9 (n=1, batch ?)": 55.24,
       "y26_swa_a06_03 (b82)": 55.59,
       "y26_lb_uniform (b32)": 55.52}

# stock-neutral base. Every run starts from this and overrides ONE thing.
_ALL_OFF = dict(
    # SWA off
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    # LB-TAL off
    use_lbtal=False,
    # stock TAL
    tal_alpha=0.5, tal_beta=6.0,
    # SNL1 / SCB off
    l1_scale_p=0.0, tal_beta_small=None,
    # stock gains
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_anchor", "expect": {"snl1": False, "scb": False, "dfl_gain": 1.5},
     "params": cfg(),
     "label": "ANCHOR — stock yolo26s, every mechanism OFF",
     "why": "The batch-matched baseline this whole study is read against. "
            "yolo26_custom-9 (55.24) is one run whose batch is recorded nowhere, "
            "and the existing 14-run campaign straddles b32 and b82 — its two "
            "best configs are not comparable to each other. Without this run "
            "every delta below inherits that ambiguity. It also gives a second "
            "sample of the stock model: if it lands far from 55.24, that gap is "
            "your real noise floor on this architecture and every published "
            "delta in the campaign needs re-reading."},

    {"name": "y26_dfl3", "expect": {"snl1": False, "scb": False, "dfl_gain": 3.0},
     "params": cfg(dfl=3.0),
     "label": "PROBE — dfl gain 1.5 -> 3.0, no code change",
     "why": "Answers 'is this lever even connected?' before three more runs go "
            "into it. The L1 term carries 15.8% of the loss gain while CIoU "
            "(78.9%) is already scale-invariant, so the model may simply not be "
            "limited by it. If doubling the weight leaves the result on top of "
            "the anchor, correcting that term's INTERNAL scaling cannot help "
            "either and runs 3-5 are dead. Movement in EITHER direction means "
            "the term matters and SNL1 is worth measuring."},

    {"name": "y26_snl1_p25", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": cfg(l1_scale_p=0.25),
     "label": "SNL1 p=0.25 — partial removal of the 32x size bias",
     "why": "The conservative point: ~1.6x more L1 gradient on 8 px boxes, ~0.68x "
            "on 256 px, with the divisor renormalised to mean 1 so the term's "
            "total magnitude and the dfl gain stay comparable to the anchor. "
            "Chosen over p=1.0 because full scale-invariance puts 10.5x of extra "
            "gradient on the smallest boxes, and large-object AP is already the "
            "weak column on every custom config measured on this dataset."},

    {"name": "y26_snl1_p50", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": cfg(l1_scale_p=0.5),
     "label": "SNL1 p=0.50 — half-way to scale-invariant",
     "why": "Second point of the axis, so p becomes a measured curve rather than "
            "one guess: 2.8x on 8 px, 0.50x on 256 px. With p=0.25 and the anchor "
            "at p=0 that is three points. A monotone trend in SMALL is evidence "
            "even if overall stays flat. If LARGE collapses between 0.25 and "
            "0.50 the useful range is below 0.25 and p=1.0 is pointless."},

    {"name": "y26_scb_b3", "expect": {"snl1": False, "scb": True, "dfl_gain": 1.5},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0),
     "label": "SCB beta 3.0 -> 6.0 over sqrt(area) 0 -> 64 px",
     "why": "Independent of SNL1 — it changes WHICH anchor is selected, not how "
            "the residual is weighted, so a null in one does not implicate the "
            "other. Halving beta for small objects reduces reliance on IoU where "
            "a single pixel of shift swings it. On the stock 3-level model the "
            "output branch still runs topk2=1, so one bad pick per GT has no "
            "runner-up to correct it — that is what makes the exponent worth "
            "moving off its COCO default here."},
]


def preflight(todo):
    """Fail on anything that would make a run a silent no-op."""
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    checks = {
        "BboxLoss.snl1_enabled (SNL1)": hasattr(BboxLoss, "snl1_enabled"),
        "BboxLoss.l1_scale_denom (SNL1)": hasattr(BboxLoss, "l1_scale_denom"),
        "TaskAlignedAssigner.scb_enabled (SCB)": hasattr(TAL.TaskAlignedAssigner, "scb_enabled"),
        "loss.py reads l1_scale_p": "l1_scale_p" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<40}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] the SNL1/SCB patch is not installed.")
        print("  python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False

    for r in todo:
        p = r["params"]
        if r["expect"]["snl1"]:
            bl = BboxLoss(1)
            bl.l1_scale_p = float(p["l1_scale_p"])
            if not bl.snl1_enabled():
                print(f"  [ABORT] {r['name']}: snl1_enabled() False at p={p['l1_scale_p']}")
                return False
        if r["expect"]["scb"]:
            a = TAL.TaskAlignedAssigner(topk=10, beta=6.0)
            a.beta_small = float(p["tal_beta_small"])
            a.beta_ref_px = float(p.get("tal_beta_ref_px", 64.0))
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: scb_enabled() False")
                return False

    print()
    print(f"  MODEL   {MODEL_WEIGHTS}  (STOCK — no yaml, no P2 head, no DySample)")
    print(f"  BATCH   {BATCH}  matches sweep_v6i + phase2_v6i")
    print(f"  ANCHOR  y26_anchor runs in THIS script — read every delta against it,")
    print(f"          not against yolo26_custom-9 (55.24, batch unrecorded).")
    print(f"  Read SMALL per-size. Large is unusable at n=1 (sd 2.11 pp).")

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
    """Assert at epoch 1 that the mechanism is LIVE in the constructed criterion."""
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        for h in (crit, getattr(crit, "one2many", None), getattr(crit, "one2one", None)):
            bl = getattr(h, "bbox_loss", None) if h is not None else None
            if bl is None:
                continue
            for attr, val in (("epoch", trainer.epoch), ("total_epochs", trainer.epochs)):
                if hasattr(bl, attr):
                    setattr(bl, attr, val)
        if state["verified"] or trainer.epoch < 1:
            return

        o2o = getattr(crit, "one2one", crit)
        o2m = getattr(crit, "one2many", crit)
        exp = rc["expect"]
        bl = o2o.bbox_loss

        if bl.dfl_loss is not None:
            raise RuntimeError(
                f"{rc['name']}: reg_max > 1, so the DFL branch runs and there is no L1 "
                f"term. SNL1 is meaningless on this model — check {MODEL_WEIGHTS}.")
        if exp["snl1"]:
            if not (hasattr(bl, "snl1_enabled") and bl.snl1_enabled()):
                raise RuntimeError(
                    f"{rc['name']}: l1_scale_p requested but snl1_enabled() is False "
                    f"(p={getattr(bl, 'l1_scale_p', None)}). Aborting.")
            print(f"  [guard] SNL1 live on one2one: p={bl.l1_scale_p}")
        elif getattr(bl, "l1_scale_p", 0.0) != 0.0:
            raise RuntimeError(f"{rc['name']}: expected SNL1 OFF but l1_scale_p={bl.l1_scale_p}")

        for tag, br in (("one2one", o2o), ("one2many", o2m)):
            a = getattr(br, "assigner", None)
            live = a is not None and hasattr(a, "scb_enabled") and a.scb_enabled()
            if exp["scb"] and not live:
                raise RuntimeError(
                    f"{rc['name']}: tal_beta_small requested but scb_enabled() is False "
                    f"on {tag} (beta_small={getattr(a, 'beta_small', None)}). Aborting.")
            if not exp["scb"] and live:
                raise RuntimeError(f"{rc['name']}: expected SCB OFF but it is live on {tag}")
            if exp["scb"]:
                print(f"  [guard] SCB live on {tag}: beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px")

        got = float(getattr(o2o.hyp, "dfl", 1.5))
        if abs(got - exp["dfl_gain"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: dfl gain is {got}, expected {exp['dfl_gain']}")
        print(f"  [guard] dfl gain = {got}   assigner = {type(o2o.assigner).__name__}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff or '(nothing — this is the anchor)'}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "params": rc["params"], "expect": rc["expect"], "seed": SEED,
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "loss_iso_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    by = {r["name"]: r["test_map5095"] * 100 for r in ok}
    anchor = by.get("y26_anchor")
    print()
    print("=" * 86)
    print(f"  YOLO26 LOSS ISOLATION — stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 86)
    ref = f"{anchor:.2f} (this script)" if anchor else "MISSING — run y26_anchor"
    print(f"  anchor: {ref}")
    print()
    print(f"{'run':<16}{'mAP50':>9}{'mAP50-95':>10}{'vs anchor':>11}{'h':>6}")
    print("-" * 86)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        d = f"{v - anchor:>+11.2f}" if anchor and r["name"] != "y26_anchor" else f"{'ANCHOR' if r['name'] == 'y26_anchor' else '?':>11}"
        print(f"{r['name']:<16}{r['test_map50'] * 100:>9.2f}{v:>10.2f}{d}{r['hours']:>6.1f}")
    print("-" * 86)
    for k, v in REF.items():
        print(f"  {k:<34}{v:>7.2f}")

    if anchor:
        print(f"\n  ANCHOR CHECK: this run {anchor:.2f} vs published yolo26_custom-9 55.24 "
              f"-> {anchor - 55.24:+.2f}")
        if abs(anchor - 55.24) > 0.4:
            print("    That gap IS the stock model's run-to-run spread. Every delta in the")
            print("    published 14-run campaign (best +0.35) is inside it and none of it")
            print("    is reportable without more anchor samples.")
    if anchor and "y26_dfl3" in by:
        d = by["y26_dfl3"] - anchor
        print(f"\n  PROBE: dfl 1.5 -> 3.0 moved {d:+.2f} pp")
        if abs(d) < 0.2:
            print("    Flat. The model is not limited by the L1 term, so correcting that")
            print("    term's internal scaling cannot help. Report the analysis, drop the runs.")
        else:
            print("    The L1 term is connected — SNL1 is worth measuring.")
    snl1 = sorted((r["params"]["l1_scale_p"], by[r["name"]]) for r in ok if r["expect"]["snl1"])
    if snl1 and anchor:
        print("\n  SNL1 axis (anchor is p = 0):")
        print(f"    p=0.00  {anchor:.2f}")
        for p, v in snl1:
            print(f"    p={p:.2f}  {v:.2f}   {v - anchor:+.2f}")
        print("    Monotone in SMALL is evidence even if overall is flat.")
    print("\n  Read per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print("  Do NOT read LARGE at n=1 — this dataset gave 53.72..60.66 across ten")
    print("  replicates of one config (sd 2.11 pp).")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<16} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    if only and "y26_anchor" not in only:
        print("  [!] y26_anchor not selected — deltas will fall back to the published")
        print("      55.24, whose batch is unrecorded. Prefer running the anchor too.")
    print()
    print("=" * 86)
    print(f"  YOLO26 LOSS ISOLATION — {len(todo)} runs, ~{1.8 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 86)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "seed": SEED, "hours": float("nan"),
                        "error": str(e), "expect": r["expect"], "params": r["params"],
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
