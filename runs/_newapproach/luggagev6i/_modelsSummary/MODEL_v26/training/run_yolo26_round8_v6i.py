#!/usr/bin/env python3
"""
YOLO26 ROUND 8 — two mechanisms designed FOR YOLO26, not ported into it.

=============================================================================
WHY THE PORTED MECHANISMS COULD NOT WORK
=============================================================================
SWA redistributes regression weight AMONG the positives assigned to a GT.
LB-TAL reallocates a top-k budget ACROSS pyramid levels. YOLO26 provides
neither precondition:

    one2one uses topk2=1        -> ONE positive per GT, nothing to redistribute
    one2one carries ~90% of the loss by the last epoch (o2m decays 0.8 -> 0.1)
    one2one produces every prediction (the head is NMS-free)
    LB-TAL is gated to one2many (topk2 is None) and never touches the output

That is why 14 loss runs on YOLO26 gave 8/14 above baseline, mean -0.05,
p = 0.40. Not bad hyperparameters — the operations had no room to act.

=============================================================================
THE DEFECT THESE RUNS TARGET
=============================================================================
YOLO26 is DFL-free (reg_max = 1) and replaces DFL with an L1 on ltrb
normalised by IMAGE size:

    target_ltrb = target_ltrb * stride ; target_ltrb[..., 0::2] /= imgsz[1]

so the target magnitude is proportional to object size. At 640 px:

    GT side    8 px -> target 0.0063     1x gradient
    GT side  256 px -> target 0.2000    32x gradient

For the SAME RELATIVE localisation error, a 256 px trolley contributes ~32x
the regression gradient of an 8 px bag. YOLOv12 does not have this problem:
DFL is a cross-entropy over reg_max bins whose magnitude is independent of the
target value. The bias is specific to the DFL-free head, and nobody has
touched it.

  SNL1  divides the residual by the GT's own extent^p, renormalised to mean 1
        so p redistributes gradient without rescaling the term.
        p = 0 is bit-identical to stock.

  SCB   align_metric = score^alpha * IoU^beta picks THE single anchor per GT
        when topk2 = 1. IoU is a high-variance ranking signal on small boxes and
        a stable one on large boxes, so a single global beta over-trusts IoU
        exactly where it is least reliable. SCB interpolates beta by GT size.
        On YOLOv12 this lever is weak (topk=10 mostly reorders a kept set);
        topk2=1 is what makes it worth testing here.

=============================================================================
BOUND YOUR EXPECTATIONS BEFORE READING THE RESULTS
=============================================================================
    term          gain   share   scale behaviour
    box  (CIoU)    7.5   78.9%   IoU is a ratio -> already scale-INVARIANT
    cls  (BCE)     0.5    5.3%   size-independent
    dfl  (L1)      1.5   15.8%   the biased term SNL1 corrects

The defect is real but lives in ~16% of the loss, and the dominant regression
term is already scale-fair. Do not expect anything like YOLOv12's +0.86.

RUN 1 EXISTS TO TELL YOU WHETHER TO BOTHER. It doubles the dfl gain with NO
code change. If a 2x change in that term's weight moves nothing, the model is
insensitive to it and correcting its internal scaling cannot help either — stop
after one run instead of four.

=============================================================================
CONTROL
=============================================================================
Ten runs of this exact architecture at b32/640/seed0 (rounds 4-6, whose budget
labels never took effect) form a replicate distribution:

    56.08 +- 0.19   overall mAP50-95   (n = 10)
    52.14 +- 0.23   small
    57.21 +- 2.11   large   <- do NOT read large at n=1

    DECISION BAND   55.70 .. 56.46     inside = null, outside = real

56.46 is also the best of the ten draws, which is the correct bar: to claim a
gain a config must beat the luckiest replicate of doing nothing.

Architecture FIXED at the DySample P2 variant, byte-identical to the yaml
behind all ten control runs. b32, 640, seed 0.
REQUIRES the patched loss.py / tal.py / default.yaml.

Usage:
    python run_yolo26_round8_v6i.py y26_dfl3        # the probe, run this first
    python run_yolo26_round8_v6i.py
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
PROJECT_DIR = "runs_yolo26_round8_v6i"
YAML_DIR = "arch_yamls_y26_r8"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CTRL_MEAN, CTRL_SD, CTRL_N = 56.08, 0.19, 10
BAND = (CTRL_MEAN - 2 * CTRL_SD, CTRL_MEAN + 2 * CTRL_SD)
STOCK_DFL_GAIN = 1.5

DYS_YAML = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""


RUNS = [
    {"name": "y26_dfl3", "expect": {"snl1": False, "scb": False, "dfl_gain": 3.0},
     "params": dict(dfl=3.0),
     "label": "PROBE — dfl gain 1.5 -> 3.0, no code change",
     "why": "Answers 'is this lever even connected?' before you spend three more "
            "runs on it. The L1 term carries 15.8% of the loss gain while CIoU "
            "(78.9%) is already scale-invariant, so the model may simply not be "
            "limited by it. If doubling the term's weight leaves the result inside "
            "the control band, correcting its INTERNAL scaling cannot help either "
            "and runs 2-4 are dead. If it moves in either direction, the term "
            "matters and SNL1 is worth measuring. One run either way."},

    {"name": "y26_snl1_p25", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": dict(l1_scale_p=0.25),
     "label": "SNL1 p=0.25 — partial removal of the 32x size bias",
     "why": "The conservative point. Multiplies small-object L1 gradient ~1.6x and "
            "cuts the 256 px end to ~0.68x, with the divisor renormalised to mean "
            "1 so the term's total magnitude is unchanged and the dfl gain stays "
            "comparable to the control. Chosen over p=1.0 because this dataset's "
            "large-object AP is already the weak column on every custom config, "
            "and full scale-invariance pushes 10.5x of extra gradient onto the "
            "smallest boxes."},

    {"name": "y26_snl1_p50", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": dict(l1_scale_p=0.5),
     "label": "SNL1 p=0.50 — half-way to scale-invariant",
     "why": "The second point of the axis, so p is a measured curve rather than a "
            "single guess. 2.8x on 8 px boxes, 0.50x on 256 px. With p=0.25 and "
            "the control at p=0 this gives three points; a monotone trend in "
            "small-object AP would be evidence even if the overall numbers stay "
            "inside the band. Watch LARGE — if it falls off a cliff between 0.25 "
            "and 0.50, the useful range is below 0.25 and p=1.0 is pointless."},

    {"name": "y26_scb_b3", "expect": {"snl1": False, "scb": True, "dfl_gain": 1.5},
     "params": dict(tal_beta_small=3.0, tal_beta_ref_px=64.0),
     "label": "SCB beta 3.0 -> 6.0 over sqrt(area) 0 -> 64 px",
     "why": "Independent of SNL1: it changes WHICH anchor is chosen rather than how "
            "the residual is weighted, so a null here does not implicate SNL1 or "
            "vice versa. beta=6.0 is a COCO default never varied in 29 YOLO26 runs. "
            "Halving it for small objects reduces reliance on IoU where a single "
            "pixel of shift swings it — precisely the regime topk2=1 makes "
            "decisive, since one bad pick per GT has no runner-up to correct it."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def preflight(todo):
    """Fail on anything that would make a run a silent no-op."""
    import inspect
    try:
        import ultralytics
        import ultralytics.nn.modules as M
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
        "DySample importable": hasattr(M, "DySample"),
    }
    for k, v in checks.items():
        print(f"  {k:<40}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] the round-8 patch is not installed on this machine.")
        print("  Copy ultralytics26/ultralytics/{utils/loss.py,utils/tal.py,cfg/default.yaml}")
        print("  then re-check:  python verify_patch_v6i.py --ref <...> --runtime")
        return False

    # each requested mechanism must be capable of changing something
    for r in todo:
        p = r["params"]
        if r["expect"]["snl1"]:
            bl = BboxLoss(1)
            bl.l1_scale_p = float(p["l1_scale_p"])
            if not bl.snl1_enabled():
                print(f"  [ABORT] {r['name']}: l1_scale_p={p['l1_scale_p']} but snl1_enabled() is False")
                return False
        if r["expect"]["scb"]:
            a = TAL.TaskAlignedAssigner(topk=10, beta=6.0)
            a.beta_small = float(p["tal_beta_small"])
            a.beta_ref_px = float(p["tal_beta_ref_px"])
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: tal_beta_small set but scb_enabled() is False")
                return False

    print()
    print(f"  CONTROL {CTRL_MEAN:.2f} +- {CTRL_SD:.2f} (n={CTRL_N})   band {BAND[0]:.2f} .. {BAND[1]:.2f}")
    print(f"  LARGE control sd is 2.11 pp - do NOT read a large story from n=1.")
    print(f"  Run y26_dfl3 FIRST. If it lands inside the band, stop.")

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
        # keep SWA's epoch hooks working even though round 8 does not use SWA
        for h in (crit, getattr(crit, "one2many", None), getattr(crit, "one2one", None)):
            bl = getattr(h, "bbox_loss", None) if h is not None else None
            if bl is None:
                continue
            for attr, val in (("epoch", trainer.epoch), ("total_epochs", trainer.epochs)):
                if hasattr(bl, attr):
                    setattr(bl, attr, val)
        if state["verified"] or trainer.epoch < 1:
            return

        o2o = getattr(crit, "one2one", crit)   # the branch that produces predictions
        o2m = getattr(crit, "one2many", crit)
        exp = rc["expect"]

        if exp["snl1"]:
            bl = o2o.bbox_loss
            if not (hasattr(bl, "snl1_enabled") and bl.snl1_enabled()):
                raise RuntimeError(
                    f"{rc['name']}: l1_scale_p requested but snl1_enabled() is False "
                    f"(p={getattr(bl, 'l1_scale_p', None)}, dfl_loss={bl.dfl_loss}). "
                    f"If dfl_loss is not None the model is not reg_max=1 and there is "
                    f"no L1 term to normalise. Aborting.")
            print(f"  [guard] SNL1 live on one2one: p={bl.l1_scale_p}")
        if exp["scb"]:
            for tag, br in (("one2one", o2o), ("one2many", o2m)):
                a = getattr(br, "assigner", None)
                if a is None or not (hasattr(a, "scb_enabled") and a.scb_enabled()):
                    raise RuntimeError(
                        f"{rc['name']}: tal_beta_small requested but scb_enabled() is "
                        f"False on {tag} (beta_small={getattr(a, 'beta_small', None)}). Aborting.")
                print(f"  [guard] SCB live on {tag}: beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px")
        got_dfl = float(getattr(o2o.hyp, "dfl", STOCK_DFL_GAIN))
        if abs(got_dfl - exp["dfl_gain"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: dfl gain is {got_dfl}, expected {exp['dfl_gain']}")
        print(f"  [guard] dfl gain = {got_dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    cfg = save_yaml(DYS_YAML, os.path.join(YAML_DIR, "y26_p2_dysample.yaml"))
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  cfg={cfg}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}")
    for k, v in sorted(rc["params"].items()):
        print(f"    {k:<20}{v}")
    print()
    t0 = time.time()

    model = YOLO(cfg)
    n_dys = sum(1 for m in model.model.modules() if type(m).__name__ == "DySample")
    if n_dys != 1:
        raise RuntimeError(f"{name}: graph has {n_dys} DySample, expected 1")
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] weight transfer failed: {e}")
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
           "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours, "weights": weights,
           "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "r8_params.json"), "w") as f:
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
    print()
    print("=" * 84)
    print("  YOLO26 ROUND 8 - SNL1 / SCB, mechanism-verified")
    print("=" * 84)
    print(f"  CONTROL {CTRL_MEAN:.2f} +- {CTRL_SD:.2f} (n={CTRL_N})   band {BAND[0]:.2f} .. {BAND[1]:.2f}")
    print()
    print(f"{'run':<18}{'mAP50':>9}{'mAP50-95':>10}{'vs ctrl':>9}{'z':>7}  verdict")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        z = (v - CTRL_MEAN) / CTRL_SD
        verdict = "NULL (inside band)" if BAND[0] <= v <= BAND[1] else (
            "REAL GAIN" if v > BAND[1] else "REAL LOSS")
        print(f"{r['name']:<18}{r['test_map50'] * 100:>9.2f}{v:>10.2f}{v - CTRL_MEAN:>+9.2f}{z:>+7.1f}  {verdict}")
    print("-" * 84)
    print(f"{'control (n=10)':<18}{'':>9}{CTRL_MEAN:>10.2f}")
    print()

    by = {r["name"]: r["test_map5095"] * 100 for r in ok}
    if "y26_dfl3" in by:
        v = by["y26_dfl3"]
        if BAND[0] <= v <= BAND[1]:
            print("  PROBE NULL. Doubling the dfl gain changed nothing, so the model is not")
            print("  limited by the L1 term. SNL1 corrects a real bias inside a term that")
            print("  does not matter here - report the analysis, do not chase the runs.")
        else:
            print(f"  PROBE MOVED ({v - CTRL_MEAN:+.2f}). The L1 term is connected; SNL1 is worth measuring.")
    snl1 = [(r["params"]["l1_scale_p"], by[r["name"]]) for r in ok if r["expect"]["snl1"]]
    if snl1:
        print()
        print("  SNL1 axis (p vs overall) - control is p = 0:")
        print(f"    p=0.00  {CTRL_MEAN:.2f}   (control, n={CTRL_N})")
        for p, v in sorted(snl1):
            print(f"    p={p:.2f}  {v:.2f}   {v - CTRL_MEAN:+.2f}")
        print("    monotone in SMALL is evidence even if overall stays inside the band.")
        print("    run CocoEvalAllFolders_luggage.py - overall will not show it.")
    print()
    print("  Do NOT read the large column at n=1 - control sd there is 2.11 pp.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<18} {r['weights']}")
    print()
    print(f"saved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 8 - {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 84)
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
                        "error": str(e), "expect": r["expect"], "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
