#!/usr/bin/env python3
"""
YOLO26 ROUND 7 — the arch x loss combination, with a HARD no-op guard.

=============================================================================
READ THIS FIRST — WHY THE GUARD EXISTS
=============================================================================
Rounds 4-6 passed LB-TAL budgets while the STOCK loss.py was installed. The
config system accepted `use_lbtal=True`, the run header printed it, and my
preflight validated the budget dict — but stock loss.py never reads the flag,
so the assigner was never built. Ten runs with ten different budget labels were
one configuration:

    overall 55.89 .. 56.46   sd 0.19    <- the budgets changed nothing
    large   53.72 .. 60.66   sd 2.11    <- and large is this noisy regardless

The old preflight checked that tal.py HAS the assigner class and that
default.yaml ACCEPTS the keys. It never checked that loss.py READS the flag.
Those are three different files and only the third one matters.

This runner asserts, at epoch 1 of every run, that the mechanism is actually
live in the constructed criterion — not that it was requested. A silent no-op
now raises instead of producing a plausible number.

=============================================================================
THE CONTROL YOU ALREADY HAVE
=============================================================================
The ten collapsed runs are ten replicates of exactly this architecture at
b32/640/seed0. That is a real control distribution:

    dysample arch, stock loss, n=10:   56.08 +- 0.19   overall mAP50-95
                                       52.14 +- 0.23   small
                                       57.21 +- 2.11   large

So a candidate needs NO replicates of its own. A single run outside
56.08 +- 0.38 (2 sd) is a real effect; inside is a real null.

    DECISION BAND      > 56.46  or  < 55.70   -> something happened
                       55.70 .. 56.46         -> null, mechanism does nothing

Note that 56.46 is also the best of the ten draws. That is the correct bar: to
claim a gain, a config must beat the luckiest replicate of doing nothing.

LARGE IS NOT USABLE HERE. sd 2.11 means differences under ~4 pp are unreadable.
Do not read a large-object story out of these four runs.

=============================================================================
THE FOUR — a screen, not a sweep
=============================================================================
Every loss result on YOLO26 so far came from the stock 3-LEVEL model at b82 and
was never seed-checked (best +0.35, 8/13 above anchor, p=0.29). So there is no
trustworthy ranking to refine. These four ask only: does either mechanism move
anything at all on THIS architecture?

  1 swa0603    SWA sqrt 0.6->0.3 px48 boost 2.0   the size-weighted regression term
  2 lbuni      LB-TAL uniform                     per-level budget, no constants
  3 lbp2k2     LB-TAL fixed {4:2,8:3,16:4,32:4}   the budget rounds 4-6 never ran
  4 swa_lb     both                               interaction, if either works alone

If all four land in the band, both ports are null on YOLO26 and that is a
finding worth reporting — the same code gave +0.86 on YOLOv12, and a clean
negative transfer result is publishable. If one moves, sweep around it next,
with a known noise floor to size the sweep against.

Architecture FIXED at the DySample P2 variant, byte-identical to the yaml behind
all ten control runs. b32, 640, seed 0.
REQUIRES the PATCHED loss.py, tal.py, default.yaml and nn modules.

Usage:
    python run_yolo26_round7_v6i.py
    python run_yolo26_round7_v6i.py y26_dys_lbp2k2
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
PROJECT_DIR = "runs_yolo26_round7_v6i"
YAML_DIR = "arch_yamls_y26_r7"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# 10-replicate control: dysample arch, stock loss, b32/640/seed0
CTRL_MEAN, CTRL_SD, CTRL_N = 56.08, 0.19, 10
CTRL_SMALL, CTRL_LARGE = 52.14, 57.21
BAND = (CTRL_MEAN - 2 * CTRL_SD, CTRL_MEAN + 2 * CTRL_SD)   # 55.70 .. 56.46
BASELINE_B82 = 55.24   # yolo26_custom-9, n=1, DIFFERENT BATCH — not directly comparable


def _swa(a0, a1, boost=2.0, px=48, mode="sqrt", **extra):
    return dict(alpha_start=a0, alpha_end=a1, alpha_min=0.0, alpha_max=1.0,
                area_weight_mode=mode, area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost, **extra)


def _lb(mode="uniform", level_topk=None):
    return dict(use_lbtal=True, lbtal_mode=mode, lbtal_level_topk=level_topk,
                lbtal_min_level_k=1, lbtal_quality_gate=0.0)

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
    {"name": "y26_dys_swa0603", "expect": {"swa": True, "lbtal": False},
     "params": _swa(0.6, 0.3),
     "label": "SWA sqrt 0.6->0.3 px48 boost 2.0, stock assigner",
     "why": "The size-weighted regression term on its own. It reweights the "
            "positives that already exist rather than changing which anchors are "
            "positive, so it is the mechanism least entangled with the P2 head. "
            "On the 3-level model at b82 this was the best loss config measured "
            "(+0.35), but that ranking came from 13 unreplicated runs with p=0.29 "
            "and cannot be trusted — treat this as untested."},

    {"name": "y26_dys_lbuni", "expect": {"swa": False, "lbtal": True},
     "params": _lb("uniform"),
     "label": "LB-TAL uniform: ceil(topk/n_levels) per level, no dataset constants",
     "why": "The principled form. On a 4-level model every level gets "
            "ceil(10/4)=3 slots, summing to 12 and capped at topk=10, so it is a "
            "pure re-allocation with no fitted numbers. If per-level budgeting "
            "helps at all on YOLO26, this is the version that should show it and "
            "the version that needs no justification in a paper."},

    {"name": "y26_dys_lbp2k2", "expect": {"swa": False, "lbtal": True},
     "params": _lb("fixed", {4: 2, 8: 3, 16: 4, 32: 4}),
     "label": "LB-TAL fixed {4:2, 8:3, 16:4, 32:4} — the budget rounds 4-6 never actually ran",
     "why": "The exact budget labelled y26_p2k2_hi, whose 56.46 turned out to be "
            "the luckiest of ten replicates of doing nothing. Running it for real "
            "closes that loop, and lets you state what the budget does rather "
            "than withdrawing a number with nothing to replace it."},

    {"name": "y26_dys_swa_lb", "expect": {"swa": True, "lbtal": True},
     "params": _swa(0.6, 0.3, **_lb("fixed", {4: 2, 8: 3, 16: 4, 32: 4})),
     "label": "SWA 0.6->0.3 + LB-TAL fixed {4:2,8:3,16:4,32:4}",
     "why": "The two act at different stages — LB-TAL picks which anchors are "
            "positive, SWA weights them once picked — so they can compose. On "
            "YOLOv12 the best combination beat both parents. Only interpretable "
            "if a parent moves; if runs 1 and 2 are both null, read this as a "
            "fourth replicate of the control."},
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

    ok_cls = hasattr(TAL, "LevelBalancedTaskAlignedAssigner")
    ok_swa = hasattr(BboxLoss, "swa_weight")
    ok_read = "use_lbtal" in inspect.getsource(v8DetectionLoss.__init__)
    print(f"  tal.py  has LevelBalancedTaskAlignedAssigner : {ok_cls}")
    print(f"  BboxLoss has swa_weight                      : {ok_swa}")
    print(f"  loss.py READS use_lbtal in v8DetectionLoss   : {ok_read}   <-- the one rounds 4-6 missed")
    if not (ok_cls and ok_swa and ok_read):
        print()
        print("  [ABORT] the PATCHED loss.py is not installed. Every run would be a")
        print("  replicate of the control. Install it, then re-run:")
        print("      python verify_patch_v6i.py --ref <ultralytics26/ultralytics> --install --runtime")
        return False
    if not hasattr(M, "DySample"):
        print("  [ABORT] DySample not importable")
        return False

    for r in todo:
        p = r["params"]
        if r["expect"]["swa"]:
            bl = BboxLoss(1)
            for k, v in p.items():
                if hasattr(bl, k):
                    setattr(bl, k, v)
            if not bl.swa_enabled():
                print(f"  [ABORT] {r['name']}: swa params given but swa_enabled() is False (no-op)")
                return False
        if r["expect"]["lbtal"]:
            A = TAL.LevelBalancedTaskAlignedAssigner
            lt = p.get("lbtal_level_topk")
            a = A(topk=10, level_topk_mode=p["lbtal_mode"], level_topk=lt, min_level_k=1)
            b = a._per_level_budget([4.0, 8.0, 16.0, 32.0])
            if lt and {float(k): v for k, v in b.items()} != {float(k): v for k, v in lt.items()}:
                print(f"  [ABORT] {r['name']}: budget resolved to {b}, expected {lt}")
                return False
            print(f"  {r['name']:<18} budget -> { {int(k): v for k, v in b.items()} }")

    print()
    print(f"  CONTROL  {CTRL_MEAN:.2f} +- {CTRL_SD:.2f}  (n={CTRL_N}, same arch, stock loss, b32)")
    print(f"  DECISION BAND  {BAND[0]:.2f} .. {BAND[1]:.2f}   inside = null, outside = real")
    print(f"  LARGE has sd 2.11 in the control - do NOT read a large story from n=1.")

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
    """Feed the epoch to SWA AND assert the mechanism is live in the built criterion."""
    state = {"verified": False}

    def holders(crit):
        for h in (crit, getattr(crit, "one2many", None), getattr(crit, "one2one", None)):
            if h is not None and getattr(h, "bbox_loss", None) is not None:
                yield h

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        n = 0
        for h in holders(crit):
            for attr, val in (("epoch", trainer.epoch), ("total_epochs", trainer.epochs)):
                if hasattr(h.bbox_loss, attr):
                    setattr(h.bbox_loss, attr, val)
            n += 1
        if n == 0:
            if trainer.epoch >= 1:
                raise RuntimeError("epoch callback reached NO BboxLoss - criterion layout changed")
            return
        if state["verified"] or trainer.epoch < 1:
            return

        # ---- THE GUARD: assert the mechanism is LIVE, not merely requested ----
        o2m = getattr(crit, "one2many", crit)   # LB-TAL installs here (topk2 is None)
        if rc["expect"]["lbtal"]:
            got = type(getattr(o2m, "assigner", None)).__name__
            if got != "LevelBalancedTaskAlignedAssigner":
                raise RuntimeError(
                    f"{rc['name']}: use_lbtal was requested but one2many.assigner is "
                    f"{got}. The patched loss.py is NOT installed - this run would be a "
                    f"replicate of the control. Aborting instead of producing a number.")
            print(f"  [guard] LB-TAL live: {got}")
        if rc["expect"]["swa"]:
            bl = o2m.bbox_loss
            if not (hasattr(bl, "swa_enabled") and bl.swa_enabled()):
                raise RuntimeError(
                    f"{rc['name']}: SWA params requested but swa_enabled() is False "
                    f"(alpha_start={getattr(bl, 'alpha_start', None)}). Aborting.")
            print(f"  [guard] SWA live: alpha(t={trainer.epoch})={bl.get_dynamic_alpha():.3f} "
                  f"mode={bl.area_weight_mode}")
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
        print(f"    {k:<22}{v}")
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
        with open(os.path.join(save_dir, "r7_params.json"), "w") as f:
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
    print("  YOLO26 ROUND 7 - arch x loss, mechanism-verified")
    print("=" * 84)
    print(f"  CONTROL {CTRL_MEAN:.2f} +- {CTRL_SD:.2f} (n={CTRL_N})   "
          f"decision band {BAND[0]:.2f} .. {BAND[1]:.2f}")
    print()
    print(f"{'run':<20}{'mAP50':>9}{'mAP50-95':>10}{'vs ctrl':>9}{'z':>7}  verdict")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        z = (v - CTRL_MEAN) / CTRL_SD
        verdict = "NULL (inside band)" if BAND[0] <= v <= BAND[1] else (
            "REAL GAIN" if v > BAND[1] else "REAL LOSS")
        print(f"{r['name']:<20}{r['test_map50'] * 100:>9.2f}{v:>10.2f}"
              f"{v - CTRL_MEAN:>+9.2f}{z:>+7.1f}  {verdict}")
    print("-" * 84)
    print(f"{'control (n=10)':<20}{'':>9}{CTRL_MEAN:>10.2f}")
    print()

    gains = [r for r in ok if r["test_map5095"] * 100 > BAND[1]]
    if not gains:
        print("  ALL NULL. Neither SWA nor LB-TAL moves this architecture beyond the")
        print("  noise of doing nothing. Given the same code gave +0.86 pp on YOLOv12,")
        print("  that is a clean negative-transfer result and worth reporting as one.")
        print(f"  Report the architecture alone: {CTRL_MEAN:.2f} +- {CTRL_SD:.2f} (n={CTRL_N}).")
    else:
        print(f"  MOVED: {', '.join(r['name'] for r in gains)}")
        print(f"  Sweep around the winner next. With sd {CTRL_SD:.2f} you need steps > 0.4 pp")
        print("  to resolve, and 2 replicates of the winner before reporting a number.")
    print()
    print("  Do NOT read the large column at n=1 - control sd there is 2.11 pp.")
    print(f"  Baseline {BASELINE_B82:.2f} is n=1 at b82 and NOT batch-matched to these.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print()
    print(f"saved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 7 - {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
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
                        "error": str(e), "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
