#!/usr/bin/env python3
"""
YOLO26 LOSS STUDY — 5 loss experiments on stock, best 2 carried onto the arch.

=============================================================================
PHASE A — 5 loss experiments, STOCK yolo26s, everything else OFF
=============================================================================
Base = _ALL_OFF: stock CIoU + L1 + BCE, stock TAL (topk 10 / a 0.5 / b 6.0),
gains 7.5 / 0.5 / 1.5, no SWA, no LB-TAL. Model is the shipped `yolo26s.pt` —
no yaml, no P2 head, no DySample. Each run overrides ONE thing, so the delta
against the published baseline is that mechanism's own effect.

Reference (already measured, do not re-run):
    yolo26_custom-9   55.24   stock baseline
    y26_swa_a06_03    55.59   best of the 14-run ported campaign (b82)
    14-run campaign    8/14 above baseline, mean -0.05, sign test p = 0.40

  y26_dfl3      dfl 1.5 -> 3.0        PROBE — is the L1 term even connected?
  y26_snl1_p25  l1_scale_p 0.25       SNL1, conservative
  y26_snl1_p50  l1_scale_p 0.50       SNL1, half-way to scale-invariant
  y26_scb_b3    tal_beta_small 3.0    size-conditioned beta
  y26_beta4     tal_beta 6.0 -> 4.0   CONTROL for scb: is it the SIZE-CONDITIONING
                                      or just a lower global beta?

y26_beta4 is what makes y26_scb_b3 interpretable. If both improve by the same
amount, SCB's benefit is simply "beta 6.0 is too high on this dataset" and the
size-conditioning adds nothing — a much weaker claim, and one you would rather
find out now than in review.

=============================================================================
PHASE B — the best 2 from phase A, on the DySample architecture
=============================================================================
Phase B reads phase A's summary.json, takes the top 2 by test mAP50-95, and
re-runs them on the P2 + DySample graph at BATCH_ARCH.

WHY THIS IS NOT REDUNDANT WITH PHASE A. The two axes did not compose on either
detector. On YOLOv12: arch alone +1.21, loss alone +0.86, both together +0.84 —
the stride-4 head supplies what SWA was compensating for. On YOLO26 the ported
mechanisms went from null on the stock model to actively harmful on the P2 one
(-0.43 to -0.82, z -2.2 to -4.3). So a loss gain on stock does NOT imply a gain
on the architecture, and phase B is the run that tells you whether your final
model should carry the loss change at all.

BATCH_ARCH is separate from BATCH_LOSS: the P2 head adds a stride-4 level
(160x160 at 640 px) and needs more memory. Set it to whatever fits. Phase B
numbers are internally comparable and comparable to the b32 arch runs if you
leave it at 32; they are NOT comparable to phase A across a batch change.

=============================================================================
BOUND YOUR EXPECTATIONS
=============================================================================
    term          gain   share   scale behaviour
    box  (CIoU)    7.5   78.9%   IoU is a ratio -> already scale-INVARIANT
    cls  (BCE)     0.5    5.3%   size-independent
    dfl  (L1)      1.5   15.8%   the biased term SNL1 corrects

SNL1 targets a real 32x size bias (YOLO26 normalises the L1 target by IMAGE
size, so gradient scales with object size) but that bias lives in ~16% of the
loss while the dominant regression term is already scale-fair. Do not expect
YOLOv12's +0.86.

Read SMALL per-size. Overall resolves ~0.4 pp. Large is unusable at n=1: ten
replicates of one config on this dataset spanned 53.72..60.66, sd 2.11 pp.

REQUIRES the patched loss.py / tal.py / default.yaml (SNL1 + SCB).

Usage:
    python run_yolo26_loss_study_v6i.py                    # phase A then B
    python run_yolo26_loss_study_v6i.py --phase a
    python run_yolo26_loss_study_v6i.py --phase b
    python run_yolo26_loss_study_v6i.py --phase b --pick y26_snl1_p25,y26_scb_b3
    python run_yolo26_loss_study_v6i.py --phase a --only y26_dfl3
"""

import argparse
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
PROJECT_DIR = "runs_yolo26_loss_study_v6i"
YAML_DIR = "arch_yamls_y26_lossstudy"
EPOCHS = 70
IMG_SIZE = 640                 # eval MUST also be 640 (the 896 lesson)

# ---- SET THESE ---------------------------------------------------------------
BATCH_LOSS = 82                # phase A, stock 3-level model
BATCH_ARCH = 32                # phase B, P2 + DySample. 32 keeps it comparable
                               # to every recorded arch run; lower it if you OOM.
N_CARRY = 2                    # how many phase-A configs go into phase B
# ------------------------------------------------------------------------------

WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24               # yolo26_custom-9, stock, published
ARCH_REF = 56.08               # DySample arch, stock loss, n=10, b32, sd 0.19

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0,
    l1_scale_p=0.0, tal_beta_small=None,
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d

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
     "params": cfg(dfl=3.0),
     "label": "PROBE — dfl gain 1.5 -> 3.0, no code change",
     "why": "Answers 'is this lever connected?' before the others are believed. The "
            "L1 term carries 15.8% of the loss gain while CIoU (78.9%) is already "
            "scale-invariant, so the model may not be limited by it at all. Flat "
            "here means SNL1 corrects a real bias inside a term that does not "
            "matter, and the p=0.25/0.50 results should be read as nulls rather "
            "than as evidence against the mechanism."},

    {"name": "y26_snl1_p25", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": cfg(l1_scale_p=0.25),
     "label": "SNL1 p=0.25 — partial removal of the 32x size bias",
     "why": "Conservative point: ~1.6x more L1 gradient on 8 px boxes, ~0.68x on "
            "256 px, divisor renormalised to mean 1 so the term's magnitude and the "
            "dfl gain stay comparable to stock. Preferred over p=1.0 as a starting "
            "point because full scale-invariance puts 10.5x extra gradient on the "
            "smallest boxes and large-object AP is already the weak column here."},

    {"name": "y26_snl1_p50", "expect": {"snl1": True, "scb": False, "dfl_gain": 1.5},
     "params": cfg(l1_scale_p=0.5),
     "label": "SNL1 p=0.50 — half-way to scale-invariant",
     "why": "Second point, so p is a curve and not a guess: 2.8x on 8 px, 0.50x on "
            "256 px. With stock at p=0 that is three points. A monotone trend in "
            "SMALL is evidence even if overall is flat. If LARGE collapses between "
            "0.25 and 0.50 then the useful range is below 0.25 and p=1.0 is dead."},

    {"name": "y26_scb_b3", "expect": {"snl1": False, "scb": True, "dfl_gain": 1.5},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0),
     "label": "SCB — beta 3.0 -> 6.0 over sqrt(area) 0 -> 64 px",
     "why": "Changes WHICH anchor is selected rather than how the residual is "
            "weighted, so it is independent of SNL1. With topk2=1 the alignment "
            "metric picks THE single anchor per GT in the branch that produces "
            "every prediction, and IoU is a high-variance ranking signal on small "
            "boxes — one pixel of shift swings it. Lower beta there trusts "
            "classification score more where IoU is least reliable."},

    {"name": "y26_beta4", "expect": {"snl1": False, "scb": False, "dfl_gain": 1.5},
     "params": cfg(tal_beta=4.0),
     "label": "CONTROL for SCB — global beta 6.0 -> 4.0, no size conditioning",
     "why": "The run that makes y26_scb_b3 interpretable. beta=6.0 is a COCO "
            "default never varied in 29 recorded YOLO26 runs, so if simply lowering "
            "it globally gives the same gain as SCB, then SCB's size-conditioning "
            "contributes nothing and the finding collapses to 'beta was mistuned'. "
            "Cheap insurance against a reviewer asking exactly this."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def preflight(todo, phase):
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
    }
    if phase == "b":
        checks["DySample importable"] = hasattr(M, "DySample")
    for k, v in checks.items():
        print(f"  {k:<40}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] patch not installed:")
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
            a = TAL.TaskAlignedAssigner(topk=10, beta=float(p.get("tal_beta", 6.0)))
            a.beta_small = float(p["tal_beta_small"])
            a.beta_ref_px = float(p.get("tal_beta_ref_px", 64.0))
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: scb_enabled() False")
                return False

    b = BATCH_LOSS if phase == "a" else BATCH_ARCH
    model = f"{MODEL_WEIGHTS} (STOCK)" if phase == "a" else "P2 + DySample yaml"
    ref = f"baseline {BASELINE:.2f}" if phase == "a" else f"arch ref {ARCH_REF:.2f} (n=10, sd 0.19, b32)"
    print()
    print(f"  PHASE {phase.upper()}   model={model}   batch={b}   imgsz={IMG_SIZE}   seed={SEED}")
    print(f"  read against: {ref}")
    if phase == "b" and BATCH_ARCH != 32:
        print(f"  [!] BATCH_ARCH={BATCH_ARCH} != 32 — results are NOT comparable to the")
        print(f"      recorded arch runs ({ARCH_REF:.2f} and the 13-config ablation).")
    print(f"  Read SMALL per-size. Large is unusable at n=1 (sd 2.11 pp).")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    names = [rn(r["name"], phase) for r in todo]
    clash = sorted({f"{n} -> {bd}" for n in names for bd in bases
                    if os.path.isdir(os.path.join(bd, n))}) if not OVERWRITE_EXISTING else []
    if clash:
        print()
        print("  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def rn(name, phase):
    """Run directory name: phase B is suffixed so A and B never collide."""
    return name if phase == "a" else f"{name}_arch"


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
        exp, bl = rc["expect"], getattr(crit, "one2one", crit).bbox_loss

        if bl.dfl_loss is not None:
            raise RuntimeError(f"{rc['name']}: reg_max > 1 — no L1 term, SNL1 is meaningless here")
        if exp["snl1"] and not (hasattr(bl, "snl1_enabled") and bl.snl1_enabled()):
            raise RuntimeError(
                f"{rc['name']}: l1_scale_p requested but snl1_enabled() is False "
                f"(p={getattr(bl, 'l1_scale_p', None)}). Aborting.")
        if not exp["snl1"] and getattr(bl, "l1_scale_p", 0.0) != 0.0:
            raise RuntimeError(f"{rc['name']}: expected SNL1 OFF but l1_scale_p={bl.l1_scale_p}")
        if exp["snl1"]:
            print(f"  [guard] SNL1 live: p={bl.l1_scale_p}")

        for tag, br in (("one2one", o2o), ("one2many", o2m)):
            a = getattr(br, "assigner", None)
            live = a is not None and hasattr(a, "scb_enabled") and a.scb_enabled()
            if exp["scb"] and not live:
                raise RuntimeError(f"{rc['name']}: SCB requested but scb_enabled() False on {tag}")
            if not exp["scb"] and live:
                raise RuntimeError(f"{rc['name']}: expected SCB OFF but live on {tag}")
            if exp["scb"]:
                print(f"  [guard] SCB live on {tag}: beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px")

        got_b = float(getattr(o2o.assigner, "beta", 6.0))
        want_b = float(rc["params"].get("tal_beta", 6.0))
        if abs(got_b - want_b) > 1e-6:
            raise RuntimeError(f"{rc['name']}: tal_beta is {got_b}, expected {want_b}")
        got_d = float(getattr(o2o.hyp, "dfl", 1.5))
        if abs(got_d - exp["dfl_gain"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: dfl gain is {got_d}, expected {exp['dfl_gain']}")
        print(f"  [guard] dfl={got_d}  beta={got_b}  assigner={type(o2o.assigner).__name__}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc, phase):
    name = rn(rc["name"], phase)
    batch = BATCH_LOSS if phase == "a" else BATCH_ARCH
    if phase == "a":
        src, kind = MODEL_WEIGHTS, "stock yolo26s"
    else:
        src = save_yaml(DYS_YAML, os.path.join(YAML_DIR, "y26_p2_dysample.yaml"))
        kind = "P2 + DySample"
    print()
    print("=" * 78)
    print(f"  PHASE {phase.upper()}  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={src} ({kind})  imgsz={IMG_SIZE}  batch={batch}  epochs={EPOCHS}  seed={SEED}")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff}")
    print()
    t0 = time.time()

    model = YOLO(src)
    if phase == "b":
        n_dys = sum(1 for m in model.model.modules() if type(m).__name__ == "DySample")
        if n_dys != 1:
            raise RuntimeError(f"{name}: graph has {n_dys} DySample, expected 1")
        try:
            model.load(MODEL_WEIGHTS)
        except Exception as e:
            print(f"  [warn] weight transfer failed: {e}")
    state = attach_callbacks(model, rc)

    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "base": rc["name"], "phase": phase, "params": rc["params"],
           "expect": rc["expect"], "seed": SEED, "model": src, "imgsz": IMG_SIZE,
           "batch": batch, "hours": hours, "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "loss_study_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
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


def summary_path(phase):
    return os.path.join(PROJECT_DIR, f"summary_phase_{phase}.json")


def load_phase(phase):
    p = summary_path(phase)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def pick_top(n=N_CARRY):
    res = [r for r in load_phase("a") if r.get("test_map5095") == r.get("test_map5095")]
    res = [r for r in res if r["base"] != "y26_dfl3"]  # the probe is diagnostic, not a config
    res.sort(key=lambda r: -r["test_map5095"])
    return [r["base"] for r in res[:n]]


def summarise(res, phase):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    ref = BASELINE if phase == "a" else ARCH_REF
    reflbl = "baseline 55.24" if phase == "a" else f"arch ref {ARCH_REF:.2f} (n=10, sd 0.19)"
    batch = BATCH_LOSS if phase == "a" else BATCH_ARCH
    print()
    print("=" * 86)
    print(f"  PHASE {phase.upper()} — {'stock yolo26s' if phase == 'a' else 'P2 + DySample'}, "
          f"b{batch}/{IMG_SIZE}, seed {SEED}")
    print("=" * 86)
    print(f"{'run':<18}{'mAP50':>9}{'mAP50-95':>10}{'vs ref':>9}{'h':>6}   verdict")
    print("-" * 86)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        d = v - ref
        vd = "above ref" if d > 0.4 else ("below ref" if d < -0.4 else "flat (< 0.4 pp)")
        print(f"{r['name']:<18}{r['test_map50'] * 100:>9.2f}{v:>10.2f}{d:>+9.2f}{r['hours']:>6.1f}   {vd}")
    print("-" * 86)
    print(f"  reference: {reflbl}")

    by = {r["base"]: r["test_map5095"] * 100 for r in ok}
    if phase == "a":
        if "y26_dfl3" in by:
            d = by["y26_dfl3"] - BASELINE
            print(f"\n  PROBE  dfl 1.5 -> 3.0 moved {d:+.2f} pp")
            if abs(d) < 0.2:
                print("    Flat. The model is not limited by the L1 term, so SNL1 corrects a")
                print("    real bias inside a term that does not matter. Read the p results as")
                print("    nulls about the TERM, not about the mechanism.")
        snl1 = sorted((r["params"]["l1_scale_p"], by[r["base"]]) for r in ok if r["expect"]["snl1"])
        if snl1:
            print("\n  SNL1 axis (stock is p = 0):")
            print(f"    p=0.00  {BASELINE:.2f}   (published baseline)")
            for p, v in snl1:
                print(f"    p={p:.2f}  {v:.2f}   {v - BASELINE:+.2f}")
            print("    Monotone in SMALL is evidence even if overall is flat.")
        if "y26_scb_b3" in by and "y26_beta4" in by:
            s, g = by["y26_scb_b3"] - BASELINE, by["y26_beta4"] - BASELINE
            print(f"\n  SCB vs its CONTROL")
            print(f"    y26_scb_b3  (size-conditioned)  {s:+.2f}")
            print(f"    y26_beta4   (global beta 4.0)   {g:+.2f}")
            print(f"    difference                      {s - g:+.2f}")
            if abs(s - g) < 0.3:
                print("    Same within noise -> the gain is 'beta 6.0 was mistuned', NOT")
                print("    size-conditioning. Report it as a hyperparameter finding.")
            elif s > g:
                print("    SCB beats the flat-beta control -> the size-conditioning itself helps.")
        top = pick_top()
        print(f"\n  PHASE B WILL CARRY: {top or '(nothing — no completed phase A runs)'}")
        print(f"    python {os.path.basename(__file__)} --phase b")
        print(f"    override with --pick name1,name2 ; BATCH_ARCH is currently {BATCH_ARCH}")
    else:
        a = {r["base"]: r["test_map5095"] * 100 for r in load_phase("a")
             if r.get("test_map5095") == r.get("test_map5095")}
        print("\n  DOES THE LOSS SURVIVE THE ARCHITECTURE?")
        print(f"    {'config':<16}{'phase A':>10}{'vs base':>9}{'phase B':>10}{'vs arch':>9}")
        for r in sorted(ok, key=lambda x: -x["test_map5095"]):
            b_ = r["base"]
            av = f"{a[b_]:.2f}" if b_ in a else "  -  "
            ad = f"{a[b_] - BASELINE:+.2f}" if b_ in a else "  -  "
            bv = r["test_map5095"] * 100
            print(f"    {b_:<16}{av:>10}{ad:>9}{bv:>10.2f}{bv - ARCH_REF:>+9.2f}")
        print("\n    Gains that vanish here are the same non-composition seen on both")
        print("    detectors: arch +1.21 / loss +0.86 / together +0.84 on YOLOv12, and")
        print("    ported mechanisms going null -> harmful on YOLO26's P2 model.")
        if BATCH_ARCH != 32:
            print(f"\n    [!] BATCH_ARCH={BATCH_ARCH}: 'vs arch' is NOT valid, the {ARCH_REF:.2f}")
            print(f"        reference was measured at b32.")

    print("\n  Per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print("  Do NOT read LARGE at n=1 — sd 2.11 pp on this dataset.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<18} {r['weights']}")


def execute(todo, phase):
    if not preflight(todo, phase):
        return False
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = summary_path(phase)
    res = []
    for r in todo:
        try:
            res.append(run_one(r, phase))
        except Exception as e:
            print(f"  [ERROR] run '{rn(r['name'], phase)}' failed: {e}")
            res.append({"name": rn(r["name"], phase), "base": r["name"], "phase": phase,
                        "seed": SEED, "hours": float("nan"), "error": str(e),
                        "expect": r["expect"], "params": r["params"],
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, phase)
    print(f"\nsaved -> {out_path}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--only", default="", help="phase A: comma-separated run names")
    ap.add_argument("--pick", default="", help="phase B: comma-separated names, overrides auto-select")
    args = ap.parse_args()
    BY = {r["name"]: r for r in RUNS}

    if args.phase in ("a", "ab"):
        only = set(x for x in args.only.split(",") if x)
        todo = [r for r in RUNS if not only or r["name"] in only]
        print()
        print("=" * 86)
        print(f"  PHASE A — {len(todo)} loss experiments on stock {MODEL_WEIGHTS}, "
              f"~{1.8 * len(todo):.1f} GPU-h")
        print(f"  {', '.join(r['name'] for r in todo)}")
        print("=" * 86)
        if not execute(todo, "a") and args.phase == "ab":
            sys.exit(1)

    if args.phase in ("b", "ab"):
        picks = [x for x in args.pick.split(",") if x] or pick_top()
        bad = [p for p in picks if p not in BY]
        if bad:
            sys.exit(f"[ABORT] unknown config(s) for phase B: {bad}")
        if not picks:
            sys.exit("[ABORT] phase B has nothing to carry — run phase A first, "
                     "or pass --pick name1,name2")
        todo = [BY[p] for p in picks]
        print()
        print("=" * 86)
        print(f"  PHASE B — {len(todo)} configs on the P2 + DySample arch at batch "
              f"{BATCH_ARCH}, ~{1.8 * len(todo):.1f} GPU-h")
        print(f"  carrying: {', '.join(picks)}"
              f"{'  (auto-selected from phase A)' if not args.pick else '  (--pick)'}")
        print("=" * 86)
        execute(todo, "b")
