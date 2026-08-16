#!/usr/bin/env python3
"""
YOLO26 SCB SWEEP — tune the one mechanism that showed a real effect.

=============================================================================
WHY SCB AND NOT SNL1
=============================================================================
Comparing configs to EACH OTHER is baseline-independent: a low baseline shifts
every row equally. Against the hyperparameter floor (dfl gain 55.37, global
beta 55.37 — two unrelated trivial nudges landing identically):

    y26_scb_b3      55.66   +0.30 above floor   <- SCB
    y26_alpha075    55.59   +0.22
    y26_swa_a06_03  55.59   +0.22               <- best ported mechanism
    y26_snl1_p25    55.49   +0.12
    y26_scb_b4s2    55.49   +0.12
    y26_snl1_p50    55.48   +0.11               <- SNL1

Two pairs of DIFFERENT configs landed on identical values (dfl3 == beta4,
snl1_p25 == scb_b4s2), which puts run-to-run noise around 0.05-0.10 pp rather
than the 0.28 the earlier campaign suggested. On that scale SCB's +0.30 over
the floor is roughly 3 sd, it is the best of all 21 loss configs tried on
YOLO26, and it beats the best ported mechanism.

SNL1 is at the floor (+0.11/+0.12). The dfl-gain probe already explained why:
doubling that term's weight moved only +0.13, so it corrects a real 32x size
bias inside a term the model is not limited by. Not worth more runs.

=============================================================================
WHAT IS ACTUALLY UNTUNED
=============================================================================
SCB has three parameters. One has been varied, and badly:

    tal_beta_small    3.0    varied once. 2.0 was tried ONLY inside
                             y26_scb_b4s2, which ALSO dropped tal_beta 6.0 ->
                             4.0 - so beta_small and beta were confounded and
                             that run tells us nothing about beta_small alone.
    tal_beta_ref_px  64.0    NEVER varied in any run.
    tal_beta          6.0    varied globally (y26_beta4, +0.13) - stays fixed
                             here so every run below is a single-factor change.

=============================================================================
DESIGN — two 3-point axes crossing at the existing y26_scb_b3
=============================================================================
    beta_small axis (ref_px = 64):     2.0  --  3.0*  --  4.0
    ref_px axis (beta_small = 3.0):     32  --   64*  --  128

    * = y26_scb_b3, already measured at 55.66. It is the shared centre, so all
    four runs below are single-factor changes from a known point and no new
    control is needed.

WHAT EACH AXIS ASKS

  beta_small  how far to discount IoU for the smallest objects. 3.0 helped;
              2.0 discounts harder, 4.0 is a gentler touch. A peak at 3.0 means
              the setting is right. Monotone means the optimum is outside the
              range and worth chasing.

  ref_px      where the conditioning stops. At 64 it covers all of COCO-small
              (sqrt-area < 32) and half of medium (32-96). The phase-A
              decomposition put SCB's own effect in SMALL (+0.47 vs the global
              control) and almost nothing in medium (+0.09), so 32 sharpens it
              onto the bucket that responded and 128 extends it into one that
              did not. If 32 matches or beats 64, the mechanism is a
              small-object tool and should be described as one.

RUNS ARE ORDERED BY VALUE, so stopping early still leaves a usable result:

  1 y26_scb_r32    ~30% to beat the centre. Defines WHAT the mechanism is: the
                   decomposition put SCB's effect at +0.47 in small and +0.09 in
                   medium, so half the conditioning at ref_px=64 is spent where
                   nothing happened. If 32 matches or wins, SCB is a small-object
                   tool and should be described as one.
  2 y26_scb_s2     ~30%. More of what worked, and the first CLEAN measurement of
                   beta_small=2.0 (scb_b4s2 used it but also dropped tal_beta).
  3 y26_scb_r128   ~15% to win, but completes an axis never varied in any run.
  4 y26_scb_s4     ~20%. Makes beta_small a curve; also tests whether a mild
                   touch suffices, which is a more portable claim than a constant.

None of these is likely to move the number much. The realistic payoff is being
able to write "broad optimum over ref_px 32-64, beta_small 2-4" instead of
quoting one fitted value.

=============================================================================
READING IT
=============================================================================
    floor (hyperparameter nudges)   55.37
    y26_scb_b3 (centre)             55.66   +0.30
    apparent noise                  ~0.05-0.10 pp

    beats 55.66 by > 0.15   -> better setting, worth adopting
    within 0.10 of 55.66    -> flat, the centre is fine
    below 55.50             -> that direction is wrong

Also read SMALL and PRECISION per-size: SCB's signature is precision (+2.02 on
scb_b3, best in the project) with small (+0.24). If a setting improves overall
but flattens precision, it is not the same mechanism working harder.

Stock yolo26s.pt, b82/640/seed 0 — matches every other loss run.
REQUIRES the patched tal.py (SCB). No new code.

Usage:
    python run_yolo26_scb_sweep_v6i.py
    python run_yolo26_scb_sweep_v6i.py y26_scb_s2 y26_scb_r32
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
PROJECT_DIR = "runs_yolo26_scb_sweep_v6i"
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
FLOOR = 55.37          # dfl3 and beta4, two unrelated trivial nudges, identical
CENTRE = 55.66         # y26_scb_b3: beta_small 3.0, ref_px 64, beta 6.0
NOISE = 0.10           # from two exact ties between different configs

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0,
    l1_scale_p=0.0, tal_beta_small=None,
    box=7.5, cls=0.5, dfl=1.5,
)


def scb(beta_small, ref_px):
    d = dict(_ALL_OFF)
    d.update(tal_beta_small=beta_small, tal_beta_ref_px=ref_px)
    return d


RUNS = [
    {"name": "y26_scb_r32", "axis": "ref_px", "params": scb(3.0, 32.0),
     "label": "ref_px 64 -> 32  (beta_small 3.0 held)",
     "why": "ref_px has never been varied in any run. At 64 the conditioning "
            "covers all of COCO-small (sqrt-area < 32) AND half of medium "
            "(32-96). The phase-A decomposition put SCB's own effect almost "
            "entirely in SMALL (+0.47 vs the global-beta control) and nearly "
            "nothing in medium (+0.09), so restricting it to 32 targets exactly "
            "the bucket that responded. If this matches or beats the centre, SCB "
            "is a small-object mechanism and should be described as one."},

    {"name": "y26_scb_s2", "axis": "beta_small", "params": scb(2.0, 64.0),
     "label": "beta_small 3.0 -> 2.0  (ref_px 64 held)",
     "why": "Discount IoU harder for the smallest objects. beta_small=2.0 has "
            "never been measured cleanly: y26_scb_b4s2 used it but dropped "
            "tal_beta to 4.0 at the same time, confounding the two. This is the "
            "single-factor version. With 3.0 at the centre and 4.0 below it, a "
            "peak at 3.0 says the setting is already right; monotone toward 2.0 "
            "says the optimum is lower than anything tried."},

    {"name": "y26_scb_r128", "axis": "ref_px", "params": scb(3.0, 128.0),
     "label": "ref_px 64 -> 128  (beta_small 3.0 held)",
     "why": "The opposite direction: extend the conditioning across all of "
            "medium and into large. Makes the ref_px axis two-sided, and tests "
            "whether the +0.09 medium result was a ceiling of the mechanism's "
            "reach or a real limit. If 128 wins, the effect is not small-object-"
            "specific and the whole IoU-reliability story needs rewriting - "
            "which is worth finding out before it goes in a paper."},

    {"name": "y26_scb_s4", "axis": "beta_small", "params": scb(4.0, 64.0),
     "label": "beta_small 3.0 -> 4.0  (ref_px 64 held)",
     "why": "The gentler side, needed to make the axis three points rather than "
            "a direction. Without it a good result at 2.0 cannot be told apart "
            "from 'any reduction helps and the amount does not matter'. Also the "
            "cheapest version of the mechanism - if 4.0 matches 3.0, the useful "
            "claim is 'a mild discount suffices', which is a stronger and more "
            "portable statement than a tuned constant."},
]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import v8DetectionLoss
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    ok_cls = hasattr(TAL.TaskAlignedAssigner, "scb_enabled")
    ok_read = "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__)
    print(f"  TaskAlignedAssigner.scb_enabled       {ok_cls}")
    print(f"  loss.py reads tal_beta_small          {ok_read}")
    if not (ok_cls and ok_read):
        print()
        print("  [ABORT] SCB patch not installed.")
        print("  python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False
    for r in todo:
        p = r["params"]
        a = TAL.TaskAlignedAssigner(topk=10, alpha=p["tal_alpha"], beta=p["tal_beta"])
        a.beta_small = float(p["tal_beta_small"])
        a.beta_ref_px = float(p["tal_beta_ref_px"])
        if not a.scb_enabled():
            print(f"  [ABORT] {r['name']}: scb_enabled() False")
            return False
        print(f"  {r['name']:<14} beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px   OK")
    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock)  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  centre y26_scb_b3 = {CENTRE:.2f}   hyperparameter floor {FLOOR:.2f}   "
          f"baseline {BASELINE:.2f}")
    print(f"  apparent noise ~{NOISE:.2f} pp -> beating {CENTRE:.2f} by >0.15 is a real")
    print(f"  improvement; within 0.10 means the centre setting is already fine.")
    print(f"  Read SMALL and PRECISION too - SCB's signature is precision (+2.02).")

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
    """Assert at epoch 1 that SCB is live with exactly the requested settings."""
    state = {"verified": False}
    p = rc["params"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        for tag in ("one2one", "one2many"):
            a = getattr(getattr(crit, tag, crit), "assigner", None)
            if a is None or not (hasattr(a, "scb_enabled") and a.scb_enabled()):
                raise RuntimeError(f"{rc['name']}: SCB not live on {tag}")
            for attr, want in (("beta_small", p["tal_beta_small"]),
                               ("beta_ref_px", p["tal_beta_ref_px"]),
                               ("beta", p["tal_beta"]), ("alpha", p["tal_alpha"])):
                got = float(getattr(a, attr))
                if abs(got - float(want)) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag}.{attr} is {got}, expected {want}")
            print(f"  [guard] {tag}: beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px  alpha {a.alpha}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {name}   [{rc['axis']} axis]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}")
    print(f"  beta_small={rc['params']['tal_beta_small']}  "
          f"ref_px={rc['params']['tal_beta_ref_px']}  beta={rc['params']['tal_beta']}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "axis": rc["axis"], "params": rc["params"], "seed": SEED,
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "scb_sweep_params.json"), "w") as f:
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
    print()
    print("=" * 84)
    print(f"  SCB SWEEP — stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 84)
    print(f"{'run':<16}{'beta_s':>8}{'ref_px':>8}{'mAP50-95':>10}{'vs centre':>11}   verdict")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        dv = v - CENTRE
        vd = ("BETTER" if dv > 0.15 else "wrong direction" if v < 55.50 else "flat")
        print(f"{r['name']:<16}{r['params']['tal_beta_small']:>8.1f}"
              f"{r['params']['tal_beta_ref_px']:>8.0f}{v:>10.2f}{dv:>+11.2f}   {vd}")
    print("-" * 84)
    print(f"{'y26_scb_b3 (centre)':<16}{3.0:>8.1f}{64:>8.0f}{CENTRE:>10.2f}")
    print(f"{'hyperparam floor':<16}{'':>8}{'':>8}{FLOOR:>10.2f}")
    print(f"{'baseline':<16}{'':>8}{'':>8}{BASELINE:>10.2f}")

    print("\n  beta_small axis (ref_px = 64):")
    for lbl, n in (("2.0", "y26_scb_s2"), ("3.0", None), ("4.0", "y26_scb_s4")):
        v = CENTRE if n is None else by.get(n)
        tag = "  <- centre" if n is None else ""
        print(f"    {lbl:>5}   {v:.2f}{tag}" if v else f"    {lbl:>5}   (not run)")
    print("\n  ref_px axis (beta_small = 3.0):")
    for lbl, n in (("32", "y26_scb_r32"), ("64", None), ("128", "y26_scb_r128")):
        v = CENTRE if n is None else by.get(n)
        tag = "  <- centre" if n is None else ""
        print(f"    {lbl:>5}   {v:.2f}{tag}" if v else f"    {lbl:>5}   (not run)")

    best = max(by.values()) if by else 0
    if best > CENTRE + 0.15:
        w = max(by, key=by.get)
        print(f"\n  {w} improves on the centre by {best - CENTRE:+.2f}. Confirm the")
        print(f"  per-size signature matches SCB's (precision up, small up) before")
        print(f"  adopting it - a gain with flat precision is a different effect.")
    else:
        print(f"\n  Nothing beat the centre by more than 0.15. beta_small 3.0 / ref_px 64")
        print(f"  is the setting, and SCB is a one-parameter mechanism with a broad")
        print(f"  optimum - which is a better claim than a tuned constant.")
    print(f"\n  Per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print(f"  Do NOT read LARGE at n=1 - sd 2.11 pp on this dataset.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<16} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print()
    print("=" * 84)
    print(f"  YOLO26 SCB SWEEP — {len(todo)} runs, ~{1.8 * len(todo):.1f} GPU-h")
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
            res.append({"name": r["name"], "axis": r["axis"], "seed": SEED,
                        "hours": float("nan"), "error": str(e), "params": r["params"],
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
