#!/usr/bin/env python3
"""
PEU-DFL ablation — per-edge uncertainty attenuation.

=============================================================================
WHAT MOTIVATED THIS (measured, not assumed)
=============================================================================
diag_per_edge_dfl.py on r10_nwd_fixedc/best.pt, 53,185 foreground anchors:

    edge      resid(bins)   resid(px)   resid / box dimension
    left        0.117         1.34            5.43%
    top         0.430         4.96            6.67%
    right       0.126         1.44            5.61%
    bottom      0.192         2.19            3.40%

Two conclusions:

1. NOT QUANTISATION-LIMITED. Every residual is SUB-BIN (0.12-0.43 of a bin) and
   saturation is 0.00%. The DFL expectation already interpolates far below bin
   width, so neither finer bins (A-DFL) nor per-edge reweighting by aspect ratio
   (AR-DFL) addresses the actual error.

2. THE TOP EDGE IS THE PROBLEM. It is 2.26x worse than the bottom edge and 3.6x
   worse than the width edges. Aspect ratio, scale and quantisation do not
   explain that asymmetry. The physical explanation does: the bottom edge is
   ground contact — sharp and unambiguous; the top edge is handles, telescoping
   poles, straps and occlusion by the person carrying the bag. It is ambiguous
   for the annotator as much as for the network.

=============================================================================
THE MECHANISM
=============================================================================
Forcing a network to fit an intrinsically ambiguous target is label-noise
fitting. The standard remedy is learned attenuation (Kendall & Gal 2017;
KL-Loss, He et al. 2019; Gaussian YOLO) — which normally requires an extra head
predicting variance.

DFL ALREADY EMITS A DISTRIBUTION PER EDGE. Its variance is a free aleatoric
uncertainty estimate that every implementation discards by taking only the
expectation. PEU-DFL uses it:

    mu_e  = sum_i p_i * i                 <- already the decode
    var_e = sum_i p_i * (i - mu_e)^2      <- free, currently thrown away
    s_e   = log var_e
    L_e  <- L_e * clip(exp(-beta * s_e)) + lambda * s_e

At the optimum var_e = L_e / lambda, so variance tracks difficulty: hard edges
(top) get attenuated, easy ones (bottom) do not. Self-calibrating, ZERO new
parameters, ZERO architecture change, so pretrained weights load unchanged.

Weights are MEAN-NORMALISED to 1 -> PEU redistributes DFL weight across edges
without changing total DFL magnitude. Without that, any gain is confounded with
simply raising the dfl gain (the flaw in the earlier AR-DFL ablation).

=============================================================================
RUNS
=============================================================================
  peu_anchor      stock DFL — the beat-target
  peu_b05         beta=0.5, lambda=1.0  (default; mild attenuation)
  peu_b10         beta=1.0, lambda=1.0  (full Kendall form)
  peu_b05_nodet   beta=0.5, variance NOT detached (gradient flows through it)
  peu_fixed_top   CONTROL: fixed per-edge weights matching the MEASURED
                  residual ratio, no uncertainty at all. If this matches the
                  learned version, the adaptivity claim is dead and the finding
                  is simply "down-weight the top edge".
  peu_b05_lam2    lambda=2.0 — stronger pull against declaring uncertainty

REQUIRES loss.py (this folder, with PEU) at ultralytics/utils/loss.py and the
adfl_* + peu_* keys whitelisted in cfg/default.yaml.

Usage:
  python run_peu_ablation.py
  python run_peu_ablation.py peu_anchor peu_b05 peu_fixed_top   # the decisive 3
  python run_peu_ablation.py --seeds
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_peu"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# measured relative residuals (left, top, right, bottom) from the diagnostic
MEASURED_REL = (5.43, 6.67, 5.61, 3.40)

_BASE = dict(
    use_satal=False, tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    swa_mode="scale", swa_alpha=0.0, swa_boost=1.0, swa_size_axis="width",
    box_loss_type="ciou",
    use_nwd=False, use_class_weights=False, use_vfl=False, use_loss_clip=False,
    use_ardfl=False, use_adfl=False,
)
_PEU_OFF = dict(use_peu=False, peu_beta=0.5, peu_lambda=1.0, peu_detach=True,
                peu_warmup_epochs=5, peu_min_var=0.25, peu_w_clip=3.0, peu_log=True)


def cfg(**over):
    d = dict(_BASE)
    d.update(_PEU_OFF)
    d.update(over)
    return d


def peu(**over):
    return cfg(use_peu=True, **over)


# The control uses AR-DFL's fixed per-edge weights, set to the INVERSE of the
# measured residuals and mean-normalised — i.e. exactly what a perfectly
# informed fixed scheme would do. loss.py forbids use_peu + use_ardfl together,
# so this run is unambiguously the fixed-weight arm.
_inv = [1.0 / r for r in MEASURED_REL]
_m = sum(_inv) / 4
_FIXED_W = [round(v / _m, 3) for v in _inv]          # (l, t, r, b)

RUNS = [
    {"name": "peu_anchor", "rank": 0, "kind": "baseline",
     "label": "stock DFL — beat-target",
     "params": cfg()},

    {"name": "peu_b05", "rank": 1, "kind": "learned",
     "label": "beta=0.5 lambda=1.0, detached variance (default)",
     "params": peu(peu_beta=0.5, peu_lambda=1.0, peu_detach=True)},

    {"name": "peu_fixed_top", "rank": 2, "kind": "CONTROL",
     "label": f"fixed per-edge weights from the MEASURED residuals {_FIXED_W} "
              f"(l,t,r,b) — no uncertainty. Separates 'adaptive' from "
              f"'just down-weight the top edge'",
     "params": cfg(use_ardfl=True,
                   ardfl_w_weight=_FIXED_W[0], ardfl_h_weight=_FIXED_W[1],
                   ardfl_ar_gate=False)},

    {"name": "peu_b10", "rank": 3, "kind": "learned",
     "label": "beta=1.0 lambda=1.0 — full Kendall attenuation",
     "params": peu(peu_beta=1.0, peu_lambda=1.0, peu_detach=True)},

    {"name": "peu_b05_lam2", "rank": 4, "kind": "learned",
     "label": "beta=0.5 lambda=2.0 — stronger penalty on claiming uncertainty",
     "params": peu(peu_beta=0.5, peu_lambda=2.0, peu_detach=True)},

    {"name": "peu_b05_nodet", "rank": 5, "kind": "learned",
     "label": "beta=0.5, variance NOT detached — gradient flows through the weight",
     "params": peu(peu_beta=0.5, peu_lambda=1.0, peu_detach=False)},
]

SEED_RUNS = ["peu_anchor", "peu_b05"]
SEEDS_LIST = [0, 42, 123]


def _loss_fingerprint():
    try:
        import ultralytics.utils.loss as L
        return {"path": L.__file__,
                "md5": hashlib.md5(open(L.__file__, "rb").read()).hexdigest()[:12],
                "has_peu": hasattr(L, "peu_report")}
    except Exception as e:
        return {"error": str(e)}


def on_train_epoch_start(trainer):
    try:
        from ultralytics.utils.loss import set_epoch
        set_epoch(trainer.epoch, getattr(trainer, "epochs", EPOCHS))
    except Exception:
        pass
    try:
        de_parallel(trainer.model).current_epoch = trainer.epoch
    except Exception:
        pass


def on_train_epoch_end(trainer):
    """Per-edge variance telemetry — this IS the hypothesis test.

    If the top edge really carries the highest aleatoric uncertainty it should
    show the largest var and DFL loss and the smallest attenuation weight. If it
    does not, PEU is attenuating something other than what the diagnostic found.
    """
    try:
        from ultralytics.utils.loss import peu_report
        r = peu_report(reset=True)
        if not r:
            return
        order = ("left", "top", "right", "bottom")
        print("  [PEU] var   " + "  ".join(f"{k}={r['var'][k]:.3f}" for k in order))
        print("  [PEU] w     " + "  ".join(f"{k}={r['w'][k]:.3f}" for k in order))
        print("  [PEU] dfl   " + "  ".join(f"{k}={r['loss'][k]:.3f}" for k in order))
        top_is_worst = r["var"]["top"] == max(r["var"].values())
        print(f"  [PEU] top edge has highest variance: {top_is_worst}")
    except Exception:
        pass


def run_one(rc, seed=SEED, with_test=False):
    name = rc["name"] if seed == SEED else f"{rc['name']}_s{seed}"
    print(f"\n{'=' * 76}\n  RUN {name}   [{rc.get('kind')}]  seed={seed}\n  {rc['label']}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=100, close_mosaic=10, seed=seed, deterministic=True,
              exist_ok=False)
    kw.update(copy.deepcopy(rc["params"]))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "peu_params.json"), "w") as f:
            json.dump({"name": name, "kind": rc.get("kind"), "label": rc["label"],
                       "params": rc["params"], "seed": seed, "epochs": EPOCHS,
                       "measured_rel_residual": dict(zip(("left", "top", "right", "bottom"),
                                                         MEASURED_REL)),
                       "fixed_control_weights": _FIXED_W,
                       "loss_file": _loss_fingerprint()}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json: {e}")

    def _m(rd, *ks):
        for k in ks:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "kind": rc.get("kind"), "seed": seed, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    if with_test:
        try:
            tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
            out["test_map50"], out["test_map5095"] = float(tm.box.map50), float(tm.box.map)
        except Exception as e:
            print(f"  [warn] test eval: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--seeds", action="store_true")
    args = ap.parse_args()

    print(f"\n{'=' * 76}")
    print(f"  PEU-DFL ABLATION  @{IMG_SIZE}px  {EPOCHS} epochs")
    print(f"  measured per-edge relative residual (l,t,r,b) = {MEASURED_REL}")
    print(f"  fixed-control weights                         = {_FIXED_W}")
    print(f"  loss file: {_loss_fingerprint()}")
    print(f"{'=' * 76}")

    if args.seeds:
        todo = [(r, s) for r in RUNS if r["name"] in SEED_RUNS for s in SEEDS_LIST]
    else:
        unknown = set(args.runs) - {r["name"] for r in RUNS}
        if unknown:
            sys.exit(f"unknown: {sorted(unknown)}\navailable: {[r['name'] for r in RUNS]}")
        sel = RUNS if not args.runs else [r for r in RUNS if r["name"] in set(args.runs)]
        todo = [(r, SEED) for r in sel]

    for r, s in todo:
        print(f"  {r.get('rank','?'):>2}  {r['name']:<16s} seed={s}  {r['kind']}")
    print(f"{'=' * 76}\n")

    res = [run_one(r, seed=s, with_test=args.with_test) for r, s in todo]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "peu_summary_seeds.json" if args.seeds else "peu_summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    a = next((r["val_map5095"] for r in res if r["name"] == "peu_anchor"), None)
    print(f"\n{'=' * 76}\n  RESULTS (val)\n{'=' * 76}")
    print(f"{'run':<18s}{'kind':<11s}{'mAP50':>8s}{'mAP50-95':>11s}{'delta':>9s}{'h':>6s}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x["val_map5095"] if x["val_map5095"] == x["val_map5095"] else -9)):
        d = f"{(r['val_map5095']-a)*100:+.2f}" if a and r["name"] != "peu_anchor" else "—"
        print(f"{r['name']:<18s}{str(r['kind']):<11s}{r['val_map50']*100:>8.2f}"
              f"{r['val_map5095']*100:>11.2f}{d:>9s}{r['hours']:>5.1f}")
    print("\nREAD peu_fixed_top FIRST: if it matches peu_b05, the contribution is")
    print("'down-weight the top edge', not 'learn per-edge uncertainty'. Both are")
    print("publishable, but they are different papers — do not conflate them.")
    print(f"\nsaved -> {out}")
