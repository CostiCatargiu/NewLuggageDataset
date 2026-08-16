#!/usr/bin/env python3
"""
YOLO26 LOSS ROUND 2 — two configs, both on stock yolo26s, both from the phase-A
decomposition. No seeds, no architecture.

=============================================================================
WHAT PHASE A ACTUALLY SHOWED
=============================================================================
Deltas vs baseline 55.24, stock yolo26s, b82, n=1 each:

    run              overall   small  medium   large    prec  recall
    y26_scb_b3         +0.42   +0.24   +0.87   -1.43   +2.02   +0.15
    y26_snl1_p25       +0.25   +0.05   +0.97   -4.37   +0.56   +0.18
    y26_snl1_p50       +0.24   +0.42   +0.60   -3.38   +1.15   +0.45
    y26_dfl3           +0.13   +0.18   +0.25   -2.01   -0.62   +0.95
    y26_beta4          +0.13   -0.22   +0.78   -1.04   +0.45   +1.16

The control run (y26_beta4, global beta 6.0 -> 4.0) separates two effects that
were confounded inside SCB:

                  scb_b3   beta4   difference = size-conditioning's own effect
    small          +0.24   -0.22       +0.47
    medium         +0.87   +0.78       +0.09
    precision      +2.02   +0.45       +1.57
    recall         +0.15   +1.16       -1.01

LOWERING BETA GLOBALLY buys medium (+0.78) and recall (+1.16) and LOSES small.
SIZE-CONDITIONING buys small (+0.47) and precision (+1.57) and gives the recall
back. They act on different buckets, and the current SCB leaves the global
reduction unused: it keeps beta = 6.0 for everything above 64 px.

=============================================================================
READ THIS BEFORE TRUSTING THE ABOVE
=============================================================================
All five phase-A runs sit within 0.29 pp of each other, and the published
14-run campaign at this batch had sd 0.28. So the whole decomposition rests on
1.0-1.5 sd differences. The per-bucket pattern is coherent in a way pure scatter
usually is not, but rounds 4-6 of this project produced an entire mechanistic
story out of exactly this kind of margin and it evaporated under a proper
control. Treat these two runs as leads, not as a sweep around a known optimum.

DECISION THRESHOLDS, given no seed repeats exist:

    >= +0.80 vs baseline    real, outside anything noise explains
    +0.60 .. +0.80          promising, needs a seed before it is a number
    <  +0.60                indistinguishable from y26_scb_b3's +0.42

=============================================================================
THE TWO
=============================================================================
  y26_scb_b4s2   tal_beta 4.0 + tal_beta_small 2.0
                 global reduction AND size-conditioning, the two effects the
                 control separated. Best-motivated config available.

  y26_alpha075   tal_alpha 0.5 -> 0.75
                 the other exponent, never varied in 34 recorded YOLO26 runs.
                 Not a refinement of a 1-sd result - an unexplored axis.

Deliberately NOT included: SNL1 x SCB (SNL1's ceiling is the 15.8% loss share,
and the dfl3 probe moved only +0.13, so that term is a weak lever), and
tal_beta_ref_px (a third point on an axis whose first two points differ by
1 sd).

Stock yolo26s.pt, b82/640/seed 0 — matches phase A exactly.
REQUIRES the patched loss.py / tal.py / default.yaml.

Usage:
    python run_yolo26_loss_round2_v6i.py
    python run_yolo26_loss_round2_v6i.py y26_alpha075
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
PROJECT_DIR = "runs_yolo26_loss_round2_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches phase A
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24
PHASE_A = {"y26_scb_b3": 55.66, "y26_snl1_p25": 55.49, "y26_snl1_p50": 55.48,
           "y26_dfl3": 55.37, "y26_beta4": 55.37}
NOISE = 0.28                          # sd of the published 14-run campaign, b82
REAL = BASELINE + 0.80                # >= this is outside noise

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


RUNS = [
    {"name": "y26_scb_b4s2",
     "expect": {"scb": True, "alpha": 0.5, "beta": 4.0},
     "params": cfg(tal_beta=4.0, tal_beta_small=2.0, tal_beta_ref_px=64.0),
     "label": "beta 2.0 -> 4.0 over sqrt(area) 0 -> 64 px (global reduction + conditioning)",
     "why": "y26_beta4 showed the global reduction buys medium (+0.78) and recall "
            "(+1.16) while losing small (-0.22). y26_scb_b3 showed the size-"
            "conditioning buys small (+0.47) and precision (+1.57) on top of a "
            "beta that stayed at 6.0 everywhere above 64 px. This config takes "
            "both: the ceiling drops to 4.0 so medium and recall keep the global "
            "gain, and the floor drops to 2.0 so small keeps the conditioning "
            "gain. If they compose it clears +0.80. If it lands at ~+0.42 the two "
            "effects were the same effect measured twice and the decomposition "
            "was noise - which is the honest failure mode to watch for."},

    {"name": "y26_alpha075",
     "expect": {"scb": False, "alpha": 0.75, "beta": 6.0},
     "params": cfg(tal_alpha=0.75),
     "label": "tal_alpha 0.5 -> 0.75, beta untouched",
     "why": "align_metric = score^alpha * IoU^beta, and alpha has never been "
            "varied in 34 recorded YOLO26 runs - it is the one axis of this "
            "metric nobody has touched. With topk2=1 the metric picks THE single "
            "anchor per GT in the branch that produces every prediction, so its "
            "balance matters more here than on a topk=10 assigner. Raising alpha "
            "is the complement of lowering beta: instead of trusting IoU less, "
            "trust the classification score more. The beta results already "
            "suggest the balance is mistuned for this dataset. Lower prior than "
            "run 1, but it fails for a different reason if it fails, which is "
            "worth more than a third point on an axis resolved to 1 sd."},
]


def preflight(todo):
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
        "TaskAlignedAssigner.scb_enabled": hasattr(TAL.TaskAlignedAssigner, "scb_enabled"),
        "loss.py reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<38}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] SCB patch not installed.")
        print("  python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False

    for r in todo:
        p = r["params"]
        if r["expect"]["scb"]:
            a = TAL.TaskAlignedAssigner(topk=10, alpha=p["tal_alpha"], beta=p["tal_beta"])
            a.beta_small = float(p["tal_beta_small"])
            a.beta_ref_px = float(p["tal_beta_ref_px"])
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: scb_enabled() False "
                      f"(beta_small={a.beta_small} beta={a.beta})")
                return False
            print(f"  {r['name']:<15} beta {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px  OK")
        else:
            print(f"  {r['name']:<15} alpha={p['tal_alpha']} beta={p['tal_beta']} (no SCB)  OK")

    print()
    print(f"  MODEL   {MODEL_WEIGHTS} (STOCK)   batch {BATCH}   imgsz {IMG_SIZE}   seed {SEED}")
    print(f"  baseline {BASELINE:.2f}   best phase A {max(PHASE_A.values()):.2f} (y26_scb_b3)")
    print(f"  campaign sd at this batch {NOISE:.2f} -> only >= {REAL:.2f} is outside noise")
    print(f"  Read SMALL, MEDIUM, PRECISION and RECALL separately - the phase-A")
    print(f"  decomposition lives in those columns, not in overall mAP.")

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
    """Assert at epoch 1 that the exponents and SCB state are what the name claims."""
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        exp = rc["expect"]
        for tag in ("one2one", "one2many"):
            br = getattr(crit, tag, crit)
            a = getattr(br, "assigner", None)
            if a is None:
                raise RuntimeError(f"{rc['name']}: no assigner on {tag}")
            live = hasattr(a, "scb_enabled") and a.scb_enabled()
            if live != exp["scb"]:
                raise RuntimeError(
                    f"{rc['name']}: SCB is {'ON' if live else 'OFF'} on {tag}, expected "
                    f"{'ON' if exp['scb'] else 'OFF'} (beta_small={getattr(a, 'beta_small', None)})")
            for attr, want in (("alpha", exp["alpha"]), ("beta", exp["beta"])):
                got = float(getattr(a, attr))
                if abs(got - want) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag}.{attr} is {got}, expected {want}")
            msg = f"  [guard] {tag}: alpha={a.alpha} beta={a.beta}"
            if live:
                msg += f"  SCB {a.beta_small} -> {a.beta} @ {a.beta_ref_px}px"
            print(msg)
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
        raise RuntimeError(f"{name}: the guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "params": rc["params"], "expect": rc["expect"], "seed": SEED,
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "loss_r2_params.json"), "w") as f:
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
    print(f"  YOLO26 LOSS ROUND 2 — stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 84)
    print(f"{'run':<16}{'mAP50':>9}{'mAP50-95':>10}{'vs base':>9}   verdict")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        d = v - BASELINE
        vd = ("REAL (outside noise)" if v >= REAL else
              "promising, needs a seed" if d >= 0.60 else
              "inside noise of the existing +0.42")
        print(f"{r['name']:<16}{r['test_map50'] * 100:>9.2f}{v:>10.2f}{d:>+9.2f}   {vd}")
    print("-" * 84)
    for n, v in sorted(PHASE_A.items(), key=lambda kv: -kv[1]):
        print(f"  phase A  {n:<16}{v:>7.2f}{v - BASELINE:>+9.2f}")
    print(f"  baseline {'yolo26_custom-9':<16}{BASELINE:>7.2f}")

    by = {r["name"]: r["test_map5095"] * 100 for r in ok}
    if "y26_scb_b4s2" in by:
        v = by["y26_scb_b4s2"]
        print(f"\n  DID THE TWO EFFECTS COMPOSE?")
        print(f"    y26_beta4    (global only)        +0.13")
        print(f"    y26_scb_b3   (conditioning only)  +0.42")
        print(f"    y26_scb_b4s2 (both)               {v - BASELINE:+.2f}")
        if v - BASELINE >= 0.80:
            print("    They composed. Check per-size: medium/recall from the global")
            print("    reduction, small/precision from the conditioning.")
        elif abs(v - PHASE_A['y26_scb_b3']) < 0.3:
            print("    Landed on top of scb_b3 -> the 'two effects' were one effect")
            print("    measured twice, and the phase-A decomposition was 1-sd scatter.")
        else:
            print("    Did not compose. Same pattern as every other combination in this")
            print("    project - arch+loss on both detectors, SWA+LB-TAL on YOLOv12.")
    if "y26_alpha075" in by:
        v = by["y26_alpha075"]
        print(f"\n  ALPHA AXIS   alpha 0.5 -> 0.75 moved {v - BASELINE:+.2f} pp")
        print(f"    (beta 6.0 -> 4.0 moved +0.13 for comparison)")

    print(f"\n  Per-size: CocoEvalAllFolders_luggage.py on each best.pt.")
    print(f"  The phase-A decomposition lives in small / medium / precision / recall.")
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
    print(f"  YOLO26 LOSS ROUND 2 — {len(todo)} runs, ~{1.8 * len(todo):.1f} GPU-h")
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
                        "error": str(e), "expect": r["expect"], "params": r["params"],
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
