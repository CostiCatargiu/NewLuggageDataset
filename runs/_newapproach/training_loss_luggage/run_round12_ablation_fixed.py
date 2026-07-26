#!/usr/bin/env python3
"""
Round 12 — STRUCTURAL LOSS MECHANISMS (change HOW the loss is computed, not weights).

WHY THIS ROUND
  R1-R11 (~80 runs) reweighted the same CIoU+DFL signal and exhausted that space.
  The best gain was +0.64% mAP50-95 from EIoU+class_balance. This round changes
  the STRUCTURE of the loss — the geometry of IoU, the quantization of DFL, the
  per-class loss choice, the task weighting, and the assignment metric.

MECHANISMS (all in loss_v3_luggage.py, all OFF by default):
  [M1] GB-EIoU     — gradient-balanced width/height EIoU. Gives width 2.5x more
                      shape-penalty gradient than height, matching the 33x72px
                      mean box shape where width error hurts IoU 2.5x more.
  [M2] AA-DFL      — aspect-aware DFL with sqrt-compressed bins for narrow edges.
                      Width edges use 4 out of 16 DFL bins; compression gives 2x
                      effective resolution where it matters.
  [M3] CC-Box      — class-conditional box loss. Bags (highest shape variance) get
                      DIoU (center-only, no shape penalty); backpack/trolley get
                      GB-EIoU (consistent tall shape benefits from shape penalty).
  [M4] LTW         — learned task weighting (Kendall 2018). Replaces fixed
                      box=7.5/cls=0.5/dfl=1.5 with learnable weights that adapt.
  [M5] SA-TAL      — shape-aware TAL: assignment uses GB-EIoU overlap instead of
                      CIoU. Changes WHICH anchors get supervised, not the weights.
  [M6] Consistency — auxiliary IoU-DFL consistency loss.

RUNS (each isolates one mechanism, then combines winners):
  r12_anchor        all OFF (stock loss baseline)
  r12_gb_eiou       [M1] only — GB-EIoU as base metric
  r12_eiou_control  CONTROL   — standard EIoU, isolates GB's gradient balancing
  r12_aa_dfl        [M2] only — aspect-aware DFL
  r12_cc_box        [M3] only — class-conditional (bag=DIoU, rest=GB-EIoU)
  r12_ltw           [M4] only — learned task weighting
  r12_sa_tal        [M5] only — shape-aware TAL assignment
  r12_consistency   [M6] only — IoU-DFL consistency loss
  r12_gb_eiou_cls   [M1] + class_weights (combine with R11 winner)
  r12_full_struct   [M1]+[M2]+[M3]+[M5] — all structural, no learned weights
  r12_full_all      [M1]+[M2]+[M3]+[M4]+[M5]+[M6] — everything on

REQUIRES:
  1. Copy loss_v3_luggage.py -> ultralytics/utils/loss.py
  2. Whitelist the new config keys (done automatically below)
  3. Call attach_epoch_tracking(model) for LTW warmup (done automatically)

DECISION RULE: candidate iff val mAP50-95 > anchor + 0.3%. Run candidates with
  3 seeds; test once on full dataset.

Usage:
  python run_round12_ablation.py                      # all runs
  python run_round12_ablation.py r12_gb_eiou          # only named run(s)
  python run_round12_ablation.py --with-test          # also eval test set
"""

import sys
import time
import gc
import json
import os
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# STEP 0 — REGISTER CUSTOM KEYS
# =============================================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_DEFAULTS = {
    # Previous rounds (still available)
    "class_weights": None,
    "normalize_class_weights": True,
    "use_vfl": False,
    "vfl_alpha": 0.75,
    "vfl_gamma": 2.0,
    "small_obj_cls_boost": 1.0,
    "iou_ratio": 1.0,
    "nwd_c": 3.0,
    "small_obj_boost": 1.0,
    "small_obj_area_thresh": 36.0,
    "use_inner_iou": False,
    "inner_iou_ratio_small": 0.7,
    "inner_iou_ratio_large": 1.0,
    "use_ar_penalty": False,
    "ar_penalty_lambda": 0.05,
    "ar_penalty_tall_extra": 0.5,
    "ar_penalty_max": 1.0,
    # Round 12 — structural mechanisms
    "box_metric": "ciou",
    "aa_dfl": False,
    "aa_dfl_gamma": 0.5,
    "cc_box": False,
    "cc_box_bag_metric": "diou",
    "cc_box_bag_class": 1,
    "learned_task_weights": False,
    "ltw_warmup": 10,
    "sa_tal": False,
    "iou_dfl_consistency": 0.0,
}

for _k, _v in _CUSTOM_DEFAULTS.items():
    DEFAULT_CFG_DICT.setdefault(_k, _v)
    if not hasattr(DEFAULT_CFG, _k):
        setattr(DEFAULT_CFG, _k, _v)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_newluggage_r12"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# Class weights: sqrt inverse-frequency, mean-normalized (R11 winner)
CLASS_WEIGHTS = [1.0599, 1.1663, 0.7738]

# =============================================================================
# ALL-OFF BASELINE
# =============================================================================
_ALL_OFF = dict(
    box_metric="ciou",
    aa_dfl=False, aa_dfl_gamma=0.5,
    cc_box=False, cc_box_bag_metric="diou", cc_box_bag_class=1,
    learned_task_weights=False, ltw_warmup=10,
    sa_tal=False,
    iou_dfl_consistency=0.0,
    class_weights=None,
    use_vfl=False,
    small_obj_cls_boost=1.0,
    iou_ratio=1.0,
    small_obj_boost=1.0,
    use_inner_iou=False,
    use_ar_penalty=False,
)

# =============================================================================
# ABLATION RUNS
# =============================================================================
RUNS = [
    # --- Anchor baseline (stock loss) ---
    {
        "name": "r12_anchor",
        "label": "[0/10] Stock baseline — all OFF",
        "params": {**_ALL_OFF},
        "seed": SEED,
    },

    # --- Single-mechanism isolation ---
    {
        "name": "r12_gb_eiou",
        "label": "[1/10] [M1] GB-EIoU only — width-balanced shape penalty",
        "params": {**_ALL_OFF, "box_metric": "gb_eiou"},
        "seed": SEED,
    },
    {
        "name": "r12_eiou_control",
        "label": "[1b/11] CONTROL: standard EIoU (50/50 w/h) — isolates GB's gradient balancing",
        "params": {**_ALL_OFF, "box_metric": "eiou"},
        "seed": SEED,
    },
    {
        "name": "r12_aa_dfl",
        "label": "[2/10] [M2] Aspect-Aware DFL only — compressed width bins",
        "params": {**_ALL_OFF, "aa_dfl": True, "aa_dfl_gamma": 0.5},
        "seed": SEED,
    },
    {
        "name": "r12_cc_box",
        "label": "[3/10] [M3] Class-Conditional Box — bag=DIoU, rest=GB-EIoU",
        "params": {
            **_ALL_OFF,
            "box_metric": "gb_eiou",
            "cc_box": True, "cc_box_bag_metric": "diou", "cc_box_bag_class": 1,
        },
        "seed": SEED,
    },
    {
        "name": "r12_ltw",
        "label": "[4/10] [M4] Learned Task Weights — Kendall uncertainty",
        "params": {**_ALL_OFF, "learned_task_weights": True, "ltw_warmup": 10},
        "seed": SEED,
    },
    {
        "name": "r12_sa_tal",
        "label": "[5/10] [M5] Shape-Aware TAL — GB-EIoU assignment",
        "params": {**_ALL_OFF, "sa_tal": True},
        "seed": SEED,
    },
    {
        "name": "r12_consistency",
        "label": "[6/10] [M6] IoU-DFL Consistency — auxiliary coherence loss",
        "params": {**_ALL_OFF, "iou_dfl_consistency": 0.5},
        "seed": SEED,
    },

    # --- Combinations ---
    {
        "name": "r12_gb_eiou_cls",
        "label": "[7/10] [M1] + class_weights — combine with R11 winner",
        "params": {
            **_ALL_OFF,
            "box_metric": "gb_eiou",
            "class_weights": CLASS_WEIGHTS,
        },
        "seed": SEED,
    },
    {
        "name": "r12_full_struct",
        "label": "[8/10] [M1]+[M2]+[M3]+[M5] — all structural, fixed task weights",
        "params": {
            **_ALL_OFF,
            "box_metric": "gb_eiou",
            "aa_dfl": True, "aa_dfl_gamma": 0.5,
            "cc_box": True, "cc_box_bag_metric": "diou", "cc_box_bag_class": 1,
            "sa_tal": True,
            "class_weights": CLASS_WEIGHTS,
        },
        "seed": SEED,
    },
    {
        "name": "r12_full_all",
        "label": "[9/10] ALL mechanisms ON — M1+M2+M3+M4+M5+M6",
        "params": {
            **_ALL_OFF,
            "box_metric": "gb_eiou",
            "aa_dfl": True, "aa_dfl_gamma": 0.5,
            "cc_box": True, "cc_box_bag_metric": "diou", "cc_box_bag_class": 1,
            "learned_task_weights": True, "ltw_warmup": 10,
            "sa_tal": True,
            "iou_dfl_consistency": 0.5,
            "class_weights": CLASS_WEIGHTS,
        },
        "seed": SEED,
    },
]

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================
def preflight():
    """Verify loss_v3_luggage.py is the active loss module."""
    from ultralytics.utils import loss as loss_mod
    src = loss_mod.__file__
    # Check for our marker
    import inspect
    source = inspect.getsource(loss_mod.v8DetectionLoss)
    has_v3 = ("loss_v3_luggage" in source or "GB-EIoU" in source
              or "ShapeAwareTAL" in source)
    # Fixed-file markers: make_dfl_proj (M2 redesign) + LTW optimizer registry.
    has_fixes = hasattr(loss_mod, "make_dfl_proj") and hasattr(loss_mod, "_LTW_REGISTRY")
    if has_v3 and has_fixes:
        print(f"[preflight] OK  active loss module: {src} (fixed build)")
        return True
    if has_v3 and not has_fixes:
        print("[preflight] FAIL  this is the ORIGINAL loss_v3_luggage.py, not the fixed build.")
        print("             Known-broken in that version:")
        print("               M2 aa_dfl  — bin compression inverted + decode mismatch")
        print("               M4 ltw     — params never optimized; box gain 844x too small")
        print("               M5 sa_tal  — GB-EIoU arg order inverted (AR from prediction)")
        print("               M6 consist — IndexError, and zero gradient")
        print("             -> copy loss_v3_luggage_fixed.py to ultralytics/utils/loss.py")
        return False
    else:
        print(f"[preflight] WARN  active loss module does NOT appear to be loss_v3_luggage.py")
        print(f"             Source: {src}")
        print(f"             To fix: copy loss_v3_luggage.py -> ultralytics/utils/loss.py")
        return False


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_run(run_cfg, with_test=False):
    name = run_cfg["name"]
    label = run_cfg["label"]
    params = run_cfg["params"]
    seed = run_cfg["seed"]

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  name={name}  seed={seed}")
    print(f"{'='*72}\n")

    model = YOLO(MODEL_WEIGHTS)

    # Attach epoch tracking for Kendall warmup
    try:
        from ultralytics.utils.loss import attach_epoch_tracking
        attach_epoch_tracking(model)
        print("[setup] Epoch tracking attached")
    except ImportError:
        if params.get("learned_task_weights"):
            raise SystemExit(
                "[setup] FATAL: learned_task_weights=True but attach_epoch_tracking "
                "is missing. Without it the LTW params never reach the optimizer "
                "and 'learned' weights silently stay fixed. Use loss_v3_luggage_fixed.py."
            )
        print("[setup] WARN: attach_epoch_tracking not found (older loss file?)")

    t0 = time.time()
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=seed,
        name=name,
        project=PROJECT_DIR,
        exist_ok=True,
        patience=30,
        close_mosaic=10,
        deterministic=True,
        amp=True,
        val=True,
        # Loss params
        **params,
    )
    elapsed = time.time() - t0

    # Collect val metrics
    metrics = {
        "name": name,
        "seed": seed,
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "params": {k: (v if not isinstance(v, list) else v) for k, v in params.items()},
    }
    if results is not None:
        try:
            r = results.results_dict
            metrics["val"] = {
                "mAP50": round(r.get("metrics/mAP50(B)", 0), 5),
                "mAP50_95": round(r.get("metrics/mAP50-95(B)", 0), 5),
                "precision": round(r.get("metrics/precision(B)", 0), 5),
                "recall": round(r.get("metrics/recall(B)", 0), 5),
            }
        except Exception as e:
            metrics["val_error"] = str(e)

    # Save metrics
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, f"{name}_seed{seed}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[{name}] saved -> {out_path}")

    # Optional test evaluation
    if with_test:
        print(f"\n[{name}] Evaluating on test set...")
        best_path = os.path.join(PROJECT_DIR, name, "weights", "best.pt")
        if os.path.exists(best_path):
            test_model = YOLO(best_path)
            test_results = test_model.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH)
            if test_results is not None:
                try:
                    tr = test_results.results_dict
                    metrics["test"] = {
                        "mAP50": round(tr.get("metrics/mAP50(B)", 0), 5),
                        "mAP50_95": round(tr.get("metrics/mAP50-95(B)", 0), 5),
                    }
                    with open(out_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    print(f"[{name}] test eval error: {e}")
            del test_model

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = set(sys.argv[1:])
    with_test = "--with-test" in args
    args.discard("--with-test")

    # Filter runs if specific names given
    if args:
        runs = [r for r in RUNS if r["name"] in args]
        if not runs:
            print(f"No matching runs found. Available: {[r['name'] for r in RUNS]}")
            sys.exit(1)
    else:
        runs = RUNS

    # Preflight
    if not preflight():
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != "y":
            sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  ROUND 12 (FIXED) — Structural Loss Mechanisms ({len(runs)} runs)")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}, seed={SEED}")
    print(f"{'='*72}\n")

    all_results = []
    for i, run in enumerate(runs):
        print(f"\n>>> Run {i+1}/{len(runs)}: {run['name']}")
        result = train_run(run, with_test=with_test)
        all_results.append(result)

    # Summary
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*72}")
    print("  ROUND 12 SUMMARY")
    print(f"{'='*72}")
    print(f"{'Name':<25s} {'mAP50':>8s} {'mAP50-95':>10s} {'Time':>8s}")
    print("-" * 55)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<25s} {v.get('mAP50', 0):>8.4f} {v.get('mAP50_95', 0):>10.4f} {r.get('elapsed_s', 0):>7.0f}s")
    print(f"\nSummary saved: {summary_path}")
