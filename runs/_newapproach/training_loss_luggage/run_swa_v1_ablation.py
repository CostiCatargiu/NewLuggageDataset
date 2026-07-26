#!/usr/bin/env python3
"""
SWA v1 Alpha Schedule Ablation — 8 runs exploring alpha_start / alpha_end
combinations on the v1 mechanism. NO BOOST (small_obj_boost=1.0).

This uses your CURRENT loss.py (v1 SWA) — no file swap needed.

BACKGROUND
  v1 tested these alpha schedules (ALWAYS coupled with small_obj_boost):
    0.5 → 0.25   boost=2.0   px=48   → test mAP50-95 = 56.52
    0.6 → 0.6    boost=1.75  px=48   → test mAP50-95 = 56.61  (best SWA)
    0.7 → 0.3    boost=2.0   px=48   → test mAP50-95 = 56.38
    0.9 → 0.4    boost=2.0   px=48   → test mAP50-95 = 56.29
    baseline (all OFF)                → test mAP50-95 = 56.85

  PROBLEM: alpha and boost were always coupled — we can't tell which caused
  the mAP50-95 drop. Maybe the alpha schedule is fine and the 1/area boost
  (400:1 spread) is what kills precision.

  THIS ABLATION: same v1 mechanism, same alpha_start/alpha_end ranges,
  but boost=1.0 (OFF). Pure alpha isolation.

  We fill the GAPS in v1's alpha grid:
    v1 tested:  0.9→0.4 | 0.7→0.3 | 0.6→0.6 | 0.5→0.25
    We test:    0.9→0.3 | 0.9→0.5 | 0.9→0.6 | 0.7→0.4 | 0.7→0.5 | 0.7→0.6 | 0.5→0.3

RUNS (8 total, NO boost):
  swa1_anchor          alpha OFF                   baseline
  swa1_09_03           alpha 0.9→0.3  min/max 0.3/0.9   wide spread
  swa1_09_05           alpha 0.9→0.5  min/max 0.5/0.9   stays high
  swa1_09_06           alpha 0.9→0.6  min/max 0.6/0.9   barely anneals
  swa1_07_04           alpha 0.7→0.4  min/max 0.4/0.7   between v1 points
  swa1_07_05           alpha 0.7→0.5  min/max 0.5/0.7   tight spread
  swa1_07_06           alpha 0.7→0.6  min/max 0.6/0.7   barely anneals
  swa1_05_03           alpha 0.5→0.3  min/max 0.3/0.5   higher floor than v1

REQUIRES:
  loss_function/loss2.py copied to ultralytics/utils/loss.py
  This is the latest v1 SWA implementation (R7-R8, backward compatible with R2-R5).

  BEFORE RUNNING:
    cp loss_function/loss2.py /path/to/ultralytics/utils/loss.py

Usage:
  python run_swa_v1_ablation.py                        # all 8 runs
  python run_swa_v1_ablation.py swa1_09_05             # single run
  python run_swa_v1_ablation.py --with-test            # include test eval
"""

import sys
import time
import gc
import json
import os
import torch
from ultralytics import YOLO

# =============================================================================
# STEP 0 — REGISTER CUSTOM KEYS (v1 parameter names)
# =============================================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_LOSS_DEFAULTS = {
    # v1 SWA
    "alpha_start": 0.0,
    "alpha_end": 0.0,
    "alpha_min": 0.0,
    "alpha_max": 0.0,
    "small_obj_px": 48,
    "small_obj_boost": 1.0,
    "area_weight_min": 0.4,
    "area_weight_mode": "inv",
    # area weighting
    "weight_renorm": 1,
    "area_mode": "fixed",
    "area_ref_px": 64.0,
    "area_gamma": 0.5,
    "area_w_cap": 3.0,
    # per-class boost
    "small_obj_boost_backpack": -1.0,
    "small_obj_boost_bag": -1.0,
    "small_obj_boost_trolley": -1.0,
    # DFL boost
    "dfl_small_boost": 1.0,
    # NWD
    "nwd_ratio": 0.0,
    "nwd_c": 64.0,
    # center/crowd
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    "center_loss_mode": "small",
    "center_crowd_iou": 0.1,
    # clipping
    "iou_clip_start": 999.0,
    "iou_clip_end": 999.0,
    "dfl_clip_start": 999.0,
    "dfl_clip_end": 999.0,
    "use_percentile_clip": False,
    "clip_percentile": 0.95,
    "use_loss_clip": False,
    # TAL
    "tal_topk": 10,
    "tal_alpha": 0.5,
    "tal_beta": 6.0,
    # SATAL
    "use_satal": False,
    "satal_alpha_small": 1.2,
    "satal_beta_small": 4.5,
    "satal_alpha_large": 1.0,
    "satal_beta_large": 6.0,
    "satal_small_area": 0.0025,
    "satal_large_area": 0.0225,
    "satal_topk_factor": 1.3,
    # shape TAL
    "use_shape_tal": False,
    "shape_gamma": 0.1,
    "shape_min": 0.1,
    # IoU type
    "iou_type": "ciou",
}

for _k, _v in _CUSTOM_LOSS_DEFAULTS.items():
    DEFAULT_CFG_DICT.setdefault(_k, _v)
    if not hasattr(DEFAULT_CFG, _k):
        setattr(DEFAULT_CFG, _k, _v)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_swa_v1_ablation"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# ALL-OFF BASELINE (v1 parameter names)
# =============================================================================
_ALL_OFF = dict(
    alpha_start=0.0,
    alpha_end=0.0,
    alpha_min=0.0,
    alpha_max=0.0,
    small_obj_px=48,
    small_obj_boost=1.0,              # NO BOOST — entire ablation
    area_weight_mode="inv",
    nwd_ratio=0.0,
    iou_type="ciou",
    use_satal=False,
    use_loss_clip=False,
    use_shape_tal=False,
    center_loss_weight_init=0.0,
    dfl_small_boost=1.0,
)

# =============================================================================
# 8 RUNS — alpha_start→alpha_end, NO boost
# =============================================================================
RUNS = [
    # ---- Baseline ----
    {
        "name": "swa1_anchor",
        "label": "[0/7] Anchor — alpha OFF (stock CIoU)",
        "params": {**_ALL_OFF},
        "seed": SEED,
    },

    # ---- Start=0.9, vary end (0.3, 0.5, 0.6) ----
    # v1 only tested 0.9→0.4 (with boost=2.0)
    {
        "name": "swa1_09_03",
        "label": "[1/7] alpha 0.9→0.3 — wide spread, aggressive start",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.9, "alpha_end": 0.3,
            "alpha_min": 0.3, "alpha_max": 0.9,
        },
        "seed": SEED,
    },
    {
        "name": "swa1_09_05",
        "label": "[2/7] alpha 0.9→0.5 — stays high, tight spread",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.9, "alpha_end": 0.5,
            "alpha_min": 0.5, "alpha_max": 0.9,
        },
        "seed": SEED,
    },
    {
        "name": "swa1_09_06",
        "label": "[3/7] alpha 0.9→0.6 — barely anneals from 0.9",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.9, "alpha_end": 0.6,
            "alpha_min": 0.6, "alpha_max": 0.9,
        },
        "seed": SEED,
    },

    # ---- Start=0.7, vary end (0.4, 0.5, 0.6) ----
    # v1 only tested 0.7→0.3 (with boost=2.0)
    {
        "name": "swa1_07_04",
        "label": "[4/7] alpha 0.7→0.4 — moderate start, moderate end",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.7, "alpha_end": 0.4,
            "alpha_min": 0.4, "alpha_max": 0.7,
        },
        "seed": SEED,
    },
    {
        "name": "swa1_07_05",
        "label": "[5/7] alpha 0.7→0.5 — moderate start, tight spread",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.7, "alpha_end": 0.5,
            "alpha_min": 0.5, "alpha_max": 0.7,
        },
        "seed": SEED,
    },
    {
        "name": "swa1_07_06",
        "label": "[6/7] alpha 0.7→0.6 — barely anneals from 0.7",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.7, "alpha_end": 0.6,
            "alpha_min": 0.6, "alpha_max": 0.7,
        },
        "seed": SEED,
    },

    # ---- Start=0.5, end=0.3 ----
    # v1 tested 0.5→0.25. Higher floor here.
    {
        "name": "swa1_05_03",
        "label": "[7/7] alpha 0.5→0.3 — gentle start, higher floor than v1",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.5, "alpha_end": 0.3,
            "alpha_min": 0.3, "alpha_max": 0.5,
        },
        "seed": SEED,
    },
]


# =============================================================================
# EPOCH CALLBACK — CRITICAL for v1 alpha annealing
# =============================================================================
# Without this, the alpha stays frozen at alpha_start for the entire training.
# This callback syncs the current epoch into the loss object so the
# alpha_start → alpha_end schedule actually anneals.

def on_train_epoch_start(trainer):
    """Sync epoch into the custom loss (drives alpha / clip schedules)."""
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
    except Exception:
        pass


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
    print(f"{'='*72}")

    a_start = params.get("alpha_start", 0.0)
    a_end = params.get("alpha_end", 0.0)
    boost = params.get("small_obj_boost", 1.0)

    if a_start == 0.0:
        sched = "OFF"
    elif a_start == a_end:
        sched = f"{a_start} constant"
    else:
        sched = f"{a_start} → {a_end}"

    print(f"  Alpha: {sched}")
    print(f"  Boost: {boost} (OFF)")
    print(f"  Implementation: v1 (loss_satal3.py)\n")

    model = YOLO(MODEL_WEIGHTS)

    # CRITICAL: register epoch callback for alpha annealing
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("[setup] Epoch callback registered (alpha annealing active)")

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
        **params,
    )
    elapsed = time.time() - t0

    metrics = {
        "name": name,
        "seed": seed,
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "alpha_schedule": {
            "alpha_start": a_start,
            "alpha_end": a_end,
            "alpha_min": params.get("alpha_min", 0.0),
            "alpha_max": params.get("alpha_max", 0.0),
            "small_obj_boost": boost,
            "implementation": "v1",
        },
        "params": {k: v for k, v in params.items()},
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

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, f"{name}_seed{seed}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[{name}] saved -> {out_path}")

    if with_test:
        print(f"\n[{name}] Evaluating on test set...")
        best_path = os.path.join(PROJECT_DIR, name, "weights", "best.pt")
        if os.path.exists(best_path):
            test_model = YOLO(best_path)
            test_results = test_model.val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH
            )
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

    if args:
        runs = [r for r in RUNS if r["name"] in args]
        if not runs:
            print(f"No matching runs. Available: {[r['name'] for r in RUNS]}")
            sys.exit(1)
    else:
        runs = RUNS

    # Print overview
    print(f"\n{'='*72}")
    print(f"  SWA v1 ALPHA SCHEDULE ABLATION ({len(runs)} runs)")
    print(f"  Implementation: v1 (current loss.py, 1/area weighting)")
    print(f"  Boost: OFF (1.0) — isolating alpha schedule effect")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}, seed={SEED}")
    print(f"{'='*72}")
    print(f"\n  v1 previously tested (WITH boost):")
    print(f"    0.9→0.4 boost=2.0  | 0.7→0.3 boost=2.0  | 0.6 const boost=1.75  | 0.5→0.25 boost=2.0")
    print(f"\n  Now testing (WITHOUT boost) — fill the gaps:\n")
    print(f"  {'Name':<20s} {'Start':>6s} {'End':>6s} {'Min':>6s} {'Max':>6s}")
    print(f"  {'-'*46}")
    for r in runs:
        p = r["params"]
        print(f"  {r['name']:<20s} {p.get('alpha_start', 0):>6.1f} "
              f"{p.get('alpha_end', 0):>6.1f} "
              f"{p.get('alpha_min', 0):>6.1f} "
              f"{p.get('alpha_max', 0):>6.1f}")
    print()

    all_results = []
    for i, run in enumerate(runs):
        print(f"\n>>> Run {i+1}/{len(runs)}: {run['name']}")
        result = train_run(run, with_test=with_test)
        all_results.append(result)

    # Save summary
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print results
    print(f"\n\n{'='*72}")
    print("  RESULTS — v1 Alpha Schedule (boost=1.0, no size boost)")
    print(f"{'='*72}")
    print(f"  v1 reference (WITH boost):")
    print(f"    0.9→0.4 = 56.29 | 0.7→0.3 = 56.38 | 0.6 const = 56.61 | 0.5→0.25 = 56.52")
    print(f"    baseline (OFF)  = 56.85")
    print()
    print(f"{'Name':<20s} {'Schedule':>10s} {'mAP50':>8s} {'mAP50-95':>10s} "
          f"{'Prec':>7s} {'Recall':>7s}")
    print("-" * 68)
    for r in all_results:
        v = r.get("val", {})
        s = r.get("alpha_schedule", {})
        a_s = s.get("alpha_start", 0)
        a_e = s.get("alpha_end", 0)
        sched = "OFF" if a_s == 0 else (f"{a_s} const" if a_s == a_e else f"{a_s}→{a_e}")
        print(f"{r['name']:<20s} {sched:>10s} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {v.get('precision', 0):>7.4f} "
              f"{v.get('recall', 0):>7.4f}")

    # Deltas
    anchor = next((r for r in all_results if r["name"] == "swa1_anchor"), None)
    if anchor and "val" in anchor:
        b50 = anchor["val"]["mAP50"]
        b95 = anchor["val"]["mAP50_95"]
        print(f"\n  Deltas vs anchor (mAP50={b50:.4f}, mAP50-95={b95:.4f}):")
        best_name, best_d95 = "swa1_anchor", 0.0
        for r in all_results:
            if r["name"] == "swa1_anchor":
                continue
            v = r.get("val", {})
            if not v:
                continue
            d50 = v.get("mAP50", 0) - b50
            d95 = v.get("mAP50_95", 0) - b95
            marker = " <<<" if d95 > 0.001 else ""
            print(f"    {r['name']:<20s} Δ50={d50:+.4f}  Δ50-95={d95:+.4f}{marker}")
            if d95 > best_d95:
                best_d95 = d95
                best_name = r["name"]

        if best_d95 > 0:
            best_r = next(r for r in all_results if r["name"] == best_name)
            best_s = best_r.get("alpha_schedule", {})
            print(f"\n  >>> BEST: {best_name} (Δ50-95={best_d95:+.4f})")
            print(f"  >>> alpha {best_s['alpha_start']} → {best_s['alpha_end']}")
            print(f"\n  NEXT: Add boost on top of this schedule, or test same schedule on v2.")
        else:
            print(f"\n  >>> No alpha schedule beat baseline on mAP50-95.")
            print(f"  >>> Confirms: alpha flattening itself hurts, not just the boost.")

    print(f"\nSummary saved: {summary_path}")
