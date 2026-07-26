#!/usr/bin/env python3
"""
SWA v1 Boost + px Ablation — 8 runs sweeping small_obj_boost AND small_obj_px
on top of the winning alpha schedule 0.9→0.3.

PREVIOUS RESULTS:
  alpha 0.9→0.3, boost=1.0  → mAP50-95 = 0.5748  (+1.13% vs v6_default2)
  v6_default2 baseline      → mAP50-95 = 0.5684

DESIGN: 4 boost values x 2 px thresholds = 8 runs
  boost: 1.2 (gentle), 1.5 (moderate), 2.0 (strong), 3.0 (aggressive)
  px:    36 (tight — only very small objects boosted)
         48 (wide — most small objects boosted, previously used in R2-R8)

  Dataset stats: mean_width=33px, 40% objects <48px.
    px=36: boosts objects with max_side < 36px → ~25% of objects
    px=48: boosts objects with max_side < 48px → ~40% of objects

REQUIRES:
  loss_function/loss2.py copied to ultralytics/utils/loss.py

  BEFORE RUNNING:
    cp loss_function/loss2.py /path/to/ultralytics/utils/loss.py

Usage:
  python run_swa_v1_boost_ablation.py
  python run_swa_v1_boost_ablation.py swa1c_b120_px36
  python run_swa_v1_boost_ablation.py --with-test
"""

import sys
import time
import gc
import json
import os
import torch
from ultralytics import YOLO

# =============================================================================
# STEP 0 — REGISTER CUSTOM KEYS
# =============================================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_LOSS_DEFAULTS = {
    "alpha_start": 0.0, "alpha_end": 0.0, "alpha_min": 0.0, "alpha_max": 0.0,
    "small_obj_px": 48, "small_obj_boost": 1.0,
    "area_weight_min": 0.4, "area_weight_mode": "inv",
    "weight_renorm": 1, "area_mode": "fixed",
    "area_ref_px": 64.0, "area_gamma": 0.5, "area_w_cap": 3.0,
    "small_obj_boost_backpack": -1.0, "small_obj_boost_bag": -1.0,
    "small_obj_boost_trolley": -1.0, "dfl_small_boost": 1.0,
    "nwd_ratio": 0.0, "nwd_c": 64.0,
    "center_loss_weight_init": 0.0, "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35, "center_loss_mode": "small",
    "center_crowd_iou": 0.1,
    "iou_clip_start": 999.0, "iou_clip_end": 999.0,
    "dfl_clip_start": 999.0, "dfl_clip_end": 999.0,
    "use_percentile_clip": False, "clip_percentile": 0.95, "use_loss_clip": False,
    "tal_topk": 10, "tal_alpha": 0.5, "tal_beta": 6.0,
    "use_satal": False, "satal_alpha_small": 1.2, "satal_beta_small": 4.5,
    "satal_alpha_large": 1.0, "satal_beta_large": 6.0,
    "satal_small_area": 0.0025, "satal_large_area": 0.0225,
    "satal_topk_factor": 1.3,
    "use_shape_tal": False, "shape_gamma": 0.1, "shape_min": 0.1,
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
PROJECT_DIR = "runs_swa_v1_boost"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# WINNING ALPHA SCHEDULE (fixed for all runs)
# =============================================================================
_ALPHA_BASE = dict(
    alpha_start=0.9,
    alpha_end=0.3,
    alpha_min=0.3,
    alpha_max=0.9,
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
# 8 RUNS — 4 boost x 2 px threshold
#
# boost:  1.2 (gentle), 1.5 (moderate), 2.0 (strong), 3.0 (aggressive)
# px:     36 (tight, ~25% objects), 48 (wide, ~40% objects)
# =============================================================================
RUNS = [
    # ---- px=36 (tight — only the smallest objects get boosted) ----
    {
        "name": "swa1c_b120_px36",
        "label": "[1/8] boost=1.2, px=36 — gentle boost, tight threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 1.2, "small_obj_px": 36},
        "seed": SEED,
    },
    {
        "name": "swa1c_b150_px36",
        "label": "[2/8] boost=1.5, px=36 — moderate boost, tight threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 1.5, "small_obj_px": 36},
        "seed": SEED,
    },
    {
        "name": "swa1c_b200_px36",
        "label": "[3/8] boost=2.0, px=36 — strong boost, tight threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 2.0, "small_obj_px": 36},
        "seed": SEED,
    },
    {
        "name": "swa1c_b300_px36",
        "label": "[4/8] boost=3.0, px=36 — aggressive boost, tight threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 3.0, "small_obj_px": 36},
        "seed": SEED,
    },

    # ---- px=48 (wide — most small objects get boosted) ----
    {
        "name": "swa1c_b120_px48",
        "label": "[5/8] boost=1.2, px=48 — gentle boost, wide threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 1.2, "small_obj_px": 48},
        "seed": SEED,
    },
    {
        "name": "swa1c_b150_px48",
        "label": "[6/8] boost=1.5, px=48 — moderate boost, wide threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 1.5, "small_obj_px": 48},
        "seed": SEED,
    },
    {
        "name": "swa1c_b200_px48",
        "label": "[7/8] boost=2.0, px=48 — strong boost, wide threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 2.0, "small_obj_px": 48},
        "seed": SEED,
    },
    {
        "name": "swa1c_b300_px48",
        "label": "[8/8] boost=3.0, px=48 — aggressive boost, wide threshold",
        "params": {**_ALPHA_BASE, "small_obj_boost": 3.0, "small_obj_px": 48},
        "seed": SEED,
    },
]


# =============================================================================
# EPOCH CALLBACK — CRITICAL for alpha annealing
# =============================================================================
def on_train_epoch_start(trainer):
    """Sync epoch into the custom loss (drives alpha schedule)."""
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
    px = params.get("small_obj_px", 48)

    print(f"  Alpha: {a_start} → {a_end}")
    print(f"  Boost: {boost}")
    print(f"  Px threshold: {px}")
    print(f"  Loss: loss2.py (v1 SWA)\n")

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("[setup] Epoch callback registered")

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
        "config": {
            "alpha_start": a_start,
            "alpha_end": a_end,
            "small_obj_boost": boost,
            "small_obj_px": px,
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

    print(f"\n{'='*72}")
    print(f"  SWA v1 BOOST + PX SWEEP ({len(runs)} runs)")
    print(f"  Alpha: 0.9→0.3 (fixed, proven winner)")
    print(f"  Sweep: 4 boost (1.2, 1.5, 2.0, 3.0) x 2 px (36, 48)")
    print(f"  Reference: alpha-only (boost=1.0) = 0.5748 mAP50-95")
    print(f"  Baseline:  v6_default2              = 0.5684 mAP50-95")
    print(f"  Loss: loss2.py, epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}")
    print(f"{'='*72}")
    print(f"\n  {'Name':<22s} {'Boost':>6s} {'Px':>4s}")
    print(f"  {'-'*34}")
    for r in runs:
        p = r["params"]
        print(f"  {r['name']:<22s} {p['small_obj_boost']:>6.1f} {p['small_obj_px']:>4d}")
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

    # Print results as grid
    print(f"\n\n{'='*72}")
    print("  BOOST x PX RESULTS (alpha 0.9→0.3 fixed)")
    print(f"{'='*72}")
    print(f"  Reference: alpha-only (boost=1.0) = 0.5748 mAP50-95")
    print(f"  Baseline:  v6_default2              = 0.5684 mAP50-95")
    print()
    print(f"{'Name':<22s} {'Boost':>6s} {'Px':>4s} {'mAP50':>8s} {'mAP50-95':>10s} "
          f"{'Prec':>7s} {'Recall':>7s}")
    print("-" * 62)
    for r in all_results:
        v = r.get("val", {})
        c = r.get("config", {})
        print(f"{r['name']:<22s} {c.get('small_obj_boost', 0):>6.1f} "
              f"{c.get('small_obj_px', 0):>4d} "
              f"{v.get('mAP50', 0):>8.4f} {v.get('mAP50_95', 0):>10.4f} "
              f"{v.get('precision', 0):>7.4f} {v.get('recall', 0):>7.4f}")

    # Grid view
    print(f"\n  mAP50-95 Grid (boost x px):")
    print(f"  {'':>10s} {'px=36':>10s} {'px=48':>10s}")
    for boost_val in [1.2, 1.5, 2.0, 3.0]:
        row = f"  boost={boost_val:<4.1f}"
        for px_val in [36, 48]:
            match = next(
                (r for r in all_results
                 if r.get("config", {}).get("small_obj_boost") == boost_val
                 and r.get("config", {}).get("small_obj_px") == px_val),
                None
            )
            if match and "val" in match:
                val = match["val"]["mAP50_95"]
                delta = val - 0.5748
                row += f" {val:>8.4f} ({delta:+.4f})"
            else:
                row += f" {'N/A':>8s} {'':>8s}"
        print(row)

    # Deltas
    alpha_only = 0.5748
    baseline = 0.5684
    print(f"\n  Deltas vs alpha-only (0.5748) and baseline (0.5684):")
    best_name, best_val = None, -1
    for r in all_results:
        v = r.get("val", {})
        if not v:
            continue
        m95 = v.get("mAP50_95", 0)
        d_alpha = m95 - alpha_only
        d_base = m95 - baseline
        marker = " <<< BEATS ALPHA-ONLY" if d_alpha > 0.001 else ""
        print(f"    {r['name']:<22s} vs alpha={d_alpha:+.4f}  "
              f"vs base={d_base:+.4f} ({d_base/baseline*100:+.2f}%){marker}")
        if m95 > best_val:
            best_val = m95
            best_name = r["name"]

    if best_name:
        best_r = next(r for r in all_results if r["name"] == best_name)
        best_c = best_r.get("config", {})
        print(f"\n  >>> BEST: {best_name} "
              f"(boost={best_c.get('small_obj_boost')}, px={best_c.get('small_obj_px')}, "
              f"mAP50-95={best_val:.4f})")

    print(f"\nSummary saved: {summary_path}")
