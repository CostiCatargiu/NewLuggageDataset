#!/usr/bin/env python3
"""
SWA v1 Alpha Schedule Ablation 2 — Push alpha limits further.

PREVIOUS RESULTS (ablation 1, boost=1.0 OFF):
  BEST:  swa1_09_03  alpha 0.9→0.3  mAP50-95 = 0.5748  (+0.63 vs baseline)
         swa1_05_03  alpha 0.5→0.3  mAP50-95 = 0.5718  (+0.33)
         swa1_07_05  alpha 0.7→0.5  mAP50-95 = 0.5712  a(+0.27)
  WORST: swa1_09_06  alpha 0.9→0.6  mAP50-95 = 0.5638  (-0.47)
  BASE:  r9_anchor   OFF            mAP50-95 = 0.5685

  Pattern: wider spread wins, higher start wins, lower end wins.
  0.9→0.3 is the widest tested. Does the trend continue?

THIS ABLATION (4 runs, push extremes):
  swa1b_095_02     alpha 0.95→0.2   highest start, lowest end
  swa1b_09_02      alpha 0.9→0.2    same start as winner, even lower end
  swa1b_09_01      alpha 0.9→0.1    same start, push end to near-zero
  swa1b_095_03     alpha 0.95→0.3   higher start than winner, same end

REQUIRES:
  loss_function/loss2.py copied to ultralytics/utils/loss.py

  BEFORE RUNNING:
    cp loss_function/loss2.py /path/to/ultralytics/utils/loss.py

Usage:
  python run_swa_v1_ablation2.py
  python run_swa_v1_ablation2.py --with-test
  python run_swa_v1_ablation2.py swa1b_09_02
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
PROJECT_DIR = "runs_swa_v1_ablation2"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# ALL-OFF BASELINE
# =============================================================================
_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=48, small_obj_boost=1.0,
    area_weight_mode="inv", nwd_ratio=0.0, iou_type="ciou",
    use_satal=False, use_loss_clip=False, use_shape_tal=False,
    center_loss_weight_init=0.0, dfl_small_boost=1.0,
)

# =============================================================================
# 4 RUNS — push the extremes, NO boost
#
# Previous best: 0.9→0.3 = 0.5748 mAP50-95
# Trend: wider spread + higher start + lower end = better
# =============================================================================
RUNS = [
    # {
    #     "name": "swa1b_095_02",
    #     "label": "[1/6] alpha 0.95→0.2 — highest start, lowest end tested",
    #     "params": {
    #         **_ALL_OFF,
    #         "alpha_start": 0.95, "alpha_end": 0.2,
    #         "alpha_min": 0.2, "alpha_max": 0.95,
    #     },
    #     "seed": SEED,
    # },
    # {
    #     "name": "swa1b_09_02",
    #     "label": "[2/6] alpha 0.9→0.2 — winner's start, lower end",
    #     "params": {
    #         **_ALL_OFF,
    #         "alpha_start": 0.9, "alpha_end": 0.2,
    #         "alpha_min": 0.2, "alpha_max": 0.9,
    #     },
    #     "seed": SEED,
    # },
    # {
    #     "name": "swa1b_09_01",
    #     "label": "[3/6] alpha 0.9→0.1 — winner's start, near-zero end",
    #     "params": {
    #         **_ALL_OFF,
    #         "alpha_start": 0.9, "alpha_end": 0.1,
    #         "alpha_min": 0.1, "alpha_max": 0.9,
    #     },
    #     "seed": SEED,
    # },
    # {
    #     "name": "swa1b_095_03",
    #     "label": "[4/6] alpha 0.95→0.3 — higher start, same end as winner",
    #     "params": {
    #         **_ALL_OFF,
    #         "alpha_start": 0.95, "alpha_end": 0.3,
    #         "alpha_min": 0.3, "alpha_max": 0.95,
    #     },
    #     "seed": SEED,
    # },

    # # ---- Start=0.8 (gap between 0.7 and 0.9 — never tested) ----
    # {
    #     "name": "swa1b_08_02",
    #     "label": "[5/6] alpha 0.8→0.2 — fill the 0.8 start gap, wide spread",
    #     "params": {
    #         **_ALL_OFF,
    #         "alpha_start": 0.8, "alpha_end": 0.2,
    #         "alpha_min": 0.2, "alpha_max": 0.8,
    #     },
    #     "seed": SEED,
    # },
    {
        "name": "swa1b_08_03",
        "label": "[6/6] alpha 0.8→0.3 — fill the 0.8 start gap, same end as winner",
        "params": {
            **_ALL_OFF,
            "alpha_start": 0.8, "alpha_end": 0.3,
            "alpha_min": 0.3, "alpha_max": 0.8,
        },
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

    print(f"  Alpha: {a_start} → {a_end}")
    print(f"  Boost: {boost} (OFF)")
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
        "alpha_schedule": {
            "alpha_start": a_start,
            "alpha_end": a_end,
            "alpha_min": params.get("alpha_min", 0.0),
            "alpha_max": params.get("alpha_max", 0.0),
            "small_obj_boost": boost,
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
    print(f"  SWA v1 ALPHA — PUSH LIMITS ({len(runs)} runs)")
    print(f"  Previous best: 0.9→0.3 = 0.5748 mAP50-95")
    print(f"  Question: does wider spread keep improving?")
    print(f"  Boost: OFF (1.0), loss: loss2.py")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}, seed={SEED}")
    print(f"{'='*72}")
    print(f"\n  {'Name':<22s} {'Start':>6s} {'End':>6s}")
    print(f"  {'-'*36}")
    for r in runs:
        p = r["params"]
        print(f"  {r['name']:<22s} {p['alpha_start']:>6.2f} {p['alpha_end']:>6.2f}")
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
    print("  RESULTS — Push Alpha Limits (boost=1.0)")
    print(f"{'='*72}")
    print(f"  Reference: swa1_09_03 (0.9→0.3) = 0.5748 mAP50-95")
    print(f"             baseline (OFF)        = 0.5685 mAP50-95")
    print()
    print(f"{'Name':<22s} {'Schedule':>10s} {'mAP50':>8s} {'mAP50-95':>10s} "
          f"{'Prec':>7s} {'Recall':>7s}")
    print("-" * 68)
    for r in all_results:
        v = r.get("val", {})
        s = r.get("alpha_schedule", {})
        sched = f"{s.get('alpha_start', 0)}→{s.get('alpha_end', 0)}"
        print(f"{r['name']:<22s} {sched:>10s} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {v.get('precision', 0):>7.4f} "
              f"{v.get('recall', 0):>7.4f}")

    # Deltas vs previous best
    prev_best = 0.5748
    baseline = 0.5685
    print(f"\n  Deltas:")
    for r in all_results:
        v = r.get("val", {})
        if not v:
            continue
        d_prev = v.get("mAP50_95", 0) - prev_best
        d_base = v.get("mAP50_95", 0) - baseline
        marker = " <<< NEW BEST" if d_prev > 0.001 else ""
        print(f"    {r['name']:<22s} vs prev_best={d_prev:+.4f}  vs baseline={d_base:+.4f}{marker}")

    print(f"\nSummary saved: {summary_path}")
