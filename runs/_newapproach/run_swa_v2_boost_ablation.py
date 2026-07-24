#!/usr/bin/env python3
"""
SWA v2 Bounded Boost Ablation — 4 runs testing the v2 bounded boost mechanism
on top of the proven alpha 0.9→0.3 schedule.

BACKGROUND
  v1 boost (1/area, 400:1 spread) hurt mAP50-95 in ALL 8 configs tested.
  v2 boost is fundamentally different:
    - Width-keyed (pixels), not area-keyed (grid cells)
    - Bounded in [1, swa_boost], not unbounded 1/area
    - Fixed threshold (stable), not batch-max normalized (jittery)
    - Max ratio is e.g. 3:1, not 400:1

  Question: does the v2 bounded boost stack where v1 failed?

  Alpha 0.9→0.3 is proven (+1.13% vs baseline on v1). We keep the same
  alpha concept on v2 and only test the boost.

RUNS (4 total):
  swa2_b125    alpha 0.9→0.3, boost=1.25, thresh=33px (gentle)
  swa2_b150    alpha 0.9→0.3, boost=1.5,  thresh=33px (moderate)
  swa2_b200    alpha 0.9→0.3, boost=2.0,  thresh=33px (strong)
  swa2_b300    alpha 0.9→0.3, boost=3.0,  thresh=33px (aggressive)

  thresh=33px = dataset mean width. Objects narrower than 33px get boosted.

REQUIRES:
  loss_function/loss.py (v2 SWA) copied to ultralytics/utils/loss.py
  THIS IS A DIFFERENT LOSS FILE THAN THE v1 RUNS!

  BEFORE RUNNING:
    cp loss_function/loss.py /path/to/ultralytics/utils/loss.py

Usage:
  python run_swa_v2_boost_ablation.py
  python run_swa_v2_boost_ablation.py swa2_b150
  python run_swa_v2_boost_ablation.py --with-test
"""

import sys
import time
import gc
import json
import os
import torch
from ultralytics import YOLO

# =============================================================================
# STEP 0 — REGISTER CUSTOM KEYS (v2 parameter names)
# =============================================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_LOSS_DEFAULTS = {
    # SATAL (all OFF)
    "use_satal": False,
    "tal_topk": 10, "tal_alpha": 0.5, "tal_beta": 6.0,
    "satal_alpha_small": 1.5, "satal_beta_small": 3.0,
    "satal_alpha_large": 1.0, "satal_beta_large": 6.0,
    "satal_small_area": 0.0025, "satal_large_area": 0.0225,
    "satal_topk_factor": 1.5,
    # SWA v2 (bounded)
    "swa_mode": "scale",
    "swa_alpha": 0.0,
    "swa_alpha_end": None,
    "swa_size_axis": "width",
    "swa_boost": 1.0,
    "swa_width_thresh_px": 24.0,
    "swa_area_thresh_px2": 1024.0,
    # box metric
    "box_loss_type": "ciou",
    "wiou_alpha": 1.9, "wiou_delta": 3.0, "wiou_momentum": 0.02,
    # NWD (OFF)
    "use_nwd": False, "nwd_mode": "blend", "nwd_weight": 0.5,
    "nwd_c_px": 12.0, "nwd_small_width_px": 24.0, "nwd_debug": False,
    # cls (OFF)
    "use_class_weights": False, "class_counts": None,
    "use_vfl": False, "vfl_alpha": 0.75, "vfl_gamma": 2.0,
    # clipping (OFF)
    "use_loss_clip": False, "iou_clip": 2.0, "dfl_clip": 5.0,
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
PROJECT_DIR = "runs_swa_v2_boost"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# BASE CONFIG — alpha 0.9→0.3 on v2, everything else OFF
# =============================================================================
_BASE = dict(
    swa_mode="blend",
    swa_alpha=0.9,
    swa_alpha_end=0.3,
    swa_size_axis="width",
    swa_width_thresh_px=33.0,
    swa_area_thresh_px2=1024.0,
    box_loss_type="ciou",
    use_satal=False,
    use_nwd=False,
    use_class_weights=False,
    use_vfl=False,
    use_loss_clip=False,
)

# =============================================================================
# 4 RUNS — v2 bounded boost sweep
# =============================================================================
RUNS = [
    {
        "name": "swa2_b125",
        "label": "[1/4] v2 boost=1.25, thresh=33px — gentle bounded boost",
        "params": {**_BASE, "swa_boost": 1.25},
        "seed": SEED,
    },
    {
        "name": "swa2_b150",
        "label": "[2/4] v2 boost=1.5, thresh=33px — moderate bounded boost",
        "params": {**_BASE, "swa_boost": 1.5},
        "seed": SEED,
    },
    {
        "name": "swa2_b200",
        "label": "[3/4] v2 boost=2.0, thresh=33px — strong bounded boost",
        "params": {**_BASE, "swa_boost": 2.0},
        "seed": SEED,
    },
    {
        "name": "swa2_b300",
        "label": "[4/4] v2 boost=3.0, thresh=33px — aggressive bounded boost",
        "params": {**_BASE, "swa_boost": 3.0},
        "seed": SEED,
    },
]


# =============================================================================
# EPOCH CALLBACK — for alpha annealing
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
# PREFLIGHT — verify v2 loss is active
# =============================================================================
def preflight():
    """Verify the active loss module is v2 SWA (loss_function/loss.py)."""
    import inspect
    from ultralytics.utils import loss as loss_mod

    src = loss_mod.__file__
    source = inspect.getsource(loss_mod)

    has_swa_boost = "swa_boost" in source
    has_size_weight = "_size_weight" in source
    has_swa_mode = "swa_mode" in source
    has_width_thresh = "swa_width_thresh_px" in source

    # Make sure it's NOT v1 (v1 has "small_obj_boost", v2 has "swa_boost")
    has_v1_boost = "small_obj_boost" in source and "swa_boost" not in source

    if has_v1_boost:
        print(f"[preflight] FAIL — v1 loss detected (has small_obj_boost but no swa_boost)")
        print(f"             Source: {src}")
        print(f"\n  You need the v2 loss. Run:")
        print(f"    cp loss_function/loss.py /path/to/ultralytics/utils/loss.py")
        return False

    if has_swa_boost and has_size_weight and has_swa_mode and has_width_thresh:
        print(f"[preflight] OK  v2 SWA loss active: {src}")
        return True
    else:
        print(f"[preflight] FAIL — v2 SWA features not found")
        print(f"             Source: {src}")
        print(f"\n  Run: cp loss_function/loss.py /path/to/ultralytics/utils/loss.py")
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
    print(f"{'='*72}")

    boost = params.get("swa_boost", 1.0)
    thresh = params.get("swa_width_thresh_px", 33.0)
    alpha = params.get("swa_alpha", 0.0)
    alpha_end = params.get("swa_alpha_end", None)

    print(f"  Alpha: {alpha} → {alpha_end}")
    print(f"  Boost: {boost} (v2 bounded, width-keyed)")
    print(f"  Thresh: {thresh}px")
    print(f"  Loss: loss_function/loss.py (v2 SWA)\n")

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
            "swa_alpha": alpha,
            "swa_alpha_end": alpha_end,
            "swa_boost": boost,
            "swa_width_thresh_px": thresh,
            "swa_mode": params.get("swa_mode", "blend"),
            "implementation": "v2_bounded",
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

    if not preflight():
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != "y":
            sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  SWA v2 BOUNDED BOOST ABLATION ({len(runs)} runs)")
    print(f"  Alpha: 0.9→0.3 (fixed, proven winner)")
    print(f"  Boost: v2 bounded, width-keyed, thresh=33px")
    print(f"  Sweep: 1.25, 1.5, 2.0, 3.0")
    print(f"  v1 alpha-only reference: 0.5748 mAP50-95")
    print(f"  v6_default2 baseline:    0.5684 mAP50-95")
    print(f"  Loss: loss_function/loss.py (v2)")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}, seed={SEED}")
    print(f"{'='*72}")
    print(f"\n  {'Name':<16s} {'Boost':>6s} {'Thresh':>7s} {'Mechanism':>12s}")
    print(f"  {'-'*44}")
    for r in runs:
        p = r["params"]
        print(f"  {r['name']:<16s} {p['swa_boost']:>6.2f} {p['swa_width_thresh_px']:>6.0f}px {'v2 bounded':>12s}")
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
    print("  v2 BOUNDED BOOST RESULTS (alpha 0.9→0.3)")
    print(f"{'='*72}")
    print(f"  v1 alpha-only (boost=1.0): 0.5748 mAP50-95")
    print(f"  v6_default2 baseline:      0.5684 mAP50-95")
    print(f"  v1 best boost (2.0@px48):  0.5722 mAP50-95 (hurt)")
    print()
    print(f"{'Name':<16s} {'Boost':>6s} {'mAP50':>8s} {'mAP50-95':>10s} "
          f"{'Prec':>7s} {'Recall':>7s}")
    print("-" * 52)
    for r in all_results:
        v = r.get("val", {})
        c = r.get("config", {})
        print(f"{r['name']:<16s} {c.get('swa_boost', 0):>6.2f} "
              f"{v.get('mAP50', 0):>8.4f} {v.get('mAP50_95', 0):>10.4f} "
              f"{v.get('precision', 0):>7.4f} {v.get('recall', 0):>7.4f}")

    # Deltas
    alpha_only = 0.5748
    v1_best_boost = 0.5722
    baseline = 0.5684
    print(f"\n  Comparison:")
    for r in all_results:
        v = r.get("val", {})
        if not v:
            continue
        m95 = v.get("mAP50_95", 0)
        c = r.get("config", {})
        d_alpha = m95 - alpha_only
        d_v1boost = m95 - v1_best_boost
        d_base = m95 - baseline
        marker = " <<< BEATS ALPHA-ONLY!" if d_alpha > 0.001 else \
                 " <<< beats v1 boost" if d_v1boost > 0.001 else ""
        print(f"    {r['name']:<16s} vs alpha-only={d_alpha:+.4f}  "
              f"vs v1-boost={d_v1boost:+.4f}  vs baseline={d_base:+.4f}{marker}")

    print(f"\nSummary saved: {summary_path}")
