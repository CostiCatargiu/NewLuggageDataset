#!/usr/bin/env python3
"""
Round 2 — new value regions for the 3 phases (6 runs, no default).

Motivated by the Round-1 section sweep (see runs_luggage_section_sweep):
  - SWA was the only section with a real gain (+0.8 mAP50, small-object recall),
    but alpha/boost were confounded and px was never varied → 3 new SWA points.
  - TAL was only ever tested LOOSER than stock and got worse → test the
    STRICT direction (2 runs).
  - Clips were inert at alpha=0 because per-sample losses never reached the
    caps → retest clips on top of SWA-high, where weights actually spike (1 run).

Runs:
  1. r2_swa_boost25   — SWA 0.5→0.25, boost 2.5 @48px  (boost strength, branch now live)
  2. r2_swa_const06   — SWA constant 0.6, boost 1.75 @48px  (no-decay hypothesis)
  3. r2_swa_px32      — SWA 0.5→0.25, boost 2.0 @32px  (concentrated boost)
  4. r2_tal_strict    — TAL 8/0.5/8.0  (strict direction, untested)
  5. r2_tal_beta7     — TAL 10/0.5/7.0 (beta-only strictening)
  6. r2_swahigh_clip  — SWA 0.9→0.4 b2.0 + loose clips (clips where they engage)

NOTE: SWA runs (1,2,3,6) require the per-anchor stride fix in
      BboxLoss._compute_weights.

Usage:
  python run_round2_sweep.py
"""

import time
import gc
import copy
import json
import os
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v4i.yolov12_70percentage/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_luggage_round2"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared "off" blocks
# =============================================================================
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0,
)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# =============================================================================
# ROUND 2 CONFIGS
# =============================================================================

# --- SWA: boost strength at the productive alpha (branch is live now) ---
R2_SWA_BOOST25 = dict(
    alpha_start=0.5, alpha_end=0.25, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=2.5,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)

# --- SWA: constant alpha 0.6, no decay ---
R2_SWA_CONST06 = dict(
    alpha_start=0.6, alpha_end=0.6, alpha_min=0.6, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=1.75,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)

# --- SWA: tighter px threshold — concentrate boost on the truly tiny ---
R2_SWA_PX32 = dict(
    alpha_start=0.5, alpha_end=0.25, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=32, small_obj_boost=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)

# --- TAL: strict direction ---
R2_TAL_STRICT = dict(
    **_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=8, tal_alpha=0.5, tal_beta=8.0,
)

# --- TAL: mild beta-only strictening ---
R2_TAL_BETA7 = dict(
    **_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=10, tal_alpha=0.5, tal_beta=7.0,
)

# --- CLIP retested where it can engage: on top of SWA-high ---
R2_SWAHIGH_CLIP = dict(
    alpha_start=0.9, alpha_end=0.4, alpha_min=0.3, alpha_max=0.95,
    small_obj_px=48, small_obj_boost=2.0,
    **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=40.0, iou_clip_end=30.0,   # eff. 4.0 → 3.0
    dfl_clip_start=50.0, dfl_clip_end=40.0,   # eff. 5.0 → 4.0
)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER
# =============================================================================
RUNS = [
    {"name": "r2_swa_boost25",  "label": "SWA 0.5→0.25 (min 0.2/max 0.8), boost 2.5 @ 48px — boost strength",      "params": R2_SWA_BOOST25,  "seed": 0},
    {"name": "r2_swa_const06",  "label": "SWA constant 0.6 (no decay, min 0.6/max 0.8), boost 1.75 @ 48px",        "params": R2_SWA_CONST06,  "seed": 0},
    {"name": "r2_swa_px32",     "label": "SWA 0.5→0.25 (min 0.2/max 0.8), boost 2.0 @ 32px — concentrated boost",  "params": R2_SWA_PX32,     "seed": 0},
    {"name": "r2_tal_strict",   "label": "TAL 8/0.5/8.0 — strict direction (untested)",                            "params": R2_TAL_STRICT,   "seed": 0},
    {"name": "r2_tal_beta7",    "label": "TAL 10/0.5/7.0 — beta-only strictening",                                 "params": R2_TAL_BETA7,    "seed": 0},
    {"name": "r2_swahigh_clip", "label": "SWA 0.9→0.4 (min 0.3/max 0.95) b2.0 @48 + loose clips 40→30/50→40",      "params": R2_SWAHIGH_CLIP, "seed": 0},
]


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


def run_one(run_cfg):
    name = run_cfg["name"]
    label = run_cfg["label"]
    params = run_cfg["params"]
    seed = run_cfg.get("seed", 0)

    print(f"\n{'=' * 70}")
    print(f"  RUN: {name}  (seed {seed})")
    print(f"  {label}")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    train_kwargs = {
        "data": DATA_YAML,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": PROJECT_DIR,
        "name": name,
        "patience": 100,
        "close_mosaic": 10,
        "seed": seed,
        "deterministic": True,
    }
    train_kwargs.update(copy.deepcopy(params))

    results = model.train(**train_kwargs)
    elapsed = (time.time() - start_time) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    # ---- persist ground-truth config next to the run results ----
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": label, "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
                       "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    # ---- val-split mAP50 from training results ----
    val_map50 = float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key])
                break
    except Exception:
        pass

    # ---- explicit TEST-split evaluation on best.pt ----
    test_map50, test_map5095 = float("nan"), float("nan")
    try:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        test_model = YOLO(best_pt)
        tm = test_model.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                            batch=BATCH, device=DEVICE, workers=WORKERS,
                            project=PROJECT_DIR, name=f"{name}_test")
        test_map50 = float(tm.box.map50)
        test_map5095 = float(tm.box.map)
        del test_model, tm
    except Exception as e:
        print(f"  [WARN] test eval failed: {e}")

    # Free GPU memory before the next run
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "label": label, "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "test_map50": test_map50,
            "test_map5095": test_map5095}


def main():
    print(f"\n{'=' * 70}")
    print(f"  ROUND 2 SWEEP — 6 RUNS (3 SWA new-region + 2 TAL strict + 1 SWA+clip)")
    print(f"  Runs: {', '.join(r['name'] for r in RUNS)}")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = []

    for run_cfg in RUNS:
        try:
            result = run_one(run_cfg)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "label": run_cfg["label"],
                      "seed": run_cfg.get("seed", 0),
                      "elapsed_h": float("nan"), "val_map50": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}
        summary.append(result)

        # incremental summary dump — survives a crash mid-study
        try:
            os.makedirs(PROJECT_DIR, exist_ok=True)
            with open(os.path.join(PROJECT_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

    total_elapsed = (time.time() - overall_start) / 3600

    print(f"\n{'=' * 70}")
    print(f"  ALL RUNS COMPLETE ({total_elapsed:.2f}h total)")
    print(f"{'=' * 70}")
    print(f"  {'Run':<24}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 68}")
    for r in summary:
        def fmt(v, pct=True):
            if v != v:  # NaN
                return "n/a"
            return f"{v * 100:.2f}%" if pct else f"{v:.2f}"
        print(f"  {r['name']:<24}{fmt(r['elapsed_h'], pct=False):>9}"
              f"{fmt(r['val_map50']):>11}{fmt(r['test_map50']):>12}"
              f"{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()