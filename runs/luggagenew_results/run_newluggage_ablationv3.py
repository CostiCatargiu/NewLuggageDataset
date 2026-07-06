#!/usr/bin/env python3
"""
Round 3 — SATAL (Scale-Adaptive TAL, Section E): prior-winner revival,
balanced variant, combination with the winning SWA recipe, and a
cross-file anchor.

Background:
  - Rounds 1-2 established: SWA works (+0.8 mAP50 via small-object recall,
    winner: constant alpha 0.6, boost 1.75 @48px), uniform TAL deviation
    hurts (stock 10/0.5/6.0 optimal, U-curve both directions), clips inert.
  - SATAL is the untested assigner hypothesis: loosen assignment ONLY for
    small objects, keep large at stock — sidesteps the uniform-loosening
    pathology.
  - R3/R4 configs revive the two best candidates from the prior SATAL
    experiment series (EXPERIMENTS R1-R6), reconstructed on top of that
    study's baseline: base TAL 12/0.6/5.0, alpha_small 1.2, beta_small 4.5,
    thresholds 0.0025/0.0225, topk_factor 1.3.

Runs (~11h total):
  1. r3_satal_r3     — R3 revival: beta_small 5.0, factor 1.5 (small topk≈18)
  2. r3_satal_r4     — R4 balanced: + alpha_small 1.3, small_area 0.0035
  3. r3_satal_r3_swa — R3 SATAL + SWA const 0.6 (the combined candidate)
  4. r3_swa_anchor   — SWA const 0.6, stock TAL, SATAL off
                       (cross-file reproducibility check: must land within
                        ±0.35 of r2_swa_const06 = 83.19 / 56.61, else the
                        loss file has behavioral drift and the round is void)

PREFLIGHT — MANDATORY before launch:
  [ ] loss file: class weighting behind a toggle and OFF by default
      (hardcoded ALWAYS-ON confounds every comparison to Rounds 1-2)
  [ ] use_nwd pinned False here (file defaults disagree between __init__
      and BboxLoss — do not rely on either)
  [ ] E2EDetectLoss NameError (v8DetectionLossLuggage) fixed
  [ ] ultralytics/utils/satal.py importable, assigner accepts set_imgsz
  [ ] per-anchor stride fix present (SWA runs 3,4 depend on it)
  [ ] epoch-sync callback works with this file (watch [Alpha] on run 3/4)

Sanity checks on epoch 1:
  - run 1/2 config printout: use_satal True, SWA alphas all 0.0, CW off
  - run 3/4: [Alpha] flat at 0.600
  - run 4: use_satal False, tal 10/0.5/6.0

Reference numbers (70% subset, 70ep, seed 0, test):
  default 82.54/56.84 | swa_const06 83.19/56.61 | noise floor ±0.35

Usage:
  python run_round3_satal.py
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
PROJECT_DIR = "runs_luggage_round3_satal"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared blocks
# =============================================================================
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0,
)
_SWA_CONST06 = dict(   # the Round-2 winning recipe
    alpha_start=0.6, alpha_end=0.6, alpha_min=0.6, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=1.75,
)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
_NWD_OFF = dict(use_nwd=False)   # pin explicitly — file defaults disagree

# =============================================================================
# SATAL — prior-study baseline + the two revived candidates
# =============================================================================
# Base TAL feeds the assigner's stock/large side. These are the prior SATAL
# study's baseline values (from its SATAL_PARAMS "works well" block), NOT
# the Round-1/2 stock TAL.
_SATAL_BASE_TAL = dict(
    tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_large=1.0, satal_beta_large=6.0,
)

_SATAL_R3 = dict(                 # prior best — EXACT revival
    use_satal=True, **_SATAL_BASE_TAL,
    satal_alpha_small=1.2,        # baseline value (R3 changed only beta/factor)
    satal_beta_small=5.0,         # R3 change (from 4.5)
    satal_topk_factor=1.5,        # R3 change (from 1.3) → small topk ≈ 18
    satal_small_area=0.0025,
    satal_large_area=0.0225,
)

_SATAL_R4 = dict(                 # "balanced" — all levers gently
    use_satal=True, **_SATAL_BASE_TAL,
    satal_alpha_small=1.3,
    satal_beta_small=5.0,
    satal_topk_factor=1.5,
    satal_small_area=0.0035,      # small zone widened to ~38px
    satal_large_area=0.0225,
)

# =============================================================================
# RUN CONFIGS
# =============================================================================
R3_SATAL_R3 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF, **_NWD_OFF, **_SATAL_R3)

R3_SATAL_R4 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF, **_NWD_OFF, **_SATAL_R4)

R3_SATAL_R3_SWA = dict(**_SWA_CONST06, **_CENTER_OFF, **_CLIP_OFF, **_NWD_OFF, **_SATAL_R3)

R3_SWA_ANCHOR = dict(**_SWA_CONST06, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
                     **_NWD_OFF, use_satal=False)

RUNS = [
    {"name": "r3_satal_r3",     "label": "SATAL R3 revival — base 12/0.6/5.0, a_s1.2/b_s5.0, factor 1.5, thresh 0.0025/0.0225", "params": R3_SATAL_R3,     "seed": 0},
    {"name": "r3_satal_r4",     "label": "SATAL R4 balanced — a_s1.3/b_s5.0, factor 1.5, thresh 0.0035/0.0225",                 "params": R3_SATAL_R4,     "seed": 0},
    {"name": "r3_satal_r3_swa", "label": "SATAL R3 + SWA const 0.6 boost 1.75 @48 — combined candidate",                        "params": R3_SATAL_R3_SWA, "seed": 0},
    {"name": "r3_swa_anchor",   "label": "SWA const 0.6 boost 1.75 @48, stock TAL, SATAL off — cross-file anchor",              "params": R3_SWA_ANCHOR,   "seed": 0},
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
    print(f"  ROUND 3 — SATAL: R3 revival + R4 + combined + anchor (4 runs)")
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
    print(f"  {'Run':<26}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 70}")
    for r in summary:
        def fmt(v, pct=True):
            if v != v:  # NaN
                return "n/a"
            return f"{v * 100:.2f}%" if pct else f"{v:.2f}"
        print(f"  {r['name']:<26}{fmt(r['elapsed_h'], pct=False):>9}"
              f"{fmt(r['val_map50']):>11}{fmt(r['test_map50']):>12}"
              f"{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()