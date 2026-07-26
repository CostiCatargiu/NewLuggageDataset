#!/usr/bin/env python3
"""
Round 4 — regression-loss & assignment upgrades (Section I).

Motivated by Rounds 1-3 on the full test split:
  - mAP50-95 was STUCK at ~0.565 across every TAL/CLIP tweak; tight
    localization is the ceiling, small objects the weakest cell (~0.48-0.51).
  - SATAL raised mAP50/recall but LOWERED mAP50-95 (looser positives, looser
    boxes). Recall (~0.74) trailed precision (~0.82) everywhere.

This round changes what Rounds 1-3 never touched: the box REGRESSION loss and
its pairing with SATAL assignment. Everything is layered on the Round-2 winner
(SWA constant 0.6, boost 1.75 @48px) so the box loss is the isolated variable.

New loss levers (all in loss_satal_swa_plus_v2.py, Section I):
  box_loss_type : 'ciou' | 'mpdiou' | 'wiou' | 'focaler'
  use_nwd + nwd_mode='small_only'  : Wasserstein term for tiny boxes only
  swa_smooth    : continuous small-object boost (no hard step; stride-fixed)
  use_class_weighting : gate the (previously always-on) class weights
  cls_mode      : 'bce' | 'qfl'

Runs — ORDERED BY DESCENDING CONFIDENCE OF A MEANINGFUL GAIN.
(each config PINS every toggle for reproducibility — no default drift)
  1. r4_satal_mpdiou   — SATAL R3 assignment + MPDIoU. Highest expected value:
                         attacks BOTH weaknesses (SATAL's recall/mAP50 + MPDIoU
                         tight boxes), clear mechanism, MPDIoU is parameter-free.
  2. r4_mpdiou         — SWA const06 + MPDIoU. Cleanest low-variance positive;
                         isolates "does a tighter box loss lift mAP50-95".
  3. r4_satal_wiou     — SATAL R3 + Wise-IoU v3. Same "have both" idea, higher
                         ceiling but more variance (WIoU alpha/delta/EMA).
  4. r4_nwd_small      — SWA const06 + NWD small_only. Targets the weakest cell
                         (small mAP50-95) directly; upside high but nwd_C may
                         need a second pass off the debug printout.
  5. r4_wiou           — SWA const06 + Wise-IoU v3. Moderate, tuning-sensitive.
  6. r4_focaler        — SWA const06 + Focaler-CIoU. Lower prob, moderate size.
  7. r4_swa_smooth     — SWA const06 continuous boost (stride-fixed). High prob
                         of a SMALL gain (it is basically the winner, debugged).
  8. r4_baseline_clean — SWA const06, CIoU, class-weighting OFF. Reference/
                         denominator, not an improvement bet — but you may want
                         to run it FIRST anyway to lock the clean baseline number.

PREFLIGHT — MANDATORY before launch:
  [ ] ultralytics loss module points at loss_satal_swa_plus_v2.py
      (drop-in replacement; same relative imports .metrics / .tal)
  [ ] the new hyp keys (box_loss_type, wiou_*, focaler_*, swa_smooth,
      swa_boost_power, use_loss_clip, use_class_weighting, cls_mode, qfl_beta,
      use_nwd, nwd_mode, nwd_C) are accepted by your cfg whitelist — the same
      way alpha_start / satal_* are already passed through model.train()
  [ ] epoch-0 config banner prints the intended [I] box_loss_type and [F] flag
  [ ] ultralytics/utils/satal.py importable (runs 1 & 3)
  [ ] per-anchor stride fix now lives in BboxLoss._compute_weights (this file)

Sanity checks on epoch 1 banner:
  - run r4_satal_mpdiou / r4_satal_wiou: use_satal True, box mpdiou/wiou
  - runs on SWA const06: [Alpha] flat 0.600
  - run r4_nwd_small: use_nwd True, nwd_mode small_only
  - run r4_baseline_clean: box ciou, Class Weighting OFF, use_satal False

Reference numbers (70% subset, 70ep, seed 0, test):
  default 82.54/56.84 | swa_const06 83.19/56.61 | noise floor +/-0.35

Usage:
  python run_newluggage_ablationv4.py
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
PROJECT_DIR = "runs_luggage_round4"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared blocks (identical to Rounds 2-3)
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

# --- SATAL R3 assignment (prior best; from Round 3) ---
_SATAL_R3 = dict(
    use_satal=True,
    tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_alpha_small=1.2, satal_beta_small=5.0,
    satal_topk_factor=1.5,
    satal_small_area=0.0025, satal_large_area=0.0225,
)

# Common Section-I / toggle pins so nothing drifts on defaults.
# Individual runs override only the lever under test.
_PINS = dict(
    use_satal=False,
    use_nwd=False, nwd_mode="small_only", nwd_C=4.0,
    box_loss_type="ciou",
    swa_smooth=False, swa_boost_power=0.5,
    use_loss_clip=False,           # inert in Rounds 1-3; off for a clean signal
    use_class_weighting=True,      # keep ON to match Rounds 1-3 (except clean ref)
    cls_mode="bce",
)


def _cfg(**overrides):
    """Build a run config: pins -> center/clip off -> SWA-const06 -> stock TAL -> overrides."""
    c = dict(_PINS)
    c.update(_CENTER_OFF)
    c.update(_CLIP_OFF)
    c.update(_SWA_CONST06)
    c.update(_TAL_STOCK)
    c.update(overrides)
    return c


# =============================================================================
# RUN CONFIGS
# =============================================================================
R4_BASELINE_CLEAN = _cfg(box_loss_type="ciou", use_class_weighting=False)
R4_MPDIOU         = _cfg(box_loss_type="mpdiou")
R4_WIOU           = _cfg(box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)
R4_FOCALER        = _cfg(box_loss_type="focaler", focaler_d=0.0, focaler_u=0.95)
R4_NWD_SMALL      = _cfg(box_loss_type="ciou", use_nwd=True, nwd_mode="small_only", nwd_C=4.0)
R4_SWA_SMOOTH     = _cfg(box_loss_type="ciou", swa_smooth=True, swa_boost_power=0.5)

# SATAL R3 assignment paired with a tight box loss ("have both" experiments).
# _SATAL_R3 overrides use_satal + TAL params; box loss overrides the pin.
R4_SATAL_MPDIOU = _cfg(box_loss_type="mpdiou", **_SATAL_R3)
R4_SATAL_WIOU   = _cfg(box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02, **_SATAL_R3)

# Ordered by descending confidence of a meaningful improvement (see docstring).
RUNS = [
    {"name": "r4_satal_mpdiou",   "label": "[1/8] SATAL R3 assignment + MPDIoU -- recall AND tight boxes",          "params": R4_SATAL_MPDIOU,   "seed": 0},
    {"name": "r4_mpdiou",         "label": "[2/8] SWA const06 + MPDIoU box loss (tight localization)",              "params": R4_MPDIOU,         "seed": 0},
    {"name": "r4_satal_wiou",     "label": "[3/8] SATAL R3 assignment + Wise-IoU v3",                               "params": R4_SATAL_WIOU,     "seed": 0},
    {"name": "r4_nwd_small",      "label": "[4/8] SWA const06 + NWD small_only (C=4.0) on CIoU base",               "params": R4_NWD_SMALL,      "seed": 0},
    {"name": "r4_wiou",           "label": "[5/8] SWA const06 + Wise-IoU v3 (1.9/3.0, mom 0.02)",                   "params": R4_WIOU,           "seed": 0},
    {"name": "r4_focaler",        "label": "[6/8] SWA const06 + Focaler-CIoU (d0.0/u0.95)",                         "params": R4_FOCALER,        "seed": 0},
    {"name": "r4_swa_smooth",     "label": "[7/8] SWA const06 continuous boost (stride-fixed, power 0.5)",          "params": R4_SWA_SMOOTH,     "seed": 0},
    {"name": "r4_baseline_clean", "label": "[8/8] SWA const06, CIoU, class-weighting OFF -- clean reference",       "params": R4_BASELINE_CLEAN, "seed": 0},
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
    print(f"  ROUND 4 -- regression-loss & assignment upgrades ({len(RUNS)} runs)")
    print(f"  Runs (confidence order): {', '.join(r['name'] for r in RUNS)}")
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

        # incremental summary dump -- survives a crash mid-study
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
    print(f"  {'Run':<22}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 66}")
    for r in summary:
        def fmt(v, pct=True):
            if v != v:  # NaN
                return "n/a"
            return f"{v * 100:.2f}%" if pct else f"{v:.2f}"
        print(f"  {r['name']:<22}{fmt(r['elapsed_h'], pct=False):>9}"
              f"{fmt(r['val_map50']):>11}{fmt(r['test_map50']):>12}"
              f"{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()
