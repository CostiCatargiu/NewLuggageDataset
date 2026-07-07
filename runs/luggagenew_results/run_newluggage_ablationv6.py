#!/usr/bin/env python3
"""
Round 6 — classification loss & untested combinations (loss_satal_swa_plus_v2.py).

What Rounds 1-5 established (28 models, full test split):
  - mAP50-95 ceiling at ~0.568: no box loss (CIoU/MPDIoU/WIoU/Focaler), no
    TAL/SATAL variant, no NWD mode broke through. The ceiling is architectural.
  - SWA trades localization for recall: SWA-OFF holds best mAP50-95 (0.5684),
    SWA-const06 holds best mAP50 balance (0.8343+precision 0.8271).
  - WIoU set study-best precision (0.8337) on SWA-const06 base.
  - SATAL raises mAP50 (+0.3%) but LOWERS mAP50-95 (-2.1%), every time.
  - BAG class is the bottleneck: 22% of data, 20.6% small, 74% precision vs
    94% recall → false-positive problem, NOT a recall problem.

What was NEVER tested:
  1. QFL (Quality Focal Loss) for classification — cls_mode='qfl' exists in the
     loss file but every run across 5 rounds pinned cls_mode='bce'. QFL modulates
     gradients by |target - pred|^beta → harder examples get stronger signal.
     This directly targets the bag confusion problem (high recall, low precision).
  2. WIoU + QFL on the SWA-OFF base — combines the best box loss (WIoU, proven
     precision winner) with the best mAP50-95 base (SWA-OFF) AND the untested
     classification improvement (QFL). The theoretically strongest combination.
  3. MPDIoU on SWA-OFF — R5 planned this (r5_mpdiou_default) but it never ran.
     MPDIoU penalizes corner misalignment → tightest boxes on the best base.
  4. Aggressive bag class weighting — current sqrt-dampened weights give bag only
     1.53x trolley's weight despite a 15% performance gap. Removing the sqrt
     dampening gives bag ~2.3x trolley's weight.
  5. 120 epochs — all 28 models trained for 70 epochs; the weapon experiments in
     the same codebase used 200-300ep. 70 may be leaving performance on the table.

Runs — ORDERED BY DESCENDING CONFIDENCE OF A MEANINGFUL GAIN:
  1. r6_wiou_qfl_default  — WIoU + QFL on SWA-OFF. Theoretically strongest combo:
                            best box loss × best mAP50-95 base × untested cls.
  2. r6_qfl_default       — QFL on SWA-OFF + CIoU. Isolates QFL impact on the
                            best mAP50-95 base (clean signal, one variable).
  3. r6_mpdiou_default    — MPDIoU on SWA-OFF. The R5 run that never executed.
                            Corner-point alignment on the tight-localization base.
  4. r6_qfl_swa           — QFL on SWA-const06 + CIoU. Isolates QFL impact on
                            the best balanced base (the production model).
  5. r6_bag_boost_qfl     — Aggressive bag weights (no sqrt dampening) + QFL on
                            SWA-const06. Directly targets the 15% bag gap.
  6. r6_wiou_qfl_swa      — WIoU + QFL on SWA-const06. The full combo on the
                            production base — precision maximizer.
  7. r6_satal_soft_qfl    — Softened SATAL (factor 1.3) + QFL + CIoU on SWA-const06.
                            Can SATAL's mAP50 gain survive with QFL fixing its
                            precision penalty? Lowest confidence but highest ceiling.
  8. r6_best_120ep        — PLACEHOLDER: re-train the best of runs 1-7 for 120
                            epochs. Execute manually after analyzing runs 1-7.

PREFLIGHT — MANDATORY before launch:
  [ ] ultralytics loss module points at loss_satal_swa_plus_v2.py
  [ ] hyp keys whitelisted: box_loss_type, wiou_*, cls_mode, qfl_beta,
      use_class_weighting, class_weight_mode, use_nwd, use_loss_clip, swa_*
  [ ] class_weight_mode patch applied to loss_satal_swa_plus_v2.py (see below):
      Add to v8DetectionLoss.__init__ Section F, after line 718:
        self.class_weight_mode = getattr(h, 'class_weight_mode', 'sqrt')
        if self.class_weight_mode == 'linear':
            self.class_weights = inv_freq / inv_freq.mean()  # no sqrt dampening
      This enables 'linear' mode for run 5 (r6_bag_boost_qfl).
  [ ] epoch-0 banner: check [G] prints "QFL" for runs 1-6, [F] prints weights

Sanity checks on epoch 1 banner:
  - runs 1,2,3: [Alpha] flat 0.000 (SWA off)
  - runs 4,5,6: [Alpha] flat 0.600 (SWA const06)
  - run 7: use_satal True, satal_topk_factor 1.3
  - ALL runs: [G] Cls Loss: QFL
  - run 5: class weights should show ~[0.92, 1.41, 0.67] (linear mode)

Reference numbers (70% subset, 70ep, seed 0, test):
  v6_default2 82.54/56.84 (best mAP50-95) | swa_const06 83.19/56.61
  r4_wiou 83.19/56.43 P=83.37 (best precision) | noise floor +/-0.35

Usage:
  python run_newluggage_ablationv6.py
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
PROJECT_DIR = "runs_luggage_round6"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared blocks (identical to Rounds 2-5)
# =============================================================================
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0,
)
_SWA_CONST06 = dict(   # Round-2 winner (constant alpha 0.6, no decay)
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

# SATAL R3 assignment, SOFTENED: topk_factor 1.3 (was 1.5) -> less loosening
_SATAL_R3_SOFT = dict(
    use_satal=True,
    tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_alpha_small=1.2, satal_beta_small=5.0,
    satal_topk_factor=1.3,               # softened from 1.5 → smaller mAP50-95 hit
    satal_small_area=0.0025, satal_large_area=0.0225,
)

_WIOU = dict(wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)

# Toggle pins — every lever explicitly set, no default drift.
_PINS = dict(
    use_satal=False,
    use_nwd=False, nwd_mode="small_only", nwd_C=4.0,
    box_loss_type="ciou",
    swa_smooth=False, swa_boost_power=0.5,
    use_loss_clip=False,
    use_class_weighting=True,
    cls_mode="bce",                # overridden to 'qfl' in all R6 runs
    qfl_beta=2.0,
    class_weight_mode="sqrt",      # 'sqrt' = Rounds 1-5 default; 'linear' = aggressive
)


def _cfg(swa=_SWA_CONST06, **overrides):
    """pins -> center/clip off -> chosen SWA base -> stock TAL -> per-run overrides."""
    c = dict(_PINS)
    c.update(_CENTER_OFF)
    c.update(_CLIP_OFF)
    c.update(swa)
    c.update(_TAL_STOCK)
    c.update(overrides)
    return c


# =============================================================================
# RUN CONFIGS
# =============================================================================

# --- 1. WIoU + QFL on SWA-OFF: theoretically strongest combination ---
R6_WIOU_QFL_DEFAULT = _cfg(
    swa=_SWA_OFF,
    box_loss_type="wiou", **_WIOU,
    cls_mode="qfl", qfl_beta=2.0,
)

# --- 2. QFL on SWA-OFF + CIoU: isolate QFL impact on best mAP50-95 base ---
R6_QFL_DEFAULT = _cfg(
    swa=_SWA_OFF,
    cls_mode="qfl", qfl_beta=2.0,
)

# --- 3. MPDIoU on SWA-OFF: the R5 run that never executed ---
R6_MPDIOU_DEFAULT = _cfg(
    swa=_SWA_OFF,
    box_loss_type="mpdiou",
    cls_mode="qfl", qfl_beta=2.0,
)

# --- 4. QFL on SWA-const06: isolate QFL on the production base ---
R6_QFL_SWA = _cfg(
    swa=_SWA_CONST06,
    cls_mode="qfl", qfl_beta=2.0,
)

# --- 5. Aggressive bag weights (no sqrt) + QFL on SWA-const06 ---
#     Requires class_weight_mode patch in loss file (see PREFLIGHT).
R6_BAG_BOOST_QFL = _cfg(
    swa=_SWA_CONST06,
    cls_mode="qfl", qfl_beta=2.0,
    class_weight_mode="linear",    # no sqrt dampening → bag gets ~2.3x trolley
)

# --- 6. WIoU + QFL on SWA-const06: full combo on production base ---
R6_WIOU_QFL_SWA = _cfg(
    swa=_SWA_CONST06,
    box_loss_type="wiou", **_WIOU,
    cls_mode="qfl", qfl_beta=2.0,
)

# --- 7. Softened SATAL + QFL + CIoU on SWA-const06 ---
R6_SATAL_SOFT_QFL = _cfg(
    swa=_SWA_CONST06,
    cls_mode="qfl", qfl_beta=2.0,
    **_SATAL_R3_SOFT,
)


# Ordered by descending confidence of a meaningful improvement.
RUNS = [
    {"name": "r6_wiou_qfl_default",  "label": "[1/7] WIoU + QFL on SWA-OFF -- best box × best base × untested cls",   "params": R6_WIOU_QFL_DEFAULT,  "seed": 0},
    {"name": "r6_qfl_default",       "label": "[2/7] QFL on SWA-OFF + CIoU -- isolate QFL on best mAP50-95 base",     "params": R6_QFL_DEFAULT,       "seed": 0},
    {"name": "r6_mpdiou_default",    "label": "[3/7] MPDIoU + QFL on SWA-OFF -- tight corners on tight-loc base",      "params": R6_MPDIOU_DEFAULT,    "seed": 0},
    {"name": "r6_qfl_swa",           "label": "[4/7] QFL on SWA-const06 + CIoU -- QFL on the production base",        "params": R6_QFL_SWA,           "seed": 0},
    {"name": "r6_bag_boost_qfl",     "label": "[5/7] Aggressive bag weight (linear) + QFL -- target 15% bag gap",     "params": R6_BAG_BOOST_QFL,     "seed": 0},
    {"name": "r6_wiou_qfl_swa",      "label": "[6/7] WIoU + QFL on SWA-const06 -- precision maximizer combo",         "params": R6_WIOU_QFL_SWA,     "seed": 0},
    {"name": "r6_satal_soft_qfl",    "label": "[7/7] Softened SATAL (1.3) + QFL + SWA -- can QFL fix SATAL's hit?",    "params": R6_SATAL_SOFT_QFL,   "seed": 0},
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
    print(f"  ROUND 6 -- QFL classification + untested combos ({len(RUNS)} runs)")
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
