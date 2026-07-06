#!/usr/bin/env python3
"""
Full Dataset — 10-run section sweep (3 SWA + 3 CLIP + 3 TAL + 1 DEFAULT).

Each section is swept along its own axis with everything else deactivated,
so per-section curves are directly attributable:

  SWA  (Section A): alpha schedule sweep  0.5→0.25 / 0.7→0.3 / 0.8→0.4
  CLIP (Section C): clip tightness sweep  loose / medium / tight
  TAL  (Section D): assigner sweep        proven / beta-isolation / aggressive probe

Run order: DEFAULT first (denominator), then SWA, CLIP, TAL.

NOTE: the 3 SWA runs require the per-anchor stride fix in
      BboxLoss._compute_weights — do not launch before it is merged.

Usage:
  python run_section_sweep.py
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
PROJECT_DIR = "runs_luggage_section_sweep"

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
# DEFAULT — everything off (control)
# =============================================================================
DEFAULT_ALL_OFF = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# SECTION A — SWA sweep (stock TAL, no clips)
# Monotone alpha axis: if mAP rises through 0.8→0.4, that justifies testing 0.9.
# =============================================================================
SWA_1_MODERATE = dict(
    alpha_start=0.5, alpha_end=0.25, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=1.75,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)
SWA_2_PROVEN = dict(   # matches your 83.12% run's schedule
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)
SWA_3_HIGH = dict(     # alpha_max raised so the 0.8 start isn't clamped
    alpha_start=0.9, alpha_end=0.4, alpha_min=0.3, alpha_max=0.95,
    small_obj_px=48, small_obj_boost=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
)

# =============================================================================
# SECTION C — CLIP sweep (alpha=0, stock TAL)
# Effective per-sample caps = value / 10. Tightness increases 1 → 3;
# CLIP_3 is expected to HURT — it brackets the failure mode.
# =============================================================================
CLIP_1_LOOSE = dict(
    **_SWA_OFF, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=40.0, iou_clip_end=30.0,   # eff. 4.0 → 3.0
    dfl_clip_start=50.0, dfl_clip_end=40.0,   # eff. 5.0 → 4.0
)
CLIP_2_MEDIUM = dict(
    **_SWA_OFF, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=30.0, iou_clip_end=20.0,   # eff. 3.0 → 2.0
    dfl_clip_start=40.0, dfl_clip_end=30.0,   # eff. 4.0 → 3.0
)
CLIP_3_TIGHT = dict(   # near your original defaults — the known-bad regime
    **_SWA_OFF, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=20.0, iou_clip_end=10.0,   # eff. 2.0 → 1.0
    dfl_clip_start=10.0, dfl_clip_end=5.0,    # eff. 1.0 → 0.5
)

# =============================================================================
# SECTION D — TAL sweep (no SWA, no clips)
# =============================================================================
TAL_1_BEST = dict(     # your ablation winner
    **_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0,
)
TAL_2_BETA_ISO = dict( # same topk/beta, stock alpha — isolates alpha vs beta
    **_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=13, tal_alpha=0.5, tal_beta=4.0,
)
TAL_3_AGGRESSIVE = dict(  # overshoot probe — brackets the optimum from above
    **_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=15, tal_alpha=0.7, tal_beta=3.0,
)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER
# =============================================================================
RUNS = [
    {"name": "v6_default",      "label": "DEFAULT — all off, stock TAL (10/0.5/6.0)",                     "params": DEFAULT_ALL_OFF},

    {"name": "v6_swa_moderate", "label": "SWA 0.5→0.25 (min 0.2/max 0.8), boost 1.75 @ 48px",             "params": SWA_1_MODERATE},
    {"name": "v6_swa_proven",   "label": "SWA 0.7→0.3 (min 0.2/max 0.8), boost 2.0 @ 48px",               "params": SWA_2_PROVEN},
    {"name": "v6_swa_high",     "label": "SWA 0.9→0.4 (min 0.3/max 0.95), boost 2.0 @ 48px",              "params": SWA_3_HIGH},

    {"name": "v6_clip_loose",   "label": "CLIP loose — iou 40→30, dfl 50→40 (eff. 4→3 / 5→4)",            "params": CLIP_1_LOOSE},
    {"name": "v6_clip_medium",  "label": "CLIP medium — iou 30→20, dfl 40→30 (eff. 3→2 / 4→3)",           "params": CLIP_2_MEDIUM},
    {"name": "v6_clip_tight",   "label": "CLIP tight — iou 20→10, dfl 10→5 (eff. 2→1 / 1→0.5)",           "params": CLIP_3_TIGHT},

    {"name": "v6_tal_best",     "label": "TAL 13/0.7/4.0 — ablation winner",                              "params": TAL_1_BEST},
    {"name": "v6_tal_beta",     "label": "TAL 13/0.5/4.0 — beta isolation (stock alpha)",                 "params": TAL_2_BETA_ISO},
    {"name": "v6_tal_probe",    "label": "TAL 15/0.7/3.0 — aggressive overshoot probe",                   "params": TAL_3_AGGRESSIVE},
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

    print(f"\n{'=' * 70}")
    print(f"  RUN: {name}")
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
        "seed": 0,
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
                       "seed": 0}, f, indent=2)
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

    # ---- explicit TEST-split evaluation on best.pt (the thesis number) ----
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

    return {"name": name, "label": label, "elapsed_h": elapsed,
            "val_map50": val_map50, "test_map50": test_map50,
            "test_map5095": test_map5095}


def main():
    print(f"\n{'=' * 70}")
    print(f"  SECTION SWEEP — 10 RUNS (1 default + 3 SWA + 3 CLIP + 3 TAL)")
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