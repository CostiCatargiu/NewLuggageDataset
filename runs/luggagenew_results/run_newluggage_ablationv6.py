#!/usr/bin/env python3
"""
Round 7 — NEW loss term + first stacking of winners (loss_satal_swa_plus_v2.py).

Rounds 1-6 established (28 models): the mAP50-95 ceiling (~0.568) is flat across
20 non-SATAL configs, and every prior round changed ONE lever at a time. Two
things were never done:
  (a) a loss that models CLASS CONFUSION — the bag bottleneck is 74% precision /
      94% recall = confident cross-class false positives, and every loss so far
      is class-agnostic on geometry.
  (b) STACKING the individual winners together (WIoU precision + px32 small-obj
      boost + the dormant Section-B center loss).

NEW loss update (Section J): class-confusion repulsion. For each foreground
predicted box, penalize its IoU with the highest-overlap DIFFERENT-class GT box
(RepGT-style). Minimizing it pushes a "bag" prediction off a backpack/trolley
object. Pure loss — no architecture, no resolution change. Verified: penalizes
cross-class overlap only, ignores same-class (never hurts real detections).

Also turns ON, for the first time in the whole study:
  - Section B center loss (aux L1 on small-object centers) — coded but off in all
    42 prior configs; targets bag's 20.6%-small distribution.
  - px32 small-object boost (from r2_swa_px32, best small-obj mAP50-95 0.5119)
    stacked with WIoU (r4_wiou, best precision 0.8337) — never combined.

Runs — ORDERED BY DESCENDING CONFIDENCE OF A GAIN (4 runs, 50 epochs):
  1. r7_stack         — WIoU + px32 boost + center loss + repulsion. The full
                        stack-the-winners recipe; highest ceiling for a new best.
  2. r7_wiou_rep      — WIoU + repulsion only. Isolates the NEW term vs r4_wiou
                        (same recipe minus repulsion) — the clean causal test.
  3. r7_stack_bag     — stack + aggressive bag weighting (class_weight_mode
                        linear). Maximal pressure on the bag precision problem.
  4. r7_wiou_center   — WIoU + px32 boost + center loss (no repulsion). Isolates
                        the dormant center-loss contribution.

Compare against: r4_wiou (0.8319/0.5643, P0.8337) and r2_swa_px32 (P0.8206,
small 0.5119). Baseline reference: swa_const06 83.19/56.60.

PREFLIGHT:
  [ ] ultralytics loss = loss_satal_swa_plus_v2.py
  [ ] whitelist the TWO new keys: use_repulsion, repulsion_weight
      (center_loss_*, class_weight_mode, box_loss_type, wiou_* already used)
  [ ] epoch-0 banner: [J] repulsion: True (w=0.3) for runs 1-3; [B] center_loss
      init 0.05 for runs 1,3,4

Speed: EPOCHS=50 (val curve plateaus ~ep40), cache=True. Bump to 70 for the
final chosen config only.

Usage:
  python run_newluggage_ablationv7.py
"""

import time
import gc
import copy
import json
import os
import torch
from ultralytics import YOLO

DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v4i.yolov12_70percentage/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_luggage_round7"

EPOCHS = 70          # plateau hits ~ep40; 50 for sweeps, bump to 70 for final
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared blocks
# =============================================================================
_SWA_CONST06 = dict(         # Round-2 winner (px48)
    alpha_start=0.6, alpha_end=0.6, alpha_min=0.6, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=1.75,
)
_SWA_CONST06_PX32 = dict(    # const06 base + the px32 small-obj boost (r2_swa_px32)
    alpha_start=0.6, alpha_end=0.6, alpha_min=0.6, alpha_max=0.8,
    small_obj_px=32, small_obj_boost=2.0,
)
_CENTER_OFF = dict(center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35)
_CENTER_ON  = dict(center_loss_weight_init=0.05, center_loss_weight_min=0.01, center_loss_decay_epochs=35)
_CLIP_OFF = dict(iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
_WIOU = dict(box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)
_REP = dict(use_repulsion=True, repulsion_weight=0.3)

_PINS = dict(
    use_satal=False,
    use_nwd=False, nwd_mode="small_only", nwd_C=4.0,
    box_loss_type="ciou",
    swa_smooth=False, swa_boost_power=0.5,
    use_loss_clip=False,
    use_class_weighting=True, class_weight_mode="sqrt",
    cls_mode="bce", qfl_beta=2.0,
    use_repulsion=False, repulsion_weight=0.3,
)


def _cfg(swa=_SWA_CONST06, center=_CENTER_OFF, **overrides):
    c = dict(_PINS)
    c.update(center)
    c.update(_CLIP_OFF)
    c.update(swa)
    c.update(_TAL_STOCK)
    c.update(overrides)
    return c


# =============================================================================
# RUN CONFIGS
# =============================================================================
R7_STACK       = _cfg(swa=_SWA_CONST06     , center=_CENTER_ON, **_WIOU, **_REP)
R7_WIOU_REP    = _cfg(swa=_SWA_CONST06,      center=_CENTER_OFF, **_WIOU, **_REP)
R7_STACK_BAG   = _cfg(swa=_SWA_CONST06     , center=_CENTER_ON, **_WIOU, **_REP, class_weight_mode="linear")
R7_WIOU_CENTER = _cfg(swa=_SWA_CONST06     , center=_CENTER_ON, **_WIOU)

RUNS = [
    {"name": "r7_stack",       "label": "[1/4] WIoU + center loss + repulsion on const06 -- stack the winners", "params": R7_STACK,       "seed": 0},
    {"name": "r7_wiou_rep",    "label": "[2/4] WIoU + repulsion only -- isolate the NEW term vs r4_wiou",         "params": R7_WIOU_REP,    "seed": 0},
    {"name": "r7_stack_bag",   "label": "[3/4] stack + aggressive bag weighting (linear) -- max bag pressure",    "params": R7_STACK_BAG,   "seed": 0},
    {"name": "r7_wiou_center", "label": "[4/4] WIoU + center loss (const06) -- isolate center-loss effect",    "params": R7_WIOU_CENTER, "seed": 0},
]


def on_train_epoch_start(trainer):
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
    name = run_cfg["name"]; label = run_cfg["label"]; params = run_cfg["params"]; seed = run_cfg.get("seed", 0)
    print(f"\n{'=' * 70}\n  RUN: {name}  (seed {seed})\n  {label}\n{'=' * 70}\n")
    start_time = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    train_kwargs = {
        "data": DATA_YAML, "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
        "device": DEVICE, "workers": WORKERS, "project": PROJECT_DIR, "name": name,
        "patience": 100, "close_mosaic": 10, "seed": seed, "deterministic": True,
        "cache": True,
    }
    train_kwargs.update(copy.deepcopy(params))
    results = model.train(**train_kwargs)
    elapsed = (time.time() - start_time) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": label, "params": params, "epochs": EPOCHS,
                       "imgsz": IMG_SIZE, "batch": BATCH, "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    val_map50 = float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd: val_map50 = float(rd[key]); break
    except Exception:
        pass

    test_map50, test_map5095 = float("nan"), float("nan")
    try:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        test_model = YOLO(best_pt)
        tm = test_model.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        test_map50 = float(tm.box.map50); test_map5095 = float(tm.box.map)
        del test_model, tm
    except Exception as e:
        print(f"  [WARN] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"name": name, "label": label, "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "test_map50": test_map50, "test_map5095": test_map5095}


def main():
    print(f"\n{'=' * 70}\n  ROUND 7 -- repulsion loss + stack-the-winners ({len(RUNS)} runs)\n"
          f"  Runs (confidence order): {', '.join(r['name'] for r in RUNS)}\n{'=' * 70}")
    overall_start = time.time(); summary = []
    for run_cfg in RUNS:
        try:
            result = run_one(run_cfg)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "label": run_cfg["label"], "seed": run_cfg.get("seed", 0),
                      "elapsed_h": float("nan"), "val_map50": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"), "error": str(e)}
        summary.append(result)
        try:
            os.makedirs(PROJECT_DIR, exist_ok=True)
            with open(os.path.join(PROJECT_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
    total_elapsed = (time.time() - overall_start) / 3600
    print(f"\n{'=' * 70}\n  ALL RUNS COMPLETE ({total_elapsed:.2f}h total)\n{'=' * 70}")
    print(f"  {'Run':<20}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 64}")
    for r in summary:
        def fmt(v, pct=True):
            if v != v: return "n/a"
            return f"{v * 100:.2f}%" if pct else f"{v:.2f}"
        print(f"  {r['name']:<20}{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
              f"{fmt(r['test_map50']):>12}{fmt(r['test_map5095']):>12}")
        if r.get("error"): print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()