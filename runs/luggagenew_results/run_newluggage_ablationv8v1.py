#!/usr/bin/env python3
"""
Round 8 — dataset-adapted loss sections (loss_satal_swa_plus_v3.py).

What Rounds 1-7 established (31 models, full test split):
  - SWA is the one repeatable mAP50/recall win (const06: 83.19, R+1.5).
  - WIoU is the one box-loss win (P=0.8337, mAP50-95 held) — but SWA+WIoU was
    NEVER combined cleanly (r5_wiou_swa missing; r7 contaminated by the buggy
    center loss). QFL/VFL do not work on this setup. Repulsion inert at w=0.3.
    Uniform TAL loosening hurts; clips inert; NWD hurt at every C tried.
  - mAP50-95 stuck at ~0.568 across 20 non-SATAL configs. Dataset analysis
    shows train→test shift (AR 1.46→2.58, area −23%) — loss levers may be
    capped by data; Round 8 also measures the ACTUAL noise floor.

Round-8 levers (all new, all in v3, all default-OFF; see loss file header):
  [A2] area_weight_mode sqrt — spread 1/area emphasis over small+medium
  [A2] per-class small-obj boost — bag boosted hardest (smallest class +
       precision bottleneck)
  [B2] center loss FIXED (per-anchor stride, size-normalized, 'crowd' mode)
  [K]  small-object cls boost — attacks the ranking gap (AR50_s 0.96 vs
       R50_s 0.71) without QFL/VFL
  [L]  bag asymmetric penalty — negative-term-only upweight of bag logits on
       backpack/trolley anchors
  [M]  AR-aware TAL — per-GT beta relax for tall/narrow boxes (median test
       AR 2.58); stock behavior below AR 2.0

Runs (ORDERED BY DESCENDING CONFIDENCE OF A GAIN), all on SWA-const06 base:
  1. r8_anchor       — const06, CIoU, everything new OFF. Cross-file anchor:
                       MUST land within ±0.35 of r2_swa_const06 (83.19/56.61)
                       and r3_swa_anchor (82.96/56.44), else v3 has drifted.
  2. r8_swa_wiou     — const06 + WIoU. The missing combination of the two
                       proven winners. Highest expected value in the study.
  3. r8_cls_swa      — const06 + Section K (boost 1.75 @48px).
  4. r8_area_sqrt    — const06 + area_weight_mode='sqrt'.
  5. r8_bag_penalty  — const06 + Section L (w=2.0).
  6. r8_artal        — const06 + Section M (thresh 2.0, scale 2.0, relax 2.0).
  7. r8_boost_bag    — const06 + per-class boost (bag 2.5 / others 1.75).
  8. r8_center_crowd — const06 + WIoU + FIXED center loss, crowd mode.
                       (also the r7_stack post-mortem: if this is fine, the
                       v2 center-loss bug caused the collapse)
  9. r8_anchor_s1    — run 1 with seed 1   ── measure the noise floor:
 10. r8_anchor_s2    — run 1 with seed 2   ── std of 3 identical configs.

PREFLIGHT — MANDATORY before launch:
  [ ] ultralytics loss module points at loss_satal_swa_plus_v3.py
  [ ] whitelist the NEW hyp keys (same mechanism as alpha_start / satal_*):
      area_weight_mode, small_obj_boost_backpack, small_obj_boost_bag,
      small_obj_boost_trolley, center_loss_mode, center_crowd_iou,
      use_cls_swa, cls_swa_boost, use_bag_penalty, bag_penalty_weight,
      bag_class_id, use_artal, artal_ar_thresh, artal_ar_scale,
      artal_beta_relax
  [ ] dataset class order is backpack=0, bag=1, trolley=2 (bag_class_id=1)
  [ ] epoch-0 banner: [A2]/[K]/[L]/[M] lines print intended values

Sanity checks on epoch 1 banner:
  - run 1/9/10: [A2] area_weight_mode inv, [K]/[L]/[M] all False, box ciou
  - run 2: box wiou; run 3: [K] True (boost=1.75 @ 48px)
  - run 4: [A2] area_weight_mode sqrt; run 5: [L] True (w=2.0, cls=1)
  - run 6: [M] True (thresh=2.0, scale=2.0, relax=2.0)
  - run 7: [A2] class boosts (bp/bg/tr): [1.75 2.5 1.75]
  - run 8: box wiou, [B] center_loss_init 0.05 (mode=crowd)
  - ALL runs: [C] deprecated line, use_loss_clip stays False

Reference numbers (70% subset, 70ep, seed 0, test):
  swa_const06 83.19/56.61 | v6_default2 82.54/56.84 (best 50-95)
  r4_wiou 83.19/56.43 P=0.8337 | declared noise ±0.35 (runs 9-10 measure it)

Usage:
  python run_newluggage_ablationv8.py
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
PROJECT_DIR = "runs_luggage_round8"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# Shared blocks (identical to Rounds 2-7)
# =============================================================================
_SWA_CONST06 = dict(   # the Round-2 winner — base for every Round-8 run
    alpha_start=0.6, alpha_end=0.6, alpha_min=0.6, alpha_max=0.8,
    small_obj_px=48, small_obj_boost=1.75,
)
_CENTER_OFF = dict(center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35)
_CENTER_CROWD = dict(center_loss_weight_init=0.05, center_loss_weight_min=0.01,
                     center_loss_decay_epochs=35, center_loss_mode="crowd", center_crowd_iou=0.1)
_CLIP_OFF = dict(iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
_WIOU = dict(box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)

# Toggle pins — EVERY lever pinned so nothing drifts on defaults.
# Each run overrides only the lever under test.
_PINS = dict(
    # pre-v3 pins (match Rounds 4-7)
    use_satal=False,
    use_nwd=False, nwd_mode="small_only", nwd_C=4.0,
    box_loss_type="ciou",
    swa_smooth=False, swa_boost_power=0.5,
    use_loss_clip=False,
    use_class_weighting=True, class_weight_mode="sqrt",   # ON to match Rounds 1-7
    cls_mode="bce", qfl_beta=2.0,
    use_repulsion=False, repulsion_weight=0.3,
    # v3 pins (all OFF/legacy — r8_anchor must reproduce v2 numbers)
    area_weight_mode="inv",
    small_obj_boost_backpack=-1.0, small_obj_boost_bag=-1.0, small_obj_boost_trolley=-1.0,
    center_loss_mode="small", center_crowd_iou=0.1,
    use_cls_swa=False, cls_swa_boost=1.75,
    use_bag_penalty=False, bag_penalty_weight=2.0, bag_class_id=1,
    use_artal=False, artal_ar_thresh=2.0, artal_ar_scale=2.0, artal_beta_relax=2.0,
)


def _cfg(center=_CENTER_OFF, **overrides):
    """pins -> center -> clip off -> SWA-const06 -> stock TAL -> per-run overrides."""
    c = dict(_PINS)
    c.update(center)
    c.update(_CLIP_OFF)
    c.update(_SWA_CONST06)
    c.update(_TAL_STOCK)
    c.update(overrides)
    return c


# =============================================================================
# RUN CONFIGS
# =============================================================================
R8_ANCHOR       = _cfg()
R8_SWA_WIOU     = _cfg(**_WIOU)
R8_CLS_SWA      = _cfg(use_cls_swa=True, cls_swa_boost=1.75)
R8_AREA_SQRT    = _cfg(area_weight_mode="sqrt")
R8_BAG_PENALTY  = _cfg(use_bag_penalty=True, bag_penalty_weight=2.0)
R8_ARTAL        = _cfg(use_artal=True, artal_ar_thresh=2.0, artal_ar_scale=2.0, artal_beta_relax=2.0)
R8_BOOST_BAG    = _cfg(small_obj_boost_bag=2.5)  # backpack/trolley fall back to 1.75
R8_CENTER_CROWD = _cfg(center=_CENTER_CROWD, **_WIOU)

RUNS = [
<<<<<<< HEAD
    # {"name": "r8_anchor",       "label": "[1/10] const06 CIoU, all v3 levers OFF -- cross-file anchor (~83.19/56.61)",  "params": R8_ANCHOR,       "seed": 0},
    # {"name": "r8_swa_wiou",     "label": "[2/10] const06 + WIoU -- the missing combo of both proven winners",           "params": R8_SWA_WIOU,     "seed": 0},
    # {"name": "r8_cls_swa",      "label": "[3/10] const06 + [K] small-obj cls boost 1.75 @48px -- ranking gap",          "params": R8_CLS_SWA,      "seed": 0},
    # {"name": "r8_area_sqrt",    "label": "[4/10] const06 + [A2] sqrt area weight -- spread emphasis small+medium",      "params": R8_AREA_SQRT,    "seed": 0},
    # {"name": "r8_bag_penalty",  "label": "[5/10] const06 + [L] bag negative-term penalty w=2.0 -- bag precision",       "params": R8_BAG_PENALTY,  "seed": 0},
    # {"name": "r8_artal",        "label": "[6/10] const06 + [M] AR-aware TAL (2.0/2.0/2.0) -- tall-narrow assignment",   "params": R8_ARTAL,        "seed": 0},
    # {"name": "r8_boost_bag",    "label": "[7/10] const06 + [A2] per-class boost bag 2.5 / others 1.75",                 "params": R8_BOOST_BAG,    "seed": 0},
    {"name": "r8_center_crowd", "label": "[8/10] const06 + WIoU + [B2] FIXED center loss (crowd) -- r7 post-mortem",    "params": R8_CENTER_CROWD, "seed": 0},
    # {"name": "r8_anchor_s1",    "label": "[9/10] anchor, seed 1 -- noise floor",                                        "params": R8_ANCHOR,       "seed": 1},
    # {"name": "r8_anchor_s2",    "label": "[10/10] anchor, seed 2 -- noise floor",                                       "params": R8_ANCHOR,       "seed": 2},
=======
    {"name": "r8_anchor",       "label": "[1/10] const06 CIoU, all v3 levers OFF -- cross-file anchor (~83.19/56.61)",  "params": R8_ANCHOR,       "seed": 0},
    {"name": "r8_swa_wiou",     "label": "[2/10] const06 + WIoU -- the missing combo of both proven winners",           "params": R8_SWA_WIOU,     "seed": 0},
    {"name": "r8_cls_swa",      "label": "[3/10] const06 + [K] small-obj cls boost 1.75 @48px -- ranking gap",          "params": R8_CLS_SWA,      "seed": 0},
    {"name": "r8_area_sqrt",    "label": "[4/10] const06 + [A2] sqrt area weight -- spread emphasis small+medium",      "params": R8_AREA_SQRT,    "seed": 0},
    {"name": "r8_bag_penalty",  "label": "[5/10] const06 + [L] bag negative-term penalty w=2.0 -- bag precision",       "params": R8_BAG_PENALTY,  "seed": 0},
    {"name": "r8_artal",        "label": "[6/10] const06 + [M] AR-aware TAL (2.0/2.0/2.0) -- tall-narrow assignment",   "params": R8_ARTAL,        "seed": 0},
    {"name": "r8_boost_bag",    "label": "[7/10] const06 + [A2] per-class boost bag 2.5 / others 1.75",                 "params": R8_BOOST_BAG,    "seed": 0},
    {"name": "r8_center_crowd", "label": "[8/10] const06 + WIoU + [B2] FIXED center loss (crowd) -- r7 post-mortem",    "params": R8_CENTER_CROWD, "seed": 0},
    {"name": "r8_anchor_s1",    "label": "[9/10] anchor, seed 1 -- noise floor",                                        "params": R8_ANCHOR,       "seed": 1},
    {"name": "r8_anchor_s2",    "label": "[10/10] anchor, seed 2 -- noise floor",                                       "params": R8_ANCHOR,       "seed": 2},
>>>>>>> 65867c5c1c381541bfad2d0cc2c95a67f576af6a
]


def on_train_epoch_start(trainer):
    """Sync epoch into the custom loss (drives alpha / center schedules)."""
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
    print(f"  ROUND 8 — dataset-adapted sections (v3 loss) — {len(RUNS)} RUNS")
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

    # noise floor from the three anchor seeds
    anchors = [r for r in summary if r["name"].startswith("r8_anchor")]
    vals = [r["test_map50"] for r in anchors if r["test_map50"] == r["test_map50"]]
    if len(vals) >= 2:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        print(f"\n  Noise floor (anchor seeds, n={len(vals)}): "
              f"mAP50 mean={mean * 100:.2f}%  std={std * 100:.2f} pts")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 65867c5c1c381541bfad2d0cc2c95a67f576af6a
