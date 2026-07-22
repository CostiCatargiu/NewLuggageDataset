#!/usr/bin/env python3
"""
Round 5 — evidence-driven follow-up to Round 4 (uses loss_satal_swa_plus_v2.py).

What Round 4 established (full test split):
  - WIoU set a NEW study-best precision 0.8337 while holding mAP50-95 0.5643
    -> the one genuine box-loss win. Chase it.
  - The overall mAP50-95 ceiling (0.5684, v6_default2 = SWA-OFF default) was NOT
    beaten by any box loss on the SWA-const06 base. Note: SWA-OFF is the best
    mAP50-95 recipe (SWA trades tight-loc for recall), so the right base to try
    to BEAT 0.5684 is SWA-OFF, not SWA-const06.
  - SATAL+tight-box did NOT recover mAP50-95 (stayed ~0.54). The penalty scales
    with how much SATAL loosens small-object assignment -> loosen LESS
    (topk_factor 1.3 instead of 1.5) to keep its recall/mAP50 with a smaller hit.
  - NWD small_only with C=4.0 HURT small objects (0.5015 vs CIoU 0.5096).
    C=4 was too high -> bracket it (C=2 and C=6) off the debug printout.

Round-5 runs (ORDERED BY DESCENDING CONFIDENCE OF A GAIN):
  1. r5_wiou_default    — WIoU on SWA-OFF default. Best shot to BEAT mAP50-95
                          0.5684: proven box loss on the proven mAP50-95 base.
  2. r5_mpdiou_default  — MPDIoU on SWA-OFF default. Same idea, tight-loc variant.
  3. r5_wiou_swa        — WIoU on SWA-const06. Confirms the R4 precision record
                          (reproducibility) and yields the high-precision model.
  4. r5_satal13_mpdiou  — SATAL (topk_factor 1.3, softened) + MPDIoU. Keep the
                          recall/mAP50 upside with a smaller mAP50-95 penalty.
  5. r5_nwd_c2          — NWD small_only C=2.0 on CIoU/SWA-const06 (retune down).
  6. r5_nwd_c6          — NWD small_only C=6.0 (bracket the other side).
  7. r5_anchor_default  — SWA-OFF, CIoU, stock TAL. Cross-file reproducibility
                          anchor; MUST land within +/-0.35 of v6_default2
                          (82.54 / 56.84) or the loss file has drifted.

PREFLIGHT — MANDATORY before launch:
  [ ] ultralytics loss module points at loss_satal_swa_plus_v2.py
  [ ] new hyp keys whitelisted (box_loss_type, wiou_*, use_nwd, nwd_mode, nwd_C,
      use_loss_clip, use_class_weighting, cls_mode, swa_*) — as in Round 4
  [ ] ultralytics/utils/satal.py importable (run 4)
  [ ] epoch-0 banner prints intended [I] box_loss_type and [A] alpha values

Sanity checks on epoch 1 banner:
  - runs 1,2,7: [Alpha] flat 0.000 (SWA off), use_satal False
  - run 3: [Alpha] flat 0.600, box wiou
  - run 4: use_satal True, satal_topk_factor 1.5->(internally topk*1.3), box mpdiou
  - runs 5,6: use_nwd True, nwd_mode small_only, nwd_C 2.0 / 6.0

Reference numbers (70% subset, 70ep, seed 0, test):
  v6_default2 82.54/56.84 (best mAP50-95) | swa_const06 83.19/56.61
  r4_wiou P=83.37 (best precision) | noise floor +/-0.35

Usage:
  python run_newluggage_ablationv5.py
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
PROJECT_DIR = "runs_luggage_round5"

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
_SATAL_R3_LOOSE13 = dict(
    use_satal=True,
    tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_alpha_small=1.2, satal_beta_small=5.0,
    satal_topk_factor=1.3,               # softened from 1.5
    satal_small_area=0.0025, satal_large_area=0.0225,
)

_WIOU = dict(wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)

# Toggle pins so nothing drifts on defaults; each run overrides only its lever.
_PINS = dict(
    use_satal=False,
    use_nwd=False, nwd_mode="small_only", nwd_C=4.0,
    box_loss_type="ciou",
    swa_smooth=False, swa_boost_power=0.5,
    use_loss_clip=False,
    use_class_weighting=True,            # keep ON to match Rounds 1-4
    cls_mode="bce",
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
R5_WIOU_DEFAULT   = _cfg(swa=_SWA_OFF,      box_loss_type="wiou", **_WIOU)
R5_MPDIOU_DEFAULT = _cfg(swa=_SWA_OFF,      box_loss_type="mpdiou")
R5_WIOU_SWA       = _cfg(swa=_SWA_CONST06,  box_loss_type="wiou", **_WIOU)
R5_SATAL13_MPDIOU = _cfg(swa=_SWA_CONST06,  box_loss_type="mpdiou", **_SATAL_R3_LOOSE13)
R5_NWD_C2         = _cfg(swa=_SWA_CONST06,  box_loss_type="ciou", use_nwd=True, nwd_mode="small_only", nwd_C=2.0)
R5_NWD_C6         = _cfg(swa=_SWA_CONST06,  box_loss_type="ciou", use_nwd=True, nwd_mode="small_only", nwd_C=6.0)
R5_ANCHOR_DEFAULT = _cfg(swa=_SWA_OFF,      box_loss_type="ciou")

RUNS = [
    {"name": "r5_wiou_default",   "label": "[1/7] WIoU on SWA-OFF default -- best shot to beat mAP50-95 0.5684",   "params": R5_WIOU_DEFAULT,   "seed": 0},
    {"name": "r5_mpdiou_default", "label": "[2/7] MPDIoU on SWA-OFF default -- tight-loc on the mAP50-95 base",    "params": R5_MPDIOU_DEFAULT, "seed": 0},
    {"name": "r5_wiou_swa",       "label": "[3/7] WIoU on SWA-const06 -- confirm R4 precision record (0.8337)",    "params": R5_WIOU_SWA,       "seed": 0},
    {"name": "r5_satal13_mpdiou", "label": "[4/7] SATAL topk 1.3 (softened) + MPDIoU -- recall with smaller hit",  "params": R5_SATAL13_MPDIOU, "seed": 0},
    {"name": "r5_nwd_c2",         "label": "[5/7] NWD small_only C=2.0 -- retune down (C=4 hurt small objects)",   "params": R5_NWD_C2,         "seed": 0},
    {"name": "r5_nwd_c6",         "label": "[6/7] NWD small_only C=6.0 -- bracket the other side",                 "params": R5_NWD_C6,         "seed": 0},
    {"name": "r5_anchor_default", "label": "[7/7] SWA-OFF CIoU stock -- reproducibility anchor (~82.54/56.84)",    "params": R5_ANCHOR_DEFAULT, "seed": 0},
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

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "label": label, "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "test_map50": test_map50,
            "test_map5095": test_map5095}


def main():
    print(f"\n{'=' * 70}")
    print(f"  ROUND 5 -- evidence-driven follow-up ({len(RUNS)} runs)")
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
            if v != v:
                return "n/a"
            return f"{v * 100:.2f}%" if pct else f"{v:.2f}"
        print(f"  {r['name']:<22}{fmt(r['elapsed_h'], pct=False):>9}"
              f"{fmt(r['val_map50']):>11}{fmt(r['test_map50']):>12}"
              f"{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()