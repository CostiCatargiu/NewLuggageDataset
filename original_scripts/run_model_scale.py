#!/usr/bin/env python3
"""
NEW-DATASET BASELINE — pure stock loss, EVERYTHING off. Clean slate.

=============================================================================
WHY
=============================================================================
A new dataset version means every historical number is void. All ~90 configs
measured so far (57.43 / 57.63 anchors, SATAL -2.96, SNATAL null, clipping
null, posboost -0.67, QFL ~0) were on LuggageDataset.v5i and CANNOT be quoted
as a reference for the new data. This script re-establishes the anchor.

Run order:
  1. ms_s          yolov12s pure stock  -> THE NEW ANCHOR (1.5 h)
  2. ms_s_seed1/2  same, seeds 1 and 2  -> the new noise floor (3 h)
  3. ms_m / ms_l   capacity probe, only once 1 and 2 exist

Do 1 before anything else. Without it no later delta on this dataset means
anything, and it also tells you immediately whether the new data behaves like
v5i (~57.6) or is a materially different problem.

=============================================================================
WHAT "ALL OFF" MEANS HERE
=============================================================================
Genuinely pure stock Ultralytics: CIoU + DFL + BCE + stock
TaskAlignedAssigner (topk 10, alpha 0.5, beta 6.0), gains 7.5/0.5/1.5.

No SWA, no center loss, no clipping (use_loss_clip=False as well as the 999
caps), no NWD, no DFL-entropy, no SATAL/SNATAL/ARTAL/LBA, no AR-DFL, no PEU,
no QFL, no pos-boost, no freq-weight, no cls-SWA, no bag penalty, no
repulsion, and — unlike the v5i r0/r9/r10 lineage — NO class weighting.
Nothing in _ALL_OFF evaluates to True.

The neutral-config guarantee: this must reproduce plain yolov12 training. If
it does not, stop and fix that before reading any mechanism result.

=============================================================================
WHAT IS HELD FIXED
=============================================================================
70 epochs, 640px, seed 0, close_mosaic 10, SGD auto, patience 100.

  !! BATCH IS NOT HELD FIXED across model sizes — it cannot be. yolov12s at
     58 filled 22.2 GB of the 4090; yolov12l needs ~16. Batch changes the
     effective LR schedule, so s -> l is a CAPACITY comparison, not a
     single-variable ablation. ms_s_b16 exists to remove that confound if the
     l result turns out to matter.

REQUIRES lossv2updated.py installed as ultralytics/utils/loss.py.

Usage:
    python run_model_scale.py                 # everything active in RUNS
    python run_model_scale.py ms_s            # a subset
"""

import sys
import time
import gc
import copy
import json
import os
import hashlib

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION
# =============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! POINT THIS AT THE **NEW** DATASET VERSION BEFORE RUNNING.
# !! Everything below assumes a clean slate. The path currently shown is the
# !! OLD v5i set that produced all ~90 historical configs.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
PROJECT_DIR = "runs_newdata_baseline"

EPOCHS = 70
IMG_SIZE = 640            # eval MUST also be 640 (the 896 lesson)
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100

WITH_TEST = True
# NO REFERENCE YET — this is a NEW dataset version.
# 57.43 / 57.63 and every one of the ~90 historical configs were measured on
# LuggageDataset.v5i. They are NOT comparable to results on the new data and
# must not be quoted as a baseline for it. ms_s below CREATES the new anchor;
# once it lands, set this to that number and everything after is measurable.
BASELINE_TEST_MAP5095 = None

# =============================================================================
# Everything-off loss base — identical to run_assigner_isolated.py
# =============================================================================
_ALL_OFF = dict(
    # SWA off: alpha 0 -> area weight multiplied by 0 -> pure score weighting
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    # center off
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
    # clip off (999 -> effective cap 99.9, never binds)
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    # NWD off, DFL-entropy off
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    # assigners off -> stock TaskAlignedAssigner
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    # cls: pure stock BCE — class weighting OFF too (see note below)
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    # every other optional cls/box mechanism explicitly off
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    # gains at stock
    box=7.5, cls=0.5, dfl=1.5,
)

# NOTE: nothing in _ALL_OFF is True. This is plain yolov12 training.
# On the OLD v5i data the pure-stock yolov12s anchor was 57.63 (class
# weighting OFF) and 57.43 with class weighting ON. Both are v5i numbers and
# are quoted here ONLY as a sanity range - if the new data lands wildly away
# from ~57-58, check the dataset before checking the loss.



# =============================================================================
# RUNS — model and batch live on the run, not in params
# =============================================================================
# BATCH GUIDE for a 24 GB 4090 @640px (yolov12s@58 measured 22.2 GB):
#   yolov12s  58   ~22.2 GB   (measured on v5i)
#   yolov12m  32   start here; drop to 24 if OOM
#   yolov12l  16   start here; drop to 12 if OOM
# If you OOM, halve the batch rather than lowering imgsz — 640 must be held.
# =============================================================================
# SWA-sqrt 0.7->0.3 — the historical v5i "best" (r0a_swa_a07_03_sqrt = 57.86)
# =============================================================================
# Worth one run on the new data: it topped 55 v5i configs, so re-testing it on
# the corrected dataset is the cleanest statement of the selection-bias point.
#
# It was probably never real: expected best-of-55 under the v5i noise
# distribution was 57.84 and it scored 57.86; and the combo study used it as a
# parent three times (sqrt+entropy 57.45, sqrt+NWD 57.04, all three 57.04) —
# it failed to replicate every time.
#
# TWO DELIBERATE DEVIATIONS from the historical run:
#  1. use_class_weighting is OFF here. The v5i run had it ON (v2-lineage
#     default). Keeping it off makes SWA the ONLY variable vs ms_s.
#  2. small_obj_px is an AREA test (area < px^2), NOT a side test. Mean object
#     area was 41x90 = 3690 px^2 on v5i and is 39x55 = 2145 px^2 now, while
#     48^2 = 2304. So the mean object is now BELOW the threshold: px=48 has
#     flipped from "boost the smaller minority" to "boost the majority".
#     The px36 variant restores the original scope
#     (36^2 = 1296 -> 0.60 of mean area, matching v5i's 2304/3690 = 0.62).
_SWA_SQRT_0703 = dict(
    _ALL_OFF,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0, area_weight_mode="sqrt",
)
_SWA_SQRT_0703_PX36 = dict(
    _ALL_OFF,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=36, small_obj_boost=2.0, area_weight_mode="sqrt",
)

RUNS = [
    # >>>>>>>>>>>>>>>>>>>>>>>>>  ACTIVE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # 1. THE ANCHOR — run first. It is the reference ms_s_sqrt0703 is measured
    #    against, and it tells you whether the new data behaves like v5i.
    {"name": "ms_s", "model": "yolov12s.pt", "batch": 54,
     "label": "yolov12s, pure stock, all OFF — NEW-DATA ANCHOR",
     "params": dict(_ALL_OFF)},

    # 2. SWA-sqrt 0.7->0.3, literal v5i replication (px 48).
    #    Same model, batch and seed as ms_s, so SWA is the only variable.
    {"name": "ms_s_sqrt0703", "model": "yolov12s.pt", "batch": 54,
     "label": "yolov12s + SWA sqrt 0.7->0.3 px48 boost2.0 — v5i best replicated",
     "params": dict(_SWA_SQRT_0703)},
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # QUEUED
    #
    # Scope-matched sqrt (px 36): boosts the same FRACTION of instances it did
    # on v5i. Run if px48 is ambiguous — separates "SWA does nothing" from
    # "px48 no longer means what it meant".
    # {"name": "ms_s_sqrt0703_px36", "model": "yolov12s.pt", "batch": 58,
    #  "label": "yolov12s + SWA sqrt 0.7->0.3 px36 boost2.0 — scope-matched",
    #  "params": dict(_SWA_SQRT_0703_PX36)},
    #
    # Seeds for the new anchor — needed before trusting any delta under ~1.0.
    # {"name": "ms_s_seed1", "model": "yolov12s.pt", "batch": 58,
    #  "label": "anchor seed 1", "params": dict(_ALL_OFF, seed=1)},
    # {"name": "ms_s_seed2", "model": "yolov12s.pt", "batch": 58,
    #  "label": "anchor seed 2", "params": dict(_ALL_OFF, seed=2)},
    #
    # Capacity probe. Batch MUST drop on a 24 GB card, so s -> l carries a
    # batch confound; ms_s_b20 removes it if the result matters.
    # {"name": "ms_l", "model": "yolov12l.pt", "batch": 20,
    #  "label": "yolov12l, pure stock, all OFF — capacity probe",
    #  "params": dict(_ALL_OFF)},
    # {"name": "ms_m", "model": "yolov12m.pt", "batch": 32,
    #  "label": "yolov12m, pure stock, all OFF — mid capacity",
    #  "params": dict(_ALL_OFF)},
    # {"name": "ms_s_b20", "model": "yolov12s.pt", "batch": 20,
    #  "label": "yolov12s at batch 20 — batch-matched control for ms_l",
    #  "params": dict(_ALL_OFF)},
]


# =============================================================================
def _loss_fingerprint():
    """Record WHICH loss file actually ran."""
    try:
        import ultralytics.utils.loss as L
        p = L.__file__
        return {"path": p,
                "md5": hashlib.md5(open(p, "rb").read()).hexdigest()[:12],
                "has_snatal": hasattr(L, "SupplyNormalizedTaskAlignedAssigner")}
    except Exception as e:
        return {"error": str(e)}


def on_train_epoch_start(trainer):
    """Push the epoch into the custom loss (inert here — all schedules off —
    but keeps the loss state consistent and matches the other runners)."""
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            try:
                crit.epoch = epoch
                if hasattr(crit, "_sync_bbox_loss_state"):
                    crit._sync_bbox_loss_state()
            except Exception:
                pass


def run_one(rc):
    name, params = rc["name"], rc["params"]
    model_w, batch = rc["model"], rc["batch"]
    print(f"\n{'=' * 76}\n  RUN {name}\n  {rc['label']}\n"
          f"  model={model_w}  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(model_w)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    meta = {"name": name, "label": rc["label"], "params": params,
            "model": model_w, "batch": batch, "epochs": EPOCHS,
            "imgsz": IMG_SIZE, "seed": SEED, "close_mosaic": CLOSE_MOSAIC,
            "hours": round(hours, 3), "loss_file": _loss_fingerprint()}
    try:
        with open(os.path.join(save_dir, "scale_params.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "model": model_w, "batch": batch, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    if WITH_TEST:
        try:
            tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
                device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                name=f"{name}_test")
            out["test_map50"] = float(tm.box.map50)
            out["test_map5095"] = float(tm.box.map)
            # tm.box.maps is PER-CLASS, not per-size-bucket. Ultralytics .val()
            # cannot produce small/medium/large — run
            # CocoEvalAllFolders_luggage.py on best.pt for the size buckets,
            # which is where every real effect in this project has lived.
            if hasattr(tm.box, "maps") and tm.box.maps is not None:
                out["test_ap_per_class"] = [float(v) for v in tm.box.maps]
        except Exception as e:
            print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    key, key50 = "test_map5095", "test_map50"
    print(f"\n{'=' * 84}\n  RESULTS ({'test' if WITH_TEST else 'val'} split)\n{'=' * 84}")
    print(f"{'run':<12}{'model':<14}{'batch':>6}{'mAP50':>9}{'mAP50-95':>11}"
          f"{'vs ref':>10}{'h':>6}")
    print("-" * 84)
    for r in sorted(res, key=lambda x: -(x[key] if x[key] == x[key] else -9)):
        d = ("%+10.2f" % ((r[key] - BASELINE_TEST_MAP5095) * 100)
             if BASELINE_TEST_MAP5095 else "%10s" % "—")
        print(f"{r['name']:<12}{r['model']:<14}{r['batch']:>6}"
              f"{r[key50] * 100:>9.2f}{r[key] * 100:>11.2f}{d}{r['hours']:>6.1f}")
    print()
    if BASELINE_TEST_MAP5095 is None:
        print("  No reference set — this IS the new-data anchor.")
        print("  Set BASELINE_TEST_MAP5095 to the ms_s test mAP50-95 above,")
        print("  then every later run on this dataset becomes measurable.")
        print("  Do NOT compare against 57.43 / 57.63 — those are v5i numbers.")
    else:
        print("  On v5i the config-population sd was 0.29. Re-derive it on the")
        print("  new data with ms_s_seed1/seed2 before trusting any small delta.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]

    print(f"\n{'=' * 84}")
    print(f"  MODEL-SCALE BASELINE  @{IMG_SIZE}px, {EPOCHS} epochs, stock loss (all phases OFF)")
    print(f"  loss file: {_loss_fingerprint()}")
    _ref = ("NONE - this run creates it" if BASELINE_TEST_MAP5095 is None
            else "%.2f" % (BASELINE_TEST_MAP5095 * 100))
    print(f"  data:      {DATA_YAML}")
    print(f"  reference: {_ref}")
    print(f"{'=' * 84}")
    for r in todo:
        print(f"  {r['name']:<10} {r['model']:<14} batch {r['batch']:<4} {r['label']}")
    print(f"{'=' * 84}\n")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "scale_summary.json")

    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "model": r["model"], "batch": r["batch"],
                        "hours": float("nan"), "val_map50": float("nan"),
                        "val_map5095": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out, "w") as f:      # incremental dump — survives a crash
            json.dump(res, f, indent=2)

    summarise(res, out)