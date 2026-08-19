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

# ---------------------------------------------------------------------------
# ALPHA DOSE-RESPONSE — the seed-free test of whether +0.86 is a mechanism
# ---------------------------------------------------------------------------
# ms_s_sqrt0703 (alpha 0.7->0.3) scored 55.64 vs the 54.77 anchor = +0.86.
# Single seed, so it could be a draw. Rather than spend 3 h on seeds, run the
# alpha schedule at two more doses: a REAL mechanism produces structure across
# alpha, noise produces a scatter. This is exactly how the v5i clipping phase
# was exposed — four doses, non-monotone ordering, spread = 1 sd = noise.
#   0.5->0.25  (mild)   0.7->0.3  (done, +0.86)   0.9->0.4  (strong)
# Everything else identical: px48, boost 2.0, sqrt, batch 54, seed 0.
_SWA_SQRT_A09_04 = dict(
    _ALL_OFF,
    alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
    small_obj_px=48, small_obj_boost=2.0, area_weight_mode="sqrt",
)
_SWA_SQRT_A05_025 = dict(
    _ALL_OFF,
    alpha_start=0.5, alpha_end=0.25, alpha_min=0.25, alpha_max=0.5,
    small_obj_px=48, small_obj_boost=2.0, area_weight_mode="sqrt",
)

# ---------------------------------------------------------------------------
# BOX-GAIN CONTROL — is +0.86 the aiming, or just more box loss?
# ---------------------------------------------------------------------------
# lossv2updated blends  weight = alpha*area_weight + (1-alpha)*score_weight
# but still divides by target_scores_sum. area_weight is max-normalised (<=1)
# while score_weight is typically well below that, so raising alpha raises the
# EFFECTIVE box gain. This is the same confound v3_dflgain17 was built to
# expose for PEU, and posboost died of it.
# TO SET THE GAIN: read epoch-1 train/box_loss from
#   runs_newdata_baseline/ms_s/results.csv        (stock)
#   runs_newdata_baseline/ms_s_sqrt0703/results.csv
# and set box = 7.5 * (sqrt_box_loss / stock_box_loss). 9.0 below is a
# placeholder for a ~1.2x ratio — REPLACE IT with the measured value.
_BOX_GAIN_CTRL = dict(_ALL_OFF, box=9.0)

# ---------------------------------------------------------------------------
# AR-DFL — re-derived for the CORRECTED aspect ratio
# ---------------------------------------------------------------------------
# The v5i weights (w=1.458, h=0.542) came from h/w = 2.69, which we now know
# was a rendering artefact of squashing 640x360 into 512x512. The derivation
# was sound; the constant was wrong. True AR is 1.55, so:
#     w = 2*AR/(1+AR) = 1.216      h = 2/(1+AR) = 0.784
# On v5i v3_ardfl_w scored -0.22 (inside noise) — never actually refuted,
# just run with the wrong numbers. This is the arm most directly invalidated
# by the distortion discovery.
_ARDFL_CORRECTED = dict(
    _ALL_OFF, use_ardfl=True, ardfl_mode="fixed",
    ardfl_w_weight=1.216, ardfl_h_weight=0.784,
)

# ---------------------------------------------------------------------------
# NWD — rescaled for the smaller objects
# ---------------------------------------------------------------------------
# NWD comes from AI-TOD and is built for TINY objects. v6i is 60% small vs
# v5i's 40%, and mean area fell 42%, so the mechanism is much closer to its
# design regime than it was. But C must move with object scale: W2 is a
# distance in stride-normalised coords, and the mean box diagonal at stride 8
# went 12.36 -> 8.43, a factor 0.682. C tuned at 4.0 on v5i therefore maps to
#     4.0 * 0.682 = 2.73
# NOTE: nwd_C=64 saturates NWD to inert on stride-normalised boxes — that bug
# is what made r10_nwd_fixedc (57.75) look like the v5i "best mod" when the
# mechanism was ~93% switched off. Do not use 64.
_NWD_C273 = dict(_ALL_OFF, use_nwd=True, nwd_mode="blend",
                 nwd_weight=0.3, nwd_C=2.73)
_NWD_C4 = dict(_ALL_OFF, use_nwd=True, nwd_mode="blend",
               nwd_weight=0.3, nwd_C=4.0)

# ---------------------------------------------------------------------------
# DFL GAIN — the direction never tested on either dataset
# ---------------------------------------------------------------------------
# v3_dflgain17 tested 1.5 -> 1.7 and lost 0.72. Down was never tried, and on
# v5i the curves said down was right: train DFL fell 19.6% while val DFL was
# flat, with val/train crossing 1.0 at epoch 10. Re-check that divergence on
# v6i from ms_s/results.csv before reading these.
_DFL_12 = dict(_ALL_OFF, dfl=1.2)
_DFL_10 = dict(_ALL_OFF, dfl=1.0)

_B = 54          # batch used by ms_s and ms_s_sqrt0703 — keep it fixed


def _swa(a0, a1, boost, px=48, mode="sqrt"):
    """SWA config built on _ALL_OFF — EVERYTHING else stays off.

    Guarantees the only keys that ever differ from the anchor are the six SWA
    ones. alpha anneals a0 -> a1 and is clamped to [a1, a0]; area_weight_mode
    reshapes the area weight BEFORE small_obj_boost multiplies it.

    Reminder: small_obj_px is an AREA test (area < px^2), not a side test.
    v6i mean object area is 2145 px^2 and 48^2 = 2304, so px=48 boosts the
    MAJORITY of instances. px=36 (1296) restores v5i's boosted fraction.
    """
    return dict(
        _ALL_OFF,
        alpha_start=a0, alpha_end=a1, alpha_min=a1, alpha_max=a0,
        small_obj_px=px, small_obj_boost=boost, area_weight_mode=mode,
    )

RUNS = [
    # ===================== DONE (v6i, batch 54, seed 0) ====================
    #   ms_s            54.77   <- THE ANCHOR
    #   ms_s_sqrt0703   55.64   +0.86   (mAP50 +0.95, P +0.57, R +0.70,
    #                                    small +0.65, med +0.54, large +2.26,
    #                                    backpack +1.55, bag +0.68, trolley +0.36)
    #   Every cell positive — NOT the redistribution signature that LBA and
    #   clipping showed on v5i. Worth taking seriously, and worth controlling.
    # {"name": "ms_s", "model": "yolov12s.pt", "batch": _B,
    #  "label": "yolov12s, pure stock, all OFF — NEW-DATA ANCHOR",
    #  "params": dict(_ALL_OFF)},
    # {"name": "ms_s_sqrt0703", "model": "yolov12s.pt", "batch": _B,
    #  "label": "yolov12s + SWA sqrt 0.7->0.3 px48 boost2.0",
    #  "params": dict(_SWA_SQRT_0703)},

    # =====================================================================
    # GROUP 1 — BOOST sweep at alpha 0.7->0.3 (3 runs, ~4.5 h)
    # =====================================================================
    # RUN THIS GROUP FIRST. boost is the MAGNITUDE knob; alpha is the BLEND
    # knob. If boost=1.0 (sqrt shape, no magnitude change) comes back flat,
    # then the sqrt SHAPE contributes nothing and Group 2 is a sweep over an
    # effective box-gain increase, not over a mechanism. You can Ctrl-C after
    # this group and save 9 h.
    #   v5i priors (vs its 57.43 anchor):
    #     boost 1.0  -0.39     boost 1.5  -0.14
    #     boost 2.0  +0.43     boost 2.5  -0.25
    #   i.e. on v5i the shape alone LOST; only boost 2.0 won, and it was the
    #   config selected as best-of-55. v6i boost 2.0 = +0.86.
    {"name": "ms_s_sqrt_a0703_b10", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.7->0.3 px48 BOOST 1.0 — shape only, THE decisive control",
     "params": _swa(0.7, 0.3, 1.0)},
    {"name": "ms_s_sqrt_a0703_b15", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.7->0.3 px48 BOOST 1.5",
     "params": _swa(0.7, 0.3, 1.5)},
    {"name": "ms_s_sqrt_a0703_b25", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.7->0.3 px48 BOOST 2.5 — over-dose",
     "params": _swa(0.7, 0.3, 2.5)},

    # =====================================================================
    # GROUP 2 — ALPHA sweep at boost 2.0, sqrt, px48 (5 runs, ~7.5 h)
    # =====================================================================
    # Fills the schedule curve around the 0.7->0.3 point that scored +0.86.
    # A real mechanism degrades SMOOTHLY away from its optimum. A lone spike
    # with neighbours at or below the anchor is the best-of-N signature — which
    # is exactly what v5i showed (only 0.7->0.3 beat baseline; every other
    # schedule was -0.03 to -0.56).
    #   v5i priors:  0.5->0.25 -0.56 | 0.6->0.3 -0.26 | 0.7->0.3 +0.43
    #                0.7->0.4  -0.35 | 0.8->0.4 -0.03 | 0.9->0.4 -0.16
    #                0.9->0.3  NEVER RUN on either dataset
    {"name": "ms_s_sqrt_a09_04", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.9->0.4 px48 boost2.0", "params": _swa(0.9, 0.4, 2.0)},
    {"name": "ms_s_sqrt_a08_04", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.8->0.4 px48 boost2.0", "params": _swa(0.8, 0.4, 2.0)},
    {"name": "ms_s_sqrt_a06_03", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.6->0.3 px48 boost2.0", "params": _swa(0.6, 0.3, 2.0)},
    {"name": "ms_s_sqrt_a07_04", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.7->0.4 px48 boost2.0 — shallower decay",
     "params": _swa(0.7, 0.4, 2.0)},
    {"name": "ms_s_sqrt_a09_03", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.9->0.3 px48 boost2.0 — widest decay, NEVER RUN",
     "params": _swa(0.9, 0.3, 2.0)},

    # =====================================================================
    # GROUP 3 — SCOPE (1 run, ~1.5 h)
    # =====================================================================
    # px is an AREA test. v6i mean area 2145 px^2 vs 48^2 = 2304, so px48
    # boosts the majority. px36 (1296) restores the fraction v5i boosted.
    # Separates "SWA works" from "px48 became a global reweighting".
    {"name": "ms_s_sqrt_a0703_px36", "model": "yolov12s.pt", "batch": _B,
     "label": "SWA sqrt 0.7->0.3 boost2.0 PX 36 — scope-matched to v5i",
     "params": _swa(0.7, 0.3, 2.0, px=36)},

    # ---------------- ROUND 2: what is the +0.86 actually made of? --------
    # Run these once the dose-response reads out. SET box= FROM THE MEASURED
    # box_loss RATIO before running the control (see _BOX_GAIN_CTRL above).
    # {"name": "ms_s_boxgain", "model": "yolov12s.pt", "batch": _B,
    #  "label": "stock with box gain raised — magnitude control for sqrt0703",
    #  "params": dict(_BOX_GAIN_CTRL)},
    # {"name": "ms_s_sqrt0703_px36", "model": "yolov12s.pt", "batch": _B,
    #  "label": "SWA sqrt 0.7->0.3 px36 — scope-matched to v5i's boosted fraction",
    #  "params": dict(_SWA_SQRT_0703_PX36)},

    # ---------------- ROUND 3: mechanisms whose PREMISE changed -----------
    # Only these three. Everything else from v5i stays closed: SATAL
    # (-2.96/-35.18) and the global TAL sweep failed because stock alpha/beta/
    # topk is a sharp optimum, which is a property of TAL not of the data;
    # clipping was a non-monotone null; posboost -0.67 was a redistribution;
    # QFL was neutral and needs gain compensation to even be testable.
    #
    # {"name": "ms_s_ardfl_corr", "model": "yolov12s.pt", "batch": _B,
    #  "label": "AR-DFL w=1.216 h=0.784 — re-derived for the TRUE AR 1.55",
    #  "params": dict(_ARDFL_CORRECTED)},
    # {"name": "ms_s_nwd_c273", "model": "yolov12s.pt", "batch": _B,
    #  "label": "NWD blend w=0.3 C=2.73 — rescaled for the smaller objects",
    #  "params": dict(_NWD_C273)},
    # {"name": "ms_s_dfl12", "model": "yolov12s.pt", "batch": _B,
    #  "label": "dfl gain 1.5 -> 1.2 — the direction never tested",
    #  "params": dict(_DFL_12)},
    # {"name": "ms_s_dfl10", "model": "yolov12s.pt", "batch": _B,
    #  "label": "dfl gain 1.5 -> 1.0", "params": dict(_DFL_10)},
    # {"name": "ms_s_nwd_c4", "model": "yolov12s.pt", "batch": _B,
    #  "label": "NWD blend w=0.3 C=4.0 — literal v5i value, for comparison",
    #  "params": dict(_NWD_C4)},
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
