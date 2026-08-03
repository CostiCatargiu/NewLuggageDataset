#!/usr/bin/env python3
"""
Round 0 — 4-section phased ablation sweep (12 runs).

Sections match loss.py exactly:
  r0_default — ALL sections off, stock TAL (the anchor for every comparison)
  PHASE A — SWA alpha-schedule sweep: 0.9->0.4, 0.7->0.3, 0.5->0.25
            (all px=48, boost 2.0; min/max set inert per schedule)
  PHASE B — Center loss (auxiliary small-object center term), isolated
  PHASE C — IoU/DFL adaptive clipping dose-response (loose/mid/tight)
            on top of the HIGH-alpha SWA base, where weights spike hardest
  PHASE D — TAL: two separate axes
            D1 candidate quantity (topk=6)
            D2/D3 metric composition (alpha/beta ratio)

Dataset context (luggage v4i, native 512x512, trained at 640 -> boxes x1.25):
  mean box 33x72 @512, mean h/w ~2.7 (tall & skinny)
  <32px max-side: 18.1% | <48px: 40.3% | <64px: ~58%
  small_obj_px is a PIXEL AREA threshold (area < px^2 at 640). With tall
  boxes, area-side T ~= max-side 1.31*T @512, so px=48 flags ~57% of
  instances as small (broad boost). If all Phase-A alphas land close
  together, revisit px scope (24/36) before revisiting alpha.

IMPORTANT — small_obj_px is SHARED by Sections A and B:
  Phase B runs keep small_obj_px=48 but disable the SWA weighting via
  alpha=0 / boost=1.0 (with alpha=0 the area weight is multiplied by zero,
  so px has no Section-A effect). px=0 would silently kill the center loss.

PHASE C rationale (effective per-sample cap = value/10):
  Base = SWA 0.9->0.4 px48 b2.0 (control = r0a_swa_a09_04, no extra run).
  With alpha up to 0.9 and boost 2.0, weighted per-sample IoU loss peaks
  ~2.5-3.0, so:
    loose 30->20 / 35->25 (eff 3.0->2.0 / 3.5->2.5) — outliers only (<~1%)
    mid   20->12 / 25->15 (eff 2.0->1.2 / 2.5->1.5) — clips the tail
    tight 10->6  / 12->8  (eff 1.0->0.6 / 1.2->0.8) — cuts into the boosted dist
  Read the per-epoch clip-rate log: ~0% on 'loose' would void that run.

PHASE D rationale:
  TAL alignment = score^alpha * IoU^beta. Candidate RANKING depends only on
  the ratio beta/alpha; absolute values also shape the soft target scores.
  This dataset's boxes are small/tall/thin -> early IoU is the noisiest
  signal, yet stock ratio 6/0.5 = 12 makes assignment IoU-dominated.
    D1 tal_topk=6, 0.5/6.0 — quantity axis: fewer, better positives
    D2 0.5/4.0, topk 10    — composition: soften IoU emphasis (ratio 8)
    D3 1.0/6.0, topk 10    — composition: raise cls voice (TOOD original, ratio 6)

REQUIRES: the fixed loss.py (pixel-area SWA, pixel-space center loss,
          clip-rate logging).

Usage:
  python run_r0_sweep.py               # run everything not yet completed
  python run_r0_sweep.py r0c_clip_mid  # run only the named run(s)
"""

import sys
import time
import gc
import copy
import json
import os
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:  # very old ultralytics
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_newluggage5_r0"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Shared blocks
# =============================================================================
# SWA fully off, px=0 (safe ONLY when the center loss is also off)
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0,
)
# SWA weighting off but px kept alive at 48 — REQUIRED for Phase B runs,
# because the center loss reads the same small_obj_px threshold.
_SWA_OFF_KEEP_PX48 = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=48, small_obj_boost=1.0,
)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# High-alpha SWA base reused by Phase C (run as r0a_swa_a09_04 = C control)
_SWA_HIGH = dict(alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
                 small_obj_px=48, small_obj_boost=2.0)

# =============================================================================
# DEFAULT — all four sections off / stock
# =============================================================================
R0_DEFAULT = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# PHASE A — SWA alpha schedules (px=48, boost 2.0; min/max inert per schedule)
# =============================================================================
R0A_SWA_A09_04 = dict(**_SWA_HIGH,
                      **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)   # also Phase-C control
R0A_SWA_A07_03 = dict(alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
                      small_obj_px=48, small_obj_boost=2.0,
                      **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
R0A_SWA_A05_025 = dict(alpha_start=0.5, alpha_end=0.25, alpha_min=0.25, alpha_max=0.5,
                       small_obj_px=48, small_obj_boost=2.0,
                       **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# PHASE A2 — SWA follow-up: four DISTINCT mechanisms (not more of the same).
# Success criterion, fixed in advance: AP50-95_small must improve by >= +0.5
# over the default; otherwise Section A closes with a negative result.
# =============================================================================
# A2-1 Targeted scope: boost only the truly tiny (~18% of instances)
R0A2_PX24_TGT = dict(alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
                     small_obj_px=24, small_obj_boost=2.0, dfl_small_boost=1.0,
                     **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-2 High contrast, no decay: max dose on the tiny bin only, held all run
R0A2_PX24_HI = dict(alpha_start=0.7, alpha_end=0.7, alpha_min=0.7, alpha_max=0.7,
                    small_obj_px=24, small_obj_boost=4.0, dfl_small_boost=1.0,
                    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-3 Inverted (rising) schedule: size emphasis during the LATE refinement
# phase (incl. the close_mosaic window) instead of the early chaotic phase
R0A2_RISE = dict(alpha_start=0.2, alpha_end=0.8, alpha_min=0.2, alpha_max=0.8,
                 small_obj_px=36, small_obj_boost=2.0, dfl_small_boost=1.0,
                 **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-4 DFL-only boost: alpha=0 (no area blending), boost ONLY the DFL term for
# small objects -> targets box-edge precision, the diagnosed deficit
# (AP50_small 0.79 vs AP50-95_small 0.52 with AR50_small ~0.96).
# REQUIRES the dfl_small_boost patch in loss.py.
R0A2_DFLBOOST = dict(alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
                     small_obj_px=36, small_obj_boost=1.0, dfl_small_boost=2.5,
                     **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# PHASE B — center loss ISOLATED (SWA weighting off, px kept at 48)
# =============================================================================
R0B_CENTER_W05 = dict(
    **_SWA_OFF_KEEP_PX48, **_CLIP_OFF, **_TAL_STOCK,
    center_loss_weight_init=0.5, center_loss_weight_min=0.01, center_loss_decay_epochs=35,
)
R0B_CENTER_W10 = dict(
    **_SWA_OFF_KEEP_PX48, **_CLIP_OFF, **_TAL_STOCK,
    center_loss_weight_init=1.0, center_loss_weight_min=0.01, center_loss_decay_epochs=35,
)

# =============================================================================
# PHASE C — clipping dose-response on the HIGH-alpha SWA base
#           (control = r0a_swa_a09_04; effective cap = value/10)
# =============================================================================
R0C_CLIP_LOOSE = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=30.0, iou_clip_end=20.0,   # eff. 3.0 -> 2.0
    dfl_clip_start=35.0, dfl_clip_end=25.0,   # eff. 3.5 -> 2.5
)
R0C_CLIP_MID = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=20.0, iou_clip_end=12.0,   # eff. 2.0 -> 1.2
    dfl_clip_start=25.0, dfl_clip_end=15.0,   # eff. 2.5 -> 1.5
)
R0C_CLIP_TIGHT = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=10.0, iou_clip_end=6.0,    # eff. 1.0 -> 0.6
    dfl_clip_start=12.0, dfl_clip_end=8.0,    # eff. 1.2 -> 0.8
)
# ISOLATED C — clipping WITHOUT SWA (compare vs r0_default).
# At alpha=0 the weight is just the TAL score weight (<= ~1), so per-sample
# losses live in ~0.1-1.2: caps must sit INSIDE that range to ever fire
# (the C-on-A cap values would be inert here, as Round 1 showed).
# Distinguishes "clipping helps in general" from "clipping only counteracts
# the spikes Section A introduces".
R0C_CLIP_SOLO = dict(
    **_SWA_OFF, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=8.0, iou_clip_end=5.0,     # eff. 0.8 -> 0.5
    dfl_clip_start=10.0, dfl_clip_end=7.0,    # eff. 1.0 -> 0.7
)

# =============================================================================
# PHASE D — TAL: quantity axis vs composition axis (everything else off)
# =============================================================================
R0D_TAL_TOPK6 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=6, tal_alpha=0.5, tal_beta=6.0)
R0D_TAL_BETA4 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=10, tal_alpha=0.5, tal_beta=4.0)
R0D_TAL_TOOD = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                    tal_topk=10, tal_alpha=1.0, tal_beta=6.0)
# Harden IoU emphasis (ratio 12->16): select positives more by overlap
# quality -> train regression on well-localized anchors. Mirror of beta4;
# together they give both directions of the composition axis.
R0D_TAL_BETA8 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=10, tal_alpha=0.5, tal_beta=8.0)
# Loose direction of the quantity axis: with topk6 and stock 10 this makes a
# 3-point dose-response curve (6/10/13). Geometry note: tiny boxes (~20x45 @640)
# only have ~11 in-box candidates, so topk13 adds positives mostly to
# medium/large objects (lower-aligned ones). Expected at-or-below stock -- run
# to close the axis, not because it is favored.
R0D_TAL_TOPK13 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                      tal_topk=13, tal_alpha=0.7, tal_beta=4.0)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER
# =============================================================================
RUNS = [
    # {"name": "r0_default",      "phase": "-", "label": "All 4 sections OFF / stock TAL — anchor",                          "params": R0_DEFAULT},
    # --- Phase A: SWA alpha schedules ---
    # {"name": "r0a_swa_a09_04",  "phase": "A", "label": "SWA 0.9->0.4, px48, boost 2.0 — strong/long — C control",          "params": R0A_SWA_A09_04},
    # {"name": "r0a_swa_a07_03",  "phase": "A", "label": "SWA 0.7->0.3, px48, boost 2.0 — medium",                           "params": R0A_SWA_A07_03},
    # {"name": "r0a_swa_a05_025", "phase": "A", "label": "SWA 0.5->0.25, px48, boost 2.0 — mild",                            "params": R0A_SWA_A05_025},
    # --- Phase A2: SWA follow-up, four distinct mechanisms ---
    {"name": "r0a2_px24_tgt",   "phase": "A2", "label": "SWA 0.9->0.4, px24 (targeted ~18%), boost 2.0 — scope",            "params": R0A2_PX24_TGT},
    {"name": "r0a2_px24_hi",    "phase": "A2", "label": "SWA const 0.7, px24, boost 4.0 — high contrast, no decay",         "params": R0A2_PX24_HI},
    {"name": "r0a2_rise",       "phase": "A2", "label": "SWA RISING 0.2->0.8, px36, boost 2.0 — late-phase emphasis",       "params": R0A2_RISE},
    {"name": "r0a2_dflboost",   "phase": "A2", "label": "DFL-only boost 2.5 @px36, alpha=0 — box-edge precision target",    "params": R0A2_DFLBOOST},
    # # --- Phase B: center loss, isolated ---
    # {"name": "r0b_center_w05",  "phase": "B", "label": "Center loss w=0.5->0.01 over 35ep, px48 (SWA weighting off)",      "params": R0B_CENTER_W05},
    # {"name": "r0b_center_w10",  "phase": "B", "label": "Center loss w=1.0->0.01 over 35ep, px48 (SWA weighting off)",      "params": R0B_CENTER_W10},
    # # --- Phase C: clipping dose-response on SWA-high base ---
    # {"name": "r0c_clip_loose",  "phase": "C", "label": "SWA-high + clips 30->20/35->25 (eff 3->2 / 3.5->2.5) — outliers",  "params": R0C_CLIP_LOOSE},
    # {"name": "r0c_clip_mid",    "phase": "C", "label": "SWA-high + clips 20->12/25->15 (eff 2->1.2 / 2.5->1.5) — tail",    "params": R0C_CLIP_MID},
    # {"name": "r0c_clip_tight",  "phase": "C", "label": "SWA-high + clips 10->6/12->8 (eff 1->0.6 / 1.2->0.8) — deep",      "params": R0C_CLIP_TIGHT},
    # {"name": "r0c_clip_solo",   "phase": "C", "label": "Clips ONLY 8->5/10->7 (eff 0.8->0.5 / 1->0.7), no SWA — isolated", "params": R0C_CLIP_SOLO},
    # --- Phase D: TAL, two axes ---
    {"name": "r0d_tal_topk13", "phase": "D", "label": "TAL topk=13, 0.7/4.0 \u2014 loose quantity (completes 6/10/13 curve)", "params": R0D_TAL_TOPK13},
    {"name": "r0d_tal_topk6",   "phase": "D", "label": "TAL topk=6, 0.5/6.0 — quantity axis (fewer, better positives)",    "params": R0D_TAL_TOPK6},
    {"name": "r0d_tal_beta4",   "phase": "D", "label": "TAL 10, 0.5/4.0 — soften IoU emphasis (ratio 12->8)",              "params": R0D_TAL_BETA4},
    {"name": "r0d_tal_tood",    "phase": "D", "label": "TAL 10, 1.0/6.0 — TOOD original, raise cls voice (ratio 6)",       "params": R0D_TAL_TOOD},
    {"name": "r0d_tal_beta8",   "phase": "D", "label": "TAL 10, 0.5/8.0 \u2014 harden IoU emphasis (ratio 12->16)",             "params": R0D_TAL_BETA8},
]


# =============================================================================
# Epoch sync — drives alpha / clip / center-decay schedules in the custom loss
# =============================================================================
def on_train_epoch_start(trainer):
    """Push trainer.epoch into the custom loss, DDP-safe.

    The loss reads self._model.current_epoch on the *unwrapped* DetectionModel,
    so set attributes via de_parallel(trainer.model), never on a DDP wrapper.
    criterion.epoch is set directly as a second path (criterion lives on the
    model in current ultralytics; older versions kept it on the trainer).
    """
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


def run_one(run_cfg):
    name = run_cfg["name"]
    label = run_cfg["label"]
    params = run_cfg["params"]
    seed = run_cfg.get("seed", SEED)

    print(f"\n{'=' * 70}")
    print(f"  RUN: {name}  (phase {run_cfg.get('phase', '?')}, seed {seed})")
    print(f"  {label}")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

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
        "exist_ok": False,
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
            json.dump({"name": name, "phase": run_cfg.get("phase"), "label": label,
                       "params": params, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                       "batch": BATCH, "seed": seed}, f, indent=2)
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

    return {"name": name, "phase": run_cfg.get("phase"), "label": label,
            "seed": seed, "elapsed_h": elapsed, "val_map50": val_map50,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
    """A run counts as done if its summary entry exists with a test score."""
    path = os.path.join(PROJECT_DIR, "summary.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for r in json.load(f):
                if r.get("name") == name and r.get("test_map50") == r.get("test_map50"):
                    return True
    except Exception:
        pass
    return False


def load_summary():
    path = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def main():
    only = set(sys.argv[1:])  # optional: run only named runs
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 70}")
    print("  ROUND 0 SWEEP — default + A (SWA alpha) + B (center) + C (clips) + D (TAL)")
    print(f"  Runs: {', '.join(r['name'] for r in todo)}")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in todo:
        if not only and already_done(run_cfg["name"]):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (found in summary.json)")
            continue

        try:
            result = run_one(run_cfg)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "phase": run_cfg.get("phase"),
                      "label": run_cfg["label"], "seed": run_cfg.get("seed", SEED),
                      "elapsed_h": float("nan"), "val_map50": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}

        # replace stale entry if re-running, else append
        if result["name"] in done_names:
            summary = [r for r in summary if r["name"] != result["name"]]
        summary.append(result)
        done_names.add(result["name"])

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
    print(f"  {'Run':<24}{'Ph':>3}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 71}")

    def fmt(v, pct=True):
        if v != v:  # NaN
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    for r in sorted(summary, key=lambda x: x["name"]):
        print(f"  {r['name']:<24}{str(r.get('phase', '?')):>3}"
              f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
              f"{fmt(r['test_map50']):>12}{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()