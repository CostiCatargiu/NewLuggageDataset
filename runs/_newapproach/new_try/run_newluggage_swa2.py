#!/usr/bin/env python3
"""
SWA area-weight shape study (Round SWA2) — 10 overnight runs.

MOTIVATION
  The single best run so far is r0a_swa_a07_03_sqrt (mAP50-95 +0.43, small +0.47
  vs the r9_anchor baseline of 57.43 / 52.19). BUT that run changed TWO things at
  once vs the anchor: the area-weight SHAPE (area_weight_mode='sqrt') AND the
  small-object boost (small_obj_boost=2.0). Every sqrt run ever done used boost=2.0,
  so we cannot yet tell whether the win comes from the sqrt shape or the boost.

  Also: sqrt was only ever tested at two schedules (0.7->0.3 and 0.9->0.4), and the
  'log' shape has never been tested at all.

WHAT alpha DOES (from loss2.py):
  weight = alpha * area_weight + (1 - alpha) * score_weight   (line 590)
    alpha = 1.0 -> 100% size-based weighting (emphasize small objects)
    alpha = 0.0 -> pure stock TAL score weighting (== anchor/baseline)
  alpha anneals linearly alpha_start -> alpha_end over training, clamped to
  [alpha_min, alpha_max]. 'area_weight_mode' reshapes area_weight (inv|sqrt|log)
  BEFORE the small_obj_boost multiplier is applied.

DESIGN (10 runs, all px=48, center/clip/TAL OFF so ONLY the SWA block varies):
  Block A — isolate sqrt SHAPE vs BOOST (sqrt @ 0.7->0.3):
      swa2_sqrt_a07_b1   boost 1.0   (shape only — is sqrt itself the win?)
      swa2_sqrt_a07_b15  boost 1.5
      swa2_sqrt_a07_b25  boost 2.5   (stronger dose; inv control dropped — inv never helps)
  Block B — sqrt SCHEDULE sweep @ boost 2.0 (fills untested sqrt x schedule gaps):
      swa2_sqrt_a08_04   0.8->0.4
      swa2_sqrt_a06_03   0.6->0.3
      swa2_sqrt_a05_025  0.5->0.25
      swa2_sqrt_a07_03   0.7->0.3   (winner re-run / repro sanity anchor)
  Block C — 'log' SHAPE (never tested) @ 0.7->0.3:
      swa2_log_a07_b2    boost 2.0
      swa2_log_a07_b1    boost 1.0
  Block D — best-guess STACK: sqrt winner + NWD blend (both reproducible winners):
      swa2_sqrt_a07_nwd  0.7->0.3 sqrt boost2 + use_nwd blend, nwd_weight=0.3,
                         nwd_C=4.0  (CORRECTED: loss2.py nwd_C is stride-normalized
                         ~2-6, NOT the old pixel-space 64 which saturated NWD to inert)

DECISION RULE (fixed in advance, val split, before test eval):
  candidate iff val mAP50-95 > anchor + 0.5 OR val AP50-95_small > anchor + 0.8.
  Candidates + winner-repro then get seeds 1,2 before any conclusion.

Usage:
  python run_newluggage_swa2.py                    # all runs not yet done
  python run_newluggage_swa2.py swa2_sqrt_a07_b1   # only named run(s)
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
# CONFIGURATION  (identical training settings to run_newluggage_ablation.py)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_newluggage5_swa2"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Shared OFF blocks — everything except the SWA area block is stock/inert.
# =============================================================================
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
# NWD off (loss2.py gates NWD on use_nwd, default False)
_NWD_OFF = dict(use_nwd=False, nwd_weight=0.0, nwd_C=64.0)


def _swa(alpha_start, alpha_end, area_mode, boost, **extra):
    """Build an SWA config. alpha_min/max bracket the schedule so it isn't clamped."""
    lo = min(alpha_start, alpha_end)
    hi = max(alpha_start, alpha_end)
    cfg = dict(
        alpha_start=alpha_start, alpha_end=alpha_end, alpha_min=lo, alpha_max=hi,
        small_obj_px=48, small_obj_boost=boost, area_weight_mode=area_mode,
        **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
    )
    cfg.update(extra)
    return cfg


# --- Block A: sqrt shape vs boost (sqrt @ 0.7->0.3) ---
SWA2_SQRT_A07_B1  = _swa(0.7, 0.3, "sqrt", 1.0, **_NWD_OFF)
SWA2_SQRT_A07_B15 = _swa(0.7, 0.3, "sqrt", 1.5, **_NWD_OFF)
SWA2_SQRT_A07_B25 = _swa(0.7, 0.3, "sqrt", 2.5, **_NWD_OFF)

# --- Block B: sqrt schedule sweep @ boost 2.0 ---
SWA2_SQRT_A08_04  = _swa(0.8, 0.4,  "sqrt", 2.0, **_NWD_OFF)
SWA2_SQRT_A06_03  = _swa(0.6, 0.3,  "sqrt", 2.0, **_NWD_OFF)
SWA2_SQRT_A05_025 = _swa(0.5, 0.25, "sqrt", 2.0, **_NWD_OFF)
SWA2_SQRT_A07_03  = _swa(0.7, 0.3,  "sqrt", 2.0, **_NWD_OFF)   # winner repro

# --- Block C: log shape @ 0.7->0.3 ---
SWA2_LOG_A07_B2 = _swa(0.7, 0.3, "log", 2.0, **_NWD_OFF)
SWA2_LOG_A07_B1 = _swa(0.7, 0.3, "log", 1.0, **_NWD_OFF)

# --- Block D: sqrt winner + NWD blend (nwd_C corrected 64 -> 4) ---
SWA2_SQRT_A07_NWD = _swa(0.7, 0.3, "sqrt", 2.0,
                         use_nwd=True, nwd_mode="blend", nwd_weight=0.3, nwd_C=4.0)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER (winner repro early as a sanity anchor)
# =============================================================================
RUNS = [
    {"name": "swa2_sqrt_a07_03",  "phase": "B", "label": "sqrt 0.7->0.3 boost2.0 — WINNER REPRO / sanity anchor", "params": SWA2_SQRT_A07_03},
    # Block A — isolate shape vs boost
    {"name": "swa2_sqrt_a07_b1",  "phase": "A", "label": "sqrt 0.7->0.3 boost1.0 — SHAPE ONLY (is sqrt the win?)", "params": SWA2_SQRT_A07_B1},
    {"name": "swa2_sqrt_a07_b15", "phase": "A", "label": "sqrt 0.7->0.3 boost1.5 — mid dose",                     "params": SWA2_SQRT_A07_B15},
    {"name": "swa2_sqrt_a07_b25", "phase": "A", "label": "sqrt 0.7->0.3 boost2.5 — stronger dose",                "params": SWA2_SQRT_A07_B25},
    # Block B — sqrt schedule sweep @ boost 2.0
    {"name": "swa2_sqrt_a08_04",  "phase": "B", "label": "sqrt 0.8->0.4 boost2.0 — stronger schedule",            "params": SWA2_SQRT_A08_04},
    {"name": "swa2_sqrt_a06_03",  "phase": "B", "label": "sqrt 0.6->0.3 boost2.0 — milder schedule",              "params": SWA2_SQRT_A06_03},
    {"name": "swa2_sqrt_a05_025", "phase": "B", "label": "sqrt 0.5->0.25 boost2.0 — mildest schedule",            "params": SWA2_SQRT_A05_025},
    # Block C — log shape
    {"name": "swa2_log_a07_b2",   "phase": "C", "label": "log 0.7->0.3 boost2.0 — gentler-than-sqrt shape",       "params": SWA2_LOG_A07_B2},
    {"name": "swa2_log_a07_b1",   "phase": "C", "label": "log 0.7->0.3 boost1.0 — log shape only",                "params": SWA2_LOG_A07_B1},
    # Block D — stack
    {"name": "swa2_sqrt_a07_nwd", "phase": "D", "label": "sqrt 0.7->0.3 boost2.0 + NWD blend w0.3 C4.0 (corrected)","params": SWA2_SQRT_A07_NWD},
]


# =============================================================================
# Epoch sync — drives alpha / clip / center-decay schedules in the custom loss
# =============================================================================
def on_train_epoch_start(trainer):
    """Push trainer.epoch into the custom loss, DDP-safe."""
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
    print("  SWA2 SWEEP — area-weight shape (sqrt/log) x boost x schedule")
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
