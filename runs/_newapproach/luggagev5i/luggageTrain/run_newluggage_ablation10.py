#!/usr/bin/env python3
"""
Round 9b — 3 NEW mechanisms + improved NWD/DFL + anchor (8 trainings).

CONTEXT (30+ prior runs):
  Every reweighting/assignment variant landed within ±0.5 mAP50-95 of anchor.
  Diagnosed deficit (stable across ALL rounds):
    AR50_small ~0.96 but AP50-95_small ~0.51
    → small objects are FOUND but loosely BOXED at strict IoU.

  VFL (cls-side IoU-aware ranking) confirmed null — the gap is REGRESSION
  quality, not classification calibration. This round attacks only the
  regression path, with three new ideas plus improved versions of NWD/DFL.

ORIGINAL MECHANISMS IN THIS ROUND (see RECONCILED note below):

  r9b_iarw       [Section F, NEW-9] IoU-Aware Regression Weighting (iarw_gamma).
                  Per-anchor regression boost proportional to (1-IoU).detach().
                  NOT implemented in loss2.py -> removed; run == anchor.

  r9b_nwd_adapt  [NEW-5b/5c] Adaptive NWD with annealing.
                  loss2.py implements a PLAIN NWD blend only; the adaptive /
                  anneal knobs (nwd_adaptive, nwd_anneal, nwd_anneal_min) are
                  removed. Surviving lever: use_nwd + nwd_mode='blend' +
                  nwd_weight + nwd_C.

  r9b_dfl_gated  [NEW-3b] IoU-gated DFL boost (dfl_small_boost + dfl_iou_gated).
                  NOT in loss2.py -> removed; run == anchor at px36.

  r9b_iarw_nwd   IARW + NWD. iarw_gamma removed (not in loss2.py); only the
                  plain NWD blend survives (use_nwd + nwd_weight + nwd_C).

  r9b_iarw_dfl   IARW + IoU-gated DFL. Both mechanisms (iarw_gamma,
                  dfl_small_boost, dfl_iou_gated) are not in loss2.py -> removed;
                  run == anchor at px36.

RECONCILED TO loss2.py (the only loss implementation present in this repo):
  - NWD keys renamed: nwd_ratio -> nwd_weight, nwd_c -> nwd_C; NWD gated by
    use_nwd (default False). loss2.py has only a plain NWD blend.
  - cls_loss -> cls_mode ('bce' | 'qfl'); vfl_* removed.
  - Removed (no loss2.py impl): iarw_gamma, dfl_small_boost, dfl_iou_gated,
    nwd_adaptive, nwd_anneal, nwd_anneal_min, weight_renorm, area_*.
  CONSEQUENCE: only NWD-blend runs (r9b_nwd_adapt, r9b_iarw_nwd) stay live;
  the IARW / DFL-gated runs collapse to the anchor until loss2.py implements
  them. CAUTION: loss2.py nwd_C is STRIDE-NORMALIZED (~4.0), not pixels — the
  legacy 64.0 saturates NWD (~inert); retune to ~2-6.

VERIFY AT LAUNCH (config banner):
  r9b_anchor     : cls_mode bce | use_nwd False | nwd_weight/C 0.0/64.0
  r9b_iarw       : == anchor (iarw_gamma removed)
  r9b_iarw_lo    : == anchor (iarw_gamma removed)
  r9b_nwd_adapt  : use_nwd True nwd_mode blend nwd_weight/C 0.3/64.0
  r9b_dfl_gated  : == anchor at px36 (dfl_small_boost/iou_gated removed)
  r9b_iarw_nwd   : use_nwd True nwd_mode blend nwd_weight/C 0.3/64.0
  r9b_iarw_dfl   : == anchor at px36 (iarw/dfl_small_boost/iou_gated removed)

DECISION RULE (same as R9, fixed before any eval):
  A mechanism is a CANDIDATE only if val mAP50-95 > anchor + 0.5 OR
  val AP50-95_small > anchor + 0.8. Candidates (and the anchor) then get
  seeds 1 and 2 before any conclusion; test eval happens once, at the end,
  on the multi-seed survivors only (pass --with-test to force earlier).

Usage:
  python run_newluggage_ablation9.py                 # all runs not yet completed
  python run_newluggage_ablation9.py r9b_iarw        # only named run(s)
  python run_newluggage_ablation9.py --with-test     # also eval test split per run (discouraged)
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

PROJECT_DIR = "runs_newluggage_r10"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Shared OFF blocks — every run states EVERY custom param explicitly, so the
# saved ablation_params.json is ground truth (the eval JSONs never were).
# =============================================================================
# RECONCILED TO loss2.py:
#  - Removed (no loss2.py impl): weight_renorm, area_* , dfl_small_boost,
#    dfl_iou_gated, nwd_adaptive, nwd_anneal, nwd_anneal_min, iarw_gamma, vfl_*.
#  - Renamed to loss2.py names: nwd_ratio -> nwd_weight, nwd_c -> nwd_C,
#    cls_loss -> cls_mode. NWD is gated by use_nwd in loss2.py.
#  CONSEQUENCE: the IARW, IoU-gated-DFL and adaptive/anneal NWD mechanisms are
#  NOT in loss2.py, so runs built around them collapse toward the anchor. Only
#  the plain NWD blend (use_nwd + nwd_weight + nwd_C) survives as a live lever.
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_boost=1.0,
)
# NWD off block: loss2.py gates NWD on use_nwd (default False).
_TARGETED_OFF = dict(use_nwd=False, nwd_weight=0.0, nwd_C=64.0)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
# loss2.py uses cls_mode ('bce' | 'qfl'); no VFL, so vfl_* are removed.
_CLS_BCE = dict(cls_mode="bce")

# =============================================================================
# RUN CONFIGS
# =============================================================================

# Fresh anchor — every NEW path inert; the comparison basis for this round.
R9B_ANCHOR = dict(
    **_SWA_OFF, small_obj_px=48,  # px inert here (boost=1, nwd off)
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 1) IARW — IoU-Aware Regression Weighting (Section F, NEW-9)
#    NOTE: iarw_gamma is NOT implemented in loss2.py and was removed. This run
#    now equals the anchor and no longer tests IARW until loss2.py adds it.
# ---------------------------------------------------------------------------
R9B_IARW = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# Conservative IARW variant — same note as above (iarw_gamma removed).
R9B_IARW_LO = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 2) NWD BLEND (Section A'')
#    NOTE: loss2.py implements a plain NWD blend only. The adaptive/anneal
#    knobs (nwd_adaptive, nwd_anneal, nwd_anneal_min) are NOT implemented and
#    were removed; dfl_small_boost/dfl_iou_gated also removed. What remains is
#    use_nwd=True + nwd_mode='blend' + nwd_weight + nwd_C. CAUTION: loss2.py's
#    nwd_C is STRIDE-NORMALIZED (default 4.0), not pixels — the old 64.0 will
#    saturate NWD (~inert); retune nwd_C to ~2-6.
# ---------------------------------------------------------------------------
R9B_NWD_ADAPT = dict(
    **_SWA_OFF, small_obj_px=48,
    use_nwd=True, nwd_mode="blend", nwd_weight=0.3, nwd_C=64.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 3) IoU-GATED DFL BOOST (Section A', NEW-3b)
#    NOTE: dfl_small_boost and dfl_iou_gated are NOT in loss2.py and were
#    removed. This run now equals the anchor until loss2.py implements them.
# ---------------------------------------------------------------------------
R9B_DFL_GATED = dict(
    **_SWA_OFF, small_obj_px=36,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 4) COMPOSITION: IARW + NWD blend
#    NOTE: iarw_gamma and the adaptive/anneal NWD knobs are NOT in loss2.py and
#    were removed; only the plain NWD blend survives (use_nwd + nwd_weight +
#    nwd_C). Same nwd_C caution as run 2.
# ---------------------------------------------------------------------------
R9B_IARW_NWD = dict(
    **_SWA_OFF, small_obj_px=48,
    use_nwd=True, nwd_mode="blend", nwd_weight=0.3, nwd_C=64.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 5) COMPOSITION: IARW + IoU-gated DFL
#    NOTE: iarw_gamma, dfl_small_boost and dfl_iou_gated are NOT in loss2.py and
#    were removed. This run now equals the anchor until loss2.py implements
#    those mechanisms.
# ---------------------------------------------------------------------------
R9B_IARW_DFL = dict(
    **_SWA_OFF, small_obj_px=36,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER (anchor first — everything is judged against it)
# =============================================================================
RUNS = [
    {"name": "r9b_anchor",    "phase": "-",    "label": "Fresh anchor — all optional paths inert",                            "params": R9B_ANCHOR},
    {"name": "r9b_iarw",      "phase": "F",    "label": "IARW (iarw_gamma removed in loss2.py -> == anchor)",                 "params": R9B_IARW},
    {"name": "r9b_iarw_lo",   "phase": "F",    "label": "IARW lo (iarw_gamma removed in loss2.py -> == anchor)",             "params": R9B_IARW_LO},
    {"name": "r9b_nwd_adapt", "phase": "A''",  "label": "NWD blend weight=0.3 @px48 (adaptive/anneal removed in loss2.py)",   "params": R9B_NWD_ADAPT},
    {"name": "r9b_dfl_gated", "phase": "A'",   "label": "px36 (dfl_small_boost/iou_gated removed in loss2.py -> == anchor)",  "params": R9B_DFL_GATED},
    {"name": "r9b_iarw_nwd",  "phase": "F+A''","label": "NWD blend weight=0.3 @px48 (iarw removed in loss2.py)",             "params": R9B_IARW_NWD},
    {"name": "r9b_iarw_dfl",  "phase": "F+A'", "label": "px36 (iarw/dfl_small_boost/iou_gated removed in loss2.py -> anchor)","params": R9B_IARW_DFL},
    # After this round: seeds 1,2 for the anchor + any candidate passing the
    # decision rule in the header. Add entries like:
    # {"name": "r9b_anchor_s1", "phase": "-", "label": "anchor seed 1", "params": R9B_ANCHOR, "seed": 1},
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


def run_one(run_cfg, with_test=False):
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

    # ---- VAL-split metrics from training results (selection basis) ----
    val_map50, val_map5095 = float("nan"), float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key])
                break
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if key in rd:
                val_map5095 = float(rd[key])
                break
    except Exception:
        pass

    # ---- optional TEST eval (default OFF: selection must stay on val) ----
    test_map50, test_map5095 = float("nan"), float("nan")
    if with_test:
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
            "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "val_map5095": val_map5095,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
    """A run counts as done if its summary entry exists with a val score."""
    path = os.path.join(PROJECT_DIR, "summary.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for r in json.load(f):
                if r.get("name") == name and r.get("val_map5095") == r.get("val_map5095"):
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
    args = [a for a in sys.argv[1:]]
    with_test = "--with-test" in args
    only = {a for a in args if not a.startswith("--")}
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 70}")
    print("  ROUND 9b — IARW + adaptive NWD + IoU-gated DFL (regression-only attacks)")
    print(f"  Runs: {', '.join(r['name'] for r in todo)}")
    if with_test:
        print("  [!] --with-test: test-split eval per run (leaks test into selection)")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in todo:
        if not only and already_done(run_cfg["name"]):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (found in summary.json)")
            continue

        try:
            result = run_one(run_cfg, with_test=with_test)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "phase": run_cfg.get("phase"),
                      "label": run_cfg["label"], "seed": run_cfg.get("seed", SEED),
                      "elapsed_h": float("nan"),
                      "val_map50": float("nan"), "val_map5095": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}

        if result["name"] in done_names:
            summary = [r for r in summary if r["name"] != result["name"]]
        summary.append(result)
        done_names.add(result["name"])

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
    print(f"  {'Run':<18}{'Ph':>4}{'Time(h)':>9}{'val mAP50':>11}{'val 50-95':>11}"
          f"{'tst mAP50':>11}{'tst 50-95':>11}")
    print(f"  {'-' * 75}")

    def fmt(v, pct=True):
        if v != v:  # NaN
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    anchor = next((r for r in summary if r["name"] == "r9b_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>4}"
                f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
                f"{fmt(r.get('val_map5095', float('nan'))):>11}"
                f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if (anchor and r["name"] != "r9b_anchor"
                and r.get("val_map5095") == r.get("val_map5095")
                and anchor.get("val_map5095") == anchor.get("val_map5095")):
            d = (r["val_map5095"] - anchor["val_map5095"]) * 100
            line += f"   ({'+' if d >= 0 else ''}{d:.2f} vs anchor)"
        print(line)
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    print("\n  DECISION RULE: candidate iff val mAP50-95 > anchor+0.5"
          " (or small AP50-95 > anchor+0.8, from the run's own val logs).")
    print("  Candidates AND the anchor -> seeds 1,2 before any conclusion;"
          " test eval once, at the end, on multi-seed survivors only.")


if __name__ == "__main__":
    main()