#!/usr/bin/env python3
"""
Round 10 — REGRESSION-SIGNAL mechanisms (not reweighting).

WHY THIS ROUND IS DIFFERENT
  Rounds 2-9 all landed within +/-0.5 mAP50-95 of anchor. Diagnosis (stable
  across 30+ runs): AR50_small ~0.96 but AP50-95_small ~0.51 -> small objects
  are FOUND but LOOSELY BOXED at strict IoU. Every prior mechanism (area weight,
  small_obj_boost, dfl_small_boost, NWD blend, IARW, SATAL) is loss-side
  REWEIGHTING of the same CIoU+DFL signal. Reweighting is zero-sum under
  weight_renorm and cannot add localization information -> the plateau.

  Round 10 changes the REGRESSION SIGNAL ITSELF:

  r10_alpha_iou   [NEW-10] alpha-IoU (power-IoU). loss = 1 - IoU^alpha (+CIoU
                  penalty). Concentrates gradient in the HIGH-IoU regime, which
                  is exactly where AP75/AP90 live. One knob, directly targets
                  the strict-threshold deficit. alpha=3.0.

  r10_l1_smooth   [NEW-11] Pixel-space Smooth-L1 residual auxiliary on L/T/R/B.
                  (1-IoU) gradient FLATTENS as the box tightens; Smooth-L1 keeps
                  a non-vanishing gradient near the optimum -> supplies the
                  "last few pixels" signal. small-object gated, added post-clip.

  r10_dfl_entropy [NEW-12] DFL distribution sharpening. Penalizes the entropy of
                  each side's softmax bin distribution so loose/multi-modal edge
                  distributions become sharp & unimodal -> crisper decoded edges.
                  Shapes the PREDICTION, not the loss weight. small-gated.

  r10_nwd_fixedc  [NEW-13] Adaptive NWD with SIZE-ADAPTIVE temperature. The R9
                  NWD used a fixed c=64px, which saturates (nwd~1 -> inert) for
                  the small boxes it targets. Here c = 0.5*sqrt(area_px), so NWD
                  stays discriminative in the small regime. Combined with the
                  R9b adaptive+anneal ratio schedule.

  r10_tightness   [NEW-14] Asymmetric tightness penalty. Loose small boxes spill
                  OVER the GT; charge over-extended sides (normalized by object
                  diagonal) to pull boxes to hug the object. small-gated.

  r10_combo       alpha-IoU 2.0 + Smooth-L1 0.05 + DFL-entropy 0.03. These are
                  orthogonal: alpha-IoU reshapes the overlap gradient, L1 adds
                  sub-pixel residual gradient, entropy sharpens the DFL bins.

REQUIRES (loss.py, this version): NEW-10..14 params. You MUST also whitelist the
  new keys in whatever ultralytics cfg patch already whitelists iarw_gamma/
  nwd_ratio, or model.train() will drop them and every run silently == anchor:
    alpha_iou, l1_aux_weight, l1_aux_beta, l1_aux_small_only, l1_balanced,
    l1_balanced_alpha, l1_balanced_gamma, dfl_entropy_weight,
    dfl_entropy_small_only, nwd_c_adaptive, nwd_c_k, tightness_gamma,
    tightness_small_only

VERIFY AT LAUNCH (loss.py config banner must show):
  r10_anchor     : alpha_iou=1.0 l1_aux_weight=0.0 dfl_entropy=0.0
                   nwd_c_adaptive=0 tightness_gamma=0.0   (all inert)
  r10_alpha_iou  : alpha_iou=3.0, everything else off
  r10_l1_smooth  : l1_aux_weight=0.05 Smooth-L1, everything else off
  r10_dfl_entropy: dfl_entropy=0.05, everything else off
  r10_nwd_fixedc : nwd_ratio=0.3 adaptive+anneal + nwd_c_adaptive=1
  r10_tightness  : tightness_gamma=0.5, everything else off
  r10_combo      : alpha_iou=2.0 + l1_aux_weight=0.05 + dfl_entropy=0.03

DECISION RULE (fixed before any eval, same as R9):
  candidate iff val mAP50-95 > anchor+0.5 OR val AP50-95_small > anchor+0.8.
  Candidates + anchor -> seeds 1,2; test eval once, at the end, on survivors.

Usage:
  python run_newluggage_ablation10.py                # all runs
  python run_newluggage_ablation10.py r10_alpha_iou  # only named run(s)
  python run_newluggage_ablation10.py --with-test    # also eval test (discouraged)
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
except ImportError:
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
# Shared OFF blocks — every run states EVERY custom param explicitly.
# =============================================================================
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_boost=1.0,
    weight_renorm=1, area_mode="fixed", area_ref_px=64.0, area_gamma=0.5,
    area_w_cap=3.0,
)
_TARGETED_OFF = dict(
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
)
_IARW_OFF = dict(iarw_gamma=0.0)
# --- Round 10 OFF block (all new mechanisms inert) ---
_R10_OFF = dict(
    alpha_iou=1.0,
    l1_aux_weight=0.0, l1_aux_beta=2.0, l1_aux_small_only=1,
    l1_balanced=0, l1_balanced_alpha=0.5, l1_balanced_gamma=1.5,
    dfl_entropy_weight=0.0, dfl_entropy_small_only=1,
    nwd_c_adaptive=0, nwd_c_k=0.5,
    tightness_gamma=0.0, tightness_small_only=1,
)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
_CLS_BCE = dict(cls_loss="bce", vfl_alpha=0.75, vfl_gamma=2.0)


def _base(**overrides):
    """Fully-inert config; overrides flip on exactly the mechanism under test."""
    cfg = dict(
        **_SWA_OFF, small_obj_px=48,
        **_TARGETED_OFF, **_IARW_OFF, **_R10_OFF,
        **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
    )
    cfg.update(overrides)
    return cfg


# =============================================================================
# RUN CONFIGS
# =============================================================================
R10_ANCHOR      = _base()                                        # all inert
R10_ALPHA_IOU   = _base(alpha_iou=3.0)                           # NEW-10
R10_L1_SMOOTH   = _base(l1_aux_weight=0.05, l1_balanced=0,       # NEW-11
                        l1_aux_beta=2.0, l1_aux_small_only=1)
R10_DFL_ENTROPY = _base(dfl_entropy_weight=0.05,                 # NEW-12
                        dfl_entropy_small_only=1)
R10_NWD_FIXEDC  = _base(nwd_ratio=0.3, nwd_c=64.0,               # NEW-13 (+5b/5c)
                        nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
                        nwd_c_adaptive=1, nwd_c_k=0.5, small_obj_px=48)
R10_TIGHTNESS   = _base(tightness_gamma=0.5, tightness_small_only=1)  # NEW-14
R10_COMBO       = _base(alpha_iou=2.0,                           # NEW-10+11+12
                        l1_aux_weight=0.05, l1_aux_beta=2.0, l1_aux_small_only=1,
                        dfl_entropy_weight=0.03, dfl_entropy_small_only=1)

RUNS = [
    {"name": "r10_anchor",      "phase": "-",    "label": "Fresh anchor — all NEW-10..14 inert",                 "params": R10_ANCHOR},
    {"name": "r10_alpha_iou",   "phase": "R10",  "label": "alpha-IoU 3.0 — power-IoU, high-IoU gradient focus",  "params": R10_ALPHA_IOU},
    {"name": "r10_l1_smooth",   "phase": "R10",  "label": "Smooth-L1 pixel aux 0.05 (small) — sub-pixel signal", "params": R10_L1_SMOOTH},
    {"name": "r10_dfl_entropy", "phase": "R10",  "label": "DFL entropy sharpen 0.05 (small) — crisper edges",    "params": R10_DFL_ENTROPY},
    {"name": "r10_nwd_fixedc",  "phase": "R10",  "label": "Adaptive NWD 0.3 + size-adaptive c — corrected NWD",  "params": R10_NWD_FIXEDC},
    {"name": "r10_tightness",   "phase": "R10",  "label": "Asymmetric tightness 0.5 (small) — anti-spillover",   "params": R10_TIGHTNESS},
    {"name": "r10_combo",       "phase": "R10",  "label": "alpha-IoU 2.0 + L1 0.05 + DFL-ent 0.03 — orthogonal", "params": R10_COMBO},
]


# =============================================================================
# Epoch sync — drives alpha / clip / anneal schedules in the custom loss
# =============================================================================
def on_train_epoch_start(trainer):
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
        "data": DATA_YAML, "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
        "device": DEVICE, "workers": WORKERS, "project": PROJECT_DIR, "name": name,
        "patience": 100, "close_mosaic": 10, "seed": seed,
        "deterministic": True, "exist_ok": False,
    }
    train_kwargs.update(copy.deepcopy(params))

    results = model.train(**train_kwargs)
    elapsed = (time.time() - start_time) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "phase": run_cfg.get("phase"), "label": label,
                       "params": params, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                       "batch": BATCH, "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    val_map50, val_map5095 = float("nan"), float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key]); break
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if key in rd:
                val_map5095 = float(rd[key]); break
    except Exception:
        pass

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

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "phase": run_cfg.get("phase"), "label": label,
            "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "val_map5095": val_map5095,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
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
    print("  ROUND 10 — regression-signal mechanisms (alpha-IoU, L1, DFL-entropy, NWD-c, tightness)")
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
    print(f"  {'Run':<18}{'Ph':>5}{'Time(h)':>9}{'val mAP50':>11}{'val 50-95':>11}"
          f"{'tst mAP50':>11}{'tst 50-95':>11}")
    print(f"  {'-' * 76}")

    def fmt(v, pct=True):
        if v != v:
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    anchor = next((r for r in summary if r["name"] == "r10_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>5}"
                f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
                f"{fmt(r.get('val_map5095', float('nan'))):>11}"
                f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if (anchor and r["name"] != "r10_anchor"
                and r.get("val_map5095") == r.get("val_map5095")
                and anchor.get("val_map5095") == anchor.get("val_map5095")):
            d = (r["val_map5095"] - anchor["val_map5095"]) * 100
            line += f"   ({'+' if d >= 0 else ''}{d:.2f} vs anchor)"
        print(line)
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    print("\n  DECISION RULE: candidate iff val mAP50-95 > anchor+0.5"
          " (or small AP50-95 > anchor+0.8, from the run's own val logs).")
    print("  Candidates AND the anchor -> seeds 1,2; test eval once, at the end.")


if __name__ == "__main__":
    main()
