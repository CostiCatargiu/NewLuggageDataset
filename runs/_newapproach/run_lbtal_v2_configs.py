#!/usr/bin/env python3
"""
LB-TAL v2 — NEW mechanism configs designed to BEAT lb_uniform (55.57 / +0.87 small).

=============================================================================
WHY THESE (grounded in the isolated + overnight results, NOT random tuning)
=============================================================================
MEASURED SEED NOISE (lb_uniform vs lb_uniform_seed1 — same config, seed 0 vs 1):
    overall mAP50-95  0.12 pp     large mAP50-95  2.06 pp
Every claim below is stated against those numbers. Read the table with them.

  SIGNAL A [SUPPORTED] - lb_coarse_244 has the HIGHEST recall ceiling
             (AR50_small 0.9658 vs uniform 0.9595) but LOWER mAP (55.34 vs
             55.57). AR is a ceiling metric and less jittery than mAP, so this
             is the solid signal: coarse levels FIND more small objects but
             some coarse positives are LOW QUALITY and don't convert.
             FIX: quality-gate the per-level picks (drop weak coarse positives).

  SIGNAL B [WEAK — do not over-trust] - the SWA+LB-TAL combo scored 55.10.
             The original rationale was "over-boosting small collapsed LARGE",
             but the data does not show that:
               small mAP50-95  cmb 0.5047  vs uniform 0.5085  -> small went DOWN
               large mAP50-95  cmb 0.5437  vs uniform 0.5775  -> -3.4 pp
             There was no small-boost to trade against large, and the 3.4 pp
             large drop is only ~1.6x the 2.06 pp seed noise on that metric,
             from a single pair. lb_balcap therefore tests a hypothesis that is
             not yet established. Keep it, but rank it below the qg configs.

NOT a premise: "uniform == uniform_mk2 proves tuning has peaked." Those two are
the same config under deterministic=True/seed=0 — identical to 4 decimals on
all 30 metrics. That demonstrates reproducibility, not a plateau.

=============================================================================
THE 6 NEW CONFIGS
=============================================================================
  1. lb_uniform_qg50   uniform + quality_gate 0.50  — admit only per-level picks
                       >= 50% of the GT's GLOBAL best (across levels).
                       Tests SIGNAL A on the winner. RUN THIS ONE FIRST.
  2. lb_uniform_qg70   uniform + quality_gate 0.70  — stricter gate.
  3. lb_coarse_qg50    coarse {8:2,16:4,32:4} + qg 0.50 — coarse_244 had the best
                       recall ceiling; gate its weak positives to convert it to mAP.
  4. lb_balcap         'balanced_capped' mode — P3 trimmed, P4/P5 protected.
                       Directly tests SIGNAL B (do-no-harm-to-large balancing).
  5. lb_balcap_qg50    balanced_capped + quality_gate 0.50 — combine both fixes.
  6. lb_uniform_tk13_qg50  uniform + tal_topk 13 + qg 0.50 — bigger gated budget.

Baselines to beat: lb_uniform 55.57 (+0.87 small, AR50_small 0.960),
                   SWA sqrt0703 55.64 (+0.65 small), anchor 54.77.
READ small mAP + AR50_small (CocoEvalAllFolders_luggage.py), not just overall.
This script CANNOT produce per-size metrics — ultralytics val has no
small/medium/large breakdown. It records the weights path for each run so the
coco eval can be pointed at them afterwards. Overall mAP alone will not settle
this sweep: the configs are expected to differ by ~0.2-0.5 pp.

GATE SEMANTICS (changed): quality_gate is measured against the GT's best metric
ACROSS ALL LEVELS, not within each level. The per-level version was near-inert
— torch.topk returns sorted, so each level's top-1 always passed and a
uniformly-weak coarse level survived intact, which is the opposite of the
intent. 4 of the 6 configs below depend on the gate doing real work.

VERIFY FIRST: python selftest_lbtal.py  (all PASS) — the new modes/gate are
covered if you re-run it; at minimum confirm no import/shape errors.
This script also preflights the installed loss.py (path/md5/features) and
aborts rather than silently training stock TAL for 9 GPU-h.

REQUIRES lossv2updated.py (with balanced_capped + quality_gate) installed as
ultralytics/utils/loss.py, and lbtal_quality_gate whitelisted in default.yaml.

Usage:
    python run_lbtal_v2_configs.py
    python run_lbtal_v2_configs.py lb_uniform_qg50 lb_coarse_qg50
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
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_lbtal_v2"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
BASELINE_TEST_MAP5095 = 0.5477

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
    use_lbtal=False,
)


def _lb(mode, level_topk=None, min_level_k=1, quality_gate=0.0, topk=10):
    return dict(_ALL_OFF, use_lbtal=True, lbtal_mode=mode,
                lbtal_level_topk=level_topk, lbtal_min_level_k=min_level_k,
                lbtal_quality_gate=quality_gate, tal_topk=topk)


RUNS = [
    {"name": "lb_uniform_qg50", "batch": BATCH,
     "label": "uniform + quality_gate 0.50 — drop weak per-level positives (Signal A)",
     "params": _lb("uniform", quality_gate=0.50)},

    {"name": "lb_uniform_qg70", "batch": BATCH,
     "label": "uniform + quality_gate 0.70 — stricter gate",
     "params": _lb("uniform", quality_gate=0.70)},

    {"name": "lb_coarse_qg50", "batch": BATCH,
     "label": "coarse {8:2,16:4,32:4} + qg 0.50 — convert coarse_244's recall ceiling to mAP",
     "params": _lb("fixed", level_topk={8: 2, 16: 4, 32: 4}, quality_gate=0.50)},

    {"name": "lb_balcap", "batch": BATCH,
     "label": "balanced_capped — trim P3, protect P4/P5 (Signal B: do-no-harm-to-large)",
     "params": _lb("balanced_capped")},

    {"name": "lb_balcap_qg50", "batch": BATCH,
     "label": "balanced_capped + quality_gate 0.50 — both fixes together",
     "params": _lb("balanced_capped", quality_gate=0.50)},

    {"name": "lb_uniform_tk13_qg50", "batch": BATCH,
     "label": "uniform + tal_topk 13 + qg 0.50 — bigger gated budget",
     "params": _lb("uniform", quality_gate=0.50, topk=13)},
]


def loss_provenance():
    """Record which loss.py is actually installed.

    The whole sweep is a no-op if ultralytics is running stock TAL, and the
    unknown params would be silently ignored rather than raising. Cheap
    insurance: log path + md5 + a marker for the features this sweep needs.
    """
    info = {"path": None, "md5": None, "has_lbtal": False,
            "has_balanced_capped": False, "has_quality_gate": False}
    try:
        import ultralytics.utils.loss as _lm
        path = getattr(_lm, "__file__", None)
        info["path"] = path
        if path and os.path.exists(path):
            src = open(path, "rb").read()
            info["md5"] = hashlib.md5(src).hexdigest()[:12]
            txt = src.decode("utf-8", "ignore")
            info["has_lbtal"] = "use_lbtal" in txt
            info["has_balanced_capped"] = "balanced_capped" in txt
            info["has_quality_gate"] = "quality_gate" in txt
    except Exception as e:
        info["error"] = str(e)
    return info


LOSS_INFO = loss_provenance()


def preflight():
    """Abort before burning GPU-hours on a loss.py that can't run these configs."""
    print(f"  loss.py: {LOSS_INFO.get('path')}")
    print(f"  md5={LOSS_INFO.get('md5')}  lbtal={LOSS_INFO.get('has_lbtal')} "
          f"balanced_capped={LOSS_INFO.get('has_balanced_capped')} "
          f"quality_gate={LOSS_INFO.get('has_quality_gate')}")
    missing = [k for k in ("has_lbtal", "has_balanced_capped", "has_quality_gate")
               if not LOSS_INFO.get(k)]
    if missing:
        print(f"\n  [ABORT] installed loss.py is missing: {', '.join(missing)}")
        print("  Install lossv2updated.py as ultralytics/utils/loss.py first.")
        return False
    return True


def collect_metrics(tm):
    """Pull everything the val object exposes, incl. per-class.

    NOTE: ultralytics' DetMetrics has no per-size (small/medium/large)
    breakdown — that comes from CocoEvalAllFolders_luggage.py. Small mAP and
    AR50_small are the stated target of this sweep, so run the coco eval on
    the `weights` path recorded below before drawing conclusions.
    """
    out = {}
    try:
        box = tm.box
        out["map50"] = float(box.map50)
        out["map5095"] = float(box.map)
        out["precision"] = float(box.mp)
        out["recall"] = float(box.mr)
        out["map75"] = float(getattr(box, "map75", float("nan")))
        names = getattr(tm, "names", {}) or {}
        per_class = {}
        ap50 = getattr(box, "ap50", None)
        ap = getattr(box, "maps", None)
        idxs = list(getattr(box, "ap_class_index", [])) or list(range(len(names)))
        for i, ci in enumerate(idxs):
            cname = names.get(ci, str(ci)) if isinstance(names, dict) else str(ci)
            rec = {}
            if ap50 is not None and i < len(ap50):
                rec["AP50"] = float(ap50[i])
            if ap is not None and ci < len(ap):
                rec["AP50_95"] = float(ap[ci])
            if rec:
                per_class[cname] = rec
        if per_class:
            out["per_class"] = per_class
    except Exception as e:
        out["error"] = str(e)
    return out


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


def run_one(rc):
    name, params, batch = rc["name"], rc["params"], rc["batch"]
    print(f"\n{'=' * 76}\n  RUN {name}\n  {rc['label']}\n"
          f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
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
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch, "seed": SEED,
                       "loss_file": LOSS_INFO}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")
    rd = getattr(results, "results_dict", {}) or {}
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "label": rc["label"], "batch": batch, "hours": hours,
           "seed": SEED, "epochs": EPOCHS, "imgsz": IMG_SIZE,
           "save_dir": save_dir, "weights": weights,
           "params": copy.deepcopy(params), "loss_file": LOSS_INFO,
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        tm = YOLO(weights).val(
            data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
        out["test_metrics"] = collect_metrics(tm)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    ref = BASELINE_TEST_MAP5095
    print(f"\n{'=' * 76}\n  LB-TAL v2 RESULTS (test)\n{'=' * 76}")
    print(f"{'run':<22}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'vs_uniform':>12}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = ("%+9.2f" % ((r["test_map5095"] - ref) * 100)) if ref else "-"
        du = ("%+11.2f" % ((r["test_map5095"] - 0.5557) * 100))
        print(f"{r['name']:<22}{r['test_map50'] * 100:>9.2f}{r['test_map5095'] * 100:>11.2f}{d}{du}{r['hours']:>6.1f}")
    print(f"\n  anchor {ref*100:.2f} | lb_uniform 55.57 | SWA sqrt0703 55.64")
    print("  vs_uniform > 0 -> a v2 config BEAT the current LB-TAL winner.")
    print("\n  [!] Measured seed noise (lb_uniform vs lb_uniform_seed1, same config):")
    print("        overall mAP50-95  ~0.12 pp")
    print("        large   mAP50-95  ~2.06 pp")
    print("      Treat gaps below those as unresolved, not as a ranking.")
    print("\n  [!] Per-size metrics (small mAP, AR50_small) are the real target and")
    print("      are NOT produced by ultralytics val. Run CocoEvalAllFolders_luggage.py")
    print("      on the weights below:")
    for r in res:
        w = r.get("weights")
        if w:
            print(f"        {r['name']:<22} {w}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 76}\n  LB-TAL v2 — {len(todo)} new mechanism configs to beat lb_uniform")
    print(f"  runs: {', '.join(r['name'] for r in todo)}  (~{1.5*len(todo):.0f} GPU-h)")
    print(f"{'=' * 76}\n")
    if not preflight():
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "batch": r["batch"], "hours": float("nan"),
                        "val_map5095": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
