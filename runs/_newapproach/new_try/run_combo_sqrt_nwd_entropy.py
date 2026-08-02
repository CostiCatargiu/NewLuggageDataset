#!/usr/bin/env python3
"""
COMBO STUDY — stack the 3 independent winners in ONE loss (lossv2updated.py).

=============================================================================
MOTIVATION
=============================================================================
Three mechanisms each beat the baseline (57.43) on DIFFERENT size buckets, so
they are orthogonal and should compound:

  SWA-sqrt      r0a_swa_a07_03_sqrt = 57.86  — best on SMALL (52.66) + recall.
                area_weight_mode='sqrt', alpha 0.7->0.3, boost 2.0, px48.
  NWD blend     r10_nwd_fixedc      = 57.75  — best on LARGE/MED (+2.72/+0.72).
                use_nwd blend, nwd_weight 0.3, nwd_C 4.0 (STRIDE-normalized!).
  DFL-entropy   r10_dfl_entropy     = 57.71  — best on LARGE (+2.89), small +0.14.
                dfl_entropy_weight 0.05 (global edge-distribution sharpening).

All three now live in lossv2updated.py (NWD + sqrt were native; DFL-entropy was
ported from loss_dflentropy.py, global variant). This study tests whether
stacking helps AND stays attributable via the pairwise controls.

CRITICAL: nwd_C = 4.0, NOT 64. Inside BboxLoss the boxes are stride-normalized
(target_bboxes /= stride_tensor), so 64 saturates NWD to inert (the old bug).
Watch the first-batch NWD debug print: nwd.mean() should be ~0.3-0.8.

=============================================================================
DESIGN — COMBINATIONS ONLY (singles/anchor skipped; their results are known)
=============================================================================
  combo_sqrt_nwd      sqrt + NWD               (the 2 strongest, most orthogonal)
  combo_sqrt_entropy  sqrt + entropy
  combo_nwd_entropy   NWD + entropy
  combo_all           sqrt + NWD + entropy     (the 3-way)

Compared against the KNOWN single-mechanism results:
  baseline 57.43 | SWA-sqrt 57.86 | NWD 57.75 | DFL-entropy 57.71
A pair only "adds" if it beats both its parents; the triple only ships if it
beats the best pair. A combo only WINS overall if it clears 57.86 (best single).

REQUIRES lossv2updated.py copied to ultralytics/utils/loss.py, and the keys
area_weight_mode / use_nwd / nwd_* / dfl_entropy_weight whitelisted in
cfg/default.yaml.

Usage:
  python run_combo_sqrt_nwd_entropy.py                 # all not-yet-done
  python run_combo_sqrt_nwd_entropy.py combo_all       # a subset
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
# CONFIGURATION  — must match the SWA / baseline recipe (57.43 / 57.86)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_combo_sqrt_nwd_entropy"

EPOCHS = 70
IMG_SIZE = 640            # !! eval MUST also be 640 (the 896 lesson)
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

BASELINE_TEST_MAP5095 = 0.5743   # r9_anchor2 / all anchors, test split
SWA_SQRT_REF = 0.5786            # r0a_swa_a07_03_sqrt, the current best

# =============================================================================
# Mechanism building blocks
# =============================================================================
_CENTER_OFF = dict(center_loss_weight_init=0.0, center_loss_weight_min=0.0,
                   center_loss_decay_epochs=35)
_CLIP_OFF = dict(iou_clip_start=999.0, iou_clip_end=999.0,
                 dfl_clip_start=999.0, dfl_clip_end=999.0)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# SWA off = alpha schedule inert (alpha 0 -> pure score weighting = stock)
_SWA_OFF = dict(alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
                small_obj_px=48, small_obj_boost=1.0, area_weight_mode="inv")
# SWA-sqrt winner
_SWA_SQRT = dict(alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
                 small_obj_px=48, small_obj_boost=2.0, area_weight_mode="sqrt")

# NWD off / on (nwd_C = 4.0, stride-normalized!)
_NWD_OFF = dict(use_nwd=False, nwd_weight=0.0, nwd_C=4.0)
_NWD_ON = dict(use_nwd=True, nwd_mode="blend", nwd_weight=0.3, nwd_C=4.0)

# DFL-entropy off / on
_ENT_OFF = dict(dfl_entropy_weight=0.0)
_ENT_ON = dict(dfl_entropy_weight=0.05)


def _cfg(swa, nwd, ent):
    """Assemble a full config from the three mechanism blocks (+ stock rest)."""
    c = dict(**_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
    c.update(swa)
    c.update(nwd)
    c.update(ent)
    return c


# Reproducibility runs (anchor + the 3 singles) SKIPPED — their results are
# already known (anchor 57.43, sqrt 57.86, nwd 57.75, entropy 57.71). Only the
# COMBINATIONS are run here, compared against those known single-mechanism refs.
RUNS = [
    {"name": "combo_sqrt_nwd",     "params": _cfg(_SWA_SQRT, _NWD_ON,  _ENT_OFF),
     "label": "sqrt + NWD (2 strongest, orthogonal small vs large)"},
    {"name": "combo_sqrt_entropy", "params": _cfg(_SWA_SQRT, _NWD_OFF, _ENT_ON),
     "label": "sqrt + DFL-entropy"},
    {"name": "combo_nwd_entropy",  "params": _cfg(_SWA_OFF,  _NWD_ON,  _ENT_ON),
     "label": "NWD + DFL-entropy"},
    {"name": "combo_all",          "params": _cfg(_SWA_SQRT, _NWD_ON,  _ENT_ON),
     "label": "sqrt + NWD + DFL-entropy (the 3-way)"},
]


# =============================================================================
# Epoch sync — drives the SWA alpha schedule in the custom loss
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


def run_one(rc):
    name, params = rc["name"], rc["params"]
    print(f"\n{'=' * 76}\n  RUN {name}\n  {rc['label']}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=100, close_mosaic=10, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "combo_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
                       "seed": SEED}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    # TEST-split eval at 640 (NEVER 896)
    try:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        tm = YOLO(best_pt).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                               batch=BATCH, device=DEVICE, workers=WORKERS,
                               project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def already_done(name, summary):
    for r in summary:
        if r.get("name") == name and r.get("test_map5095") == r.get("test_map5095"):
            return True
    return False


def load_summary():
    p = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def summarise(summary):
    # No anchor run here; compare deltas against the known SWA-sqrt best (57.86),
    # since a combo only matters if it beats the best single mechanism.
    ref = SWA_SQRT_REF

    def pc(v):
        return f"{v * 100:.2f}" if v == v else "n/a"

    print(f"\n{'=' * 62}\n  COMBO RESULTS (test split)\n{'=' * 62}")
    print(f"  {'run':<20}{'mAP50':>8}{'mAP50-95':>10}{'d_sqrt':>9}{'h':>6}")
    print("  " + "-" * 53)
    for r in sorted(summary, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = "—"
        if r["test_map5095"] == r["test_map5095"]:
            d = f"{(r['test_map5095'] - ref) * 100:+.2f}"
        hrs = f"{r['hours']:.1f}" if r.get("hours") == r.get("hours") else "n/a"
        print(f"  {r['name']:<20}{pc(r['test_map50']):>8}{pc(r['test_map5095']):>10}{d:>9}{hrs:>6}")
        if r.get("error"):
            print(f"      FAILED: {r['error']}")

    print(f"\n  Known single-mechanism refs (test mAP50-95):")
    print(f"    baseline      {BASELINE_TEST_MAP5095 * 100:.2f}")
    print(f"    SWA-sqrt      {SWA_SQRT_REF * 100:.2f}  <- d_sqrt is measured vs THIS")
    print(f"    NWD           57.75")
    print(f"    DFL-entropy   57.71")
    print(f"  A combo only 'wins' if it clears the best single (57.86).")


def main():
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 76}")
    print(f"  COMBO sqrt+NWD+entropy  @{IMG_SIZE}px, {EPOCHS}ep, batch {BATCH}")
    print(f"  runs: {', '.join(r['name'] for r in todo)}")
    print(f"{'=' * 76}")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    summary = load_summary()
    done = {r["name"] for r in summary}

    for rc in todo:
        if not only and already_done(rc["name"], summary):
            print(f"\n  [SKIP] {rc['name']} already completed.")
            continue
        try:
            res = run_one(rc)
        except Exception as e:
            print(f"\n  [ERROR] {rc['name']} failed: {e}")
            res = {"name": rc["name"], "hours": float("nan"),
                   "val_map50": float("nan"), "val_map5095": float("nan"),
                   "test_map50": float("nan"), "test_map5095": float("nan"),
                   "error": str(e)}
        summary = [r for r in summary if r["name"] != res["name"]]
        summary.append(res)
        done.add(res["name"])
        with open(out_path, "w") as f:            # incremental, crash-safe
            json.dump(summary, f, indent=2)

    summarise(summary)


if __name__ == "__main__":
    main()
