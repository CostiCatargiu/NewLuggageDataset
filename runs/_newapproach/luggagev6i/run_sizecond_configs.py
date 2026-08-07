#!/usr/bin/env python3
"""
SIZE-CONDITIONAL LB-TAL (the F9 mechanism) + the justified combos.

=============================================================================
WHY — straight from the footprint diagnostic (F9)
=============================================================================
The 4-pass footprint proved ONE GLOBAL BUDGET CANNOT SERVE ALL THREE SIZES:
  * small GTs peak at P3=4 and are P4-SUPPLY-limited (only 2.46 candidates),
  * large GTs need a LOW P3 (501 P3 candidates -> any P3 budget forces junk
    stride-8 positives that collapse large IoU 0.876->0.720).
p4wide {8:4,16:7,32:1} is the best single global 3-tuple (cmb_p4wide: overall
55.60, small 51.15 = +1.17, the project's best small AP). F9 states verbatim
that a genuinely SIZE-CONDITIONAL per-level budget "would DOMINATE any single
global 3-tuple". This runs exactly that mechanism.

size_cond gives EACH GT a per-level budget chosen by its OWN max-side size:
    small  -> {8:5,16:4,32:1}   fine-heavy (peak P3 + all its P4 supply)
    medium -> {8:4,16:6,32:1}   balanced
    large  -> {8:1,16:7,32:2}   coarse-heavy (minimal P3 -> no junk stride-8)
Thresholds small<48<=medium<=96<large px, matching the footprint convention.

=============================================================================
THE RUNS (ordered by expected payoff)
=============================================================================
  1. cmb_sizecond   sqrt0703 + size_cond (default F9 budgets) — THE headline
                    candidate. If F9 is right it beats cmb_p4wide (small 51.15).
  2. lb_sizecond    size_cond on the all-off base — ATTRIBUTION for #1 (is the
                    win the allocation, or the sqrt combination?). Also: does it
                    beat lb_p4wide (small 50.97) / lb_uniform (50.85) alone?
  3. cmb_sizecond_aggr  aggressive small budget {8:6,16:3,32:1} for small,
                    large {8:1,16:8,32:1} — pushes the size split harder.
  4. cmb_p4wide_clsswa  cmb_p4wide + cls_swa 1.75 — the THIRD axis (cls/scoring)
                    stacked on the current best. Fills the empty scoring stage;
                    cls_swa alone was +0.41 small and did NOT fail (unlike qfl).
  5. cmb_sizecond_clsswa  size_cond + sqrt + cls_swa — all three axes on the
                    F9 allocation. The full stack, run last (needs #1 to work).
  6. cmb_p4wide_seed1   seed-1 confirm of the current best, so the new configs
                    are compared against a SEED-VERIFIED baseline not a lucky one.

Baselines: cmb_p4wide 55.60 / small 51.15 (current best), lb_uniform 55.57,
SWA sqrt0703 55.64, anchor 54.77. Seed noise 0.12 overall / 2.06 large.
READ small mAP (CocoEvalAllFolders_luggage.py) — the whole point is small.

VERIFY FIRST (5 min, no training): diag_anchor_footprint.py with the size_cond
assigner, confirm small GTs draw ~5/4/1 and large GTs ~1/7/2, i.e. large's
stride-8 count drops toward the stock 0.16 while small's P3 stays ~4-5.

REQUIRES lossv2updated.py (with mode='size_cond' + set_gt_sizes) installed as
ultralytics/utils/loss.py; lbtal_size_* whitelisted in default.yaml.

Usage:
    python run_sizecond_configs.py
    python run_sizecond_configs.py cmb_sizecond lb_sizecond
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
PROJECT_DIR = "runs_sizecond"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
BASELINE_TEST_MAP5095 = 0.5477
CURRENT_BEST = 0.5560   # cmb_p4wide

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

_SQRT0703 = dict(
    _ALL_OFF,
    area_weight_mode="sqrt",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)


def _lb_sizecond(base, size_budgets=None, quality_gate=0.0, cls_swa=None):
    cfg = dict(base, use_lbtal=True, lbtal_mode="size_cond",
               lbtal_quality_gate=quality_gate)
    if size_budgets is not None:
        cfg["lbtal_size_budgets"] = size_budgets
    if cls_swa is not None:
        cfg["use_cls_swa"] = True
        cfg["cls_swa_boost"] = cls_swa
        cfg["small_obj_px"] = cfg.get("small_obj_px", 48) or 48
    return cfg


def _sq_p4wide(cls_swa=None):
    cfg = dict(_SQRT0703, use_lbtal=True, lbtal_mode="fixed",
               lbtal_level_topk={8: 4, 16: 7, 32: 1})
    if cls_swa is not None:
        cfg["use_cls_swa"] = True
        cfg["cls_swa_boost"] = cls_swa
    return cfg


_AGGR = {"small": {8: 6, 16: 3, 32: 1},
         "medium": {8: 4, 16: 6, 32: 1},
         "large": {8: 1, 16: 8, 32: 1}}

RUNS = [
    {"name": "cmb_sizecond", "batch": BATCH,
     "label": "sqrt0703 + size_cond (F9 default budgets) — the headline candidate",
     "params": _lb_sizecond(_SQRT0703)},

    {"name": "lb_sizecond", "batch": BATCH,
     "label": "size_cond on all-off base — attribution + vs lb_p4wide/lb_uniform",
     "params": _lb_sizecond(_ALL_OFF)},

    {"name": "cmb_sizecond_aggr", "batch": BATCH,
     "label": "sqrt0703 + size_cond aggressive {s:6/3/1, l:1/8/1} — harder size split",
     "params": _lb_sizecond(_SQRT0703, size_budgets=_AGGR)},

    {"name": "cmb_p4wide_clsswa", "batch": BATCH,
     "label": "cmb_p4wide + cls_swa 1.75 — third axis (scoring) on the current best",
     "params": _sq_p4wide(cls_swa=1.75)},

    {"name": "cmb_sizecond_clsswa", "batch": BATCH,
     "label": "sqrt0703 + size_cond + cls_swa 1.75 — full 3-axis stack on F9 alloc",
     "params": _lb_sizecond(_SQRT0703, cls_swa=1.75)},

    {"name": "cmb_p4wide_seed1", "batch": BATCH, "seed": 1,
     "label": "cmb_p4wide SEED 1 — confirm the current best before comparing",
     "params": _sq_p4wide()},
]


def loss_provenance():
    info = {"path": None, "md5": None, "has_size_cond": False, "has_set_gt_sizes": False}
    try:
        import ultralytics.utils.loss as _lm
        path = getattr(_lm, "__file__", None)
        info["path"] = path
        if path and os.path.exists(path):
            src = open(path, "rb").read()
            info["md5"] = hashlib.md5(src).hexdigest()[:12]
            txt = src.decode("utf-8", "ignore")
            info["has_size_cond"] = "size_cond" in txt
            info["has_set_gt_sizes"] = "set_gt_sizes" in txt
    except Exception as e:
        info["error"] = str(e)
    return info


LOSS_INFO = loss_provenance()


def preflight():
    print(f"  loss.py: {LOSS_INFO.get('path')}")
    print(f"  md5={LOSS_INFO.get('md5')}  size_cond={LOSS_INFO.get('has_size_cond')} "
          f"set_gt_sizes={LOSS_INFO.get('has_set_gt_sizes')}")
    missing = [k for k in ("has_size_cond", "has_set_gt_sizes") if not LOSS_INFO.get(k)]
    if missing:
        print(f"\n  [ABORT] installed loss.py missing: {', '.join(missing)}")
        print("  Install the updated lossv2updated.py as ultralytics/utils/loss.py.")
        return False
    return True


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
    seed = rc.get("seed", 0)
    print(f"\n{'=' * 76}\n  RUN {name}  (seed {seed})\n  {rc['label']}\n"
          f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch, "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")
    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "seed": seed, "batch": batch, "hours": hours,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
            data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    print(f"\n{'=' * 76}\n  SIZE-CONDITIONAL RESULTS (test)\n{'=' * 76}")
    print(f"{'run':<22}{'seed':>5}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'vs_best':>9}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = ("%+9.2f" % ((r["test_map5095"] - BASELINE_TEST_MAP5095) * 100))
        vb = ("%+8.2f" % ((r["test_map5095"] - CURRENT_BEST) * 100))
        print(f"{r['name']:<22}{r['seed']:>5}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{d}{vb}{r['hours']:>6.1f}")
    print(f"\n  anchor {BASELINE_TEST_MAP5095*100:.2f} | current best cmb_p4wide {CURRENT_BEST*100:.2f}")
    print("  vs_best > 0 -> beats cmb_p4wide on OVERALL. But the target is SMALL mAP:")
    print("  run CocoEvalAllFolders_luggage.py on best.pt (cmb_p4wide small = 51.15).")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 76}\n  SIZE-CONDITIONAL LB-TAL (F9) + combos — {len(todo)} runs")
    print(f"  target: beat cmb_p4wide (overall 55.60, small 51.15)")
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
            res.append({"name": r["name"], "seed": r.get("seed", 0), "batch": r["batch"],
                        "hours": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
