#!/usr/bin/env python3
"""
OVERNIGHT TUNING — push the two v6i winners further + confirm + combine.

=============================================================================
STARTING POINT (isolated v6i results, s48/l128, anchor = 54.77)
=============================================================================
  LB-TAL uniform  : 55.57 (+0.80 overall, +0.87 SMALL, AR50_small 0.951->0.960)
  SWA sqrt0703    : 55.64 (+0.87 overall, +0.65 small, gain mostly in noisy large)

Both are single-seed. LB-TAL showed a MONOTONIC dose-response: flatter per-level
allocation -> bigger gain (proportional +0.05 -> fixed442 +0.54 -> uniform +0.80).
The curve has NOT peaked, so "even flatter / coarse-biased" is the natural next
probe. SWA is near its ceiling (9-config sweep, optimum fuzzy), so we only fine-
tune px and CONFIRM it, rather than chase it.

=============================================================================
THE 6 OVERNIGHT RUNS (ordered by expected payoff)
=============================================================================
  1. cmb_lbU_swa0703   COMBINE the two winners (assignment + box). Different loss
                       stages -> stackable. Highest-value: could beat both.
  2. lb_coarse_244     LB-TAL fixed {8:2,16:4,32:4} — deliberately coarse-biased,
                       extends the "flatter wins" trend PAST uniform.
  3. lb_uniform_mk2    LB-TAL uniform + min_level_k=2 — guarantees coarse levels
                       even more budget (flatter-still, cleaner than fixed).
  4. lb_uniform_tk13   LB-TAL uniform + tal_topk=13 — more absolute budget/level
                       while keeping the equal split.
  5. lb_uniform_seed1  SEED CONFIRM the current LB-TAL winner (settle +0.80).
  6. swa0703_px44      SWA sqrt0.7->0.3 b2.0 with small_obj_px=44 (~true mean
                       object side ~46px) — the one untested SWA knob worth a look.

READ per-SIZE buckets (CocoEvalAllFolders_luggage.py on best.pt), esp. SMALL
mAP and AR50_small — that is where the real effect lives. Overall mAP can be flat
while small moves.

VERIFY FIRST: python selftest_lbtal.py  (all PASS) before launching LB-TAL runs.

REQUIRES lossv2updated.py installed as ultralytics/utils/loss.py, with the
use_lbtal/lbtal_* and SWA keys whitelisted in cfg/default.yaml.

Usage:
    python run_overnight_tune.py              # all 6, in order
    python run_overnight_tune.py cmb_lbU_swa0703 lb_coarse_244   # a subset
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
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_overnight_tune"

EPOCHS = 70
IMG_SIZE = 640            # eval MUST also be 640 (the 896 lesson)
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100

BASELINE_TEST_MAP5095 = 0.5477   # v6i pure-stock anchor (ms_s)

# =============================================================================
# Everything-off base (identical to run_lbtal_isolated.py — pure stock).
# =============================================================================
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

# SWA sqrt0703 winner block (box-side). px defaults to 48 (the winning value).
_SWA0703 = dict(
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    area_weight_mode="sqrt", small_obj_boost=2.0, small_obj_px=48,
)


def _combo(**overrides):
    return dict(_ALL_OFF, **overrides)


# =============================================================================
# THE 6 RUNS
# =============================================================================
RUNS = [
    # 1) COMBINE the two winners: LB-TAL uniform (assignment) + SWA sqrt0703 (box).
    #    Different loss stages -> stackable. Highest expected payoff.
    {"name": "cmb_lbU_swa0703", "batch": BATCH, "seed": 0,
     "label": "COMBO LB-TAL uniform + SWA sqrt0.7->0.3 b2.0 px48 (assignment+box)",
     "params": _combo(
         use_lbtal=True, lbtal_mode="uniform", lbtal_level_topk=None, lbtal_min_level_k=1,
         **_SWA0703)},

    # 2) LB-TAL coarse-biased {8:2,16:4,32:4} — push PAST uniform along the
    #    "flatter/coarser wins" trend the dose-response showed.
    {"name": "lb_coarse_244", "batch": BATCH, "seed": 0,
     "label": "LB-TAL fixed {8:2,16:4,32:4} — coarse-biased, extends flatter-wins trend",
     "params": _combo(use_lbtal=True, lbtal_mode="fixed",
                      lbtal_level_topk={8: 2, 16: 4, 32: 4}, lbtal_min_level_k=1)},

    # 3) LB-TAL uniform + min_level_k=2 — guarantee coarse levels more budget
    #    (flatter-still, cleaner than a hand-fixed dict).
    {"name": "lb_uniform_mk2", "batch": BATCH, "seed": 0,
     "label": "LB-TAL uniform + min_level_k=2 — floor coarse-level budget higher",
     "params": _combo(use_lbtal=True, lbtal_mode="uniform",
                      lbtal_level_topk=None, lbtal_min_level_k=2)},

    # 4) LB-TAL uniform + tal_topk=13 — more absolute budget per level, equal split.
    {"name": "lb_uniform_tk13", "batch": BATCH, "seed": 0,
     "label": "LB-TAL uniform + tal_topk=13 — larger per-level budget",
     "params": _combo(use_lbtal=True, lbtal_mode="uniform",
                      lbtal_level_topk=None, lbtal_min_level_k=1, tal_topk=13)},

    # 5) SEED CONFIRM the current LB-TAL winner (uniform, seed 1) — settle +0.80.
    {"name": "lb_uniform_seed1", "batch": BATCH, "seed": 1,
     "label": "LB-TAL uniform — SEED 1 confirmation of the +0.80 winner",
     "params": _combo(use_lbtal=True, lbtal_mode="uniform",
                      lbtal_level_topk=None, lbtal_min_level_k=1)},

    # 6) SWA px fine-tune: px=44 (~true mean object side ~46px). The one untested
    #    SWA knob worth a look; SWA is otherwise near its ceiling.
    {"name": "swa0703_px44", "batch": BATCH, "seed": 0,
     "label": "SWA sqrt0.7->0.3 b2.0 px44 — align 'small' to true mean object size",
     "params": _combo(
         alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
         area_weight_mode="sqrt", small_obj_boost=2.0, small_obj_px=44)},
]


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
    name, params, batch, seed = rc["name"], rc["params"], rc["batch"], rc["seed"]
    print(f"\n{'=' * 76}\n  RUN {name}  (seed {seed})\n  {rc['label']}\n"
          f"  model={MODEL_WEIGHTS}  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")

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
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch,
                       "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "seed": seed, "batch": batch, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
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
    ref = BASELINE_TEST_MAP5095
    print(f"\n{'=' * 76}\n  OVERNIGHT TUNING RESULTS (test split)\n{'=' * 76}")
    print(f"{'run':<20}{'seed':>5}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = ("%+10.2f" % ((r["test_map5095"] - ref) * 100)) if ref else "%10s" % "-"
        print(f"{r['name']:<20}{r['seed']:>5}{r['test_map50'] * 100:>9.2f}"
              f"{r['test_map5095'] * 100:>11.2f}{d}{r['hours']:>6.1f}")
    print(f"\n  anchor = {ref * 100:.2f} | LB-TAL uniform = 55.57 | SWA sqrt0703 = 55.64")
    print("  Then read per-SIZE buckets (CocoEvalAllFolders_luggage.py) — small mAP")
    print("  and AR50_small are where the real effect lives.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]

    print(f"\n{'=' * 76}")
    print(f"  OVERNIGHT TUNING  @{IMG_SIZE}px, {EPOCHS}ep, yolov12s")
    print(f"  anchor {BASELINE_TEST_MAP5095 * 100:.2f} | tuning LB-TAL uniform + SWA sqrt0703")
    print(f"  runs ({len(todo)}): {', '.join(r['name'] for r in todo)}")
    print(f"  est. ~{1.5 * len(todo):.0f} GPU-h total")
    print(f"{'=' * 76}\n")
    print("  REMINDER: run  python selftest_lbtal.py  first — all checks must PASS.\n")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")

    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "seed": r["seed"], "batch": r["batch"],
                        "hours": float("nan"), "val_map50": float("nan"),
                        "val_map5095": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

    summarise(res, out_path)
