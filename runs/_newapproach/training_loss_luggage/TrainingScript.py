# train_loss_ablation.py
"""
Ablation study for loss4.py — luggage-specific loss enhancements on YOLOv12s.

loss4.py defaults to stock Ultralytics loss (all features OFF). Each run enables
one mechanism at a time so you can measure its exact delta. The two experimental
features (AR penalty, Inner-IoU) each branch off the SIZE_BOOST config
independently, so a regression can be attributed to the right one.

  Run 0: baseline        — stock loss, zero custom features
  Run 1: class_balance   — + class weights (sqrt inverse-frequency)
  Run 2: nwd_blend       — + NWD/CIoU blend (iou_ratio=0.5)
  Run 3: size_boost      — + size-adaptive IoU/DFL/cls boost
  Run 4: ar_penalty      — Run 3 + AR penalty ONLY   (isolated)
  Run 5: inner_iou       — Run 3 + Inner-IoU ONLY    (isolated)
  Run 6: full            — Run 3 + AR penalty + Inner-IoU

Same architecture, config, and seed across runs; only loss params change.

IMPORTANT — this script assumes loss4.py has REPLACED ultralytics/utils/loss.py
(loss4.py uses relative imports `from .metrics import ...`, so it must live inside
the ultralytics.utils package). The preflight below verifies the custom loss is
actually active and aborts if it isn't, so you never silently train stock loss.
"""

import os
import json
import math
import time
import statistics

import torch

# ============================================================
# STEP 0 — REGISTER CUSTOM LOSS KEYS WITH ULTRALYTICS
# ------------------------------------------------------------
# Ultralytics rejects unknown train() kwargs with a SyntaxError
# ("'x' is not a valid YOLO argument"). We register our custom keys
# with their NEUTRAL (stock-equivalent) defaults so that:
#   (a) train() accepts them, and
#   (b) any run that omits a key inherits the no-op default.
# This MUST run before any get_cfg / YOLO(...) call.
# ============================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_LOSS_DEFAULTS = {
    # classification
    "class_weights": None,            # list[nc] | None (None -> uniform, stock)
    "normalize_class_weights": True,
    "use_vfl": False,
    "vfl_alpha": 0.75,
    "vfl_gamma": 2.0,
    "small_obj_cls_boost": 1.0,       # 1.0 -> off
    # box regression
    "iou_ratio": 1.0,                 # 1.0 -> pure CIoU (stock)
    "nwd_c": 3.0,
    "small_obj_boost": 1.0,           # 1.0 -> off
    "small_obj_area_thresh": 36.0,
    "use_inner_iou": False,
    "inner_iou_ratio_small": 0.7,
    "inner_iou_ratio_large": 1.0,
    "use_ar_penalty": False,
    "ar_penalty_lambda": 0.05,
    "ar_penalty_tall_extra": 0.5,
    "ar_penalty_max": 1.0,
}

for _k, _v in _CUSTOM_LOSS_DEFAULTS.items():
    DEFAULT_CFG_DICT.setdefault(_k, _v)
    if not hasattr(DEFAULT_CFG, _k):
        setattr(DEFAULT_CFG, _k, _v)

from ultralytics import YOLO  # noqa: E402  (import after key registration)
from ultralytics.cfg import get_cfg  # noqa: E402


# ============================================================
# PATHS — UPDATE THESE FOR YOUR MACHINE
# ============================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubsetNEW/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_loss4_ablation"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ============================================================
# FIXED TRAINING PARAMS (same for ALL runs)
# ============================================================
EPOCHS = 80
IMGSZ = 640
BATCH = 58
WORKERS = 8
PATIENCE = 20
CLOSE_MOSAIC = 10

# Seeds: single seed is fine for a first pass, but mAP run-to-run variance on a
# subset can be ±0.5-1%, which may exceed the effect of subtle features (NWD).
# For publishable deltas, use e.g. [42, 1, 2] and read the mean±std summary.
SEEDS = [42]

# Skip a run if its best.pt already exists (resume-friendly across crashes).
RESUME_IF_DONE = True

# Which split to evaluate for the ablation comparison. Use 'val' during the sweep
# (consistent with early stopping); evaluate the winner on 'test' separately.
EVAL_SPLIT = "val"

# ============================================================
# YOLOv12s ORIGINAL ARCHITECTURE (nc=3 for luggage)
# ============================================================
ARCH_YOLOV12S = """# YOLOv12s original — loss4 ablation study
nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

# ============================================================
# PRE-COMPUTE class weights (sqrt inverse-frequency, mean-normalised)
#   _COUNTS MUST be in data.yaml `names:` order (NOT alphabetical).
#   These are full-split counts; only ratios matter for weighting, and ratios
#   are preserved under (stratified) subsampling, so the 30% subset is fine.
#   backpack=11491, bag=9490, trolley=21557
# ============================================================
_COUNTS = [11491.0, 9490.0, 21557.0]
_NC = len(_COUNTS)
_TOTAL = sum(_COUNTS)
_INV_FREQ = [_TOTAL / (c * _NC) for c in _COUNTS]
_SQRT_INV = [math.sqrt(w) for w in _INV_FREQ]
_MEAN_SQRT = sum(_SQRT_INV) / _NC
CLASS_WEIGHTS_SQRT = [round(w / _MEAN_SQRT, 4) for w in _SQRT_INV]
# Result: ~[1.0599, 1.1664, 0.7738]

# ============================================================
# LOSS CONFIGURATIONS
#   Runs 0-3 are a strict additive stack.
#   Runs 4 and 5 each add ONE experimental feature onto Run 3 (isolated).
#   Run 6 combines both. loss4 defaults are no-ops, so omitted keys stay stock.
# ============================================================
_THROUGH_SIZE_BOOST = {  # everything up to and including Run 3
    "class_weights": CLASS_WEIGHTS_SQRT,
    "normalize_class_weights": True,
    "iou_ratio": 0.5,
    "nwd_c": 3.0,
    "small_obj_boost": 1.5,
    "small_obj_cls_boost": 1.3,
    "small_obj_area_thresh": 36.0,
}
_AR_PARAMS = {
    "use_ar_penalty": True,
    "ar_penalty_lambda": 0.05,
    "ar_penalty_tall_extra": 0.5,
    "ar_penalty_max": 1.0,
}
_INNER_PARAMS = {
    "use_inner_iou": True,
    "inner_iou_ratio_small": 0.7,
    "inner_iou_ratio_large": 1.0,
}

LOSS_CONFIGS = [
    {"name": "run0_baseline",
     "description": "Stock YOLOv12s loss — all loss4 features OFF",
     "params": {}},

    {"name": "run1_class_balance",
     "description": "Class-balance weights only (sqrt inverse-frequency)",
     "params": {"class_weights": CLASS_WEIGHTS_SQRT, "normalize_class_weights": True}},

    {"name": "run2_nwd_blend",
     "description": "Class-balance + NWD/CIoU blend (iou_ratio=0.5)",
     "params": {"class_weights": CLASS_WEIGHTS_SQRT, "normalize_class_weights": True,
                "iou_ratio": 0.5, "nwd_c": 3.0}},

    {"name": "run3_size_boost",
     "description": "Class-balance + NWD + size-adaptive boost (IoU/DFL 1.5x, cls 1.3x)",
     "params": dict(_THROUGH_SIZE_BOOST)},

    {"name": "run4_ar_penalty",
     "description": "Run 3 + aspect-ratio penalty ONLY (isolated experimental)",
     "params": {**_THROUGH_SIZE_BOOST, **_AR_PARAMS}},

    {"name": "run5_inner_iou",
     "description": "Run 3 + Inner-IoU ONLY (isolated experimental)",
     "params": {**_THROUGH_SIZE_BOOST, **_INNER_PARAMS}},

    {"name": "run6_full",
     "description": "Run 3 + AR penalty + Inner-IoU (everything on)",
     "params": {**_THROUGH_SIZE_BOOST, **_AR_PARAMS, **_INNER_PARAMS}},
]


# ============================================================
# PREFLIGHT CHECKS
# ============================================================
def preflight():
    """Fail fast BEFORE launching hours of training if something is misconfigured."""
    print("\n[preflight] checking configuration ...")

    # 1) Custom keys are accepted by the config system.
    try:
        cfg = get_cfg(overrides={"iou_ratio": 0.5, "small_obj_boost": 1.5,
                                 "class_weights": CLASS_WEIGHTS_SQRT})
        assert getattr(cfg, "iou_ratio", None) == 0.5
        print("[preflight] OK  custom loss keys accepted by get_cfg")
    except Exception as e:
        raise SystemExit(f"[preflight] FAIL  custom keys rejected: {e}\n"
                         f"  -> the STEP 0 registration block did not take effect.")

    # 2) The ACTIVE loss module is loss4 (not stock). loss4's v8DetectionLoss has a
    #    _compute_cls_loss method and BboxLoss has an iou_ratio attr; stock has neither.
    from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
    is_custom = hasattr(v8DetectionLoss, "_compute_cls_loss") and "iou_ratio" in (
        BboxLoss.__init__.__code__.co_names + BboxLoss.__init__.__code__.co_varnames
    )
    if not is_custom:
        raise SystemExit(
            "[preflight] FAIL  ultralytics.utils.loss is the STOCK loss, not loss4.\n"
            "  -> Replace ultralytics/utils/loss.py with loss4.py (it uses relative\n"
            "     imports, so it must live inside the ultralytics.utils package).\n"
            "  Without this, every run would silently train identical stock loss.")
    print("[preflight] OK  active loss module is loss4 (custom)")

    # 3) Dataset yaml exists.
    if not os.path.isfile(DATA_YAML):
        raise SystemExit(f"[preflight] FAIL  DATA_YAML not found: {DATA_YAML}")
    print("[preflight] OK  dataset yaml found")
    print("[preflight] class order reminder: _COUNTS must match data.yaml names[]:")
    print(f"            {list(zip(['idx0','idx1','idx2'], _COUNTS))}  weights={CLASS_WEIGHTS_SQRT}\n")


# ============================================================
# HELPERS
# ============================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)


def extract_metrics(m):
    """Pull overall + per-class AP out of an Ultralytics DetMetrics object, robustly."""
    out = {}
    try:
        out["mAP50"] = float(m.box.map50)
        out["mAP50_95"] = float(m.box.map)
        out["precision"] = float(m.box.mp)
        out["recall"] = float(m.box.mr)
    except Exception as e:
        out["overall_error"] = str(e)
    try:
        names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
        per = {}
        for i, c in enumerate(list(m.box.ap_class_index)):
            per[names[int(c)]] = {"AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])}
        out["per_class"] = per
    except Exception as e:
        out["per_class_error"] = str(e)
    return out


def run_one(yaml_path, run_cfg, seed):
    """Train (or resume) one config at one seed, then eval on EVAL_SPLIT. Returns metrics dict."""
    run_name = f"{run_cfg['name']}_seed{seed}"
    run_dir = os.path.join(PROJECT_DIR, run_name)
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs(run_dir, exist_ok=True)
    save_json(run_cfg, os.path.join(run_dir, "loss_config.json"))

    already_done = RESUME_IF_DONE and os.path.isfile(best_pt)
    if already_done:
        print(f"    [resume] best.pt exists — skipping training for {run_name}")
    else:
        model = YOLO(yaml_path)
        model.load(MODEL_WEIGHTS)
        model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            workers=WORKERS, project=PROJECT_DIR, name=run_name, patience=PATIENCE,
            close_mosaic=CLOSE_MOSAIC, seed=seed, deterministic=True, exist_ok=True,
            **run_cfg["params"],
        )

    # Evaluate best weights on the comparison split (same for every run).
    metrics = YOLO(best_pt).val(data=DATA_YAML, imgsz=IMGSZ, split=EVAL_SPLIT,
                                project=PROJECT_DIR, name=f"{run_name}_eval", exist_ok=True)
    result = extract_metrics(metrics)
    save_json(result, os.path.join(run_dir, "metrics.json"))
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    preflight()

    ts = int(time.time())
    yaml_path = os.path.join("ultralytics", "cfg", "models", "v12", "yolov12s_loss4_ablation.yaml")
    save_yaml(ARCH_YOLOV12S, yaml_path)

    print("=" * 70)
    print("  LOSS4 ABLATION — YOLOv12s on Luggage Dataset")
    print("=" * 70)
    print(f"  data={DATA_YAML}")
    print(f"  imgsz={IMGSZ}  epochs={EPOCHS}  batch={BATCH}  seeds={SEEDS}  eval_split={EVAL_SPLIT}")
    print(f"  class_weights(sqrt, mean-norm)={CLASS_WEIGHTS_SQRT}")
    print(f"  configs={len(LOSS_CONFIGS)}  total_runs={len(LOSS_CONFIGS) * len(SEEDS)}")
    print("=" * 70)

    # results[config_name][seed] = metrics dict
    results = {c["name"]: {} for c in LOSS_CONFIGS}

    total = len(LOSS_CONFIGS) * len(SEEDS)
    idx = 0
    for cfg in LOSS_CONFIGS:
        for seed in SEEDS:
            idx += 1
            print(f"\n{'=' * 70}\n  [{idx}/{total}] {cfg['name']}  seed={seed}\n"
                  f"  {cfg['description']}\n{'=' * 70}")
            try:
                results[cfg["name"]][seed] = run_one(yaml_path, cfg, seed)
                m = results[cfg["name"]][seed]
                print(f"    -> mAP50={m.get('mAP50')}  mAP50-95={m.get('mAP50_95')}")
                if "per_class" in m:
                    for cls_name, ap in m["per_class"].items():
                        print(f"       {cls_name:10s} AP50={ap['AP50']:.4f}  AP50-95={ap['AP50_95']:.4f}")
            except Exception as e:
                results[cfg["name"]][seed] = {"error": str(e)}
                print(f"    -> FAILED: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # AGGREGATE + COMPARISON TABLE (mean±std over seeds; delta vs baseline)
    # ------------------------------------------------------------------
    def agg(config_name, key):
        vals = [s.get(key) for s in results[config_name].values() if isinstance(s, dict) and key in s]
        if not vals:
            return None, None
        return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    base_map, _ = agg("run0_baseline", "mAP50_95")
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY (eval on '%s')" % EVAL_SPLIT)
    print("=" * 70)
    header = f"  {'config':22s} {'mAP50':>16s} {'mAP50-95':>16s} {'Δ50-95':>10s}"
    print(header)
    print("  " + "-" * 66)
    summary = {}
    for c in LOSS_CONFIGS:
        name = c["name"]
        map50_m, map50_s = agg(name, "mAP50")
        map_m, map_s = agg(name, "mAP50_95")
        summary[name] = {"mAP50_mean": map50_m, "mAP50_std": map50_s,
                         "mAP50_95_mean": map_m, "mAP50_95_std": map_s,
                         "per_seed": results[name]}
        if map_m is None:
            print(f"  {name:22s} {'FAILED':>16s}")
            continue
        d = "" if base_map is None else f"{(map_m - base_map):+.4f}"
        s50 = f"{map50_m:.4f}±{map50_s:.4f}" if len(SEEDS) > 1 else f"{map50_m:.4f}"
        s95 = f"{map_m:.4f}±{map_s:.4f}" if len(SEEDS) > 1 else f"{map_m:.4f}"
        print(f"  {name:22s} {s50:>16s} {s95:>16s} {d:>10s}")
    print("=" * 70)
    if len(SEEDS) == 1:
        print("  NOTE: single seed — treat small Δ (< ~0.5-1% mAP) as within noise.")
        print("        Set SEEDS=[42,1,2] and re-run for mean±std you can trust.")

    save_json({
        "timestamp": ts,
        "training_config": {"model": "YOLOv12s", "loss_file": "loss4.py", "data": DATA_YAML,
                            "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                            "seeds": SEEDS, "eval_split": EVAL_SPLIT},
        "class_weights_sqrt": CLASS_WEIGHTS_SQRT,
        "configs": {c["name"]: c for c in LOSS_CONFIGS},
        "summary": summary,
    }, os.path.join(PROJECT_DIR, f"ablation_summary_{ts}.json"))
    print(f"\n  Summary saved to {os.path.join(PROJECT_DIR, f'ablation_summary_{ts}.json')}")


if __name__ == "__main__":
    main()