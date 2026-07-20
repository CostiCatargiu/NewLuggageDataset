#!/usr/bin/env python3
"""
Round 10 — loss SHAPE + target perturbation (7 trainings).

CONTEXT (R9 + R9b = 10 mechanism runs, all null on mAP50-95):
  40+ runs proved that gradient RESCALING is null — area weighting, IARW,
  DFL boost, NWD blend, clipping, TAL tuning all land within ±0.5pt of anchor.
  The model converges to the same optimum regardless of weighting path.

  Diagnosed: mAP50=0.827 vs mAP50-95=0.574 -> 0.253pt gap. Boxes are "roughly
  right" (IoU≥0.50) but not "precisely right" (IoU≥0.75+). This is the prize.

INSIGHT FOR R10: every prior mechanism changed HOW MUCH the model learns from
  each sample, not WHAT it learns. The loss SURFACE itself has a fundamental
  property: CIoU gradients are nearly UNIFORM across IoU range. The model has
  no extra incentive to push from IoU=0.7 to IoU=0.9 vs from IoU=0.3 to
  IoU=0.5. R10 attacks this with mechanisms that change the loss SHAPE or the
  regression TARGETS themselves.

NEW MECHANISMS:

  r10_focal_iou    [Section G, NEW-10] Focal-IoU loss.
                    loss = IoU^gamma * (1 - IoU). Amplifies gradient at HIGH IoU.
                    A box at IoU=0.8 gets 0.8^gamma more gradient than at IoU=0.2
                    (which gets 0.2^gamma). The model is REWARDED MORE for
                    refining already-good boxes. Directly targets the mAP50 vs
                    mAP50-95 gap. gamma=0 -> stock CIoU.

  r10_wise_iou     [Section H, NEW-11] Wise-IoU outlier-aware gain.
                    exp(loss/mean_loss - 1) per sample. Down-weights noisy GT
                    and annotation outliers that the model wastes capacity on.
                    Focuses learning on "clean, learnable" examples.

  r10_jitter       [Section I, NEW-12] Box jitter — label smoothing for regression.
                    Randomly perturb GT box edges ±3% during training. Prevents
                    overfitting to exact GT coordinates. Annealed to zero.

  r10_focal_wise   Focal-IoU + Wise-IoU. The loss shape (focal) changes WHAT
                    IoU range is rewarded; the outlier gain (wise) changes WHICH
                    samples the model focuses on. Orthogonal.

  r10_focal_jitter Focal-IoU + box jitter. Loss shape + target regularization.
                    These attack different axes: focal changes the objective
                    function, jitter changes the targets.

  r10_all3         All three: focal_iou + wise_iou + jitter. If singles show
                    signal, this tests composition.

REQUIRES: loss.py with NEW-10 (focal_iou_gamma), NEW-11 (wise_iou),
  NEW-12 (box_jitter, box_jitter_anneal) whitelisted in cfg patch.

VERIFY AT LAUNCH (config banner):
  r10_anchor       : focal_iou_gamma=0 | wise_iou=0 | box_jitter=0
  r10_focal_iou    : focal_iou_gamma=1.0
  r10_focal_iou_lo : focal_iou_gamma=0.5
  r10_wise_iou     : wise_iou=1
  r10_jitter       : box_jitter=0.03, anneal
  r10_focal_wise   : focal_iou_gamma=1.0, wise_iou=1
  r10_focal_jitter : focal_iou_gamma=1.0, box_jitter=0.03
  r10_all3         : focal_iou_gamma=1.0, wise_iou=1, box_jitter=0.03

DECISION RULE (same as prior rounds, fixed before any eval):
  Candidate iff val mAP50-95 > anchor + 0.5 OR val AP50-95_small > anchor + 0.8.

Usage:
  python run_newluggage_ablation9.py                 # all runs not yet completed
  python run_newluggage_ablation9.py r10_focal_iou   # only named run(s)
  python run_newluggage_ablation9.py --with-test     # also eval test split (discouraged)
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
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_boost=1.0,
    # NEW-1/NEW-2: inert at alpha=0 (renorm is identity, area weight unused)
    weight_renorm=1, area_mode="fixed", area_ref_px=64.0, area_gamma=0.5,
    area_w_cap=3.0,
)
_TARGETED_OFF = dict(
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
)
_IARW_OFF = dict(iarw_gamma=0.0)
_FOCAL_OFF = dict(focal_iou_gamma=0.0)
_WISE_OFF = dict(wise_iou=0)
_JITTER_OFF = dict(box_jitter=0.0, box_jitter_anneal=1)
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

# =============================================================================
# RUN CONFIGS — every param explicit, ablation_params.json is ground truth
# =============================================================================

_ALL_OFF = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, **_IARW_OFF, **_FOCAL_OFF, **_WISE_OFF, **_JITTER_OFF,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# Fresh anchor — every NEW path inert
R10_ANCHOR = dict(**_ALL_OFF)

# ---------------------------------------------------------------------------
# 1) FOCAL-IoU — amplify gradients at HIGH IoU (Section G, NEW-10)
#    loss = IoU^gamma * (1 - IoU). With gamma=1.0:
#      IoU=0.3 -> focal_weight=0.3 (less gradient for rough boxes)
#      IoU=0.7 -> focal_weight=0.7 (more gradient for decent boxes)
#      IoU=0.9 -> focal_weight=0.9 (strong gradient for refinement)
#    The model is REWARDED MORE for refining already-good boxes than for
#    coarsely correcting bad ones. This is the opposite of what IARW did
#    (which boosted bad boxes). Directly targets the mAP50->mAP50-95 gap.
# ---------------------------------------------------------------------------
R10_FOCAL = dict(**_ALL_OFF, focal_iou_gamma=1.0)

# Conservative: gamma=0.5 (milder focal effect)
R10_FOCAL_LO = dict(**_ALL_OFF, focal_iou_gamma=0.5)

# ---------------------------------------------------------------------------
# 2) WISE-IoU — outlier-aware gradient gain (Section H, NEW-11)
#    exp(loss/mean_loss - 1) per sample. At the batch mean: gain=1.0.
#    Easy examples (loss < mean): gain < 1 -> slightly down-weighted.
#    Hard/noisy examples (loss >> mean): gain > 1 but with diminishing
#    returns (exp sublinear vs raw ratio). Focuses on "learnable" samples.
# ---------------------------------------------------------------------------
R10_WISE = dict(**_ALL_OFF, wise_iou=1)

# ---------------------------------------------------------------------------
# 3) BOX JITTER — label smoothing for regression (Section I, NEW-12)
#    ±3% random perturbation on each GT box edge after TAL assignment.
#    Prevents overfitting to exact GT coordinates. Annealed: full jitter
#    early -> zero at end. Like dropout but for regression targets.
# ---------------------------------------------------------------------------
R10_JITTER = dict(**_ALL_OFF, box_jitter=0.03, box_jitter_anneal=1)

# ---------------------------------------------------------------------------
# 4) FOCAL + WISE — loss shape + outlier awareness
#    Focal changes WHAT IoU range is rewarded; Wise changes WHICH samples
#    the model focuses on. Orthogonal axes.
# ---------------------------------------------------------------------------
R10_FOCAL_WISE = dict(**_ALL_OFF, focal_iou_gamma=1.0, wise_iou=1)

# ---------------------------------------------------------------------------
# 5) FOCAL + JITTER — loss shape + target regularization
#    Focal changes the objective function; jitter changes the targets.
#    Different layers of the training pipeline.
# ---------------------------------------------------------------------------
R10_FOCAL_JITTER = dict(**_ALL_OFF, focal_iou_gamma=1.0, box_jitter=0.03, box_jitter_anneal=1)

# ---------------------------------------------------------------------------
# 6) ALL THREE — focal + wise + jitter
#    If singles show signal, test full composition.
# ---------------------------------------------------------------------------
R10_ALL3 = dict(**_ALL_OFF, focal_iou_gamma=1.0, wise_iou=1, box_jitter=0.03, box_jitter_anneal=1)

# =============================================================================
# RUNS TO EXECUTE
# =============================================================================
RUNS = [
    {"name": "r10_anchor",       "phase": "-", "label": "Fresh anchor — all NEW paths inert",                      "params": R10_ANCHOR},
    {"name": "r10_focal_iou",    "phase": "G", "label": "Focal-IoU gamma=1.0 — amplify high-IoU gradients",        "params": R10_FOCAL},
    {"name": "r10_focal_iou_lo", "phase": "G", "label": "Focal-IoU gamma=0.5 — conservative variant",              "params": R10_FOCAL_LO},
    {"name": "r10_wise_iou",     "phase": "H", "label": "Wise-IoU — outlier-aware gradient gain",                  "params": R10_WISE},
    {"name": "r10_jitter",       "phase": "I", "label": "Box jitter 3% annealed — regression label smoothing",     "params": R10_JITTER},
    {"name": "r10_focal_wise",   "phase": "G+H","label": "Focal-IoU 1.0 + Wise-IoU — shape + outlier awareness",   "params": R10_FOCAL_WISE},
    {"name": "r10_focal_jitter", "phase": "G+I","label": "Focal-IoU 1.0 + jitter 3% — shape + target reg",         "params": R10_FOCAL_JITTER},
    {"name": "r10_all3",         "phase": "GHI","label": "Focal-IoU + Wise-IoU + jitter — full composition",        "params": R10_ALL3},
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
    print("  ROUND 10 — Focal-IoU + Wise-IoU + box jitter (loss shape + targets)")
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

    anchor = next((r for r in summary if r["name"] == "r10_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>4}"
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
    print("  Candidates AND the anchor -> seeds 1,2 before any conclusion;"
          " test eval once, at the end, on multi-seed survivors only.")


if __name__ == "__main__":
    main()