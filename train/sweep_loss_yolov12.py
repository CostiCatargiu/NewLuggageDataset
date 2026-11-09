#!/usr/bin/env python3
"""
YOLOv12-s LOSS GRID SEARCH — Table 6, every combination, stock yolov12s graph, LuggageDataset v6i
Environment A (YOLOv12 fork + ultralytics_patches/v12).

The scale-weighted alignment loss (paper Section 4.1.1). Each positive's box regression
weight blends an inverse-area term with the usual score weight. The blend factor alpha
follows a linear curriculum from alpha_start (epoch 0) to alpha_end (last epoch), and objects
whose longest side is below p_small pixels get an extra boost rho.

THE GRID (Table 6) — every combination below is trained:

    parameter          meaning                       values                               n   paper
    alpha_start        blend at epoch 0              0.6 0.65 0.7 0.75 0.8 0.85 0.9       7   0.7
    alpha_end          blend at the last epoch       0.3 0.35 0.4 0.5                     4   0.3
    area_weight_mode   area exponent                 inv = A^-1, pow075 = A^-3/4,         3   sqrt
                                                     sqrt = A^-1/2
    small_obj_px       p_small, longest side (px)    24 28 30 32 36                       5   32
    small_obj_boost    boost rho below p_small       1.0 1.25 1.5 1.75 2.0 2.25 2.5       7   2.0

    7 x 4 x 3 x 5 x 7 = 2940 combinations, plus one stock-loss reference run = 2941 runs.

alpha_min / alpha_max, the clamp of the curriculum, are always the two endpoints.

ORDER: the stock reference first, then the paper setting, then the rest of the grid. Runs
already finished (status OK in the results JSON) are skipped, so the script can be stopped and
started again and it resumes where it stopped.

COST: about one GPU-hour per run at 70 epochs and batch 54 on one GPU. To run a smaller grid,
delete values from GRID; to run one setting, put it in ONLY.

All runs: stock yolov12s.pt graph, batch 54, 70 epochs, 640 px, seed 0. To run the grid on
the paper's custom graph, set MODEL to configs/yolov12/yolov12s-lsshift.yaml and BATCH to 32.

Usage:
  python sweep_loss_yolov12.py

Results: one folder per run in PROJECT_DIR, and PROJECT_DIR_results.json with the test-split
mAP50 / mAP50-95 of every run (written after every run). The paper selected on mAP@50 of
small objects from the COCO evaluator, original_scripts/CocoEvalAllFolders_luggage.py.
"""

import gc
import itertools
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = str(ROOT / "data" / "luggage_v6i.yaml")  # edit: path to your LuggageDataset v6i data.yaml
MODEL = "yolov12s.pt"  # stock graph, pretrained
WEIGHTS = "yolov12s.pt"  # used only when MODEL is a .yaml
PROJECT_DIR = "runs_loss_yolov12"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 70
BATCH = 54
SEED = 0

# =============================================================================
# THE GRID (Table 6) — every combination is trained
# =============================================================================
GRID = dict(
    alpha_start=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],      # n = 7
    alpha_end=[0.3, 0.35, 0.4, 0.5],                         # n = 4
    area_weight_mode=["inv", "pow075", "sqrt"],              # n = 3   A^-1, A^-3/4, A^-1/2
    small_obj_px=[24, 28, 30, 32, 36],                       # n = 5
    small_obj_boost=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],  # n = 7
)
PAPER = dict(alpha_start=0.7, alpha_end=0.3, area_weight_mode="sqrt", small_obj_px=32, small_obj_boost=2.0)

ONLY = []  # e.g. ["y12_as070_ae030_sqrt_px32_rho200"] to train only these runs; empty = all

# Every custom loss term OFF (= stock YOLOv12 loss). Passed explicitly because the
# patched default.yaml ships the paper's loss settings, not neutral values.
_ALL_OFF = dict(
    small_obj_px=0, small_obj_boost=1.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
)


def num(v):
    """0.65 -> '065', 2.0 -> '200', 32 -> '32' (for run names)."""
    return f"{round(v * 100):03d}" if isinstance(v, float) else str(v)


def swa_run(alpha_start, alpha_end, area_weight_mode, small_obj_px, small_obj_boost):
    name = (f"y12_as{num(alpha_start)}_ae{num(alpha_end)}_{area_weight_mode}"
            f"_px{small_obj_px}_rho{num(small_obj_boost)}")
    desc = (f"alpha {alpha_start}->{alpha_end}, {area_weight_mode}, p_small {small_obj_px}, "
            f"rho {small_obj_boost}")
    params = dict(_ALL_OFF, alpha_start=alpha_start, alpha_end=alpha_end,
                  alpha_min=min(alpha_start, alpha_end), alpha_max=max(alpha_start, alpha_end),
                  area_weight_mode=area_weight_mode, small_obj_px=small_obj_px,
                  small_obj_boost=small_obj_boost)
    return {"name": name, "desc": desc, "params": params}


def build_runs():
    paper = swa_run(**PAPER)
    paper["desc"] = "PAPER: " + paper["desc"]
    runs = [{"name": "y12_stock", "desc": "REFERENCE: stock loss, every custom term off",
             "params": dict(_ALL_OFF)}, paper]
    for combo in itertools.product(*GRID.values()):
        r = swa_run(**dict(zip(GRID, combo)))
        if r["name"] != paper["name"]:
            runs.append(r)
    return runs


RUNS = build_runs()


# =============================================================================
def sync_epoch(trainer):
    """The alpha curriculum reads criterion.epoch; keep it in step with the trainer."""
    m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    m.current_epoch = trainer.epoch
    crit = getattr(m, "criterion", None)
    if crit is not None:
        crit.epoch = trainer.epoch
        if hasattr(crit, "_sync_bbox_loss_state"):
            crit._sync_bbox_loss_state()


def test_eval(save_dir):
    w = Path(save_dir) / "weights" / "best.pt"
    r = YOLO(str(w)).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                         workers=WORKERS, plots=False, verbose=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


def differs(params):
    return {k: v for k, v in params.items() if _ALL_OFF.get(k, "__") != v}


def run_one(run):
    print(f"\n{'#' * 78}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: LOSS   model={MODEL}   batch={BATCH}")
    print(f"# differs from stock: {differs(run['params'])}")
    print(f"{'#' * 78}\n")
    start_time = time.time()
    out = {"name": run["name"], "desc": run["desc"], "params": differs(run["params"]), "status": "OK",
           "test_mAP50": None, "test_mAP50-95": None}
    try:
        model = YOLO(MODEL)
        if MODEL.endswith(".yaml"):
            model.load(WEIGHTS)
        model.add_callback("on_train_epoch_start", sync_epoch)
        model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
            workers=WORKERS, project=PROJECT_DIR, name=run["name"], exist_ok=True,
            patience=100, close_mosaic=10, seed=SEED, deterministic=True,
            **run["params"],
        )
        out["test_mAP50"], out["test_mAP50-95"] = test_eval(model.trainer.save_dir)
    except Exception as e:
        out["status"] = f"FAILED: {e}"
        print(f"\n  FAILED: {run['name']} -- {e}")
    finally:
        out["hours"] = round((time.time() - start_time) / 3600, 2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main():
    out_path = f"{PROJECT_DIR}_results.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {r["name"] for r in results if r["status"] == "OK"}
    results = [r for r in results if r["status"] == "OK"]  # failed runs are retried
    todo = [r for r in RUNS if r["name"] not in done and (not ONLY or r["name"] in ONLY)]

    total_start = time.time()
    print(f"\n{'=' * 78}")
    print(f"  YOLOv12-s LOSS GRID SEARCH — {len(RUNS)} runs ({len(done)} done, {len(todo)} to train)")
    print(f"{'=' * 78}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Model: {MODEL}   Epochs: {EPOCHS}   ImgSize: {IMG_SIZE}   Batch: {BATCH}")
    for k, v in GRID.items():
        print(f"  {k:<18} n={len(v)}  {v}")
    print(f"  paper setting: {PAPER}")
    print(f"{'=' * 78}\n")

    for i, run in enumerate(todo):
        print(f"\n>>> Run {i + 1}/{len(todo)}: {run['name']}")
        results.append(run_one(run))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    ok = sorted((r for r in results if r["status"] == "OK"), key=lambda r: -r["test_mAP50"])
    print(f"\n{'=' * 78}")
    print(f"  DONE ({(time.time() - total_start) / 3600:.2f}h this session)   test split, best 20 by mAP50")
    print(f"{'=' * 78}")
    print(f"  {'run':<38}{'mAP50':>8}{'mAP50-95':>10}")
    for r in ok[:20]:
        print(f"  {r['name']:<38}{r['test_mAP50']:>8.2f}{r['test_mAP50-95']:>10.2f}")
    for r in results:
        if r["status"] != "OK":
            print(f"  {r['name']:<38}  {r['status']}")
    print(f"\n  all results -> {out_path}\n")


if __name__ == "__main__":
    main()
