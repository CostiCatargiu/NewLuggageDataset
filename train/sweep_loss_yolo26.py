#!/usr/bin/env python3
"""
YOLO26-s LOSS GRID SEARCH — Table 6, every combination, stock yolo26s graph, LuggageDataset v6i
Environment B (ultralytics 8.4.117 + ultralytics_patches/v26).

The task-aligned assigner (paper Section 4.1.2). Each (ground truth, anchor) pair is scored
m = s^alpha * u^beta (s: predicted class score, u: IoU); the top-k anchors per ground truth
become positives and m also sets the target magnitude.

THE GRID (Table 6) — every combination below is trained:

    parameter        meaning                                   values                       n   paper
    tal_beta         IoU exponent beta (stock 6)               0 0.5 1 1.5 2 3 4 5 6        9   1.0
    tal_alpha        score exponent alpha (stock 0.5)          0.25 0.4 0.5 0.6 0.75 1.0    6   0.5
    topk             one-to-many top-k (stock 10; the          7 10 12 14                   4   10
                     one-to-one branch keeps 7 -> 1)
    tal_beta_sel /   separate beta for anchor SELECTION        (1,6) (6,1) (1,3) (3,1)      6   not adopted
    tal_beta_tgt     and for TARGET magnitudes                 (2,6) (6,2)
    tal_beta_small   size-conditional beta: this value at      1.5 2.0 2.25 2.5 3.0         5   not adopted
                     the smallest boxes, rising to beta 6
                     at 64 px

The beta split and the size-conditional beta are alternatives to a single beta (each sets
the IoU exponent its own way), so they are not crossed with tal_beta. Every beta variant is
crossed with alpha and top-k:

    A  single beta        9 beta        x 6 alpha x 4 top-k = 216
    B  beta split         6 (sel, tgt)  x 6 alpha x 4 top-k = 144
    C  size-cond. beta    5 beta_small  x 6 alpha x 4 top-k = 120
                                                    total     480 runs

ORDER: the stock assigner first (beta 6, alpha 0.5, top-k 10), then the paper setting
(beta 1), then the rest of grid A, then B, then C. Runs already finished (status OK in the
results JSON) are skipped, so the script can be stopped and started again and it resumes
where it stopped.

COST: about one GPU-hour per run at 70 epochs and batch 82 on one GPU. To run a smaller grid,
delete values from the lists below; to run one setting, put it in ONLY.

top-k is a constructor argument of the assigner, not a training key, so it is set through an
on_train_start callback once the loss exists.

All runs: stock yolo26s.pt graph, batch 82, 70 epochs, 640 px, seed 0. To run the grid on
the paper's custom graph, set MODEL to configs/yolo26/yolo26-p2dys-p234rich.yaml and BATCH to
48 (the PAN rows of the checkpoint are then remapped, see remap_pan).

Usage:
  python sweep_loss_yolo26.py

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
MODEL = "yolo26s.pt"  # stock graph, pretrained
WEIGHTS = "yolo26s.pt"  # used only when MODEL is a .yaml
PROJECT_DIR = "runs_loss_yolo26"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
EPOCHS = 70
BATCH = 82
SEED = 0
REMAP_ROWS = range(17, 23)  # stock PAN/head rows in yolo26s.pt
REMAP_SHIFT = 6             # their index in graphs with an inserted P2 level

# =============================================================================
# THE GRID (Table 6) — every combination is trained
# =============================================================================
BETA = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]                  # n = 9
ALPHA = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]                              # n = 6
TOPK = [7, 10, 12, 14]                                                # n = 4
BETA_SPLIT = [(1.0, 6.0), (6.0, 1.0), (1.0, 3.0), (3.0, 1.0), (2.0, 6.0), (6.0, 2.0)]  # n = 6 (sel, tgt)
BETA_SMALL = [1.5, 2.0, 2.25, 2.5, 3.0]                               # n = 5
BETA_SMALL_REF = 6.0  # beta reached at BETA_SMALL_REF_PX (the patched default, 64 px)
STOCK = dict(beta=6.0, alpha=0.5, topk=10)
PAPER = dict(beta=1.0, alpha=0.5, topk=10)

ONLY = []  # e.g. ["y26_b1_a05_k10"] to train only these runs; empty = all

# Stock YOLO26 assigner. The patched default.yaml is already neutral (tal_beta_sel /
# tal_beta_tgt / tal_beta_small default to null = off); written out for clarity.
_ALL_OFF = dict(tal_alpha=0.5, tal_beta=6.0)
STOCK_TOPK = 10


def num(v):
    """0.25 -> '025', 1.0 -> '1', 2.25 -> '225' (for run names)."""
    s = f"{v:g}"
    return s.replace(".", "") if "." in s else s


def run(name, desc, topk, **params):
    return {"name": f"{name}_k{topk}", "desc": f"{desc}, top-k {topk}",
            "params": dict(_ALL_OFF, **params), "topk": None if topk == STOCK_TOPK else topk}


def build_runs():
    a = [run(f"y26_b{num(b)}_a{num(al)}", f"beta {b}, alpha {al}", k, tal_beta=b, tal_alpha=al)
         for b, al, k in itertools.product(BETA, ALPHA, TOPK)]
    b = [run(f"y26_sel{num(s)}_tgt{num(t)}_a{num(al)}", f"beta select {s} / target {t}, alpha {al}", k,
             tal_beta_sel=s, tal_beta_tgt=t, tal_alpha=al)
         for (s, t), al, k in itertools.product(BETA_SPLIT, ALPHA, TOPK)]
    c = [run(f"y26_bs{num(bs)}_a{num(al)}", f"beta {bs} at smallest boxes -> {BETA_SMALL_REF:g} at 64 px, "
             f"alpha {al}", k, tal_beta=BETA_SMALL_REF, tal_beta_small=bs, tal_alpha=al)
         for bs, al, k in itertools.product(BETA_SMALL, ALPHA, TOPK)]
    first = [f"y26_b{num(STOCK['beta'])}_a{num(STOCK['alpha'])}_k{STOCK['topk']}",
             f"y26_b{num(PAPER['beta'])}_a{num(PAPER['alpha'])}_k{PAPER['topk']}"]
    head = [next(r for r in a if r["name"] == n) for n in first]
    head[0]["desc"] = "REFERENCE (stock): " + head[0]["desc"]
    head[1]["desc"] = "PAPER: " + head[1]["desc"]
    return head + [r for r in a + b + c if r["name"] not in first]


RUNS = build_runs()


# =============================================================================
def remap_pan(model):
    """Graphs with an inserted P2 level: copy checkpoint rows 17-22 to rows 23-28."""
    ref = YOLO(WEIGHTS).model.float()
    if len(model.model.model) <= len(ref.model):
        return  # three-level graph: nothing is shifted
    sd, moved = model.model.state_dict(), {}
    for k, v in ref.state_dict().items():
        parts = k.split(".", 2)  # model.<row>.<rest>
        if parts[0] == "model" and parts[1].isdigit() and int(parts[1]) in REMAP_ROWS:
            k2 = f"model.{int(parts[1]) + REMAP_SHIFT}.{parts[2]}"
            if k2 in sd and sd[k2].shape == v.shape:
                moved[k2] = v
    model.model.load_state_dict(moved, strict=False)
    print(f"  [remap] {len(moved)} tensors copied from checkpoint rows 17-22 to 23-28")


def set_topk(topk):
    """Create the loss up front and set the one-to-many top-k (one-to-one stays 7 -> 1)."""
    def cb(trainer):
        m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        if getattr(m, "criterion", None) is None:
            m.criterion = m.init_criterion()
        m.criterion.one2many.assigner.topk = topk
        print(f"  [topk] one-to-many top-k set to {topk}")
    return cb


def test_eval(save_dir):
    w = Path(save_dir) / "weights" / "best.pt"
    r = YOLO(str(w)).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                         workers=WORKERS, plots=False, verbose=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


def differs(run):
    d = {k: v for k, v in run["params"].items() if _ALL_OFF.get(k, "__") != v}
    if run.get("topk") is not None:
        d["topk"] = run["topk"]
    return d


def run_one(run):
    print(f"\n{'#' * 78}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: LOSS   model={MODEL}   batch={BATCH}")
    print(f"# differs from stock: {differs(run)}")
    print(f"{'#' * 78}\n")
    start_time = time.time()
    out = {"name": run["name"], "desc": run["desc"], "params": differs(run), "status": "OK",
           "test_mAP50": None, "test_mAP50-95": None}
    try:
        model = YOLO(MODEL)
        if MODEL.endswith(".yaml"):
            model.load(WEIGHTS)
            remap_pan(model)
        if run.get("topk") is not None:
            model.add_callback("on_train_start", set_topk(run["topk"]))
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
    print(f"  YOLO26-s LOSS GRID SEARCH — {len(RUNS)} runs ({len(done)} done, {len(todo)} to train)")
    print(f"{'=' * 78}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Model: {MODEL}   Epochs: {EPOCHS}   ImgSize: {IMG_SIZE}   Batch: {BATCH}")
    print(f"  beta        n={len(BETA)}  {BETA}")
    print(f"  alpha       n={len(ALPHA)}  {ALPHA}")
    print(f"  top-k       n={len(TOPK)}  {TOPK}")
    print(f"  beta split  n={len(BETA_SPLIT)}  {BETA_SPLIT}")
    print(f"  beta small  n={len(BETA_SMALL)}  {BETA_SMALL}")
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
    print(f"  {'run':<30}{'mAP50':>8}{'mAP50-95':>10}")
    for r in ok[:20]:
        print(f"  {r['name']:<30}{r['test_mAP50']:>8.2f}{r['test_mAP50-95']:>10.2f}")
    for r in results:
        if r["status"] != "OK":
            print(f"  {r['name']:<30}  {r['status']}")
    print(f"\n  all results -> {out_path}\n")


if __name__ == "__main__":
    main()
