#!/usr/bin/env python3
"""
Luggage Architecture Ablation — NEXT set (post-analysis).

Context (what the first arch sweep already told us, on test_full_dataset):
  - Architecture broke the bag-class ceiling that ~100 loss/SWA runs never could:
    bag AP50-95 went from a hard 45.9-47.5 band (loss era) to 48.1-49.8 (arch).
    Best bag = arch_globalctx_p2 (49.77); best balance = arch_dysample (57.57).
  - ShapeCBAM was the WORST run and lowest on bag (48.14) -> its "wide=bag" premise
    is contradicted by the data (94% of ALL classes are tall). DROPPED here.
  - Overall mAP50-95 sits at ~57.5 regardless; the discriminating metric is bag /
    small-object AP, so these runs are judged on per-class bag + mAP50-95_small.

WHAT THIS SCRIPT RUNS (only what moves the paper, ShapeCBAM removed):
  arch_baseline      — stock yolov12s head, SAME training/loss config as the arch
                       runs. ATTRIBUTION CONTROL: proves the bag gain is
                       architectural, not from the (inherited) SATAL/TAL config.
                       *** run this first — it is the missing reference. ***
  arch_p2head        — REAL 4-level P2 detection head (stride 4). The strongest
                       untested lever. The prior sweep only had the light ZGP2Fuse
                       (P2->P3 injection); this adds a dedicated stride-4 head for
                       the 33px objects. Highest upside for bag + small.
  arch_p2head_gctx   — P2 head + ZGGlobalContext per level. Combines the two best
                       signals so far (P2 detail + global context, which won bag).
  arch_detail_aux    — ZGSmallDetail + DetectAuxDual. Clean (no ShapeCBAM). Targets
                       small-object SCORING, which is the real bottleneck
                       (AR50_small ~96% but mAP50-95_small ~52%).
  arch_full_noshape  — kitchen sink WITHOUT ShapeCBAM: DySample head + ZGP2Fuse +
                       ZGSmallDetail + ZGGlobalContext + DetectAuxDual. The honest
                       "everything that actually helped" combination.

DELIBERATELY NOT RUN:
  arch_shape_p2_detail, and any ShapeCBAM combo — ShapeCBAM lost on every metric
  including bag; re-testing it wastes GPU. (The old arch_full carried ShapeCBAM as
  dead weight; arch_full_noshape replaces it.)

IMPORTANT — keep the loss/assigner config identical to the first arch sweep.
  The first arch runs evaluated with use_satal:true (+ tal_topk 12, alpha 0.6,
  beta 5), inherited from your default config, NOT set in the script. This script
  likewise does not set them, so it inherits the same default. Do NOT change that
  default between runs or arch_baseline stops being a valid control.

MODULES USED (all from nn_modules/, all zero-gated identity-init):
  ZGP2Fuse, ZGGlobalContext, ZGSmallDetail, DetectAuxDual, DySample.

REQUIRES:
  nn_modules/ copied to ultralytics/nn/modules/
  (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage:
  python run_luggage_arch_ablation_next.py                    # all, in priority order
  python run_luggage_arch_ablation_next.py arch_baseline      # single run
  python run_luggage_arch_ablation_next.py --with-test        # + test-split eval
"""

import time
import gc
import sys
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
PROJECT_DIR = "runs_luggage_arch"
YAML_DIR = "arch_yamls_luggage"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21  # Detect index in stock yolov12s
BATCH = 58            # global fallback if a run has no per-config "batch"
EPOCHS = 70
SEED = 0
AUX_W = 0.5

# =============================================================================
# BUILDING BLOCKS
# =============================================================================

# Standard YOLOv12s backbone (layers 0-8)
BACKBONE = """nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]               # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]        # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]        # 2  backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]        # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]        # 4  backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]              # 5
  - [-1, 4, A2C2f, [512, True, 4]]           # 6  backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]             # 7
  - [-1, 4, A2C2f, [1024, True, 1]]          # 8  backbone P5 (stride 32)
"""

# --- HEADS ---

# Stock nearest-neighbor head (layers 9-20, P3/P4/P5 heads @ 14/17/20)
HEAD_STOCK = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
"""

# DySample head (content-aware upsampling instead of nearest)
HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                        # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, DySample, [2]]                        # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
"""

# --- TAILS (appended after head, before Detect) ---

# A. Stock 3-level Detect @ 21  (attribution baseline: no architecture change)
TAIL_STOCK_DETECT = """  - [[14, 17, 20], 1, Detect, [nc]]                # 21
"""

# B. Real 4-level P2 detection head (stride 4). Extends the top-down path one level
#    down and fuses with backbone P2 (layer 2). Detect gets P2/P3/P4/P5;
#    strides auto-compute to [4, 8, 16, 32].  (Detect @ 24)
TAIL_P2HEAD = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[23, 14, 17, 20], 1, Detect, [nc]]             # 24  4-level: P2/P3/P4/P5
"""

# C. P2 head + ZGGlobalContext on every level (combine the two best signals).
#    (Detect @ 28)
TAIL_P2HEAD_GCTX = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 + global context
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                 # 26  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]                # 27  P5 + global context
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28  4-level: P2/P3/P4/P5
"""

# D. ZGSmallDetail + DetectAuxDual (clean, no ShapeCBAM).  (Detect @ 22)
TAIL_DETAIL_AUX = f"""  - [14, 1, ZGSmallDetail, [256, 3, 5]]             # 21  P3 detail guard
  - [[21, 17, 20, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 22
"""

# E. Full stack WITHOUT ShapeCBAM (on DySample head): P2Fuse + SmallDetail +
#    GlobalContext + DetectAuxDual.  (Detect @ 26)
TAIL_FULL_NOSHAPE = f"""  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2->P3 detail fusion
  - [21, 1, ZGSmallDetail, [256, 3, 5]]            # 22  P3 detail guard
  - [22, 1, ZGGlobalContext, [256]]                # 23  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 24  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 25  P5 + global context
  - [[23, 24, 25, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 26
"""

# =============================================================================
# ASSEMBLE ARCHITECTURES (priority order)
# =============================================================================
# NOTE on per-config "batch": the 4-level P2 head detects at stride 4 (160x160
# feature maps), which is memory-heavy, so those runs get a smaller batch; the
# DetectAuxDual runs add aux towers (training-only) so they get a moderate batch.
# These are STARTING points for a ~24GB GPU — tune to your VRAM. Any run without
# a "batch" key falls back to the global BATCH.
RUNS = [
    {
        "name": "arch_baseline",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_STOCK_DETECT,
        "batch": 58,
        "desc": "[1/5] Stock head, SAME config — ATTRIBUTION CONTROL (run first)",
    },
    {
        "name": "arch_p2head",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2HEAD,
        "batch": 32,
        "desc": "[2/5] Real 4-level P2 detection head (stride 4) — top untested lever",
    },
    {
        "name": "arch_p2head_gctx",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2HEAD_GCTX,
        "batch": 28,
        "desc": "[3/5] P2 head + GlobalContext — combines the two best bag/small signals",
    },
    {
        "name": "arch_detail_aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DETAIL_AUX,
        "batch": 48,
        "desc": "[4/5] ZGSmallDetail + DetectAuxDual — targets small-object scoring",
    },
    {
        "name": "arch_full_noshape",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_FULL_NOSHAPE,
        "batch": 32,
        "desc": "[5/5] Kitchen sink w/o ShapeCBAM: DySample + P2Fuse + Detail + GCtx + Aux",
    },
]

# =============================================================================
# HELPERS
# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping Detect layer if index changed.

    For the 4-level P2-head runs the Detect layer gains a level, so only
    shape-matching keys transfer; the P2 tower initializes fresh (expected).
    """
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src = f"model.{DETECT_SRC_IDX}."
    pfx_dst = f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items()
                if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} keys")
    return model


# =============================================================================
# TRAINING
# =============================================================================
def run_experiment(run, with_test=False):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml"], yaml_path)

    batch = run.get("batch", BATCH)  # per-config batch, fallback to global BATCH

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    start = time.time()
    model = None  # ensure defined for the finally-block even if YOLO() raises
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

        head = type(model.model.model[-1]).__name__
        strides = model.model.stride.tolist()
        print(f"  head = {head}, levels = {model.model.model[-1].nl}, strides = {strides}")

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=batch,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=30,
            close_mosaic=10,
            seed=SEED,
            deterministic=True,
            amp=True,
            val=True,
        )

        elapsed = (time.time() - start) / 3600

        metrics = {"name": run["name"], "status": "OK", "time_h": round(elapsed, 2)}
        if results is not None:
            try:
                r = results.results_dict
                metrics["val"] = {
                    "mAP50": round(r.get("metrics/mAP50(B)", 0), 5),
                    "mAP50_95": round(r.get("metrics/mAP50-95(B)", 0), 5),
                    "precision": round(r.get("metrics/precision(B)", 0), 5),
                    "recall": round(r.get("metrics/recall(B)", 0), 5),
                }
            except Exception as e:
                metrics["val_error"] = str(e)

        # Optional test evaluation
        if with_test:
            best_path = os.path.join(PROJECT_DIR, run["name"], "weights", "best.pt")
            if os.path.exists(best_path):
                test_model = YOLO(best_path)
                test_results = test_model.val(data=DATA_YAML, split="test",
                                              imgsz=IMG_SIZE, batch=batch)
                if test_results is not None:
                    tr = test_results.results_dict
                    metrics["test"] = {
                        "mAP50": round(tr.get("metrics/mAP50(B)", 0), 5),
                        "mAP50_95": round(tr.get("metrics/mAP50-95(B)", 0), 5),
                    }
                del test_model

        out_path = os.path.join(PROJECT_DIR, f"{run['name']}_metrics.json")
        os.makedirs(PROJECT_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return metrics

    except Exception as e:
        elapsed = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time_h": round(elapsed, 2)}
    finally:
        # Guarded cleanup: model may be None if YOLO() raised before assignment.
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = set(sys.argv[1:])
    with_test = "--with-test" in args
    args.discard("--with-test")

    if args:
        runs = [r for r in RUNS if r["name"] in args]
        if not runs:
            print(f"No matching runs. Available: {[r['name'] for r in RUNS]}")
            sys.exit(1)
    else:
        runs = RUNS

    os.makedirs(YAML_DIR, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  LUGGAGE ARCHITECTURE ABLATION — NEXT ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes, 40% small, AR=2.69)")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, seed={SEED}  (per-config batch below)")
    print(f"{'=' * 72}")
    for r in runs:
        print(f"  [batch {r.get('batch', BATCH):>3}] {r['desc']}")
    print(f"{'=' * 72}\n")

    all_results = []
    for run in runs:
        result = run_experiment(run, with_test=with_test)
        all_results.append(result)

    # Summary
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_next_summary.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 72}")
    print("  RESULTS")
    print(f"{'=' * 72}")
    print(f"{'Name':<28s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>8s} {'Time':>6s}")
    print("-" * 64)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<28s} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {r['status']:>8s} {r.get('time_h', 0):>5.1f}h")

    print(f"\nSummary saved: {summary_path}")
