#!/usr/bin/env python3
"""
Luggage Architecture Ablation — 8 runs targeting the 3 luggage-specific problems:
  1. 40% small objects (33px mean width)
  2. 94% tall objects (AR=2.69)
  3. Bag class weak (AP50-95=0.497, highest shape variance)

MODULES USED (all from nn_modules/, all zero-gated identity-init):
  ShapeCBAM       — shape-adaptive attention: H/V/Square convolutions mixed per
                    instance. DESIGNED FOR LUGGAGE (H=bags, V=trolleys, S=backpacks).
  ZGP2Fuse        — injects stride-4 backbone features into P3 as gated residual.
                    NOT a full P2 head. Enriches P3 with high-res detail for 33px objects.
  ZGGlobalContext  — gentle per-level global context (avg pool + MLP + gated add).
                    Won 41 rounds on weapons. Dataset-agnostic context enrichment.
  ZGSmallDetail   — small-kernel detail guard at P3. Counteracts RF-shift from
                    neck enhancements. k=3 + k=5 depthwise, gated.
  DetectAuxDual   — dual-path: main sees fused features, aux sees raw. Forces
                    backbone to preserve detail for small objects.
  DySample        — content-aware upsampling (ICCV 2023). Preserves thin object
                    edges during FPN upsampling. Replaces nearest-neighbor.
  C2fShapeCBAM    — C2f block with ShapeCBAM attention integrated.

RUNS (8 total, each isolating or combining modules):
  arch_shapecbam       — C2fShapeCBAM in neck P3+P4 (shape-adaptive attention)
  arch_p2fuse          — ZGP2Fuse (P2→P3 detail injection for small objects)
  arch_globalctx       — ZGGlobalContext per level (proven winner from weapons)
  arch_dysample        — DySample upsampling (content-aware, preserves thin edges)
  arch_detail_aux      — ZGSmallDetail + DetectAuxDual (small detail + dual supervision)
  arch_globalctx_p2    — ZGGlobalContext + ZGP2Fuse (context + small enrichment)
  arch_shape_p2_detail — ShapeCBAM + ZGP2Fuse + ZGSmallDetail (full luggage stack)
  arch_full            — ShapeCBAM + ZGP2Fuse + ZGGlobalContext + ZGSmallDetail + DetectAuxDual

REQUIRES:
  nn_modules/ copied to ultralytics/nn/modules/
  (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage:
  python run_luggage_arch_ablation.py
  python run_luggage_arch_ablation.py arch_shapecbam
  python run_luggage_arch_ablation.py --with-test
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
BATCH = 58
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

# Stock nearest-neighbor head (layers 9-20, Detect @ 21)
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

# 1. ShapeCBAM at P3+P4 (Detect @ 23)
TAIL_SHAPECBAM = """  - [14, 1, C2fShapeCBAM, [256, False, -1]]        # 21  P3 + shape attention
  - [17, 1, C2fShapeCBAM, [512, False, -1]]        # 22  P4 + shape attention
  - [[21, 22, 20], 1, Detect, [nc]]                # 23
"""

# 2. ZGP2Fuse only (Detect @ 22)
TAIL_P2FUSE = f"""  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2→P3 detail fusion
  - [[21, 17, 20], 1, Detect, [nc]]                # 22
"""

# 3. GlobalContext per level (Detect @ 24)
TAIL_GLOBALCTX = """  - [14, 1, ZGGlobalContext, [256]]                # 21  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 22  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 23  P5 + global context
  - [[21, 22, 23], 1, Detect, [nc]]                # 24
"""

# 4. DySample — uses HEAD_DYSAMPLE + stock Detect @ 21
TAIL_STOCK_DETECT = """  - [[14, 17, 20], 1, Detect, [nc]]                # 21
"""

# 5. ZGSmallDetail + DetectAuxDual (Detect @ 23)
TAIL_DETAIL_AUX = f"""  - [14, 1, ZGSmallDetail, [256, 3, 5]]             # 21  P3 detail guard
  - [[21, 17, 20, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 22
"""

# 6. GlobalContext + P2Fuse (Detect @ 25)
TAIL_GLOBALCTX_P2 = f"""  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2→P3 detail fusion
  - [21, 1, ZGGlobalContext, [256]]                # 22  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 23  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 24  P5 + global context
  - [[22, 23, 24], 1, Detect, [nc]]                # 25
"""

# 7. ShapeCBAM + P2Fuse + SmallDetail (Detect @ 25)
TAIL_SHAPE_P2_DETAIL = f"""  - [14, 1, C2fShapeCBAM, [256, False, -1]]        # 21  P3 + shape attention
  - [[21, 2], 1, ZGP2Fuse, []]                     # 22  P2→P3 detail fusion
  - [22, 1, ZGSmallDetail, [256, 3, 5]]            # 23  P3 detail guard
  - [17, 1, C2fShapeCBAM, [512, False, -1]]        # 24  P4 + shape attention
  - [[23, 24, 20], 1, Detect, [nc]]                # 25
"""

# 8. Full luggage stack (Detect @ 29)
TAIL_FULL = f"""  - [14, 1, C2fShapeCBAM, [256, False, -1]]        # 21  P3 + shape attention
  - [[21, 2], 1, ZGP2Fuse, []]                     # 22  P2→P3 detail fusion
  - [22, 1, ZGSmallDetail, [256, 3, 5]]            # 23  P3 detail guard
  - [23, 1, ZGGlobalContext, [256]]                # 24  P3 + global context
  - [17, 1, C2fShapeCBAM, [512, False, -1]]        # 25  P4 + shape attention
  - [25, 1, ZGGlobalContext, [512]]                # 26  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 27  P5 + global context
  - [[24, 26, 27, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 28
"""

# =============================================================================
# ASSEMBLE ARCHITECTURES
# =============================================================================
RUNS = [
    # --- Ordered by confidence (highest first) ---
    {
        "name": "arch_p2fuse",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2FUSE,
        "desc": "[1/8] ZGP2Fuse — inject stride-4 features into P3 for 33px objects (HIGHEST conf)",
    },
    {
        "name": "arch_shapecbam",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_SHAPECBAM,
        "desc": "[2/8] ShapeCBAM at P3+P4 — shape-adaptive H/V/Square attention for luggage",
    },
    {
        "name": "arch_globalctx",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GLOBALCTX,
        "desc": "[3/8] ZGGlobalContext per level — proven weapon winner, dataset-agnostic",
    },
    {
        "name": "arch_detail_aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DETAIL_AUX,
        "desc": "[4/8] ZGSmallDetail + DetectAuxDual — detail guard + dual supervision",
    },
    {
        "name": "arch_globalctx_p2",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GLOBALCTX_P2,
        "desc": "[5/8] GlobalContext + P2Fuse — combines #1 and #3, should be best if both help",
    },
    {
        "name": "arch_dysample",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_STOCK_DETECT,
        "desc": "[6/8] DySample upsampling — content-aware, preserves thin object edges",
    },
    {
        "name": "arch_shape_p2_detail",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_SHAPE_P2_DETAIL,
        "desc": "[7/8] ShapeCBAM + P2Fuse + SmallDetail — full luggage-specific stack (higher risk)",
    },
    {
        "name": "arch_full",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_FULL,
        "desc": "[8/8] EVERYTHING — ShapeCBAM + P2 + Detail + GlobalCtx + AuxDual (highest risk)",
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
    """Load pretrained weights, remapping Detect layer if index changed."""
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

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {BATCH}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    start = time.time()
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
            batch=BATCH,
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
                                              imgsz=IMG_SIZE, batch=BATCH)
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
    print(f"  LUGGAGE ARCHITECTURE ABLATION ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes, 40% small, AR=2.69)")
    print(f"  epochs={EPOCHS}, img={IMG_SIZE}, batch={BATCH}, seed={SEED}")
    print(f"{'=' * 72}")
    for r in runs:
        print(f"  {r['desc']}")
    print(f"{'=' * 72}\n")

    all_results = []
    for run in runs:
        result = run_experiment(run, with_test=with_test)
        all_results.append(result)

    # Summary
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_summary.json")
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
