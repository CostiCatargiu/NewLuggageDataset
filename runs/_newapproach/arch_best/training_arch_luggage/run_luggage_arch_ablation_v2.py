#!/usr/bin/env python3
"""
Luggage Architecture Ablation v2 — tuned to the ACTUAL LuggageDataset.v5i split.

Split facts (from DATASETanalyze_luggage2.py, LuggageDatasetSplit.txt):
  classes (3): backpack (27%), bag (22%), trolley (51%)   -> trolley-dominant
  40.3% small (<48px, mean W 33px)  -> 17,131 small train instances
  94.0% tall  (h/w 2.69, only 1.1% wide / 4.9% square)    -> nearly everything is thin+tall
  bag = weakest class: lowest h/w (2.23), most shape-varied (needs the most help)

DESIGN CHANGES vs v1:
  - DROPPED ShapeCBAM / C2fShapeCBAM runs. Their premise (bags=wide, backpacks=square,
    trolleys=tall) is contradicted by the data: ALL three classes are overwhelmingly
    tall (backpack 2.55, bag 2.23, trolley 2.96) and wide boxes are ~1%. The horizontal
    branch has nothing to exploit here, so shape-adaptive attention is not justified.
  - ADDED a real 4-level P2 detection head (stride 4). v1 avoided one citing "~374 small
    instances" — that number is from a different (weapons) dataset. Luggage has 17,131
    small train instances, which comfortably supports a dedicated P2 head. This is now
    the headline small-object experiment.
  - ADDED an explicit stock baseline run (arch_baseline) so all deltas are measured
    against a same-conditions control, not against numbers in comments.
  - Kept the modules that genuinely fit the small + thin problem: ZGP2Fuse (light P2->P3
    detail injection), DySample (content-aware upsampling — preserves thin/tall edges),
    ZGSmallDetail, ZGGlobalContext, DetectAuxDual (dual detail supervision).
  - Hardened plumbing: env-overridable paths + startup existence checks, safe cleanup
    (no NameError masking), custom-module import sanity check.

MODULES USED (all from nn_modules/, all zero-gated identity-init):
  ZGP2Fuse        — injects stride-4 backbone features into P3 as a gated residual.
  ZGGlobalContext — gentle per-level global context (avg pool + MLP + gated add).
  ZGSmallDetail   — small-kernel detail guard at P3 (k=3 + k=5 depthwise, gated).
  DetectAuxDual   — dual-path: main sees fused features, aux sees raw. Preserves detail.
  DySample        — content-aware upsampling (ICCV 2023). Replaces nearest-neighbor.

RUNS (9 total, ordered by confidence):
  arch_baseline        — stock yolov12s head (CONTROL)
  arch_p2fuse          — ZGP2Fuse (light P2->P3 detail injection for 33px objects)
  arch_p2head          — real 4-level P2 detection head (stride 4) — NEW headline run
  arch_dysample        — DySample upsampling (content-aware, preserves thin/tall edges)
  arch_globalctx       — ZGGlobalContext per level (dataset-agnostic context)
  arch_detail_aux      — ZGSmallDetail + DetectAuxDual (detail guard + dual supervision)
  arch_p2head_dysample — P2 head + DySample (both small-object mechanisms combined)
  arch_globalctx_p2    — ZGGlobalContext + ZGP2Fuse (context + small enrichment)
  arch_full            — DySample head + ZGP2Fuse + ZGSmallDetail + ZGGlobalContext + AuxDual

REQUIRES:
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage:
  python run_luggage_arch_ablation_v2.py
  python run_luggage_arch_ablation_v2.py arch_p2head
  python run_luggage_arch_ablation_v2.py --with-test
  DATA_YAML=/path/data.yaml PRETRAINED=yolov12s.pt python run_luggage_arch_ablation_v2.py
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
# CONFIGURATION  (env-overridable so the script is portable)
# =============================================================================
DATA_YAML = os.environ.get(
    "DATA_YAML", "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
)
PRETRAINED = os.environ.get("PRETRAINED", "yolov12s.pt")
PROJECT_DIR = os.environ.get("PROJECT_DIR", "runs_luggage_arch")
YAML_DIR = os.environ.get("YAML_DIR", "arch_yamls_luggage")
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = int(os.environ.get("WORKERS", 8))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))   # native imgs are 512; 640 upscales, helps small objs
DETECT_SRC_IDX = 21  # Detect index in stock yolov12s
BATCH = int(os.environ.get("BATCH", 58))
EPOCHS = int(os.environ.get("EPOCHS", 70))
SEED = int(os.environ.get("SEED", 0))
AUX_W = 0.5

# =============================================================================
# BUILDING BLOCKS
# =============================================================================

# Standard YOLOv12s backbone (layers 0-8). nc=3: backpack, bag, trolley.
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

# DySample head (content-aware upsampling instead of nearest — preserves thin/tall edges)
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

# 0. Stock 3-level Detect @ 21 (baseline / dysample control)
TAIL_STOCK_DETECT = """  - [[14, 17, 20], 1, Detect, [nc]]                # 21
"""

# 1. ZGP2Fuse only — light P2->P3 detail injection (Detect @ 22)
TAIL_P2FUSE = """  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2->P3 detail fusion
  - [[21, 17, 20], 1, Detect, [nc]]                # 22
"""

# 2. Real 4-level P2 detection head (stride 4). Extends the top-down path one level
#    down to stride 4 and fuses with backbone P2 (layer 2). Detect gets P2/P3/P4/P5;
#    strides auto-compute to [4, 8, 16, 32]. (Detect @ 24)
TAIL_P2HEAD = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[23, 14, 17, 20], 1, Detect, [nc]]             # 24  4-level: P2/P3/P4/P5
"""

# 2b. P2 head with DySample upsample into the P2 level (Detect @ 24)
TAIL_P2HEAD_DYS = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[23, 14, 17, 20], 1, Detect, [nc]]             # 24  4-level: P2/P3/P4/P5
"""

# 3. GlobalContext per level (Detect @ 24)
TAIL_GLOBALCTX = """  - [14, 1, ZGGlobalContext, [256]]                # 21  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 22  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 23  P5 + global context
  - [[21, 22, 23], 1, Detect, [nc]]                # 24
"""

# 4. ZGSmallDetail + DetectAuxDual (Detect @ 22)
TAIL_DETAIL_AUX = f"""  - [14, 1, ZGSmallDetail, [256, 3, 5]]             # 21  P3 detail guard
  - [[21, 17, 20, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 22
"""

# 5. GlobalContext + P2Fuse (Detect @ 25)
TAIL_GLOBALCTX_P2 = """  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2->P3 detail fusion
  - [21, 1, ZGGlobalContext, [256]]                # 22  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 23  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 24  P5 + global context
  - [[22, 23, 24], 1, Detect, [nc]]                # 25
"""

# 6. Full stack (no ShapeCBAM): P2Fuse + SmallDetail + GlobalContext + AuxDual,
#    on top of the DySample head (Detect @ 27)
TAIL_FULL = f"""  - [[14, 2], 1, ZGP2Fuse, []]                     # 21  P2->P3 detail fusion
  - [21, 1, ZGSmallDetail, [256, 3, 5]]            # 22  P3 detail guard
  - [22, 1, ZGGlobalContext, [256]]                # 23  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                # 24  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]               # 25  P5 + global context
  - [[23, 24, 25, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 26
"""

# =============================================================================
# ASSEMBLE ARCHITECTURES
# =============================================================================
RUNS = [
    {
        "name": "arch_baseline",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_STOCK_DETECT,
        "desc": "[1/9] Stock yolov12s head — CONTROL (measure all deltas against this)",
    },
    {
        "name": "arch_p2fuse",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2FUSE,
        "desc": "[2/9] ZGP2Fuse — light P2->P3 detail injection for 33px objects",
    },
    {
        "name": "arch_p2head",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2HEAD,
        "desc": "[3/9] Real 4-level P2 detection head (stride 4) — HEADLINE small-object run",
    },
    {
        "name": "arch_dysample",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_STOCK_DETECT,
        "desc": "[4/9] DySample upsampling — content-aware, preserves thin/tall edges",
    },
    {
        "name": "arch_globalctx",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GLOBALCTX,
        "desc": "[5/9] ZGGlobalContext per level — dataset-agnostic context enrichment",
    },
    {
        "name": "arch_detail_aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DETAIL_AUX,
        "desc": "[6/9] ZGSmallDetail + DetectAuxDual — detail guard + dual supervision",
    },
    {
        "name": "arch_p2head_dysample",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_P2HEAD_DYS,
        "desc": "[7/9] P2 head + DySample — both small-object mechanisms combined",
    },
    {
        "name": "arch_globalctx_p2",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GLOBALCTX_P2,
        "desc": "[8/9] GlobalContext + P2Fuse — context + small enrichment",
    },
    {
        "name": "arch_full",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_FULL,
        "desc": "[9/9] EVERYTHING (no ShapeCBAM) — DySample + P2Fuse + Detail + GlobalCtx + AuxDual",
    },
]


# =============================================================================
# HELPERS
# =============================================================================
def preflight():
    """Fail fast on missing inputs / missing custom modules before any training."""
    problems = []
    if not os.path.exists(DATA_YAML):
        problems.append(f"DATA_YAML not found: {DATA_YAML}")
    # PRETRAINED may be a bare name ultralytics auto-downloads; only warn if it looks
    # like a path that doesn't exist.
    if os.sep in PRETRAINED and not os.path.exists(PRETRAINED):
        problems.append(f"PRETRAINED not found: {PRETRAINED}")
    try:
        from ultralytics.nn.modules import (  # noqa: F401
            ZGP2Fuse, ZGGlobalContext, ZGSmallDetail, DySample, DetectAuxDual,
        )
    except Exception as e:
        problems.append(
            "Custom modules not importable from ultralytics.nn.modules "
            f"(did you copy nn_modules/ into ultralytics/nn/modules/?): {e}"
        )
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping the Detect block if its index changed.

    Note: for the 4-level P2-head run the Detect layer has one extra level, so only
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

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {BATCH}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    model = None
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

    preflight()
    os.makedirs(YAML_DIR, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  LUGGAGE ARCHITECTURE ABLATION v2 ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes: backpack/bag/trolley; 40% small, 94% tall)")
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
