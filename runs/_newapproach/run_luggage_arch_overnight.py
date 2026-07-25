#!/usr/bin/env python3
"""
Luggage Architecture Ablation — OVERNIGHT set (6 runs, push for higher performance).

DIAGNOSIS driving these runs (from the arch results so far):
  - mAP50 ~82.5 but mAP50-95 ~57.6  -> a ~25pt gap = boxes found, not TIGHT.
  - AR50_small ~96% but mAP50-95_small ~52% = small objects FOUND but mis-scored /
    mis-ranked and loosely localized.
  So the lever is LOCALIZATION QUALITY + SCORING, not more recall (already saturated).
  Every run below stacks ONE new lever on the proven P2-head base and targets that gap.
  Bag is NOT targeted here (it plateaued at ~49.8 across all architectures -> data/loss
  problem, not features).

RUNS (all use verified nn_modules; batch + imgsz are per-config):
  arch_dysample_p2_gctx  — combine the 3 proven winners: DySample head + real 4-level
                           P2 head (DySample upsample) + GlobalContext per level.
  arch_deepp3aux         — ZGSmallDetail@P3 + DetectAuxDualDeepP3 (DEEPER P3 cls/box
                           towers). Directly attacks small-object SCORING quality.
  arch_p2head_coordatt   — P2 head + CoordinateAttention per level. H/W-factorized
                           attention = long-range VERTICAL context for 94%-tall objects
                           (the attention ShapeCBAM should have been).
  arch_p2head_strip      — P2 head + ZGStrip at P3/P4. Anisotropic strip conv = a fixed
                           tall receptive-field prior matched to the dataset statistic.
  arch_p2head_decoupled  — P2 head + DetectDecoupled. Decoupled cls/reg towers usually
                           improve high-IoU localization (the mAP50<->mAP50-95 gap).
  arch_p2head_gctx_hires — the current best config (P2 head + GlobalContext) at imgsz
                           896. Higher res = more pixels on 33px objects (highest-EV
                           single lever). Batch dropped to fit VRAM.

REQUIRES:
  nn_modules/ copied to ultralytics/nn/modules/
  (block.py, conv.py, head.py, tasks.py, __init__.py)

IMPORTANT: keep the loss/assigner config identical to the earlier arch sweep
  (use_satal:true, tal_topk 12, alpha 0.6, beta 5 — inherited from your default, NOT
  set here). Do not change it, so these stay comparable to the previous arch runs.

Usage:
  python run_luggage_arch_overnight.py                 # all 6, in order
  python run_luggage_arch_overnight.py arch_deepp3aux  # one run
  python run_luggage_arch_overnight.py --with-test     # + test-split eval
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
IMG_SIZE = 640         # global fallback if a run has no per-config "imgsz"
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21    # Detect index in stock yolov12s
BATCH = 32             # global fallback if a run has no per-config "batch"
EPOCHS = 70
SEED = 0
AUX_W = 0.5

# =============================================================================
# BUILDING BLOCKS
# =============================================================================

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

# --- TAILS ---

# 1. DySample head + DySample-P2 head + GlobalContext x4  (Detect @ 28)
TAIL_DYS_P2_GCTX = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 + global context
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                 # 26  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]                # 27  P5 + global context
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 2. SmallDetail@P3 + DetectAuxDualDeepP3 (deeper P3 scoring towers)  (Detect @ 22)
TAIL_DEEPP3AUX = f"""  - [14, 1, ZGSmallDetail, [256, 3, 5]]             # 21  P3 detail guard
  - [[21, 17, 20, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, {AUX_W}]]  # 22
"""

# 3. P2 head + CoordinateAttention x4  (Detect @ 28)
TAIL_P2_COORDATT = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, CoordinateAttention, [256]]             # 24  P2 + coord attention
  - [14, 1, CoordinateAttention, [256]]             # 25  P3 + coord attention
  - [17, 1, CoordinateAttention, [512]]             # 26  P4 + coord attention
  - [20, 1, CoordinateAttention, [1024]]            # 27  P5 + coord attention
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 4. P2 head + ZGStrip at P3/P4 (anisotropic tall receptive field)  (Detect @ 26)
TAIL_P2_STRIP = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [14, 1, ZGStrip, [256]]                         # 24  P3 + strip conv
  - [17, 1, ZGStrip, [512]]                         # 25  P4 + strip conv
  - [[23, 24, 25, 20], 1, Detect, [nc]]             # 26  P2/P3/P4/P5
"""

# 5. P2 head + DetectDecoupled (decoupled cls/reg towers)  (Detect @ 24)
TAIL_P2_DECOUPLED = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[23, 14, 17, 20], 1, DetectDecoupled, [nc]]    # 24  4-level decoupled head
"""

# 6. P2 head + GlobalContext x4 (== current best config, run at high res)  (Detect @ 28)
TAIL_P2_GCTX = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 + global context
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 + global context
  - [17, 1, ZGGlobalContext, [512]]                 # 26  P4 + global context
  - [20, 1, ZGGlobalContext, [1024]]                # 27  P5 + global context
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# =============================================================================
# ASSEMBLE (priority order for overnight)
# =============================================================================
RUNS = [
    {
        "name": "arch_dysample_p2_gctx",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYS_P2_GCTX,
        "batch": 24,
        "desc": "[1/6] DySample + 4-level P2 + GlobalContext — combine the 3 winners",
    },
    {
        "name": "arch_deepp3aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEEPP3AUX,
        "batch": 36,
        "desc": "[2/6] SmallDetail + DetectAuxDualDeepP3 — deeper P3 towers for scoring",
    },
    {
        "name": "arch_p2head_coordatt",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_COORDATT,
        "batch": 30,
        "desc": "[3/6] P2 head + CoordinateAttention — H/W-factorized attn for tall objs",
    },
    {
        "name": "arch_p2head_strip",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_STRIP,
        "batch": 30,
        "desc": "[4/6] P2 head + ZGStrip — anisotropic tall receptive-field prior",
    },
    {
        "name": "arch_p2head_decoupled",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_DECOUPLED,
        "batch": 28,
        "desc": "[5/6] P2 head + DetectDecoupled — decoupled towers for tight boxes",
    },
    {
        "name": "arch_p2head_gctx_hires",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX,
        "batch": 16,
        "imgsz": 896,
        "desc": "[6/6] Best config (P2 + GlobalContext) at imgsz 896 — resolution lever",
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
    """Load pretrained weights, remapping the Detect block if its index changed.

    For the 4-level P2-head / decoupled / deep-P3 heads only shape-matching keys
    transfer; the new towers initialize fresh (expected, near-identity at init).
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

    batch = run.get("batch", BATCH)      # per-config batch
    imgsz = run.get("imgsz", IMG_SIZE)   # per-config image size

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {SEED}")
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
            imgsz=imgsz,
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

        if with_test:
            best_path = os.path.join(PROJECT_DIR, run["name"], "weights", "best.pt")
            if os.path.exists(best_path):
                test_model = YOLO(best_path)
                test_results = test_model.val(data=DATA_YAML, split="test",
                                              imgsz=imgsz, batch=batch)
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

    os.makedirs(YAML_DIR, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  LUGGAGE ARCHITECTURE ABLATION — OVERNIGHT ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes, 40% small, AR=2.69)")
    print(f"  epochs={EPOCHS}, seed={SEED}  (per-config batch/imgsz below)")
    print(f"{'=' * 72}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{r.get('imgsz', IMG_SIZE)}] {r['desc']}")
    print(f"{'=' * 72}\n")

    all_results = []
    for run in runs:
        result = run_experiment(run, with_test=with_test)
        all_results.append(result)

    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_overnight_summary.json")
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
