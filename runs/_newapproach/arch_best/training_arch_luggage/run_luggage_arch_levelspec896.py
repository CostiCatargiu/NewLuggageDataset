#!/usr/bin/env python3
"""
Luggage Architecture — arch_levelspec at imgsz 896.

=============================================================================
WHAT THIS IS
=============================================================================
arch_levelspec is the best 640 architecture in the corpus (36 runs):

    gctx2 (ZGGlobalContext2, avg+max) @ P2, P3
    ZGDSConv(k=9)                     @ P4, P5
    DySample neck + 4-level P2 head

  @640 b32 = 83.34 mAP50 / 58.02 mAP50-95
  vs assigner-matched baseline v12s_default2 (82.77 / 57.63):
      +0.69% mAP50 | +0.68% mAP50-95 | +0.98% sm50-95 | +1.89% lg50-95
      +2.09% precision | -1.97% recall | -0.44% md50-95
  -> positive on 8 of 10 metrics, the only 640 architecture to manage that.

It is also a VALIDATED local optimum. Round 5 perturbed it in four independent
directions and every one was worse:
    remove ZGDSConv from P4/P5   57.58  (-0.44)   <- dsconv IS load-bearing
    shrink snake kernel k 9->5   57.79  (-0.23)   <- k=9 is right despite being
                                                     2-4x the object extent
    shift boundary to P2 / P3P5  57.86  (-0.16)
    add DetectAuxDual            57.68  (-0.34)   <- lifts recall, costs more

The mechanism is ROUTING, not modules: the same two modules STACKED on the same
levels (arch_dysgctx2_dsconv4) scored 57.53, SEPARATED by level they score
58.02. +0.49 from routing alone -- larger than any single module contributes
anywhere in the corpus.

=============================================================================
WHY 896
=============================================================================
Resolution is the only lever that has reliably beaten the 640 band, and it has
been confirmed three times:
    gctx5   57.67 -> hires10   58.92   (+1.25, mean of both test sets)
    dysample 57.61 -> hires4    58.95   (+1.34)
    decoupled2 57.52 -> hires3  59.11   (+1.59)
Current best @896 = arch_decoupled2_hires3, 59.11 mean (59.13 / 59.09), and it
is the most cross-set-stable run in the corpus (drift -0.04).

levelspec starts 0.50 above decoupled2's 640 score (58.02 vs 57.52). Applying
the observed +1.25..+1.59 resolution gain puts this run at roughly 59.3-59.6,
i.e. a new overall best if the gain transfers at all.

Two reasons it may transfer BETTER than average:
  * gctx2's max-pool branch resolves the salient peak over 2x the linear extent
    at 896 (the avg branch is largely resolution-insensitive).
  * ZGDSConv sits at P4/P5. At 640 the mean object is 4.5 / 2.25 cells tall
    there, so a 9-tap snake overshoots. At 896 it is 6.3 / 3.2 cells -- closer
    to matched, and round 5 already showed k=9 beats k=5, meaning the module
    wants MORE spatial extent, not less.
One reason it may transfer WORSE: levelspec is already the most precision-heavy
model in the corpus (P 82.61 / R 73.76). Resolution usually buys recall, which
is exactly what it lacks -- but if the precision bias is architectural rather
than resolution-limited, the recall ceiling follows it to 896.

=============================================================================
NOTES
=============================================================================
  * The YAML is byte-identical to the 640 arch_levelspec tail. ONLY imgsz and
    batch differ, so the 640 vs 896 delta is attributable to resolution alone.
  * batch 16 mirrors arch_p2head_gctx_hires10 (same 4-level P2 head at 896).
    ZGDSConv sits at P4/P5 only -- 56x56 and 28x28 at 896 -- so it adds little
    over the gctx variant. If it OOMs in the first epoch, drop to 12
    (arch_decoupled2_hires3 ran at 12).
  * Loss/assigner untouched -> directly comparable to every other arch run.
  * A stock-v12s @896 baseline does NOT exist. Every "% improvement" quoted for
    an 896 run is currently measured against the 640 baseline, which conflates
    architecture with resolution. RUN [2] below fixes that; it is commented out
    so this file trains only what was asked for.

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_levelspec896.py                # train
  python run_luggage_arch_levelspec896.py --build-only   # construct + param count
  python run_luggage_arch_levelspec896.py --with-test    # + test-split eval
"""

import time
import gc
import sys
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
PROJECT_DIR = "runs_luggage_arch6"
YAML_DIR = "arch_yamls_luggage"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 896
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 16
EPOCHS = 70
SEED = 0

# =============================================================================
# ARCHITECTURE — byte-identical to the 640 arch_levelspec
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

# gctx2 on the FINE levels, snake conv on the COARSE levels. No level gets both.
TAIL_LEVELSPEC = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2      (small)
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2      (small)
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake conv (medium)
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake conv (large)
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

RUNS = [
    {
        "name": "arch_levelspec_hires",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LEVELSPEC,
        "batch": 16,
        "imgsz": 896,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "arch_levelspec @896 b16 — the 640 champion at high resolution",
        "ref": "levelspec@640 58.02 | best@896 decoupled2_hires3 59.11 | target >59.11",
    },
    # ---- the missing 896 baseline (uncomment to make % claims clean) ----------
    # {
    #     "name": "v12s_default_hires",
    #     "yaml": None,                 # stock yolov12s.yaml, no custom modules
    #     "batch": 20,
    #     "imgsz": 896,
    #     "levels": 3,
    #     "strides": [8, 16, 32],
    #     "desc": "stock YOLOv12s @896 — the baseline every 896 % claim needs",
    #     "ref": "v12s_default2@640 = 57.63",
    # },
]


# =============================================================================
# HELPERS
# =============================================================================
def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping the Detect block if its index changed."""
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


def build_model(run):
    """Construct, then assert head nl and the stride pyramid before training."""
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml"], yaml_path)

    model = YOLO(yaml_path)
    det = model.model.model[-1]
    nl = getattr(det, "nl", None)
    strides = [int(s) for s in model.model.stride.tolist()]

    if run.get("levels") is not None and nl != run["levels"]:
        raise RuntimeError(f"head nl={nl}, expected {run['levels']}")
    if run.get("strides") is not None and strides != run["strides"]:
        raise RuntimeError(f"strides={strides}, expected {run['strides']}")

    load_pretrained_with_detect_remap(model)
    print(f"  [build] OK  head={type(det).__name__} nl={nl} strides={strides}")
    return model, yaml_path


# =============================================================================
# TRAINING
# =============================================================================
def run_experiment(run, with_test=False, build_only=False):
    batch = run.get("batch", BATCH)
    imgsz = run.get("imgsz", IMG_SIZE)

    print(f"\n{'#' * 78}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# beat: {run.get('ref', '-')}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 78}\n")

    start = time.time()
    model = None
    try:
        model, yaml_path = build_model(run)

        if build_only:
            n = sum(p.numel() for p in model.model.parameters())
            print(f"  [build-only] {run['name']}: {n / 1e6:.2f}M params, yaml={yaml_path}")
            return {"name": run["name"], "status": "BUILD-OK",
                    "params_M": round(n / 1e6, 2), "imgsz": imgsz,
                    "batch": batch, "time_h": 0.0}

        results = model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=imgsz, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=run["name"],
            patience=30, close_mosaic=10, seed=SEED, deterministic=True,
            amp=True, val=True,
        )

        elapsed = (time.time() - start) / 3600
        metrics = {"name": run["name"], "status": "OK", "imgsz": imgsz,
                   "batch": batch, "seed": SEED, "time_h": round(elapsed, 2)}
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
                tm = YOLO(best_path)
                tr = tm.val(data=DATA_YAML, split="test", imgsz=imgsz, batch=batch)
                if tr is not None:
                    rd = tr.results_dict
                    metrics["test"] = {
                        "mAP50": round(rd.get("metrics/mAP50(B)", 0), 5),
                        "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0), 5),
                    }
                del tm

        os.makedirs(PROJECT_DIR, exist_ok=True)
        with open(os.path.join(PROJECT_DIR, f"{run['name']}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return metrics

    except Exception as e:
        elapsed = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback
        traceback.print_exc()
        return {"name": run["name"], "status": "FAILED", "error": str(e),
                "time_h": round(elapsed, 2)}
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
    build_only = "--build-only" in args
    runs = [r for r in RUNS if r.get("yaml")]

    os.makedirs(YAML_DIR, exist_ok=True)
    print(f"\n{'=' * 78}")
    print("  arch_levelspec @896  —  the 640 champion at high resolution")
    print(f"  {'BUILD-ONLY' if build_only else f'epochs={EPOCHS}, seed={SEED}'}")
    print("  640 reference : levelspec       83.34 / 58.02  (b32)")
    print("  896 to beat   : decoupled2_hires3      59.11  (mean of both test sets)")
    print("  expectation   : ~59.3-59.6 if the +1.25..1.59 resolution gain transfers")
    print(f"{'=' * 78}\n")

    all_results = [run_experiment(r, with_test=with_test, build_only=build_only)
                   for r in runs]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    suffix = "levelspec896_build_check" if build_only else "levelspec896_summary"
    path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 78}")
    print("  RESULT")
    print(f"{'=' * 78}")
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<24s} b{r.get('batch', 0):<3d} @{r.get('imgsz', 0)}  "
              f"mAP50 {v.get('mAP50', 0):.4f}  mAP50-95 {v.get('mAP50_95', 0):.4f}  "
              f"P {v.get('precision', 0):.4f}  R {v.get('recall', 0):.4f}  "
              f"[{r['status']}]  {r.get('time_h', 0):.1f}h")
        if r.get("params_M"):
            print(f"{'':<24s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<24s} {r['error']}")
    print(f"\nSummary saved: {path}")
