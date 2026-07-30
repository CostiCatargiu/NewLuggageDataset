#!/usr/bin/env python3
"""
Luggage Architecture — NOVEL BLOCKS ablation (aspect-ratio-steered geometry).

=============================================================================
MOTIVATION
=============================================================================
The 17+ prior 640px runs explored ONE family (spatial-attention / pooled-
context / P2-detail) and saturated at a 0.90pt plateau (mAP50-95 56.94->57.84)
with zero additivity. Every one of those modules is a KNOWN published block
(CBAM/BAM, Coordinate Attention CVPR'21, DySample ICCV'23, StarNet CVPR'24,
DSConv ICCV'23) and every one keeps SQUARE convolution geometry, doing only
attention/context on top.

The dataset's defining property is untouched by all of them:
  94% of objects are TALL (h/w > 1.25), mean AR 2.69,
  per-class AR: bag 2.23 < backpack 2.55 < trolley 2.96.

And the real bottleneck is LOCALIZATION, not detection:
  mAP50 ~83 vs mAP50-95 ~57.6  =>  25pt gap, while AR50_small ~96%.
Objects are FOUND; boxes are not TIGHT. Feature-attention cannot close a gap
that lives in box geometry.

=============================================================================
THE NOVEL CONTRIBUTION: aspect-ratio-steered CONVOLUTION GEOMETRY
=============================================================================
Instead of attention on square features, we steer the convolution GEOMETRY
itself toward the object's vertical extent. Three new blocks (in block.py):

  ARSC  (Aspect-Ratio-Steered Conv) -- FLAGSHIP.
        Per-LOCATION scalar r(p) in [0,1] blends a tall (k x 1) depthwise kernel
        with a square 3x3 kernel:  y = (1-r)*sq + r*tall. r is predicted by a
        3x3 conv->sigmoid; regions on tall objects push r->1. Zero-gated
        residual (identity at init). Novel: learns HOW tall the kernel should
        be at every pixel -- ZGStrip/ZGDSConv use a FIXED elongated kernel;
        CoordAtt/GCtx2 keep square geometry.

  ARSPP (Anisotropic Strip Pyramid Pooling).
        The tall-object analogue of ASPP: a multi-scale pyramid of VERTICAL
        strip convs (k in {3,7,11}) instead of square dilated kernels, so
        vertical context is gathered at multiple scales (bag height -> trolley
        full extent) while horizontal (mostly background) context is not wasted.

  ARGate (Aspect-Ratio Gate) -- lightweight CONTROL.
        Applies a tall (k x 1) branch gated by a single GLOBAL per-image
        verticality scalar g = sigmoid(MLP(GAP(x))). Adapts only globally, not
        per-location. Isolates the value of PER-LOCATION adaptivity when
        compared against ARSC.

=============================================================================
THE ABLATION (7 runs) -- tells the fixed-vs-adaptive geometry story
=============================================================================
  [1] arch_strip_baseline  ZGStrip @P3P4P5  -- FIXED tall geometry (control)
  [2] arch_arsc            ARSC    @P3P4P5  -- per-location adaptive (flagship)
  [3] arch_arsc_p3p4       ARSC    @P3P4    -- placement ablation
  [4] arch_argate          ARGate  @P3P4P5  -- GLOBAL adaptivity (control)
  [5] arch_arspp           ARSPP   @P3P4P5  -- multi-scale vertical pyramid
  [6] arch_arspp_p3p4      ARSPP   @P3P4    -- placement ablation
  [7] arch_arsc_gctx2      ARSC + ZGGlobalContext2 @P3P4P5 -- additivity test

Expected narrative (to be confirmed empirically):
  fixed strip (1)  <  global gate (4)  <  per-location ARSC (2)
  => adaptivity, and specifically PER-LOCATION adaptivity, is what helps.
  (7) tests whether geometry-steering is additive with the best KNOWN module.

=============================================================================
COMPARABILITY
=============================================================================
Same backbone / stock head / pretrained init / epochs / seed / batch policy as
the earlier ARCH runs. Loss & assigner untouched (use_satal / topk=12 /
alpha=0.6 / beta=5.0 inherited from defaults), so mAP is directly comparable to
the 640 plateau references (gctx22 57.84 | coordatt3 57.78 | baseline 57.63).

REQUIRES  nn_modules/ copied to ultralytics/nn/modules/
          (block.py + tasks.py + __init__.py must carry ARSC/ARSPP/ARGate)

Usage
  python run_luggage_arch_novel.py                     # all 7
  python run_luggage_arch_novel.py arch_arsc           # single run
  python run_luggage_arch_novel.py --build-only        # construct only (no train)
  python run_luggage_arch_novel.py --with-test         # + test-split eval
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
DETECT_SRC_IDX = 21    # Detect index in stock yolov12s
BATCH = 58             # global fallback
EPOCHS = 70
SEED = 0

# --- Run-mode toggles (flip these instead of passing CLI flags) -------------
# BUILD_ONLY = True  -> construct every model and verify wiring, NO training
#                       (~seconds per run; use this first to catch build bugs).
# BUILD_ONLY = False -> full training.
# WITH_TEST  = True  -> after training, also evaluate on the test split.
# CLI flags --build-only / --with-test still work and OVERRIDE these if given.
BUILD_ONLY = True
WITH_TEST = False

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

# Stock nearest-neighbor head (layers 9-20)
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

# ============================================================================
# TAILS
# ============================================================================

# [1] ZGStrip @P3P4P5 -- FIXED tall geometry control. (Detect @ 24)
#     Single fixed vertical strip length everywhere; the baseline that ARSC's
#     per-location adaptivity must beat to justify the novelty.
TAIL_STRIP = """  - [14, 1, ZGStrip, [256, 11]]                    # 21  P3 + fixed strip k=11
  - [17, 1, ZGStrip, [512, 11]]                    # 22  P4 + fixed strip k=11
  - [20, 1, ZGStrip, [1024, 11]]                   # 23  P5 + fixed strip k=11
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [2] ARSC @P3P4P5 -- per-location adaptive geometry (FLAGSHIP). (Detect @ 24)
TAIL_ARSC = """  - [14, 1, ARSC, [256, 7]]                        # 21  P3 + aspect-ratio-steered conv
  - [17, 1, ARSC, [512, 7]]                        # 22  P4 + aspect-ratio-steered conv
  - [20, 1, ARSC, [1024, 7]]                       # 23  P5 + aspect-ratio-steered conv
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [3] ARSC @P3P4 only (P5 clean). (Detect @ 23)
#     Tall+small objects concentrate at high-res; P5 (large) may not need
#     geometry steering. Fewer params, matches ZGDSConv placement logic.
TAIL_ARSC_P3P4 = """  - [14, 1, ARSC, [256, 7]]                        # 21  P3 + ARSC
  - [17, 1, ARSC, [512, 7]]                        # 22  P4 + ARSC
  - [[21, 22, 20], 1, Detect, [nc]]                 # 23  P3(ARSC)/P4(ARSC)/P5(clean)
"""

# [4] ARGate @P3P4P5 -- GLOBAL adaptivity control. (Detect @ 24)
TAIL_ARGATE = """  - [14, 1, ARGate, [256, 7]]                      # 21  P3 + global-gated tall branch
  - [17, 1, ARGate, [512, 7]]                      # 22  P4 + global-gated tall branch
  - [20, 1, ARGate, [1024, 7]]                     # 23  P5 + global-gated tall branch
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [5] ARSPP @P3P4P5 -- multi-scale vertical pyramid. (Detect @ 24)
TAIL_ARSPP = """  - [14, 1, ARSPP, [256, [3, 7, 11]]]              # 21  P3 + vertical strip pyramid
  - [17, 1, ARSPP, [512, [3, 7, 11]]]              # 22  P4 + vertical strip pyramid
  - [20, 1, ARSPP, [1024, [3, 7, 11]]]             # 23  P5 + vertical strip pyramid
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [6] ARSPP @P3P4 only (P5 clean). (Detect @ 23)
TAIL_ARSPP_P3P4 = """  - [14, 1, ARSPP, [256, [3, 7, 11]]]              # 21  P3 + vertical pyramid
  - [17, 1, ARSPP, [512, [3, 7, 11]]]              # 22  P4 + vertical pyramid
  - [[21, 22, 20], 1, Detect, [nc]]                 # 23  P3(ARSPP)/P4(ARSPP)/P5(clean)
"""

# [7] ARSC + ZGGlobalContext2 @P3P4P5 -- additivity test. (Detect @ 27)
#     Geometry-steering (localization axis) + the #1 known feature module
#     (global context axis). Orthogonal bottlenecks -> should be additive if
#     ARSC genuinely attacks localization rather than duplicating feature work.
TAIL_ARSC_GCTX2 = """  - [14, 1, ARSC, [256, 7]]                        # 21  P3 + ARSC
  - [17, 1, ARSC, [512, 7]]                        # 22  P4 + ARSC
  - [20, 1, ARSC, [1024, 7]]                       # 23  P5 + ARSC
  - [21, 1, ZGGlobalContext2, [256]]                # 24  P3 + global ctx
  - [22, 1, ZGGlobalContext2, [512]]                # 25  P4 + global ctx
  - [23, 1, ZGGlobalContext2, [1024]]               # 26  P5 + global ctx
  - [[24, 25, 26], 1, Detect, [nc]]                 # 27
"""

# =============================================================================
# ASSEMBLE ARCHITECTURES
# =============================================================================
RUNS = [
    {
        "name": "arch_strip_baseline",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_STRIP,
        "batch": 56,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[1/7] ZGStrip @P3P4P5 -- FIXED tall geometry (control for ARSC)",
    },
    {
        "name": "arch_arsc",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARSC,
        "batch": 56,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[2/7] ARSC @P3P4P5 -- per-location aspect-ratio-steered conv (FLAGSHIP)",
    },
    {
        "name": "arch_arsc_p3p4",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARSC_P3P4,
        "batch": 56,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[3/7] ARSC @P3P4 (P5 clean) -- placement ablation",
    },
    {
        "name": "arch_argate",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARGATE,
        "batch": 56,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[4/7] ARGate @P3P4P5 -- GLOBAL adaptivity (isolates per-location value)",
    },
    {
        "name": "arch_arspp",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARSPP,
        "batch": 52,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[5/7] ARSPP @P3P4P5 -- multi-scale vertical strip pyramid (tall ASPP)",
    },
    {
        "name": "arch_arspp_p3p4",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARSPP_P3P4,
        "batch": 54,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[6/7] ARSPP @P3P4 (P5 clean) -- placement ablation",
    },
    {
        "name": "arch_arsc_gctx2",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_ARSC_GCTX2,
        "batch": 50,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[7/7] ARSC + GCtx2 @P3P4P5 -- geometry + best-known feature (additivity)",
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
    """Load pretrained weights, remapping the Detect layer if its index shifted.

    Any layers added before Detect (ARSC/ARSPP/ARGate/GCtx2) initialize fresh;
    with zero-gated residuals (gamma=0) they are identity at init, so the
    pretrained backbone/head/box path transfers cleanly.
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


def build_model(run):
    """Construct model, verify head levels and stride pyramid."""
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml"], yaml_path)

    model = YOLO(yaml_path)

    det = model.model.model[-1]
    nl = getattr(det, "nl", None)
    strides = [int(s) for s in model.model.stride.tolist()]

    exp_nl = run.get("levels")
    exp_st = run.get("strides")
    if exp_nl is not None and nl != exp_nl:
        raise RuntimeError(f"head nl={nl}, expected {exp_nl}")
    if exp_st is not None and strides != exp_st:
        raise RuntimeError(f"strides={strides}, expected {exp_st}")

    load_pretrained_with_detect_remap(model)
    print(f"  [build] OK  head={type(det).__name__} nl={nl} strides={strides}")
    return model, yaml_path


# =============================================================================
# TRAINING
# =============================================================================
def run_experiment(run, with_test=False, build_only=False):
    batch = run.get("batch", BATCH)

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Imgsz {IMG_SIZE}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    start = time.time()
    model = None
    try:
        model, yaml_path = build_model(run)

        if build_only:
            n_params = sum(p.numel() for p in model.model.parameters())
            print(f"  [build-only] {run['name']}: {n_params / 1e6:.2f}M params, "
                  f"yaml={yaml_path}")
            return {"name": run["name"], "status": "BUILD-OK",
                    "params_M": round(n_params / 1e6, 2), "time_h": 0.0}

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

        metrics = {"name": run["name"], "status": "OK",
                   "batch": batch, "time_h": round(elapsed, 2)}
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
    # Start from the module-level toggles; CLI flags override them if present.
    with_test = WITH_TEST
    build_only = BUILD_ONLY
    if "--with-test" in args:
        with_test = True
    if "--no-test" in args:
        with_test = False
    if "--build-only" in args:
        build_only = True
    if "--train" in args:
        build_only = False
    for flag in ("--with-test", "--no-test", "--build-only", "--train"):
        args.discard(flag)

    if args:
        unknown = args - {r["name"] for r in RUNS}
        if unknown:
            print(f"Unknown run name(s): {sorted(unknown)}")
            print(f"Available: {[r['name'] for r in RUNS]}")
            sys.exit(1)
        runs = [r for r in RUNS if r["name"] in args]
    else:
        runs = RUNS

    if not runs:
        print("Nothing to run -- RUNS is empty.")
        sys.exit(1)

    os.makedirs(YAML_DIR, exist_ok=True)

    mode = "BUILD-ONLY (no training)" if build_only else f"epochs={EPOCHS}, seed={SEED}"
    print(f"\n{'=' * 78}")
    print(f"  LUGGAGE ARCH — NOVEL BLOCKS ({len(runs)} runs @640)")
    print(f"  Dataset: luggage (3 classes, 40% small, 94% tall, AR=2.69)")
    print(f"  Novelty: aspect-ratio-STEERED convolution geometry (ARSC/ARSPP/ARGate)")
    print(f"  {mode}  (per-config batch below)")
    print(f"  Beat-target: gctx22 = 57.84 | coordatt3 = 57.78 | baseline = 57.63")
    print(f"{'=' * 78}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{IMG_SIZE}] {r['desc']}")
    print(f"{'=' * 78}\n")

    all_results = []
    for run in runs:
        all_results.append(run_experiment(run, with_test=with_test,
                                          build_only=build_only))

    suffix = "novel_build_check" if build_only else "novel_summary"
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 78}")
    print("  RESULTS")
    print(f"{'=' * 78}")
    print(f"{'Name':<28s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>10s} {'Time':>6s}")
    print("-" * 70)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<28s} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {r['status']:>10s} "
              f"{r.get('time_h', 0):>5.1f}h")
        if r.get("params_M"):
            print(f"{'':<28s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<28s} {r['error']}")

    print(f"\nSummary saved: {summary_path}")
