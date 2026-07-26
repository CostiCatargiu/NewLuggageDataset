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
                           attention = long-range VERTICAL context for 94%-tall objects.
  arch_p2head_strip      — P2 head + ZGStrip at P3/P4. Anisotropic strip conv = a fixed
                           tall receptive-field prior matched to the dataset statistic.
  arch_p2head_decoupled  — P2 head + DetectDecoupled. Decoupled cls/reg towers usually
                           improve high-IoU localization (the mAP50<->mAP50-95 gap).
  arch_p2head_gctx_hires — the current best config (P2 head + GlobalContext) at higher
                           imgsz. More pixels on 33px objects. Batch dropped to fit.

-----------------------------------------------------------------------------
BUILD-FAILURE FIXES (why this file changed)
-----------------------------------------------------------------------------
Two runs died at YOLO(yaml) — model construction, before a single training step:

1) arch_p2head_coordatt
     conv.py:724  `return x * a_h * a_w`
     RuntimeError: size of tensor a (128) must match tensor b (256) at dim 1
   128 = 256 * 0.50 (the 's' width multiplier). The module was CONSTRUCTED with the
   raw YAML arg 256 while the feature map arriving carried 128 channels -> i.e.
   CoordinateAttention is NOT listed in the width-scaling set inside parse_model()
   (ultralytics/nn/tasks.py), unlike ZGGlobalContext which is (the gctx runs built).

   PROPER FIX (recommended, do it once):
     grep -n "ZGGlobalContext" ultralytics/nn/tasks.py
     ...and add CoordinateAttention / ZGStrip / ZGSmallDetail to that same `if m in {...}`
     set so `c2 = make_divisible(min(c2, max_channels) * width, 8)` is applied.

   WORKAROUND USED HERE: the YAML channel args are templated ({c256}/{c512}/{c1024}).
   Each config is built with the NOMINAL values first; if that raises, it is rebuilt
   with PRE-SCALED values (256->128, 512->256, 1024->512). Whichever convention
   constructs is the one kept on disk. This is scale-aware, so it stays correct if
   SCALE is changed, and it is a no-op for modules that are already registered.

2) arch_p2head_decoupled
     head.py:643  `torch.cat((self.cv2[i](box[i]), self.cv3[i](cls[i])), 1)`
     RuntimeError: Expected size 64 but got size 16 for tensor number 1
   The mismatched dim is SPATIAL, not channels (64 vs 16 at build probe s=256 =
   P2 vs P4, exactly two pyramid levels apart). That is the signature of a head that
   splits its input list in half — `box, cls = x[:self.nl], x[self.nl:]` with
   `self.nl = len(ch) // 2`. Fed 4 tensors it read P2/P3 as the box path and P4/P5 as
   the cls path. So DetectDecoupled is a DUAL-input head like DetectAuxDualDeepP3, not
   a drop-in Detect.

   Verify with:  sed -n '600,660p' ultralytics/nn/modules/head.py
   If `self.nl = len(ch) // 2` is intentional -> the 8-input (duplicated) YAML below is
   correct. If it is a copy-paste leftover from the aux head -> fix the module instead
   (`self.nl = len(ch)`, `box = cls = x`) and the 4-input variant will win on its own.

   HANDLED HERE: the Detect line is templated ({det_in}); 4-input is tried first, then
   8-input (each level passed twice). Post-build the head's `nl` and strides are
   asserted against the expected pyramid, so a variant that builds but wires the wrong
   levels is rejected rather than silently trained for 3 hours.

Nothing about the loss/assigner path was touched: use_satal / tal_topk / alpha / beta
are still inherited from the defaults, so these stay comparable to the earlier sweep.

REQUIRES:
  nn_modules/ copied to ultralytics/nn/modules/
  (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage (queue is controlled by commenting/uncommenting entries in RUNS):
  python run_luggage_arch_overnight.py                      # every uncommented run in RUNS
  python run_luggage_arch_overnight.py arch_deepp3aux       # one run by name
  python run_luggage_arch_overnight.py --build-only         # construct only, no training
  python run_luggage_arch_overnight.py --with-test          # + test-split eval
"""

import time
import gc
import sys
import json
import math
import os
import itertools

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

# Must match the `scales:` entry actually used below. Used only to compute the
# pre-scaled channel fallback for modules missing from parse_model's scaling set.
SCALE_WIDTH = 0.50
MAX_CHANNELS = 1024


def scaled_ch(c):
    """Reproduce ultralytics parse_model channel scaling: make_divisible(min(c, max)*w, 8)."""
    v = min(c, MAX_CHANNELS) * SCALE_WIDTH
    return int(math.ceil(v / 8) * 8)


# Substitution axes for the build-variant fallback ---------------------------
CH_NOMINAL = {"c256": 256, "c512": 512, "c1024": 1024}
CH_SCALED = {k: scaled_ch(v) for k, v in CH_NOMINAL.items()}

DET_IN_SINGLE = {"det_in": "23, 14, 17, 20"}
DET_IN_DUAL = {"det_in": "23, 14, 17, 20, 23, 14, 17, 20"}

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
# NOTE: {c256}/{c512}/{c1024} mark channel args on modules whose parse_model
# registration is unverified. ZGGlobalContext is verified-scaled (its runs built
# fine), so those stay hard-coded and byte-identical to the proven configs.

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
TAIL_DEEPP3AUX = f"""  - [14, 1, ZGSmallDetail, [{{c256}}, 3, 5]]           # 21  P3 detail guard
  - [[21, 17, 20, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, {AUX_W}]]  # 22
"""

# 3. P2 head + CoordinateAttention x4  (Detect @ 28)
TAIL_P2_COORDATT = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, CoordinateAttention, [{c256}]]          # 24  P2 + coord attention
  - [14, 1, CoordinateAttention, [{c256}]]          # 25  P3 + coord attention
  - [17, 1, CoordinateAttention, [{c512}]]          # 26  P4 + coord attention
  - [20, 1, CoordinateAttention, [{c1024}]]         # 27  P5 + coord attention
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 4. P2 head + ZGStrip at P3/P4 (anisotropic tall receptive field)  (Detect @ 26)
TAIL_P2_STRIP = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [14, 1, ZGStrip, [{c256}]]                      # 24  P3 + strip conv
  - [17, 1, ZGStrip, [{c512}]]                      # 25  P4 + strip conv
  - [[23, 24, 25, 20], 1, Detect, [nc]]             # 26  P2/P3/P4/P5
"""

# 5. P2 head + DetectDecoupled (decoupled cls/reg towers)  (Detect @ 24)
#    {det_in} = 4 inputs, or each level duplicated if the head splits its list in half.
TAIL_P2_DECOUPLED = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[{det_in}], 1, DetectDecoupled, [nc]]          # 24  4-level decoupled head
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
# Comment / uncomment entries here to control the queue. With no CLI args, every
# uncommented entry runs, in order.
# "levels"  = expected number of detection levels on the head (asserted post-build)
# "strides" = expected stride pyramid (asserted post-build)
RUNS = [
    {
        "name": "arch_dysample_p2_gctx",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYS_P2_GCTX,
        "batch": 36,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[1/6] DySample + 4-level P2 + GlobalContext — combine the 3 winners",
    },
    {
        "name": "arch_deepp3aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEEPP3AUX,
        "batch": 52,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[2/6] SmallDetail + DetectAuxDualDeepP3 — deeper P3 towers for scoring",
    },
    {
        "name": "arch_p2head_coordatt",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_COORDATT,
        "batch": 36,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[3/6] P2 head + CoordinateAttention — H/W-factorized attn for tall objs",
    },
    {
        "name": "arch_p2head_strip",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_STRIP,
        "batch": 36,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[4/6] P2 head + ZGStrip — anisotropic tall receptive-field prior",
    },
    {
        "name": "arch_p2head_decoupled",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_DECOUPLED,
        "batch": 36,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[5/6] P2 head + DetectDecoupled — decoupled towers for tight boxes",
    },
    {
        "name": "arch_p2head_gctx_hires",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX,
        "batch": 16,          # dropped from 36 to fit 896 in VRAM
        "imgsz": 896,         # the whole point of this run — the resolution lever
        "levels": 4,
        "strides": [4, 8, 16, 32],
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


def build_variants(yaml_template):
    """Cartesian product of the substitution axes this template actually uses.

    Ordered so the CORRECT-if-parse_model-is-patched convention is tried first;
    the workaround convention is only reached if that raises.
    """
    axes = []
    if any(k in yaml_template for k in ("{c256}", "{c512}", "{c1024}")):
        axes.append([("ch=nominal", CH_NOMINAL), ("ch=pre-scaled", CH_SCALED)])
    if "{det_in}" in yaml_template:
        axes.append([("det=4-input", DET_IN_SINGLE), ("det=8-input(dual)", DET_IN_DUAL)])

    if not axes:
        return [("as-written", {})]

    variants = []
    for combo in itertools.product(*axes):
        label = " + ".join(lbl for lbl, _ in combo)
        subs = {}
        for _, d in combo:
            subs.update(d)
        variants.append((label, subs))
    return variants


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


def build_model(run):
    """Try each YAML variant until one constructs AND wires the expected pyramid.

    Returns (model, yaml_path, variant_label). Raises RuntimeError if all fail.
    """
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    variants = build_variants(run["yaml"])
    errors = []

    for label, subs in variants:
        text = run["yaml"].format(**subs)
        save_yaml(text, yaml_path)
        model = None
        try:
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
            print(f"  [build] OK via [{label}]  head={type(det).__name__} "
                  f"nl={nl} strides={strides}")
            return model, yaml_path, label

        except Exception as e:
            errors.append(f"[{label}] {type(e).__name__}: {e}")
            print(f"  [build] variant [{label}] rejected -- {e}")
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("all build variants failed:\n    " + "\n    ".join(errors))


# =============================================================================
# TRAINING
# =============================================================================
def run_experiment(run, with_test=False, build_only=False):
    batch = run.get("batch", BATCH)      # per-config batch
    imgsz = run.get("imgsz", IMG_SIZE)   # per-config image size

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    start = time.time()
    model = None
    try:
        model, yaml_path, variant = build_model(run)

        if build_only:
            n_params = sum(p.numel() for p in model.model.parameters())
            print(f"  [build-only] {run['name']}: {n_params / 1e6:.2f}M params, "
                  f"yaml={yaml_path}")
            return {"name": run["name"], "status": "BUILD-OK",
                    "variant": variant, "params_M": round(n_params / 1e6, 2),
                    "time_h": 0.0}

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

        metrics = {"name": run["name"], "status": "OK", "variant": variant,
                   "imgsz": imgsz, "batch": batch, "time_h": round(elapsed, 2)}
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

        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h) via [{variant}]")
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
    for flag in ("--with-test", "--build-only"):
        args.discard(flag)

    if args:
        unknown = args - {r["name"] for r in RUNS}
        if unknown:
            print(f"Unknown run name(s): {sorted(unknown)}")
            print(f"Available (uncommented in RUNS): {[r['name'] for r in RUNS]}")
            sys.exit(1)
        runs = [r for r in RUNS if r["name"] in args]
    else:
        runs = RUNS          # no args = every uncommented entry in RUNS

    if not runs:
        print("Nothing to run — RUNS is empty (all entries commented out?).")
        sys.exit(1)

    os.makedirs(YAML_DIR, exist_ok=True)

    mode = "BUILD-ONLY (no training)" if build_only else f"epochs={EPOCHS}, seed={SEED}"
    print(f"\n{'=' * 72}")
    print(f"  LUGGAGE ARCHITECTURE ABLATION — OVERNIGHT ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes, 40% small, AR=2.69)")
    print(f"  {mode}  (per-config batch/imgsz below)")
    print(f"  width={SCALE_WIDTH} -> fallback channels 256->{scaled_ch(256)}, "
          f"512->{scaled_ch(512)}, 1024->{scaled_ch(1024)}")
    print(f"{'=' * 72}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{r.get('imgsz', IMG_SIZE)}] {r['desc']}")
    print(f"{'=' * 72}\n")

    all_results = []
    for run in runs:
        result = run_experiment(run, with_test=with_test, build_only=build_only)
        all_results.append(result)

    suffix = "build_check" if build_only else "overnight_summary"
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 72}")
    print("  RESULTS")
    print(f"{'=' * 72}")
    print(f"{'Name':<28s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>10s} {'Time':>6s}")
    print("-" * 72)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<28s} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {r['status']:>10s} "
              f"{r.get('time_h', 0):>5.1f}h")
        if r.get("variant"):
            print(f"{'':<28s} built via [{r['variant']}]")
        if r.get("error"):
            print(f"{'':<28s} {r['error']}")

    print(f"\nSummary saved: {summary_path}")