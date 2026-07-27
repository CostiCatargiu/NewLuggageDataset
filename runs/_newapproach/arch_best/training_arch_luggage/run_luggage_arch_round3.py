#!/usr/bin/env python3
"""
Luggage Architecture Ablation — ROUND 3 (3 runs: 1 @640 + 2 @896).

=============================================================================
WHERE WE ARE
=============================================================================
640 plateau, mAP50-95, 17 architectures:  56.94 -> 57.64  (0.70pt wide)
  coordatt3 57.64 | decoupled_gctx_asym 57.61 | gctx5 57.60 | dysample_p2_gctx5
  57.58 | dysample 57.57 | globalctx_p2 57.57 | decoupled_coordatt 57.46 |
  decoupled2 57.46 | decoupled_gctx 57.44 | detail_aux4 57.36 | strip4 57.33 |
  p2fuse2 57.19 | p2head6 57.18 | globalctx 57.15 | full_noshape2 57.14 |
  deepp3aux2 57.07 | shapecbam 56.94            (v6_default2 baseline = 56.84)

Only ONE thing has ever broken that band: RESOLUTION.
  hires10 = gctx5 @896 = 58.82  (+1.22 over its own 640 twin, ~2x the whole band)

Round-2 result (decoupling): a measured LOSS, confirmed twice.
  gctx5    57.60 -> decoupled_gctx    57.44   (-0.16)
  coordatt3 57.64 -> decoupled_coordatt 57.46 (-0.18)
Enhancement belongs on BOTH box and cls paths, not cls-only. And the asym run
refuted the capacity-dilution story for the large-object tax: stripping gctx off
the P4/P5 cls path made large WORSE (57.32 vs 58.14), i.e. gctx on P4/P5 was
HELPING large. The tax is assigner-side (topk=12 spread over 4 levels + satal),
not a head-capacity problem, so no head variant will recover it.

=============================================================================
THE THREE RUNS
=============================================================================
[1] arch_gctx2  (640, batch 28)  ---------------------------------------------
    ZGGlobalContext2 (avg+max pool) in place of ZGGlobalContext (avg only).
    BYTE-IDENTICAL to the gctx5 tail except the module name -> a true
    single-variable ablation against your strongest, best-validated lever.
      * ZGGlobalContext2 is in block.py and registered in parse_model's
        width-scaling set (tasks.py), so channel args stay hard-coded.
      * It was written for round41 on the WEAPON dataset and has NEVER been run
        on luggage.
      * Mechanism: objects here are 94% tall, mean 33x72px, ~5 per image in
        clutter. An avg-pooled global descriptor is dominated by background;
        max-pool adds the salient peak.
      * batch 28 (NOT 32) to match arch_p2head_gctx exactly.
    If it wins it also upgrades hires10 for free, since hires10 IS gctx5 @896.
    Reference to beat: gctx5 = 57.60 / 83.07 / small 79.74 / large 59.37.

[2] arch_dysample_hires  (896, batch 24)  ------------------------------------
    arch_dysample at 896. Verified 3-LEVEL: [[14,17,20], 1, Detect, [nc]] —
    there is no P2 level at all, so its entire +0.73 at 640 came from
    content-aware upsampling.
      * Tops the all-metrics aggregate at 640 (mean rank 5.4/17), #1 precision,
        #2 mAP50.
      * DySample is a LEARNED upsampler: it predicts sampling offsets, so more
        input pixels = more signal for those offsets. Most resolution-hungry
        module in the set.
      * At 896 the P3 level (stride 8) is 112x112 — roughly the effective
        resolution P2 had at 448. A 3-level head may get P2-like small-object
        coverage WITHOUT paying the P2 large-object tax. Every 896 run so far
        has been 4-level; this cell is completely untested.
      * Cheap: 3 levels at 896 fits batch 24 comfortably.
    Reference: dysample @640 = 57.57 / 82.96 / small 79.62 / large 60.08.

[3] arch_decoupled2_hires  (896, batch 12)  ----------------------------------
    arch_p2head_decoupled2 at 896 — the best 640 SMALL-object config.
      * decoupled2 is #1 at 640 on BOTH small metrics (mAP50_small 80.06,
        mAP50-95_small 52.91).
      * 896 amplifies small objects more than anything else in the corpus:
        hires10 gained +2.35 small mAP50 and +2.39 small mAP50-95.
      * So this is the best shot at a standout SMALL-object number, as opposed
        to a standout aggregate.
    IMPORTANT — faithful reproduction. decoupled2 was NOT task-decoupled. Its
    Detect line resolved to the 8-input DUPLICATED fallback
    [23,14,17,20, 23,14,17,20], so box and cls read IDENTICAL features and the
    win came from cv3 being rebuilt as a deeper DWConv->Conv tower — a FAT CLS
    TOWER over shared features. That duplicated wiring is written out
    EXPLICITLY below with decoupled_levels: 0, so the routing assertion
    documents the intent instead of a build fallback deciding it silently.
    Reference: decoupled2 @640 = 57.46 / 82.69 / small 80.06 / large 56.90.

Not chosen for the 896 slots: coordatt3. Highest 640 base (57.64) so the naive
extrapolation is the biggest (~58.9), but it is 14th of 17 on large mAP50-95 and
resolution will not fix that; coordinate attention also has no mechanism that
scales specially with input size. decoupled_gctx @896 was dropped outright — at
57.44 it is the measured-worse twin of gctx5, whose 896 result (58.82) you
already have, so it would cost the most expensive run type to confirm ~58.7.

=============================================================================
BUILD ROBUSTNESS
=============================================================================
  * Every Detect input list is written EXPLICITLY. No {det_in} templating, so
    box/cls routing can never be chosen by a silent build fallback.
  * assert_routing() parses the generated YAML, splits the Detect from-list in
    half and counts levels whose cls source differs from its box source. It runs
    BEFORE YOLO(yaml) so a mis-wired head cannot reach model.train().
      run [3] declares decoupled_levels: 0 -> duplicated inputs are the INTENT.
  * ZGGlobalContext / ZGGlobalContext2 are verified-scaled -> hard-coded args.
  * Post-build: head nl + stride pyramid asserted (3-level run expects
    [8,16,32]; 4-level runs expect [4,8,16,32]).
  * --build-only constructs all three with every assertion and does NOT train.

Loss/assigner path untouched: use_satal / tal_topk / alpha / beta inherited from
defaults, so these stay comparable to the earlier ARCH runs. (Note the arch
family carries use_satal=true / topk=12 / a=0.6 / b=5.0, which is NOT the
v6_default2 setting — keep comparisons within the arch family.)

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_round3.py                      # all 3, in order
  python run_luggage_arch_round3.py --build-only         # construct only, no training
  python run_luggage_arch_round3.py arch_gctx2 --with-test
"""

import time
import gc
import re
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
PROJECT_DIR = "runs_luggage_arch3"
YAML_DIR = "arch_yamls_luggage"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640         # global fallback if a run has no per-config "imgsz"
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21    # Detect index in stock yolov12s
BATCH = 28             # global fallback if a run has no per-config "batch"
EPOCHS = 70
SEED = 0

SCALE_WIDTH = 0.50
MAX_CHANNELS = 1024


def scaled_ch(c):
    """Reproduce ultralytics parse_model channel scaling: make_divisible(min(c, max)*w, 8)."""
    v = min(c, MAX_CHANNELS) * SCALE_WIDTH
    return int(math.ceil(v / 8) * 8)


CH_NOMINAL = {"c256": 256, "c512": 512, "c1024": 1024}
CH_SCALED = {k: scaled_ch(v) for k, v in CH_NOMINAL.items()}

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

# [1] P2 head + ZGGlobalContext2 (avg+max) per level.
#     Byte-identical to the gctx5 tail except ZGGlobalContext -> ZGGlobalContext2.
#     (Detect @ 28, 4 levels)
TAIL_P2_GCTX2 = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + avg+max global context
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + avg+max global context
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + avg+max global context
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 + avg+max global context
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [2] Stock 3-level Detect on the DySample neck. NO P2 level.  (Detect @ 21)
TAIL_STOCK_DETECT = """  - [[14, 17, 20], 1, Detect, [nc]]                 # 21  P3/P4/P5 (no P2)
"""

# [3] P2 head + DetectDecoupled fed DUPLICATED inputs — the faithful
#     reproduction of arch_p2head_decoupled2 (fat cls tower over SHARED
#     features, NOT task decoupling). decoupled_levels: 0 makes that explicit.
#     (Detect @ 24, 4 levels)
TAIL_P2_FATCLS = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [[23, 14, 17, 20, 23, 14, 17, 20], 1, DetectDecoupled, [nc]]  # 24  box==cls (shared)
"""

# =============================================================================
# ASSEMBLE
# =============================================================================
RUNS = [
    {
        "name": "arch_gctx2",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX2,
        "batch": 28,          # matches arch_p2head_gctx exactly
        "imgsz": 640,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[1/3] @640 b28 — ZGGlobalContext2 (avg+max) vs gctx5 57.60, single variable",
    },
    {
        "name": "arch_dysample_hires",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_STOCK_DETECT,
        "batch": 24,
        "imgsz": 896,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[2/3] @896 b24 — DySample 3-level (no P2) at high res vs 57.57",
    },
    {
        "name": "arch_decoupled2_hires",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_FATCLS,
        "batch": 12,          # fat cls tower on P2 stride-4 @896 = 224x224
        "imgsz": 896,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 0,   # duplicated inputs are INTENDED here
        "desc": "[3/3] @896 b12 — best-small 640 head (fat cls tower) at high res vs 57.46",
    },
    {
        "name": "arch_gctx22_hires",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX2,
        "batch": 16,          # same cost as hires10 (gctx2 ~= gctx in params/FLOPs)
        "imgsz": 896,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[4/4] @896 b16 — BEST 640 model at high res; target > 59.19",
    },
    # ---- seed repeats, ready to uncomment (turns the 0.22 lead into a measurement) ----
    # {
    #     "name": "arch_gctx22_s1",
    #     "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX2,
    #     "batch": 28, "imgsz": 640, "seed": 1,
    #     "levels": 4, "strides": [4, 8, 16, 32],
    #     "desc": "[s1] @640 b28 seed 1 — reproducibility of arch_gctx22",
    # },
    # {
    #     "name": "arch_gctx22_s2",
    #     "yaml": BACKBONE + HEAD_STOCK + TAIL_P2_GCTX2,
    #     "batch": 28, "imgsz": 640, "seed": 2,
    #     "levels": 4, "strides": [4, 8, 16, 32],
    #     "desc": "[s2] @640 b28 seed 2 — reproducibility of arch_gctx22",
    # },
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

    Only the CHANNEL axis exists ({c256}/{c512}/{c1024}), tried nominal-first.
    There is deliberately no {det_in} axis: Detect input lists are explicit.
    """
    axes = []
    if any(k in yaml_template for k in ("{c256}", "{c512}", "{c1024}")):
        axes.append([("ch=nominal", CH_NOMINAL), ("ch=pre-scaled", CH_SCALED)])

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


DET_FROM_RE = re.compile(
    r"-\s*\[\s*\[([^\]]*)\]\s*,\s*\d+\s*,\s*(Detect[A-Za-z0-9_]*)\s*,"
)


def parse_det_from(yaml_text):
    """Return (head_name, from_list) for the last multi-input Detect* layer."""
    matches = DET_FROM_RE.findall(yaml_text)
    if not matches:
        return None, None
    src, head = matches[-1]
    return head, [int(t.strip()) for t in src.split(",") if t.strip()]


def assert_routing(run, yaml_text):
    """Verify box/cls feature sources match the declared intent.

    DetectDecoupled sets self.nl = len(ch)//2 and splits its input list in half:
    the first nl feed cv2 (box), the last nl feed cv3 (cls). Identical halves
    mean the head is NOT decoupled -- it is a fat cls tower over shared
    features. Run [3] WANTS that (decoupled_levels: 0) because it reproduces
    arch_p2head_decoupled2; any other value is a wiring bug. Runs without a
    decoupled_levels key use a plain Detect and are skipped.
    """
    exp = run.get("decoupled_levels")
    if exp is None:
        return
    head, frm = parse_det_from(yaml_text)
    if frm is None:
        raise RuntimeError("routing check: no multi-input Detect* layer found in YAML")
    if len(frm) % 2 != 0:
        raise RuntimeError(
            f"routing check: {head} needs an even input count (2*nl), got {len(frm)}"
        )
    nl = len(frm) // 2
    box, cls = frm[:nl], frm[nl:]
    n_dec = sum(1 for b, c in zip(box, cls) if b != c)
    if n_dec != exp:
        raise RuntimeError(
            f"routing check FAILED for {head}: {n_dec}/{nl} levels decoupled, "
            f"expected {exp}.  box={box}  cls={cls}"
        )
    kind = "SHARED features (fat cls tower)" if n_dec == 0 else f"{n_dec}/{nl} decoupled"
    print(f"  [routing] {head}: nl={nl}  box={box}  cls={cls}  -> {kind}")


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
    """Routing check -> construct -> assert head nl -> assert stride pyramid."""
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    variants = build_variants(run["yaml"])
    errors = []

    for label, subs in variants:
        text = run["yaml"].format(**subs)
        save_yaml(text, yaml_path)
        model = None
        try:
            assert_routing(run, text)

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
    batch = run.get("batch", BATCH)
    imgsz = run.get("imgsz", IMG_SIZE)
    seed  = run.get("seed", SEED)      # per-config seed (for reproducibility repeats)

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {seed}")
    print(f"{'#' * 72}\n")

    start = time.time()
    model = None
    try:
        model, yaml_path, variant = build_model(run)

        if build_only:
            n_params = sum(p.numel() for p in model.model.parameters())
            print(f"  [build-only] {run['name']}: {n_params / 1e6:.2f}M params, "
                  f"yaml={yaml_path}")
            return {"name": run["name"], "status": "BUILD-OK", "variant": variant,
                    "params_M": round(n_params / 1e6, 2), "imgsz": imgsz,
                    "batch": batch, "time_h": 0.0}

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
            seed=seed,
            deterministic=True,
            amp=True,
            val=True,
        )

        elapsed = (time.time() - start) / 3600

        metrics = {"name": run["name"], "status": "OK", "variant": variant,
                   "imgsz": imgsz, "batch": batch, "seed": seed,
                   "time_h": round(elapsed, 2)}
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
    print(f"  LUGGAGE ARCH -- ROUND 3 ({len(runs)} runs)")
    print(f"  Dataset: luggage (3 classes, 40% small, mean box 33x72px, AR=2.69)")
    print(f"  {mode}")
    print(f"  640 refs: gctx5 57.60 | dysample 57.57 | decoupled2 57.46 (small 80.06/52.91)")
    print(f"  896 ref : hires10 58.82 (small 80.60/53.52, large 58.62)")
    print(f"{'=' * 78}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{r.get('imgsz', IMG_SIZE)}] {r['desc']}")
    print(f"{'=' * 78}\n")

    all_results = []
    for run in runs:
        all_results.append(run_experiment(run, with_test=with_test, build_only=build_only))

    suffix = "round3_build_check" if build_only else "round3_summary"
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 78}")
    print("  RESULTS")
    print(f"{'=' * 78}")
    print(f"{'Name':<26s} {'img':>5s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>10s} {'Time':>6s}")
    print("-" * 78)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<26s} {r.get('imgsz', 0):>5d} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {r['status']:>10s} "
              f"{r.get('time_h', 0):>5.1f}h")
        if r.get("params_M"):
            print(f"{'':<26s} {r['params_M']}M params  built via [{r.get('variant')}]")
        if r.get("error"):
            print(f"{'':<26s} {r['error']}")

    print(f"\nSummary saved: {summary_path}")
