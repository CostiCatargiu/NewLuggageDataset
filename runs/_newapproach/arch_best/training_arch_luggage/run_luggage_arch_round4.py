#!/usr/bin/env python3
"""
Luggage Architecture Ablation — ROUND 4 (3 runs @640).

=============================================================================
WHAT THE 27-RUN CORPUS ACTUALLY ESTABLISHED
=============================================================================
Assigner-matched baseline (v12s_default2, satal=true/topk=12/a=0.6/b=5.0 — the
SAME assigner every arch run uses):
    82.77 mAP50 | 57.63 mAP50-95 | sm50 79.59 | sm50-95 52.17
    md50-95 67.55 | lg50-95 59.85 | P 80.92 | R 75.24

Current best:  arch_dysample_p2_gctx2 = 57.91  (DySample head + P2 + gctx2 x4, b36)
    vs baseline: +0.28 overall, +0.54 sm50, +0.58 sm50-95, +1.82 lg
                 -1.29 md   <-- its ONLY deficit
Runner-up:     arch_gctx22 = 57.84 (P2 + gctx2 x4, b28) — best small (80.54/53.12),
                                    but weak large (57.92)

THREE FACTS THIS ROUND IS BUILT ON
----------------------------------
(1) gctx2 (avg+max) > gctx (avg) is the ONLY twice-replicated lever:
        gctx5   57.67 -> gctx22             57.84   (+0.17, P2 head)
        dys_p2_gctx5 57.63 -> dys_p2_gctx2  57.91   (+0.28, DySample+P2 head)
    Its `reduction` hyperparameter has NEVER been touched (see run [3]).

(2) STACKING SAME-PURPOSE LEVERS FAILS; STACKING DIFFERENT-DEFICIT LEVERS WORKS.
    Six pairings of "proven winners" that all buy the same thing (small-object
    context) landed BELOW the baseline:
        gctx2+coordatt 57.08 | gctx2+dysample2 57.11 | dysample+coordatt 57.31
        gctx2+detail_aux 57.35 | gctx+P2Fuse 57.57 | (round-1 combos)
    The ONE combo that worked pairs levers fixing DIFFERENT size buckets:
        gctx22 (small: +0.95 sm50-95, large: -1.92)
      + DySample (large: +2.24 standalone)
      = dysample_p2_gctx2 (+0.07 overall vs gctx22, +3.74 LARGE)
    => Design rule for this round: only combine modules whose per-size profiles
       are COMPLEMENTARY, never modules that both target small objects.

(3) MEDIUM OBJECTS ARE COMPLETELY UNATTACKED.
    ZERO of 27 arch runs beat the baseline on md50-95 (67.55). It is the only
    bucket with a clean sweep against us, and it is 39.9% of test instances —
    MORE than small (39.7%). The best model loses 1.29 there.
    The one module with strong medium is ZGDSConv (snake conv): md 67.32 and
    lg 62.56 as a 3-level standalone (arch_dsconv, 57.77, b48) — best medium AND
    best large of the overnight-2 round. It has NEVER met the P2 head.

=============================================================================
RUNS — ORDERED BY CONFIDENCE OF IMPROVING ON 57.91
=============================================================================
  [1] arch_dysgctx2_r4       (b36)  P(improve) ~50%  |  expected size ~+0.1..0.2
        TUNE THE ONE REPLICATED LEVER — the safest bet in the set.
        ZGGlobalContext2(c1, c2, reduction=8) builds its MLP bottleneck as
        hidden = max(8, c1 // reduction). After the s-scale width multiplier
        (0.50) the P2/P3 levels carry c1=128, so hidden = 16 — a 16-channel
        bottleneck compressing a 2*128 = 256-dim avg+max descriptor. That is
        very tight, and `reduction` has never been varied on any dataset.
        reduction=4 doubles it for a few thousand params and no real FLOPs.
        WHY IT RANKS FIRST: changes exactly ONE variable; runs at the SAME
        batch 36 as its reference so there is no step-count confound; cannot
        destabilise (same topology, wider MLP); cheapest run of the three.
        WHY IT MIGHT NOT: `reduction` is a mild hyperparameter — the effect
        could easily land inside the ~0.15 noise floor.

  [2] arch_dysgctx2_dsconv   (b32)  P(improve) ~40%  |  expected size ~+0.3..0.5
        BEST MODEL + ZGDSConv at P3/P4 — the biggest prize if it lands.
        Targets a MEASURED deficit: the best model loses -1.29 on medium, the
        only bucket where it is behind the baseline, and medium is 39.9% of
        test instances (MORE than small at 39.7%). ZGDSConv is the only module
        in the corpus strong on medium (67.32) AND large (62.56), and it has
        never met the P2 head. Snake conv is also a literal shape prior:
        94% of boxes are tall, mean h/w = 2.69, mean box 33x72px.
        WHY IT MIGHT NOT: six of seven combos have failed, and this one stacks
        two zero-gated residual modules IN SERIES on the same levels (gctx2
        then dsconv at P3/P4) — structurally the same pattern that sank
        gctx2_coordatt (57.08) and gctx2_detail_aux (57.35). Two gamma gates in
        sequence can also simply cancel.

  [3] arch_levelspec         (b32)  P(improve) ~30%  |  highest INFORMATION value
        PER-LEVEL MODULE SPECIALISATION — the only untested structural idea.
        All 27 runs so far apply ONE module family UNIFORMLY to every pyramid
        level. The per-size evidence says that is wrong:
            gctx2    helps small (+0.95 sm50-95), hurts large (-1.92)
            ZGDSConv helps medium + large (67.32 / 62.56)
        So route them by level instead of stacking them everywhere:
            gctx2    -> P2, P3   (where small objects live)
            ZGDSConv -> P4, P5   (where medium/large live)
        Doubles as the control for run [2]: comparing them isolates whether
        gctx2 on P4/P5 is a net NEGATIVE (gctx22 large = 57.92 vs gctx5 large
        = 60.20 hints that avg+max may hurt at coarse levels).
        WHY IT RANKS LAST: the earlier `asym` run already tested "remove gctx
        from P4/P5" and large got WORSE (57.32 vs 58.14). Different module
        (gctx not gctx2) on a decoupled head, so not decisive — but it points
        the wrong way, and this run strips the best-validated module off half
        the pyramid. A negative result here is still worth having.

QUEUE RATIONALE: r4 first because it is fastest and safest, so an interrupted
overnight still banks the cleanest result. levelspec last because it is the one
whose NEGATIVE outcome is publishable on its own.
Note the ordering by EXPECTED VALUE would swap [1] and [2]: 40% x 0.4pt beats
50% x 0.15pt. Ordered here by P(improve), as requested.

Deliberately NOT included: more same-purpose combos (rule 2), anything on the
coordatt axis (coordatt3 is 14th-15th on large and every coordatt combo failed),
ZGStar (56.82, below baseline), DetectDeepCls (57.49, below baseline).

=============================================================================
NOTES
=============================================================================
  * Batch is per-config and VRAM-driven, NOT pinned. With EPOCHS fixed, lower
    batch = more optimizer steps, so heavier 4-level configs get ~1.5-2x the
    steps of the 3-level ones. Across the corpus r(batch, mAP50-95) = -0.52.
    Run [3] therefore keeps b36 to match its reference exactly; runs [1]/[2]
    drop to b32 for the extra modules and are the LESS favoured side of that
    gradient (i.e. any gain they show is conservative).
  * arch_gctx2_dysample collapsed at b62 (41.87). Nothing here exceeds b36.
  * Arg passing verified in tasks.py: for this module set parse_model builds
    `args = [c1, c2, *args[1:]]`, so [256, 4] -> (c1, c2, reduction=4) and
    [512, 9] -> (c1, c2, k=9). Both ZGGlobalContext2 and ZGDSConv are in the
    width-scaling set, so channel args stay nominal (no {c256} templating).
  * Loss/assigner untouched: use_satal / tal_topk / alpha / beta inherited from
    defaults -> directly comparable to the whole arch family.

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_round4.py                     # all 3, in order
  python run_luggage_arch_round4.py --build-only        # construct only, no training
  python run_luggage_arch_round4.py arch_dysgctx2_dsconv --with-test
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
PROJECT_DIR = "runs_luggage_arch4"
YAML_DIR = "arch_yamls_luggage"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 32
EPOCHS = 70
SEED = 0

SCALE_WIDTH = 0.50
MAX_CHANNELS = 1024


def scaled_ch(c):
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

# DySample neck — part of the current best model (arch_dysample_p2_gctx2).
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

# [1] Best model + ZGDSConv at P3/P4 (attacks the -1.29 medium deficit).
#     gctx2 stays on all 4 levels; snake conv is inserted AFTER it on P3/P4 so
#     the Detect reads P2(gctx2) / P3(gctx2+snake) / P4(gctx2+snake) / P5(gctx2).
TAIL_DYSGCTX2_DSCONV = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 + avg+max global ctx
  - [25, 1, ZGDSConv, [256, 9]]                     # 28  P3 + snake conv k=9
  - [26, 1, ZGDSConv, [512, 9]]                     # 29  P4 + snake conv k=9
  - [[24, 28, 29, 27], 1, Detect, [nc]]             # 30  P2 / P3+snake / P4+snake / P5
"""

# [2] Per-level specialisation: gctx2 on the FINE levels only (P2/P3),
#     ZGDSConv on the COARSE levels only (P4/P5). No level gets both.
TAIL_LEVELSPEC = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2      (small)
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2      (small)
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake conv (medium)
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake conv (large)
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [3] Byte-identical to arch_dysample_p2_gctx2 except reduction 8 -> 4.
#     hidden = max(8, c1//reduction): P2/P3 bottleneck 16ch -> 32ch.
TAIL_DYSGCTX2_R4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256, 4]]             # 24  P2 + gctx2 (reduction=4)
  - [14, 1, ZGGlobalContext2, [256, 4]]             # 25  P3 + gctx2 (reduction=4)
  - [17, 1, ZGGlobalContext2, [512, 4]]             # 26  P4 + gctx2 (reduction=4)
  - [20, 1, ZGGlobalContext2, [1024, 4]]            # 27  P5 + gctx2 (reduction=4)
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# =============================================================================
# ASSEMBLE
# =============================================================================
RUNS = [
    {
        "name": "arch_dysgctx2_r4",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYSGCTX2_R4,
        "batch": 36,          # matches arch_dysample_p2_gctx2 exactly -> no step-count confound
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "~50% -- highest P(improve). One variable, on the only twice-replicated lever.",
        "desc": "[1/3] b36 — gctx2 reduction 8->4 (single-variable, safest, cheapest)",
        "ref": "arch_dysample_p2_gctx2 = 57.91 (reduction=8)",
    },
    {
        "name": "arch_dysgctx2_dsconv",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYSGCTX2_DSCONV,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "~40% -- lower P(improve) but LARGEST expected magnitude (+0.3..0.5).",
        "desc": "[2/3] b32 — best model + snake conv @P3P4 (attacks the -1.29 medium)",
        "ref": "arch_dysample_p2_gctx2 = 57.91 (md 66.26)",
    },
    {
        "name": "arch_levelspec",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LEVELSPEC,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "~30% -- lowest P(improve), HIGHEST information value either way.",
        "desc": "[3/3] b32 — gctx2 @P2P3 + snake @P4P5 (per-level specialisation)",
        "ref": "arch_dysample_p2_gctx2 = 57.91 (uniform gctx2 x4)",
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
    """Channel-arg fallback for modules outside parse_model's width-scaling set.

    Nothing in this round needs it (ZGGlobalContext2 and ZGDSConv are both
    width-scaled), so every template returns a single "as-written" variant.
    Kept so a future CoordinateAttention run can drop straight in.
    """
    if any(k in yaml_template for k in ("{c256}", "{c512}", "{c1024}")):
        return [("ch=nominal", CH_NOMINAL), ("ch=pre-scaled", CH_SCALED)]
    return [("as-written", {})]


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
    errors = []

    for label, subs in build_variants(run["yaml"]):
        text = run["yaml"].format(**subs)
        save_yaml(text, yaml_path)
        model = None
        try:
            model = YOLO(yaml_path)

            det = model.model.model[-1]
            nl = getattr(det, "nl", None)
            strides = [int(s) for s in model.model.stride.tolist()]

            exp_nl, exp_st = run.get("levels"), run.get("strides")
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
    seed = run.get("seed", SEED)

    print(f"\n{'#' * 76}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# beat: {run.get('ref', '-')}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {seed}")
    print(f"{'#' * 76}\n")

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
                tr = test_model.val(data=DATA_YAML, split="test",
                                    imgsz=imgsz, batch=batch)
                if tr is not None:
                    rd = tr.results_dict
                    metrics["test"] = {
                        "mAP50": round(rd.get("metrics/mAP50(B)", 0), 5),
                        "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0), 5),
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

    os.makedirs(YAML_DIR, exist_ok=True)
    mode = "BUILD-ONLY (no training)" if build_only else f"epochs={EPOCHS}, seed={SEED}"
    print(f"\n{'=' * 78}")
    print(f"  LUGGAGE ARCH -- ROUND 4 ({len(runs)} runs @640)")
    print(f"  Dataset: 3 classes | 39.7% small, 39.9% medium, 20.4% large | mean box 33x72px")
    print(f"  {mode}")
    print(f"  baseline v12s_default2 : 82.77 / 57.63 / sm 79.59-52.17 / md 67.55 / lg 59.85")
    print(f"  best so far dys_p2_gctx2: 82.99 / 57.91 / sm 80.13-52.75 / md 66.26 / lg 61.66")
    print(f"{'=' * 78}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3}] {r['desc']}")
    print(f"{'=' * 78}\n")

    all_results = [run_experiment(r, with_test=with_test, build_only=build_only)
                   for r in runs]

    suffix = "round4_build_check" if build_only else "round4_summary"
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 78}")
    print("  RESULTS")
    print(f"{'=' * 78}")
    print(f"{'Name':<26s} {'batch':>6s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>10s} {'Time':>6s}")
    print("-" * 78)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<26s} {r.get('batch', 0):>6d} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {r['status']:>10s} {r.get('time_h', 0):>5.1f}h")
        if r.get("params_M"):
            print(f"{'':<26s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<26s} {r['error']}")
    print(f"\nSummary saved: {summary_path}")
