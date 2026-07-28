#!/usr/bin/env python3
"""
Luggage Architecture Ablation — OVERNIGHT 2 (8 runs @640).

=============================================================================
STRATEGY: 5 proven-combo + 3 new-module
=============================================================================
The prior 17 640px runs found 4 individual winners but showed ZERO additivity
when stacking them naively (dysample_p2_gctx5 = 57.63 did NOT beat gctx22 =
57.84). However, many two-way and three-way pairings of the winners were
NEVER TESTED:

  Proven winners (single-lever, mAP50-95 on test_full_dataset):
    gctx22     = 57.84  ZGGlobalContext2 (avg+max pool) — the #1 winner
    coordatt3  = 57.78  CoordinateAttention per level    — #2
    detail_aux4= 57.64  ZGSmallDetail + DetectAuxDual    — #3 (strong on large)
    dysample   = 57.61  DySample head (content-aware up)  — #4
    gctx5      = 57.67  P2 head + ZGGlobalContext         — #5

  Combinations ALREADY tested (and their result):
    dysample + P2 + gctx   = 57.63  (no additivity: gctx + dys cancelled)
    gctx + P2Fuse          = 57.57  (no additivity)
    P2 + gctx              = 57.67  (worked: = gctx5)
    P2 + coordatt          = 57.78  (worked: = coordatt3)

  Combinations NEVER tested (this script):
    [1] gctx2 + dysample         — the #1 lever (global ctx) + #4 (upsampling)
    [2] gctx2 + coordatt         — the #1 (global ctx) + #2 (spatial attn)
    [3] dysample + coordatt      — #4 (upsampling) + #2 (spatial attn)
    [4] dysample + gctx2 + P2    — #4 + #1 + P2 head (with gctx2 instead of gctx)
    [5] gctx2 + detail_aux       — #1 + aux supervision (targets scoring quality)

  The key hypothesis: the prior stacking test (dys+P2+gctx = 57.63) used
  ZGGlobalContext (avg-only). The #1 winner was ZGGlobalContext2 (avg+max).
  Its +0.24 over gctx may come from a COMPLEMENTARY signal (max-pool salient
  peaks) that DySample/CoordAtt don't provide, so gctx2 combos may show
  additivity where gctx combos didn't.

  The 3 speculative runs test genuinely NEW axes:
    [6] DetectDeepCls          — deeper cls tower (4-block vs stock 2-block)
    [7] ZGStar per level       — multiplicative feature mixing (StarNet 2024)
    [8] ZGDSConv@P3P4          — dynamic snake conv (shape-following for tall objs)

=============================================================================
RUNS (ordered by confidence)
=============================================================================
  [1] arch_gctx2_dysample    — GCtx2 x3 + DySample head (NEVER combined)
  [2] arch_gctx2_coordatt    — GCtx2 x3 + CoordAtt x3 (NEVER combined)
  [3] arch_dysample_coordatt — DySample head + CoordAtt x3 (NEVER combined)
  [4] arch_dysample_p2_gctx2 — DySample + P2 + GCtx2 (gctx2 NOT gctx)
  [5] arch_gctx2_detail_aux  — GCtx2 x3 + SmallDetail + DetectAuxDual
  [6] arch_deepcls           — DetectDeepCls: 4-block cls tower (head capacity)
  [7] arch_star              — ZGStar x3: multiplicative mixing (NEW nonlinearity)
  [8] arch_dsconv             — ZGDSConv@P3P4: snake conv (shape-following)

=============================================================================
MODULE REGISTRATION
=============================================================================
All modules are imported AND width-scaled in nn_modules/tasks.py:
  ZGGlobalContext2, ZGSmallDetail, DySample, DetectAuxDual, DetectDeepCls,
  ZGStar, ZGDSConv — all in the scaling/dispatch sets.
  CoordinateAttention — has custom dispatch in tasks.py but is NOT width-scaled;
  uses {c256}/{c512}/{c1024} templates with build_variants() fallback
  (nominal tried first, pre-scaled as fallback).

Loss/assigner path untouched: use_satal / tal_topk / alpha / beta inherited
from defaults (use_satal=true / topk=12 / a=0.6 / b=5.0), so these stay
comparable to the earlier ARCH runs.

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/
  (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_overnight2.py                            # all 8
  python run_luggage_arch_overnight2.py arch_gctx2_dysample        # single
  python run_luggage_arch_overnight2.py --build-only               # construct
  python run_luggage_arch_overnight2.py --with-test                # + test eval
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
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21    # Detect index in stock yolov12s
BATCH = 58             # global fallback
EPOCHS = 70
SEED = 0
AUX_W = 0.5

# Width scaling for modules NOT in parse_model's scaling set (CoordinateAttention).
# Must match the `scales:` s entry: [0.50, 0.50, 1024].
SCALE_WIDTH = 0.50
MAX_CHANNELS = 1024


def scaled_ch(c):
    """Reproduce ultralytics parse_model channel scaling: make_divisible(min(c, max)*w, 8)."""
    v = min(c, MAX_CHANNELS) * SCALE_WIDTH
    return int(math.ceil(v / 8) * 8)


# Substitution axes for CoordinateAttention build-variant fallback
CH_NOMINAL = {"c256": 256, "c512": 512, "c1024": 1024}
CH_SCALED = {k: scaled_ch(v) for k, v in CH_NOMINAL.items()}

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

# ============================================================================
# TAILS — PROVEN COMBINATIONS (runs 1-5)
# ============================================================================

# [1] GCtx2 x3 levels + DySample upsampling. (Detect @ 24)
#     gctx22 (57.84) used stock head; dysample (57.61) used DySample head.
#     NEVER combined: does content-aware upsampling feed better features
#     into the global context that made gctx22 #1?
TAIL_GCTX2_DYSAMPLE = """  - [14, 1, ZGGlobalContext2, [256]]                # 21  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 22  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 23  P5 + avg+max global ctx
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [2] GCtx2 x3 + CoordinateAttention x3. (Detect @ 27)
#     gctx22 adds GLOBAL (channel-wise) context; coordatt adds SPATIAL
#     (H/W-factorized position). Orthogonal mechanisms — gctx operates on
#     pool(HW) -> channel, coordatt operates on H-pool and W-pool -> spatial.
#     NEVER combined.
TAIL_GCTX2_COORDATT = """  - [14, 1, ZGGlobalContext2, [256]]                # 21  P3 + global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 22  P4 + global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 23  P5 + global ctx
  - [21, 1, CoordinateAttention, [{c256}]]          # 24  P3 + coord attn
  - [22, 1, CoordinateAttention, [{c512}]]          # 25  P4 + coord attn
  - [23, 1, CoordinateAttention, [{c1024}]]         # 26  P5 + coord attn
  - [[24, 25, 26], 1, Detect, [nc]]                 # 27
"""

# [3] DySample head + CoordinateAttention x3. (Detect @ 24)
#     DySample improves upsampling quality (thin-edge preservation); CoordAtt
#     improves H/W spatial attention (94% tall objects). Different axes.
#     NEVER combined.
TAIL_DYSAMPLE_COORDATT = """  - [14, 1, CoordinateAttention, [{c256}]]          # 21  P3 + coord attn
  - [17, 1, CoordinateAttention, [{c512}]]          # 22  P4 + coord attn
  - [20, 1, CoordinateAttention, [{c1024}]]         # 23  P5 + coord attn
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [4] DySample head + P2 head + GCtx2 x4 levels. (Detect @ 28)
#     The prior stacking test (dysample_p2_gctx5 = 57.63) used ZGGlobalContext
#     (avg-only). The +0.24 lift from gctx -> gctx2 (avg+max) may come from
#     a complementary max-pool signal that DySample does not provide.
#     Also: DySample upsample INTO the P2 level (content-aware stride 4->8).
TAIL_DYS_P2_GCTX2 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4 (content-aware)
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + avg+max global ctx
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 + avg+max global ctx
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [5] GCtx2 x3 + ZGSmallDetail + DetectAuxDual. (Detect @ 25)
#     gctx22 (57.84) was the overall winner. detail_aux4 (57.64) was strong
#     on large objects (mAP50_large 86.87, best in the set). GCtx2 enriches
#     global context; aux supervision independently forces the backbone to
#     preserve detail for small objects. Different mechanisms.
#     NEVER combined.
TAIL_GCTX2_DETAIL_AUX = f"""  - [14, 1, ZGGlobalContext2, [256]]                # 21  P3 + avg+max global ctx
  - [17, 1, ZGGlobalContext2, [512]]                # 22  P4 + avg+max global ctx
  - [20, 1, ZGGlobalContext2, [1024]]               # 23  P5 + avg+max global ctx
  - [21, 1, ZGSmallDetail, [256, 3, 5]]             # 24  P3 detail guard
  - [[24, 22, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 25
"""

# ============================================================================
# TAILS — SPECULATIVE (runs 6-8)
# ============================================================================

# [6] DetectDeepCls — 4-block cls tower (stock=2). (Detect @ 21)
#     Never run on luggage. The 3 classes are all tall (AR 2.23-2.96), so the
#     classifier must discriminate on texture/detail, not shape. A deeper cls
#     tower gives more capacity for those subtle differences.
TAIL_DEEPCLS = """  - [[14, 17, 20], 1, DetectDeepCls, [nc]]          # 21  4-block cls tower
"""

# [7] ZGStar per level — multiplicative feature mixing. (Detect @ 24)
#     Completely different nonlinearity from attention/context (no spatial maps,
#     no pooling). Element-wise product of two projections = implicit polynomial
#     feature expansion. NEVER run on any dataset.
TAIL_STAR = """  - [14, 1, ZGStar, [256, 4]]                      # 21  P3 + star mixing
  - [17, 1, ZGStar, [512, 4]]                      # 22  P4 + star mixing
  - [20, 1, ZGStar, [1024, 4]]                     # 23  P5 + star mixing
  - [[21, 22, 23], 1, Detect, [nc]]                 # 24
"""

# [8] ZGDSConv at P3/P4 — Dynamic Snake Convolution. (Detect @ 23)
#     Kernel snakes along elongated structures with cumulative per-tap offsets.
#     94% tall objects = direct shape prior match. Pure PyTorch (F.grid_sample),
#     avoids the torchvision deform_conv2d SIGSEGV that killed ZGDCN.
#     Applied at P3/P4 only (tall objects strongest there); P5 left clean.
TAIL_DSCONV = """  - [14, 1, ZGDSConv, [256, 9]]                    # 21  P3 + snake conv k=9
  - [17, 1, ZGDSConv, [512, 9]]                    # 22  P4 + snake conv k=9
  - [[21, 22, 20], 1, Detect, [nc]]                 # 23  P3(snake)/P4(snake)/P5(clean)
"""

# =============================================================================
# ASSEMBLE ARCHITECTURES
# =============================================================================
RUNS = [
    # --- PROVEN COMBOS (5 runs) ---
    # {
    #     "name": "arch_gctx2_dysample",
    #     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTX2_DYSAMPLE,
    #     "batch": 62,
    #     "levels": 3,
    #     "strides": [8, 16, 32],
    #     "desc": "[1/8] GCtx2 + DySample head — #1 winner + content-aware upsampling",
    # },
    {
        "name": "arch_gctx2_coordatt",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GCTX2_COORDATT,
        "batch": 48,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[2/8] GCtx2 + CoordAtt — global (channel) + spatial (H/W) attention",
    },
    {
        "name": "arch_dysample_coordatt",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYSAMPLE_COORDATT,
        "batch": 58,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[3/8] DySample + CoordAtt — edge-preserving upsample + H/W attention",
    },
    {
        "name": "arch_dysample_p2_gctx2",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_DYS_P2_GCTX2,
        "batch": 36,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "desc": "[4/8] DySample + P2 + GCtx2 — retry stacking with gctx2 (not gctx)",
    },
    {
        "name": "arch_gctx2_detail_aux",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_GCTX2_DETAIL_AUX,
        "batch": 52,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[5/8] GCtx2 + SmallDetail + AuxDual — #1 winner + dual supervision",
    },
    # --- SPECULATIVE (3 runs) ---
    {
        "name": "arch_deepcls",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEEPCLS,
        "batch": 58,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[6/8] DetectDeepCls — 4-block cls tower (head capacity for 3 tall classes)",
    },
    {
        "name": "arch_star",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_STAR,
        "batch": 52,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[7/8] ZGStar x3 — multiplicative mixing (polynomial expansion, StarNet 2024)",
    },
    {
        "name": "arch_dsconv",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DSCONV,
        "batch": 48,
        "levels": 3,
        "strides": [8, 16, 32],
        "desc": "[8/8] ZGDSConv@P3P4 — dynamic snake conv for 94%-tall objects",
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
    """Generate substitution variants for CoordinateAttention channel args.

    CoordinateAttention is NOT in parse_model's width-scaling set, so its YAML
    channel args must be pre-scaled manually. Try nominal first (in case the
    module gets added to the scaling set), then pre-scaled as fallback.

    Templates without {c256}/{c512}/{c1024} placeholders return a single
    "as-written" variant with no substitutions — i.e. a no-op for configs
    that don't use CoordinateAttention.
    """
    if any(k in yaml_template for k in ("{c256}", "{c512}", "{c1024}")):
        return [
            ("ch=nominal", CH_NOMINAL),
            ("ch=pre-scaled", CH_SCALED),
        ]
    return [("as-written", {})]


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping Detect layer if index changed.

    For DetectDeepCls, cv3 is rebuilt so only cv2 (box) weights transfer
    from the pretrained Detect — cls trains fresh (intended behavior).
    For DetectAuxDual, aux towers are train-only and initialize fresh.
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
        text = run["yaml"].format(**subs) if subs else run["yaml"]
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
    batch = run.get("batch", BATCH)

    print(f"\n{'#' * 72}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Batch {batch}  Imgsz {IMG_SIZE}  Epochs {EPOCHS}  seed {SEED}")
    print(f"{'#' * 72}\n")

    start = time.time()
    model = None
    try:
        model, yaml_path, variant = build_model(run)

        if build_only:
            n_params = sum(p.numel() for p in model.model.parameters())
            print(f"  [build-only] {run['name']}: {n_params / 1e6:.2f}M params, "
                  f"yaml={yaml_path}, variant=[{variant}]")
            return {"name": run["name"], "status": "BUILD-OK",
                    "variant": variant, "params_M": round(n_params / 1e6, 2),
                    "time_h": 0.0}

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

        metrics = {"name": run["name"], "status": "OK", "variant": variant,
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
    print(f"  LUGGAGE ARCH — OVERNIGHT 2 ({len(runs)} runs @640)")
    print(f"  Dataset: luggage (3 classes, 40% small, 94% tall, AR=2.69)")
    print(f"  {mode}  (per-config batch below)")
    print(f"  Beat-target: gctx22 = 57.84  |  coordatt3 = 57.78  |  baseline = 57.63")
    print(f"{'=' * 78}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{IMG_SIZE}] {r['desc']}")
    print(f"{'=' * 78}\n")

    all_results = []
    for run in runs:
        all_results.append(run_experiment(run, with_test=with_test,
                                          build_only=build_only))

    suffix = "overnight2_build_check" if build_only else "overnight2_summary"
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
        if r.get("variant"):
            print(f"{'':<28s} built via [{r['variant']}]")
        if r.get("params_M"):
            print(f"{'':<28s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<28s} {r['error']}")

    print(f"\nSummary saved: {summary_path}")
