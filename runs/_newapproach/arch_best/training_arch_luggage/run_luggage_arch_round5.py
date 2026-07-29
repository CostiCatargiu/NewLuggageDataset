#!/usr/bin/env python3
"""
Luggage Architecture Ablation — ROUND 5 (5 runs @640): DISSECT arch_levelspec.

=============================================================================
WHERE WE ARE
=============================================================================
Assigner-matched baseline  v12s_default2 (satal=true topk=12 a=0.6 b=5.0):
    82.77 / 57.63 | sm 79.59-52.17 | md 88.66-67.55 | lg 81.18-59.85 | P 80.92 R 75.24

NEW BEST (round 4)  arch_levelspec = 58.02  (b32)
    gctx2 @P2,P3  +  ZGDSConv(k=9) @P4,P5   on the DySample neck + P2 head
    vs baseline: +0.57 mAP50, +0.39 50-95, +0.55 sm50, +0.51 sm50-95,
                 +0.56 md50, -0.30 md50-95, +0.51 lg50, +1.13 lg50-95,
                 +1.69 P, -1.48 R
    -> positive on 8 of 10 metrics; the first architecture in 30 runs to do that.

THE ONE BIG FINDING: ROUTING BEATS STACKING.
    arch_dysgctx2_dsconv4  57.53   gctx2 AND dsconv on the SAME levels (P3,P4)
    arch_levelspec         58.02   gctx2 and dsconv on SEPARATE levels
    Same two modules, same batch 32, same base -> +0.49 from routing alone,
    larger than any single module's contribution in the whole corpus.
    (7 of 8 same-level stacks have now failed; every level-separated one worked.)

=============================================================================
WHY THIS ROUND DISSECTS RATHER THAN ADDS
=============================================================================
30 architectures have bought +0.39 over baseline. Adding a 31st module is worth
~+0.1-0.4. But arch_levelspec changed TWO things at once versus its predecessor
(arch_dysample_p2_gctx2, 57.91) and nobody has separated them:
    (a) REMOVED gctx2 from P4/P5
    (b) ADDED ZGDSConv(k=9) to P4/P5
Runs [1] and [2] below split that. If (a) is the whole effect, the recipe gets
simpler AND cheaper, and every future run inherits it.

KERNEL/OBJECT-SCALE MISMATCH (drives runs [2] and [3])
    mean box = 33w x 72h px. In feature-map cells:
        P3 (stride 8):  4.1 x  9.0 cells   <- k=9 snake spans exactly the height
        P4 (stride 16): 2.1 x  4.5 cells   <- k=9 is 2x the object
        P5 (stride 32): 1.0 x  2.25 cells  <- k=9 is 4x the object
    ZGDSConv snakes a 1D kernel of k taps along an elongated structure. At P5 a
    9-tap path is four times longer than the object it is supposed to follow, so
    most taps sample background. levelspec still won with that mismatch, which
    is exactly why the (a)-vs-(b) split matters.

THE NEW BOTTLENECK IS RECALL (drives run [5])
    levelspec  P 82.61 / R 73.76      baseline  P 80.92 / R 75.24
    It is the most precise model in the corpus and the least sensitive. mAP is an
    area under the PR curve, so a recall ceiling truncates the curve and caps mAP
    no matter how good precision is. Recovering ~1.5 points of recall without
    losing the precision is worth more than another feature module.
    Aux supervision is the corpus's high-recall mechanism (detail_aux4 R 75.10,
    deepp3aux2 R 75.30) AND it is a LOSS-PATH mechanism, not another zero-gated
    residual on the same tensor -> the "don't stack on one level" rule does not
    apply to it.

=============================================================================
RUNS — all @640, all b32 (b28 for [5]) so they are mutually comparable
        AND directly comparable to arch_levelspec (58.02, b32)
=============================================================================
  [1] arch_ls_clean    gctx2 @P2,P3 ; P4/P5 get NOTHING.
        THE MISSING CONTROL. Isolates effect (a) from effect (b). If this lands
        at ~58.0 then ZGDSConv contributes nothing and levelspec's win was
        entirely "stop putting gctx2 on the coarse levels" — a simpler, cheaper,
        more defensible result. If it lands ~57.6 then dsconv is load-bearing.
        Fewest parameters of any run in this round. Either outcome is decisive.

  [2] arch_ls_k5       gctx2 @P2,P3 ; ZGDSConv(k=5) @P4,P5.
        Single variable vs levelspec: k 9 -> 5. At P4 the mean object is 4.5
        cells tall, at P5 2.25 — a 5-tap path is a far better match than 9.
        If k=5 > k=9 the kernel was oversized and the same fix applies wherever
        dsconv is used.

  [3] arch_ls_shift    gctx2 @P2 only ; ZGDSConv(k=9) @P3,P4,P5.
        Moves the routing boundary down one level, which ALSO puts k=9 at P3
        where 9 taps == the 9-cell mean object height (the one level where the
        kernel is scale-matched). Tests boundary placement and kernel match in
        the same run. gctx2 is left only at P2, the level where small objects
        genuinely live.

  [4] arch_ls_nodys    levelspec with the STOCK nearest-neighbour neck.
        DySample as a standalone scored 57.61 vs the 57.63 baseline — i.e.
        literally nothing — yet every recent run inherits it. It has never been
        ablated inside levelspec. If this matches 58.02, drop DySample: fewer
        params, no content-aware upsampler to justify. If it drops, DySample is
        load-bearing only in combination, which is itself worth reporting.

  [5] arch_ls_aux      levelspec + DetectAuxDual (main = the 4 enhanced levels,
        aux = the 4 CLEAN neck levels), aux_weight 0.5, b28.
        Attacks the -1.48 recall deficit, levelspec's only real weakness.
        Aux towers are train-only and dropped at inference -> zero deployed cost.
        Main sees context-enriched features, aux sees raw neck features, so the
        backbone must satisfy both -> the documented DetectAuxDual design.

WHAT THIS ROUND DELIBERATELY DOES NOT DO
    No new modules. No same-level stacking (7/8 failed). Nothing on the coordatt
    axis (14th-15th on large, every combo failed). No reduction retune
    (arch_dysgctx2_r42 settled it: reduction=8 beats 4 by 0.36).

HONEST EXPECTATION
    Runs [1] and [4] are SIMPLIFICATIONS — their best outcome is "same score,
    fewer parts", which is worth having but will not move the number. Runs [2]
    and [3] are worth ~+0.1-0.3 if the scale-mismatch reading is right. Run [5]
    is the only one that could plausibly add >+0.3, via recall.
    The architecture axis is near its ceiling (30 runs -> +0.39). The bigger
    measured levers remain elsewhere: 896 resolution (+1.2..1.4, confirmed 3x),
    satal=false (+0.84 overall / +6.39 large, measured on identical arch), and
    model scale s->m (untested).

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_round5.py                  # all 5, in order
  python run_luggage_arch_round5.py --build-only     # construct only, no training
  python run_luggage_arch_round5.py arch_ls_clean --with-test
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
PROJECT_DIR = "runs_luggage_arch5"
YAML_DIR = "arch_yamls_luggage"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 32
EPOCHS = 70
SEED = 0
AUX_W = 0.5

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

# --- TAILS ---
# Shared prefix for every run: P3 -> stride-4 upsample, concat backbone P2,
# build the P2 head. Node numbering is identical across all five tails.

# [1] gctx2 @P2,P3 ; P4/P5 CLEAN. The missing control.  (Detect @ 26, 4 levels)
TAIL_LS_CLEAN = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [[24, 25, 17, 20], 1, Detect, [nc]]             # 26  P4/P5 raw neck features
"""

# [2] gctx2 @P2,P3 ; ZGDSConv k=5 @P4,P5.  (Detect @ 28, 4 levels)
TAIL_LS_K5 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGDSConv, [512, 5]]                     # 26  P4 + snake k=5 (obj 4.5 cells)
  - [20, 1, ZGDSConv, [1024, 5]]                    # 27  P5 + snake k=5 (obj 2.25 cells)
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [3] gctx2 @P2 only ; ZGDSConv k=9 @P3,P4,P5.  (Detect @ 28, 4 levels)
TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9 (obj 9.0 cells: MATCHED)
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [4] levelspec exactly, but on the STOCK nearest neck (see HEAD_STOCK).
TAIL_LS_NODYS = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# [5] levelspec + DetectAuxDual. 2*nl = 8 inputs:
#     main = [24,25,26,27] (enhanced)  |  aux = [23,14,17,20] (clean neck)
TAIL_LS_AUX = f"""  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 28
"""

# =============================================================================
# ASSEMBLE
# =============================================================================
RUNS = [
    {
        "name": "arch_ls_clean",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_CLEAN,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "decisive either way -- isolates 'removed gctx2' from 'added dsconv'",
        "desc": "[1/5] b32 — gctx2 @P2P3, P4/P5 CLEAN (the missing control)",
        "ref": "arch_levelspec 58.02 | arch_dysample_p2_gctx2 57.91",
    },
    {
        "name": "arch_ls_shift",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "~40% -- boundary + puts k=9 at the one scale-matched level (P3)",
        "desc": "[2/5] b32 — gctx2 @P2 only, snake k=9 @P3P4P5 (boundary shift)",
        "ref": "arch_levelspec 58.02",
    },
    {
        "name": "arch_ls_aux",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_AUX,
        "batch": 28,          # aux towers are train-only but cost VRAM
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "aux_levels": 4,      # DetectAuxDual: main half must differ from aux half
        "confidence": "~35% -- the only run that could add >+0.3, via the -1.48 recall gap",
        "desc": "[3/5] b28 — levelspec + DetectAuxDual (main=enhanced, aux=clean neck)",
        "ref": "arch_levelspec 58.02 (P 82.61 / R 73.76)",
    },
    {
        "name": "arch_ls_k5",
        "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_K5,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "~35% -- single variable k 9->5; k=9 is 2-4x the object at P4/P5",
        "desc": "[4/5] b32 — snake kernel k=9 -> k=5 at P4/P5 (scale match)",
        "ref": "arch_levelspec 58.02 (k=9)",
    },
    {
        "name": "arch_ls_nodys",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_LS_NODYS,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "confidence": "simplification -- DySample standalone was worth -0.02 vs baseline",
        "desc": "[5/5] b32 — levelspec on the stock nearest neck (drop DySample)",
        "ref": "arch_levelspec 58.02 (DySample neck)",
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

    Nothing here needs it (ZGGlobalContext2, ZGDSConv and DetectAuxDual are all
    registered), so every template returns a single "as-written" variant.
    """
    if any(k in yaml_template for k in ("{c256}", "{c512}", "{c1024}")):
        return [("ch=nominal", CH_NOMINAL), ("ch=pre-scaled", CH_SCALED)]
    return [("as-written", {})]


DET_FROM_RE = re.compile(
    r"-\s*\[\s*\[([^\]]*)\]\s*,\s*\d+\s*,\s*(Detect[A-Za-z0-9_]*)\s*,"
)


def parse_det_from(yaml_text):
    matches = DET_FROM_RE.findall(yaml_text)
    if not matches:
        return None, None
    src, head = matches[-1]
    return head, [int(t.strip()) for t in src.split(",") if t.strip()]


def assert_routing(run, yaml_text):
    """For DetectAuxDual: main half and aux half must read DIFFERENT nodes.

    DetectAuxDual sets self.nl = len(ch)//2 and splits its input list in half —
    main sees x[:nl] (context-enriched), aux sees x[nl:] (raw detail). Feeding
    the same nodes twice makes the aux gradient teach the backbone nothing,
    which is the exact failure mode documented in the module's own docstring.
    Runs without `aux_levels` use a plain Detect and skip this check.
    """
    exp = run.get("aux_levels")
    if exp is None:
        return
    head, frm = parse_det_from(yaml_text)
    if frm is None or len(frm) % 2 != 0:
        raise RuntimeError(f"routing check: {head} needs an even input count, got {frm}")
    nl = len(frm) // 2
    main, aux = frm[:nl], frm[nl:]
    n_diff = sum(1 for a, b in zip(main, aux) if a != b)
    if nl != exp or n_diff != exp:
        raise RuntimeError(
            f"routing check FAILED for {head}: nl={nl} (expected {exp}), "
            f"{n_diff}/{nl} levels have distinct main/aux sources.\n"
            f"    main={main}  aux={aux}"
        )
    print(f"  [routing] {head}: nl={nl}  main={main}  aux={aux}  -> {n_diff}/{nl} distinct")


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping the Detect block if its index changed.

    For DetectAuxDual the aux towers (cv2a/cv3a) are train-only and always
    initialise fresh — expected, not a warning.
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
    """Routing check (pre-build) -> construct -> assert head nl -> assert strides."""
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    errors = []

    for label, subs in build_variants(run["yaml"]):
        text = run["yaml"].format(**subs)
        save_yaml(text, yaml_path)
        model = None
        try:
            assert_routing(run, text)
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

    print(f"\n{'#' * 78}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# {run['confidence']}")
    print(f"# beat: {run.get('ref', '-')}")
    print(f"# Batch {batch}  Imgsz {imgsz}  Epochs {EPOCHS}  seed {seed}")
    print(f"{'#' * 78}\n")

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
            data=DATA_YAML, epochs=EPOCHS, imgsz=imgsz, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=run["name"],
            patience=30, close_mosaic=10, seed=seed, deterministic=True,
            amp=True, val=True,
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
                tm = YOLO(best_path)
                tr = tm.val(data=DATA_YAML, split="test", imgsz=imgsz, batch=batch)
                if tr is not None:
                    rd = tr.results_dict
                    metrics["test"] = {
                        "mAP50": round(rd.get("metrics/mAP50(B)", 0), 5),
                        "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0), 5),
                    }
                del tm

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
    print(f"\n{'=' * 80}")
    print(f"  LUGGAGE ARCH -- ROUND 5: DISSECT arch_levelspec ({len(runs)} runs @640)")
    print(f"  mean box 33x72px -> cells: P3 4.1x9.0 | P4 2.1x4.5 | P5 1.0x2.25")
    print(f"  {mode}")
    print(f"  baseline v12s_default2 : 82.77 / 57.63 | md 67.55 | lg 59.85 | P 80.92 R 75.24")
    print(f"  TARGET  arch_levelspec : 83.34 / 58.02 | md 67.26 | lg 60.98 | P 82.61 R 73.76")
    print(f"{'=' * 80}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3}] {r['desc']}")
        print(f"         {r['confidence']}")
    print(f"{'=' * 80}\n")

    all_results = [run_experiment(r, with_test=with_test, build_only=build_only)
                   for r in runs]

    suffix = "round5_build_check" if build_only else "round5_summary"
    summary_path = os.path.join(PROJECT_DIR, f"{PROJECT_DIR}_{suffix}.json")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("  RESULTS   (arch_levelspec reference: 83.34 mAP50 / 58.02 mAP50-95)")
    print(f"{'=' * 80}")
    print(f"{'Name':<20s} {'batch':>6s} {'mAP50':>8s} {'mAP50-95':>10s} {'P':>7s} {'R':>7s} {'Status':>9s} {'Time':>6s}")
    print("-" * 80)
    for r in all_results:
        v = r.get("val", {})
        print(f"{r['name']:<20s} {r.get('batch', 0):>6d} {v.get('mAP50', 0):>8.4f} "
              f"{v.get('mAP50_95', 0):>10.4f} {v.get('precision', 0):>7.4f} "
              f"{v.get('recall', 0):>7.4f} {r['status']:>9s} {r.get('time_h', 0):>5.1f}h")
        if r.get("params_M"):
            print(f"{'':<20s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<20s} {r['error']}")
    print(f"\nSummary saved: {summary_path}")
