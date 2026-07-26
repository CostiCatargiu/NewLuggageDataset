#!/usr/bin/env python3
"""
Luggage Architecture Ablation — DECOUPLED round 2 (5 runs: 4 @640 + 1 @896).

=============================================================================
WHY THIS FILE CHANGED — arch_p2head_decoupled2 WAS NEVER DECOUPLED
=============================================================================
DetectDecoupled.__init__ does:

    n = len(ch) // 2
    box_ch, cls_ch = ch[:n], ch[n:]
    super().__init__(nc, box_ch)          # -> self.nl = n

so self.nl is HALF the input count. That is intentional and documented: the head
is dual-input by design, taking 2*nl feature maps [box..., cls...].

The previous overnight script templated the Detect input list as {det_in} and
tried 4-input FIRST:

    4-input  -> len(ch)=4 -> nl=2 -> fails the `levels: 4` assertion -> REJECTED
    8-input  -> len(ch)=8 -> nl=4 -> PASSES

and the 8-input fallback was `[23, 14, 17, 20, 23, 14, 17, 20]` — every level
passed TWICE. So box and cls read IDENTICAL features. arch_p2head_decoupled2 was
not a task-decoupled head at all; it was a stock P2 head whose cv3 got rebuilt as
a deeper DWConv->Conv tower. It won small-object metrics at 640 as a FAT CLS
TOWER, not as a decoupled head.

Consequences for this round:
  * {det_in} templating is REMOVED. Every Detect input list is now written
    explicitly, so routing can never be decided by a silent build fallback.
  * A new routing assertion (assert_routing) parses the generated YAML, splits
    the Detect from-list in half, and counts how many levels have DIFFERENT box
    and cls sources. A run that expects real decoupling and gets duplicated
    sources is REJECTED before training, not discovered 3 hours later.
  * arch_decoupled_dysample is DROPPED. It relied on {det_in}, so it would have
    taken the duplicated fallback -> "P2/DySample neck + fat cls tower", which is
    a near-duplicate of arch_dysample (57.58) and arch_dysample_p2_gctx5 (57.59).
    Near-zero new information.

=============================================================================
WHAT THE 640 RESULTS ACTUALLY SAY
=============================================================================
640 plateau, mAP50-95, 14 distinct architectures:  56.94 -> 57.64  (0.70pt wide)
  coordatt3 57.64 | gctx5 57.60 | dysample_p2_gctx5 57.59 | dysample 57.58
  globalctx_p2 57.57 | decoupled2 57.47 | detail_aux4 57.36 | strip4 57.33
  p2fuse2 57.19 | p2head6 57.18 | globalctx 57.15 | full_noshape2 57.15
  deepp3aux2 57.07 | shapecbam 56.94                      (v6_default2 = 56.84)

NO ADDITIVITY: stacking the three proven winners (dysample_p2_gctx5 = 57.59) did
not beat the best single lever (gctx5 = 57.60). The P2/context/attention family
is a flat plateau worth ~+0.75, and which mechanism you pick barely matters.

decoupled2's real position: best at 640 on SMALL (small mAP50-95 52.92,
mAP50_small 80.06) but rank 6/15 on overall mAP50-95 (57.47), and the WORST large
mAP50-95 in the whole set (56.90, -6.56 vs baseline 63.46).

THE UNATTACKED AXIS — the large-object tax. Large mAP50-95, baseline 63.46:
  decoupled2 56.90 | p2head6 57.47 | coordatt3 57.71 | p2fuse2 58.09
  gctx5 59.37 | strip4 59.42 | detail_aux4 59.66 | dysample 60.09
  globalctx_p2 60.32 | shapecbam 60.99
Large is 20.4% of test instances and it is the ONLY bucket where all 15 runs are
worse than baseline. Every run in the original proposal routed enhancements to
the CLS branch; none of them touched this. Hence run [3] below.

The only change that ever broke the 640 plateau was RESOLUTION: hires10 @896 =
58.82 (+1.22 over its own 640 twin gctx5), nearly 2x the width of the entire 640
band. Hence run [5] below.

=============================================================================
RUNS
=============================================================================
  [1] arch_decoupled_gctx        box=clean, cls=GlobalContext (4/4 levels decoupled)
                                 First GENUINE decoupling test. gctx is the most
                                 validated lever (gctx5, globalctx_p2, hires10).
  [2] arch_decoupled_coordatt    box=clean, cls=CoordinateAttention (4/4)
                                 Also a clean ablation vs coordatt3 (57.64, which
                                 put attention on BOTH paths) -> does attention
                                 belong on cls only, or everywhere?
  [3] arch_decoupled_gctx_asym   cls=gctx on P2/P3 only; P4/P5 cls read the SAME
                                 clean features as box (2/4 decoupled). NEW IDEA:
                                 tests whether the large-object tax is capacity
                                 dilution from rebuilding uniform cv3 towers over
                                 4 levels. Leaves the levels that already worked
                                 untouched. One-line routing change vs [1].
  [4] arch_decoupled_cosine      REWIRED. DetectDecoupledCosine with the explicit
                                 gctx cls path (was silently taking the duplicated
                                 fallback). Cosine cls scoring is the only run here
                                 testing SCORE CALIBRATION rather than features.
  [5] arch_decoupled_gctx_hires  Run [1] at imgsz 896, batch 16. Resolution is the
                                 only lever with demonstrated headroom.

  (commented out, ready to queue) arch_decoupled_gctx2 — ZGGlobalContext2 (avg+max
  pool) instead of ZGGlobalContext (avg only). Already in block.py, written for
  round41 on the weapon dataset, NEVER run on luggage.

Expectation setting: given a 0.70pt-wide 640 plateau with zero additivity, runs
[1][2][4] should be expected to land 57.3-57.8. Their value is ablation clarity
(does real decoupling beat the fat cls tower?). Runs [3] and [5] are the ones with
a mechanism-level reason to move the number.

=============================================================================
BUILD ROBUSTNESS
=============================================================================
  * CoordinateAttention IS in parse_model's handling set (tasks.py: `elif m is
    CoordinateAttention: args = [ch[f], c2]`), so nominal channel args build fine.
    The {c256}/{c512}/{c1024} templating is kept as a harmless scale-aware
    fallback in case nn_modules/tasks.py is not the patched copy.
  * ZGGlobalContext is verified-scaled -> its args stay hard-coded.
  * DetectDecoupledCosine signature is (nc, scale=16.0, ch=()). parse_model calls
    it as (nc, ch_list), so ch_list lands in `scale`; the module has an
    isinstance(scale, (list, tuple)) guard that swaps them. YAML passes [nc] only.
  * Post-build, head nl + stride pyramid + CLS/BOX ROUTING are all asserted.
    --build-only constructs every run with the assertions and does NOT train.

Nothing about the loss/assigner path is set here — use_satal / tal_topk / alpha /
beta stay inherited from the defaults, so these remain comparable to the earlier
ARCH runs (note: the arch family inherits use_satal=true / topk=12 / a=0.6 / b=5.0,
which is NOT the v6_default2 setting — keep comparisons within the arch family).

REQUIRES
  nn_modules/ copied to ultralytics/nn/modules/ (block.py, conv.py, head.py, tasks.py, __init__.py)

Usage
  python run_luggage_arch_decoupled640.py                     # all 5, in order
  python run_luggage_arch_decoupled640.py --build-only        # construct only, no training
  python run_luggage_arch_decoupled640.py arch_decoupled_gctx --with-test
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
# NOTE: the {det_in} axis is GONE. Detect input lists are explicit in every tail
# so that box/cls routing can never be chosen by a silent build fallback.
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

# --- TAILS ---
# DetectDecoupled / DetectDecoupledCosine take 2*nl inputs, ordered
# [box_P2, box_P3, box_P4, box_P5, cls_P2, cls_P3, cls_P4, cls_P5].
# The FIRST half feeds cv2 (box regression), the SECOND half feeds cv3 (cls).
# Box always reads the CLEAN neck features (23, 14, 17, 20).

# [1] box=CLEAN + cls=GlobalContext on all 4 levels  (Detect @ 28)
TAIL_DEC_GCTX = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 cls-path
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 cls-path
  - [17, 1, ZGGlobalContext, [512]]                 # 26  P4 cls-path
  - [20, 1, ZGGlobalContext, [1024]]                # 27  P5 cls-path
  - [[23, 14, 17, 20, 24, 25, 26, 27], 1, DetectDecoupled, [nc]]  # 28  box=clean, cls=gctx
"""

# [2] box=CLEAN + cls=CoordinateAttention on all 4 levels  (Detect @ 28)
TAIL_DEC_COORDATT = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, CoordinateAttention, [{c256}]]          # 24  P2 cls-path
  - [14, 1, CoordinateAttention, [{c256}]]          # 25  P3 cls-path
  - [17, 1, CoordinateAttention, [{c512}]]          # 26  P4 cls-path
  - [20, 1, CoordinateAttention, [{c1024}]]         # 27  P5 cls-path
  - [[23, 14, 17, 20, 24, 25, 26, 27], 1, DetectDecoupled, [nc]]  # 28  box=clean, cls=attn
"""

# [3] ASYMMETRIC: cls=gctx on P2/P3 only; P4/P5 cls reuse the CLEAN box features.
#     Hypothesis: the large-object tax (-3 to -6.6 large mAP50-95 in every P2 run)
#     comes from rebuilding uniform cv3 towers across 4 levels, diluting the levels
#     that were already fine. So enhance ONLY the levels that needed it.
#     Note 17/20 appear in BOTH halves on purpose -> 2/4 levels truly decoupled.
#     (Detect @ 26)
TAIL_DEC_GCTX_ASYM = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 cls-path (enhanced)
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 cls-path (enhanced)
  - [[23, 14, 17, 20, 24, 25, 17, 20], 1, DetectDecoupled, [nc]]  # 26  P4/P5 cls = clean
"""

# [4] box=CLEAN + cls=GlobalContext + DetectDecoupledCosine (cosine cls scoring).
#     Previously used {det_in} and silently took the duplicated fallback, so the
#     "decoupled" part was never actually tested alongside cosine. Now explicit.
#     YAML passes [nc]; `scale` defaults to 16.0 inside the module.  (Detect @ 28)
TAIL_DEC_COSINE = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext, [256]]                 # 24  P2 cls-path
  - [14, 1, ZGGlobalContext, [256]]                 # 25  P3 cls-path
  - [17, 1, ZGGlobalContext, [512]]                 # 26  P4 cls-path
  - [20, 1, ZGGlobalContext, [1024]]                # 27  P5 cls-path
  - [[23, 14, 17, 20, 24, 25, 26, 27], 1, DetectDecoupledCosine, [nc]]  # 28  + cosine cls
"""

# [opt] ZGGlobalContext2 (avg+max pool) instead of avg-only. Never run on luggage.
TAIL_DEC_GCTX2 = """  - [14, 1, nn.Upsample, [None, 2, "nearest"]]     # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 cls-path (avg+max)
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 cls-path (avg+max)
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 cls-path (avg+max)
  - [20, 1, ZGGlobalContext2, [1024]]               # 27  P5 cls-path (avg+max)
  - [[23, 14, 17, 20, 24, 25, 26, 27], 1, DetectDecoupled, [nc]]  # 28
"""

# =============================================================================
# ASSEMBLE
# =============================================================================
# "levels"            = expected number of detection levels (asserted post-build)
# "strides"           = expected stride pyramid (asserted post-build)
# "decoupled_levels"  = expected count of levels whose cls source != box source
#                       (asserted pre-build from the YAML — catches the
#                        duplicated-input bug that produced decoupled2)
RUNS = [
    {
        "name": "arch_decoupled_gctx",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_GCTX,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 4,
        "desc": "[1/5] Decoupled: box=clean, cls=GlobalContext — first REAL decoupling test",
    },
    {
        "name": "arch_decoupled_coordatt",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_COORDATT,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 4,
        "desc": "[2/5] Decoupled: box=clean, cls=CoordinateAttention — vs coordatt3 (both paths)",
    },
    {
        "name": "arch_decoupled_gctx_asym",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_GCTX_ASYM,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 2,          # P2/P3 enhanced; P4/P5 cls == box (clean)
        "desc": "[3/5] Asymmetric: cls=gctx on P2/P3 only — attack the large-object tax",
    },
    {
        "name": "arch_decoupled_cosine",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_COSINE,
        "batch": 32,
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 4,
        "desc": "[4/5] DetectDecoupledCosine + gctx cls path — cosine score calibration",
    },
    {
        "name": "arch_decoupled_gctx_hires",
        "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_GCTX,
        "batch": 16,          # dropped from 32 to fit 896 in VRAM
        "imgsz": 896,         # the only lever with demonstrated headroom (+1.22)
        "levels": 4,
        "strides": [4, 8, 16, 32],
        "decoupled_levels": 4,
        "desc": "[5/5] Run [1] at imgsz 896 — decoupling x resolution",
    },
    # ---- ready to queue, not in the current batch -----------------------------
    # {
    #     "name": "arch_decoupled_gctx2",
    #     "yaml": BACKBONE + HEAD_STOCK + TAIL_DEC_GCTX2,
    #     "batch": 32,
    #     "levels": 4,
    #     "strides": [4, 8, 16, 32],
    #     "decoupled_levels": 4,
    #     "desc": "[opt] ZGGlobalContext2 (avg+max) cls path — never run on luggage",
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

    Only the CHANNEL axis remains ({c256}/{c512}/{c1024}), ordered so the
    correct-if-tasks.py-is-patched convention is tried first; the pre-scaled
    workaround is only reached if that raises. The {det_in} axis was removed on
    purpose — see the module docstring.
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
    """Verify box and cls actually read DIFFERENT features on the expected levels.

    DetectDecoupled sets self.nl = len(ch)//2 and splits its input list in half:
    the first nl feed cv2 (box), the last nl feed cv3 (cls). If the two halves are
    identical the head is NOT decoupled -- it is just a rebuilt cls tower. That is
    exactly what silently happened to arch_p2head_decoupled2, so it is now a hard
    pre-build check rather than something to discover after a 3-hour train.
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
            f"routing check FAILED for {head}: {n_dec}/{nl} levels truly decoupled, "
            f"expected {exp}.  box={box}  cls={cls}\n"
            f"    Identical box/cls sources mean the head is NOT decoupled -- it is a "
            f"fat-cls-tower run (the arch_p2head_decoupled2 bug)."
        )
    print(f"  [routing] {head}: nl={nl}  box={box}  cls={cls}  "
          f"decoupled={n_dec}/{nl} (expected {exp})")


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """Load pretrained weights, remapping the Detect block if its index changed.

    For the 4-level P2-head / decoupled heads only shape-matching keys transfer;
    the new towers initialize fresh (expected, near-identity at init).
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

    Checks, in order: box/cls routing (pre-build, from the YAML text), then
    construction, then head nl, then the stride pyramid. Returns
    (model, yaml_path, variant_label). Raises RuntimeError if all variants fail.
    """
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    variants = build_variants(run["yaml"])
    errors = []

    for label, subs in variants:
        text = run["yaml"].format(**subs)
        save_yaml(text, yaml_path)
        model = None
        try:
            # Routing is a property of the YAML, not of the built graph -- check it
            # first so a mis-wired head can never reach model.train().
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
                    "decoupled_levels": run.get("decoupled_levels"),
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
                   "imgsz": imgsz, "batch": batch,
                   "decoupled_levels": run.get("decoupled_levels"),
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
    n640 = sum(1 for r in runs if r.get("imgsz", IMG_SIZE) == 640)
    print(f"\n{'=' * 72}")
    print(f"  LUGGAGE ARCH -- DECOUPLED round 2 "
          f"({len(runs)} runs: {n640} @640 + {len(runs) - n640} @higher-res)")
    print(f"  Dataset: luggage (3 classes, 40% small, AR=2.69)")
    print(f"  {mode}  (per-config batch/imgsz below)")
    print(f"  width={SCALE_WIDTH} -> fallback channels 256->{scaled_ch(256)}, "
          f"512->{scaled_ch(512)}, 1024->{scaled_ch(1024)}")
    print(f"  640 reference: gctx5 57.60 | coordatt3 57.64 | decoupled2 57.47 "
          f"(small 52.92, large 56.90)")
    print(f"  896 reference: hires10 58.82 (small 53.52, large 58.62)")
    print(f"{'=' * 72}")
    for r in runs:
        print(f"  [b{r.get('batch', BATCH):>3} @{r.get('imgsz', IMG_SIZE)} "
              f"dec={r.get('decoupled_levels', '-')}/4] {r['desc']}")
    print(f"{'=' * 72}\n")

    all_results = []
    for run in runs:
        result = run_experiment(run, with_test=with_test, build_only=build_only)
        all_results.append(result)

    suffix = "decoupled640_build_check" if build_only else "decoupled640_summary"
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
            print(f"{'':<28s} built via [{r['variant']}]  "
                  f"decoupled={r.get('decoupled_levels')}/4")
        if r.get("params_M"):
            print(f"{'':<28s} {r['params_M']}M params")
        if r.get("error"):
            print(f"{'':<28s} {r['error']}")

    print(f"\nSummary saved: {summary_path}")
