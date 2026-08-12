#!/usr/bin/env python3
"""
YOLO26 ARCHITECTURE — port of the v6i custom head modules.

=============================================================================
WHAT WAS PORTED
=============================================================================
Three classes from the YOLOv12 fork (arch_best/nn) into ultralytics26:

    DySample          content-aware upsampler, replaces nn.Upsample P3 -> P2
    ZGGlobalContext2  gated avg+max global-context broadcast (gamma = 0 at init)
    ZGDSConv          zero-gated dynamic snake conv, k=9, shape prior for
                      elongated ground truths (trolleys = 51% of instances)

All three are channel-preserving and identity at initialisation, so they do not
perturb pretrained yolo26s weights at epoch 0. The other ~28 ZG* variants and
the *V6 duplicates were NOT ported — they are dead code in every winning
configuration and porting them would be untested surface.

THE P2 HEAD COMES FREE. On YOLOv12 the 4-level head had to be hand-built and it
was the single largest architecture gain of the campaign. ultralytics ships
yolo26-p2.yaml with Detect at strides 4/8/16/32 already. So the two custom YAMLs
here are built on the SHIPPED p2 graph, and the only thing they test is whether
the three modules add anything ON TOP of a P2 head YOLO26 already has.

=============================================================================
WHY ALL FOUR RUNS, AND WHY NONE IS OPTIONAL
=============================================================================
  1. y26_3lvl_b32   stock yolo26s, 3 levels   -> is the P2 head worth anything?
  2. y26_p2_b32     stock yolo26-p2, 4 levels -> the ARCHITECTURE ANCHOR
  3. y26_lsshift    p2 + DySample + gctx(P2) + snake(P3,P4,P5)
  4. y26_gctxp3     as 3, plus gctx at P3 with the snake stacked on it

2 minus 1 isolates the P2 head. 3 minus 2 isolates the custom modules. 4 minus 3
isolates the extra P3 context — the 0.04 pp pair that topped the v6i campaign
(ls_shift_gctxP3 56.02 vs arch_ls_shift 55.98).

!! BATCH IS 32, NOT 82. A 4-level head plus ZGDSConv will not fit at 82: the
   snake runs a Python loop of k=9 grid_sample calls per axis per level. That
   means these runs are NOT comparable to the loss sweep (anchor 55.24 @ b82).
   Run 1 exists so this arm carries its own same-batch reference. Do not quote a
   delta across the two batches.

v6i / YOLOv12 references, for context only — different detector, different
batch, not a baseline for anything here:
    ls_shift_gctxP3  56.02   small 51.58   large 56.14
    arch_ls_shift    55.98   small 51.22   large 60.09
    yolov12s stock   54.77   small 49.98   large 57.73

=============================================================================
COST WARNING
=============================================================================
ZGDSConv is the expensive module: 2 axes x 9 taps = 18 grid_sample calls per
level per forward, in a Python loop. On v12 at 3 levels it roughly doubled step
time. Here it runs on P3/P4/P5 (deliberately NOT P2 — at stride 4 that is
160x160 @640 and would dominate). Expect ~2-3 h per run rather than 1.6.

REQUIRES the patched ultralytics26: DySample / ZGGlobalContext2 / ZGDSConv in
nn.modules.block, exported in nn/modules/__init__.py, and registered in
parse_model (ZG* in base_modules, DySample in its own branch).

Usage:
    python run_yolo26_arch_v6i.py            # all four, in order
    python run_yolo26_arch_v6i.py y26_p2_b32 y26_lsshift
"""

import copy
import gc
import hashlib
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"      # transfer what matches; custom layers init fresh
PROJECT_DIR = "runs_yolo26_arch_v6i"
YAML_DIR = "arch_yamls_y26_v6i"   # the custom YAMLs are WRITTEN here at runtime
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                        # see the batch warning above
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False
SCALE = "s"

CUSTOM_MODULES = ("DySample", "ZGGlobalContext2", "ZGDSConv")

# =============================================================================
# THE CUSTOM MODEL YAMLS ARE EMBEDDED AND WRITTEN AT RUNTIME
# =============================================================================
# Same pattern as run_arch_refine_v6i.py: the script owns its topologies and
# save_yaml() writes them into YAML_DIR before each run. Nothing has to be
# copied into the installed ultralytics package, so the script is portable
# across machines and the exact topology used is recorded next to the results.
# yolo26s.yaml and yolo26-p2.yaml are NOT embedded — they ship with ultralytics
# and are referenced by name.
# =============================================================================

LSSHIFT_YAML = """# YOLO26 + the winning v6i custom head (arch_ls_shift analogue)
#
# Built on the SHIPPED yolo26-p2 backbone/head, not on a transplanted YOLOv12
# graph. Only three things are added, mirroring arch_ls_shift:
#   1. the P3 -> P2 nn.Upsample is replaced by DySample (content-aware)
#   2. ZGGlobalContext2 on the P2 detection feature
#   3. ZGDSConv k=9 on the P3 / P4 / P5 detection features
#
# WHY THE P2 HEAD COMES FREE: ultralytics ships yolo26-p2.yaml with a 4-level
# Detect at strides 4/8/16/32. On YOLOv12 the P2 head had to be hand-built and
# it was the single largest architecture gain (+1.24 small). Here it is native,
# so this file only tests whether the three CUSTOM modules add anything on top.
#
# ZGDSConv is deliberately NOT applied at P2. It runs a Python loop of k=9
# grid_sample calls per axis; at stride 4 (160x160 @640) that is the most
# expensive place in the graph, and arch_ls_shift did not use it there either.
#
# All three modules are identity at initialisation (gamma=0 / near-zero-init
# offsets), so the pretrained yolo26s weights are not perturbed at epoch 0.

nc: 3
end2end: True # YOLO26 one2one head — NMS-free
reg_max: 1 # DFL-free; the third loss term is L1 on ltrb
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]] # 6
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]] # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 1, C2PSA, [1024, 1]] # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12  cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15  cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8)

  - [-1, 1, DySample, [2]] # 17  <-- was nn.Upsample; content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # 18  cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4)

  - [-1, 1, Conv, [128, 3, 2]] # 20
  - [[-1, 16], 1, Concat, [1]] # 21
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8)

  - [-1, 1, Conv, [256, 3, 2]] # 23
  - [[-1, 13], 1, Concat, [1]] # 24
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16)

  - [-1, 1, Conv, [512, 3, 2]] # 26
  - [[-1, 10], 1, Concat, [1]] # 27
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32)

  # --- custom refinement, channel-preserving and identity at init ---
  - [19, 1, ZGGlobalContext2, [128]] # 29  P2 + gated avg/max global context
  - [22, 1, ZGDSConv, [256, 9]] # 30  P3 + snake k=9
  - [25, 1, ZGDSConv, [512, 9]] # 31  P4 + snake k=9
  - [28, 1, ZGDSConv, [1024, 9]] # 32  P5 + snake k=9

  - [[29, 30, 31, 32], 1, Detect, [nc]] # 33  Detect(P2, P3, P4, P5)
"""

GCTXP3_YAML = """# YOLO26 + the best v6i custom head (ls_shift_gctxP3 analogue)
#
# Identical to yolo26-lsshift.yaml except that P3 also gets a
# ZGGlobalContext2 with the snake STACKED on top of it. On v6i this was the
# best architecture of the whole campaign: ls_shift_gctxP3 56.02 vs
# arch_ls_shift 55.98 — a 0.04 pp gap on one seed, i.e. a tie, but it also
# held the best small (51.58). Run BOTH or neither; the pair is the test.
#
# Built on the SHIPPED yolo26-p2 backbone/head, not on a transplanted YOLOv12
# graph. Only three things are added, mirroring arch_ls_shift:
#   1. the P3 -> P2 nn.Upsample is replaced by DySample (content-aware)
#   2. ZGGlobalContext2 on the P2 detection feature
#   3. ZGDSConv k=9 on the P3 / P4 / P5 detection features
#
# WHY THE P2 HEAD COMES FREE: ultralytics ships yolo26-p2.yaml with a 4-level
# Detect at strides 4/8/16/32. On YOLOv12 the P2 head had to be hand-built and
# it was the single largest architecture gain (+1.24 small). Here it is native,
# so this file only tests whether the three CUSTOM modules add anything on top.
#
# ZGDSConv is deliberately NOT applied at P2. It runs a Python loop of k=9
# grid_sample calls per axis; at stride 4 (160x160 @640) that is the most
# expensive place in the graph, and arch_ls_shift did not use it there either.
#
# All three modules are identity at initialisation (gamma=0 / near-zero-init
# offsets), so the pretrained yolo26s weights are not perturbed at epoch 0.

nc: 3
end2end: True # YOLO26 one2one head — NMS-free
reg_max: 1 # DFL-free; the third loss term is L1 on ltrb
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]] # 6
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]] # 8
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 1, C2PSA, [1024, 1]] # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12  cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15  cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8)

  - [-1, 1, DySample, [2]] # 17  <-- was nn.Upsample; content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # 18  cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4)

  - [-1, 1, Conv, [128, 3, 2]] # 20
  - [[-1, 16], 1, Concat, [1]] # 21
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8)

  - [-1, 1, Conv, [256, 3, 2]] # 23
  - [[-1, 13], 1, Concat, [1]] # 24
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16)

  - [-1, 1, Conv, [512, 3, 2]] # 26
  - [[-1, 10], 1, Concat, [1]] # 27
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32)

  # --- custom refinement, channel-preserving and identity at init ---
  - [19, 1, ZGGlobalContext2, [128]] # 29  P2 + gated global context
  - [22, 1, ZGGlobalContext2, [256]] # 30  P3 + gated global context
  - [30, 1, ZGDSConv, [256, 9]] # 31  P3 + snake k=9, STACKED on the gctx above
  - [25, 1, ZGDSConv, [512, 9]] # 32  P4 + snake k=9
  - [28, 1, ZGDSConv, [1024, 9]] # 33  P5 + snake k=9

  - [[29, 31, 32, 33], 1, Detect, [nc]] # 34  Detect(P2, P3, P4, P5)
"""

RUNS = [
    {"name": "y26_3lvl_b32", "cfg": "yolo26s.yaml", "custom": False,
     "label": "stock yolo26s, 3 levels (P3/P4/P5) — same-batch reference",
     "why": "The loss-sweep anchor (55.24) was trained at batch 82; nothing in "
            "this file can be compared to it. This run re-establishes stock "
            "YOLO26 at batch 32 so runs 2-4 have a legitimate reference, and it "
            "also gives the P2-head delta when subtracted from run 2."},

    {"name": "y26_p2_b32", "cfg": "yolo26-p2.yaml", "custom": False,
     "label": "stock yolo26-p2, 4 levels (P2/P3/P4/P5) — THE ARCHITECTURE ANCHOR",
     "why": "The shipped P2 head with no custom modules. On YOLOv12 the P2 head "
            "alone was worth +1.24 on small. If it delivers that here, most of "
            "the v6i architecture gain is available for free and the custom "
            "modules have little left to add — which is exactly what runs 3 and "
            "4 are there to measure."},

    {"name": "y26_lsshift", "cfg": "yolo26-lsshift.yaml", "yaml": LSSHIFT_YAML, "custom": True,
     "label": "p2 + DySample + ZGGlobalContext2(P2) + ZGDSConv k9 (P3,P4,P5)",
     "why": "The arch_ls_shift analogue: the same three modules in the same "
            "positions, on YOLO26's own p2 graph rather than a transplanted "
            "YOLOv12 one. Minus run 2, this is the clean custom-module delta."},

    {"name": "y26_gctxp3", "cfg": "yolo26-gctxp3.yaml", "yaml": GCTXP3_YAML, "custom": True,
     "label": "as y26_lsshift, plus ZGGlobalContext2 at P3 with the snake stacked on it",
     "why": "The ls_shift_gctxP3 analogue — best overall AND best small of the "
            "whole v6i campaign, though only 0.04 pp above arch_ls_shift on one "
            "seed. Run it with y26_lsshift or not at all; the pair is the test, "
            "and a gap under ~0.3 pp between them means nothing."},
]


def save_yaml(content, path):
    """Write a model YAML next to the results, the way the v12 arch runners do."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def resolve_cfg(rc):
    """Path (custom, written here) or bare name (stock, shipped with ultralytics)."""
    if rc.get("yaml"):
        return save_yaml(rc["yaml"], os.path.join(YAML_DIR, rc["cfg"]))
    return rc["cfg"]


# ================================================================== preflight
def env_provenance():
    info = {"tasks_md5": None, "block_md5": None, "modules": {}, "cfg_found": {},
            "parse_model_registered": {}}
    try:
        import ultralytics.nn.modules.block as _b
        import ultralytics.nn.tasks as _t
        for mod, key in ((_b, "block_md5"), (_t, "tasks_md5")):
            p = getattr(mod, "__file__", None)
            if p and os.path.exists(p):
                info[key] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
        import ultralytics.nn.modules as M
        for n in CUSTOM_MODULES:
            info["modules"][n] = hasattr(M, n) or hasattr(_b, n)
        src = open(_t.__file__, encoding="utf-8").read()
        info["parse_model_registered"] = {
            "ZGGlobalContext2": "ZGGlobalContext2," in src,
            "ZGDSConv": "ZGDSConv," in src,
            "DySample": "elif m is DySample:" in src,
        }
    except Exception as e:
        info["import_error"] = str(e)
    try:
        import ultralytics.cfg as _c
        base = os.path.join(os.path.dirname(_c.__file__), "models", "26")
        for r in RUNS:
            # custom YAMLs are written by this script, so they always resolve.
            # For stock ones, ultralytics STRIPS THE SCALE LETTER at load time:
            # "yolo26s.yaml" is served by the file "yolo26.yaml". Checking for the
            # literal filename reports a false missing, so try the de-scaled name.
            if r.get("yaml"):
                info["cfg_found"][r["cfg"]] = "written by this script"
                continue
            stem, ext = os.path.splitext(r["cfg"])
            cands = [r["cfg"]]
            if stem and stem[-1] in "nsmlx":
                cands.append(stem[:-1] + ext)
            info["cfg_found"][r["cfg"]] = any(
                os.path.isfile(os.path.join(base, c)) for c in cands)
    except Exception as e:
        info["cfg_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  md5: tasks={ENV.get('tasks_md5')}  block={ENV.get('block_md5')}")
    print(f"  custom classes importable : {ENV.get('modules')}")
    print(f"  registered in parse_model : {ENV.get('parse_model_registered')}")
    print(f"  model yamls found         : {ENV.get('cfg_found')}")

    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import ultralytics: {ENV['import_error']}")
        return False
    need_custom = any(r["custom"] for r in todo)
    missing = [k for k, v in ENV["modules"].items() if not v]
    if need_custom and missing:
        print(f"\n  [ABORT] custom classes not importable: {missing}. The ultralytics "
              f"on the import path is not the patched tree.")
        return False
    unreg = [k for k, v in ENV["parse_model_registered"].items() if not v]
    if need_custom and unreg:
        print(f"\n  [ABORT] not registered in parse_model: {unreg}. YAML parsing would "
              f"raise KeyError, or worse, silently mis-shape the args.")
        return False
    stock_cfgs = {r["cfg"] for r in todo if not r.get("yaml")}
    missing_cfg = [k for k, v in ENV["cfg_found"].items() if v is False and k in stock_cfgs]
    if missing_cfg:
        print(f"\n  [ABORT] stock model yaml(s) not found in cfg/models/26: {missing_cfg}")
        return False

    print(f"\n  [!] BATCH={BATCH}. These runs are NOT comparable to the loss sweep "
          f"(anchor 55.24 @ b82).\n      Compare only within this file, against "
          f"y26_p2_b32.")
    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


# ======================================================================= train
def count_custom(model):
    """How many custom layers the built graph actually contains."""
    seen = {}
    for m in model.modules():
        n = type(m).__name__
        if n in CUSTOM_MODULES:
            seen[n] = seen.get(n, 0) + 1
    return seen


def run_one(rc):
    name = rc["name"]
    cfg = resolve_cfg(rc)          # writes the custom YAML into YAML_DIR
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}"
          + ("   (written by this script)" if rc.get("yaml") else "   (shipped with ultralytics)"))
    print(f"  scale={SCALE}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    built = count_custom(model.model)
    print(f"  custom layers in the built graph: {built or 'none'}")
    if rc["custom"] and not built:
        raise RuntimeError(
            f"{name}: the yaml declares custom modules but the built graph contains "
            f"none. parse_model silently dropped them — the run would be a plain p2 "
            f"model wearing a custom name.")
    if not rc["custom"] and built:
        raise RuntimeError(f"{name}: expected a stock graph but found {built}.")

    if MODEL_WEIGHTS:
        try:
            model.load(MODEL_WEIGHTS)
            print(f"  transferred what matches from {MODEL_WEIGHTS}; custom layers "
                  f"initialise fresh (identity at init by construction)")
        except Exception as e:
            print(f"  [warn] weight transfer failed: {e} — training from scratch")

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    results = model.train(**kw)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "cfg": rc["cfg"], "cfg_path": cfg, "hours": hours,
           "weights": weights, "seed": SEED, "batch": BATCH, "custom_layers": built,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    by = {r["name"]: r["test_map5095"] for r in res if r["test_map5095"] == r["test_map5095"]}
    print(f"\n{'=' * 86}\n  YOLO26 ARCHITECTURE — v6i @{IMG_SIZE}, b{BATCH}, seed {SEED}\n{'=' * 86}")
    print(f"{'run':<18}{'cfg':<24}{'mAP50':>8}{'mAP50-95':>10}{'vs p2':>8}{'h':>6}  custom")
    print('-' * 86)
    anchor = by.get("y26_p2_b32")
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        vs = f"{(r['test_map5095'] - anchor) * 100:+8.2f}" if anchor else f"{'n/a':>8}"
        cl = ",".join(f"{k}x{v}" for k, v in (r.get("custom_layers") or {}).items()) or "-"
        print(f"{r['name']:<18}{r['cfg']:<24}{r['test_map50'] * 100:>8.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{vs}{r['hours']:>6.1f}  {cl}")
    print('-' * 86)
    print("  THE THREE DELTAS THIS FILE EXISTS TO MEASURE:")
    def d(a, b, lbl):
        if a in by and b in by:
            print(f"    {lbl:<34}{(by[a] - by[b]) * 100:+6.2f} pp")
        else:
            print(f"    {lbl:<34}   (needs {a} and {b})")
    d("y26_p2_b32", "y26_3lvl_b32", "P2 head, free and native")
    d("y26_lsshift", "y26_p2_b32", "custom modules on top of P2")
    d("y26_gctxp3", "y26_lsshift", "extra P3 context")
    print("\n  Read SMALL per size: on v12 the P2 head was worth +1.24 there and the")
    print("  custom modules added little beyond it. If that repeats, the honest")
    print("  conclusion is that YOLO26 needs the P2 head and not the modules.")
    print("  YOLO26 seed noise is unmeasured; phase-1 implies a floor near 0.45 pp.")
    print("  Per-size: CocoEvalAllFolders_luggage.py on best.pt")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<18} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 86}\n  YOLO26 ARCHITECTURE — {len(todo)} runs (~{2.5 * len(todo):.1f} GPU-h, "
          f"snake convs are slow)")
    print(f"  {', '.join(r['name'] for r in todo)}\n{'=' * 86}\n")
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "cfg": r["cfg"], "hours": float("nan"),
                        "error": str(e), "test_map50": float("nan"),
                        "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
