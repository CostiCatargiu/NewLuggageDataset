#!/usr/bin/env python3
"""
YOLO26 OVERNIGHT — 4 runs, ~6.6 GPU-h. Built from the YOLO26 evidence only.

=============================================================================
THE DIAGNOSTIC THESE RUNS COME FROM
=============================================================================
Across 23 custom YOLO26 runs, ZERO beat the baseline on large objects. But the
per-size breakdown says the three failures are not the same failure:

  LARGE            mAP50   mAP50-95    AR50   AR50-95   P50-95    R50
  yolo26_custom-9  81.75      60.87   92.97     76.30    67.72   75.00
  y26_p2_dysample  74.66      53.72   91.56     71.73    61.18   70.00
  delta            -7.08      -7.14   -1.41     -4.56    -6.54   -5.00

AR50_large barely moves (-1.41) while mAP50_large falls 7.08. The large boxes
are STILL BEING PRODUCED — average recall proves it. They are being SCORED too
low to survive the ranking. R50_large drops 5 pp at the operating point while
max achievable recall is intact. That is a confidence-calibration failure, not
a detection failure and not a localisation failure. And this is an NMS-free
head, so there is no re-ranking stage to repair a bad score.

Compare the other two, which fail differently:
  y26_p2_dys3   mAP50_lg -0.02, mAP50-95_lg -5.81  -> localisation only
  y26_lsshift   mAP50_lg -1.30, mAP50-95_lg -2.75  -> mild, P50-95 +0.77

MECHANISM (hypothesis, not a result). align_metric = score^a * IoU^b with
topk2=1 in the one2one branch gives each large GT exactly ONE anchor. Adding a
stride-4 level makes P2 anchors eligible for large GTs. A P2 anchor's
classifier has seen almost only small objects, so it wins the assignment on
IoU and then scores the object poorly. That predicts exactly the table above:
box present, score wrong.

If the hypothesis is right the fix is in the ASSIGNER, not the graph — and
LB-TAL is already ported. `_per_level_budget` falls back to min_level_k for any
stride absent from the budget dict, so a 4-level model's P2 is directly
controllable. Runs 1 and 2 are a TWO-SIDED test of that.

=============================================================================
THE FOUR
=============================================================================
 1 y26_dys_p2starve   best arch + LB {4:1, 8:3, 16:4, 32:4}   STARVE P2
 2 y26_dys_p2rich     best arch + LB {4:4, 8:3, 16:2, 32:1}   ENRICH P2
 3 y26_p2_wide        best arch, P2 branch 128 -> 256 ch      new topology
 4 y26_wide_starve     wide P2 + LB {4:1, 8:3, 16:4, 32:4}     both levers (2x2 cell)

Runs 1 and 2 are deliberately opposed. Any single budget that helps is just
another draw from a 0.46 sd distribution; a MONOTONE contrast across two
opposed budgets is evidence. Read them together or not at all.

WHAT WOULD CONFIRM THE HYPOTHESIS
  p2starve: mAP50_large recovers toward 81 while AR50_large stays near 91.6
            (score fixed, detection unchanged), small holds above 51.5
  p2rich:   large falls FURTHER, below 53.72
Anything else — both up, both down, or starve helping small too — falsifies it,
and the assignment story should be dropped rather than patched.

WHY RUN 4 COMPLETES A 2x2 INSTEAD OF ADDING A NINTH ONE-OFF
Runs 3 and 4 differ only in the loss, runs 1 and 4 only in the width, and
y26_p2_dysample (55.94, already run) is the fourth corner:

                       stock loss          starve P2
    P2 = 128 ch     y26_p2_dysample     y26_dys_p2starve
    P2 = 256 ch     y26_p2_wide         y26_wide_starve

Every effect quoted in this project so far came from subtracting two single
runs, and those estimates contradicted each other: gctx measured +0.72 in one
context and -1.16 in another, a 1.88 pp swing for the same module. A factorial
MEASURES the interaction rather than assuming it is zero, and the summary
prints both main effects and the interaction term.

NO SEED REPEAT IS INCLUDED, at the user's request. Say plainly what that costs:
the 8 b32 architecture runs have mean 55.11 and sd 0.46, and y26_p2_dysample
sits at +1.82 sd where the expected maximum of 8 draws is about +1.42. The
noise floor has never been measured in 24 YOLO26 runs. So nothing here is
separable from noise on its own. What partly rescues it is the DESIGN: a 2x2
gives each main effect from four runs instead of two, and the starve/enrich
pair is a two-sided contrast. A consistent SIGN across opposed conditions is
evidence even when a single delta is not.

Batch 32, imgsz 640 — matches arch rounds 1-3 exactly, so every number here is
directly comparable to y26_p2_dysample 55.94 and y26_p2_b32 55.03.
REQUIRES the module port (DySample) on the import path.

Usage:
    python run_yolo26_overnight4_v6i.py
    python run_yolo26_overnight4_v6i.py y26_dys_p2starve y26_dys_p2rich
"""

import gc
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_overnight4_v6i"
YAML_DIR = "arch_yamls_y26_ov4"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CUSTOM_MODULES = ("DySample", "ZGGlobalContext2", "ZGDSConv")

BEST = 0.5594          # y26_p2_dysample   <- the number to beat / to replicate
ANCHOR = 0.5503        # y26_p2_b32
BASE_B82 = 0.5524      # yolo26_custom-9 stock baseline (b82, not b32-comparable)
FAMILY_SD = 0.46       # sd of the 8 b32 architecture runs, in pp

# stock loss values, so an LB run differs from the arch runs in ONE place
_STOCK = dict(use_lbtal=False)


def _lb(level_topk):
    """LB-TAL fixed per-level budget. Keys are STRIDES in px: 4=P2 ... 32=P5."""
    return dict(use_lbtal=True, lbtal_mode="fixed", lbtal_level_topk=level_topk,
                lbtal_min_level_k=1, lbtal_quality_gate=0.0)

DYS_YAML = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""

WIDE_YAML = """nc: 3
end2end: True
reg_max: 1
scales: # model compound scaling constants, i.e. 'model=yolo26n-p2.yaml' will call yolo26-p2.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 329 layers, 2,662,400 parameters, 2,662,400 gradients, 9.5 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 329 layers, 9,765,856 parameters, 9,765,856 gradients, 27.8 GFLOPs
  m: [0.50, 1.00, 512] # summary: 349 layers, 21,144,288 parameters, 21,144,288 gradients, 91.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 489 layers, 25,815,520 parameters, 25,815,520 gradients, 115.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 489 layers, 57,935,232 parameters, 57,935,232 gradients, 256.9 GFLOPs

# YOLO26n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [256, True]] # 19 (P2/4-xsmall) WIDENED 128->256

  - [-1, 1, Conv, [256, 3, 2]] # 20 match widened P2
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""


RUNS = [
    {"name": "y26_dys_p2starve", "cfg": "y26_p2_dysample.yaml", "yaml": DYS_YAML,
     "expect": {"DySample": 1}, "seed": SEED,
     "params": _lb({4: 1, 8: 3, 16: 4, 32: 4}),
     "label": "best arch + LB-TAL fixed {4:1, 8:3, 16:4, 32:4} — starve P2",
     "why": "The direct test. If large objects are being assigned to stride-4 "
            "anchors and then mis-scored, capping P2 at ONE slot per GT should "
            "hand them back to P4/P5. CONFIRMS if mAP50_large climbs toward 81 "
            "while AR50_large stays near 91.6 — score repaired, detection "
            "untouched — and small holds above 51.5. If large recovers but "
            "small collapses to baseline, P2 was carrying both and the trade is "
            "real rather than a bug."},

    {"name": "y26_dys_p2rich", "cfg": "y26_p2_dysample.yaml", "yaml": DYS_YAML,
     "expect": {"DySample": 1}, "seed": SEED,
     "params": _lb({4: 4, 8: 3, 16: 2, 32: 1}),
     "label": "best arch + LB-TAL fixed {4:4, 8:3, 16:2, 32:1} — enrich P2",
     "why": "The opposite budget, and the reason run 1 is worth believing. One "
            "budget that happens to help is a draw from a 0.46 sd distribution. "
            "A budget and its mirror moving large in OPPOSITE directions is a "
            "mechanism. CONFIRMS if large falls below 53.72, i.e. worse than "
            "doing nothing. If both budgets improve large, the effect is LB-TAL "
            "regularisation generally and has nothing to do with P2, which is "
            "worth knowing and would kill the size-conditioned idea."},

    {"name": "y26_p2_wide", "cfg": "y26_p2_wide.yaml", "yaml": WIDE_YAML,
     "expect": {"DySample": 1}, "seed": SEED, "params": dict(_STOCK),
     "label": "best arch, P2 branch widened 128 -> 256 (64 -> 128 ch at scale s)",
     "why": "Every one of the 9 architecture runs so far kept the shipped "
            "yolo26-p2 skeleton and only bolted modules onto the Detect inputs. "
            "Channel width was never varied. At scale s the P2 branch carries "
            "just 64 channels while P3 carries 128 — the finest level, where "
            "the small objects are, is the THINNEST. Doubling it is a different "
            "kind of change from adding a block and is one of the few levers "
            "plausibly worth more than the 0.46 sd noise floor. Costs a little "
            "latency at stride 4; watch that small clears 52.39 to justify it."},

    {"name": "y26_wide_starve", "cfg": "y26_p2_wide.yaml", "yaml": WIDE_YAML,
     "expect": {"DySample": 1}, "seed": SEED,
     "params": _lb({4: 1, 8: 3, 16: 4, 32: 4}),
     "label": "wide P2 + LB-TAL starve P2 — the both-levers cell, and the best shot at a record",
     "why": "This completes a 2x2 rather than adding a ninth one-off: "
            "{P2 128ch, P2 256ch} x {stock loss, starve P2}, with the already-run "
            "y26_p2_dysample (55.94) sitting in the empty corner. That matters "
            "here specifically. Every 'effect' quoted in this project came from "
            "subtracting two single runs, and those estimates contradicted each "
            "other badly — gctx measured +0.72 and -1.16 depending on its "
            "neighbour. A factorial MEASURES the interaction instead of assuming "
            "it is zero. If wide and starve are each positive and this cell "
            "exceeds both, that is a real super-additive result and the best "
            "model in the project. If this cell lands between its parents the "
            "levers are redundant, which is equally worth knowing. Widening P2 "
            "gives the finest level more capacity; starving P2 stops it "
            "capturing large GTs it then mis-scores. The two are aimed at "
            "opposite ends of the size range and have no obvious reason to "
            "conflict — which is exactly the kind of prediction that has been "
            "wrong before here, so read the cell, not the reasoning."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def preflight(todo):
    try:
        import ultralytics
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        from ultralytics.utils import tal as TAL
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    missing = [n for n in CUSTOM_MODULES if not hasattr(M, n)]
    if missing:
        print(f"  [ABORT] modules not importable: {missing} — run patch_ultralytics_modules.py")
        return False
    if not hasattr(TAL, "LevelBalancedTaskAlignedAssigner"):
        print("  [ABORT] LevelBalancedTaskAlignedAssigner missing from utils/tal.py")
        return False
    src = open(T.__file__, encoding="utf-8").read()
    if "elif m is DySample:" not in src:
        print("  [ABORT] DySample not registered in parse_model")
        return False
    print(f"  modules     : OK   LB-TAL: OK")

    # a budget dict on a 4-level model must resolve all four strides, not silently
    # fall back to min_level_k on P2 — that would make runs 1 and 2 identical.
    A = TAL.LevelBalancedTaskAlignedAssigner
    for r in todo:
        lt = r["params"].get("lbtal_level_topk")
        if not lt:
            continue
        inst = A(topk=10, level_topk_mode="fixed", level_topk=lt, min_level_k=1)
        got = inst._per_level_budget([4.0, 8.0, 16.0, 32.0])
        want = {float(k): v for k, v in lt.items()}
        if {float(k): v for k, v in got.items()} != want:
            print(f"  [ABORT] {r['name']}: budget resolved to {got}, expected {want}")
            return False
        print(f"  budget check: {r['name']:<18} -> { {int(k): v for k, v in got.items()} }  OK")

    print(f"\n  [!] b{BATCH}/{IMG_SIZE} — directly comparable to the arch rounds.")
    print(f"      y26_p2_dysample {BEST * 100:.2f} | y26_p2_b32 {ANCHOR * 100:.2f} | "
          f"family sd {FAMILY_SD:.2f} pp")
    print(f"      Treat anything under {FAMILY_SD * 2:.1f} pp as unresolved until seed 1 lands.")

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


def count_custom(model):
    seen = {}
    for m in model.modules():
        n = type(m).__name__
        if n in CUSTOM_MODULES:
            seen[n] = seen.get(n, 0) + 1
    return seen


def attach_epoch_callback(model):
    """Feed the epoch to SWA. E2E builds TWO v8DetectionLoss objects; walk both."""
    def iter_bbox_losses(crit):
        for holder in (crit, getattr(crit, "one2many", None), getattr(crit, "one2one", None)):
            bl = getattr(holder, "bbox_loss", None) if holder is not None else None
            if bl is not None:
                yield bl

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        n = 0
        for bl in iter_bbox_losses(crit):
            for attr, val in (("current_epoch", trainer.epoch), ("total_epochs", trainer.epochs)):
                if hasattr(bl, attr):
                    setattr(bl, attr, val)
            n += 1
        if n == 0 and trainer.epoch >= 1:
            raise RuntimeError("epoch callback reached NO BboxLoss — criterion layout changed")

    model.add_callback("on_train_epoch_start", on_epoch_start)


def run_one(rc):
    name, seed = rc["name"], rc.get("seed", SEED)
    cfg = save_yaml(rc["yaml"], os.path.join(YAML_DIR, rc["cfg"]))
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={seed}")
    for k, v in sorted(rc["params"].items()):
        print(f"    {k:<22}{v}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    built = count_custom(model.model)
    print(f"  custom layers: {built}   expected: {rc['expect']}")
    if built != rc["expect"]:
        raise RuntimeError(f"{name}: graph has {built}, yaml declares {rc['expect']}")
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] weight transfer failed: {e}")
    attach_epoch_callback(model)

    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "cfg": rc["cfg"], "custom_layers": built, "seed": seed,
           "params": rc["params"], "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "ov4_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
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
    print(f"\n{'=' * 86}\n  YOLO26 OVERNIGHT 4 — s/{IMG_SIZE}/b{BATCH}\n{'=' * 86}")
    print(f"{'run':<20}{'seed':>5}{'mAP50':>9}{'mAP50-95':>10}{'vs best':>9}{'vs anchor':>11}{'h':>6}")
    print('-' * 86)
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        print(f"{r['name']:<20}{r.get('seed', SEED):>5}{r['test_map50'] * 100:>9.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{(r['test_map5095'] - BEST) * 100:>+9.2f}"
              f"{(r['test_map5095'] - ANCHOR) * 100:>+11.2f}{r['hours']:>6.1f}")
    print('-' * 86)
    print(f"{'y26_p2_dysample':<20}{0:>5}{81.09:>9.2f}{BEST * 100:>10.2f}   <- to beat / replicate")
    print(f"{'y26_p2_b32':<20}{0:>5}{80.29:>9.2f}{ANCHOR * 100:>10.2f}   <- anchor")

    # ---- 2x2 factorial: {P2 width} x {P2 assignment budget} -------------------
    cells = {("128", "stock"): BEST}  # y26_p2_dysample, already run
    for k, n in ((("128", "starve"), "y26_dys_p2starve"),
                 (("256", "stock"), "y26_p2_wide"),
                 (("256", "starve"), "y26_wide_starve")):
        if n in by:
            cells[k] = by[n]
    if len(cells) == 4:
        c = {k: v * 100 for k, v in cells.items()}
        print(f"\n  2x2  P2 WIDTH x P2 BUDGET        stock      starve     effect")
        for w in ("128", "256"):
            print(f"    P2 = {w:>3} ch                  {c[(w, 'stock')]:8.2f}{c[(w, 'starve')]:12.2f}"
                  f"{c[(w, 'starve')] - c[(w, 'stock')]:+11.2f}")
        print(f"    width effect              {c[('256', 'stock')] - c[('128', 'stock')]:+8.2f}"
              f"{c[('256', 'starve')] - c[('128', 'starve')]:+12.2f}")
        main_w = ((c[("256", "stock")] + c[("256", "starve")])
                  - (c[("128", "stock")] + c[("128", "starve")])) / 2
        main_b = ((c[("128", "starve")] + c[("256", "starve")])
                  - (c[("128", "stock")] + c[("256", "stock")])) / 2
        inter = (c[("256", "starve")] - c[("256", "stock")]) - (c[("128", "starve")] - c[("128", "stock")])
        print(f"\n    main effect  width         {main_w:+.2f} pp")
        print(f"    main effect  starve        {main_b:+.2f} pp")
        print(f"    INTERACTION                {inter:+.2f} pp")
        if abs(inter) > 2 * FAMILY_SD:
            print(f"      |interaction| > 2 sd — the levers are NOT independent. Every")
            print(f"      single-run 'effect' quoted in this project is context-bound,")
            print(f"      which is what gctx (+0.72 / -1.16) already suggested.")
        else:
            print(f"      Within noise — additive is a defensible assumption for these two.")
        best_cell = max(c, key=c.get)
        print(f"\n    best cell: P2={best_cell[0]}ch / {best_cell[1]}  =  {c[best_cell]:.2f}"
              f"  ({c[best_cell] - BEST * 100:+.2f} vs y26_p2_dysample)")
        print(f"    NOTE: no seed repeat was run. With family sd {FAMILY_SD:.2f} pp, treat")
        print(f"    anything under ~{2 * FAMILY_SD:.1f} pp as unresolved.")

    if "y26_dys_p2starve" in by and "y26_dys_p2rich" in by:
        s, r = by["y26_dys_p2starve"] * 100, by["y26_dys_p2rich"] * 100
        print(f"\n  P2 BUDGET CONTRAST   starve {s:.2f}   enrich {r:.2f}   spread {s - r:+.2f} pp")
        print("    The overall numbers are secondary. Run the per-size eval and read LARGE:")
        print("      hypothesis holds  -> starve mAP50_large climbs toward 81, AR50_large stays ~91.6,")
        print("                           enrich falls below 53.72")
        print("      both improve      -> generic LB-TAL regularisation, NOT a P2 effect")
        print("      neither moves     -> assignment is not the mechanism; drop the story")

    print(f"\n  Per-size is where all four are decided — overall mAP will not show it:")
    for r in res:
        if r.get("weights"):
            print(f"    CocoEvalAllFolders_luggage.py  {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 86}\n  YOLO26 OVERNIGHT — {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
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
            res.append({"name": r["name"], "cfg": r["cfg"], "seed": r.get("seed", SEED),
                        "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
