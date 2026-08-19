#!/usr/bin/env python3
"""
YOLO26 ROUND 6 — the P4/P5 allocation axis, with the positive COUNT held fixed.

=============================================================================
THE CONFOUND THIS ROUND REMOVES
=============================================================================
Round 5 said the large-object recovery came from feeding P4/P5 (+3.37 pp main
effect) rather than from starving P2 (+0.68). That reading is not safe, because
tal_topk is hardcoded to 10 (loss.py:1402) and the assigner CAPS the union of
the per-level picks at topk. The five budgets run so far do not sum to the same
number:

    p2k1_lo  {4:1, 8:3, 16:2, 32:1} =  7   under cap -> ~7 positives per GT
    rich     {4:4, 8:3, 16:2, 32:1} = 10   at cap    -> 10
    starve   {4:1, 8:3, 16:4, 32:4} = 12   capped    -> 10
    p2k2_hi  {4:2, 8:3, 16:4, 32:4} = 13   capped    -> 10
    p2k4_hi  {4:4, 8:3, 16:4, 32:4} = 15   capped    -> 10

Every "P4/P5 high" cell exceeded the cap and every "low" cell did not, so the
main effect I reported also contains a change in how many positives each GT
receives. That was my error: I designed the square without checking the cap.

One thing already argues composition matters on its own — rich and p2k4_hi both
deliver 10 positives per GT and differ by 5.10 pp on large (54.51 vs 59.61).
But that is a single pair, and the axis has never been swept cleanly.

=============================================================================
THE DESIGN — every budget sums to EXACTLY 10
=============================================================================
The cap never binds, so the allocation each run receives is exactly the one
written, and the number of positives per GT is identical across all of them.
The only thing varying is WHERE those ten positives go.

    run           budget                     P4+P5 share   P4:P5
    rich (DONE)   {4:4, 8:3, 16:2, 32:1}        3 / 10      2:1
    s10_bal       {4:2, 8:3, 16:3, 32:2}        5 / 10      3:2
    s10_hi        {4:1, 8:2, 16:3, 32:4}        7 / 10      3:4
    s10_p45       {4:1, 8:1, 16:4, 32:4}        8 / 10      4:4
    s10_p5        {4:1, 8:1, 16:2, 32:6}        8 / 10      2:6

rich, s10_bal, s10_hi, s10_p45 form a monotone share axis at 3/5/7/8 out of 10.
If large climbs monotonically along it, the mechanism is established cleanly for
the first time in this project. If it is flat or non-monotone, then the round-5
"+3.37" was the CAP doing the work, not the allocation, and the whole P4/P5
story collapses the same way the P2 story did.

s10_p45 and s10_p5 carry the SAME share (8/10) split 4:4 versus 2:6. That pair
separates P4 from P5 at fixed share and fixed count — the only clean way to ask
which level actually carries large objects.

=============================================================================
WHAT TO READ
=============================================================================
LARGE, mAP50-95. Reference points, all b32/640/seed0:

    yolo26_custom-9  60.87    stock baseline, still unbeaten on large in 30 runs
    y26_p2k4_hi      59.61    best custom large so far
    y26_dys_p2starve 58.55
    y26_p2k2_hi      57.53    best OVERALL (56.46, +1.22 vs baseline)
    y26_dys_p2rich   54.51
    y26_p2_dysample  53.72    no budget at all

Overall mAP is decoration here: it spanned 55.91-56.46 across all of round 5
while large spanned 5.10 pp. Family sd on overall is 0.49 pp, so ignore overall
differences under ~1.0 pp.

The prize: something that holds small near 52.4 AND pushes large past 59.61.
p2k2_hi currently sits at small 52.37 / large 57.53.

Architecture is FIXED at y26_p2_dysample — byte-identical to the yaml that
produced 55.94 — so the assigner budget is the only thing that varies.
Batch 32, imgsz 640, seed 0.
REQUIRES the module port (DySample) on the import path.

Usage:
    python run_yolo26_round6_v6i.py
    python run_yolo26_round6_v6i.py y26_s10_p45 y26_s10_p5
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
PROJECT_DIR = "runs_yolo26_round6_v6i"
YAML_DIR = "arch_yamls_y26_r6"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

TAL_TOPK = 10  # hardcoded in loss.py; every budget below sums to exactly this

# measured, b32/640/seed0 — (overall, small, large)
REF = {
    "yolo26_custom-9":  (0.5524, 0.5100, 0.6087),
    "y26_p2_dysample":  (0.5594, 0.5239, 0.5372),
    "y26_dys_p2rich":   (0.5607, 0.5199, 0.5451),
    "y26_dys_p2starve": (0.5603, 0.5220, 0.5855),
    "y26_p2k2_hi":      (0.5646, 0.5237, 0.5753),
    "y26_p2k4_hi":      (0.5591, 0.5169, 0.5961),
}
FAMILY_SD = 0.49  # pp, overall


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


RUNS = [
    {"name": "y26_s10_bal", "seed": SEED, "budget": {4: 2, 8: 3, 16: 3, 32: 2},
     "label": "sum-10 balanced — P4+P5 share 5/10",
     "why": "The first point above rich (3/10) on the share axis. rich is the "
            "only sum-10 budget measured so far and it sits at large 54.51, the "
            "worst of the five. If share is the variable, this should land "
            "between rich and starve. It also keeps P2 at 2, matching p2k2_hi, "
            "so the P2 slot is held while the share moves."},

    {"name": "y26_s10_hi", "seed": SEED, "budget": {4: 1, 8: 2, 16: 3, 32: 4},
     "label": "sum-10 P4/P5-heavy — share 7/10",
     "why": "Third point on the axis. With rich (3/10) and s10_bal (5/10) this "
            "gives 3/5/7 at a constant ten positives per GT. Three points is the "
            "minimum that can distinguish a monotone trend from noise, which is "
            "the whole reason for running the axis rather than one more budget."},

    {"name": "y26_s10_p45", "seed": SEED, "budget": {4: 1, 8: 1, 16: 4, 32: 4},
     "label": "sum-10 maximal P4/P5 — share 8/10, split 4:4",
     "why": "The endpoint of the axis, and the closest sum-10 analogue of "
            "p2k4_hi's {16:4, 32:4} — which holds the best custom large at "
            "59.61 but got there with a budget summing to 15 that the cap cut "
            "to 10. If this reproduces ~59.6, the allocation is what matters and "
            "the cap was irrelevant. If it lands near 54, then the round-5 "
            "effect was the cap, not the composition, and the P4/P5 story dies "
            "the same way the P2 story did."},

    {"name": "y26_s10_p5", "seed": SEED, "budget": {4: 1, 8: 1, 16: 2, 32: 6},
     "label": "sum-10, share 8/10 but split 2:6 toward P5",
     "why": "Same share as s10_p45, same count, different split. This is the "
            "only clean way to ask WHICH coarse level carries large objects. "
            "P5 is stride 32, where a large box needs the fewest anchors to "
            "cover it; P4 is stride 16. Nothing measured so far separates them "
            "— every budget moved both together. If s10_p5 matches s10_p45 the "
            "levels are interchangeable and only the coarse/fine ratio matters; "
            "if they diverge, one specific level is the lever."},
]

for _r in RUNS:  # the design depends on this; fail loudly rather than silently
    assert sum(_r["budget"].values()) == TAL_TOPK, (_r["name"], _r["budget"])
    _r["params"] = _lb(_r["budget"])
    _r["share"] = (_r["budget"][16] + _r["budget"][32]) / TAL_TOPK


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
    if not hasattr(M, "DySample"):
        print("  [ABORT] DySample not importable — run patch_ultralytics_modules.py")
        return False
    if not hasattr(TAL, "LevelBalancedTaskAlignedAssigner"):
        print("  [ABORT] LevelBalancedTaskAlignedAssigner missing from utils/tal.py")
        return False
    if "elif m is DySample:" not in open(T.__file__, encoding="utf-8").read():
        print("  [ABORT] DySample not registered in parse_model")
        return False

    # the cap must genuinely not bind, or every run collapses toward the same
    # allocation and the axis measures nothing.
    src = open(os.path.join(os.path.dirname(ultralytics.__file__), "utils", "loss.py"),
               encoding="utf-8").read()
    if "v8DetectionLoss(model, tal_topk=10)" not in src:
        print("  [WARN] one2many tal_topk is not the expected 10 — budgets sum to "
              f"{TAL_TOPK} and the sum-10 design assumes topk=10. Check loss.py.")
    A = TAL.LevelBalancedTaskAlignedAssigner
    for r in todo:
        lt = r["params"]["lbtal_level_topk"]
        got = A(topk=TAL_TOPK, level_topk_mode="fixed", level_topk=lt,
                min_level_k=1)._per_level_budget([4.0, 8.0, 16.0, 32.0])
        if {float(k): v for k, v in got.items()} != {float(k): v for k, v in lt.items()}:
            print(f"  [ABORT] {r['name']}: budget resolved to {got}, expected {lt}")
            return False
        print(f"  {r['name']:<14} { {int(k): v for k, v in got.items()} }  sum={sum(lt.values())}"
              f"  P4+P5 share={r['share']:.1f}  OK")

    print(f"\n  [!] arch FIXED at y26_p2_dysample; only the budget varies, and every")
    print(f"      budget sums to exactly {TAL_TOPK} so the cap never binds.")
    print(f"      Read LARGE: baseline 60.87 | p2k4_hi 59.61 | p2k2_hi 57.53 | rich 54.51")
    print(f"      Overall sd {FAMILY_SD:.2f} pp — ignore overall deltas under ~1.0 pp.")

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


def attach_epoch_callback(model):
    """Feed the epoch to SWA. E2E builds TWO v8DetectionLoss objects; walk both."""
    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        n = 0
        for holder in (crit, getattr(crit, "one2many", None), getattr(crit, "one2one", None)):
            bl = getattr(holder, "bbox_loss", None) if holder is not None else None
            if bl is None:
                continue
            for attr, val in (("current_epoch", trainer.epoch), ("total_epochs", trainer.epochs)):
                if hasattr(bl, attr):
                    setattr(bl, attr, val)
            n += 1
        if n == 0 and trainer.epoch >= 1:
            raise RuntimeError("epoch callback reached NO BboxLoss — criterion layout changed")
    model.add_callback("on_train_epoch_start", on_epoch_start)


def run_one(rc):
    name, seed = rc["name"], rc.get("seed", SEED)
    cfg = save_yaml(DYS_YAML, os.path.join(YAML_DIR, "y26_p2_dysample.yaml"))
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  cfg={cfg}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={seed}")
    print(f"  budget = {rc['budget']}   sum={sum(rc['budget'].values())}   "
          f"P4+P5 share={rc['share']:.1f}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(cfg)
    n_dys = sum(1 for m in model.model.modules() if type(m).__name__ == "DySample")
    print(f"  DySample layers: {n_dys}  (expected 1)")
    if n_dys != 1:
        raise RuntimeError(f"{name}: graph has {n_dys} DySample, expected 1")
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
    out = {"name": name, "budget": rc["budget"], "share": rc["share"], "seed": seed,
           "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "r6_params.json"), "w") as f:
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    print(f"\n{'=' * 88}\n  YOLO26 ROUND 6 — sum-10 budgets, arch fixed at y26_p2_dysample")
    print(f"{'=' * 88}")
    print(f"{'run':<16}{'budget':<28}{'share':>7}{'mAP50':>9}{'mAP50-95':>10}{'vs p2k2_hi':>12}{'h':>6}")
    print('-' * 88)
    BEST = REF["y26_p2k2_hi"][0]
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        b = "{" + ", ".join(f"{k}:{v}" for k, v in sorted(r["budget"].items())) + "}"
        print(f"{r['name']:<16}{b:<28}{r['share']:>7.1f}{r['test_map50'] * 100:>9.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{(r['test_map5095'] - BEST) * 100:>+12.2f}{r['hours']:>6.1f}")
    print('-' * 88)
    for n, (o, s, l) in sorted(REF.items(), key=lambda kv: -kv[1][0]):
        print(f"  {n:<20}overall {o * 100:6.2f}   small {s * 100:6.2f}   large {l * 100:6.2f}")

    print(f"\n  THE AXIS — fill in LARGE from CocoEvalAllFolders_luggage.py:")
    print(f"    share 0.3  rich       large 54.51   (measured)")
    for r in sorted(ok, key=lambda x: x["share"]):
        print(f"    share {r['share']:.1f}  {r['name']:<11} large ____")
    print(f"\n    monotone rise with share -> allocation is the mechanism, cleanly, at last")
    print(f"    flat or non-monotone     -> round 5's +3.37 was the CAP, not the")
    print(f"                                allocation, and the P4/P5 story dies too")
    print(f"\n    s10_p45 (4:4) vs s10_p5 (2:6), same share and count:")
    print(f"      similar   -> only the coarse/fine ratio matters, P4 and P5 interchangeable")
    print(f"      divergent -> one specific level carries large objects")
    print(f"\n    s10_p45 near 59.6 would reproduce p2k4_hi WITHOUT the cap and settle it.")
    for r in ok:
        if r.get("weights"):
            print(f"      {r['name']:<14} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 88}\n  YOLO26 ROUND 6 — {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}\n{'=' * 88}\n")
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
            res.append({"name": r["name"], "budget": r["budget"], "share": r["share"],
                        "seed": r.get("seed", SEED), "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
