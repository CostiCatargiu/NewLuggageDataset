#!/usr/bin/env python3
"""
YOLO26 ROUND 5 — separate the two factors inside the p2starve result.

=============================================================================
WHAT ROUND 4 FOUND, AND WHAT IT DID NOT
=============================================================================
y26_dys_p2starve is the best YOLO26 model in the project:

  LARGE            mAP50   mAP50-95    AR50     P50     R50
  yolo26_custom-9  81.75      60.87   92.97   79.07   75.00
  y26_p2_dysample  74.66      53.72   91.56   77.33   70.00
  y26_dys_p2starve 81.54      58.55   93.71   81.36   74.33
                   +6.87      +4.83   +2.14   +4.02   +4.33   vs dysample

Large mAP50 went from 74.66 back to 81.54 against a baseline of 81.75 — the
collapse is essentially repaired — and small only fell 52.39 -> 52.20. Versus
the stock baseline it is better on every bucket except large, where the deficit
shrank from -7.15 to -2.32.

BUT THE EXPERIMENT WAS CONFOUNDED. starve and rich were built as mirrors, so
they differ in THREE places, not one:

    starve  {4:1, 8:3, 16:4, 32:4}
    rich    {4:4, 8:3, 16:2, 32:1}
             ^^^        ^^^^  ^^^^

The +4.05 pp spread on large could come from STARVING P2, or from FEEDING
P4/P5, or from both. Nothing in round 4 distinguishes them. That was a design
error on my part: a mirrored pair looks like a clean contrast and is not one.

=============================================================================
THE 2x2 THAT SEPARATES THEM
=============================================================================
                     P4/P5 low {16:2,32:1}   P4/P5 high {16:4,32:4}
    P2 = 1 slot        y26_p2k1_lo   NEW     y26_dys_p2starve  56.03 / lg 58.55
    P2 = 4 slots       y26_dys_p2rich        y26_p2k4_hi       NEW
                       56.07 / lg 54.51

Two cells are already measured and they sit on one diagonal, so the two new
runs complete the square. Read on LARGE, not on overall mAP — overall barely
moved across every round-4 cell (55.46 to 56.07) while large spanned 4.4 pp.

  P2 is the cause      -> p2k1_lo lands near 58.5, p2k4_hi near 54.5
  P4/P5 is the cause   -> p2k1_lo lands near 54.5, p2k4_hi near 58.5
  both matter          -> both new cells land in between, and the interaction
                          term in the summary is the honest description

=============================================================================
THE THIRD RUN — the frontier point
=============================================================================
{4:2} sits between the two P2 settings already tried. starve gave up 0.19 pp of
small (52.39 -> 52.20) to win 4.83 on large. If two slots holds large near 58
while returning small to 52.4, it is strictly better than anything measured on
either detector. If small and large just trade smoothly, then {4:1} vs {4:2} is
a dial you set from the deployment cost of missing a distant bag versus a
trolley, and that is a legitimate result to report rather than a failure.

=============================================================================
STANDING CAVEAT
=============================================================================
No seed repeat has been run in 28 YOLO26 trainings. The six homogeneous b32
architecture runs have sd 0.49 pp on overall mAP50-95, so:

  overall differences below ~1.0 pp   NOT interpretable
  the large-object effects (+4.83)    far outside that, and they reproduced a
                                      prediction made before the run

Treat the large column as the signal and the overall column as decoration
until a seed repeat exists.

Architecture is FIXED at y26_p2_dysample for all three runs — byte-identical to
the yaml that produced 55.94 — so the only thing varying is the assigner budget.
Batch 32, imgsz 640, seed 0.
REQUIRES the module port (DySample) on the import path.

Usage:
    python run_yolo26_round5_v6i.py
    python run_yolo26_round5_v6i.py y26_p2k4_hi
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
PROJECT_DIR = "runs_yolo26_round5_v6i"
YAML_DIR = "arch_yamls_y26_r5"
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

# measured, b32/640/seed0 — the four numbers this round is read against
REF = {
    "y26_p2_dysample":  (0.5594, 0.5239, 0.5372),   # (overall, small, large)
    "y26_dys_p2starve": (0.5603, 0.5220, 0.5855),   # best so far
    "y26_dys_p2rich":   (0.5607, 0.5199, 0.5451),
    "yolo26_custom-9":  (0.5524, 0.5100, 0.6087),   # stock baseline (b82)
}
FAMILY_SD = 0.49  # pp, six homogeneous b32 arch runs


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
    {"name": "y26_p2k4_hi", "seed": SEED, "params": _lb({4: 4, 8: 3, 16: 4, 32: 4}),
     "cell": ("4", "high"),
     "label": "P2=4 slots, P4/P5 HIGH — starve's budget with ONLY the P2 slot changed",
     "why": "The single most informative run available. It differs from "
            "y26_dys_p2starve (56.03, large 58.55) in ONE number: P2 gets four "
            "slots instead of one. If large falls back toward 54, P2 capture is "
            "the mechanism and the story holds. If large stays near 58.5, then "
            "the win came from feeding P4/P5 and the P2 explanation I have been "
            "building for two rounds is wrong."},

    {"name": "y26_p2k1_lo", "seed": SEED, "params": _lb({4: 1, 8: 3, 16: 2, 32: 1}),
     "cell": ("1", "low"),
     "label": "P2=1 slot, P4/P5 LOW — rich's budget with ONLY the P2 slot changed",
     "why": "The opposite corner, and the reason the square is worth completing "
            "rather than running p2k4_hi alone. It differs from y26_dys_p2rich "
            "(56.07, large 54.51) only in the P2 slot. Together with p2k4_hi it "
            "gives each main effect from four runs instead of two, and yields an "
            "interaction term. Round 4's overall interaction was -0.15 (additive) "
            "while its LARGE interaction was enormous — starve was +4.83 at 64ch "
            "and -0.69 at 128ch — so on this axis interaction is the rule, not "
            "the exception, and it needs measuring rather than assuming."},

    {"name": "y26_p2k2_hi", "seed": SEED, "params": _lb({4: 2, 8: 3, 16: 4, 32: 4}),
     "cell": ("2", "high"),
     "label": "P2=2 slots, P4/P5 HIGH — the frontier point between {4:1} and {4:4}",
     "why": "starve traded 0.19 pp of small (52.39 -> 52.20) for 4.83 on large. "
            "Two slots asks whether that trade is a smooth dial or a cliff. If "
            "small returns to 52.4 while large holds near 58, this is strictly "
            "better than anything measured on either detector and becomes the "
            "config to report. If small and large simply trade, then the slot "
            "count is a deployment choice — distant bags versus trolleys — which "
            "is a publishable finding in its own right, not a failed run."},
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
    if not hasattr(M, "DySample"):
        print("  [ABORT] DySample not importable — run patch_ultralytics_modules.py")
        return False
    if not hasattr(TAL, "LevelBalancedTaskAlignedAssigner"):
        print("  [ABORT] LevelBalancedTaskAlignedAssigner missing from utils/tal.py")
        return False
    if "elif m is DySample:" not in open(T.__file__, encoding="utf-8").read():
        print("  [ABORT] DySample not registered in parse_model")
        return False
    print("  modules     : OK   LB-TAL: OK")

    # every stride must resolve from the dict; a silent min_level_k fallback on
    # P2 would collapse these three runs into each other.
    A = TAL.LevelBalancedTaskAlignedAssigner
    for r in todo:
        lt = r["params"]["lbtal_level_topk"]
        got = A(topk=10, level_topk_mode="fixed", level_topk=lt,
                min_level_k=1)._per_level_budget([4.0, 8.0, 16.0, 32.0])
        if {float(k): v for k, v in got.items()} != {float(k): v for k, v in lt.items()}:
            print(f"  [ABORT] {r['name']}: budget resolved to {got}, expected {lt}")
            return False
        print(f"  budget      : {r['name']:<14} -> { {int(k): v for k, v in got.items()} }  OK")

    print(f"\n  [!] arch FIXED at y26_p2_dysample; only the assigner budget varies.")
    print(f"      Read LARGE. starve 58.55 | rich 54.51 | dysample 53.72 | baseline 60.87")
    print(f"      Overall sd is {FAMILY_SD:.2f} pp — ignore overall deltas under ~1.0 pp.")

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
    print(f"  budget = {rc['params']['lbtal_level_topk']}")
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
    out = {"name": name, "cell": rc["cell"], "seed": seed, "params": rc["params"],
           "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "r5_params.json"), "w") as f:
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
    print(f"\n{'=' * 84}\n  YOLO26 ROUND 5 — assigner budget, arch fixed at y26_p2_dysample")
    print(f"{'=' * 84}")
    print(f"{'run':<20}{'P2':>4}{'P4/P5':>7}{'mAP50':>9}{'mAP50-95':>10}{'vs starve':>11}{'h':>6}")
    print('-' * 84)
    S = REF["y26_dys_p2starve"][0]
    for r in sorted([x for x in res if x["name"] in by], key=lambda x: -x["test_map5095"]):
        print(f"{r['name']:<20}{r['cell'][0]:>4}{r['cell'][1]:>7}{r['test_map50'] * 100:>9.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{(r['test_map5095'] - S) * 100:>+11.2f}{r['hours']:>6.1f}")
    print('-' * 84)
    for n, (o, s, l) in REF.items():
        print(f"{n:<20}{'':>11}{'':>9}{o * 100:>10.2f}   small {s * 100:.2f}  large {l * 100:.2f}")

    print(f"\n  OVERALL sd is {FAMILY_SD:.2f} pp. The decision is in the LARGE column —")
    print(f"  run CocoEvalAllFolders_luggage.py and fill this in:")
    print(f"\n    2x2 ON LARGE          P4/P5 low     P4/P5 high")
    print(f"      P2 = 1 slot          p2k1_lo ___      58.55  (starve)")
    print(f"      P2 = 4 slots           54.51 (rich)   p2k4_hi ___")
    print(f"\n    p2k1_lo ~58.5 and p2k4_hi ~54.5  -> P2 capture is the mechanism")
    print(f"    p2k1_lo ~54.5 and p2k4_hi ~58.5  -> the win was feeding P4/P5, not starving P2")
    print(f"    both in between                  -> both factors contribute; report the interaction")
    print(f"\n    p2k2_hi is the frontier probe: small back to 52.4 AND large near 58")
    print(f"    would beat every config measured on either detector.")
    for r in res:
        if r.get("weights"):
            print(f"      {r['name']:<16} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 84}\n  YOLO26 ROUND 5 — {len(todo)} runs, ~{1.65 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}\n{'=' * 84}\n")
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
            res.append({"name": r["name"], "cell": r["cell"], "seed": r.get("seed", SEED),
                        "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
