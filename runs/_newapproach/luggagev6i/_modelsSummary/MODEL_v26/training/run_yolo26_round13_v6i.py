#!/usr/bin/env python3
r"""
ROUND 13 — is the P2 head width the reason the P2 level lost?

ONE run, deterministic, exact. No replicates needed.

=============================================================================
THE OBSERVATION
=============================================================================
`y26_p2_b32` (stock yolo26-p2, plain nn.Upsample, no DySample) scored 55.03
against `y26_stock_b32` 55.76. Adding a P2 level LOST 0.73 on a dataset that is
60% small objects. That is backwards, and there is a mechanism for it.

From Detect.__init__:

    c2, c3 = max((16, ch[0] // 4, reg_max * 4)), max(ch[0], min(nc, 100))

Both head widths are tied to ch[0] -- the FINEST pyramid level. With nc = 3 the
`min(nc, 100) = 3` floor never binds, so c3 IS ch[0]:

    stock 3-level   ch = (128, 256, 512)        ch[0] = 128 -> c3 = 128, c2 = 32
    + P2            ch = (64, 128, 256, 512)    ch[0] =  64 -> c3 =  64, c2 = 16

So "add a P2 level" is really "add a P2 level AND halve both detection heads on
every level". The two have never been separated. On COCO (nc = 80) the floor
absorbs most of it (128 -> 80); at nc = 3 it is a clean halving, so this is a
LOW-CLASS-COUNT pathology and this dataset is exactly where it bites.

`y26_p2_wide` is not this experiment: it widened layer 19 from 128 to 256, which
changes P2 FEATURE capacity and head width together, and it ran on the DySample
graph where sd is 0.19.

=============================================================================
WHY ONE RUN IS ENOUGH
=============================================================================
The sd 0.19 that makes the architecture table unreadable comes from DySample ->
F.grid_sample, whose CUDA backward uses atomic adds. THIS GRAPH HAS NO DYSAMPLE.
The stock path reproduces bit-identically (y26_base_rep matched yolo26_custom-9
across all 118 metric values), so a single run here is an exact measurement, the
same way the b82 loss campaign was.

    lands near/above 55.76  -> the head halving caused the loss. P2 does help
                               this dataset, and the fix is one line.
    stays near 55.03        -> P2 genuinely does not pay at 640. Cleanly closed,
                               and the -0.73 is not an artifact.

Both outcomes are reportable. There is no null result here, only two answers.

=============================================================================
TRANSFER IS EQUAL ON BOTH SIDES
=============================================================================
yolo26s.pt is a 3-level checkpoint: cv2/cv3 are ModuleLists of length 3 built
for ch = (128, 256, 512). Any 4-level model mismatches them by shape at every
index, so NEITHER y26_p2_b32 NOR this run inherits a pretrained head. The
comparison is fresh-head vs fresh-head; the only variable is c2/c3.

=============================================================================
COST
=============================================================================
c3 128 vs 64 roughly doubles the classification head. It runs at every level,
and P2 is 160x160 at 640 input, so this is the expensive one: about +0.9 GFLOPs
at P2, ~3-4% on a 28 GFLOP model. Affordable. Note max(ch) would have set
c3 = 512 from P5 -- a 4x head at P2 resolution -- which is why head_ch is an
explicit value and not max(ch).

    Usage:
        python run_yolo26_round13_v6i.py                 # the one run
        python run_yolo26_round13_v6i.py --with-control  # + re-run y26_p2_b32
"""

import gc
import json
import os
import sys
import time

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
CFG_P2 = "yolo26-p2.yaml"  # 4 levels, plain nn.Upsample -> deterministic
CFG_3LVL = "yolo26.yaml"  # 3 levels, the stock graph
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_round13_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32  # matches y26_p2_b32 (55.03) and y26_stock_b32 (55.76)
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

STOCK_B32 = 55.76  # y26_stock_b32   — 3 levels, c3 = 128
P2_B32 = 55.03  # y26_p2_b32      — 4 levels, c3 =  64  (the -0.73)
HEAD_CH = 128  # pin c3 to the stock 3-level width
# =============================================================================

RUNS = [
    {"name": "y26_p2_headref128", "cfg": CFG_P2, "head_ch": HEAD_CH, "want_c3": 128, "want_nl": 4,
     "label": f"yolo26-p2, head width pinned at {HEAD_CH} (the stock 3-level value)",
     "why": "Isolates head width from pyramid depth. y26_p2_b32 changed both at once because "
            "c3 is derived from ch[0]. Compare against y26_p2_b32 = 55.03: both are 4-level "
            "and both have fresh heads, so c3 is the only variable."},

    {"name": "y26_3lvl_head64", "cfg": CFG_3LVL, "head_ch": 64, "want_c3": 64, "want_nl": 3,
     "label": "stock 3-level with the head SHRUNK to 64 (the P2 graph's value)",
     "why": "The other diagonal, and the run that makes the square readable. y26_stock_b32 "
            "55.76 keeps its pretrained head; every 4-level run trains one from scratch, so "
            "the -0.73 confounds depth, head width AND transfer. This cell is 3-level with a "
            "FRESH 64-wide head, so: p2@64 - 3lvl@64 is the pure P2-level effect (transfer and "
            "width both held), and 3lvl@128 - 3lvl@64 exposes the transfer penalty by "
            "subtraction. Without it, run 1 alone cannot attribute anything."},

    {"name": "y26_p2_headref0", "cfg": CFG_P2, "head_ch": 0, "want_c3": 64, "want_nl": 4,
     "label": "yolo26-p2 stock head width — determinism control",
     "why": "Optional. Should reproduce y26_p2_b32 = 55.03 exactly. Run it if you want to "
            "confirm this box is still bit-deterministic on a no-DySample graph before "
            "trusting single-run comparisons."},
]


def preflight(todo):
    """Prove head_ch exists and actually moves c2/c3 before spending 2 GPU-h."""
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    if not hasattr(Detect, "head_ch"):
        print("  [ABORT] Detect.head_ch missing — head.py is not patched on this machine.")
        return False

    # Build throwaway heads and prove the knob changes the width it claims to.
    for tag, ch, hc, want in (("yolo26-p2 (4 lvl)", (64, 128, 256, 512), 0, 64),
                              ("yolo26-p2 (4 lvl)", (64, 128, 256, 512), HEAD_CH, HEAD_CH),
                              ("yolo26   (3 lvl)", (128, 256, 512), 0, 128),
                              ("yolo26   (3 lvl)", (128, 256, 512), 64, 64)):
        Detect.head_ch = hc
        got = Detect(nc=3, reg_max=1, end2end=True, ch=ch).cv3[0][-1].in_channels
        print(f"  {tag}  head_ch={hc:<4} -> c3={got:<4} expected {want}")
        if got != want:
            Detect.head_ch = 0
            print("  [ABORT] the knob is not doing what these runs claim.")
            return False
    Detect.head_ch = 0  # leave the class clean; run_one sets it per run
    print(f"  {'knob verified on both graphs':<44}True")

    print()
    print(f"  weights {MODEL_WEIGHTS}  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  reference: y26_stock_b32 {STOCK_B32} (3 lvl, c3=128, PRETRAINED head)")
    print(f"             y26_p2_b32   {P2_B32} (4 lvl, c3= 64, fresh head)")
    print("  no DySample in either graph -> deterministic -> n=1 is exact")
    clash = [os.path.join(PROJECT_DIR, r["name"]) for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))] if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that the BUILT model has the head width and depth this run claims."""
    state = {"verified": False}

    def on_epoch_start(trainer):
        if state["verified"] or trainer.epoch < 1:
            return
        det = trainer.model.model[-1]
        got = det.cv3[0][-1].in_channels
        if got != rc["want_c3"]:
            raise RuntimeError(f"{rc['name']}: built c3={got}, expected {rc['want_c3']}. "
                               f"Detect.head_ch did not reach the constructed model — this run "
                               f"would measure nothing, exactly like rounds 4-6.")
        if det.nl != rc["want_nl"]:
            raise RuntimeError(f"{rc['name']}: built {det.nl} levels, expected {rc['want_nl']} "
                               f"— wrong cfg for this cell of the square.")
        o2o = det.one2one_cv3[0][-1].in_channels if hasattr(det, "one2one_cv3") else None
        if o2o is not None and o2o != rc["want_c3"]:
            raise RuntimeError(f"{rc['name']}: one2one c3={o2o}, expected {rc['want_c3']}")
        print(f"  [guard] c3={got} on cv3 and one2one_cv3 | c2={det.cv2[0][-1].in_channels} "
              f"| levels={det.nl} | reg_max={det.reg_max}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    t0 = time.time()

    Detect.head_ch = rc["head_ch"]  # class attribute, read in Detect.__init__ at build time
    try:
        model = YOLO(rc["cfg"])
        if MODEL_WEIGHTS:
            # 3-level checkpoint: backbone/neck transfer; heads only where shapes match.
            model.load(MODEL_WEIGHTS)
        state = attach_guard(model, rc)
        results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    finally:
        Detect.head_ch = 0  # never leave the class attribute set for a later build

    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "head_ch": rc["head_ch"], "cfg": rc["cfg"], "nl": rc["want_nl"],
           "c3": rc["want_c3"], "batch": BATCH, "imgsz": IMG_SIZE, "seed": SEED,
           "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    if not ok:
        print("\nno completed runs.")
        return
    by = {r["name"]: r["test_map5095"] * 100 for r in ok}

    print("\n" + "=" * 78)
    print("  ROUND 13 — RESULTS")
    print("=" * 78)
    print(f"{'run':<24}{'lvls':>5}{'c3':>5}{'head':>10}{'mAP50-95':>10}{'hours':>8}")
    print("-" * 62)
    print(f"{'y26_stock_b32':<24}{3:>5}{128:>5}{'PRETRAINED':>10}{STOCK_B32:>10.2f}{'-':>8}")
    print(f"{'y26_p2_b32':<24}{4:>5}{64:>5}{'fresh':>10}{P2_B32:>10.2f}{'-':>8}")
    for r in ok:
        print(f"{r['name']:<24}{r['nl']:>5}{r['c3']:>5}{'fresh':>10}"
              f"{r['test_map5095'] * 100:>10.2f}{r['hours']:>8.2f}")

    p2_128 = by.get("y26_p2_headref128")
    l3_64 = by.get("y26_3lvl_head64")
    ctrl = by.get("y26_p2_headref0")

    print("\n  THE SQUARE (all fresh-head cells)")
    print(f"{'':<14}{'c3=64':>10}{'c3=128':>10}")
    print(f"{'3 levels':<14}{(f'{l3_64:.2f}' if l3_64 else '—'):>10}{f'{STOCK_B32:.2f}*':>10}")
    print(f"{'4 levels (P2)':<14}{P2_B32:>10.2f}{(f'{p2_128:.2f}' if p2_128 else '—'):>10}")
    print("    * pretrained head — NOT fresh, so it is not part of the clean square")

    print("\n  DECOMPOSITION")
    if p2_128 is not None:
        print(f"    head width, 4 lvl   p2@128 - p2@64      {p2_128 - P2_B32:+.2f}")
    if l3_64 is not None:
        print(f"    P2 level, c3=64     p2@64  - 3lvl@64    {P2_B32 - l3_64:+.2f}")
        print(f"    transfer penalty    3lvl@128 - 3lvl@64  {STOCK_B32 - l3_64:+.2f}   "
              f"(head width + pretrained head, confounded)")
    if p2_128 is not None and l3_64 is not None:
        print(f"    P2 level, c3=128    p2@128 - 3lvl@64    {p2_128 - l3_64:+.2f}   "
              f"(upper bound; 3lvl@128-fresh not run)")

    print("\n  READ IT")
    if ctrl is not None:
        d = abs(ctrl - P2_B32)
        print(f"    determinism control differs from y26_p2_b32 by {d:.2f}")
        if d > 0.05:
            print("    -> this graph is NOT bit-deterministic on this box. Everything below")
            print("       needs replicates before it means anything.")
    if p2_128 is not None:
        d = p2_128 - P2_B32
        if d > 0.4:
            print(f"    Head width is worth {d:+.2f} on the P2 graph. The -0.73 attributed to")
            print("    'adding P2' was substantially the head halving, and the fix is one line.")
        elif abs(d) <= 0.4:
            print(f"    Head width moved {d:+.2f} — it is NOT the explanation. The c3 coupling")
            print("    is real in the code but does not cost accuracy here.")
        else:
            print(f"    Wider head is WORSE ({d:+.2f}). Unexpected — suspect the extra head")
            print("    capacity is overfitting at P2 resolution, and report it as such.")
    if l3_64 is not None and p2_128 is not None:
        if P2_B32 - l3_64 > 0.2:
            print("    With width and transfer held constant, the P2 LEVEL itself is positive.")
            print("    That is the opposite of what the raw 55.03 vs 55.76 suggested.")
        elif P2_B32 - l3_64 < -0.2:
            print("    Even with width and transfer held constant, P2 costs accuracy at 640.")
            print("    A clean negative — the level, not the head, is the problem.")
        else:
            print("    The P2 level is neutral once width and transfer are held constant;")
            print("    the raw -0.73 was almost entirely those two confounds.")
    print("=" * 78 + "\n")


def main():
    todo = RUNS if "--with-control" in sys.argv else RUNS[:2]
    if not preflight(todo):
        return
    print(f"\n  {len(todo)} run(s), ~{1.8 * len(todo):.1f} GPU-h\n")
    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"\n  [FAIL] {rc['name']}: {ex}\n")
            res.append({"name": rc["name"], "head_ch": rc["head_ch"], "error": str(ex),
                        "hours": 0.0, "test_map50": float("nan"), "test_map5095": float("nan")})
    try:
        with open(f"{PROJECT_DIR}__runs.json", "w") as f:
            json.dump({"stock_b32": STOCK_B32, "p2_b32": P2_B32, "batch": BATCH,
                       "seed": SEED, "results": res}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] results json not saved: {ex}")
    summarise(res)


if __name__ == "__main__":
    main()
