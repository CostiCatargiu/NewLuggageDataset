#!/usr/bin/env python3
r"""
Is tal_topk BINDING, and for which object sizes?  (v2 — exact counts, train
split, mosaic-aware)

THE QUESTION. TAL restricts candidates to anchors whose CENTRE falls inside the
GT box (select_candidates_in_gts). If a GT holds fewer than `topk` anchor
centres summed over the three levels, the budget is not binding for that GT:
every candidate is taken, and RANKING them differently cannot change the
assignment. Where topk is slack, tal_beta's SELECTION term is mathematically
inert and only its target-magnitude term does anything.

That is a testable explanation for the beta plateau, so it decides whether
ablating topk is worth GPU time. It needs no model and no forward pass — only
label geometry and the anchor grid — so it runs in seconds on CPU.

    python probe_topk_binding_v2.py


=============================================================================
WHAT v1 GOT RIGHT, AND THE THREE THINGS IT MISSES
=============================================================================
v1's logic is correct: pool < topk => topk cannot bind. Three problems change
the answer rather than refine it.

1. IT COUNTS THE TEST SPLIT.
   Assignment happens during TRAINING. Test geometry is a different sample and
   is not what any assigner ever saw. Fixed: defaults to train/labels, and
   reports every split so the difference is visible rather than assumed.

2. IT IGNORES MOSAIC — this is the big one.
   close_mosaic=10 of EPOCHS=70, so 60 of 70 epochs train on mosaic4. The
   mosaic canvas is 2*imgsz and RandomPerspective scales it by (1-scale, 1+scale)
   = (0.5, 1.5) at the campaign's scale=0.5, so an object arrives at roughly
   0.25x-0.75x its letterboxed linear size, ~0.5x on average. Linear 0.5x is
   AREA 0.25x, hence pool 0.25x. A GT with a 13-anchor pool at eval geometry
   has ~3 during the 86% of training that matters. Any binding statistic
   computed at eval scale is an upper bound and a loose one. Reported as
   scenarios below, because the exact factor is a distribution, not a number.

3. IT USES A CONTINUOUS EXPECTATION, (w/s)*(h/s), NOT A COUNT.
   That is the mean over grid phase. But the statistic in question is
   P(pool < topk), and averaging over phase before thresholding understates the
   spread — a GT with expectation 4.4 has an actual integer count anywhere from
   2 to 9 depending on where it sits. The labels give cx and cy, so the exact
   count is available and no phase averaging is needed. Fixed: exact.


=============================================================================
WHAT THE EXISTING DIAGNOSTIC ALREADY SHOWS, AND WHY IT IS NOT ENOUGH
=============================================================================
DATASET_v6i/raw/diag_anchor_footprin_results.txt, section 0/1:

    size      pool s8   s16   s32   TOTAL    selected_stock
    small     9.82  2.46  0.62   12.90       7.73     <- BELOW the budget of 10
    medium   42.38 10.63  2.71   55.72       9.87
    large   501.02 125.28 31.37  657.67      9.79

Small objects receive 7.73 positives against a budget of 10. The budget is
SLACK for them; medium and large saturate it. That is the finding, and it is
already measured.

But 7.73 < 10 has TWO possible causes and that document cannot separate them:
    (a) pool < 10, pure geometry — the budget is unreachable
    (b) pool >= 10 but the surplus candidates have alignment metric ~0 and are
        pruned by the metric > eps test, which the same document invokes for
        s32 ("those few candidates carry an alignment metric of ~0, so they
        fail the metric>eps test regardless of budget")
Only (a) makes beta's selection term INERT. Under (b) the ranking still
chooses, and lowering topk would be a different intervention entirely.

This probe isolates (a): it is pure geometry, so it answers the half that does
not depend on the model.

TWO CAVEATS ON REUSING THAT TABLE, both easy to miss:
  * IT IS YOLOv12. weights = runs_newl_luggagev6i/yolov12s_default/best.pt. The
    POOL column is pure geometry and transfers to YOLO26 exactly (same 3 levels,
    same strides, same imgsz). The SELECTED column depends on the alignment
    metric and therefore on the model, so 7.73 is an estimate for v26, not a
    measurement of it.
  * IT USES TAXONOMY A (max side 48/96 px). Every results JSON in this project
    uses TAXONOMY C (COCO area 32^2/96^2). The buckets are NOT the same
    population. This probe prints BOTH so the two literatures can be joined.


=============================================================================
AND THE THING THAT IS ACTUALLY BURIED IN THAT FILE
=============================================================================
Finding F3 of the same document reports that the topk axis has ALREADY been
swept on v6i — indirectly, through LB-TAL's per-level P3 budget, which is what
actually binds (F2: "the only binding constraint any scheme applies is how many
stride-8 positives it allows"):

      P3 budget   config                          mAP50-95
          8       lb_prop {8:8,16:2,32:1}           0.5482
        ~6.6      stock (global top-10)             0.5477
          5       lb_uniform_tk13                   0.5542
          4       lb_uniform                        0.5557   <- peak, +0.80
          3       NOT TESTED                           ?
          2       lb_coarse_244                     0.5534

Single-peaked, with a peak worth +0.80 mAP50-95 over stock. That is LARGER on
mAP50-95 than anything tal_beta achieved — beta COSTS mAP50-95.

Before anyone gets excited: those are YOLOv12 runs, scored on mAP50-95, under
taxonomy A. Not one of them has been run on YOLO26 and not one has been scored
on mAP50_small. So this is a lead, not a result. But it means "topk is
untested" is wrong — the correct statement is that it was tested on the other
model, on the metric this campaign has twice been burned by.
"""

import glob
import math
import os
from collections import Counter

# ============================== CONFIG (no args — edit here) =================
DATASET = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12"
SPLITS = ("train", "valid", "test")     # train is the one that matters
IMGSZ = 640
STRIDES = (8, 16, 32)

TOPK_O2M = 10        # E2ELoss: v8DetectionLoss(model, tal_topk=10)
TOPK_O2O = 7         # E2ELoss: v8DetectionLoss(model, tal_topk=7, tal_topk2=1)

# Mosaic scenarios. 60 of 70 epochs are mosaic4 at scale=0.5, so the linear
# size multiplier is roughly uniform on (0.25, 0.75). 1.00 = the final 10
# epochs after close_mosaic, which is also the geometry at eval.
SCENARIOS = [
    ("eval / post-close_mosaic", 1.00),
    ("mosaic, mean scale",       0.50),
    ("mosaic, small end",        0.25),
]

# Taxonomy A: max side px @640 (the footprint diagnostic).
# Taxonomy C: COCO area, sqrt(area) px (every results JSON in this project).
TAX = {
    "A  max side 48/96":   ("maxside", 48.0, 96.0),
    "C  COCO area 32/96":  ("sqrtarea", 32.0, 96.0),
}
# ============================================================================


def anchors_inside_exact(cx, cy, w, h, imgsz=IMGSZ, strides=STRIDES):
    """EXACT number of anchor centres inside the box, summed over levels.

    Centres sit at (i + 0.5) * s. A centre is inside [x1, x2] when
        x1 <= (i + 0.5) * s <= x2   <=>   x1/s - 0.5 <= i <= x2/s - 0.5
    so the count is a floor minus a ceil, clamped to the grid. No phase
    averaging: cx and cy fix the phase, which is the point of doing this from
    labels rather than from a formula.
    """
    x1, x2 = cx - w / 2.0, cx + w / 2.0
    y1, y2 = cy - h / 2.0, cy + h / 2.0
    total, per_level = 0, []
    for s in strides:
        n_cells = int(imgsz // s)
        ix0 = max(0, math.ceil(x1 / s - 0.5))
        ix1 = min(n_cells - 1, math.floor(x2 / s - 0.5))
        iy0 = max(0, math.ceil(y1 / s - 0.5))
        iy1 = min(n_cells - 1, math.floor(y2 / s - 0.5))
        n = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
        per_level.append(n)
        total += n
    return total, per_level


def bucket(w, h, mode, t1, t2):
    v = max(w, h) if mode == "maxside" else (w * h) ** 0.5
    return "small" if v < t1 else "medium" if v < t2 else "large"


def load(split):
    """Return [(cx, cy, w, h)] in px at IMGSZ, plus a count of skipped rows."""
    d = os.path.join(DATASET, split, "labels")
    out, bad = [], 0
    for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 5:
                    if line.strip():
                        bad += 1
                    continue
                if len(p) > 5:
                    bad += 1          # segment row; the 2 known corrupt train files
                    continue
                try:
                    cx, cy, w, h = (float(x) * IMGSZ for x in p[1:5])
                except ValueError:
                    bad += 1
                    continue
                if w <= 0 or h <= 0:
                    continue
                out.append((cx, cy, w, h))
    return out, bad


def pct(v, n):
    return 100.0 * v / n if n else 0.0


def report(gts, tax_name, mode, t1, t2, topk, scale, label):
    pools = {"small": [], "medium": [], "large": []}
    for cx, cy, w, h in gts:
        sw, sh = w * scale, h * scale
        scx, scy = cx * scale, cy * scale     # keep phase coherent with size
        n, _ = anchors_inside_exact(scx, scy, sw, sh)
        pools[bucket(sw, sh, mode, t1, t2)].append(n)

    print(f"    {label:<26} taxonomy {tax_name}   topk={topk}   linear x{scale:.2f}")
    print(f"      {'bucket':<8}{'n':>7}{'mean':>8}{'med':>6}{'p90':>6}"
          f"{'pool<k':>9}{'pool=0':>8}   {'verdict':<22}")
    for k in ("small", "medium", "large"):
        v = sorted(pools[k])
        if not v:
            continue
        n = len(v)
        under = sum(1 for x in v if x < topk)
        zero = sum(1 for x in v if x == 0)
        u = pct(under, n)
        verdict = ("NOT binding" if u > 50 else "binding" if u < 10 else "mixed")
        # what the assigner can actually take, ignoring the metric>eps prune
        cap = sum(min(x, topk) for x in v) / n
        print(f"      {k:<8}{n:>7}{sum(v)/n:>8.1f}{v[n//2]:>6}{v[int(n*0.9)]:>6}"
              f"{u:>8.0f}%{pct(zero,n):>7.0f}%   {verdict:<14} take~{cap:.2f}")
    print()
    return pools


def main():
    print()
    print("=" * 78)
    print("  IS tal_topk BINDING?  exact anchor-centre counts from labels")
    print("=" * 78)
    print(f"  dataset {DATASET}")
    print(f"  imgsz {IMGSZ}   strides {STRIDES}   o2m topk {TOPK_O2M}   o2o topk {TOPK_O2O}")
    print()

    data = {}
    for sp in SPLITS:
        gts, bad = load(sp)
        if not gts:
            print(f"  [skip] {sp}: no labels under {os.path.join(DATASET, sp, 'labels')}")
            continue
        data[sp] = gts
        print(f"  {sp:<7} {len(gts):>6} GTs" + (f"   ({bad} rows skipped)" if bad else ""))
    if not data:
        print("\n  nothing to do — check DATASET at the top of this file.")
        return
    print()

    main_split = "train" if "train" in data else next(iter(data))
    gts = data[main_split]

    for tax_name, (mode, t1, t2) in TAX.items():
        print("-" * 78)
        print(f"  SPLIT {main_split.upper()}   TAXONOMY {tax_name}")
        print("-" * 78)
        for label, sc in SCENARIOS:
            report(gts, tax_name, mode, t1, t2, TOPK_O2M, sc, label)

    # one2one, at eval geometry only — its budget is 7, and topk2=1 means the
    # prefilter is what feeds the argmax that produces every NMS-free prediction
    print("-" * 78)
    print(f"  ONE2ONE BRANCH (topk={TOPK_O2O}, then topk2=1 argmax)")
    print("-" * 78)
    mode, t1, t2 = TAX["C  COCO area 32/96"]
    report(gts, "C", mode, t1, t2, TOPK_O2O, 1.00, "eval geometry")

    print("=" * 78)
    print("  HOW TO READ IT")
    print("=" * 78)
    print("  'pool<k' is the fraction of GTs holding fewer anchor centres than the")
    print("  budget. For those GTs every candidate is already a positive, so the")
    print("  RANKING cannot change the assignment and tal_beta's selection term is")
    print("  inert. Only its target-magnitude term acts on them.")
    print()
    print("  IF small is NOT binding (>50%):")
    print("    - raising tal_topk cannot help small objects. Confirmed independently")
    print("      by the footprint diagnostic: they take 7.73 against a budget of 10.")
    print("    - the beta plateau has a mechanical explanation, which is a stronger")
    print("      paper claim than the plateau on its own.")
    print("    - LOWERING topk is the only version of this experiment that can bite,")
    print("      and it must be judged on mAP50_small, not mAP50-95.")
    print()
    print("  IF small IS binding (<10%):")
    print("    - the saturation story is wrong; 7.73 < 10 is then the metric>eps")
    print("      prune, not geometry, and the beta plateau needs another explanation.")
    print()
    print("  CROSS-CHECK: 'take~' is the mean positives per GT ignoring the eps prune.")
    print("  Against the footprint diagnostic's stock column (small 7.73 / med 9.87 /")
    print("  large 9.79, taxonomy A, YOLOv12) 'take~' should be an UPPER BOUND. If it")
    print("  comes out BELOW 7.73 the geometry here disagrees with that run and one")
    print("  of the two is wrong — do not proceed to GPU until that is resolved.")
    print()


if __name__ == "__main__":
    main()
