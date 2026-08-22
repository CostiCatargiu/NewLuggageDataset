#!/usr/bin/env python3
r"""
Is tal_topk binding, and for which object sizes?

Candidates in TAL are restricted to anchors whose CENTRE falls inside the GT box
(select_candidates_in_gts). If a GT contains fewer than `topk` anchor centres
across the three levels, the top-k budget is not binding for that GT: every
candidate is taken and changing topk cannot change its assignment.

That decides whether ablating tal_topk is worth GPU time. It needs no model and
no forward pass -- only the label geometry and the anchor grid -- so it runs in
seconds on CPU.

    python probe_topk_binding.py                     # defaults below
    python probe_topk_binding.py --topk 10 --imgsz 640
"""

import argparse
import glob
import os

TOPK_O2M = 10  # E2ELoss: v8DetectionLoss(model, tal_topk=10)
TOPK_O2O = 7   # E2ELoss: v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
STRIDES = (8, 16, 32)
COCO_SMALL = 32.0   # sqrt(area) in px, COCO area<32^2
COCO_MEDIUM = 96.0  # COCO 32^2<=area<96^2


def anchors_inside(w_px, h_px, strides=STRIDES):
    """Anchor centres inside a w x h box, summed over levels.

    Centres sit at (i + 0.5) * stride, so a box of side s spans floor(s/stride)
    or that plus one depending on phase. Averaging over phase gives s/stride,
    which is the expected count and is what matters in aggregate.
    """
    return sum((w_px / s) * (h_px / s) for s in strides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/test/labels")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--topk", type=int, default=TOPK_O2M)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.labels, "*.txt")))
    if not files:
        print(f"no label files under {a.labels}")
        return

    buckets = {"small": [], "medium": [], "large": []}
    for f in files:
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 5:
                    continue
                # YOLO format is normalized cx cy w h; extra columns mean segments
                w, h = float(p[3]) * a.imgsz, float(p[4]) * a.imgsz
                if w <= 0 or h <= 0:
                    continue
                side = (w * h) ** 0.5
                k = "small" if side < COCO_SMALL else "medium" if side < COCO_MEDIUM else "large"
                buckets[k].append(anchors_inside(w, h))

    print()
    print(f"  labels {a.labels}")
    print(f"  imgsz {a.imgsz}   strides {STRIDES}   topk under test {a.topk}")
    print()
    print(f"  {'bucket':<9}{'n':>8}{'mean pool':>11}{'median':>9}{'< topk':>9}{'verdict':>16}")
    print("  " + "-" * 62)
    for k in ("small", "medium", "large"):
        v = sorted(buckets[k])
        if not v:
            continue
        n = len(v)
        mean = sum(v) / n
        med = v[n // 2]
        under = 100.0 * sum(1 for x in v if x < a.topk) / n
        verdict = "NOT binding" if under > 50 else "binding" if under < 10 else "mixed"
        print(f"  {k:<9}{n:>8}{mean:>11.1f}{med:>9.1f}{under:>8.0f}%{verdict:>16}")

    allv = sorted(sum(buckets.values(), []))
    under = 100.0 * sum(1 for x in allv if x < a.topk) / len(allv)
    print("  " + "-" * 62)
    print(f"  {'all':<9}{len(allv):>8}{sum(allv) / len(allv):>11.1f}"
          f"{allv[len(allv) // 2]:>9.1f}{under:>8.0f}%")
    print()
    print("  READ IT")
    print("    'NOT binding' means the GT holds fewer anchor centres than topk, so every")
    print("    candidate is already selected and topk cannot change that bucket. If small")
    print("    is not binding, RAISING tal_topk cannot help small objects at all, and the")
    print("    only lever left on those GTs is how their few candidates are RANKED --")
    print(f"    which is tal_beta. Lowering topk below the small pool would start to bite.")
    print(f"    one2one uses topk={TOPK_O2O}; rerun with --topk {TOPK_O2O} for that branch.")
    print()


if __name__ == "__main__":
    main()
