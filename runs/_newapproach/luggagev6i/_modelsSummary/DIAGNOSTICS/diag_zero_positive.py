#!/usr/bin/env python3
r"""
ZERO-POSITIVE DIAGNOSTIC — does the assigner silently drop ground truths?
=========================================================================

Free gate for the "rank-preserving assignment" mechanism. No training, no
checkpoint written. Runs a handful of real training iterations and counts, per
branch and per object size, how often a GT ends up with NO positive anchor.


THE CLAIM BEING TESTED
----------------------
`iou_calculation` ends in `.clamp_(0)` and CIoU goes NEGATIVE for poorly-placed
boxes, so `overlaps` can be exactly 0. With

    align_metric = bbox_scores ** alpha * overlaps ** beta        (alpha .5, beta 6)

a single zero overlap makes the whole metric exactly 0. When EVERY candidate of a
GT is zero, `torch.topk` has nothing to rank and breaks ties by INDEX ORDER,
returning anchors 0..k-1 — the top-left corner of the P3 grid. Then

    mask_pos = mask_topk * mask_in_gts * mask_gt

intersects that with "anchor centre inside the GT", which is EMPTY for any GT not
at the image corner. The GT contributes nothing that iteration: no box loss, no
positive target. In one2one `select_highest_overlaps` repeats it — topk2=1 picks
index 0 and `mask_pos *= topk_idx` zeroes the row.

Nothing in the campaign covers this. LB-TAL gates on `tk_metrics > eps`, which
fails identically when every metric is 0, and it only runs on one2many.

If the effect is real it should be SIZE-DEPENDENT: small boxes have few in-GT
anchors and noisier CIoU, so they lose all of them far more often. That is also
the population NWD helped most (+0.89 small mAP50, best in the campaign) — NWD's
exp(-d/c) is strictly positive and therefore cannot collapse, which is the same
fix arrived at from the other direction.


HOW TO READ TABLE 1 — assignment health
---------------------------------------
    %cand IoU<=0     of the anchors inside a GT, how many scored CIoU <= 0
    %GT metric==0    GTs where EVERY candidate scored 0 -> topk ranks by index
    %GT 0 positives  GTs that ended the assignment with nothing
    mean pos/GT      surviving positives per GT

    small bucket >> large bucket on '%GT 0 positives'
        -> the degeneracy is real and size-dependent. Build the mechanism.
    all buckets ~0
        -> dead. Close the axis, cost 0 GPU-h.
    high but FLAT across sizes
        -> real but not a small-object story; the mechanism would not explain NWD.


HOW TO READ TABLE 2 — box quality where it counts
-------------------------------------------------
Table 1 measures CANDIDATE anchors. Only the anchor that WINS contributes a box
to the output, so candidate statistics can look bad while the prediction that
ships is fine. Table 2 measures the anchors that were actually selected.

    CIoU@sel     mean overlap of the kept anchors' boxes (one2one: exactly one,
                 so this IS the box that reaches inference)
    CIoU@best    best overlap among the kept anchors
    CIoU@avail   best overlap ANY candidate could have offered
    left on table  CIoU@avail - CIoU@best; how much the selection gave up

This exists to test one specific claim: one2one supervises 1 anchor per GT while
one2many supervises ~9.8, and one2one's CANDIDATE boxes are measurably worse
(%cand IoU<=0 of 59.3 vs 51.8 on large). If that deficit reaches the output, a
mechanism that widens one2one's REGRESSION assignment — while leaving its
 classification targets alone, so duplicate suppression is untouched — has a
 target.

    one2one CIoU@sel ~= one2many CIoU@sel
        -> the branch is fine where it counts. Extra regression supervision has
           nothing to fix. Mechanism DEAD, cost 0 GPU-h.
    one2one CIoU@sel clearly lower, and the gap WIDENS with size
        -> the deficit reaches the output. Measured target, build it.

NOTE: `overlaps` is CIoU already clamped at 0 by iou_calculation, not raw IoU.
Absolute values are therefore pessimistic; the BRANCH COMPARISON is the signal.

Run it twice — the answer is expected to differ:
    WEIGHTS = "yolo26s.pt"          start of training (COCO transfer, epoch 0)
    WEIGHTS = ".../best.pt"         a converged run, to see whether it persists


Usage:
    python diag_zero_positive.py
    python diag_zero_positive.py /path/to/best.pt
"""

import sys
from collections import defaultdict

import torch

from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
WEIGHTS = "yolo26s.pt"
BATCH = 82  # match the loss campaign so anchor/GT statistics are comparable
IMG_SIZE = 640
FRACTION = 0.10  # of the train split; ~11 iterations at b82. Raise to 0.30 when MOSAIC=0.0,
#                  which drops GTs per sample ~4x (18.7 -> 4.6).
SEED = 0

# Mosaic tiles 4 images per sample, shrinking objects and shifting the size histogram.
# The real runs use close_mosaic=10, so the final 10 epochs — and inference — see none.
# Set 0.0 to measure the regime the converged model actually ends in.
MOSAIC = 1.0

# The probe TRAINS: 12 AdamW steps at lr 1.4e-3 moved y26_identity from ~0.55 to 0.371
# mAP50-95. Set 0.0 for a true snapshot of the checkpoint (weights cannot move at lr=0;
# an explicit optimizer is required because optimizer='auto' ignores lr0).
LR0 = None  # None = ultralytics auto, 0.0 = frozen

# Max-side edges in px. Same convention as diag_miss_vs_score.py so the buckets
# line up with the existing diagnostics (NOT the COCO area buckets in the JSONs).
SMALL_MAX = 48.0
MEDIUM_MAX = 96.0
# =============================================================================

STATS = defaultdict(lambda: defaultdict(float))
_ORIG_GET_POS_MASK = TaskAlignedAssigner.get_pos_mask
_ORIG_SELECT_HIGHEST = TaskAlignedAssigner.select_highest_overlaps


def _bucket(side_px: torch.Tensor) -> torch.Tensor:
    """0 = small, 1 = medium, 2 = large, by max box side in pixels."""
    return torch.bucketize(side_px, torch.tensor([SMALL_MAX, MEDIUM_MAX], device=side_px.device))


def _branch(assigner) -> str:
    """one2one is the branch with a second top-k (topk2=1); one2many has topk2 == topk."""
    return "one2one" if assigner.topk2 != assigner.topk else "one2many"


def patched_get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
    """Record candidate-level statistics, then defer to the stock implementation."""
    mask_pos, align_metric, overlaps = _ORIG_GET_POS_MASK(
        self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
    )

    # gt_bboxes are in PIXEL units here (the caller passes anc_points * stride).
    wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp(min=0)
    side = wh.amax(-1)  # (b, n) max side, px
    valid = mask_gt[..., 0].bool()  # (b, n) real GTs, not padding

    # Recomputed rather than returned: get_pos_mask keeps mask_in_gts internal.
    in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt).bool() & valid.unsqueeze(-1)

    n_cand = in_gts.sum(-1)  # (b, n) anchors whose centre falls inside the GT
    n_le0 = (in_gts & (overlaps <= 0)).sum(-1)  # of those, how many scored CIoU <= 0
    metric_zero = align_metric.amax(-1) <= 0  # every candidate scored exactly 0
    cand_zero = mask_pos.sum(-1) <= 0  # nothing survived in-gts AND top-k

    # Stash for select_highest_overlaps, which never receives gt_bboxes.
    self._dbg_side, self._dbg_valid = side, valid

    b = _branch(self)
    buckets = _bucket(side)
    for k in range(3):
        m = valid & (buckets == k)
        if not bool(m.any()):
            continue
        s = STATS[(b, k)]
        s["n_gt"] += float(m.sum())
        s["n_cand"] += float(n_cand[m].sum())
        s["n_le0"] += float(n_le0[m].sum())
        s["gt_metric_zero"] += float((metric_zero & m).sum())
        s["gt_cand_zero"] += float((cand_zero & m).sum())
    return mask_pos, align_metric, overlaps


def patched_select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
    """Record final positive counts, and the box quality AT the anchors that were selected."""
    cand = mask_pos.clone()  # the stock implementation mutates mask_pos in place via `mask_pos *= topk_idx`
    target_gt_idx, fg_mask, mask_pos = _ORIG_SELECT_HIGHEST(self, mask_pos, overlaps, n_max_boxes, align_metric)

    side, valid = getattr(self, "_dbg_side", None), getattr(self, "_dbg_valid", None)
    if side is not None:
        n_pos = mask_pos.sum(-1)  # (b, n) positives finally held by each GT
        sel = overlaps * mask_pos
        sel_mean = sel.sum(-1) / n_pos.clamp(min=1)
        sel_best = sel.amax(-1)
        avail_best = (overlaps * cand).amax(-1)  # the best any candidate could have offered
        kept = valid & (n_pos > 0)  # quality is undefined for GTs that kept nothing

        b = _branch(self)
        buckets = _bucket(side)
        for k in range(3):
            m = valid & (buckets == k)
            if not bool(m.any()):
                continue
            s = STATS[(b, k)]
            s["n_pos_final"] += float(n_pos[m].sum())
            s["gt_zero_final"] += float(((n_pos <= 0) & m).sum())
            mk = kept & (buckets == k)
            if bool(mk.any()):
                s["n_gt_kept"] += float(mk.sum())
                s["sel_mean"] += float(sel_mean[mk].sum())
                s["sel_best"] += float(sel_best[mk].sum())
                s["avail_best"] += float(avail_best[mk].sum())
    return target_gt_idx, fg_mask, mask_pos


def report() -> None:
    """Print the per-branch, per-size table and a verdict."""
    names = ["small", "medium", "large"]
    print("\n" + "=" * 78)
    print("ZERO-POSITIVE DIAGNOSTIC")
    print(f"weights={WEIGHTS}  batch={BATCH}  imgsz={IMG_SIZE}  fraction={FRACTION}")
    print(f"size edges (MAX SIDE, px): small<{SMALL_MAX:.0f}  medium<={MEDIUM_MAX:.0f}  large>")
    print("=" * 78)

    verdict, quality = {}, {}
    for branch in ("one2many", "one2one"):
        if not any((branch, k) in STATS for k in range(3)):
            continue
        tag = "topk2=1, produces every prediction" if branch == "one2one" else "topk2 unset, auxiliary"
        print(f"\nBRANCH {branch}  ({tag})")
        print(f"{'size':<8}{'n_GT':>10}{'cand/GT':>9}{'%cand IoU<=0':>14}"
              f"{'%GT metric==0':>15}{'%GT 0 pos':>11}{'mean pos/GT':>13}")
        print("-" * 78)
        for k in range(3):
            s = STATS.get((branch, k))
            if not s or not s["n_gt"]:
                continue
            n = s["n_gt"]
            pct_zero = 100.0 * s["gt_zero_final"] / n
            print(f"{names[k]:<8}{int(n):>10}{s['n_cand'] / n:>9.1f}"
                  f"{100.0 * s['n_le0'] / max(s['n_cand'], 1):>14.1f}"
                  f"{100.0 * s['gt_metric_zero'] / n:>15.2f}"
                  f"{pct_zero:>11.2f}{s['n_pos_final'] / n:>13.2f}")
            verdict[(branch, k)] = pct_zero

        print(f"\n  box quality AT the selected anchors (clamped CIoU, not raw IoU)")
        print(f"  {'size':<8}{'n_GT':>10}{'CIoU@sel':>11}{'CIoU@best':>11}"
              f"{'CIoU@avail':>12}{'left on table':>15}")
        print("  " + "-" * 67)
        for k in range(3):
            s = STATS.get((branch, k))
            if not s or not s.get("n_gt_kept"):
                continue
            n = s["n_gt_kept"]
            sm, sb, ab = s["sel_mean"] / n, s["sel_best"] / n, s["avail_best"] / n
            print(f"  {names[k]:<8}{int(n):>10}{sm:>11.3f}{sb:>11.3f}{ab:>12.3f}{ab - sb:>15.3f}")
            quality[(branch, k)] = (sm, ab)

    print("\n" + "-" * 78)
    print("VERDICT 1 — dropped ground truths")
    small = max(verdict.get(("one2many", 0), 0.0), verdict.get(("one2one", 0), 0.0))
    large = max(verdict.get(("one2many", 2), 0.0), verdict.get(("one2one", 2), 0.0))
    if small < 0.5 and large < 0.5:
        print("  Every bucket under 0.5% — the degeneracy does not occur in practice.")
        print("  -> Rank-preserving assignment is DEAD. Close the axis, cost 0 GPU-h.")
    elif small > 2 * max(large, 0.1):
        print(f"  small {small:.2f}% vs large {large:.2f}% — size-dependent, as predicted.")
        print("  -> Real. GTs are being dropped, and it is the population NWD helped.")
    else:
        print(f"  small {small:.2f}% vs large {large:.2f}% — present but FLAT across sizes.")
        print("  -> Real but not a small-object story. It would not explain NWD's +0.89.")

    # Compare the branches on CIoU@avail, not CIoU@sel: both heads see the SAME candidate
    # anchors, so @avail is the best box each head can produce on identical input. @sel is
    # a max-over-1 for one2one against a mean-over-9.8 for one2many and is not comparable.
    print("\nVERDICT 2 — is one2one's box head actually worse, on identical candidates?")
    deltas = {}
    for k in range(3):
        a, o = quality.get(("one2many", k)), quality.get(("one2one", k))
        if a is None or o is None:
            continue
        deltas[k] = o[1] - a[1]
        print(f"  {names[k]:<8}one2many@avail {a[1]:.3f}   one2one@avail {o[1]:.3f}   "
              f"delta {o[1] - a[1]:+.3f}   (one2one ships {o[0]:.3f})")
    if not deltas:
        print("  no paired data.")
    elif all(d > -0.005 for d in deltas.values()):
        print("  one2one matches or beats one2many on identical candidates, despite training")
        print("  1 anchor per GT against ~9.8. Extra regression supervision has nothing to")
        print("  add. -> o2o_reg_k DEAD, 0 GPU-h.")
    elif deltas.get(2, 0.0) < deltas.get(0, 0.0) - 0.01:
        print("  one2one is worse AND the gap widens with size — matches the 1-vs-9.8")
        print("  supervision ratio. -> Measured target. Build o2o_reg_k.")
    else:
        print("  one2one is worse but the gap does not widen with size, so the")
        print("  supervision-ratio account does not explain it. Find the cause first.")
    print("=" * 78 + "\n")


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else WEIGHTS
    globals()["WEIGHTS"] = weights

    TaskAlignedAssigner.get_pos_mask = patched_get_pos_mask
    TaskAlignedAssigner.select_highest_overlaps = patched_select_highest_overlaps

    print(f"[patch] instrumented TaskAlignedAssigner (both branches)\n[run]   {weights} on {DATA_YAML}")
    frozen = dict(optimizer="SGD", lr0=0.0, momentum=0.0, warmup_epochs=0.0, warmup_bias_lr=0.0) if LR0 == 0.0 else {}
    YOLO(weights).train(
        data=DATA_YAML,
        epochs=1,
        fraction=FRACTION,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        mosaic=MOSAIC,
        device=0 if torch.cuda.is_available() else "cpu",
        val=False,
        save=False,
        plots=False,
        project="diag_zero_positive",
        name="probe",
        exist_ok=True,
        verbose=False,
        **frozen,
    )

    if not STATS:
        print("\n[ABORT] no statistics collected — the patch never fired. Check that this")
        print("        ultralytics is the patched MODEL_v26 copy and not a stock install.")
        return
    report()


if __name__ == "__main__":
    main()
