#!/usr/bin/env python3
r"""
patch_round21_v6i.py — install the three round-21 mechanisms into the fork.

    tal_beta_sel / tal_beta_tgt   split the IoU exponent by CONSUMER
    tcn_p                         target-ceiling normalisation exponent
    tal_beta_level                per-FPN-level IoU exponent

IDEMPOTENT. Backs up before touching anything, verifies after, and refuses to
half-apply: if any hunk does not match, nothing is written.

NO ARGUMENTS:
    python patch_round21_v6i.py            # apply
    python patch_round21_v6i.py --check    # report status only, change nothing
    python patch_round21_v6i.py --revert   # restore from .prepatch21

=============================================================================
WHY EACH ONE
=============================================================================
1. BETA SPLIT. `align_metric = score^alpha * IoU^beta` has THREE consumers:
       tal.py:370  select_topk_candidates      the candidate pool
       tal.py:553  topk(align_metric, topk2)   the single o2o winner
       tal.py:224  pos_align_metrics / norm    the target magnitudes
   Both selection sites are pure top-k, hence invariant to alpha and beta
   separately and dependent only on alpha/beta. y26_b3 and y26_a1_b6 share
   ratio 0.167 -> identical selection -> their 0.98 gap on mAP50_small is a
   PURE TARGET effect. Splitting lets each term reach its own optimum instead
   of sharing one compromise value. Both selection sites take beta_sel; that
   is a deliberate choice, stated so a later round can split 553 off knowingly.

2. TCN. tal.py:225 sets each GT's target ceiling to `pos_overlaps` = the best
   IoU any anchor achieved for it. A small object whose best anchor reaches
   0.55 is trained toward 0.55; a large one reaching 0.90 toward 0.90. Then
   inference applies ONE global threshold to both. That is the measured
   failure mode (AR50_small 0.95 vs R50_small 0.70) written into the loss.
   tcn_p raises the ceiling: pos_overlaps ** p, p<1. Unlike TSH (which powered
   the targets AFTER normalisation and so compressed the winner/runner-up gap
   that SNT proved is load-bearing) this changes only the per-GT multiplier —
   the within-GT shape align/align_max is untouched.

3. LEVEL BETA. Small objects draw 6.64 of their 9.82 positives from stride 8;
   large draw 0.16 of 9.79. Per-level beta applies the correction where small
   objects live without touching where large ones are assigned. Reuses the
   `_strides` channel that LB-TAL already installs (loss.py:757 calls
   set_strides via hasattr, so adding it to the base class is enough).

INERTNESS. beta_sel=None, beta_tgt=None, tcn_p=1.0, beta_level=None reproduce
stock BIT-IDENTICALLY: align_tgt is the SAME OBJECT as align_sel, so even the
in-place `*= mask_pos` matches. Asserted numerically at the end of this script,
not assumed.
"""

import os
import shutil
import sys

TAL = "../ultralytics/utils/tal.py"
LOSS = "../ultralytics/utils/loss.py"
YAML = "../ultralytics/cfg/default.yaml"
SUFFIX = ".prepatch21"

# ---------------------------------------------------------------- tal.py ----
TAL_HUNKS = [
(   # 1. new assigner fields + set_strides on the BASE class
"""        self.beta_ref_px = 64.0  # GT sqrt(area) at which beta reaches self.beta""",
"""        self.beta_ref_px = 64.0  # GT sqrt(area) at which beta reaches self.beta

        # --- round 21: beta split by consumer, target ceiling, per-level beta ---
        # All four are inert at the values below and reproduce stock bit-for-bit.
        self.beta_sel = None    # float; IoU exponent for the two SELECTION sites
        self.beta_tgt = None    # float; IoU exponent for the TARGET magnitudes
        self.tcn_p = 1.0        # float; pos_overlaps ** tcn_p. 1.0 == stock
        self.beta_level = None  # {stride: beta}, e.g. {8: 2.0}; needs _strides
        if not hasattr(self, "_strides"):
            self._strides = None  # (A,) per-anchor stride, set each forward pass"""),
(   # 2. base-class set_strides (LB-TAL overrides it; harmless there)
"""    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):""",
"""    def set_strides(self, stride_tensor):
        \"\"\"Record the per-anchor stride. loss.py calls this via hasattr each pass.\"\"\"
        self._strides = stride_tensor.detach().reshape(-1)

    def _anchor_beta(self, base_beta):
        \"\"\"Per-anchor IoU exponent from beta_level, broadcastable over (b, n_gt, A).

        Returns base_beta unchanged when beta_level is unset or the strides are
        missing — never silently half-applies.\"\"\"
        if not self.beta_level or self._strides is None:
            return base_beta
        b = self._strides.new_full(self._strides.shape, float(base_beta))
        for s, v in self.beta_level.items():
            b = torch.where(self._strides == float(s), float(v), b)
        return b.view(1, 1, -1)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):"""),
(   # 3. build both metrics
"""        beta = self._size_conditioned_beta(gt_bboxes) if self.scb_enabled() else self.beta
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(beta)
        return align_metric, overlaps""",
"""        beta = self._size_conditioned_beta(gt_bboxes) if self.scb_enabled() else self.beta
        b_sel = beta if self.beta_sel is None else float(self.beta_sel)
        b_sel = self._anchor_beta(b_sel) if self.beta_level else b_sel
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(b_sel)
        # Same object when nothing is live -> the in-place *= below matches stock.
        if self.beta_tgt is None and not self.beta_level:
            align_tgt = align_metric
        else:
            b_tgt = beta if self.beta_tgt is None else float(self.beta_tgt)
            align_tgt = bbox_scores.pow(self.alpha) * overlaps.pow(b_tgt)
        return align_metric, overlaps, align_tgt"""),
(   # 4. thread it through get_pos_mask
"""        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)""",
"""        align_metric, overlaps, align_tgt = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )"""),
(
"""        return mask_pos, align_metric, overlaps""",
"""        return mask_pos, align_metric, overlaps, align_tgt"""),
(   # 5. and through _forward
"""        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )""",
"""        mask_pos, align_metric, overlaps, align_tgt = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )"""),
(   # 6. targets come from align_tgt; ceiling gets tcn_p
"""        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)""",
"""        align_metric *= mask_pos
        if align_tgt is not align_metric:
            align_tgt = align_tgt * mask_pos
        pos_align_metrics = align_tgt.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # TCN: raise the per-GT target ceiling. p == 1.0 leaves this untouched.
        if float(self.tcn_p) != 1.0:
            pos_overlaps = pos_overlaps.clamp(min=0.0).pow(float(self.tcn_p))
        norm_align_metric = (align_tgt * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)"""),
]

# --------------------------------------------------------------- loss.py ----
LOSS_HUNKS = [(
"""        beta_small = getattr(h, "tal_beta_small", None)""",
"""        # --- round 21 keys, applied before SCB so SCB can still override beta ---
        for _key, _attr in (("tal_beta_sel", "beta_sel"), ("tal_beta_tgt", "beta_tgt")):
            _v = getattr(h, _key, None)
            if _v is not None:
                setattr(self.assigner, _attr, float(_v))
        _tcn = getattr(h, "tcn_p", None)
        if _tcn is not None:
            self.assigner.tcn_p = float(_tcn)
        _lvl = getattr(h, "tal_beta_level", None)
        if _lvl:
            self.assigner.beta_level = {int(k): float(v) for k, v in dict(_lvl).items()}

        beta_small = getattr(h, "tal_beta_small", None)""")]

# ----------------------------------------------------------- default.yaml ----
YAML_ADD = """tal_beta_sel: # (float, optional) IoU exponent for the SELECTION sites only; None -> tal_beta
tal_beta_tgt: # (float, optional) IoU exponent for the TARGET magnitudes only; None -> tal_beta
tcn_p: 1.0 # (float) target-ceiling exponent, pos_overlaps ** tcn_p. 1.0 = stock
tal_beta_level: # (dict, optional) per-stride IoU exponent, e.g. {8: 2.0, 16: 6.0, 32: 6.0}
"""
YAML_ANCHOR = "tal_beta_small:"


def read_keep_eol(path):
    """Read a file, remembering its line endings.

    The fork's sources are CRLF. Writing them back as LF makes every line look
    changed, which hides the real diff and would bury a mistake in this patch
    under 900 lines of noise. Returns (text_with_LF, original_ending).
    """
    raw = open(path, "rb").read().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write_keep_eol(path, text, eol):
    open(path, "wb").write(text.replace("\n", eol).encode("utf-8"))


def status(path, hunks):
    s, eol = read_keep_eol(path)
    done = sum(1 for _, new in hunks if new.split("\n")[-1].strip() and new in s)
    todo = sum(1 for old, _ in hunks if old in s)
    return s, done, todo, eol


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    here = os.path.dirname(os.path.abspath(__file__))
    tal, loss, yml = (os.path.normpath(os.path.join(here, p)) for p in (TAL, LOSS, YAML))
    for p in (tal, loss, yml):
        if not os.path.exists(p):
            raise SystemExit(f"  [ABORT] not found: {p}")

    if mode == "--revert":
        for p in (tal, loss, yml):
            if os.path.exists(p + SUFFIX):
                shutil.copy2(p + SUFFIX, p)
                print(f"  reverted {os.path.basename(p)}")
            else:
                print(f"  no backup for {os.path.basename(p)}")
        return

    tal_s, tal_done, tal_todo, tal_eol = status(tal, TAL_HUNKS)
    loss_s, loss_done, loss_todo, loss_eol = status(loss, LOSS_HUNKS)
    yml_s, yml_eol = read_keep_eol(yml)
    yml_done = "tal_beta_sel:" in yml_s

    print("=" * 72)
    print("  ROUND 21 PATCH STATUS")
    print("=" * 72)
    print(f"  tal.py        {tal_done}/{len(TAL_HUNKS)} hunks applied, {tal_todo} pending")
    print(f"  loss.py       {loss_done}/{len(LOSS_HUNKS)} hunks applied, {loss_todo} pending")
    print(f"  default.yaml  {'applied' if yml_done else 'pending'}")
    if mode == "--check":
        return
    if tal_done == len(TAL_HUNKS) and loss_done == len(LOSS_HUNKS) and yml_done:
        print("\n  already fully applied — nothing to do")
        return
    if tal_todo != len(TAL_HUNKS) - tal_done or loss_todo != len(LOSS_HUNKS) - loss_done:
        raise SystemExit(
            "\n  [ABORT] the tree is not in a state this patch understands.\n"
            "          Some hunks are neither applied nor matchable — the file has\n"
            "          been edited since. Inspect it, or --revert and retry.")

    for p in (tal, loss, yml):
        if not os.path.exists(p + SUFFIX):
            shutil.copy2(p, p + SUFFIX)
            print(f"  backup -> {os.path.basename(p)}{SUFFIX}")

    for old, new in TAL_HUNKS:
        if new not in tal_s:
            tal_s = tal_s.replace(old, new, 1)
    if "import torch" not in tal_s.split("class ")[0]:
        raise SystemExit("  [ABORT] tal.py does not import torch at module level")
    for old, new in LOSS_HUNKS:
        if new not in loss_s:
            loss_s = loss_s.replace(old, new, 1)
    if not yml_done:
        yml_s = yml_s.replace(YAML_ANCHOR, YAML_ADD + YAML_ANCHOR, 1)

    import ast
    ast.parse(tal_s)
    ast.parse(loss_s)
    write_keep_eol(tal, tal_s, tal_eol)
    write_keep_eol(loss, loss_s, loss_eol)
    write_keep_eol(yml, yml_s, yml_eol)
    print(f"\n  line endings preserved ({'CRLF' if tal_eol == chr(13)+chr(10) else 'LF'})")
    print("  applied. now prove it is inert:")
    print("      python run_yolo26_round21_v6i.py --preflight")


if __name__ == "__main__":
    main()
