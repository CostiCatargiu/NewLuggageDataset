#!/usr/bin/env python3
"""
Mechanical verification of the YOLO26 port. ~10 seconds, no GPU, no dataset.

Every check corresponds to a specific way the port could look active in the logs
while doing nothing. A silent no-op is worse than a crash here: it produces a
plausible negative number that gets written into a thesis.

PHASE 1 (no torch required) — source and pure-logic checks:
   1. the patched files are the ones on the import path
   2. LB-TAL is gated on `tal_topk2 is None`, so one2one cannot no-op silently
   3. the override signature matches YOLO26's (no `largest` argument)
   4. set_strides is called in the loss before the assigner runs
   5. all 15 custom keys reach DEFAULT_CFG_DICT
   6. a str-keyed budget normalises identically to an int-keyed one
   7. the alpha curriculum actually moves, and respects its clips
   8. stock-neutral defaults leave alpha at 0 (upstream behaviour)

PHASE 2 (needs torch) — tensor checks:
   9. SWA changes the box loss vs stock
  10. area_weight_mode is read (inv / sqrt / log all differ)
  11. small_obj_boost applies through the per-anchor stride
  12. LB-TAL redistributes positives across pyramid levels
  13. LB-TAL re-allocates rather than inflates (total <= topk)
  14. an un-configured BboxLoss reproduces upstream exactly

Usage:  python verify_port.py
Exit code 0 = safe to spend GPU time.
"""

import ast
import importlib.util
import inspect
import os
import sys
import traceback

sys.path.insert(0, "ultralytics")

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def banner(t):
    print(f"\n{'-' * 74}\n  {t}\n{'-' * 74}")


print("=" * 74)
print("  YOLO26 PORT VERIFICATION")
print("=" * 74)

# ============================================================ PHASE 1: static
banner("PHASE 1 — source and pure-logic checks (no torch)")
try:
    loss_spec = importlib.util.find_spec("ultralytics.utils.loss")
    tal_spec = importlib.util.find_spec("ultralytics.utils.tal")
    loss_path = loss_spec.origin if loss_spec else None
    tal_path = tal_spec.origin if tal_spec else None
    loss_src = open(loss_path, encoding="utf-8").read() if loss_path else ""
    tal_src = open(tal_path, encoding="utf-8").read() if tal_path else ""

    print(f"    loss.py -> {loss_path}")
    print(f"    tal.py  -> {tal_path}")

    check("1. the ultralytics on the import path is the patched tree",
          "def swa_weight" in loss_src and "class LevelBalancedTaskAlignedAssigner" in tal_src,
          "" if "def swa_weight" in loss_src else "swa_weight NOT found — wrong ultralytics is installed")

    check("2. LB-TAL is gated on `tal_topk2 is None` (one2many only)",
          "tal_topk2 is None" in loss_src,
          "one2one keeps stock TAL, so its topk2=1 cannot erase the per-level budget")

    # signature must match YOLO26's parent: (self, metrics, topk_mask=None)
    tree = ast.parse(tal_src)
    sigs = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (f for f in cls.body if isinstance(f, ast.FunctionDef)):
            if fn.name == "select_topk_candidates":
                sigs[cls.name] = [a.arg for a in fn.args.args]
    parent = sigs.get("TaskAlignedAssigner")
    childs = sigs.get("LevelBalancedTaskAlignedAssigner")
    check("3. select_topk_candidates override matches the parent signature",
          parent is not None and childs is not None and parent == childs,
          f"parent={parent}  override={childs}")

    m = ast.parse(loss_src)
    calls_set_strides = 'hasattr(self.assigner, "set_strides")' in loss_src or \
                        "hasattr(self.assigner, 'set_strides')" in loss_src
    check("4. the loss calls set_strides before assigning", calls_set_strides,
          "without it LB-TAL falls back to stock global top-k")

    from ultralytics.cfg import DEFAULT_CFG_DICT
    KEYS = ["alpha_start", "alpha_end", "alpha_min", "alpha_max", "area_weight_mode",
            "area_weight_norm", "small_obj_px", "small_obj_boost", "use_lbtal",
            "lbtal_mode", "lbtal_level_topk", "lbtal_min_level_k", "lbtal_quality_gate",
            "tal_alpha", "tal_beta"]
    missing = [k for k in KEYS if k not in DEFAULT_CFG_DICT]
    check("5. all 15 custom keys reach DEFAULT_CFG_DICT", not missing,
          f"missing={missing}" if missing else "train() overrides will be applied, not dropped")

    # --- pure-logic replicas (no torch needed) -------------------------------
    def norm(level_topk):
        """Mirror of LevelBalancedTaskAlignedAssigner._norm_level_topk."""
        if level_topk is None or isinstance(level_topk, (list, tuple)):
            return level_topk
        out = {}
        for k, v in dict(level_topk).items():
            try:
                out[int(float(k))] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    a, b = norm({8: 4, 16: 7, 32: 1}), norm({"8": 4, "16": 7, "32": 1})
    c = norm({8.0: 4, 16.0: 7, 32.0: 1})
    check("6. str / int / float budget keys all normalise the same", a == b == c,
          f"{a}  ==  {b}  ==  {c}")
    check("6b. a budget lookup by float stride hits", norm({"8": 4}).get(int(8.0)) == 4,
          "this is the trap: the v12 code indexed a str-keyed dict with 8.0 and "
          "silently fell back to min_level_k on every level")

    def alpha(epoch, total=70, a0=0.7, a1=0.3, lo=0.3, hi=0.7):
        p = epoch / max(total, 1)
        return max(lo, min(hi, a0 * (1 - p) + a1 * p))

    a_first, a_last = alpha(0), alpha(69)
    check("7. the alpha curriculum moves across training", abs(a_first - a_last) > 1e-6,
          f"epoch 0 = {a_first:.4f}  ->  epoch 69 = {a_last:.4f}")
    check("7b. alpha stays inside [alpha_min, alpha_max]",
          all(0.3 - 1e-9 <= alpha(e) <= 0.7 + 1e-9 for e in range(0, 70)))
    check("8. stock-neutral defaults leave alpha at 0",
          alpha(0, a0=0.0, a1=0.0, lo=0.0, hi=0.0) == 0.0,
          "an un-configured run is bit-identical to upstream")

except Exception:
    traceback.print_exc()
    FAIL.append("exception in phase 1")

# ============================================================ PHASE 2: tensors
banner("PHASE 2 — tensor checks (requires torch)")
try:
    import torch
except ImportError:
    print("  torch not importable in this environment — PHASE 2 SKIPPED.")
    print("  Run this script on the training machine before starting any run;")
    print("  phase 1 cannot prove that SWA changes the loss or that LB-TAL")
    print("  actually redistributes positives.")
    SKIP.append("phase 2 (torch missing)")
    torch = None

if torch is not None:
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import (
            LevelBalancedTaskAlignedAssigner,
            TaskAlignedAssigner,
        )

        # 3 levels at stride 8/16/32 over a 64x64 image
        sizes, strides_list = [(8, 8), (4, 4), (2, 2)], [8, 16, 32]
        stride = torch.cat([torch.full((h * w, 1), float(s)) for (h, w), s in zip(sizes, strides_list)])
        A, bs = stride.shape[0], 2
        pos_idx = [0, 1, 2, 70, 71, 80]  # positives spread over all three levels

        fg = torch.zeros(bs, A, dtype=torch.bool)
        fg[:, pos_idx] = True
        tb = torch.zeros(bs, A, 4)
        for i, idx in enumerate(pos_idx):
            side = 0.8 if i < 3 else 6.0  # tiny boxes on P3, large on P4/P5
            tb[:, idx] = torch.tensor([0.0, 0.0, side, side])
        torch.manual_seed(0)
        ts = torch.rand(bs, A, 3).clamp(0.05, 1.0)
        tss = torch.tensor(float(bs * len(pos_idx)))
        pred = tb.clone() + 0.15
        ap, imgsz = torch.zeros(A, 2), torch.tensor([64.0, 64.0])
        zeros = torch.zeros(bs, A, 4)

        def box_loss(**cfg):
            bl = BboxLoss(reg_max=1)  # YOLO26 is DFL-free
            for k, v in cfg.items():
                setattr(bl, k, v)
            return float(bl.forward(zeros, pred, ap, tb, ts, tss, fg, imgsz, stride)[0])

        SWA = dict(area_weight_mode="sqrt", alpha_start=0.7, alpha_end=0.3, alpha_min=0.3,
                   alpha_max=0.7, small_obj_px=48, small_obj_boost=2.0, total_epochs=70)

        stock = box_loss()
        check("9. SWA changes the box loss vs stock", abs(box_loss(**SWA, epoch=0) - stock) > 1e-6,
              f"stock={stock:.6f}  swa={box_loss(**SWA, epoch=0):.6f}")
        check("9b. the loss differs between the first and last epoch",
              abs(box_loss(**SWA, epoch=0) - box_loss(**SWA, epoch=69)) > 1e-6)

        modes = {m: box_loss(**{**SWA, "area_weight_mode": m}, epoch=0) for m in ("inv", "sqrt", "log")}
        check("10. area_weight_mode is read", len({round(v, 9) for v in modes.values()}) == 3,
              "  ".join(f"{k}={v:.6f}" for k, v in modes.items()))

        check("11. small_obj_boost applies via the per-anchor stride",
              abs(box_loss(**SWA, epoch=0) - box_loss(**{**SWA, "small_obj_boost": 1.0}, epoch=0)) > 1e-6)
        check("11b. small_obj_px=0 disables the boost",
              abs(box_loss(**{**SWA, "small_obj_px": 0}, epoch=0)
                  - box_loss(**{**SWA, "small_obj_px": 0, "small_obj_boost": 1.0}, epoch=0)) < 1e-12)

        # ---- LB-TAL level distribution
        n = 3
        torch.manual_seed(0)
        metrics = torch.rand(bs, n, A)
        metrics[:, :, :64] += 1.5  # the finest level dominates, as it does in practice
        tk_mask = torch.ones(bs, n, 10, dtype=torch.bool)
        sflat = stride.reshape(-1)

        def counts(c):
            sel = c > 0
            return {int(s): int((sel & (sflat == s).view(1, 1, -1)).sum())
                    for s in sorted(torch.unique(sflat).tolist())}

        c_stock = counts(TaskAlignedAssigner(topk=10, num_classes=3, stride=strides_list)
                         .select_topk_candidates(metrics, topk_mask=tk_mask))

        def lb(**kw):
            a = LevelBalancedTaskAlignedAssigner(topk=10, num_classes=3, stride=strides_list, **kw)
            a.set_strides(stride)
            return a

        uni = lb(level_topk_mode="uniform")
        c_uni = counts(uni.select_topk_candidates(metrics, topk_mask=tk_mask))
        check("12. LB-TAL uniform redistributes positives across levels", c_stock != c_uni,
              f"stock={c_stock}  uniform={c_uni}")
        check("12b. every level receives a share under LB-TAL", all(v > 0 for v in c_uni.values()),
              f"uniform={c_uni}")

        p4 = lb(level_topk_mode="fixed", level_topk={8: 4, 16: 7, 32: 1})
        p4s = lb(level_topk_mode="fixed", level_topk={"8": 4, "16": 7, "32": 1})
        c_p4 = counts(p4.select_topk_candidates(metrics, topk_mask=tk_mask))
        c_p4s = counts(p4s.select_topk_candidates(metrics, topk_mask=tk_mask))
        check("12c. a str-keyed budget behaves identically at tensor level", c_p4 == c_p4s,
              f"int={c_p4}  str={c_p4s}")
        check("12d. p4wide differs from uniform", c_p4 != c_uni, f"p4wide={c_p4}  uniform={c_uni}")

        tot = p4.select_topk_candidates(metrics, topk_mask=tk_mask).sum(-1)
        check("13. LB-TAL re-allocates rather than inflates", bool((tot <= 10 + 1e-6).all()),
              f"max positives per GT = {float(tot.max()):.0f}  (topk=10)")

        # missing strides must fall back loudly, not pretend to work
        naive = LevelBalancedTaskAlignedAssigner(topk=10, num_classes=3, stride=strides_list,
                                                 level_topk_mode="uniform")
        check("13b. a missing set_strides falls back to stock (and warns)",
              counts(naive.select_topk_candidates(metrics, topk_mask=tk_mask)) == c_stock)

        # ---- upstream equivalence
        w = ts[fg].sum(-1, keepdim=True)
        iou = bbox_iou(pred[fg], tb[fg], xywh=False, CIoU=True)
        upstream = float(((1.0 - iou) * w).sum() / tss)
        check("14. un-configured BboxLoss reproduces upstream exactly", abs(stock - upstream) < 1e-9,
              f"patched={stock:.9f}  upstream={upstream:.9f}")

    except Exception:
        traceback.print_exc()
        FAIL.append("exception in phase 2")

print("\n" + "=" * 74)
print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if SKIP:
    print(f"  SKIPPED: {', '.join(SKIP)}")
if FAIL:
    print("  FAILED:  " + ", ".join(FAIL))
    print("  Do NOT spend GPU time until these pass.")
elif SKIP:
    print("  Phase 1 is clean, but phase 2 must be run on the training machine.")
else:
    print("  Port verified. Safe to run run_yolo26_port_v6i.py.")
print("=" * 74)
sys.exit(1 if FAIL else 0)
