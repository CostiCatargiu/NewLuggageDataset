#!/usr/bin/env python3
r"""
diag_threshold_sweep.py — does precision survive when you lower the threshold?

=============================================================================
THE QUESTION, AND WHY IT DECIDES WHAT IS LEFT TO DO
=============================================================================
diag_miss_vs_score.py established that the miss column is SCORING-limited, not
proposal-limited, for every class, size and model:

    y26_identity   class      now   ceiling  headroom   true_miss
                   bag       68.3      95.1     +26.8        4.9
                   backpack  78.0      95.9     +17.9        4.1
                   trolley   84.7      97.3     +12.7        2.7

The boxes exist. Correctly classified, IoU >= 0.5, sitting below conf 0.25.
Only ~4% of ground truths are genuinely undetectable.

But `recovered` only says a correct box EXISTS below the threshold. It says
nothing about what ELSE comes up with it. Lowering the threshold also admits
false positives, and the whole question is the ratio.

    precision holds as conf drops  -> the model RANKS well and 0.25 is simply the
        wrong operating point. A per-class threshold is a free win: bag goes
        68.3 -> ~90 with zero retraining. Report the tuned operating point.

    precision collapses            -> the correct box exists but is ranked BELOW
        junk. That is a classification/ranking failure, and it is the first
        well-evidenced loss target this campaign has produced — a measurement,
        not a guess.

=============================================================================
TUNE ON VAL, REPORT ON TEST — THIS IS NOT OPTIONAL
=============================================================================
Picking a threshold that maximises F1 on the test split and then reporting that
same F1 on test is leakage, and a reviewer will catch it immediately. The whole
point of a threshold is that it is a FREE parameter, so it must be fitted on data
you are not reporting on.

This script therefore does both splits in one pass:

    SPLIT_TUNE   = "val"    thresholds are chosen here
    SPLIT_REPORT = "test"   those thresholds are then APPLIED here, untouched

The headline number is test performance at val-chosen thresholds. The test-optimal
row is printed too, but only as an upper bound — do not quote it as a result.

=============================================================================
HOW TP/FP ARE ASSIGNED
=============================================================================
Standard COCO-style greedy matching, done ONCE at conf=0.001:

  sort predictions by confidence descending
  for each: take the highest-IoU unmatched GT of the same class with IoU >= 0.50
            found -> TP and that GT is consumed;  otherwise -> FP

Because the matching is greedy in confidence order, thresholding at t simply
truncates the list — the TP/FP label of every surviving prediction is unchanged.
So one matching pass supports the entire sweep. Duplicate boxes on an already
matched GT count as FP, which is what makes precision fall if the model is
spraying detections.

    P = TP / (TP + FP)      R = TP / n_gt      F1 = 2PR / (P + R)

No training. Pure inference. ~1-2 min per model per split.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import torch

try:
    from ultralytics import YOLO
    from ultralytics.utils.ops import xywhn2xyxy
except Exception as e:
    print(f"[ABORT] cannot import ultralytics: {e}")
    sys.exit(1)

# ============================== CONFIG ======================================= #
WEIGHTS = [
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_round11_v6i/y26_identity/weights/best.pt",
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_combo_v6i/y26_scb3_sbb50/weights/best.pt",
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_round5_v6i/y26_p2k2_hi/weights/best.pt",
]

DATA = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
SPLIT_TUNE = "val"      # thresholds are CHOSEN here
SPLIT_REPORT = "test"   # and APPLIED here — this is what you quote
IMGSZ = 640
DEVICE = 0              # 0 for the first GPU, or "cpu"
OUT_DIR = ""            # "" = next to this script

CURRENT_CONF = 0.25     # the default every result in this project was read at
IOU_MATCH = 0.50
MAX_DET = 1000
# Thresholds to evaluate. Dense at the low end, which is where the mass is.
THRESHOLDS = [round(x, 3) for x in
              [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15,
               0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]]
# ============================================================================= #


def box_iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)


def find_label(img_path):
    base = os.path.splitext(img_path)[0].replace("\\", "/")
    parts = base.split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return "/".join(parts) + ".txt"
    return base + ".txt"


def load_gt(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    rows = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([float(x) for x in p[:5]])
    if not rows:
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    arr = np.array(rows, np.float32)
    xyxy = xywhn2xyxy(torch.from_numpy(arr[:, 1:5]), w=img_w, h=img_h).numpy()
    return arr[:, 0].astype(np.int64), xyxy


def _resolve_source(raw):
    """check_det_dataset may return a dir, a .txt listing images, or a list."""
    def expand(item):
        item = str(item)
        if item.lower().endswith(".txt") and os.path.isfile(item):
            root = os.path.dirname(item)
            out = []
            with open(item) as f:
                for line in f:
                    p = line.strip()
                    if p:
                        out.append(p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p)))
            return out
        return [item]
    items = []
    for it in (raw if isinstance(raw, (list, tuple)) else [raw]):
        items.extend(expand(it))
    return items[0] if len(items) == 1 else items


def collect(weights, data, split, imgsz, device):
    """One inference pass -> per-class (conf, is_tp) arrays + gt counts."""
    from ultralytics.data.utils import check_det_dataset
    ds = check_det_dataset(data)
    src = _resolve_source(ds.get(split) or ds.get("val"))
    model = YOLO(weights)
    print(f"    [{split}] conf=0.001 max_det={MAX_DET} ...", flush=True)
    results = model.predict(source=src, conf=0.001, iou=0.7, imgsz=imgsz, device=device,
                            stream=True, verbose=False, save=False, max_det=MAX_DET)

    recs = defaultdict(list)      # class -> [(conf, is_tp)]
    n_gt = defaultdict(int)       # class -> ground-truth count
    cap_hits = 0
    for r in results:
        oh, ow = int(r.orig_shape[0]), int(r.orig_shape[1])
        gcls, gxyxy = load_gt(find_label(r.path), ow, oh)
        for c in gcls:
            n_gt[int(c)] += 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        pxyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        pcls = r.boxes.cls.cpu().numpy().astype(np.int64)
        pconf = r.boxes.conf.cpu().numpy().astype(np.float32)
        if len(pconf) >= MAX_DET:
            cap_hits += 1
        iou = box_iou_matrix(pxyxy, gxyxy)          # (P, G)
        used = np.zeros(len(gcls), bool)
        # greedy, confidence-descending: threshold-invariant TP/FP labelling
        for pi in np.argsort(-pconf):
            c = int(pcls[pi])
            best_j, best_v = -1, IOU_MATCH
            if len(gcls):
                for gj in np.where((gcls == c) & ~used)[0]:
                    if iou[pi, gj] >= best_v:
                        best_v, best_j = iou[pi, gj], gj
            if best_j >= 0:
                used[best_j] = True
                recs[c].append((float(pconf[pi]), 1))
            else:
                recs[c].append((float(pconf[pi]), 0))
    return recs, n_gt, cap_hits


def curve(rec, n_gt):
    """rec = [(conf, is_tp)] -> {threshold: (tp, fp, P, R, F1)}."""
    if not rec:
        return {t: (0, 0, 0.0, 0.0, 0.0) for t in THRESHOLDS}
    conf = np.array([x[0] for x in rec], np.float32)
    tp = np.array([x[1] for x in rec], np.int32)
    out = {}
    for t in THRESHOLDS:
        m = conf >= t
        TP, FP = int(tp[m].sum()), int((~tp[m].astype(bool)).sum())
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / n_gt if n_gt else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        out[t] = (TP, FP, P, R, F1)
    return out


def run_one(w, names, out_dir):
    run = os.path.basename(os.path.dirname(os.path.dirname(w))) or os.path.basename(w)
    print(f"\n  === {run}")
    tune = collect(w, DATA, SPLIT_TUNE, IMGSZ, DEVICE)
    test = collect(w, DATA, SPLIT_REPORT, IMGSZ, DEVICE)
    (rec_t, gt_t, cap_t), (rec_r, gt_r, cap_r) = tune, test

    L, csv = [], ["run,split,class,threshold,tp,fp,precision,recall,f1"]
    e = L.append
    e(f"RUN: {run}")
    e(f"tune on '{SPLIT_TUNE}'  ->  report on '{SPLIT_REPORT}'   (threshold is a free")
    e(f"parameter; fitting and reporting it on the same split is leakage)")
    e(f"match IoU >= {IOU_MATCH}   max_det={MAX_DET}   cap hits: tune {cap_t}, report {cap_r}")
    if cap_t or cap_r:
        e("  *** max_det cap BOUND — low-confidence tail truncated, results suspect.")
    e("")

    classes = sorted(set(gt_t) | set(gt_r))
    chosen = {}
    e(f"{'class':<10}{'best-F1 thr':>12}{'  | tuned on ' + SPLIT_TUNE:<26}"
      f"{'  | applied to ' + SPLIT_REPORT}")
    e(f"{'':<10}{'':>12}{'P':>8}{'R':>8}{'F1':>8}   {'P':>8}{'R':>8}{'F1':>8}")
    e("-" * 78)
    for c in classes:
        cname = names.get(c, str(c))
        ct, cr = curve(rec_t.get(c, []), gt_t.get(c, 0)), curve(rec_r.get(c, []), gt_r.get(c, 0))
        for split, cv, g in ((SPLIT_TUNE, ct, gt_t.get(c, 0)), (SPLIT_REPORT, cr, gt_r.get(c, 0))):
            for t, (TP, FP, P, R, F1) in cv.items():
                csv.append(f"{run},{split},{cname},{t},{TP},{FP},{P:.4f},{R:.4f},{F1:.4f}")
        best = max(ct, key=lambda t: ct[t][4])
        chosen[c] = best
        a, b = ct[best], cr[best]
        e(f"{cname:<10}{best:>12.3f}{a[2]*100:>8.1f}{a[3]*100:>8.1f}{a[4]*100:>8.1f}   "
          f"{b[2]*100:>8.1f}{b[3]*100:>8.1f}{b[4]*100:>8.1f}")

    e("")
    e(f"WHAT IT BUYS on '{SPLIT_REPORT}' — current conf {CURRENT_CONF} vs per-class tuned")
    e(f"{'class':<10}{'':>4}{'P':>8}{'R':>8}{'F1':>8}    {'thr':>6}{'P':>8}{'R':>8}{'F1':>8}{'  dF1':>8}")
    e("-" * 78)
    tot = {"now": [0, 0], "new": [0, 0], "gt": 0}
    for c in classes:
        cname = names.get(c, str(c))
        cr = curve(rec_r.get(c, []), gt_r.get(c, 0))
        n, b = cr[CURRENT_CONF], cr[chosen[c]]
        e(f"{cname:<10}{'now':>4}{n[2]*100:>8.1f}{n[3]*100:>8.1f}{n[4]*100:>8.1f}    "
          f"{chosen[c]:>6.3f}{b[2]*100:>8.1f}{b[3]*100:>8.1f}{b[4]*100:>8.1f}"
          f"{(b[4]-n[4])*100:>+8.1f}")
        tot["now"][0] += n[0]; tot["now"][1] += n[1]
        tot["new"][0] += b[0]; tot["new"][1] += b[1]
        tot["gt"] += gt_r.get(c, 0)
    for lbl, k in (("micro now", "now"), ("micro tuned", "new")):
        TP, FP = tot[k]
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / tot["gt"] if tot["gt"] else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        e(f"{lbl:<14}{P*100:>18.1f}{R*100:>8.1f}{F1*100:>8.1f}")

    e("")
    e("READ IT")
    e("  precision roughly HOLDS at the tuned thresholds -> the model ranks well and")
    e("     0.25 was simply the wrong operating point. Free win, no retraining;")
    e("     report the tuned per-class operating point and say the default cost")
    e("     20+ points of recall.")
    e("  precision COLLAPSES -> the correct box is ranked below junk. That is a")
    e("     classification / ranking failure and the first evidence-backed loss")
    e("     target this project has had.")
    e("")
    e("  Watch the TUNE vs REPORT columns. If the tuned threshold transfers, the two")
    e("  F1 values are close. A big drop means the threshold overfitted the val split")
    e("  and per-class tuning is not as free as it looks.")
    e("")

    txt = os.path.join(out_dir, f"threshold_sweep__{run}.txt")
    cp = os.path.join(out_dir, f"threshold_sweep__{run}.csv")
    open(txt, "w").write("\n".join(L) + "\n")
    open(cp, "w").write("\n".join(csv) + "\n")
    print("\n".join(L))
    print(f"saved -> {txt}\nsaved -> {cp}")


if __name__ == "__main__":
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    device = DEVICE if (torch.cuda.is_available() or DEVICE == "cpu") else "cpu"
    if device != DEVICE:
        print("  [note] no CUDA visible — falling back to cpu (slow)")

    missing = [w for w in WEIGHTS if not os.path.isfile(w)]
    if missing:
        print("[ABORT] these weights do not exist — fix WEIGHTS in the CONFIG block:")
        for w in missing:
            print(f"    {w}")
        sys.exit(1)
    if not os.path.isfile(DATA):
        sys.exit(f"[ABORT] DATA not found: {DATA}")
    if CURRENT_CONF not in THRESHOLDS:
        sys.exit(f"[ABORT] CURRENT_CONF={CURRENT_CONF} must be in THRESHOLDS")

    from ultralytics.data.utils import check_det_dataset
    ds = check_det_dataset(DATA)
    names = ds.get("names", {})
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}

    print(f"\n  DATA  {DATA}")
    print(f"  tune on '{SPLIT_TUNE}'  ->  report on '{SPLIT_REPORT}'")
    print(f"  IMGSZ {IMGSZ}  DEVICE {device}  IOU {IOU_MATCH}  {len(WEIGHTS)} model(s)")
    for w in WEIGHTS:
        run_one(w, names, out_dir)
