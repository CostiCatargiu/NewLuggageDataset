#!/usr/bin/env python3
r"""
FALSE-POSITIVE DECOMPOSITION — what is outranking the true positives?
=====================================================================

`diag_miss_vs_score.py` established that the misses are SCORING, not PROPOSAL:
true_miss is only 2.7-5.5% per class, so the boxes exist and are ranked below
junk. It did not establish WHAT the junk is. Those cases point at completely
different conclusions:

    DUPLICATE   IoU >= 0.5 with a GT that a higher-scoring prediction already
                claimed. The head is NMS-free, so duplicate suppression is
                carried entirely by the winner/runner-up confidence gap. Leaking
                duplicates are a LOSS-SIDE target, and they would explain why SNT
                (-3.93, gap closed) and TSH (+0.11, gap widened) bracket an
                interior optimum instead of one of them winning.

    CLS         IoU >= 0.5 with a GT of a DIFFERENT class. Matches the
                misclassification rate that sat at 3.94-4.90% across all 81 runs
                and never moved. Not loss-fixable — a property of the data.

    LOC         best same-class IoU in [LOC_LO, 0.5). The object is found and
                labelled but the box is not good enough to match. This is the
                mAP50 -> mAP50-95 gap (80.18 vs 55.24) in FP form.

    BACKGROUND  best IoU with ANY GT below LOC_LO. Hallucination. A feature /
                capacity problem, not something a loss reweighting reaches.

The decomposition is reported in SCORE BANDS, because only the FPs that outrank
true positives cost AP. A pile of background junk at 0.02 is harmless; three
duplicates at 0.8 are not.


HOW TO READ IT
--------------
Look at the HIGH bands (>= 0.50) and the "outranking the median TP" summary.

    DUPLICATE dominates
        -> suppression is leaking in the NMS-free head. First evidence-backed
           loss target in the project, and it sharpens the SNT/TSH result from
           "the gap is at an interior optimum" to "here is what the gap costs".

    BACKGROUND dominates
        -> not reachable from the loss. Stop; the campaign's flatness is
           explained and the axis closes with a measurement behind it.

    CLS dominates
        -> the constant 4.4% misclassification, already known immovable.

    LOC dominates
        -> localisation, not ranking. Points back at box regression and the
           mAP50-95 gap rather than at scoring.

Matching conventions mirror diag_miss_vs_score.py (IoU 0.50, conf 0.001,
max_det 1000) so the two are directly comparable. NOTE: ultralytics'
ConfusionMatrix matches at 0.45 — do not quote these as one number.


Usage:
    python diag_fp_decomposition.py                     # WEIGHTS below
    python diag_fp_decomposition.py /path/to/best.pt
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import xywhn2xyxy

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
_RUNS = "/home/constantin/Doctorat/YoloLib/runs/detect"

# Uncomment one. Run y26_identity FIRST — it is stock, so nothing confounds the
# failure mode. The other two are only worth it if duplicates dominate there:
# comparing the three then shows whether the best loss / best arch config
# actually reduced duplicates, which no mAP table can tell you.
WEIGHTS = f"{_RUNS}/runs_yolo26_round11_v6i/y26_identity/weights/best.pt"      # stock, 55.24
# WEIGHTS = f"{_RUNS}/runs_yolo26_combo_v6i/y26_scb3_sbb50/weights/best.pt"    # best loss, 55.65
# WEIGHTS = f"{_RUNS}/runs_yolo26_round5_v6i/y26_p2k2_hi/weights/best.pt"      # best arch, 56.46 (b48)

SPLIT = "test"
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CONF = 0.001  # see everything, then band it by score
IOU_MATCH = 0.50  # a TP needs IoU >= this, same class
LOC_LO = 0.10  # below this against every GT, the box is BACKGROUND
MAX_DET = 1000

SMALL_PX, MEDIUM_PX = 48.0, 96.0
BANDS = [(0.90, 1.01), (0.70, 0.90), (0.50, 0.70), (0.25, 0.50), (0.05, 0.25), (0.0, 0.05)]
KINDS = ["dup", "cls", "loc", "bg"]
# =============================================================================


def box_iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> IoU (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)


def find_label(img_path):
    """images/.../x.jpg -> labels/.../x.txt (standard YOLO layout)."""
    base = os.path.splitext(img_path)[0]
    parts = base.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return "/".join(parts) + ".txt"
    return base + ".txt"


def load_gt(label_path, img_w, img_h):
    """YOLO txt -> (cls (K,), xyxy (K,4) native px)."""
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
    """check_det_dataset may return a dir, a .txt listing images, or a list of either.
    predict() does not read .txt list files, so expand them while preserving the split.
    """
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


def band_of(score):
    for lo, hi in BANDS:
        if lo <= score < hi:
            return (lo, hi)
    return BANDS[-1]


def classify_image(pred_xyxy, pred_cls, pred_conf, gt_cls, gt_xyxy):
    """Greedy match in score order; label every prediction TP or an FP kind.

    Returns a list of (conf, cls, kind) where kind is 'tp' or one of KINDS.
    """
    order = np.argsort(-pred_conf)
    ious = box_iou_matrix(pred_xyxy, gt_xyxy)  # (N, M)
    claimed = np.zeros(len(gt_cls), dtype=bool)
    out = []
    for i in order:
        c = int(pred_cls[i])
        row = ious[i]
        same = gt_cls == c
        best_same = float(row[same].max()) if same.any() else 0.0
        best_any = float(row.max()) if len(gt_cls) else 0.0

        if best_same >= IOU_MATCH:
            # candidate GTs of this class above threshold, best first
            cand = np.where(same & (row >= IOU_MATCH))[0]
            cand = cand[np.argsort(-row[cand])]
            free = [j for j in cand if not claimed[j]]
            if free:
                claimed[free[0]] = True
                out.append((float(pred_conf[i]), c, "tp"))
            else:
                out.append((float(pred_conf[i]), c, "dup"))  # GT already taken
        elif best_any >= IOU_MATCH:
            out.append((float(pred_conf[i]), c, "cls"))  # good box, wrong class
        elif best_same >= LOC_LO:
            out.append((float(pred_conf[i]), c, "loc"))  # right class, sloppy box
        else:
            out.append((float(pred_conf[i]), c, "bg"))
    return out


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else WEIGHTS
    data = check_det_dataset(DATA_YAML)
    names = data.get("names", {})
    src = _resolve_source(data[SPLIT])

    model = YOLO(weights)
    e2e = bool(getattr(getattr(model, "model", None), "end2end", False))
    print(f"[run]   {weights}\n[split] {SPLIT}  conf={CONF}  IoU={IOU_MATCH}  max_det={MAX_DET}")
    print(f"[head]  end2end={e2e}" + ("  (NMS-free: duplicates reach the output unsuppressed)"
                                      if e2e else "  (NMS active: duplicates above iou= are removed first)"))

    results = model.predict(source=src, conf=CONF, iou=0.7, imgsz=IMG_SIZE, max_det=MAX_DET,
                            device=DEVICE, stream=True, verbose=False)

    # counters[cls][band][kind] and the flat record needed for the median-TP cut
    counters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    tp_scores = defaultdict(list)
    fp_records = defaultdict(list)  # cls -> [(conf, kind)]
    n_imgs = cap_hits = 0

    for r in results:
        n_imgs += 1
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        if len(b) >= MAX_DET:
            cap_hits += 1
        pred_xyxy = b.xyxy.cpu().numpy()
        pred_cls = b.cls.cpu().numpy().astype(int)
        pred_conf = b.conf.cpu().numpy()

        h, w = r.orig_shape
        gt_cls, gt_xyxy = load_gt(find_label(r.path), w, h)

        for conf, c, kind in classify_image(pred_xyxy, pred_cls, pred_conf, gt_cls, gt_xyxy):
            counters[c][band_of(conf)][kind] += 1
            if kind == "tp":
                tp_scores[c].append(conf)
            else:
                fp_records[c].append((conf, kind))

    report(weights, counters, tp_scores, fp_records, names, n_imgs, cap_hits, e2e)


def report(weights, counters, tp_scores, fp_records, names, n_imgs, cap_hits, e2e):
    print("\n" + "=" * 86)
    print("FALSE-POSITIVE DECOMPOSITION")
    print(f"images={n_imgs}  max_det cap hit on {cap_hits} images  end2end={e2e}")
    print("  dup = GT already claimed by a higher-scoring box   cls = good box, wrong class")
    print("  loc = right class, IoU in [0.10, 0.50)             bg  = IoU < 0.10 with any GT")
    print("=" * 86)

    for c in sorted(counters):
        nm = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        print(f"\nCLASS {nm}")
        print(f"{'score band':<14}{'TP':>8}{'FP':>8}{'dup%':>8}{'cls%':>8}{'loc%':>8}{'bg%':>8}")
        print("-" * 62)
        for lo, hi in BANDS:
            d = counters[c].get((lo, hi))
            if not d:
                continue
            tp = d.get("tp", 0)
            fp = sum(d.get(k, 0) for k in KINDS)
            pct = [100.0 * d.get(k, 0) / fp if fp else 0.0 for k in KINDS]
            print(f"{f'{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}':<14}{tp:>8}{fp:>8}"
                  + "".join(f"{p:>8.1f}" for p in pct))

    print("\n" + "-" * 86)
    print("THE NUMBER — FPs that OUTRANK the median true positive")
    print(f"{'class':<12}{'medTP':>8}{'nFP above':>11}{'dup%':>8}{'cls%':>8}{'loc%':>8}{'bg%':>8}")
    print("-" * 63)
    pooled = defaultdict(int)
    for c in sorted(counters):
        nm = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        if not tp_scores[c]:
            continue
        med = float(np.median(tp_scores[c]))
        above = [k for s, k in fp_records[c] if s >= med]
        n = len(above)
        pct = []
        for k in KINDS:
            v = sum(1 for x in above if x == k)
            pooled[k] += v
            pct.append(100.0 * v / n if n else 0.0)
        print(f"{nm:<12}{med:>8.3f}{n:>11}" + "".join(f"{p:>8.1f}" for p in pct))

    total = sum(pooled.values())
    print("\nVERDICT")
    if not total:
        print("  no false positives above the median TP — ranking is not the constraint here.")
    else:
        share = {k: 100.0 * pooled[k] / total for k in KINDS}
        print("  pooled: " + "  ".join(f"{k}={share[k]:.1f}%" for k in KINDS))
        top = max(share, key=share.get)
        if top == "dup" and share["dup"] > 40:
            print("  -> DUPLICATES dominate. In an NMS-free head suppression is the")
            print("     winner/runner-up score gap, so this IS loss-reachable. It also")
            print("     explains the SNT/TSH bracket. First evidence-backed loss target.")
        elif top == "bg" and share["bg"] > 40:
            print("  -> BACKGROUND dominates. Hallucinated boxes are a feature/capacity")
            print("     problem; no reweighting of the loss reaches them. The loss axis")
            print("     closes here, with a measurement behind it.")
        elif top == "cls":
            print("  -> WRONG CLASS dominates. This is the 3.94-4.90% that never moved")
            print("     across 81 runs. Not loss-fixable; it is a property of the data.")
        elif top == "loc":
            print("  -> LOCALISATION dominates. The constraint is box quality, not")
            print("     ranking — consistent with the mAP50 80.18 vs mAP50-95 55.24 gap.")
        else:
            print(f"  -> mixed, largest is {top} at {share[top]:.1f}%. No single cause;")
            print("     treat any mechanism aimed at one of them as low prior.")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
