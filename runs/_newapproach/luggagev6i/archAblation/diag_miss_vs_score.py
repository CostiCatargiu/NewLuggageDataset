#!/usr/bin/env python3
r"""
diag_miss_vs_score.py — is the 26% bag-miss a PROPOSAL failure or a SCORING failure?

=============================================================================
THE QUESTION THIS ANSWERS, AND WHY IT DECIDES THE WHOLE LOSS DEBATE
=============================================================================
The collected confusion matrices were built at the DEFAULT conf threshold
(0.25). At that threshold bag misses 26.2% of its ground truths to background.
Two completely different things produce an identical "missed" cell:

  A. PROPOSAL failure   — the detector never emitted a box on that GT at all,
                          at ANY confidence. Nothing downstream (loss, threshold,
                          calibration) can recover it. It is a true recall
                          ceiling; only more/finer anchors (arch, resolution)
                          or better features help.

  B. SCORING failure    — the detector DID emit a well-placed box, but scored it
                          below 0.25, so the default-threshold matrix counts it as
                          missed. This is recoverable for free: a lower or
                          per-size threshold, or confidence calibration, gets it
                          back with ZERO retraining and — unlike SNT — zero risk
                          to recall.

This script re-runs inference at conf=0.001 (effectively no confidence filter),
matches every prediction to the ground truth by IoU, and for each MISSED-at-0.25
GT asks: was there a correctly-classified, well-localised box for it that simply
scored low? It reports the split by SIZE bucket, because the whole thesis is that
small objects are the ones being scored into oblivion.

READ:
  recovered_by_low_conf HIGH  -> SCORING failure -> a per-size threshold / cal
     is worth doing, and MORE LOSS WORK COULD STILL HELP (the classification
     head is the lever). This REOPENS the loss question, narrowly.
  recovered_by_low_conf LOW   -> PROPOSAL failure -> the box was never there.
     Loss is exhausted. Spend budget on arch / resolution. Confirms "close the
     loss campaign."

=============================================================================
WHERE TO RUN IT
=============================================================================
On the training box (weights + dataset live there). Example:

  python diag_miss_vs_score.py \
      --weights runs/detect/runs_yolo26_round11_v6i/y26_identity/weights/best.pt \
      --data    /home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml \
      --split   test --imgsz 640

Compare two models in one go (baseline vs the arch that moved the miss column):

  python diag_miss_vs_score.py \
      --weights .../y26_identity/weights/best.pt .../y26_p2k2_hi/weights/best.pt \
      --data .../data.yaml --split test

Outputs, per weights file, next to this script:
  miss_vs_score__<runname>.csv   machine-readable, one row per (class,size,band)
  miss_vs_score__<runname>.txt   the human summary + the verdict

No training. Pure inference. ~1-2 min per model on a GPU.
"""

import argparse
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


# ---- size buckets ----------------------------------------------------------
# Edges are on max(w,h) EXPRESSED AT imgsz=640-equivalent scale, so they match
# the small<48 / medium<=96 convention used throughout this project regardless
# of each image's native resolution. We compute a GT's side as a FRACTION of the
# image's longer edge, then multiply by imgsz -> a resolution-independent,
# letterbox-invariant size in the model's working frame. This sidesteps the
# stretch-vs-letterbox coordinate hazard entirely: IoU is done in NATIVE pixels
# (where preds already live), size bucketing is done on scale-normalised sides.
SMALL_PX = 48.0
MEDIUM_PX = 96.0

# confidence bands to bucket recovered boxes into
BANDS = [(0.001, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1.01)]
DEFAULT_CONF = 0.25          # the threshold the collected matrices used
IOU_MATCH = 0.50             # a "well-localised" recovered box needs IoU >= this


def box_iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> IoU (N,M). CPU numpy, small N,M per image."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return (inter / union).astype(np.float32)


def size_bucket(side_px):
    if side_px < SMALL_PX:
        return "small"
    if side_px <= MEDIUM_PX:
        return "medium"
    return "large"


def load_gt(label_path, img_w, img_h, imgsz):
    """Read a YOLO txt label.

    Returns:
      cls   (K,) int
      xyxy  (K,4) in NATIVE image pixels  -> matched against preds (also native)
      side640 (K,) max(w,h) as a fraction of the longer image edge, times imgsz
              -> the resolution-independent, letterbox-invariant size used only
                 for bucketing. IoU and bucketing are thus decoupled and each is
                 computed in the frame where it is unambiguous.
    """
    empty = (np.zeros((0,), np.int64), np.zeros((0, 4), np.float32),
             np.zeros((0,), np.float32))
    if not os.path.exists(label_path):
        return empty
    rows = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([float(x) for x in p[:5]])
    if not rows:
        return empty
    arr = np.array(rows, np.float32)
    cls = arr[:, 0].astype(np.int64)
    xywhn = arr[:, 1:5]                                   # normalised [0,1]
    # native-pixel xyxy for IoU matching against predictions
    xyxy = xywhn2xyxy(torch.from_numpy(xywhn), w=img_w, h=img_h).numpy()
    # scale-normalised side for bucketing: normalised w,h -> px on the longer
    # edge -> a 640-equivalent side independent of native resolution & padding
    longer = float(max(img_w, img_h))
    wpx = xywhn[:, 2] * img_w
    hpx = xywhn[:, 3] * img_h
    side640 = (np.maximum(wpx, hpx) / longer * imgsz).astype(np.float32)
    return cls, xyxy, side640


def find_label(img_path):
    """images/.../x.jpg -> labels/.../x.txt (standard YOLO layout)."""
    base = os.path.splitext(img_path)[0]
    if "images" in base.replace("\\", "/").split("/"):
        parts = base.replace("\\", "/").split("/")
        parts[parts.index("images")] = "labels"
        return "/".join(parts) + ".txt"
    return base + ".txt"


def _resolve_source(raw):
    """check_det_dataset may hand back: a directory, a single .txt listing image
    paths, or a list of dirs/txts. model.predict wants a dir, an image, or an
    explicit list of image paths — NOT a .txt list file. Normalise all cases to
    something predict() reads, preserving the exact image set of the split.
    """
    def expand(item):
        item = str(item)
        if item.lower().endswith(".txt") and os.path.isfile(item):
            root = os.path.dirname(item)
            paths = []
            with open(item) as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    paths.append(p if os.path.isabs(p) else os.path.normpath(
                        os.path.join(root, p)))
            return paths
        return [item]  # directory or single image — predict handles directly

    if isinstance(raw, (list, tuple)):
        out = []
        for it in raw:
            out.extend(expand(it))
        # if it collapsed to a single dir, hand the dir straight to predict
        return out[0] if len(out) == 1 else out
    return expand(raw)[0] if len(expand(raw)) == 1 else expand(raw)


def analyse(weights, data, split, imgsz, device, names):
    """Run conf=0.001 inference, match to GT, classify every GT as:
       hit@0.25 / recovered(scored below 0.25 but correct box exists) / true_miss.
    """
    model = YOLO(weights)

    # resolve the split's image list via the dataset yaml through ultralytics
    from ultralytics.data.utils import check_det_dataset
    ds = check_det_dataset(data)
    raw = ds.get(split) or ds.get("val")
    src = _resolve_source(raw)
    print(f"  [{os.path.basename(os.path.dirname(os.path.dirname(weights)))}] "
          f"predicting split='{split}' @ conf=0.001 imgsz={imgsz}")
    print(f"    source -> {src if isinstance(src, str) else f'{len(src)} images'}")

    results = model.predict(source=src, conf=0.001, iou=0.7, imgsz=imgsz,
                            device=device, stream=True, verbose=False,
                            save=False, agnostic_nms=False)

    # counters[class][size] = dict(gt, hit25, recovered, true_miss, band_counts{})
    def _new():
        return dict(gt=0, hit25=0, recovered=0, true_miss=0,
                    bands={b: 0 for b in BANDS})
    counters = defaultdict(lambda: defaultdict(_new))

    n_imgs = 0
    for r in results:
        n_imgs += 1
        img_path = r.path
        lab = find_label(img_path)
        # r.orig_shape is (h, w)
        oh, ow = int(r.orig_shape[0]), int(r.orig_shape[1])
        gcls, gxyxy, gside640 = load_gt(lab, ow, oh, imgsz)
        if len(gcls) == 0:
            continue

        # predictions are already in NATIVE image pixels — same frame as gxyxy.
        # No rescale, so no letterbox/stretch hazard in the IoU match.
        if r.boxes is None or len(r.boxes) == 0:
            pcls = np.zeros((0,), np.int64)
            pxyxy = np.zeros((0, 4), np.float32)
            pconf = np.zeros((0,), np.float32)
        else:
            pxyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pcls = r.boxes.cls.cpu().numpy().astype(np.int64)
            pconf = r.boxes.conf.cpu().numpy().astype(np.float32)

        iou = box_iou_matrix(gxyxy, pxyxy)  # (G,P), native pixels both sides

        for gi in range(len(gcls)):
            c = int(gcls[gi])
            sz = size_bucket(float(gside640[gi]))
            cell = counters[c][sz]
            cell["gt"] += 1

            # candidate preds: same class, IoU>=0.5 with this GT
            if len(pcls):
                cand = (pcls == c) & (iou[gi] >= IOU_MATCH)
            else:
                cand = np.zeros((0,), bool)

            if not cand.any():
                cell["true_miss"] += 1          # no correct box at ANY conf
                continue

            best_conf = float(pconf[cand].max())
            if best_conf >= DEFAULT_CONF:
                cell["hit25"] += 1              # already counted correct at 0.25
            else:
                cell["recovered"] += 1          # box exists, scored below 0.25
                for lo, hi in BANDS:
                    if lo <= best_conf < hi:
                        cell["bands"][(lo, hi)] += 1
                        break

    return counters, n_imgs


def report(run, counters, n_imgs, out_dir):
    lines = []
    def emit(s=""):
        lines.append(s)

    emit(f"RUN: {run}")
    emit(f"images scored: {n_imgs}   split re-eval @ conf=0.001, match IoU>= {IOU_MATCH}")
    emit(f"size edges (px, max side): small<{SMALL_PX:.0f}  medium<= {MEDIUM_PX:.0f}  large>")
    emit("")
    emit("For each (class,size): of all GTs, what fraction is")
    emit("  hit@0.25   = correct box already above default threshold")
    emit("  recovered  = correct box EXISTS but scored below 0.25  (SCORING loss)")
    emit("  true_miss  = no correct box at ANY confidence           (PROPOSAL loss)")
    emit("")
    hdr = (f"{'class':<10}{'size':<8}{'GT':>6}{'hit@.25%':>10}"
           f"{'recover%':>10}{'true_miss%':>12}")
    emit(hdr)
    emit("-" * len(hdr))

    csv = ["run,class,size,gt,hit25,recovered,true_miss,"
           "hit25_pct,recovered_pct,true_miss_pct,"
           "rec_b_001_05,rec_b_05_10,rec_b_10_25"]

    # aggregate small across classes too — the headline number
    agg = defaultdict(lambda: dict(gt=0, hit25=0, recovered=0, true_miss=0))

    for c in sorted(counters):
        cname = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        for sz in ("small", "medium", "large"):
            cell = counters[c].get(sz)
            if not cell or cell["gt"] == 0:
                continue
            g = cell["gt"]
            h, rec, tm = cell["hit25"], cell["recovered"], cell["true_miss"]
            emit(f"{cname:<10}{sz:<8}{g:>6}{100*h/g:>10.1f}"
                 f"{100*rec/g:>10.1f}{100*tm/g:>12.1f}")
            b = cell["bands"]
            csv.append(f"{run},{cname},{sz},{g},{h},{rec},{tm},"
                       f"{100*h/g:.2f},{100*rec/g:.2f},{100*tm/g:.2f},"
                       f"{b[(0.001,0.05)]},{b[(0.05,0.10)]},{b[(0.10,0.25)]}")
            a = agg[sz]
            a["gt"] += g; a["hit25"] += h; a["recovered"] += rec; a["true_miss"] += tm

    emit("")
    emit("ALL CLASSES combined, by size:")
    emit(f"{'':<10}{'size':<8}{'GT':>6}{'hit@.25%':>10}{'recover%':>10}{'true_miss%':>12}")
    emit("-" * len(hdr))
    for sz in ("small", "medium", "large"):
        a = agg[sz]
        if a["gt"] == 0:
            continue
        g = a["gt"]
        emit(f"{'':<10}{sz:<8}{g:>6}{100*a['hit25']/g:>10.1f}"
             f"{100*a['recovered']/g:>10.1f}{100*a['true_miss']/g:>12.1f}")

    emit("")
    emit("VERDICT (small bucket is the test):")
    a = agg["small"]
    if a["gt"]:
        rec_pct = 100 * a["recovered"] / a["gt"]
        tm_pct = 100 * a["true_miss"] / a["gt"]
        emit(f"  small recovered = {rec_pct:.1f}%   small true_miss = {tm_pct:.1f}%")
        if rec_pct >= tm_pct and rec_pct >= 3.0:
            emit("  -> SCORING-dominated. The boxes exist below 0.25. A per-size")
            emit("     threshold / calibration recovers them for free, and the")
            emit("     classification head IS a live lever -> narrow loss work COULD help.")
        elif tm_pct > rec_pct and tm_pct >= 3.0:
            emit("  -> PROPOSAL-dominated. The boxes are not there at any conf.")
            emit("     Loss is exhausted; spend budget on arch / resolution.")
        else:
            emit("  -> both small; the small bucket is near-saturated already.")
    emit("")

    txt = os.path.join(out_dir, f"miss_vs_score__{run}.txt")
    csvp = os.path.join(out_dir, f"miss_vs_score__{run}.csv")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(csvp, "w") as f:
        f.write("\n".join(csv) + "\n")
    print("\n".join(lines))
    print(f"\nsaved -> {txt}\nsaved -> {csvp}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="one or more best.pt paths")
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0 if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # class names from the dataset yaml
    from ultralytics.data.utils import check_det_dataset
    ds = check_det_dataset(args.data)
    names = ds.get("names", {})
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}

    for w in args.weights:
        run = os.path.basename(os.path.dirname(os.path.dirname(w))) or os.path.basename(w)
        counters, n_imgs = analyse(w, args.data, args.split, args.imgsz,
                                   args.device, names)
        report(run, counters, n_imgs, out_dir)
