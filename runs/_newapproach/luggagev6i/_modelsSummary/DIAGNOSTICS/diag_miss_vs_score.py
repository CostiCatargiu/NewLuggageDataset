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
On the training box (weights + dataset live there). Edit the CONFIG block below,
then:

  python diag_miss_vs_score.py

Outputs, per weights file, next to this script:
  miss_vs_score__<runname>.csv   machine-readable, one row per (class,size,band)
  miss_vs_score__<runname>.txt   the human summary + the verdict

No training. Pure inference. ~1-2 min per model on a GPU.
"""

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
# One or more best.pt paths. The run name is taken from the folder two levels up
# (…/<run>/weights/best.pt), which is how ultralytics lays runs out.
WEIGHTS = [
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_round11_v6i/y26_identity/weights/best.pt",
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_combo_v6i/y26_scb3_sbb50/weights/best.pt",
    "/home/constantin/Doctorat/YoloLib/runs/detect/runs_yolo26_round5_v6i/y26_p2k2_hi/weights/best.pt",
]

DATA = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
SPLIT = "test"
IMGSZ = 640
DEVICE = 0          # 0 for the first GPU, or "cpu"

# Where the .txt / .csv reports land. "" = next to this script.
OUT_DIR = ""
# ============================================================================= #


# ---- size buckets ----------------------------------------------------------
# Edges are on max(w,h) EXPRESSED AT imgsz=640-equivalent scale, so they match
# the small<48 / medium<=96 convention used throughout this project regardless
# of each image's native resolution. We compute a GT's side as a FRACTION of the
# image's longer edge, then multiply by imgsz -> a resolution-independent,
# letterbox-invariant size in the model's working frame. This sidesteps the
# stretch-vs-letterbox coordinate hazard entirely: IoU is done in NATIVE pixels
# (where preds already live), size bucketing is done on scale-normalised sides.
#
# WARNING — THESE EDGES ARE *NOT* THE COCO BUCKETS USED ELSEWHERE IN THIS PROJECT.
# Every mAP50_small / mAP50_medium / mAP50_large figure quoted in this campaign
# comes from CocoEvalAllFolders_luggage.py, which uses COCO *AREA* buckets:
#     small  area < 32^2   -> side < 32 for a square box
#     medium 32^2 .. 96^2
#     large  > 96^2
# The edges below are on MAX SIDE at 48/96, so "small" here is materially BROADER
# than COCO small. That is a defensible choice (max side is the right notion for
# "can the grid resolve it"), but it means THESE ROWS ARE NOT COMPARABLE to the
# mAP_small numbers in the results JSONs. Do not put them in the same table.
# Flip COCO_BUCKETS to True to use area-equivalent edges instead.
COCO_BUCKETS = False
SMALL_PX = 32.0 if COCO_BUCKETS else 48.0
MEDIUM_PX = 96.0

# confidence bands to bucket recovered boxes into
BANDS = [(0.001, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1.01)]
DEFAULT_CONF = 0.25          # the threshold the collected matrices used
IOU_MATCH = 0.50             # a "well-localised" recovered box needs IoU >= this

# At conf=0.001 the boxes being hunted are BY DEFINITION the lowest-scoring ones,
# and predict() keeps only the top `max_det` by score. If that cap binds, the
# low-confidence tail is silently truncated, `recovered` is undercounted, and the
# script concludes PROPOSAL-dominated — precisely the failure it exists to rule
# out. ~5 boxes/image means 300 (the default) is very likely fine, but "very
# likely" is how the batch confound survived three weeks. The report states
# outright whether the cap was ever reached.
MAX_DET = 1000

# ultralytics' ConfusionMatrix matches at IoU 0.45; this script uses 0.50. So
# `hit@0.25` here will NOT exactly equal the confusion matrix `correct` column.
# Close, but they are different measurements — do not quote them as one number.
CM_IOU_NOTE = "confusion matrices matched at IoU 0.45; this script uses 0.50"


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
                            save=False, agnostic_nms=False, max_det=MAX_DET)
    # Track whether the max_det cap ever bound. If it did, the truncated tail is
    # exactly the low-confidence region this script is measuring, and any
    # "PROPOSAL-dominated" verdict would be an artefact rather than a finding.
    cap_hits = 0
    max_preds_seen = 0

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
            max_preds_seen = max(max_preds_seen, len(pconf))
            if len(pconf) >= MAX_DET:
                cap_hits += 1

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

    return counters, n_imgs, cap_hits, max_preds_seen


def report(run, counters, n_imgs, out_dir, names, cap_hits=0, max_preds_seen=0):
    lines = []
    def emit(s=""):
        lines.append(s)

    emit(f"RUN: {run}")
    emit(f"images scored: {n_imgs}   split re-eval @ conf=0.001, match IoU>= {IOU_MATCH}")
    emit(f"size edges (px, MAX SIDE): small<{SMALL_PX:.0f}  medium<= {MEDIUM_PX:.0f}  large>")
    emit(f"NOT the COCO area buckets used in the results JSONs — do not cross-compare.")
    emit(f"{CM_IOU_NOTE}")
    emit(f"max_det={MAX_DET}   images hitting the cap: {cap_hits}   "
         f"most preds in one image: {max_preds_seen}")
    if cap_hits:
        emit("  *** WARNING: the max_det cap BOUND on some images. The truncated tail is")
        emit("      the low-confidence region this script measures, so `recovered` is")
        emit("      UNDERCOUNTED and any PROPOSAL-dominated verdict may be an artefact.")
        emit("      Raise MAX_DET and re-run before believing the verdict.")
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

    # ---- PER CLASS, all sizes pooled ---------------------------------------
    # The size verdict alone can read "near-saturated" while one CLASS is the
    # whole problem. On this dataset bag has AR50 95.6 but only ~66.5% survive at
    # conf 0.25 — a ~29pt gap that spans every size bucket, so it is invisible in
    # a size-only summary.
    percls = defaultdict(lambda: dict(gt=0, hit25=0, recovered=0, true_miss=0))
    for c in counters:
        cname = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        for sz in counters[c]:
            cell = counters[c][sz]
            a = percls[cname]
            for k in ("gt", "hit25", "recovered", "true_miss"):
                a[k] += cell[k]

    emit("")
    emit("PER CLASS, all sizes pooled:")
    emit(f"{'class':<12}{'GT':>7}{'hit@.25%':>10}{'recover%':>10}{'true_miss%':>12}"
         f"{'  verdict'}")
    emit("-" * 62)
    for cname in sorted(percls, key=lambda k: -percls[k]["gt"]):
        a = percls[cname]
        if not a["gt"]:
            continue
        g = a["gt"]
        rec, tm = 100 * a["recovered"] / g, 100 * a["true_miss"] / g
        if rec + tm < 3.0:
            v = "saturated"
        elif rec >= tm:
            v = "SCORING-limited"
        else:
            v = "PROPOSAL-limited"
        emit(f"{cname:<12}{g:>7}{100*a['hit25']/g:>10.1f}{rec:>10.1f}{tm:>12.1f}  {v}")
        csv.append(f"{run},{cname},ALL,{g},{a['hit25']},{a['recovered']},{a['true_miss']},"
                   f"{100*a['hit25']/g:.2f},{rec:.2f},{tm:.2f},,,")

    def verdict(label, a):
        if not a["gt"]:
            return
        g = a["gt"]
        rec, tm = 100 * a["recovered"] / g, 100 * a["true_miss"] / g
        emit(f"  {label}: recovered {rec:.1f}%   true_miss {tm:.1f}%")
        if rec >= tm and rec >= 3.0:
            emit("    -> SCORING-limited. The boxes EXIST below 0.25. A per-class or")
            emit("       per-size threshold recovers them with zero retraining, and the")
            emit("       classification head is a live lever -> narrow loss work COULD help.")
        elif tm > rec and tm >= 3.0:
            emit("    -> PROPOSAL-limited. No correct box at ANY confidence. Nothing in")
            emit("       the loss can recover these; spend budget on arch / resolution.")
        else:
            emit("    -> near-saturated; neither failure mode has meaningful mass.")

    emit("")
    emit("VERDICT")
    verdict("small bucket (all classes)", agg["small"])
    worst = min(percls.items(), key=lambda kv: kv[1]["hit25"] / max(kv[1]["gt"], 1),
                default=(None, None))
    if worst[0] is not None:
        emit("")
        verdict(f"worst class ({worst[0]})", worst[1])
    emit("")
    emit("  If these two disagree, believe the CLASS one — a size-pooled summary can")
    emit("  read 'saturated' while a single class carries the entire deficit.")
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
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    device = DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu"
    if device != DEVICE:
        print(f"  [note] no CUDA visible — falling back to cpu (this will be slow)")

    missing = [w for w in WEIGHTS if not os.path.isfile(w)]
    if missing:
        print("[ABORT] these weights do not exist — fix WEIGHTS in the CONFIG block:")
        for w in missing:
            print(f"    {w}")
        sys.exit(1)
    if not os.path.isfile(DATA):
        sys.exit(f"[ABORT] DATA not found: {DATA}")

    print(f"\n  DATA    {DATA}")
    print(f"  SPLIT   {SPLIT}   IMGSZ {IMGSZ}   DEVICE {device}   MAX_DET {MAX_DET}")
    print(f"  buckets small<{SMALL_PX:.0f} medium<={MEDIUM_PX:.0f} (max side"
          f"{', COCO-equivalent' if COCO_BUCKETS else ', NOT the COCO area buckets'})")
    print(f"  OUT     {out_dir}")
    print(f"  {len(WEIGHTS)} model(s)\n")

    # class names from the dataset yaml
    from ultralytics.data.utils import check_det_dataset
    ds = check_det_dataset(DATA)
    names = ds.get("names", {})
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}

    for w in WEIGHTS:
        run = os.path.basename(os.path.dirname(os.path.dirname(w))) or os.path.basename(w)
        counters, n_imgs, cap_hits, max_preds = analyse(
            w, DATA, SPLIT, IMGSZ, device, names)
        report(run, counters, n_imgs, out_dir, names, cap_hits, max_preds)
