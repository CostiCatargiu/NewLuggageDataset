"""
YOLO Inference + Detection Metrics — Multi-Split Analysis (valid + test)

Runs inference on BOTH valid and test splits in a single execution,
collects detailed TP/FP/FN statistics, and produces:
  - Per-split detailed report (printed + saved to .txt)
  - Cross-split comparison summary (why does test differ from valid?)
  - Worst-N images saved with TP/FP/FN boxes drawn

Reports include:
  1. Overall TP/FP/FN, precision, recall, F1
  2. Per-class summary (GT count, pred count, TP, FP, FN)
  3. Image-level TN/FP (specificity proxy)
  4. Confidence interval distribution
  5. Size-stratified TP/FP/FN (small/medium/large)
  6. TP vs FP confidence analysis
  7. Cross-class confusion matrix
  8. FP spatial distribution
  9. Top-N worst images
 10. VALID vs TEST comparison summary
"""

import sys
import io
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_new_weapon_dataset_full/swa_09_05_fulldataset/weights/best.pt"

DATASET_ROOT = "/home/constantin/Doctorat/GunDatasetHistogram"

# Splits to evaluate (runs both in sequence)
EVAL_SPLITS = {
    "valid": {
        "images": f"{DATASET_ROOT}/valid/images",
        "labels": f"{DATASET_ROOT}/valid/labels",
    },
    "test": {
        "images": f"{DATASET_ROOT}/test/images",
        "labels": f"{DATASET_ROOT}/test/labels",
    },
}

# Inference thresholds
CONF_THRES = 0.1
IOU_NMS = 0.60
IOU_MATCH = 0.50

# Preprocessing: "none", "clahe", "histeq"
PREPROCESS = "none"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Confidence bins
BIN_EDGES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# Size thresholds (pixels at original resolution)
SMALL_AREA_PX = 32 * 32    # 1024
MEDIUM_AREA_PX = 96 * 96   # 9216

# Worst image saving
SAVE_WORST_N = 20

# Output directory (auto-created)
OUTPUT_ROOT = str(Path(__file__).parent / "inference_debug_output")

# Report file
REPORT_FILE = str(Path(__file__).parent / "inference_analysis_report.txt")


# =========================
# Helpers
# =========================

class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def list_images(folder: str) -> list:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def bin_label(conf: float) -> str:
    for i in range(len(BIN_EDGES) - 1):
        hi = BIN_EDGES[i]
        lo = BIN_EDGES[i + 1]
        if conf < hi and conf >= lo:
            return f"{hi:.1f}-{lo:.1f}"
    if conf >= 0.9:
        return "1.0-0.9"
    return "0.1-0.0"


def safe_mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def yolo_xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
    return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0, ax2-ax1)*max(0, ay2-ay1) + max(0, bx2-bx1)*max(0, by2-by1) - inter
    return (inter / union) if union > 1e-9 else 0.0


def classify_box_size(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    area = max(0, x2-x1) * max(0, y2-y1)
    if area < SMALL_AREA_PX:
        return "small"
    elif area < MEDIUM_AREA_PX:
        return "medium"
    return "large"


def box_center_normalized(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy
    return ((x1+x2)/2)/img_w, ((y1+y2)/2)/img_h


def load_gt_yolo_labels(label_path: Path, img_w: int, img_h: int):
    gt_by_class = defaultdict(list)
    if not label_path.exists():
        return gt_by_class
    try:
        text = label_path.read_text().strip().splitlines()
    except Exception:
        return gt_by_class
    for line in text:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except Exception:
            continue
        gt_by_class[cid].append(yolo_xywhn_to_xyxy(xc, yc, w, h, img_w, img_h))
    return gt_by_class


def match_preds_to_gt(preds_xyxy, preds_conf, gt_xyxy, iou_thr):
    if not preds_xyxy and not gt_xyxy:
        return 0, 0, 0
    order = np.argsort(-np.array(preds_conf)) if preds_conf else np.array([], dtype=int)
    gt_used = [False] * len(gt_xyxy)
    tp = fp = 0
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gt_xyxy):
            if gt_used[j]: continue
            iou = iou_xyxy(preds_xyxy[i], gbox)
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1; gt_used[best_j] = True
        else:
            fp += 1
    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def match_preds_to_gt_detailed(preds_xyxy, preds_conf, gt_xyxy, iou_thr):
    tp_boxes, fp_boxes, fn_boxes = [], [], []
    if not preds_xyxy and not gt_xyxy:
        return tp_boxes, fp_boxes, fn_boxes
    order = np.argsort(-np.array(preds_conf)) if preds_conf else np.array([], dtype=int)
    gt_used = [False] * len(gt_xyxy)
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gt_xyxy):
            if gt_used[j]: continue
            iou = iou_xyxy(preds_xyxy[i], gbox)
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp_boxes.append((preds_xyxy[i], preds_conf[i], gt_xyxy[best_j], best_iou))
            gt_used[best_j] = True
        else:
            fp_boxes.append((preds_xyxy[i], preds_conf[i], best_iou))
    for j, used in enumerate(gt_used):
        if not used:
            fn_boxes.append((gt_xyxy[j],))
    return tp_boxes, fp_boxes, fn_boxes


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*prec*rec) / (prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1


# Preprocessing
def apply_clahe_bgr(bgr, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    lab2 = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def apply_histeq_bgr(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    ycrcb2 = cv2.merge((cv2.equalizeHist(y), cr, cb))
    return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)

def preprocess_image(bgr):
    mode = (PREPROCESS or "none").lower().strip()
    if mode == "none": return bgr
    if mode == "clahe": return apply_clahe_bgr(bgr, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
    if mode == "histeq": return apply_histeq_bgr(bgr)
    raise ValueError(f"Unknown PREPROCESS: {PREPROCESS}")


# =========================
# Single-split evaluation
# =========================

def evaluate_split(model, class_names, split_name, images_dir, labels_dir, output_dir):
    """Run inference on one split and return all collected stats as a dict."""

    images = list_images(images_dir)
    if not images:
        print(f"  No images found in: {images_dir}")
        return None

    labels_path = Path(labels_dir)
    split_output = Path(output_dir) / split_name
    if SAVE_WORST_N > 0:
        split_output.mkdir(parents=True, exist_ok=True)

    # Accumulators
    pred_count_per_class = defaultdict(int)
    confs_per_class = defaultdict(list)
    bin_counts = defaultdict(int)
    all_confs = []
    total_preds = 0
    gt_count_per_class = defaultdict(int)
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)
    img_tn_per_class = defaultdict(int)
    img_fp_per_class = defaultdict(int)
    size_metrics = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    tp_confs_per_class = defaultdict(list)
    fp_confs_per_class = defaultdict(list)
    fp_locations = defaultdict(list)
    per_image_errors = []
    total_images = len(images)

    print(f"\n{'#' * 70}")
    print(f"  EVALUATING SPLIT: {split_name.upper()}")
    print(f"  Images: {total_images} from {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  CONF={CONF_THRES} IOU_NMS={IOU_NMS} IOU_MATCH={IOU_MATCH} PREPROCESS={PREPROCESS}")
    print(f"{'#' * 70}")

    for idx, img_path in enumerate(images, start=1):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        img_h, img_w = im.shape[:2]

        label_path = labels_path / f"{img_path.stem}.txt"
        gt_by_class = load_gt_yolo_labels(label_path, img_w, img_h)

        for cid, gt_boxes in gt_by_class.items():
            gt_count_per_class[cid] += len(gt_boxes)

        im_for_infer = preprocess_image(im)
        results = model.predict(source=im_for_infer, conf=CONF_THRES, iou=IOU_NMS, verbose=False)
        r = results[0]
        boxes = r.boxes

        pred_by_class_xyxy = defaultdict(list)
        pred_by_class_conf = defaultdict(list)

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy().astype(float)
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
            total_preds += len(confs)
            all_confs.extend(confs.tolist())
            for c, cf, bb in zip(cls_ids, confs, xyxy):
                pred_count_per_class[c] += 1
                confs_per_class[c].append(float(cf))
                bin_counts[bin_label(float(cf))] += 1
                pred_by_class_xyxy[c].append(bb.tolist())
                pred_by_class_conf[c].append(float(cf))

        img_tp_total = img_fp_total = img_fn_total = 0
        img_class_errors = {}

        for cid in set(gt_by_class.keys()) | set(pred_by_class_xyxy.keys()):
            gt_boxes = gt_by_class.get(cid, [])
            pr_boxes = pred_by_class_xyxy.get(cid, [])
            pr_confs = pred_by_class_conf.get(cid, [])

            tp, fp, fn = match_preds_to_gt(pr_boxes, pr_confs, gt_boxes, IOU_MATCH)
            tp_per_class[cid] += tp; fp_per_class[cid] += fp; fn_per_class[cid] += fn
            img_tp_total += tp; img_fp_total += fp; img_fn_total += fn
            img_class_errors[cid] = {'tp': tp, 'fp': fp, 'fn': fn}

            tp_det, fp_det, fn_det = match_preds_to_gt_detailed(pr_boxes, pr_confs, gt_boxes, IOU_MATCH)

            for pbox, conf, gbox, iou in tp_det:
                size_metrics[cid][classify_box_size(gbox)]['tp'] += 1
                tp_confs_per_class[cid].append(conf)
            for pbox, conf, best_iou in fp_det:
                size_metrics[cid][classify_box_size(pbox)]['fp'] += 1
                fp_confs_per_class[cid].append(conf)
                cx, cy = box_center_normalized(pbox, img_w, img_h)
                fp_locations[cid].append((cx, cy))
            for (gbox,) in fn_det:
                size_metrics[cid][classify_box_size(gbox)]['fn'] += 1

        for pred_cid in pred_by_class_xyxy:
            for pbox, conf in zip(pred_by_class_xyxy[pred_cid], pred_by_class_conf[pred_cid]):
                for gt_cid in gt_by_class:
                    if gt_cid == pred_cid: continue
                    for gbox in gt_by_class[gt_cid]:
                        if iou_xyxy(pbox, gbox) > 0.3:
                            confusion_matrix[gt_cid][pred_cid] += 1
                            break

        per_image_errors.append((str(img_path), img_fp_total, img_fn_total, img_class_errors))

        for cid in class_names.keys():
            gt_present = len(gt_by_class.get(cid, [])) > 0
            pred_present = len(pred_by_class_xyxy.get(cid, [])) > 0
            if not gt_present and not pred_present:
                img_tn_per_class[cid] += 1
            elif not gt_present and pred_present:
                img_fp_per_class[cid] += 1

        if idx % 200 == 0 or idx == total_images:
            print(f"  [{split_name}] {idx}/{total_images} images... preds={total_preds}")

    # ── Print report ──
    print(f"\n{'=' * 80}")
    print(f"  RESULTS: {split_name.upper()}")
    print(f"{'=' * 80}")

    TP = sum(tp_per_class.values())
    FP = sum(fp_per_class.values())
    FN = sum(fn_per_class.values())
    GT_TOTAL = sum(gt_count_per_class.values())
    P, R, F1 = prf(TP, FP, FN)

    print(f"\n  OVERALL")
    print(f"  {'─' * 50}")
    print(f"  Total images:       {total_images}")
    print(f"  Total GT:           {GT_TOTAL}")
    print(f"  Total predictions:  {total_preds}")
    print(f"  Avg confidence:     {safe_mean(all_confs):.4f}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision={P:.4f}  Recall={R:.4f}  F1={F1:.4f}")
    print(f"  FP/image avg:       {FP/total_images:.2f}")

    print(f"\n  PER-CLASS SUMMARY")
    print(f"  {'─' * 88}")
    print(f"  {'cid':>3} {'name':<12} {'GT':>6} {'Pred':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TP/GT%':>7} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─' * 88}")
    for cid in sorted(class_names.keys()):
        name = class_names.get(cid, f"c{cid}")
        gt = gt_count_per_class.get(cid, 0)
        pred = pred_count_per_class.get(cid, 0)
        tp = tp_per_class.get(cid, 0)
        fp = fp_per_class.get(cid, 0)
        fn = fn_per_class.get(cid, 0)
        if gt == 0 and pred == 0: continue
        p, r, f1 = prf(tp, fp, fn)
        tp_gt = (100*tp/gt) if gt > 0 else 0
        print(f"  {cid:>3} {name:<12} {gt:>6} {pred:>6} {tp:>5} {fp:>5} {fn:>5} {tp_gt:>6.1f}% {p:>6.3f} {r:>6.3f} {f1:>6.3f}")

    print(f"\n  IMAGE-LEVEL SPECIFICITY")
    print(f"  {'─' * 55}")
    print(f"  {'cid':>3} {'name':<12} {'TN_img':>6} {'FP_img':>6} {'Specificity':>11}")
    for cid in sorted(class_names.keys()):
        tn = img_tn_per_class.get(cid, 0)
        fp_img = img_fp_per_class.get(cid, 0)
        if tn == 0 and fp_img == 0: continue
        spec = tn/(tn+fp_img) if (tn+fp_img)>0 else 0
        print(f"  {cid:>3} {class_names[cid]:<12} {tn:>6} {fp_img:>6} {spec:>11.4f}")

    print(f"\n  CONFIDENCE DISTRIBUTION")
    print(f"  {'─' * 30}")
    ordered_bins = ["1.0-0.9"] + [f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}" for i in range(len(BIN_EDGES)-1)]
    for b in ordered_bins:
        if b in bin_counts:
            pct = 100*bin_counts[b]/total_preds if total_preds > 0 else 0
            print(f"  {b:<8}: {bin_counts[b]:>6} ({pct:>5.1f}%)")

    print(f"\n  SIZE-STRATIFIED TP/FP/FN")
    print(f"  {'─' * 75}")
    print(f"  {'Class':<12} {'Size':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for cid in sorted(class_names.keys()):
        name = class_names.get(cid, f"c{cid}")
        for size in ['small', 'medium', 'large']:
            m = size_metrics[cid][size]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            if tp==0 and fp==0 and fn==0: continue
            p, r, f1 = prf(tp, fp, fn)
            print(f"  {name:<12} {size:<8} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
    print(f"  {'─' * 75}")
    for size in ['small', 'medium', 'large']:
        tp = sum(size_metrics[c][size]['tp'] for c in size_metrics)
        fp = sum(size_metrics[c][size]['fp'] for c in size_metrics)
        fn = sum(size_metrics[c][size]['fn'] for c in size_metrics)
        p, r, f1 = prf(tp, fp, fn)
        print(f"  {'ALL':<12} {size:<8} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")

    print(f"\n  TP vs FP CONFIDENCE")
    print(f"  {'─' * 60}")
    print(f"  {'Class':<12} {'TP mean':>8} {'TP med':>8} {'FP mean':>8} {'FP med':>8} {'Gap':>8}")
    for cid in sorted(class_names.keys()):
        name = class_names.get(cid, f"c{cid}")
        tc = tp_confs_per_class.get(cid, [])
        fc = fp_confs_per_class.get(cid, [])
        tm, tmd = (np.mean(tc), np.median(tc)) if tc else (0, 0)
        fm, fmd = (np.mean(fc), np.median(fc)) if fc else (0, 0)
        print(f"  {name:<12} {tm:>8.3f} {tmd:>8.3f} {fm:>8.3f} {fmd:>8.3f} {tm-fm:>+8.3f}")

    all_tp_c = [c for cs in tp_confs_per_class.values() for c in cs]
    all_fp_c = [c for cs in fp_confs_per_class.values() for c in cs]
    if all_fp_c:
        hi_fp = sum(1 for c in all_fp_c if c > 0.5)
        print(f"  High-conf FPs (>0.5): {hi_fp} / {len(all_fp_c)} ({hi_fp/len(all_fp_c)*100:.1f}%)")

    print(f"\n  CROSS-CLASS CONFUSION (GT=row, Pred=col)")
    print(f"  {'─' * (12 + 10*len(class_names))}")
    print(f"  {'GT\\Pred':<12}", end="")
    for cid in sorted(class_names.keys()):
        print(f"{class_names[cid]:>10}", end="")
    print()
    for gt_cid in sorted(class_names.keys()):
        print(f"  {class_names[gt_cid]:<12}", end="")
        for pred_cid in sorted(class_names.keys()):
            if gt_cid == pred_cid:
                print(f"{'--':>10}", end="")
            else:
                print(f"{confusion_matrix[gt_cid].get(pred_cid,0):>10}", end="")
        print()

    print(f"\n  FP SPATIAL DISTRIBUTION")
    print(f"  {'─' * 75}")
    for cid in sorted(class_names.keys()):
        locs = fp_locations.get(cid, [])
        if not locs: continue
        cxs = [l[0] for l in locs]; cys = [l[1] for l in locs]
        center = sum(1 for cx, cy in locs if 0.25<cx<0.75 and 0.25<cy<0.75)
        edge = len(locs) - center
        print(f"  {class_names[cid]:<12}: {len(locs)} FPs | center={center}({center/len(locs)*100:.0f}%) "
              f"edge={edge}({edge/len(locs)*100:.0f}%) | "
              f"cx={np.mean(cxs):.2f}+-{np.std(cxs):.2f} cy={np.mean(cys):.2f}+-{np.std(cys):.2f}")

    print(f"\n  TOP {SAVE_WORST_N} WORST IMAGES")
    print(f"  {'─' * 65}")
    per_image_errors.sort(key=lambda x: x[1]+x[2], reverse=True)
    print(f"  {'#':>3} {'FP':>4} {'FN':>4} {'Tot':>5} {'Image'}")
    for i, (ip, fpc, fnc, ce) in enumerate(per_image_errors[:SAVE_WORST_N]):
        print(f"  {i+1:>3} {fpc:>4} {fnc:>4} {fpc+fnc:>5} {Path(ip).name}")
        for cid, errs in sorted(ce.items()):
            if errs['fp']>0 or errs['fn']>0:
                print(f"        {class_names.get(cid, f'c{cid}')}: tp={errs['tp']} fp={errs['fp']} fn={errs['fn']}")

    # Save worst images
    if SAVE_WORST_N > 0:
        colors_map = {'tp': (0,255,0), 'fp': (0,0,255), 'fn': (0,165,255)}
        print(f"\n  Saving worst images to {split_output}...")
        for i, (ip, fpc, fnc, _) in enumerate(per_image_errors[:SAVE_WORST_N]):
            img = cv2.imread(ip)
            if img is None: continue
            ih, iw = img.shape[:2]
            lp = Path(labels_dir) / f"{Path(ip).stem}.txt"
            gt_vis = load_gt_yolo_labels(lp, iw, ih)
            im_inf = preprocess_image(img)
            res = model.predict(source=im_inf, conf=CONF_THRES, iou=IOU_NMS, verbose=False)
            r = res[0]
            for cid, gboxes in gt_vis.items():
                for gbox in gboxes:
                    x1,y1,x2,y2 = [int(v) for v in gbox]
                    cv2.rectangle(img, (x1,y1), (x2,y2), colors_map['fn'], 2)
                    cv2.putText(img, f"GT:{class_names.get(cid,'?')}", (x1,max(0,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_map['fn'], 1)
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    conf = float(box.conf[0]); cid = int(box.cls[0])
                    is_tp = any(iou_xyxy([x1,y1,x2,y2], g) >= IOU_MATCH for g in gt_vis.get(cid, []))
                    tag = "TP" if is_tp else "FP"
                    col = colors_map['tp'] if is_tp else colors_map['fp']
                    cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(img, f"{tag}:{class_names.get(cid,'?')} {conf:.2f}", (x1,max(0,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.putText(img, f"FP={fpc} FN={fnc}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imwrite(str(split_output / f"worst_{i+1:02d}_{Path(ip).name}"), img)
        print(f"  Saved {min(SAVE_WORST_N, len(per_image_errors))} images.")

    # Return summary dict for cross-split comparison
    return {
        'split': split_name,
        'total_images': total_images,
        'total_gt': GT_TOTAL,
        'total_preds': total_preds,
        'TP': TP, 'FP': FP, 'FN': FN,
        'precision': P, 'recall': R, 'f1': F1,
        'avg_conf': safe_mean(all_confs),
        'fp_per_image': FP / total_images if total_images > 0 else 0,
        'gt_per_class': dict(gt_count_per_class),
        'tp_per_class': dict(tp_per_class),
        'fp_per_class': dict(fp_per_class),
        'fn_per_class': dict(fn_per_class),
        'size_metrics': {c: {s: dict(v) for s, v in sm.items()} for c, sm in size_metrics.items()},
        'tp_confs': {c: list(v) for c, v in tp_confs_per_class.items()},
        'fp_confs': {c: list(v) for c, v in fp_confs_per_class.items()},
        'high_conf_fp_count': sum(1 for c in all_fp_c if c > 0.5),
        'total_fp_count': len(all_fp_c),
    }


# =========================
# Cross-split comparison
# =========================

def print_comparison(summaries, class_names):
    """Print side-by-side comparison of valid vs test."""
    print(f"\n\n{'=' * 80}")
    print(f"  CROSS-SPLIT COMPARISON: VALID vs TEST")
    print(f"{'=' * 80}")

    splits = list(summaries.keys())
    if len(splits) < 2:
        print("  Need both valid and test to compare.")
        return

    # Overall comparison
    print(f"\n  OVERALL METRICS")
    print(f"  {'─' * 65}")
    print(f"  {'Metric':<25}", end="")
    for s in splits:
        print(f"{s.upper():>15}", end="")
    print(f"{'Delta':>12} {'%Change':>10}")
    print(f"  {'─' * 65}")

    metrics = [
        ('Total images', 'total_images'),
        ('Total GT', 'total_gt'),
        ('Total predictions', 'total_preds'),
        ('TP', 'TP'),
        ('FP', 'FP'),
        ('FN', 'FN'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1', 'f1'),
        ('Avg confidence', 'avg_conf'),
        ('FP/image', 'fp_per_image'),
        ('High-conf FPs (>0.5)', 'high_conf_fp_count'),
    ]

    for label, key in metrics:
        vals = [summaries[s].get(key, 0) for s in splits]
        print(f"  {label:<25}", end="")
        for v in vals:
            if isinstance(v, float):
                print(f"{v:>15.4f}", end="")
            else:
                print(f"{v:>15}", end="")
        if len(vals) >= 2 and vals[0] and isinstance(vals[0], (int, float)):
            delta = vals[1] - vals[0]
            pct = (delta / vals[0] * 100) if vals[0] != 0 else 0
            if isinstance(delta, float):
                print(f"{delta:>+12.4f} {pct:>+9.1f}%", end="")
            else:
                print(f"{delta:>+12} {pct:>+9.1f}%", end="")
        print()

    # Per-class comparison
    print(f"\n  PER-CLASS PRECISION COMPARISON")
    print(f"  {'─' * 55}")
    print(f"  {'Class':<12}", end="")
    for s in splits:
        print(f"{'P_'+s:>10} {'R_'+s:>10}", end="")
    print()
    for cid in sorted(class_names.keys()):
        name = class_names[cid]
        print(f"  {name:<12}", end="")
        for s in splits:
            sm = summaries[s]
            tp = sm['tp_per_class'].get(cid, 0)
            fp = sm['fp_per_class'].get(cid, 0)
            fn = sm['fn_per_class'].get(cid, 0)
            p, r, _ = prf(tp, fp, fn)
            print(f"{p:>10.3f} {r:>10.3f}", end="")
        print()

    # Size comparison
    print(f"\n  SIZE-STRATIFIED PRECISION COMPARISON")
    print(f"  {'─' * 55}")
    print(f"  {'Size':<10}", end="")
    for s in splits:
        print(f"{'P_'+s:>10} {'R_'+s:>10}", end="")
    print()
    for size in ['small', 'medium', 'large']:
        print(f"  {size:<10}", end="")
        for s in splits:
            sm = summaries[s]
            tp = sum(sm['size_metrics'].get(c, {}).get(size, {}).get('tp', 0) for c in sm['size_metrics'])
            fp = sum(sm['size_metrics'].get(c, {}).get(size, {}).get('fp', 0) for c in sm['size_metrics'])
            fn = sum(sm['size_metrics'].get(c, {}).get(size, {}).get('fn', 0) for c in sm['size_metrics'])
            p, r, _ = prf(tp, fp, fn)
            print(f"{p:>10.3f} {r:>10.3f}", end="")
        print()

    # Key findings
    print(f"\n  KEY FINDINGS")
    print(f"  {'─' * 55}")
    v = summaries.get('valid', {})
    t = summaries.get('test', {})
    if v and t:
        p_drop = v['precision'] - t['precision']
        r_drop = v['recall'] - t['recall']
        fp_ratio = t['fp_per_image'] / v['fp_per_image'] if v['fp_per_image'] > 0 else 0

        if p_drop > 0.05:
            print(f"  !! PRECISION drops {p_drop:.3f} ({p_drop/v['precision']*100:.1f}%) from valid to test")
        if r_drop > 0.05:
            print(f"  !! RECALL drops {r_drop:.3f} ({r_drop/v['recall']*100:.1f}%) from valid to test")
        if fp_ratio > 1.5:
            print(f"  !! FP/image is {fp_ratio:.1f}x higher on test ({t['fp_per_image']:.2f}) vs valid ({v['fp_per_image']:.2f})")

        # Which class has biggest precision drop?
        worst_class = None
        worst_drop = 0
        for cid in class_names:
            vtp = v['tp_per_class'].get(cid, 0); vfp = v['fp_per_class'].get(cid, 0)
            ttp = t['tp_per_class'].get(cid, 0); tfp = t['fp_per_class'].get(cid, 0)
            vp = vtp/(vtp+vfp) if (vtp+vfp)>0 else 0
            tp_t = ttp/(ttp+tfp) if (ttp+tfp)>0 else 0
            drop = vp - tp_t
            if drop > worst_drop:
                worst_drop = drop
                worst_class = class_names[cid]
        if worst_class:
            print(f"  !! Worst precision drop: {worst_class} (-{worst_drop:.3f})")

    print()


# =========================
# Main
# =========================

def main():
    # Set up tee output
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    tee = TeeWriter(REPORT_FILE)
    sys.stdout = tee

    print(f"{'=' * 80}")
    print(f"  INFERENCE IMAGE STATISTICS — MULTI-SPLIT ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Dataset:     {DATASET_ROOT}")
    print(f"  Splits:      {', '.join(EVAL_SPLITS.keys())}")
    print(f"  CONF={CONF_THRES} IOU_NMS={IOU_NMS} IOU_MATCH={IOU_MATCH}")
    print(f"  PREPROCESS={PREPROCESS}")
    print(f"  Report file: {REPORT_FILE}")
    print(f"{'=' * 80}")

    model = YOLO(MODEL_PATH)

    if isinstance(model.names, dict):
        class_names = {int(k): v for k, v in model.names.items()}
    else:
        class_names = {i: n for i, n in enumerate(model.names)}

    summaries = {}

    for split_name, split_info in EVAL_SPLITS.items():
        result = evaluate_split(
            model, class_names, split_name,
            split_info['images'], split_info['labels'],
            OUTPUT_ROOT
        )
        if result:
            summaries[split_name] = result

    # Cross-split comparison
    if len(summaries) >= 2:
        print_comparison(summaries, class_names)

    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Report saved to: {REPORT_FILE}")
    print(f"  Worst images saved to: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    # Restore stdout
    sys.stdout = tee.stdout
    tee.close()
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
