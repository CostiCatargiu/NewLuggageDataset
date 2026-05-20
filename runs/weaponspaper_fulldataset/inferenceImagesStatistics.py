"""
YOLO Inference + Detection Metrics (Ultralytics) with TP/FP/FN + (image-level) TN

Adds (requires GT labels in YOLO format):
- TP / FP / FN via IoU matching (per class)
- TN computed at IMAGE LEVEL per class:
    TN_c = count(images with no GT of class c AND no predictions of class c)

Also reports:
- precision, recall, f1 (from TP/FP/FN)
- specificity (from image-level TN and image-level FP)

NEW:
- Optional preprocessing BEFORE detection:
  - none
  - CLAHE (adaptive equalization)
  - global histogram equalization (on luminance / Y channel)

EXTRA (requested):
- Count GT annotations per class (how many labeled boxes exist)
- Count predicted boxes per class
- Count "detected well" per class = TP (IoU-matched)
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG (hardcoded)
# =========================
# MODEL_PATH = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_custom_weapon/yolov12s_custom_train4/weights/best.pt"
MODEL_PATH = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_new_weapon_dataset_full/swa_09_05_fulldataset/weights/best.pt"
# TEST_IMAGES_DIR = r"//home/constantin/Downloads/WeSecure.v2-wesecure.yolov12/train/images"
TEST_IMAGES_DIR = r"/home/constantin/Doctorat/GunDatasetHistogram/test/images"

# Labels folder: assumes /test/labels mirrors /test/images (or your export layout)
# TEST_LABELS_DIR = str(Path(TEST_IMAGES_DIR).parent / "labels")
TEST_LABELS_DIR = '/home/constantin/Doctorat/GunDatasetHistogram/test/labels'
# Inference thresholds
CONF_THRES = 0.1
IOU_NMS = 0.60

# IoU threshold for considering a prediction a TP (matching)
IOU_MATCH = 0.50

# -------------------------
# PREPROCESSING
# -------------------------
# Options: "none", "clahe", "histeq"
PREPROCESS = "none"  # <- set to "clahe" or "histeq"

# If using CLAHE:
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Image extensions to include
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Confidence bins: 0.9–0.8, 0.8–0.7, ... 0.1–0.0
BIN_EDGES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


# =========================
# Helpers
# =========================
def list_images(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"TEST_IMAGES_DIR not found: {folder}")
    files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def bin_label(conf: float) -> str:
    for i in range(len(BIN_EDGES) - 1):
        hi = BIN_EDGES[i]
        lo = BIN_EDGES[i + 1]
        if conf < hi and conf >= lo:
            return f"{hi:.1f}–{lo:.1f}"
    if conf >= 0.9:
        return "1.0–0.9"
    return "0.1–0.0"


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def yolo_xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    """YOLO normalized xywh -> pixel xyxy"""
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    """IoU between boxes a and b in xyxy (floats)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 1e-9 else 0.0


def load_gt_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """
    Returns gt_by_class: dict[int, list[xyxy]]
    Label lines: class xc yc w h  (normalized)
    """
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
        box = yolo_xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
        gt_by_class[cid].append(box)

    return gt_by_class


def match_preds_to_gt(preds_xyxy, preds_conf, gt_xyxy, iou_thr):
    """
    Greedy matching by descending confidence.
    Returns: tp_count, fp_count, fn_count
    """
    if len(preds_xyxy) == 0 and len(gt_xyxy) == 0:
        return 0, 0, 0

    order = np.argsort(-np.array(preds_conf)) if len(preds_conf) else np.array([], dtype=int)
    preds_sorted = [preds_xyxy[i] for i in order]
    conf_sorted = [preds_conf[i] for i in order]

    gt_used = [False] * len(gt_xyxy)
    tp = 0
    fp = 0

    for pbox, _ in zip(preds_sorted, conf_sorted):
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_xyxy):
            if gt_used[j]:
                continue
            iou = iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            gt_used[best_j] = True
        else:
            fp += 1

    fn = sum(1 for used in gt_used if not used)
    return tp, fp, fn


# -------------------------
# PREPROCESSING FUNCTIONS
# -------------------------
def apply_clahe_bgr(bgr: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on LAB-L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_histeq_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization on luminance only (Y channel) to avoid color artifacts.
    Uses YCrCb (OpenCV standard).
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y2 = cv2.equalizeHist(y)
    ycrcb2 = cv2.merge((y2, cr, cb))
    return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)


def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    mode = (PREPROCESS or "none").lower().strip()
    if mode == "none":
        return bgr
    if mode == "clahe":
        return apply_clahe_bgr(bgr, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE)
    if mode == "histeq":
        return apply_histeq_bgr(bgr)
    raise ValueError(f"Unknown PREPROCESS mode: {PREPROCESS} (use: none/clahe/histeq)")


# =========================
# Main
# =========================
def main():
    model = YOLO(MODEL_PATH)

    # Resolve class names
    if isinstance(model.names, dict):
        class_names = {int(k): v for k, v in model.names.items()}
    else:
        class_names = {i: n for i, n in enumerate(model.names)}

    images = list_images(TEST_IMAGES_DIR)
    if not images:
        print(f"No images found in: {TEST_IMAGES_DIR}")
        return

    labels_dir = Path(TEST_LABELS_DIR)
    if not labels_dir.exists():
        print(f"⚠️ Labels dir not found: {labels_dir}")
        print("   TP/FP/FN/TN metrics require GT label files. Please set TEST_LABELS_DIR correctly.\n")

    # --- counts / confidence stats ---
    pred_count_per_class = defaultdict(int)       # predicted boxes per class
    confs_per_class = defaultdict(list)
    bin_counts = defaultdict(int)
    all_confs = []
    total_preds = 0

    # --- GT annotation counts ---
    gt_count_per_class = defaultdict(int)         # GT boxes per class (annotations)

    # --- detection metrics ---
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)

    # --- image-level specificity proxy ---
    img_tn_per_class = defaultdict(int)
    img_fp_per_class = defaultdict(int)

    total_images = len(images)

    print(f"Model: {MODEL_PATH}")
    print(f"Images: {total_images} from {TEST_IMAGES_DIR}")
    print(f"Labels: {labels_dir}")
    print(f"CONF_THRES={CONF_THRES}, IOU_NMS={IOU_NMS}, IOU_MATCH={IOU_MATCH}")
    print(f"PREPROCESS={PREPROCESS} (clahe clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_GRID_SIZE})")
    print("-" * 60)

    # Inference loop
    for idx, img_path in enumerate(images, start=1):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        img_h, img_w = im.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_by_class = load_gt_yolo_labels(label_path, img_w, img_h)

        # NEW: count GT annotations per class
        for cid, gt_boxes in gt_by_class.items():
            gt_count_per_class[cid] += len(gt_boxes)

        # preprocessing before inference
        im_for_infer = preprocess_image(im)

        results = model.predict(
            source=im_for_infer,
            conf=CONF_THRES,
            iou=IOU_NMS,
            verbose=False
        )
        r = results[0]
        boxes = r.boxes

        # Build pred_by_class (xyxy + conf) for matching
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

        # ----- TP/FP/FN per class via IoU matching -----
        classes_in_img = set(gt_by_class.keys()) | set(pred_by_class_xyxy.keys())
        for cid in classes_in_img:
            gt_boxes = gt_by_class.get(cid, [])
            pr_boxes = pred_by_class_xyxy.get(cid, [])
            pr_confs = pred_by_class_conf.get(cid, [])

            tp, fp, fn = match_preds_to_gt(pr_boxes, pr_confs, gt_boxes, IOU_MATCH)
            tp_per_class[cid] += tp
            fp_per_class[cid] += fp
            fn_per_class[cid] += fn

        # ----- Image-level TN / image-level FP (specificity proxy) -----
        for cid in class_names.keys():
            gt_present = len(gt_by_class.get(cid, [])) > 0
            pred_present = len(pred_by_class_xyxy.get(cid, [])) > 0

            if (not gt_present) and (not pred_present):
                img_tn_per_class[cid] += 1
            elif (not gt_present) and pred_present:
                img_fp_per_class[cid] += 1

        if idx % 50 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images} images... total preds so far: {total_preds}")

    # =========================
    # Reporting
    # =========================
    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    print("\n" + "=" * 60)
    print("OVERALL (DETECTIONS)")
    print("=" * 60)
    overall_avg_conf = safe_mean(all_confs)

    TP = sum(tp_per_class.values())
    FP = sum(fp_per_class.values())
    FN = sum(fn_per_class.values())
    GT_TOTAL = sum(gt_count_per_class.values())

    P, R, F1 = prf(TP, FP, FN)

    print(f"Total images:                      {total_images}")
    print(f"Total GT annotations (all classes): {GT_TOTAL}")
    print(f"Total predictions (all classes):    {total_preds}")
    print(f"Overall average confidence:         {overall_avg_conf:.4f}")
    print(f"TP={TP}  FP={FP}  FN={FN}")
    print(f"Precision={P:.4f}  Recall={R:.4f}  F1={F1:.4f}")

    print("\n" + "=" * 90)
    print("PER-CLASS SUMMARY (GT annotations, predictions, and correct detections)")
    print("=" * 90)
    print("cid  name                 GT      Pred    TP     FP     FN     TP/GT%   Prec    Rec     F1")
    print("-" * 90)

    for cid in sorted(class_names.keys()):
        name = class_names.get(cid, f"class_{cid}")

        gt = gt_count_per_class.get(cid, 0)
        pred = pred_count_per_class.get(cid, 0)

        tp = tp_per_class.get(cid, 0)
        fp = fp_per_class.get(cid, 0)
        fn = fn_per_class.get(cid, 0)

        if gt == 0 and pred == 0 and tp == 0 and fp == 0 and fn == 0:
            continue

        prec, rec, f1 = prf(tp, fp, fn)
        tp_over_gt = (100.0 * tp / gt) if gt > 0 else 0.0

        print(f"{cid:>3}  {name:<20}  {gt:>6}  {pred:>6}  {tp:>5}  {fp:>5}  {fn:>5}  {tp_over_gt:>7.2f}%  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")

    print("\n" + "=" * 60)
    print("PER-CLASS IMAGE-LEVEL TN/FP (Specificity proxy)")
    print("=" * 60)
    print("cid  name                 TN_img  FP_img  Specificity")
    print("-" * 60)
    for cid in sorted(class_names.keys()):
        tn = img_tn_per_class.get(cid, 0)
        fp_img = img_fp_per_class.get(cid, 0)
        if tn == 0 and fp_img == 0:
            continue
        spec = tn / (tn + fp_img) if (tn + fp_img) > 0 else 0.0
        name = class_names.get(cid, f"class_{cid}")
        print(f"{cid:>3}  {name:<20}  {tn:>6}  {fp_img:>6}  {spec:>10.4f}")

    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVAL DISTRIBUTION")
    print("=" * 60)
    ordered_bins = ["1.0–0.9"] + [f"{BIN_EDGES[i]:.1f}–{BIN_EDGES[i+1]:.1f}" for i in range(len(BIN_EDGES) - 1)]
    for b in ordered_bins:
        if b in bin_counts:
            print(f"{b:<8} : {bin_counts[b]}")

    if total_preds > 0:
        print("\nPercentages:")
        for b in ordered_bins:
            if b in bin_counts:
                pct = 100.0 * bin_counts[b] / total_preds
                print(f"{b:<8} : {pct:6.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
