#!/usr/bin/env python3
"""
Pseudo-label the under-annotated "other" class on the TRAINING set.

WHY: "other" objects are present but not annotated in many training images, so
the model is penalized as a false positive for correctly detecting them -> the
"other" class is trained on a contradictory signal (high recall, low precision).
This script uses a trained model to fill in the MISSING "other" boxes: it runs
inference on the train images, keeps the confident "other" detections that don't
overlap any existing GT box, and writes AUGMENTED label files (originals + the
new pseudo boxes). Retraining on these removes the contradictory signal.

SAFETY / SCOPE (read this):
  * TRAIN SPLIT ONLY. Never run this on val/test and evaluate on it -- scoring a
    model against its own predictions is circular and invalid. For an honest test
    metric, hand-complete a small held-out test subset instead.
  * Originals are NEVER overwritten. Augmented labels go to a NEW directory.
  * Only the "other" class is pseudo-labeled; the well-annotated weapon classes
    are left untouched.
  * High confidence threshold + IoU gate against existing GT avoid injecting
    false positives as "ground truth". Tune CONF_THRES and inspect the stats.

USAGE:
  1. Set MODEL_WEIGHTS to your trained checkpoint (e.g. a good r21 best.pt).
  2. Set DATA_YAML to the dataset yaml.
  3. python make_pseudo_labels_other.py
  4. Review the printed stats. If sensible, activate the pseudo labels by pointing
     training at OUT_LABELS_DIR (instructions printed at the end). Re-train, then
     optionally iterate (regenerate with the improved model).
"""

import os
import glob
import shutil

import yaml
import numpy as np
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================
MODEL_WEIGHTS = "runs_noaug_weapon_full/r21_arch_full/weights/best.pt"  # <-- your trained model
DATA_YAML     = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
OTHER_NAME    = "other"     # class name to pseudo-label (resolved to an id from data.yaml)

CONF_THRES    = 0.55        # keep only "other" detections at/above this confidence
IOU_SKIP      = 0.30        # skip a detection if it overlaps ANY existing GT box by >= this
                            #   (region already annotated -> don't add a duplicate/conflict)
MIN_BOX_FRAC  = 0.0008      # drop tiny specks (box area < this fraction of the image); 0 to disable
MAX_PER_IMAGE = 20          # safety cap on pseudo boxes added per image
IMGSZ         = 640
DEVICE        = 0

OUT_LABELS_DIR = "pseudo_labels_other/train"   # augmented labels written here (originals preserved)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =============================================================================
# HELPERS
# =============================================================================
def resolve_other_id(names):
    """names may be a list or an id->name dict; return the integer id of OTHER_NAME."""
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    for i, n in items:
        if str(n).strip().lower() == OTHER_NAME.lower():
            return int(i)
    raise ValueError(f"class '{OTHER_NAME}' not found in data.yaml names: {names}")


def list_train_images(data):
    """Resolve the train image paths from a YOLO data.yaml (dir, txt list, or list)."""
    root = data.get("path", "")
    train = data["train"]
    entries = train if isinstance(train, list) else [train]
    images = []
    for e in entries:
        p = e if os.path.isabs(e) else os.path.join(root, e)
        if os.path.isdir(p):
            for ext in IMG_EXTS:
                images += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
        elif p.endswith(".txt") and os.path.isfile(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ip = line if os.path.isabs(line) else os.path.join(root, line)
                    images.append(ip)
        elif os.path.isfile(p):
            images.append(p)
    return sorted(set(images))


def img_to_label_path(img_path):
    """YOLO convention: .../images/... -> .../labels/..., extension -> .txt."""
    parts = img_path.replace("\\", "/").rsplit("/images/", 1)
    if len(parts) == 2:
        lbl = parts[0] + "/labels/" + parts[1]
    else:
        lbl = os.path.splitext(img_path)[0]
    return os.path.splitext(lbl)[0] + ".txt"


def read_gt_xyxy(label_path):
    """Read existing GT (any class) as normalized xyxy boxes. Returns (N,4) array."""
    if not os.path.isfile(label_path):
        return np.zeros((0, 4), dtype=np.float32), []
    lines = [l.strip() for l in open(label_path) if l.strip()]
    boxes = []
    for l in lines:
        f = l.split()
        if len(f) < 5:
            continue
        cx, cy, w, h = map(float, f[1:5])
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return (np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)), lines


def iou_matrix(a, b):
    """IoU between boxes a (N,4) and b (M,4), all xyxy normalized. Returns (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    other_id = resolve_other_id(data["names"])
    images = list_train_images(data)
    os.makedirs(OUT_LABELS_DIR, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  PSEUDO-LABELING 'other' (id={other_id}) ON TRAIN")
    print(f"  model:  {MODEL_WEIGHTS}")
    print(f"  images: {len(images)}   conf>={CONF_THRES}   skip if IoU(GT)>= {IOU_SKIP}")
    print(f"  out:    {OUT_LABELS_DIR}  (originals preserved)")
    print(f"{'=' * 70}")
    if not images:
        print("  No train images resolved -- check DATA_YAML paths. Aborting.")
        return

    model = YOLO(MODEL_WEIGHTS)

    n_added, n_imgs_aug, n_imgs_total = 0, 0, 0
    per_image_added = []

    # Batched inference over the train images.
    BATCH = 64
    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        results = model.predict(batch, conf=CONF_THRES, imgsz=IMGSZ, device=DEVICE,
                                 verbose=False, stream=False)
        for img_path, r in zip(batch, results):
            n_imgs_total += 1
            label_path = img_to_label_path(img_path)
            gt_xyxy, orig_lines = read_gt_xyxy(label_path)

            # model's "other" detections, normalized xywh -> xyxy
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xywhn = r.boxes.xywhn.cpu().numpy()
            keep = cls == other_id
            det = xywhn[keep]

            pseudo_lines = []
            if len(det):
                det_xyxy = np.stack([det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2,
                                     det[:, 0] + det[:, 2] / 2, det[:, 1] + det[:, 3] / 2], axis=1)
                ious = iou_matrix(det_xyxy, gt_xyxy)            # (n_det, n_gt)
                max_iou = ious.max(axis=1) if gt_xyxy.shape[0] else np.zeros(len(det))
                areas = det[:, 2] * det[:, 3]
                for j in range(len(det)):
                    if max_iou[j] >= IOU_SKIP:          # region already annotated -> skip
                        continue
                    if areas[j] < MIN_BOX_FRAC:          # speck -> skip
                        continue
                    cx, cy, w, h = det[j]
                    pseudo_lines.append(f"{other_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    if len(pseudo_lines) >= MAX_PER_IMAGE:
                        break

            # write augmented label file (originals + pseudo), originals untouched on disk
            out_path = os.path.join(OUT_LABELS_DIR, os.path.basename(label_path))
            with open(out_path, "w") as f:
                if orig_lines:
                    f.write("\n".join(orig_lines))
                    if pseudo_lines:
                        f.write("\n")
                if pseudo_lines:
                    f.write("\n".join(pseudo_lines) + "\n")

            if pseudo_lines:
                n_imgs_aug += 1
                n_added += len(pseudo_lines)
                per_image_added.append(len(pseudo_lines))

        print(f"  ...{n_imgs_total}/{len(images)} images  |  +{n_added} pseudo boxes so far")

    # -------- stats --------
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"  images processed:         {n_imgs_total}")
    print(f"  images given pseudo boxes:{n_imgs_aug}  ({100*n_imgs_aug/max(n_imgs_total,1):.1f}%)")
    print(f"  total pseudo 'other' boxes added: {n_added}")
    if per_image_added:
        a = np.array(per_image_added)
        print(f"  per-augmented-image: mean {a.mean():.2f}  median {int(np.median(a))}  max {a.max()}")
    print(f"  augmented labels written to: {OUT_LABELS_DIR}")
    print(f"{'=' * 70}")
    print("\n  SANITY CHECK FIRST: open a handful of images whose pseudo boxes were added")
    print("  and confirm the new boxes are real, previously-unlabeled 'other' objects.")
    print("  If you see many boxes on non-objects -> raise CONF_THRES and re-run.\n")
    print("  TO TRAIN ON THESE (originals stay safe):")
    print("    1) back up the real train labels dir:  mv .../labels/train .../labels/train_orig")
    print(f"    2) put the pseudo labels in its place: cp -r {OUT_LABELS_DIR} .../labels/train")
    print("    3) retrain; to revert, swap train_orig back.")
    print("  (Do NOT pseudo-label or evaluate on val/test -- that is circular.)\n")


if __name__ == "__main__":
    main()
