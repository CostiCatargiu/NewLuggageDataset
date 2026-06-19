#!/usr/bin/env python3
"""
Pseudo-label / candidate-generate the under-annotated "other" class — WHOLE dataset.

WHY: "other" objects are present but not always annotated, so the model is
penalized as a false positive for correctly detecting them (high recall, low
precision). This runs a trained model over the dataset and proposes the MISSING
"other" boxes — the confident "other" detections that (a) don't overlap any
existing GT box and (b) aren't also predicted as a weapon (ambiguous → skipped).
It writes AUGMENTED label files (originals + new boxes), originals never touched.

CANDIDATE RULE (per detection), matching your spec:
  keep an "other" detection iff:
    - conf >= CONF_THRES
    - it does NOT overlap any existing GT box        (IoU < IOU_SKIP)  -> truly unlabeled
    - it does NOT overlap any weapon prediction       (IoU < WEAPON_CONFLICT_IOU)
      i.e. if the same instance is also predicted as a weapon, skip it (could be a
      misclassified weapon, not a real "other")
    - it is not a tiny speck                          (area >= MIN_BOX_FRAC)

SPLIT USAGE — READ THIS:
  * TRAIN: the augmented labels are safe to RETRAIN on (training tolerates some
    label noise). This is the automatic fix.
  * VAL / TEST: the augmented labels here are CANDIDATES FOR HUMAN REVIEW ONLY.
    Do NOT train-eval against auto-generated labels on val/test — scoring a model
    against its own predictions is circular and invalid. Open these in a labeling
    tool (CVAT / Label Studio / Roboflow), accept/reject by hand, THEN use them as
    your clean evaluation set.

Originals are never overwritten; everything goes under OUT_ROOT/<split>/.

USAGE:
  1. set MODEL_WEIGHTS to a trained checkpoint, DATA_YAML to the dataset yaml
  2. python make_pseudo_labels_other.py
  3. review the per-split stats + the manifest; eyeball some images
  4. TRAIN: swap in OUT_ROOT/train labels and retrain.
     VAL/TEST: human-review OUT_ROOT/{val,test} before any evaluation.
"""

import os
import csv
import glob

import yaml
import numpy as np
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================
MODEL_WEIGHTS = "runs_noaug_weapon_full/r21_arch_full/weights/best.pt"  # <-- your trained model
DATA_YAML     = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
OTHER_NAME    = "other"        # class to pseudo-label (resolved to an id from data.yaml names)
SPLITS        = ["train", "val", "test"]   # processed if present in the data.yaml

CONF_THRES          = 0.55     # keep "other" detections at/above this confidence
IOU_SKIP            = 0.30     # skip if it overlaps any EXISTING GT box  (already annotated)
WEAPON_CONFLICT_IOU = 0.40     # skip if it overlaps a WEAPON prediction  (ambiguous / misclassified)
WEAPON_CONFLICT_CONF= 0.25     # a weapon prediction this confident counts as a conflict
MIN_BOX_FRAC        = 0.0008   # drop specks (box area < this fraction of the image); 0 disables
MAX_PER_IMAGE       = 20       # safety cap on added boxes per image
IMGSZ               = 640
DEVICE              = 0

OUT_ROOT = "pseudo_labels_other"          # OUT_ROOT/<split>/<name>.txt  (augmented labels)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =============================================================================
# HELPERS
# =============================================================================
def resolve_other_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for i, n in items:
        if str(n).strip().lower() == OTHER_NAME.lower():
            return int(i)
    raise ValueError(f"class '{OTHER_NAME}' not found in data.yaml names: {names}")


def resolve_split_images(data, key):
    """Resolve image paths for a split key (dir, txt list, or list of either)."""
    if key not in data or data[key] in (None, ""):
        return []
    root = data.get("path", "")
    entries = data[key] if isinstance(data[key], list) else [data[key]]
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
                    if line:
                        images.append(line if os.path.isabs(line) else os.path.join(root, line))
        elif os.path.isfile(p):
            images.append(p)
    return sorted(set(images))


def img_to_label_path(img_path):
    parts = img_path.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img_path)[0]
    return os.path.splitext(lbl)[0] + ".txt"


def read_gt_xyxy(label_path):
    if not os.path.isfile(label_path):
        return np.zeros((0, 4), np.float32), []
    lines = [l.strip() for l in open(label_path) if l.strip()]
    boxes = []
    for l in lines:
        f = l.split()
        if len(f) >= 5:
            cx, cy, w, h = map(float, f[1:5])
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return (np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32)), lines


def xywhn_to_xyxy(d):
    return np.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2,
                     d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], axis=1)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def process_split(model, images, split, other_id):
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    n_added = n_imgs_aug = 0
    # predict low enough to also catch weak weapon predictions for the conflict check
    pred_conf = min(CONF_THRES, WEAPON_CONFLICT_CONF)

    BATCH = 64
    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        results = model.predict(batch, conf=pred_conf, imgsz=IMGSZ, device=DEVICE,
                                verbose=False, stream=False)
        for img_path, r in zip(batch, results):
            label_path = img_to_label_path(img_path)
            gt_xyxy, orig_lines = read_gt_xyxy(label_path)

            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xywhn = r.boxes.xywhn.cpu().numpy()

            other_sel = (cls == other_id) & (conf >= CONF_THRES)
            weapon_sel = (cls != other_id) & (conf >= WEAPON_CONFLICT_CONF)
            other = xywhn[other_sel]
            weapon_xyxy = xywhn_to_xyxy(xywhn[weapon_sel]) if weapon_sel.any() else np.zeros((0, 4), np.float32)

            pseudo = []
            if len(other):
                o_xyxy = xywhn_to_xyxy(other)
                iou_gt = iou_matrix(o_xyxy, gt_xyxy).max(1) if len(gt_xyxy) else np.zeros(len(other))
                iou_wp = iou_matrix(o_xyxy, weapon_xyxy).max(1) if len(weapon_xyxy) else np.zeros(len(other))
                areas = other[:, 2] * other[:, 3]
                for j in range(len(other)):
                    if iou_gt[j] >= IOU_SKIP:            # already annotated here
                        continue
                    if iou_wp[j] >= WEAPON_CONFLICT_IOU:  # also predicted as a weapon -> ambiguous
                        continue
                    if areas[j] < MIN_BOX_FRAC:           # speck
                        continue
                    cx, cy, w, h = other[j]
                    pseudo.append(f"{other_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    if len(pseudo) >= MAX_PER_IMAGE:
                        break

            out_path = os.path.join(out_dir, os.path.basename(label_path))
            with open(out_path, "w") as f:
                if orig_lines:
                    f.write("\n".join(orig_lines))
                    if pseudo:
                        f.write("\n")
                if pseudo:
                    f.write("\n".join(pseudo) + "\n")

            if pseudo:
                n_imgs_aug += 1
                n_added += len(pseudo)
                manifest.append((img_path, len(orig_lines), len(pseudo)))
        print(f"  [{split}] {min(start + BATCH, len(images))}/{len(images)}  +{n_added} candidates")

    # manifest of images that got candidates (handy for review / filtering)
    man_path = os.path.join(OUT_ROOT, f"manifest_{split}.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_existing_labels", "n_candidates_added"])
        w.writerows(manifest)

    return dict(split=split, images=len(images), imgs_aug=n_imgs_aug, added=n_added,
                out_dir=out_dir, manifest=man_path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    other_id = resolve_other_id(data["names"])
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("=" * 72)
    print(f"  PSEUDO-LABEL CANDIDATES for 'other' (id={other_id}) — WHOLE DATASET")
    print(f"  model: {MODEL_WEIGHTS}")
    print(f"  keep if conf>={CONF_THRES}, IoU(GT)<{IOU_SKIP}, IoU(weapon)<{WEAPON_CONFLICT_IOU}")
    print("=" * 72)

    model = YOLO(MODEL_WEIGHTS)
    summaries = []
    for split in SPLITS:
        imgs = resolve_split_images(data, split)
        if not imgs:
            print(f"  [{split}] not present / empty — skipped")
            continue
        summaries.append(process_split(model, imgs, split, other_id))

    print("\n" + "=" * 72)
    print("  DONE")
    for s in summaries:
        pct = 100 * s["imgs_aug"] / max(s["images"], 1)
        print(f"  {s['split']:<6} images={s['images']:<6} "
              f"images_with_candidates={s['imgs_aug']:<6}({pct:4.1f}%) "
              f"candidate_boxes={s['added']}")
        print(f"         labels -> {s['out_dir']}/    manifest -> {s['manifest']}")
    print("=" * 72)
    print("\n  USE:")
    print("   * TRAIN: swap OUT_ROOT/train in for your train labels (back up originals) and retrain.")
    print("   * VAL/TEST: these are CANDIDATES — human-review them (CVAT/Label Studio/Roboflow)")
    print("     before using as a clean eval set. NEVER evaluate against auto labels (circular).")
    print("   Sanity-check first: open a few train images and confirm the added boxes are real")
    print("   unlabeled 'other' objects. Too many on non-objects -> raise CONF_THRES and re-run.\n")


if __name__ == "__main__":
    main()
