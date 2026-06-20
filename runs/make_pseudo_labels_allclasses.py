#!/usr/bin/env python3
"""
Pseudo-label / candidate-generate MISSING annotations for ALL classes — whole dataset,
with candidates SPLIT PER CLASS so you can merge/review each class independently.

CANDIDATE RULE (per detection of any class c):
  keep it iff:
    - conf >= CONF_THRES
    - it does NOT overlap any existing GT box (any class)   (IoU < IOU_SKIP)
    - it does NOT overlap a DIFFERENT-class prediction       (IoU < CONFLICT_IOU
      with a box of another class at conf >= CONFLICT_CONF)
    - area >= MIN_BOX_FRAC
  the pseudo box is written with the DETECTED class id.

OUTPUTS (originals never touched):
  OUT_ROOT/<split>/<stem>.txt                       augmented = original + ALL candidates
  OUT_ROOT/by_class/<classname>/<split>/<stem>.txt  candidate boxes for THAT class only
  OUT_ROOT/manifest_<classname>_<split>.csv         per-class manifest (image, n_added)

So to use one class: merge OUT_ROOT/by_class/other into train; review the rest.
Well-annotated classes (weapons) should show very few candidates (mostly FPs at low
conf) — don't auto-merge those. NEVER auto-merge val/test for any class.
"""

import os
import csv
import glob
from collections import defaultdict

import yaml
import numpy as np
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================
MODEL_WEIGHTS = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_noaug_weapon_full/v5_tal07_loose_full/weights/best.pt"
DATA_YAML     = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
SPLITS        = ["train", "val", "test"]

CONF_THRES    = 0.15      # keep detections at/above this confidence (LOW -> noisy for clean classes)
IOU_SKIP      = 0.30      # skip if it overlaps any existing GT box (already annotated)
CONFLICT_IOU  = 0.40      # skip if it overlaps a DIFFERENT-class prediction (ambiguous)
CONFLICT_CONF = 0.15      # a different-class prediction this confident counts as a conflict
MIN_BOX_FRAC  = 0.0008    # drop specks (box area < this fraction of the image); 0 disables
MAX_PER_IMAGE = 40        # safety cap on added boxes per image (across all classes)
IMGSZ         = 640
DEVICE        = 0

OUT_ROOT = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_labels_allclasses"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =============================================================================
# HELPERS
# =============================================================================
def resolve_split_images(data, key):
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


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def process_split(model, images, split, names):
    aug_dir = os.path.join(OUT_ROOT, split)                 # original + ALL candidates
    byclass_dir = os.path.join(OUT_ROOT, "by_class")        # per-class candidate-only
    os.makedirs(aug_dir, exist_ok=True)

    per_class_count = defaultdict(int)
    per_class_manifest = defaultdict(list)                  # class -> [(img, n)]
    n_added = n_imgs_aug = 0
    pred_conf = min(CONF_THRES, CONFLICT_CONF)

    BATCH = 64
    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        results = model.predict(batch, conf=pred_conf, imgsz=IMGSZ, device=DEVICE,
                                verbose=False, stream=False)
        for img_path, r in zip(batch, results):
            label_path = img_to_label_path(img_path)
            stem = os.path.basename(label_path)
            gt_xyxy, orig_lines = read_gt_xyxy(label_path)

            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xywhn = r.boxes.xywhn.cpu().numpy()

            by_class = defaultdict(list)   # class id -> [label lines] for this image
            if len(cls):
                all_xyxy = xywhn_to_xyxy(xywhn)
                cand_idx = np.where(conf >= CONF_THRES)[0]
                iou_cg = iou_matrix(all_xyxy[cand_idx], gt_xyxy)
                iou_ca = iou_matrix(all_xyxy[cand_idx], all_xyxy)
                kept = 0
                for r_i, j in enumerate(cand_idx):
                    c = int(cls[j]); box = xywhn[j]
                    if gt_xyxy.shape[0] and iou_cg[r_i].max() >= IOU_SKIP:
                        continue
                    if ((cls != c) & (conf >= CONFLICT_CONF) & (iou_ca[r_i] >= CONFLICT_IOU)).any():
                        continue
                    if box[2] * box[3] < MIN_BOX_FRAC:
                        continue
                    by_class[c].append(f"{c} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                    kept += 1
                    if kept >= MAX_PER_IMAGE:
                        break

            all_pseudo = [ln for c in by_class for ln in by_class[c]]

            # 1) combined augmented file (original + all candidates)
            _write(os.path.join(aug_dir, stem), orig_lines + all_pseudo)

            # 2) per-class candidate-only files + manifests
            for c, lines in by_class.items():
                cname = names[c] if c < len(names) else str(c)
                _write(os.path.join(byclass_dir, cname, split, stem), lines)
                per_class_count[c] += len(lines)
                per_class_manifest[c].append((img_path, len(lines)))

            if all_pseudo:
                n_imgs_aug += 1
                n_added += len(all_pseudo)
        print(f"  [{split}] {min(start + BATCH, len(images))}/{len(images)}  +{n_added} candidates")

    # per-class manifests
    for c, rows in per_class_manifest.items():
        cname = names[c] if c < len(names) else str(c)
        with open(os.path.join(OUT_ROOT, f"manifest_{cname}_{split}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "n_candidates_added"])
            w.writerows(rows)

    pc = {names[c] if c < len(names) else c: per_class_count[c] for c in sorted(per_class_count)}
    return dict(split=split, images=len(images), imgs_aug=n_imgs_aug, added=n_added, per_class=pc)


# =============================================================================
# MAIN
# =============================================================================
def main():
    data = yaml.safe_load(open(DATA_YAML))
    names = data["names"]
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("=" * 72)
    print(f"  PSEUDO-LABEL CANDIDATES (per class) {names} — WHOLE DATASET")
    print(f"  model: {MODEL_WEIGHTS}")
    print(f"  keep if conf>={CONF_THRES}, IoU(GT)<{IOU_SKIP}, IoU(other-class pred)<{CONFLICT_IOU}")
    print("=" * 72)

    model = YOLO(MODEL_WEIGHTS)
    summaries = []
    for split in SPLITS:
        imgs = resolve_split_images(data, split)
        if not imgs:
            print(f"  [{split}] not present / empty — skipped")
            continue
        summaries.append(process_split(model, imgs, split, names))

    print("\n" + "=" * 72)
    print("  DONE — candidates per class:")
    header = f"  {'split':<6} {'images':>7} {'imgs_aug':>9} " + " ".join(f"{n:>10}" for n in names)
    print(header)
    for s in summaries:
        row = f"  {s['split']:<6} {s['images']:>7} {s['imgs_aug']:>9} " + \
              " ".join(f"{s['per_class'].get(n, 0):>10}" for n in names)
        print(row)
    print("\n  Per-class candidate labels: OUT_ROOT/by_class/<class>/<split>/")
    print("  Per-class manifests:        OUT_ROOT/manifest_<class>_<split>.csv")
    print("  A class with MANY candidates is under-annotated; FEW -> already well labeled")
    print("  (its candidates are mostly false positives — don't auto-merge those).")
    print("=" * 72)


if __name__ == "__main__":
    main()
