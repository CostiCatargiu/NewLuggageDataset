#!/usr/bin/env python3
"""
Render the pseudo-label candidates for visual inspection.

For every image that got at least one candidate, draw:
  * existing GT boxes      -> GREEN  (what was already annotated)
  * new candidate "other"  -> RED    (the proposed missing boxes)
into an output folder, so you can eyeball whether the red boxes are real
unlabeled "other" objects (calibrate the confidence threshold) and/or review
them for the test set.

Reads the augmented labels written by make_pseudo_labels_other.py (PSEUDO_ROOT)
and compares them to the ORIGINAL dataset labels to know which boxes are new.
"""
import os, glob, yaml
import cv2

DATA_YAML   = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
PSEUDO_ROOT = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_labels_other"  # augmented labels
OUT_DIR     = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_review_images"
SPLITS      = ["train", "val", "test"]
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GREEN = (0, 200, 0)      # existing GT
RED   = (0, 0, 255)      # candidate (new) "other"


def split_images(data, key):
    if key not in data or not data[key]:
        return []
    root = data.get("path", "")
    entries = data[key] if isinstance(data[key], list) else [data[key]]
    imgs = []
    for e in entries:
        p = e if os.path.isabs(e) else os.path.join(root, e)
        if os.path.isdir(p):
            for ext in IMG_EXTS:
                imgs += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
        elif p.endswith(".txt") and os.path.isfile(p):
            imgs += [l.strip() if os.path.isabs(l.strip()) else os.path.join(root, l.strip())
                     for l in open(p) if l.strip()]
    return sorted(set(imgs))


def orig_label(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    p = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(p)[0] + ".txt"


def read_lines(path):
    return [l.strip() for l in open(path)] if os.path.isfile(path) else []


def draw_box(img, line, color, names):
    f = line.split()
    if len(f) < 5:
        return
    cid = int(f[0]); cx, cy, bw, bh = map(float, f[1:5])
    h, w = img.shape[:2]
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = names[cid] if cid < len(names) else str(cid)
    cv2.putText(img, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    data = yaml.safe_load(open(DATA_YAML))
    names = data["names"]
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)

    total = 0
    for split in SPLITS:
        out = os.path.join(OUT_DIR, split)
        os.makedirs(out, exist_ok=True)
        rendered = 0
        for img_path in split_images(data, split):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            aug_path = os.path.join(PSEUDO_ROOT, split, stem + ".txt")
            orig = read_lines(orig_label(img_path))
            aug = read_lines(aug_path)
            new_lines = [l for l in aug if l and l not in orig]   # the appended candidates
            if not new_lines:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            for l in orig:
                if l:
                    draw_box(img, l, GREEN, names)      # existing GT
            for l in new_lines:
                draw_box(img, l, RED, names)            # new candidate
            cv2.imwrite(os.path.join(out, f"{stem}.jpg"), img)
            rendered += 1
        print(f"  [{split}] rendered {rendered} images with candidates -> {out}")
        total += rendered
    print(f"\n  done: {total} review images. GREEN = existing GT, RED = candidate 'other'.")
    print("  Scan the RED boxes: real unlabeled 'other' -> threshold ok; junk -> raise CONF_THRES.")


if __name__ == "__main__":
    main()
