#!/usr/bin/env python3
"""
Render the ALL-CLASS pseudo-label candidates for visual inspection.

For every image that got at least one candidate, draw:
  * existing GT boxes      -> GREEN, thin, labeled with class   (what's already annotated)
  * new candidate boxes    -> per-class BRIGHT color, thick, labeled "NEW:<class>"
so you can check whether each proposed box is a real, correctly-classified object
(missing label) or a model error.

Reads the augmented labels from PSEUDO_ROOT (original + candidates) and diffs them
against the ORIGINAL dataset labels to know which boxes are new.

CLASS_FILTER:
  None -> draw candidates of all classes.
  "pistol" (or any class name) -> only draw that class's candidates (focused review).
"""
import os, glob, yaml
import cv2

DATA_YAML   = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
PSEUDO_ROOT = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_labels_allclasses"  # augmented labels
OUT_DIR     = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_review_allclasses"
SPLITS      = ["train", "val", "test"]
CLASS_FILTER = None          # None = all classes; or e.g. "knife" to review one class
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GT_COLOR = (0, 180, 0)       # green for existing GT
# bright BGR palette for NEW candidate boxes, indexed by class id
CAND_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
               (0, 128, 255), (128, 0, 255), (255, 128, 0), (0, 255, 128)]


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
    return [l.strip() for l in open(path) if l.strip()] if os.path.isfile(path) else []


def draw(img, line, color, names, prefix=""):
    f = line.split()
    if len(f) < 5:
        return
    cid = int(f[0]); cx, cy, bw, bh = map(float, f[1:5])
    h, w = img.shape[:2]
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    thick = 3 if prefix else 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    name = names[cid] if cid < len(names) else str(cid)
    cv2.putText(img, prefix + name, (x1, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    data = yaml.safe_load(open(DATA_YAML))
    names = data["names"]
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    filt_id = names.index(CLASS_FILTER) if CLASS_FILTER else None

    total = 0
    for split in SPLITS:
        out = os.path.join(OUT_DIR, split)
        os.makedirs(out, exist_ok=True)
        rendered = 0
        for img_path in split_images(data, split):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            aug = read_lines(os.path.join(PSEUDO_ROOT, split, stem + ".txt"))
            orig = read_lines(orig_label(img_path))
            new_lines = [l for l in aug if l and l not in orig]
            if filt_id is not None:
                new_lines = [l for l in new_lines if l.split() and int(l.split()[0]) == filt_id]
            if not new_lines:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            for l in orig:
                draw(img, l, GT_COLOR, names)                      # existing GT (green)
            for l in new_lines:
                cid = int(l.split()[0])
                color = CAND_COLORS[cid % len(CAND_COLORS)]
                draw(img, l, color, names, prefix="NEW:")          # candidate (bright, thick)
            cv2.imwrite(os.path.join(out, f"{stem}.jpg"), img)
            rendered += 1
        print(f"  [{split}] rendered {rendered} images -> {out}")
        total += rendered

    tag = f"class='{CLASS_FILTER}'" if CLASS_FILTER else "all classes"
    print(f"\n  done: {total} review images ({tag}).")
    print("  GREEN = existing GT.  BRIGHT 'NEW:<class>' = candidate.")
    print("  Check each NEW box: real object of that class -> missing label; wrong -> model error.")


if __name__ == "__main__":
    main()
