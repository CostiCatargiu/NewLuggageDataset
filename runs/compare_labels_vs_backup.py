#!/usr/bin/env python3
"""
Compare current labels vs their *_backup for all splits — show what changed.

For each split it diffs LABELS_BASE/<split> against LABELS_BASE/<split>_backup,
matching boxes by IoU, and classifies every difference as:
  ADDED        : box in CURRENT with no match in backup   (new annotation)
  REMOVED      : box in BACKUP  with no match in current  (deleted annotation)
  CLASS_CHANGED: matched box (IoU >= IOU_MATCH) whose class differs (relabel)
  (matched + same class = unchanged, not reported)

Prints a per-split, per-class summary and writes a per-image CSV of the changes.
Pure file diff — needs no model. (names from data.yaml just for nicer labels.)
"""
import os, glob, csv
import numpy as np
try:
    import yaml
except Exception:
    yaml = None

# =============================================================================
# CONFIG
# =============================================================================
LABELS_BASE   = "/home/constantin/Doctorat/GunDatasetNoAugSplit/labels"  # contains <split>/ and <split>_backup/
SPLITS        = ["train", "val", "test"]
BACKUP_SUFFIX = "_backup"
IOU_MATCH     = 0.5         # boxes matched above this IoU are "the same box"
DATA_YAML     = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"  # for class names (optional)
OUT_DIR       = "."        # where per-split CSVs are written


def load_names():
    if yaml and os.path.isfile(DATA_YAML):
        n = yaml.safe_load(open(DATA_YAML)).get("names")
        if isinstance(n, dict):
            return [n[i] for i in range(len(n))]
        if n:
            return list(n)
    return None


def read_boxes(path):
    """Return (classes [N], xyxy [N,4]) from a YOLO label file."""
    cls, boxes = [], []
    if os.path.isfile(path):
        for l in open(path):
            f = l.split()
            if len(f) >= 5:
                c = int(f[0]); cx, cy, w, h = map(float, f[1:5])
                cls.append(c)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return (np.array(cls, int),
            np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32))


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


def diff_file(bak_path, cur_path):
    """Return (added, removed, changed) where each is a list of (class_old, class_new)."""
    bc, bx = read_boxes(bak_path)     # backup
    cc, cx = read_boxes(cur_path)     # current
    added, removed, changed = [], [], []

    ious = iou_matrix(bx, cx)
    used_cur = set()
    matched_bak = set()
    # greedy match each backup box to its best unused current box
    for i in range(len(bx)):
        if ious.shape[1] == 0:
            break
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= IOU_MATCH and j not in used_cur:
            used_cur.add(j); matched_bak.add(i)
            if bc[i] != cc[j]:
                changed.append((int(bc[i]), int(cc[j])))     # relabel
        # else: leave unmatched -> handled below
    for i in range(len(bx)):
        if i not in matched_bak:
            removed.append((int(bc[i]), None))
    for j in range(len(cx)):
        if j not in used_cur:
            added.append((None, int(cc[j])))
    return added, removed, changed


def main():
    names = load_names()
    cname = (lambda c: names[c] if names and c is not None and c < len(names) else str(c))

    for split in SPLITS:
        cur_dir = os.path.join(LABELS_BASE, split)
        bak_dir = os.path.join(LABELS_BASE, split + BACKUP_SUFFIX)
        if not os.path.isdir(bak_dir):
            print(f"  [{split}] no backup dir ({bak_dir}) — skipped")
            continue

        stems = set(os.path.basename(p) for p in glob.glob(os.path.join(cur_dir, "*.txt")))
        stems |= set(os.path.basename(p) for p in glob.glob(os.path.join(bak_dir, "*.txt")))

        add_cls, rem_cls, chg = {}, {}, {}     # per-class counters / change pairs
        imgs_changed = 0
        rows = []
        tot_add = tot_rem = tot_chg = 0
        for stem in sorted(stems):
            a, r, c = diff_file(os.path.join(bak_dir, stem), os.path.join(cur_dir, stem))
            if not (a or r or c):
                continue
            imgs_changed += 1
            tot_add += len(a); tot_rem += len(r); tot_chg += len(c)
            for _, nc in a:
                add_cls[nc] = add_cls.get(nc, 0) + 1
            for oc, _ in r:
                rem_cls[oc] = rem_cls.get(oc, 0) + 1
            for oc, nc in c:
                chg[(oc, nc)] = chg.get((oc, nc), 0) + 1
            rows.append((stem, len(a), len(r), len(c)))

        print(f"\n  ===== {split} =====")
        print(f"  files compared: {len(stems)}   images changed: {imgs_changed}")
        print(f"  ADDED boxes:   {tot_add}   " +
              ", ".join(f"{cname(k)}:{v}" for k, v in sorted(add_cls.items(), key=lambda x: str(x[0]))))
        print(f"  REMOVED boxes: {tot_rem}   " +
              ", ".join(f"{cname(k)}:{v}" for k, v in sorted(rem_cls.items(), key=lambda x: str(x[0]))))
        print(f"  CLASS CHANGED: {tot_chg}   " +
              ", ".join(f"{cname(o)}->{cname(n)}:{v}" for (o, n), v in sorted(chg.items(), key=lambda x: str(x[0]))))

        out_csv = os.path.join(OUT_DIR, f"label_diff_{split}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "added", "removed", "class_changed"])
            w.writerows(rows)
        print(f"  per-image diff -> {out_csv}")


if __name__ == "__main__":
    main()
