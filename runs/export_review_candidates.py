#!/usr/bin/env python3
"""
Export images + labels that gained NEW pseudo-label candidates, for human review
(upload to Roboflow / CVAT / Label Studio).

For each split it diffs the CURRENT labels against a frozen backup (BACKUP_SUB).
Any image whose label file changed (boxes added/removed/class-changed) is copied —
with its CURRENT label (originals + appended candidates) — into REVIEW_ROOT.

Output layout (all splits merged):
  REVIEW_ROOT/images/<split>__<stem>.<ext>
  REVIEW_ROOT/labels/<split>__<stem>.txt
  REVIEW_ROOT/review_manifest.csv   (split, file, added, removed, changed)
  REVIEW_ROOT/name_map.csv          (exported_name -> ORIGINAL roboflow name)  <-- TRACKING
  REVIEW_ROOT/data.yaml

NAME TRACKING (important): Roboflow names look like  name_jpg.rf.<hash>.jpg
The part BEFORE '.rf' is shared across copies; the HASH AFTER '.rf' is the unique
differentiator. STRIP_RF collapses names to the pre-'.rf' part (and de-dups with
_1/_2), which would otherwise lose the link to the original. name_map.csv records,
for every exported file, the FULL original stem (with the .rf hash) + source path,
so each reviewed file can always be mapped back. For a guaranteed round-trip to
Roboflow, set STRIP_RF = False (keeps the full unique names).
"""
import os
import csv
import glob
import shutil
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
DS            = "/home/constantin/Doctorat/GunDatasetNoAugSplit"
SPLITS        = ["train", "valid", "test"]
LABELS_SUB    = "labels"           # <DS>/<split>/labels/
BACKUP_SUB    = "labels2"          # <DS>/<split>/labels2/   (frozen pre-merge originals)
IMAGES_SUB    = "images"           # <DS>/<split>/images/
IOU_MATCH     = 0.5
DATA_YAML     = f"{DS}/data.yaml"
IMG_EXTS      = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REVIEW_ROOT   = f"{DS}/review_candidates2"
PREFIX_SPLIT  = True               # prefix exported names with split to avoid cross-split collisions
STRIP_RF      = True               # keep only the part before '.rf'; False = keep full unique names


# =============================================================================
# DIFF HELPERS
# =============================================================================
def read_boxes(path):
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


def diff_counts(bak_path, cur_path):
    bc, bx = read_boxes(bak_path)
    cc, cx = read_boxes(cur_path)
    ious = iou_matrix(bx, cx)
    used_cur, matched_bak = set(), set()
    changed = 0
    for i in range(len(bx)):
        if ious.shape[1] == 0:
            break
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= IOU_MATCH and j not in used_cur:
            used_cur.add(j); matched_bak.add(i)
            if bc[i] != cc[j]:
                changed += 1
    removed = sum(1 for i in range(len(bx)) if i not in matched_bak)
    added = sum(1 for j in range(len(cx)) if j not in used_cur)
    return added, removed, changed


# =============================================================================
# FILE HELPERS
# =============================================================================
def find_image_for_stem(split, stem):
    img_dir = os.path.join(DS, split, IMAGES_SUB)
    base = os.path.splitext(stem)[0]
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.isfile(p):
            return p
    hits = []
    for ext in IMG_EXTS:
        hits += glob.glob(os.path.join(img_dir, "**", base + ext), recursive=True)
    return hits[0] if hits else None


def strip_rf(base):
    return base.split(".rf", 1)[0] if STRIP_RF else base


def dest_paths(split, stem, img_src):
    base = strip_rf(os.path.splitext(stem)[0])
    img_ext = os.path.splitext(img_src)[1]
    name = f"{split}__{base}" if PREFIX_SPLIT else base
    return (os.path.join(REVIEW_ROOT, "images", name + img_ext),
            os.path.join(REVIEW_ROOT, "labels", name + ".txt"))


def main():
    os.makedirs(REVIEW_ROOT, exist_ok=True)
    manifest = []
    name_map = []          # <-- TRACKING: exported name <-> original roboflow name
    total_imgs = 0

    for split in SPLITS:
        cur_dir = os.path.join(DS, split, LABELS_SUB)
        bak_dir = os.path.join(DS, split, BACKUP_SUB)
        if not os.path.isdir(bak_dir):
            print(f"  [{split}] no backup dir ({bak_dir}) — skipped")
            continue
        if not os.path.isdir(cur_dir):
            print(f"  [{split}] no current labels dir ({cur_dir}) — skipped")
            continue

        stems = set(os.path.basename(p) for p in glob.glob(os.path.join(cur_dir, "*.txt")))
        stems |= set(os.path.basename(p) for p in glob.glob(os.path.join(bak_dir, "*.txt")))

        copied = missing_img = 0
        for stem in sorted(stems):                       # stem keeps the FULL .rf.<hash> name
            a, r, c = diff_counts(os.path.join(bak_dir, stem), os.path.join(cur_dir, stem))
            if not (a or r or c):
                continue
            img_src = find_image_for_stem(split, stem)
            if img_src is None:
                missing_img += 1
                print(f"    [{split}] WARNING no image found for {stem}")
                continue

            cur_lbl = os.path.join(cur_dir, stem)
            img_dst, lbl_dst = dest_paths(split, stem, img_src)
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

            # de-dup if STRIP_RF collapsed distinct originals to the same name
            if os.path.exists(img_dst):
                root_i, ext_i = os.path.splitext(img_dst)
                root_l, ext_l = os.path.splitext(lbl_dst)
                k = 1
                while os.path.exists(f"{root_i}_{k}{ext_i}"):
                    k += 1
                img_dst = f"{root_i}_{k}{ext_i}"
                lbl_dst = f"{root_l}_{k}{ext_l}"

            shutil.copy2(img_src, img_dst)
            shutil.copy2(cur_lbl, lbl_dst)

            # TRACKING: record the final exported names against the ORIGINAL roboflow name
            orig_stem = os.path.splitext(stem)[0]        # full original incl. '.rf.<hash>'
            name_map.append((
                os.path.basename(img_dst),               # exported image name
                os.path.basename(lbl_dst),               # exported label name
                split,
                orig_stem,                               # ORIGINAL stem (with .rf hash)
                os.path.basename(img_src),               # original image filename
                img_src,                                 # original full source path
            ))
            manifest.append((split, stem, a, r, c))
            copied += 1

        total_imgs += copied
        print(f"  [{split}] candidates exported: {copied}"
              + (f"   (missing images: {missing_img})" if missing_img else ""))

    with open(os.path.join(REVIEW_ROOT, "review_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "added", "removed", "class_changed"])
        w.writerows(manifest)

    # the tracking file: map every exported file back to its original roboflow name
    with open(os.path.join(REVIEW_ROOT, "name_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exported_image", "exported_label", "split",
                    "original_stem_with_rf", "original_image", "original_path"])
        w.writerows(name_map)

    if os.path.isfile(DATA_YAML):
        shutil.copy2(DATA_YAML, os.path.join(REVIEW_ROOT, "data.yaml"))

    print(f"\n  DONE — {total_imgs} images+labels exported to: {REVIEW_ROOT}")
    print(f"  manifest -> review_manifest.csv")
    print(f"  TRACKING -> name_map.csv  (exported name <-> original .rf name)")
    print("  Keep name_map.csv: after reviewing, it maps each file back to its")
    print("  original Roboflow filename so you can merge labels into the right split.")


if __name__ == "__main__":
    main()
