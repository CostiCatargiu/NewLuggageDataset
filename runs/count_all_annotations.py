#!/usr/bin/env python3
"""
Per-class annotation-incompleteness table.

For each class and split, counts existing GT instances/images and combines them
with the per-class pseudo-label manifests to express the candidates as a TRUE
incompleteness rate (relative to that class's actual annotations / footprint),
so the four classes are finally comparable.

Manifests expected (from make_pseudo_labels_allclasses.py):
    MANIFEST_DIR/manifest_<classname>_<split>.csv   columns: image, n_candidates_added
"""
import os, glob, csv, yaml
from collections import defaultdict

DATA_YAML    = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
MANIFEST_DIR = "/home/constantin/Doctorat/GunDatasetNoAugSplit/pseudo_labels_allclasses"
SPLITS       = ["train", "val", "test"]
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def lbl(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    p = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(p)[0] + ".txt"


def read_manifest(cname, split):
    path = os.path.join(MANIFEST_DIR, f"manifest_{cname}_{split}.csv")
    boxes = imgs = 0
    if os.path.isfile(path):
        with open(path) as f:
            rd = csv.reader(f)
            next(rd, None)  # header
            for row in rd:
                if len(row) >= 2 and row[1].strip().isdigit():
                    boxes += int(row[1]); imgs += 1
    return boxes, imgs


def main():
    data = yaml.safe_load(open(DATA_YAML))
    names = data["names"]
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)

    print(f"{'split':<6} {'class':<10} {'imgs_w_cls':>10} {'GT':>7} {'cand_box':>9} "
          f"{'cand_img':>9} {'%box_missing':>13} {'%img_missing':>13}")
    for split in SPLITS:
        imgs = split_images(data, split)
        gt_cnt = defaultdict(int)
        img_cnt = defaultdict(int)
        for im in imgs:
            lp = lbl(im)
            if not os.path.isfile(lp):
                continue
            present = set()
            for l in open(lp):
                f = l.split()
                if len(f) >= 5:
                    c = int(f[0]); gt_cnt[c] += 1; present.add(c)
            for c in present:
                img_cnt[c] += 1
        for cid, cname in enumerate(names):
            cb, ci = read_manifest(cname, split)
            gt = gt_cnt[cid]
            pct_box = 100 * cb / max(gt + cb, 1)
            pct_img = 100 * ci / max(img_cnt[cid], 1)
            print(f"{split:<6} {cname:<10} {img_cnt[cid]:>10} {gt:>7} {cb:>9} "
                  f"{ci:>9} {pct_box:>12.1f}% {pct_img:>12.1f}%")
        print()

    print("  %box_missing = cand_boxes / (GT + cand_boxes)  [lower bound]")
    print("  %img_missing = cand_images / images_containing_that_class")
    print("  Compare the rates ACROSS classes: if 'other' >> weapons, the gap is")
    print("  specific to 'other'. If weapons are also high, they're under-annotated too.")


if __name__ == "__main__":
    main()
