#!/usr/bin/env python3
"""
Dataset statistics for the (re-split) weapon dataset — for the paper's data chapter.

Per split (train/val/test) it reports:
  - number of images and number of annotated instances (boxes)
  - instances PER CLASS
  - object-size distribution: small / medium / large, by absolute pixel scale
    size = sqrt(box_width_px * box_height_px), using each image's real dimensions.
    Bins (defaults, configurable):  small < 48 px,  48 <= medium < 96 px,  large >= 96 px
  - per-class size breakdown (full detail)

Works with a YOLO data.yaml whose train/val/test are either image folders OR .txt
image-list files (so it runs directly on data_regrouped.yaml from the re-split).

Output: console tables + dataset_statistics.json (saved next to the data.yaml).

Usage:
  python dataset_statistics.py --data /path/to/data_regrouped.yaml
  python dataset_statistics.py --data data.yaml --small 48 --medium 96
"""

import os
import json
import math
from collections import defaultdict

from PIL import Image

# =============================================================================
# HARD-CODED CONFIG (edit here if paths/thresholds change)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/regrouped_split/data_regrouped.yaml"
SMALL = 48.0     # box scale (sqrt area, px) below this = small
MEDIUM = 96.0    # below this (and >= SMALL) = medium ; >= this = large
OUT = None       # None -> save dataset_statistics.json next to the data.yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve(data_yaml):
    """Return ({split: [image paths]}, names, out_dir)."""
    import yaml
    d = yaml.safe_load(open(data_yaml))
    root = d.get("path", "") or ""
    names = d.get("names")
    out_dir = root if root and os.path.isdir(root) else os.path.dirname(os.path.abspath(data_yaml))
    splits = {}
    for key in ("train", "val", "test"):
        v = d.get(key)
        if not v:
            continue
        v = v[0] if isinstance(v, list) else v
        p = v if os.path.isabs(v) else os.path.join(root, v)
        imgs = []
        if p.endswith(".txt") and os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if line:
                    imgs.append(line if os.path.isabs(line) else os.path.join(root, line))
        elif os.path.isdir(p):
            d2 = os.path.join(p, "images") if os.path.isdir(os.path.join(p, "images")) else p
            for dp, _, fs in os.walk(d2):
                for f in fs:
                    if f.lower().endswith(IMG_EXTS):
                        imgs.append(os.path.join(dp, f))
        splits[key] = sorted(set(imgs))
    return splits, names, out_dir


def label_path(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(lbl)[0] + ".txt"


def main():
    splits, names, out_dir = resolve(DATA_YAML)
    out = OUT or os.path.join(out_dir, "dataset_statistics.json")
    cls_name = (lambda k: names[k] if isinstance(names, dict) else names[k]) if names else (lambda k: f"class{k}")

    print(f"\n{'='*74}\n  DATASET STATISTICS  (small < {SMALL}px | medium < {MEDIUM}px | large >=)")
    print(f"  size = sqrt(box_w_px * box_h_px), per-image real dimensions\n{'='*74}")

    report = {"size_thresholds": {"small_lt": SMALL, "medium_lt": MEDIUM},
              "size_metric": "sqrt(box_area_px)", "splits": {}}
    grand = {"images": 0, "instances": 0}

    for sp, imgs in splits.items():
        n_img = len(imgs)
        per_class = defaultdict(int)
        size_bins = {"small": 0, "medium": 0, "large": 0}
        per_class_size = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
        n_inst = 0
        wh_cache = {}
        for img in imgs:
            lp = label_path(img)
            if not os.path.isfile(lp):
                continue
            try:
                W, H = Image.open(img).size
            except Exception:
                continue
            for line in open(lp):
                f = line.split()
                if len(f) < 5:
                    continue
                try:
                    c = int(float(f[0])); w = float(f[3]); h = float(f[4])
                except ValueError:
                    continue
                n_inst += 1
                per_class[c] += 1
                scale = math.sqrt(max(w * W, 0.0) * max(h * H, 0.0))
                b = "small" if scale < SMALL else ("medium" if scale < MEDIUM else "large")
                size_bins[b] += 1
                per_class_size[c][b] += 1
        report["splits"][sp] = {
            "images": n_img, "instances": n_inst,
            "boxes_per_image": round(n_inst / max(n_img, 1), 2),
            "per_class": {cls_name(k): per_class[k] for k in sorted(per_class)},
            "size_bins": size_bins,
            "size_bins_pct": {k: round(100 * v / max(n_inst, 1), 1) for k, v in size_bins.items()},
            "per_class_size": {cls_name(k): dict(per_class_size[k]) for k in sorted(per_class_size)},
        }
        grand["images"] += n_img; grand["instances"] += n_inst

        print(f"\n  [{sp}]  images={n_img}  instances={n_inst}  ({report['splits'][sp]['boxes_per_image']} boxes/img)")
        print("     per-class instances: " + ", ".join(f"{cls_name(k)}={per_class[k]}" for k in sorted(per_class)))
        print(f"     size: small={size_bins['small']} ({report['splits'][sp]['size_bins_pct']['small']}%)  "
              f"medium={size_bins['medium']} ({report['splits'][sp]['size_bins_pct']['medium']}%)  "
              f"large={size_bins['large']} ({report['splits'][sp]['size_bins_pct']['large']}%)")

    report["overall"] = grand
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n  TOTAL: {grand['images']} images, {grand['instances']} instances")
    print(f"  wrote -> {out}\n{'='*74}\n")


if __name__ == "__main__":
    main()
