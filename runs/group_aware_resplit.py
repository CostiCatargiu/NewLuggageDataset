#!/usr/bin/env python3
"""
Leakage-free re-split by NEAR-DUPLICATE CLUSTERS (no video IDs needed).

The dataset is video frames with ~30% cross-split near-duplicate leakage. We can't
group by source video (filenames don't carry it), so instead we group by visual
near-duplication: build clusters where any two images within Hamming<=THRESH are
linked (transitively, via union-find), then assign each WHOLE cluster to a single
split. Because every near-duplicate lands in the same cluster, no near-duplicate can
straddle two splits -> the new split is leakage-free at this threshold.

Assignment is stratified greedy: clusters are placed to hit the target ratios for
BOTH total images and each class count (so per-class balance is preserved).

Non-destructive: images are NOT moved. We write train/val/test .txt lists of image
paths plus a new data.yaml. Labels resolve via the usual /images/ -> /labels/ rule,
so the original folder layout is reused as-is.

Outputs (in --out dir):
  train.txt, val.txt, test.txt   (absolute image paths)
  data_regrouped.yaml            (points at the three lists)
  resplit_report.json            (cluster stats, split sizes, per-class counts, leak check)

Usage:
  python group_aware_resplit.py --data /path/to/data.yaml --thresh 5 --out regrouped_split
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HASH_SIZE = 8
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def resolve_splits(data_yaml):
    splits = {}
    try:
        import yaml
        d = yaml.safe_load(open(data_yaml))
        root = d.get("path", "") or ""
        names = d.get("names")
        for key in ("train", "val", "test"):
            v = d.get(key)
            if not v:
                continue
            v = v[0] if isinstance(v, list) else v
            p = v if os.path.isabs(v) else os.path.join(root, v)
            if os.path.isdir(os.path.join(p, "images")):
                p = os.path.join(p, "images")
            splits[key] = p
        return splits, names
    except Exception as e:
        print(f"  [warn] yaml parse failed ({e})")
        return splits, None


def list_images(d):
    out = []
    if d and os.path.isdir(d):
        for dp, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def label_path(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(lbl)[0] + ".txt"


def class_ids(img):
    lp = label_path(img)
    ids = set()
    if os.path.isfile(lp):
        for line in open(lp):
            s = line.split()
            if s:
                try:
                    ids.add(int(float(s[0])))
                except ValueError:
                    pass
    return ids


def dhash(path, size=HASH_SIZE):
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return np.uint64(val)
    except Exception:
        return None


def hamming_one_vs_many(h, arr):
    x = (arr ^ h).view(np.uint8).reshape(-1, 8)
    return _POP[x].sum(axis=1)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml")
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--out", default="regrouped_split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    splits, names = resolve_splits(args.data)
    paths = []
    for d in splits.values():
        paths += list_images(d)
    paths = sorted(set(paths))
    print(f"  pooled {len(paths)} images from {list(splits)}")

    # hashes + class ids
    hashes, kept, cls = [], [], []
    for p in paths:
        h = dhash(p)
        if h is None:
            continue
        hashes.append(h); kept.append(p); cls.append(class_ids(p))
    arr = np.array(hashes, dtype=np.uint64)
    n = len(kept)
    print(f"  hashed {n} images; clustering near-duplicates (Hamming<= {args.thresh})...")

    # union-find over near-duplicate pairs
    uf = UF(n)
    for i in range(n):
        if i % 2000 == 0:
            print(f"    clustering {i}/{n}")
        d2 = hamming_one_vs_many(arr[i], arr[i + 1:])
        for off in np.where(d2 <= args.thresh)[0]:
            uf.union(i, i + 1 + int(off))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    clusters = list(comp.values())
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"  {len(clusters)} clusters | singletons {sum(1 for s in sizes if s==1)} | largest {sizes[0]}")

    # all classes present
    all_cls = sorted({c for s in cls for c in s})
    def cluster_vec(idxs):
        v = {k: 0 for k in all_cls}
        for i in idxs:
            for k in cls[i]:
                v[k] += 1
        return v

    totals = {"img": n}
    for k in all_cls:
        totals[k] = sum(cluster_vec(c)[k] for c in clusters)
    target = {s: {"img": RATIOS[s] * totals["img"], **{k: RATIOS[s] * totals[k] for k in all_cls}} for s in RATIOS}

    # stratified greedy: place largest clusters first into the split that best fits targets
    order = sorted(range(len(clusters)), key=lambda c: len(clusters[c]), reverse=True)
    cur = {s: {"img": 0, **{k: 0 for k in all_cls}} for s in RATIOS}
    assign = {}
    for ci in order:
        vec = cluster_vec(clusters[ci]); m = len(clusters[ci])
        best, best_cost = None, None
        for s in RATIOS:
            cost = ((cur[s]["img"] + m - target[s]["img"]) / max(target[s]["img"], 1)) ** 2
            for k in all_cls:
                cost += ((cur[s][k] + vec[k] - target[s][k]) / max(target[s][k], 1)) ** 2
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, s
        assign[ci] = best
        cur[best]["img"] += m
        for k in all_cls:
            cur[best][k] += vec[k]

    split_imgs = {s: [] for s in RATIOS}
    for ci, s in assign.items():
        for i in clusters[ci]:
            split_imgs[s].append(kept[i])

    # write lists + yaml
    for s in RATIOS:
        with open(os.path.join(args.out, f"{s}.txt"), "w") as f:
            f.write("\n".join(sorted(split_imgs[s])) + "\n")
    yaml_txt = "nc: %d\n" % (len(names) if names else len(all_cls))
    if names:
        yaml_txt += "names: %s\n" % (list(names.values()) if isinstance(names, dict) else names)
    yaml_txt += "train: %s\nval: %s\ntest: %s\n" % tuple(
        os.path.abspath(os.path.join(args.out, f"{s}.txt")) for s in ("train", "val", "test"))
    open(os.path.join(args.out, "data_regrouped.yaml"), "w").write(yaml_txt)

    # verify: no cross-split near-dups remain (by construction)
    comp_of = {}
    for ci, members in enumerate(clusters):
        for i in members:
            comp_of[i] = ci
    idx_split = {}
    for s in RATIOS:
        for p in split_imgs[s]:
            idx_split[p] = s
    leak = 0  # clusters spanning >1 split (should be 0)
    for members in clusters:
        s = {idx_split[kept[i]] for i in members}
        if len(s) > 1:
            leak += 1

    report = {
        "thresh": args.thresh, "pooled_images": n, "clusters": len(clusters),
        "largest_cluster": sizes[0], "singletons": sum(1 for s in sizes if s == 1),
        "split_sizes": {s: len(split_imgs[s]) for s in RATIOS},
        "split_pct": {s: round(100 * len(split_imgs[s]) / n, 1) for s in RATIOS},
        "per_class": {s: {int(k): cur[s][k] for k in all_cls} for s in RATIOS},
        "clusters_spanning_multiple_splits": leak,
    }
    json.dump(report, open(os.path.join(args.out, "resplit_report.json"), "w"), indent=2)

    print(f"\n  new split sizes: " + " | ".join(f"{s} {len(split_imgs[s])} ({report['split_pct'][s]}%)" for s in RATIOS))
    print(f"  per-class counts:")
    for s in RATIOS:
        print(f"    {s}: " + ", ".join(f"cls{k}={cur[s][k]}" for k in all_cls))
    print(f"  clusters spanning >1 split: {leak}  (0 = leakage-free)")
    print(f"  wrote lists + data_regrouped.yaml + resplit_report.json -> {args.out}/")
    print("\n  Next: retrain with data_regrouped.yaml; these are your leakage-free splits.")


if __name__ == "__main__":
    main()
