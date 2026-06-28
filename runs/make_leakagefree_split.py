#!/usr/bin/env python3
"""
Create a LEAKAGE-FREE 70/15/15 split for the weapon dataset (video frames).

Why: the frames contain many near-duplicates, so a random/per-frame split leaks
almost-identical images across train/val/test and inflates the metrics. This script
groups near-duplicate frames into clusters and assigns each WHOLE cluster to a single
split, so no near-duplicate can straddle two subsets.

Method
  1. Pool every image referenced by the source data.yaml (train + val + test).
  2. Encode each image with a 64-bit perceptual hash (difference hash, dHash).
  3. Link any two images within Hamming distance <= THRESH; take connected
     components (union-find) = clusters of mutually near-identical frames.
  4. Stratified-greedy assignment: place clusters (largest first) into the split
     with the most remaining capacity, targeting 70/15/15 for the total image
     count AND for every class simultaneously (per-class balance).
  5. Verify: 0 clusters span more than one split  ->  leakage-free by construction.

Non-destructive: images/labels are NOT moved or modified. The script only writes
train/val/test .txt path lists + a new data.yaml (labels resolve via the usual
/images/ -> /labels/ rule).

Run:  python make_leakagefree_split.py      (edit the CONFIG block if paths differ)

Needs: numpy, pillow, pyyaml
"""

import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image

# =============================================================================
# CONFIG  (edit here)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"  # source split
OUT_DIR   = None          # None -> <dataset>/regrouped_split
THRESH    = 5             # Hamming distance: <= THRESH links two images
RATIOS    = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED      = 0
IMG_EXTS  = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


# ----------------------------------------------------------------- yaml / io
def resolve_splits(data_yaml):
    import yaml
    d = yaml.safe_load(open(data_yaml))
    root = d.get("path", "") or ""
    names = d.get("names")
    splits = {}
    for key in ("train", "val", "test"):
        v = d.get(key)
        if not v:
            continue
        v = v[0] if isinstance(v, list) else v
        p = v if os.path.isabs(v) else os.path.join(root, v)
        if p.endswith(".txt"):
            splits[key] = ("list", p)
        else:
            if os.path.isdir(os.path.join(p, "images")):
                p = os.path.join(p, "images")
            splits[key] = ("dir", p)
    return splits, names, root


def list_images(kind, p, root):
    out = []
    if kind == "dir" and os.path.isdir(p):
        for dp, _, fs in os.walk(p):
            for f in fs:
                if f.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(dp, f))
    elif kind == "list" and os.path.isfile(p):
        for line in open(p):
            s = line.strip()
            if s:
                out.append(s if os.path.isabs(s) else os.path.join(root, s))
    return out


def label_path(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(lbl)[0] + ".txt"


def class_ids(img):
    lp, ids = label_path(img), set()
    if os.path.isfile(lp):
        for line in open(lp):
            s = line.split()
            if s:
                try:
                    ids.add(int(float(s[0])))
                except ValueError:
                    pass
    return ids


# ----------------------------------------------------------------- hashing
def dhash(path, size=8):
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return np.uint64(v)
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


# ----------------------------------------------------------------- main
def main():
    splits, names, root = resolve_splits(DATA_YAML)
    dataset_dir = root if root and os.path.isdir(root) else os.path.dirname(os.path.abspath(DATA_YAML))
    out = OUT_DIR or os.path.join(dataset_dir, "regrouped_split")
    os.makedirs(out, exist_ok=True)
    print(f"  source yaml : {DATA_YAML}\n  output dir  : {out}")

    paths = []
    for kind, p in splits.values():
        paths += list_images(kind, p, root)
    paths = sorted(set(paths))
    print(f"  pooled {len(paths)} images from {list(splits)}")

    hashes, kept, cls = [], [], []
    for i, p in enumerate(paths):
        if i % 2000 == 0:
            print(f"    hashing {i}/{len(paths)}")
        h = dhash(p)
        if h is None:
            continue
        hashes.append(h); kept.append(p); cls.append(class_ids(p))
    arr = np.array(hashes, dtype=np.uint64)
    n = len(kept)
    print(f"  hashed {n} images; clustering near-duplicates (Hamming <= {THRESH})...")

    uf = UF(n)
    for i in range(n):
        if i % 2000 == 0:
            print(f"    clustering {i}/{n}")
        if i + 1 < n:
            d = hamming_one_vs_many(arr[i], arr[i + 1:])
            for off in np.where(d <= THRESH)[0]:
                uf.union(i, i + 1 + int(off))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    clusters = list(comp.values())
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"  {len(clusters)} clusters | singletons {sum(1 for s in sizes if s == 1)} | largest {sizes[0]}")

    all_cls = sorted({c for s in cls for c in s})
    def cluster_vec(idxs):
        v = {k: 0 for k in all_cls}
        for i in idxs:
            for k in cls[i]:
                v[k] += 1
        return v

    totals = {"img": n, **{k: sum(cluster_vec(c)[k] for c in clusters) for k in all_cls}}
    target = {s: {"img": RATIOS[s] * totals["img"], **{k: RATIOS[s] * totals[k] for k in all_cls}} for s in RATIOS}

    order = sorted(range(len(clusters)), key=lambda c: len(clusters[c]), reverse=True)
    cur = {s: {"img": 0, **{k: 0 for k in all_cls}} for s in RATIOS}
    assign, K = {}, max(len(all_cls), 1)
    for ci in order:
        vec, m = cluster_vec(clusters[ci]), len(clusters[ci])
        best, best_need = None, None
        for s in RATIOS:
            need = 4.0 * (target[s]["img"] - cur[s]["img"]) / max(target[s]["img"], 1.0)
            for k in all_cls:
                need += (1.0 / K) * (target[s][k] - cur[s][k]) / max(target[s][k], 1.0)
            if best_need is None or need > best_need:
                best_need, best = need, s
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
        open(os.path.join(out, f"{s}.txt"), "w").write("\n".join(sorted(split_imgs[s])) + "\n")
    yaml_txt = "nc: %d\n" % (len(names) if names else len(all_cls))
    if names:
        yaml_txt += "names: %s\n" % (list(names.values()) if isinstance(names, dict) else names)
    yaml_txt += "train: %s\nval: %s\ntest: %s\n" % tuple(
        os.path.abspath(os.path.join(out, f"{s}.txt")) for s in ("train", "val", "test"))
    open(os.path.join(out, "data_regrouped.yaml"), "w").write(yaml_txt)

    # verify: no cluster spans >1 split
    cl_of = {}
    for ci, members in enumerate(clusters):
        for i in members:
            cl_of[kept[i]] = ci
    img_split = {p: s for s in RATIOS for p in split_imgs[s]}
    cluster_splits = defaultdict(set)
    for p, s in img_split.items():
        cluster_splits[cl_of[p]].add(s)
    spanning = sum(1 for v in cluster_splits.values() if len(v) > 1)

    report = {
        "source_yaml": DATA_YAML, "hamming_threshold": THRESH, "seed": SEED,
        "total_images": n, "clusters": len(clusters),
        "singletons": sum(1 for s in sizes if s == 1), "largest_cluster": sizes[0],
        "clusters_spanning_multiple_splits": spanning,
        "split_images": {s: len(split_imgs[s]) for s in RATIOS},
        "split_pct": {s: round(100 * len(split_imgs[s]) / max(n, 1), 2) for s in RATIOS},
        "per_class_per_split": {
            s: {("class%d" % k): cur[s][k] for k in all_cls} for s in RATIOS},
    }
    json.dump(report, open(os.path.join(out, "resplit_report.json"), "w"), indent=2)

    print("\n  ==== RESULT ====")
    for s in RATIOS:
        print(f"   {s:5} {len(split_imgs[s]):6d} imgs  ({report['split_pct'][s]}%)")
    print(f"   clusters spanning >1 split: {spanning}  (0 = leakage-free)")
    print(f"   wrote: {out}/train.txt, val.txt, test.txt, data_regrouped.yaml, resplit_report.json\n")


if __name__ == "__main__":
    main()
