#!/usr/bin/env python3
"""
Build a LEAKAGE-FREE 70/15/15 weapon dataset (real images + labels on disk).

The frames contain many near-duplicates, so a random / per-frame split leaks
almost-identical images across train/val/test and inflates metrics. This script
groups near-duplicate frames into clusters and assigns each WHOLE cluster to a
single split, then MATERIALISES a ready-to-train dataset folder:

    OUT/
      train/images/*.jpg   train/labels/*.txt
      val/images/*.jpg     val/labels/*.txt
      test/images/*.jpg    test/labels/*.txt
      data.yaml            (points at the three image folders)
      resplit_report.json  (cluster stats, sizes, per-class, leak check)

Method
  1. Pool every image referenced by the source data.yaml (train+val+test).
  2. 64-bit perceptual hash (dHash) per image.
  3. Link images within Hamming <= THRESH; connected components (union-find) = clusters.
  4. Stratified-greedy: place whole clusters to hit 70/15/15 for the image count
     AND every class. No cluster is split -> no near-duplicate crosses a subset.
  5. Copy (or link) each image + its label into the new split folders.

Run:  python make_leakagefree_split.py     (edit CONFIG below if paths differ)
Needs: numpy, pillow, pyyaml
"""

import os
import json
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

# =============================================================================
# CONFIG  (edit here)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"     # source split
OUT_DIR   = "/home/constantin/Doctorat/GunDatasetNoAugSplit/leakagefree"   # new dataset root
MODE      = "copy"        # "copy" (portable) | "hardlink" (no extra disk) | "symlink"
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


# ----------------------------------------------------------------- place files
def place(src, dst, mode):
    if os.path.exists(dst):
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


# ----------------------------------------------------------------- main
def main():
    splits, names, root = resolve_splits(DATA_YAML)
    out = OUT_DIR
    os.makedirs(out, exist_ok=True)
    print(f"  source yaml : {DATA_YAML}\n  output root : {out}\n  mode        : {MODE}")

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
    print(f"  hashed {n} images; clustering (Hamming <= {THRESH})...")

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

    split_idx = {s: [] for s in RATIOS}
    for ci, s in assign.items():
        split_idx[s].extend(clusters[ci])

    # ---- materialise images + labels ----
    used_names = set()
    counts = {s: 0 for s in RATIOS}
    for s in RATIOS:
        img_dir = os.path.join(out, s, "images")
        lbl_dir = os.path.join(out, s, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in split_idx[s]:
            src_img = kept[i]
            base = os.path.basename(src_img)
            stem, ext = os.path.splitext(base)
            # guarantee unique filename across the whole dataset
            name = base
            if name in used_names:
                name = f"{stem}_{i}{ext}"
            used_names.add(name)
            place(src_img, os.path.join(img_dir, name), MODE)
            src_lbl = label_path(src_img)
            dst_lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            if os.path.isfile(src_lbl):
                place(src_lbl, dst_lbl, MODE)
            else:
                open(dst_lbl, "w").close()   # empty = background (no objects)
            counts[s] += 1
        print(f"  [{s}] wrote {counts[s]} images + labels -> {os.path.join(out, s)}")

    # ---- data.yaml ----
    name_list = (list(names.values()) if isinstance(names, dict) else names) if names else None
    yaml_txt = "path: %s\n" % os.path.abspath(out)
    yaml_txt += "train: train/images\nval: val/images\ntest: test/images\n"
    yaml_txt += "nc: %d\n" % (len(name_list) if name_list else len(all_cls))
    if name_list:
        yaml_txt += "names: %s\n" % name_list
    open(os.path.join(out, "data.yaml"), "w").write(yaml_txt)

    # ---- leak verification + report ----
    cl_of = {}
    for ci, members in enumerate(clusters):
        for i in members:
            cl_of[i] = ci
    cluster_splits = defaultdict(set)
    for s in RATIOS:
        for i in split_idx[s]:
            cluster_splits[cl_of[i]].add(s)
    spanning = sum(1 for v in cluster_splits.values() if len(v) > 1)

    report = {
        "source_yaml": DATA_YAML, "output_root": os.path.abspath(out), "mode": MODE,
        "hamming_threshold": THRESH, "seed": SEED,
        "total_images": n, "clusters": len(clusters),
        "singletons": sum(1 for s in sizes if s == 1), "largest_cluster": sizes[0],
        "clusters_spanning_multiple_splits": spanning,
        "split_images": {s: counts[s] for s in RATIOS},
        "split_pct": {s: round(100 * counts[s] / max(n, 1), 2) for s in RATIOS},
        "per_class_per_split": {s: {("class%d" % k): cur[s][k] for k in all_cls} for s in RATIOS},
    }
    json.dump(report, open(os.path.join(out, "resplit_report.json"), "w"), indent=2)

    print("\n  ==== RESULT ====")
    for s in RATIOS:
        print(f"   {s:5} {counts[s]:6d} imgs  ({report['split_pct'][s]}%)")
    print(f"   clusters spanning >1 split: {spanning}   (0 = leakage-free)")
    print(f"   dataset ready: {out}  (train it with {os.path.join(out, 'data.yaml')})\n")


if __name__ == "__main__":
    main()
