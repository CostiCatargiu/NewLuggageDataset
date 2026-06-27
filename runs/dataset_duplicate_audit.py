#!/usr/bin/env python3
"""
Near-duplicate / redundancy audit for a video-frame YOLO dataset.

Why: when a dataset is built from video frames, consecutive frames are almost
identical. Two problems follow:
  (1) REDUNDANCY  — many near-identical images inflate the set without adding
      diversity (over-optimistic "dataset size").
  (2) LEAKAGE     — if near-duplicate frames land in BOTH train and val/test, the
      model effectively sees the test images during training, which inflates the
      reported metrics. This is the serious one for a paper.

Method: perceptual hash (dHash, 64-bit) per image + Hamming distance. dHash is
robust to small changes (compression, tiny shifts) so consecutive frames hash
close together. Two images are "near-duplicate" if Hamming distance <= THRESH.
Exact duplicates (identical bytes) are also reported via MD5.

Outputs:
  - console summary: per-split redundancy, and the KEY leakage numbers
    (% of val/test images that have a near-duplicate in train).
  - duplicate_audit_report.json : full machine-readable report.
  - leakage_pairs.csv           : every cross-split near-duplicate pair (review/remove).

Dependencies: pillow, numpy (both already in the YOLO env). pyyaml optional.

Usage:
  python dataset_duplicate_audit.py
  python dataset_duplicate_audit.py --data /path/to/data.yaml --thresh 5
"""

import os
import csv
import json
import hashlib
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

# =============================================================================
# CONFIG (defaults; override on the command line)
# =============================================================================
DEFAULT_DATA = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
HASH_SIZE = 8          # dHash grid -> (HASH_SIZE*HASH_SIZE) = 64-bit hash
THRESH = 5             # Hamming distance <= THRESH  => near-duplicate (0 = identical hash)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_WITHIN_EXAMPLES = 40   # cap example pairs stored per split (counts are still exact)

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


# =============================================================================
# dataset resolution
# =============================================================================
def resolve_splits(data_yaml):
    """Return {split: image_dir} for train/val/test from a YOLO data.yaml."""
    root, splits = "", {}
    try:
        import yaml
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        root = d.get("path", "") or ""
        for key in ("train", "val", "test"):
            v = d.get(key)
            if not v:
                continue
            v = v[0] if isinstance(v, list) else v
            p = v if os.path.isabs(v) else os.path.join(root, v)
            splits[key] = p
    except Exception as e:
        print(f"  [warn] could not parse {data_yaml} ({e}); trying sibling folders")
        base = os.path.dirname(data_yaml)
        for key in ("train", "val", "test"):
            for cand in (os.path.join(base, key, "images"), os.path.join(base, key)):
                if os.path.isdir(cand):
                    splits[key] = cand
                    break
    # normalize: if a dir has an 'images' subdir, use it
    for k, p in list(splits.items()):
        if os.path.isdir(os.path.join(p, "images")):
            splits[k] = os.path.join(p, "images")
    return splits


def list_images(d):
    out = []
    if not d or not os.path.isdir(d):
        return out
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dp, f))
    return sorted(out)


# =============================================================================
# hashing
# =============================================================================
def dhash(path, size=HASH_SIZE):
    """64-bit perceptual hash as uint64; None on failure."""
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.int16)
        diff = a[:, 1:] > a[:, :-1]          # horizontal gradient -> size*size bits
        bits = diff.flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return np.uint64(val)
    except Exception:
        return None


def md5(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def hamming_one_vs_many(h, arr):
    """Hamming distance of uint64 scalar h to each element of uint64 array arr."""
    x = (arr ^ h).view(np.uint8).reshape(-1, 8)
    return _POP[x].sum(axis=1)


# =============================================================================
# audit
# =============================================================================
def audit(splits, thresh):
    data = {}   # split -> {"paths":[], "hash":uint64 array, "md5":[...]}
    for sp, d in splits.items():
        paths = list_images(d)
        print(f"  [{sp}] {len(paths)} images in {d}")
        hashes, md5s, kept = [], [], []
        for p in paths:
            h = dhash(p)
            if h is None:
                continue
            hashes.append(h); md5s.append(md5(p)); kept.append(p)
        data[sp] = {"paths": kept, "hash": np.array(hashes, dtype=np.uint64), "md5": md5s}
    report = {"thresh": thresh, "hash_bits": HASH_SIZE * HASH_SIZE, "splits": {}, "within": {}, "cross": {}}

    # ---- per-split: exact dups + within-split near-dups ----
    for sp, dd in data.items():
        n = len(dd["paths"])
        arr = dd["hash"]
        # exact (md5) duplicate groups
        groups = defaultdict(list)
        for i, m in enumerate(dd["md5"]):
            if m: groups[m].append(i)
        exact_pairs = sum(len(g) - 1 for g in groups.values() if len(g) > 1)
        # near-dup (pHash) within split
        near_count = 0
        has_near = np.zeros(n, dtype=bool)
        examples = []
        for i in range(n):
            if i + 1 >= n:
                break
            d2 = hamming_one_vs_many(arr[i], arr[i + 1:])
            hit = np.where(d2 <= thresh)[0]
            if len(hit):
                near_count += len(hit)
                has_near[i] = True
                has_near[i + 1 + hit] = True
                if len(examples) < MAX_WITHIN_EXAMPLES:
                    j = i + 1 + hit[0]
                    examples.append([os.path.basename(dd["paths"][i]),
                                     os.path.basename(dd["paths"][j]), int(d2[hit[0]])])
        report["splits"][sp] = {"images": n}
        report["within"][sp] = {
            "exact_duplicate_pairs": int(exact_pairs),
            "near_duplicate_pairs": int(near_count),
            "images_with_a_near_dup": int(has_near.sum()),
            "pct_images_with_near_dup": round(100.0 * has_near.sum() / max(n, 1), 1),
            "examples": examples,
        }

    # ---- cross-split near-dups (LEAKAGE) — ALL pairs: train-val, train-test, val-test ----
    order = [s for s in ("train", "val", "test") if s in data and len(data[s]["paths"])]
    for ai in range(len(order)):
        for bi in range(ai + 1, len(order)):
            a, b = order[ai], order[bi]      # report % of b (downstream split) leaked from a
            aarr = data[a]["hash"]
            bdd = data[b]
            leaked, pairs = 0, []
            for i, h in enumerate(bdd["hash"]):
                d2 = hamming_one_vs_many(h, aarr)
                j = int(np.argmin(d2))
                dmin = int(d2[j])
                if dmin <= thresh:
                    leaked += 1
                    pairs.append({
                        "pair": f"{a}_vs_{b}", "hamming": dmin,
                        "leaked_split": b, "leaked_image": bdd["paths"][i],
                        "ref_split": a, "ref_match": data[a]["paths"][j],
                    })
            n = len(bdd["paths"])
            report["cross"][f"{a}_vs_{b}"] = {
                "leaked_side": b, "ref_side": a,
                "near_dup_images": leaked,
                "pct_leaked": round(100.0 * leaked / max(n, 1), 1),
                "pairs": pairs,
            }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--thresh", type=int, default=THRESH)
    ap.add_argument("--out", default="duplicate_audit_report.json")
    ap.add_argument("--csv", default="leakage_pairs.csv")
    args = ap.parse_args()

    print(f"\n{'='*72}\n  NEAR-DUPLICATE / LEAKAGE AUDIT  (dHash, Hamming<= {args.thresh})\n{'='*72}")
    splits = resolve_splits(args.data)
    if not splits:
        print("  No splits found. Pass --data <data.yaml> or check paths.")
        return
    rep = audit(splits, args.thresh)

    print(f"\n  --- WITHIN-SPLIT redundancy ---")
    for sp, w in rep["within"].items():
        print(f"  [{sp}] {rep['splits'][sp]['images']} imgs | exact-dup pairs: {w['exact_duplicate_pairs']} | "
              f"near-dup pairs: {w['near_duplicate_pairs']} | "
              f"{w['pct_images_with_near_dup']}% of images have a near-duplicate")

    print(f"\n  --- CROSS-SPLIT LEAKAGE (train-val, train-test, val-test) ---")
    leak_rows = []
    if not rep["cross"]:
        print("  (need >=2 non-empty splits to compare)")
    for k, c in rep["cross"].items():
        print(f"  {k}: {c['near_dup_images']} {c['leaked_side']} images have a near-dup in "
              f"{c['ref_side']}  =>  {c['pct_leaked']}% of {c['leaked_side']} leaked")
        leak_rows.extend(c["pairs"])

    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    if leak_rows:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["pair", "hamming", "leaked_split", "leaked_image", "ref_split", "ref_match"])
            for p in leak_rows:
                wr.writerow([p["pair"], p["hamming"], p["leaked_split"], p["leaked_image"],
                             p["ref_split"], p["ref_match"]])
        print(f"\n  wrote {len(leak_rows)} cross-split near-dup pairs -> {args.csv}")
    print(f"  wrote full report -> {args.out}")

    # verdict
    print(f"\n{'='*72}")
    if rep["cross"] and all(c["near_dup_images"] == 0 for c in rep["cross"].values()):
        print("  VERDICT: no cross-split near-duplicates at this threshold — splits look clean.")
    elif rep["cross"]:
        worst = max(rep["cross"].values(), key=lambda c: c["pct_leaked"])
        print(f"  VERDICT: cross-split leakage detected (up to {worst['pct_leaked']}% of "
              f"{worst['leaked_side']} has a near-dup in {worst['ref_side']}).")
        print("  -> review the CSV; re-split so all frames from one video/clip stay in ONE split.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
