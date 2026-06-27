#!/usr/bin/env python3
"""
Leakage evidence figures for the dataset chapter.

Produces, from the ACTUAL images on the training box:

  1) leakage_montage.(png|pdf)  -- real near-duplicate frame pairs that the
     ORIGINAL split placed in different subsets (train vs test/val), shown side
     by side with their perceptual-hash Hamming distance. The visual proof.

  2) leakage_hist.(png|pdf)     -- for every test image, its nearest Hamming
     distance to ANY training image, plotted for the ORIGINAL split vs the
     leakage-free split. Original spikes near 0 (duplicates); clean split does
     not. The quantitative proof.

  3) leakage_figures_summary.json -- numbers behind the plots.

No arguments; edit the hard-coded paths below if needed.

  pip install pillow numpy matplotlib pyyaml
"""

import os, json, math
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================
DATASET_ROOT = "/home/constantin/Doctorat/GunDatasetNoAugSplit"
ORIG_YAML = os.path.join(DATASET_ROOT, "data.yaml")                       # original (leaky) split
NEW_YAML  = os.path.join(DATASET_ROOT, "regrouped_split/data_regrouped.yaml")  # leakage-free split
OUT_DIR   = DATASET_ROOT                  # where figures + json are saved
HAMMING_THR = 5                           # <= this counts as a near-duplicate
N_MONTAGE = 8                             # number of leaked pairs to show
THUMB_W = 320                             # montage thumbnail width (px)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# popcount lookup for 8-bit values
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- split loading
def resolve(data_yaml):
    """Return {split: [image paths]} from a YOLO data.yaml (folders or .txt lists)."""
    d = yaml.safe_load(open(data_yaml))
    root = d.get("path", "") or os.path.dirname(os.path.abspath(data_yaml))
    out = {}
    for key in ("train", "val", "test"):
        v = d.get(key)
        if not v:
            continue
        v = v[0] if isinstance(v, list) else v
        p = v if os.path.isabs(v) else os.path.join(root, v)
        imgs = []
        if p.endswith(".txt") and os.path.isfile(p):
            for line in open(p):
                s = line.strip()
                if s:
                    imgs.append(s if os.path.isabs(s) else os.path.join(root, s))
        elif os.path.isdir(p):
            base = os.path.join(p, "images") if os.path.isdir(os.path.join(p, "images")) else p
            for dp, _, fs in os.walk(base):
                for f in fs:
                    if f.lower().endswith(IMG_EXTS):
                        imgs.append(os.path.join(dp, f))
        out[key] = sorted(set(imgs))
    return out


# ---------------------------------------------------------------- perceptual hash
def dhash_bytes(path, cache):
    """64-bit dHash as 8 uint8 bytes; cached by path."""
    if path in cache:
        return cache[path]
    try:
        im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    except Exception:
        cache[path] = None
        return None
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()          # 64 bools
    packed = np.packbits(bits)                        # 8 uint8
    cache[path] = packed
    return packed


def hash_all(paths, cache):
    arr, keep = [], []
    for p in paths:
        h = dhash_bytes(p, cache)
        if h is not None:
            arr.append(h); keep.append(p)
    return (np.array(arr, dtype=np.uint8) if arr else np.zeros((0, 8), np.uint8)), keep


def nearest_dist(test_h, train_h):
    """For each test hash, min Hamming distance to any train hash. Returns (dist, idx)."""
    dists = np.empty(len(test_h), dtype=np.int16)
    idxs = np.empty(len(test_h), dtype=np.int64)
    for i in range(len(test_h)):
        x = np.bitwise_xor(train_h, test_h[i])       # (Ntrain, 8)
        d = _POP[x].sum(axis=1)                       # (Ntrain,)
        j = int(np.argmin(d))
        dists[i] = d[j]; idxs[i] = j
    return dists, idxs


# ---------------------------------------------------------------- montage
def thumb(path, w=THUMB_W):
    im = Image.open(path).convert("RGB")
    h = int(im.height * w / im.width)
    return im.resize((w, h), Image.BILINEAR)


def build_montage(pairs, out_base):
    """pairs: list of (train_path, test_path, dist, test_split). Stacked rows of 2."""
    if not pairs:
        print("  [montage] no leaked pairs found under threshold — skipping")
        return
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    cells = []
    for tr, te, d, sp in pairs:
        a, b = thumb(tr), thumb(te)
        H = max(a.height, b.height)
        pad, lab = 8, 34
        row = Image.new("RGB", (a.width + b.width + 3 * pad, H + lab + pad), "white")
        row.paste(a, (pad, lab)); row.paste(b, (a.width + 2 * pad, lab))
        dr = ImageDraw.Draw(row)
        dr.text((pad, 6), f"TRAIN", fill=(40, 80, 150), font=font)
        dr.text((a.width + 2 * pad, 6), f"{sp.upper()}   Hamming d = {d}", fill=(192, 57, 43), font=font)
        cells.append(row)
    W = max(c.width for c in cells)
    Htot = sum(c.height + 10 for c in cells) + 10
    canvas = Image.new("RGB", (W, Htot), "white")
    y = 10
    for c in cells:
        canvas.paste(c, (0, y)); y += c.height + 10
    canvas.save(out_base + ".png")
    canvas.save(out_base + ".pdf", "PDF", resolution=150)
    print(f"  [montage] wrote {out_base}.png / .pdf  ({len(pairs)} pairs)")


# ---------------------------------------------------------------- histogram
def build_hist(orig_d, new_d, out_base, thr=HAMMING_THR):
    bins = np.arange(0, 33, 2)
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(orig_d, bins=bins, alpha=0.62, color="#C0392B", label="original split", density=True)
    plt.hist(new_d,  bins=bins, alpha=0.62, color="#1E7A46", label="leakage-free split", density=True)
    plt.axvline(thr + 0.5, color="#444", ls="--", lw=1)
    plt.text(thr + 0.9, plt.ylim()[1] * 0.92, f"near-dup\nthreshold = {thr}", fontsize=9, color="#444")
    plt.xlabel("nearest Hamming distance from a test image to the training set")
    plt.ylabel("fraction of test images")
    plt.title("Cross-split near-duplicate distance: original vs leakage-free split")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=160)
    plt.savefig(out_base + ".pdf")
    print(f"  [hist] wrote {out_base}.png / .pdf")


# ---------------------------------------------------------------- main
def main():
    cache = {}
    summary = {"hamming_threshold": HAMMING_THR}

    print("\n== ORIGINAL split ==")
    o = resolve(ORIG_YAML)
    o_train_h, o_train_paths = hash_all(o.get("train", []), cache)
    pairs = []
    orig_test_nd = []
    for sp in ("test", "val"):
        if sp not in o:
            continue
        h, paths = hash_all(o[sp], cache)
        d, idx = nearest_dist(h, o_train_h)
        if sp == "test":
            orig_test_nd = d.tolist()
        leaked = int((d <= HAMMING_THR).sum())
        summary[f"original_{sp}"] = {
            "images": len(paths), "leaked_<=thr": leaked,
            "leaked_pct": round(100 * leaked / max(len(paths), 1), 1),
            "median_nearest": int(np.median(d)) if len(d) else None,
        }
        print(f"  {sp}: {leaked}/{len(paths)} ({summary[f'original_{sp}']['leaked_pct']}%) "
              f"have a train near-duplicate (d<={HAMMING_THR}); median nearest d={summary[f'original_{sp}']['median_nearest']}")
        order = np.argsort(d)
        for k in order:
            if d[k] <= HAMMING_THR:
                pairs.append((o_train_paths[idx[k]], paths[k], int(d[k]), sp))
            if len(pairs) >= N_MONTAGE:
                break
        if len(pairs) >= N_MONTAGE:
            break

    print("\n== LEAKAGE-FREE split ==")
    n = resolve(NEW_YAML)
    n_train_h, _ = hash_all(n.get("train", []), cache)
    nh, npaths = hash_all(n.get("test", []), cache)
    nd, _ = nearest_dist(nh, n_train_h)
    new_test_nd = nd.tolist()
    leaked = int((nd <= HAMMING_THR).sum())
    summary["clean_test"] = {
        "images": len(npaths), "leaked_<=thr": leaked,
        "leaked_pct": round(100 * leaked / max(len(npaths), 1), 1),
        "median_nearest": int(np.median(nd)) if len(nd) else None,
    }
    print(f"  test: {leaked}/{len(npaths)} ({summary['clean_test']['leaked_pct']}%) "
          f"have a train near-duplicate (d<={HAMMING_THR}); median nearest d={summary['clean_test']['median_nearest']}")

    print("\n== figures ==")
    build_montage(pairs[:N_MONTAGE], os.path.join(OUT_DIR, "leakage_montage"))
    build_hist(np.array(orig_test_nd), np.array(new_test_nd),
               os.path.join(OUT_DIR, "leakage_hist"))

    json.dump(summary, open(os.path.join(OUT_DIR, "leakage_figures_summary.json"), "w"), indent=2)
    print(f"\n  wrote {os.path.join(OUT_DIR, 'leakage_figures_summary.json')}\n")


if __name__ == "__main__":
    main()
