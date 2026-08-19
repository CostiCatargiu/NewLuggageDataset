#!/usr/bin/env python3
"""
analyze_dataset_v6i.py — dataset characterisation for LuggageDataset.v6i.yolov12

Regenerates the report in raw/LuggageDatasetSplitv6i.txt and additionally emits
machine-readable CSVs into tables/ for the paper.

NO ARGUMENTS. Edit the CONFIG block below and run:
    python analyze_dataset_v6i.py

Reads YOLO-format label files (class cx cy w h, all normalised) and the image
dimensions from the paired image files. Everything is reported in pixels at the
image's own native resolution, which for this dataset is 640 on the long side.
"""

import os
import csv
import glob
from collections import defaultdict

# ======================= CONFIG — edit these ================================
DATA_ROOT = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12"
SPLITS    = ["train", "valid", "test"]
CLASS_NAMES = ["backpack", "bag", "trolley"]

OUT_TXT   = "DATASET_ANALYSIS_v6i_regenerated.txt"
OUT_DIR   = "../tables"          # CSVs land here
WRITE_CSV = True

# ---- SIZE TAXONOMIES -------------------------------------------------------
# Four definitions exist in this project. They are NOT interchangeable.
# See ../SIZE_THRESHOLDS.md for which results were computed under which.
#
#   A  max side 48/96 px   -> dataset analysis + all diagnostics
#   B  max side 32/64 px   -> comparability reference only
#   C  AREA 1024/9216 px^2 -> every per-size mAP in the results JSONs (COCO)
#   D  max side 60/120 px  -> v5i only, WRONG on v6i, kept for traceability
#
# Each entry is (measure, {bucket: (lo, hi)}) where measure is "max_side" or "area".
TAXONOMIES = {
    "A": ("max_side", {"small": (0, 48),   "medium": (48, 96),     "large": (96, 10 ** 9)}),
    "B": ("max_side", {"small": (0, 32),   "medium": (32, 64),     "large": (64, 10 ** 9)}),
    "C": ("area",     {"small": (0, 1024), "medium": (1024, 9216), "large": (9216, 10 ** 12)}),
    "D": ("max_side", {"small": (0, 60),   "medium": (60, 120),    "large": (120, 10 ** 9)}),
}
# Which taxonomies to print in the human-readable report (CSVs always get all four).
REPORT_TAXONOMIES = ["A", "B", "C"]

# Back-compat aliases used by the section-3/4 printers.
SIZE_A = TAXONOMIES["A"][1]
SIZE_B = TAXONOMIES["B"][1]

# Shape taxonomy on h/w
SHAPE_WIDE_MAX   = 0.80
SHAPE_SQUARE_MAX = 1.25

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ============================================================================


def _img_size(path):
    """Image dimensions without a full decode. Pillow if present, else header parse."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size            # (w, h)
    except Exception:
        import struct
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        raise RuntimeError(f"cannot read size of {path} (install Pillow)")


def _bucket(v, taxonomy):
    """taxonomy is the {bucket: (lo, hi)} dict. Edges are inclusive on medium's
    upper bound, matching the dataset report's 'small < 48 <= medium <= 96 < large'."""
    lo_s, hi_s = taxonomy["small"]
    lo_m, hi_m = taxonomy["medium"]
    if v < hi_s:
        return "small"
    if v <= hi_m:
        return "medium"
    return "large"


def _measure(inst, tax_id):
    """The scalar this taxonomy buckets on."""
    return inst["area"] if TAXONOMIES[tax_id][0] == "area" else inst["max_side"]


def _bucket_by(inst, tax_id):
    return _bucket(_measure(inst, tax_id), TAXONOMIES[tax_id][1])


def _shape_bin(ratio):
    if ratio < SHAPE_WIDE_MAX:
        return "wide"
    if ratio <= SHAPE_SQUARE_MAX:
        return "square"
    return "tall"


def scan_split(root, split):
    """Return a list of per-instance dicts plus per-image bookkeeping."""
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)

    images = [p for p in sorted(glob.glob(os.path.join(img_dir, "*")))
              if p.lower().endswith(IMG_EXTS)]

    inst, n_labeled, n_background, n_missing = [], 0, 0, 0
    img_res = defaultdict(int)
    issues = []

    for ip in images:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        W, H = _img_size(ip)
        img_res[f"{W}x{H}"] += 1

        if not os.path.exists(lp):
            n_missing += 1
            continue
        rows = [r.split() for r in open(lp).read().splitlines() if r.strip()]
        if not rows:
            n_background += 1
            continue
        n_labeled += 1

        for r in rows:
            if len(r) < 5:
                issues.append(f"malformed row in {stem}.txt")
                continue
            c = int(float(r[0]))
            cx, cy, nw, nh = (float(x) for x in r[1:5])
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0) or nw <= 0 or nh <= 0:
                issues.append(f"out-of-range box in {stem}.txt")
            w_px, h_px = nw * W, nh * H
            inst.append({
                "cls": c,
                "w": w_px, "h": h_px,
                "max_side": max(w_px, h_px),
                "area": w_px * h_px,
                "ratio": (h_px / w_px) if w_px > 0 else 0.0,
                "img": stem, "IW": W, "IH": H,
            })

    return {
        "split": split, "images": len(images), "labeled": n_labeled,
        "background": n_background, "missing": n_missing,
        "inst": inst, "res": img_res, "issues": issues,
    }


def main():
    out = []
    p = out.append
    stats = {s: scan_split(DATA_ROOT, s) for s in SPLITS}
    tot_img = sum(v["images"] for v in stats.values())
    tot_ins = sum(len(v["inst"]) for v in stats.values())

    p("=" * 78)
    p("DATASET ANALYSIS REPORT — LuggageDataset.v6i")
    p("=" * 78)

    p("\n1) IMAGES & INSTANCES PER SPLIT")
    p(f"{'split':<9}{'images':<10}{'img %':<9}{'labeled':<10}{'background':<12}"
      f"{'no label file':<15}{'instances':<12}{'inst %':<9}{'avg box/img'}")
    p("-" * 78)
    for s in SPLITS:
        v = stats[s]
        n = len(v["inst"])
        p(f"{s:<9}{v['images']:<10}{v['images']/tot_img*100:<9.1f}{v['labeled']:<10}"
          f"{v['background']:<12}{v['missing']:<15}{n:<12}{n/tot_ins*100:<9.1f}"
          f"{n/max(v['images'],1):.2f}")

    p("\n2) INSTANCES PER CLASS (count and % within split)")
    p(f"{'class':<14}" + "".join(f"{s+' (n / %)':<20}" for s in SPLITS))
    p("-" * 78)
    for ci, cn in enumerate(CLASS_NAMES):
        row = f"{cn:<14}"
        for s in SPLITS:
            k = sum(1 for i in stats[s]["inst"] if i["cls"] == ci)
            tot = len(stats[s]["inst"]) or 1
            row += f"{k:>6} / {k/tot*100:>6.2f}%".ljust(20)
        p(row)

    for tag in REPORT_TAXONOMIES:
        meas, tax = TAXONOMIES[tag]
        hi_s, hi_m = tax["small"][1], tax["medium"][1]
        unit = "px^2 AREA" if meas == "area" else "px MAX SIDE"
        p(f"\n3{tag}) OBJECT SIZE ANALYSIS — TAXONOMY {tag} "
          f"(small < {hi_s}, {hi_s} <= medium <= {hi_m}, large > {hi_m}; by {unit})")
        if meas == "area":
            p("     NOTE: this is the COCO area convention — the one every per-size mAP")
            p("     in the results JSONs uses. NOT comparable with taxonomy A or B.")
        p(f"{'split':<9}{'small':<10}{'%':<9}{'medium':<10}{'%':<9}"
          f"{'large':<10}{'%':<9}{'mean W':<10}{'mean H'}")
        p("-" * 78)
        for s in SPLITS:
            ins = stats[s]["inst"]
            n = len(ins) or 1
            c = {k: sum(1 for i in ins if _bucket_by(i, tag) == k)
                 for k in ("small", "medium", "large")}
            mw = sum(i["w"] for i in ins) / n
            mh = sum(i["h"] for i in ins) / n
            p(f"{s:<9}{c['small']:<10}{c['small']/n*100:<9.1f}{c['medium']:<10}"
              f"{c['medium']/n*100:<9.1f}{c['large']:<10}{c['large']/n*100:<9.1f}"
              f"{mw:<10.0f}{mh:.0f}")

    for tag in REPORT_TAXONOMIES:
        p(f"\n4{tag}) SIZE DISTRIBUTION PER CLASS — TAXONOMY {tag} (train split)")
        p(f"{'class':<14}{'small':<10}{'medium':<10}{'large'}")
        p("-" * 78)
        for ci, cn in enumerate(CLASS_NAMES):
            ins = [i for i in stats["train"]["inst"] if i["cls"] == ci]
            c = {k: sum(1 for i in ins if _bucket_by(i, tag) == k)
                 for k in ("small", "medium", "large")}
            p(f"{cn:<14}{c['small']:<10}{c['medium']:<10}{c['large']}")

    p("\n4X) BUCKET DISAGREEMENT BETWEEN TAXONOMIES (train split)")
    p("     How many instances change bucket when the threshold definition changes.")
    p(f"{'pair':<12}{'agree':<12}{'disagree':<12}{'disagree %'}")
    p("-" * 78)
    ins = stats["train"]["inst"]
    n = len(ins) or 1
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        dis = sum(1 for i in ins if _bucket_by(i, a) != _bucket_by(i, b))
        p(f"{a} vs {b:<8}{n-dis:<12}{dis:<12}{dis/n*100:.1f}%")

    p(f"\n5) BOX SHAPE ANALYSIS (h/w ratio: wide < {SHAPE_WIDE_MAX}, "
      f"square {SHAPE_WIDE_MAX}-{SHAPE_SQUARE_MAX}, tall > {SHAPE_SQUARE_MAX})")
    p(f"{'split':<9}{'wide':<10}{'%':<9}{'square':<10}{'%':<9}{'tall':<10}{'%':<9}{'mean h/w'}")
    p("-" * 78)
    for s in SPLITS:
        ins = stats[s]["inst"]
        n = len(ins) or 1
        c = {k: sum(1 for i in ins if _shape_bin(i["ratio"]) == k)
             for k in ("wide", "square", "tall")}
        p(f"{s:<9}{c['wide']:<10}{c['wide']/n*100:<9.1f}{c['square']:<10}"
          f"{c['square']/n*100:<9.1f}{c['tall']:<10}{c['tall']/n*100:<9.1f}"
          f"{sum(i['ratio'] for i in ins)/n:.2f}")

    p("\n6) BOX SHAPE PER CLASS, PER SPLIT (mean h/w)")
    p(f"{'class':<14}" + "".join(f"{s+' (mean h/w)':<18}" for s in SPLITS))
    p("-" * 78)
    for ci, cn in enumerate(CLASS_NAMES):
        row = f"{cn:<14}"
        for s in SPLITS:
            ins = [i for i in stats[s]["inst"] if i["cls"] == ci]
            m = sum(i["ratio"] for i in ins) / max(len(ins), 1)
            row += f"{m:.2f} (n={len(ins)})".ljust(18)
        p(row)

    p("\n   Shape-bin breakdown per class (train split):")
    p(f"{'class':<14}{'wide':<10}{'square':<10}{'tall'}")
    p("-" * 78)
    for ci, cn in enumerate(CLASS_NAMES):
        ins = [i for i in stats["train"]["inst"] if i["cls"] == ci]
        c = {k: sum(1 for i in ins if _shape_bin(i["ratio"]) == k)
             for k in ("wide", "square", "tall")}
        p(f"{cn:<14}{c['wide']:<10}{c['square']:<10}{c['tall']}")

    p("\n7) DATA QUALITY")
    for s in SPLITS:
        v = stats[s]
        top = sorted(v["res"].items(), key=lambda kv: -kv[1])[:3]
        msg = f"{len(v['issues'])} issue(s)" if v["issues"] else "no issues found"
        p(f"  {s}: {msg} | top image sizes: " +
          ", ".join(f"{k} ({n})" for k, n in top))

    p("\n" + "=" * 78)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[written] {OUT_TXT}")

    if WRITE_CSV:
        os.makedirs(OUT_DIR, exist_ok=True)

        with open(os.path.join(OUT_DIR, "split_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "images", "instances", "avg_box_per_img",
                        "mean_w_px", "mean_h_px", "mean_hw_ratio"])
            for s in SPLITS:
                ins = stats[s]["inst"]; n = len(ins) or 1
                w.writerow([s, stats[s]["images"], len(ins),
                            round(len(ins)/max(stats[s]['images'],1), 2),
                            round(sum(i["w"] for i in ins)/n, 1),
                            round(sum(i["h"] for i in ins)/n, 1),
                            round(sum(i["ratio"] for i in ins)/n, 3)])

        # every taxonomy x every split — the "all thresholds" table
        with open(os.path.join(OUT_DIR, "size_distribution_all_thresholds.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["taxonomy", "measure", "small_edge", "medium_edge", "split",
                        "small", "small_pct", "medium", "medium_pct", "large", "large_pct"])
            for tag, (meas, tax) in TAXONOMIES.items():
                for s in SPLITS:
                    ins = stats[s]["inst"]; n = len(ins) or 1
                    c = [sum(1 for i in ins if _bucket_by(i, tag) == k)
                         for k in ("small", "medium", "large")]
                    w.writerow([tag, meas, tax["small"][1], tax["medium"][1], s,
                                c[0], round(c[0]/n*100, 1),
                                c[1], round(c[1]/n*100, 1),
                                c[2], round(c[2]/n*100, 1)])

        # per class, per taxonomy, per split, plus shape
        with open(os.path.join(OUT_DIR, "class_size_shape.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["taxonomy", "split", "class", "n", "small", "medium", "large",
                        "wide", "square", "tall", "mean_hw",
                        "mean_max_side_px", "mean_area_px2"])
            for tag in TAXONOMIES:
                for s in SPLITS:
                    for ci, cn in enumerate(CLASS_NAMES):
                        ins = [i for i in stats[s]["inst"] if i["cls"] == ci]
                        n = len(ins) or 1
                        c = [sum(1 for i in ins if _bucket_by(i, tag) == k)
                             for k in ("small", "medium", "large")]
                        sh = [sum(1 for i in ins if _shape_bin(i["ratio"]) == k)
                              for k in ("wide", "square", "tall")]
                        w.writerow([tag, s, cn, len(ins), *c, *sh,
                                    round(sum(i["ratio"] for i in ins)/n, 3),
                                    round(sum(i["max_side"] for i in ins)/n, 1),
                                    round(sum(i["area"] for i in ins)/n, 1)])

        # how often the taxonomies disagree, per split
        with open(os.path.join(OUT_DIR, "taxonomy_disagreement.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "pair", "n", "agree", "disagree", "disagree_pct"])
            for s in SPLITS:
                ins = stats[s]["inst"]; n = len(ins) or 1
                for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "D")):
                    dis = sum(1 for i in ins if _bucket_by(i, a) != _bucket_by(i, b))
                    w.writerow([s, f"{a}_vs_{b}", len(ins), len(ins)-dis, dis,
                                round(dis/n*100, 1)])

        print(f"[written] {OUT_DIR}/split_summary.csv, size_distribution_all_thresholds.csv, "
              f"class_size_shape.csv, taxonomy_disagreement.csv")


if __name__ == "__main__":
    main()
