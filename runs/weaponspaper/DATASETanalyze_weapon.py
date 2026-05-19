"""
Extended dataset analysis for weapon detection project — v3 (multi-threshold).

Supports multiple datasets in a single run (FULL + ablation subsets).

Key features:
  1. PER-DATASET SUMMARY: image/instance counts, density, class distribution
  2. MULTI-THRESHOLD SIZE ANALYSIS:
       - 24/72px  (tighter small definition)
       - 32/96px  (COCO standard)
       - 48/144px (looser small definition)
     Shows absolute counts + percentages for small/medium/large at each threshold.
  3. TRAIN ↔ TEST SHIFT DETECTION per threshold
  4. CROSS-DATASET COMPARISON (side-by-side) per threshold
  5. FIDELITY REPORT per threshold (ablation vs full dataset)
  6. FILENAME / SOURCE OVERLAP detection
  7. PERCEPTUAL HASH LEAKAGE check (optional)

Usage:
    python DATASETanalyze_weapon.py

Requires: Pillow + imagehash (optional, for perceptual hash leakage check)
"""

import os
import re
import glob
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 640
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']

# Multiple size threshold configurations
# Each entry: (label, small_px, medium_px)
# "small" = area < small_px², "medium" = small_px² ≤ area < medium_px², "large" = area ≥ medium_px²
SIZE_THRESHOLDS = [
    ("24/72",  24,  72),   # tighter small definition
    ("32/96",  32,  96),   # COCO standard (current default)
    ("48/144", 48, 144),   # looser small definition
    ("64/192", 64, 192),   # large-object-friendly threshold
]
DEFAULT_THRESHOLD_LABEL = "32/96"  # used for shift/fidelity reports

try:
    from PIL import Image
    import imagehash
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def get_label_files(labels_dir):
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    return [f for f in files if os.path.basename(f) != "classes.txt"]


def parse_label_file(label_path):
    annotations = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    annotations.append((cid, x, y, w, h))
                except ValueError:
                    continue
    except Exception:
        pass
    return annotations


def classify_size(w_norm, h_norm, small_px=32, medium_px=96):
    """Classify bounding box into small/medium/large based on pixel thresholds."""
    area = w_norm * h_norm
    small_area_norm = (small_px * small_px) / (IMAGE_SIZE * IMAGE_SIZE)
    medium_area_norm = (medium_px * medium_px) / (IMAGE_SIZE * IMAGE_SIZE)
    if area < small_area_norm:
        return "small"
    elif area < medium_area_norm:
        return "medium"
    return "large"


def get_image_for_label(label_path, images_dir):
    stem = Path(label_path).stem
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def format_pct(value, total):
    return 100 * value / total if total else 0.0


# ─────────────────────────────────────────────────────────────────
# Core: compute summary statistics for a single dataset
# ─────────────────────────────────────────────────────────────────
def compute_dataset_stats(dataset_root):
    """
    Computes a full statistics dictionary for a dataset.
    Returns nested dict: stats[split] = { ... metrics ... }
    Size distributions are computed for ALL threshold configurations.
    """
    stats = {}
    for split in SPLITS:
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue

        label_files = get_label_files(labels_dir)
        counts = []
        # Size counts per threshold config: {threshold_label: Counter}
        size_counts_multi = {tl: Counter() for tl, _, _ in SIZE_THRESHOLDS}
        class_counts = Counter()
        per_image_small_ratio = []
        per_class_per_image = defaultdict(list)
        box_areas = []  # normalized areas
        box_aspect_ratios = []  # h/w

        for lf in label_files:
            ann = parse_label_file(lf)
            counts.append(len(ann))
            cls_in_img = Counter()
            small_in_img = 0
            for cid, x, y, w, h in ann:
                # Classify for each threshold
                for t_label, small_px, medium_px in SIZE_THRESHOLDS:
                    sz = classify_size(w, h, small_px, medium_px)
                    size_counts_multi[t_label][sz] += 1

                # Use default threshold for small_in_img ratio
                default_sz = classify_size(w, h, 32, 96)
                class_counts[cid] += 1
                cls_in_img[cid] += 1
                if default_sz == "small":
                    small_in_img += 1
                box_areas.append(w * h)
                if w > 0:
                    box_aspect_ratios.append(h / w)
            for cid, c in cls_in_img.items():
                per_class_per_image[cid].append(c)
            per_image_small_ratio.append(
                small_in_img / len(ann) if ann else 0.0
            )

        total = sum(class_counts.values())
        # Keep backward-compatible "size_counts" as the default (32/96)
        stats[split] = {
            "n_images": len(counts),
            "n_instances": total,
            "counts": counts,
            "size_counts": dict(size_counts_multi[DEFAULT_THRESHOLD_LABEL]),
            "size_counts_multi": {tl: dict(sc) for tl, sc in size_counts_multi.items()},
            "class_counts": dict(class_counts),
            "per_image_small_ratio": per_image_small_ratio,
            "per_class_per_image": dict(per_class_per_image),
            "box_areas": box_areas,
            "box_aspect_ratios": box_aspect_ratios,
        }
    return stats


# ─────────────────────────────────────────────────────────────────
# Per-dataset print: detailed instance + per-split distributions
# ─────────────────────────────────────────────────────────────────
def print_dataset_summary(stats, label):
    print(f"\n{'=' * 70}")
    print(f"  DATASET SUMMARY: {label}")
    print(f"{'=' * 70}")

    for split in SPLITS:
        if split not in stats:
            continue
        s = stats[split]
        counts = s["counts"]
        if not counts:
            continue

        print(f"\n  📂 {split.upper()}  ({s['n_images']} images, "
              f"{s['n_instances']} instances)")
        print(f"     Inst/img    : mean {statistics.mean(counts):.2f} | "
              f"median {statistics.median(counts):.0f} | "
              f"max {max(counts)} | "
              f"std {statistics.stdev(counts) if len(counts) > 1 else 0:.2f}")

        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        pct_str = []
        for p in [50, 90, 95, 99]:
            idx = min(int(n * p / 100), n - 1)
            pct_str.append(f"P{p}={sorted_counts[idx]}")
        print(f"     Percentiles : {' | '.join(pct_str)}")

        # Density buckets
        crowded = sum(1 for c in counts if c >= 8)
        very_crowded = sum(1 for c in counts if c >= 12)
        print(f"     Density     : ≥8 obj: {crowded} ({100*crowded/n:.1f}%)"
              f"  |  ≥12 obj: {very_crowded} ({100*very_crowded/n:.1f}%)")

        total = s["n_instances"]
        # Show size distribution for ALL thresholds
        for t_label, small_px, medium_px in SIZE_THRESHOLDS:
            sc = s["size_counts_multi"][t_label]
            size_str = []
            for sz in ["small", "medium", "large"]:
                c = sc.get(sz, 0)
                size_str.append(f"{sz[0].upper()}={c:>5d} ({format_pct(c, total):>5.1f}%)")
            marker = " *" if t_label == DEFAULT_THRESHOLD_LABEL else "  "
            print(f"     Sizes [{t_label:>6s}]{marker}: {' | '.join(size_str)}")
        print(f"     (* = COCO default threshold)")

        class_str = []
        for cid, name in enumerate(CLASS_NAMES):
            c = s["class_counts"].get(cid, 0)
            class_str.append(f"{name}={format_pct(c, total):.1f}%")
        print(f"     Classes     : {' | '.join(class_str)}")

        if s["box_areas"]:
            med_area = statistics.median(s["box_areas"]) * IMAGE_SIZE * IMAGE_SIZE
            med_ar = statistics.median(s["box_aspect_ratios"]) if s["box_aspect_ratios"] else 0
            print(f"     Box stats   : median area {med_area:.0f}px² | "
                  f"median H/W ratio {med_ar:.2f}")


# ─────────────────────────────────────────────────────────────────
# Per-dataset shift detection
# ─────────────────────────────────────────────────────────────────
def print_shift_report(stats, label):
    print(f"\n  🔍 TRAIN ↔ TEST SHIFT  [{label}]")
    print("  " + "─" * 66)
    if "train" not in stats or "test" not in stats:
        print("     (missing train or test split)")
        return

    tr = stats["train"]
    te = stats["test"]
    if not tr["n_instances"] or not te["n_instances"]:
        return

    # Size shift for each threshold
    for t_label, _, _ in SIZE_THRESHOLDS:
        marker = " *" if t_label == DEFAULT_THRESHOLD_LABEL else ""
        print(f"     ── Threshold [{t_label}]{marker} ──")
        tr_sc = tr["size_counts_multi"][t_label]
        te_sc = te["size_counts_multi"][t_label]
        for sz in ["small", "medium", "large"]:
            tr_pct = format_pct(tr_sc.get(sz, 0), tr["n_instances"])
            te_pct = format_pct(te_sc.get(sz, 0), te["n_instances"])
            rel = ((te_pct - tr_pct) / tr_pct * 100) if tr_pct else 0
            tag = "⚠️" if abs(rel) > 15 else "✓"
            print(f"        {tag} {sz:>6s}: train {tr_pct:5.1f}% → test {te_pct:5.1f}%  "
                  f"(rel {rel:+5.1f}%)")

    # Class shift (unchanged)
    print(f"     ── Class shift ──")
    for cid, name in enumerate(CLASS_NAMES):
        tr_pct = format_pct(tr["class_counts"].get(cid, 0), tr["n_instances"])
        te_pct = format_pct(te["class_counts"].get(cid, 0), te["n_instances"])
        rel = ((te_pct - tr_pct) / tr_pct * 100) if tr_pct else 0
        tag = "⚠️" if abs(rel) > 15 else "✓"
        print(f"        {tag} {name:>9s}: train {tr_pct:5.1f}% → test {te_pct:5.1f}%  "
              f"(rel {rel:+5.1f}%)")


# ─────────────────────────────────────────────────────────────────
# NEW: cross-dataset side-by-side comparison
# ─────────────────────────────────────────────────────────────────
def cross_dataset_comparison(datasets):
    """
    datasets: dict of label -> stats (output of compute_dataset_stats)
    Produces side-by-side tables for size, class, density across all datasets.
    """
    labels = list(datasets.keys())

    print("\n" + "=" * 70)
    print("  📊 CROSS-DATASET COMPARISON (SIDE-BY-SIDE)")
    print("=" * 70)

    # ── Table 1: Per-split image and instance counts ──
    print("\n  ── Image / instance counts per split ──")
    header = f"  {'Split':<8} " + " ".join(f"{lbl:<22}" for lbl in labels)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for split in SPLITS:
        row = f"  {split:<8} "
        for lbl in labels:
            if split in datasets[lbl]:
                s = datasets[lbl][split]
                row += f"{s['n_images']:>5d} img / {s['n_instances']:>6d} inst  "
            else:
                row += "—".center(22) + " "
        print(row)

    # ── Table 2: Size distribution per split, per dataset, per threshold ──
    for split in SPLITS:
        print(f"\n  ── Size distribution: {split.upper()} ──")
        for t_label, small_px, medium_px in SIZE_THRESHOLDS:
            marker = " *" if t_label == DEFAULT_THRESHOLD_LABEL else ""
            print(f"     Threshold [{t_label}]{marker}  (small<{small_px}px, med<{medium_px}px)")
            header = f"       {'Size':<8} " + " ".join(f"{lbl:>22}" for lbl in labels)
            print(header)
            print("       " + "─" * (len(header) - 7))
            for sz in ["small", "medium", "large"]:
                row = f"       {sz:<8} "
                for lbl in labels:
                    if split in datasets[lbl]:
                        s = datasets[lbl][split]
                        sc = s["size_counts_multi"][t_label]
                        c = sc.get(sz, 0)
                        pct = format_pct(c, s["n_instances"])
                        row += f"{c:>6d} ({pct:>5.1f}%)       "
                    else:
                        row += "—".rjust(22) + " "
                print(row)

    # ── Table 3: Class distribution per split, per dataset ──
    for split in SPLITS:
        print(f"\n  ── Class distribution: {split.upper()} ──")
        header = f"  {'Class':<10} " + " ".join(f"{lbl:>22}" for lbl in labels)
        print(header)
        print("  " + "─" * (len(header) - 2))
        for cid, name in enumerate(CLASS_NAMES):
            row = f"  {name:<10} "
            for lbl in labels:
                if split in datasets[lbl]:
                    s = datasets[lbl][split]
                    pct = format_pct(s["class_counts"].get(cid, 0), s["n_instances"])
                    row += f"{pct:>20.1f}%  "
                else:
                    row += "—".rjust(22) + " "
            print(row)

    # ── Table 4: Per-image density (crowdedness) per split ──
    print(f"\n  ── Mean instances per image per split ──")
    header = f"  {'Split':<8} " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for split in SPLITS:
        row = f"  {split:<8} "
        for lbl in labels:
            if split in datasets[lbl]:
                s = datasets[lbl][split]
                mean_inst = statistics.mean(s["counts"]) if s["counts"] else 0
                row += f"{mean_inst:>21.2f}  "
            else:
                row += "—".rjust(22) + " "
        print(row)

    # ── Table 5: Train↔test shift summary across datasets ──
    print(f"\n  ── Train ↔ Test SHIFT (relative %) ──")

    # Size shift per threshold
    for t_label, small_px, medium_px in SIZE_THRESHOLDS:
        marker = " *" if t_label == DEFAULT_THRESHOLD_LABEL else ""
        print(f"\n     Threshold [{t_label}]{marker}")
        header = f"     {'Size':<10} " + " ".join(f"{lbl:>22}" for lbl in labels)
        print(header)
        print("     " + "─" * (len(header) - 5))
        for sz in ["small", "medium", "large"]:
            row = f"     {sz:<10} "
            for lbl in labels:
                ds = datasets[lbl]
                if "train" not in ds or "test" not in ds:
                    row += "—".rjust(22) + " "
                    continue
                tr = ds["train"]
                te = ds["test"]
                if not tr["n_instances"] or not te["n_instances"]:
                    row += "—".rjust(22) + " "
                    continue
                tr_pct = format_pct(tr["size_counts_multi"][t_label].get(sz, 0), tr["n_instances"])
                te_pct = format_pct(te["size_counts_multi"][t_label].get(sz, 0), te["n_instances"])
                rel = ((te_pct - tr_pct) / tr_pct * 100) if tr_pct else 0
                row += f"{rel:>+20.1f}%  "
            print(row)

    # Class shift
    print(f"\n     Class shift")
    header = f"     {'Class':<10} " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    print("     " + "─" * (len(header) - 5))
    for cid, name in enumerate(CLASS_NAMES):
        row = f"     {name:<10} "
        for lbl in labels:
            ds = datasets[lbl]
            if "train" not in ds or "test" not in ds:
                row += "—".rjust(22) + " "
                continue
            tr = ds["train"]
            te = ds["test"]
            if not tr["n_instances"] or not te["n_instances"]:
                row += "—".rjust(22) + " "
                continue
            tr_pct = format_pct(tr["class_counts"].get(cid, 0), tr["n_instances"])
            te_pct = format_pct(te["class_counts"].get(cid, 0), te["n_instances"])
            rel = ((te_pct - tr_pct) / tr_pct * 100) if tr_pct else 0
            row += f"{rel:>+20.1f}%  "
        print(row)


# ─────────────────────────────────────────────────────────────────
# NEW: fidelity report — does the ablation faithfully represent the full?
# ─────────────────────────────────────────────────────────────────
def fidelity_report(datasets, reference_label):
    """
    For each ablation dataset, compute how much its distributions deviate from
    the reference (full) dataset. Lower = better proxy.

    Reports max absolute deviation (in pp) and mean absolute deviation across:
      - size distribution (all 3 sizes × 3 splits = 9 metrics)
      - class distribution (3 classes × 3 splits = 9 metrics)
    """
    print("\n" + "=" * 70)
    print("  🎯 ABLATION FIDELITY REPORT")
    print("  (How well does each ablation reproduce the FULL dataset?)")
    print("=" * 70)

    if reference_label not in datasets:
        print(f"  Reference dataset '{reference_label}' not found.")
        return

    ref = datasets[reference_label]
    other_labels = [lbl for lbl in datasets if lbl != reference_label]

    if not other_labels:
        print("  (only one dataset provided — nothing to compare)")
        return

    print(f"\n  Reference: {reference_label}\n")

    for lbl in other_labels:
        target = datasets[lbl]
        class_deviations = []

        print(f"  ── {lbl} vs {reference_label} ──")

        # Size deviations per threshold
        for t_label, small_px, medium_px in SIZE_THRESHOLDS:
            size_deviations = []
            marker = " *" if t_label == DEFAULT_THRESHOLD_LABEL else ""

            for split in SPLITS:
                if split not in ref or split not in target:
                    continue
                r = ref[split]
                t = target[split]
                if not r["n_instances"] or not t["n_instances"]:
                    continue

                r_sc = r["size_counts_multi"][t_label]
                t_sc = t["size_counts_multi"][t_label]
                for sz in ["small", "medium", "large"]:
                    r_pct = format_pct(r_sc.get(sz, 0), r["n_instances"])
                    t_pct = format_pct(t_sc.get(sz, 0), t["n_instances"])
                    size_deviations.append((split, sz, t_pct - r_pct))

            size_abs = [abs(d) for _, _, d in size_deviations]
            size_max = max(size_abs) if size_abs else 0
            size_mean = statistics.mean(size_abs) if size_abs else 0

            size_deviations.sort(key=lambda x: abs(x[2]), reverse=True)
            worst_str = ""
            if size_deviations:
                split, sz, dev = size_deviations[0]
                worst_str = f" (worst: {split}/{sz} = {dev:+.2f}pp)"

            print(f"     Size [{t_label:>6s}]{marker}: max {size_max:.2f}pp | mean {size_mean:.2f}pp{worst_str}")

        # Class deviations (threshold-independent)
        for split in SPLITS:
            if split not in ref or split not in target:
                continue
            r = ref[split]
            t = target[split]
            if not r["n_instances"] or not t["n_instances"]:
                continue

            for cid, name in enumerate(CLASS_NAMES):
                r_pct = format_pct(r["class_counts"].get(cid, 0), r["n_instances"])
                t_pct = format_pct(t["class_counts"].get(cid, 0), t["n_instances"])
                class_deviations.append((split, name, t_pct - r_pct))

        class_abs = [abs(d) for _, _, d in class_deviations]
        class_max = max(class_abs) if class_abs else 0
        class_mean = statistics.mean(class_abs) if class_abs else 0

        class_deviations.sort(key=lambda x: abs(x[2]), reverse=True)
        worst_cls_str = ""
        if class_deviations:
            split, name, dev = class_deviations[0]
            worst_cls_str = f" (worst: {split}/{name} = {dev:+.2f}pp)"

        print(f"     Class deviation: max {class_max:.2f}pp | mean {class_mean:.2f}pp{worst_cls_str}")

        # Quality grade (use default threshold size + class)
        default_size_devs = []
        for split in SPLITS:
            if split not in ref or split not in target:
                continue
            r = ref[split]
            t = target[split]
            if not r["n_instances"] or not t["n_instances"]:
                continue
            r_sc = r["size_counts_multi"][DEFAULT_THRESHOLD_LABEL]
            t_sc = t["size_counts_multi"][DEFAULT_THRESHOLD_LABEL]
            for sz in ["small", "medium", "large"]:
                r_pct = format_pct(r_sc.get(sz, 0), r["n_instances"])
                t_pct = format_pct(t_sc.get(sz, 0), t["n_instances"])
                default_size_devs.append(abs(t_pct - r_pct))

        worst = max(max(default_size_devs) if default_size_devs else 0, class_max)
        if worst < 0.5:
            grade = "🟢 EXCELLENT (deviations < 0.5pp)"
        elif worst < 1.0:
            grade = "🟢 GOOD (deviations < 1.0pp)"
        elif worst < 2.0:
            grade = "🟡 ACCEPTABLE (deviations < 2.0pp)"
        else:
            grade = "🔴 POOR (deviations ≥ 2.0pp)"
        print(f"     Verdict        : {grade}")
        print()


# ─────────────────────────────────────────────────────────────────
# Filename / source overlap (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
def filename_source_analysis(dataset_root, label):
    print("\n" + "─" * 70)
    print(f"  🎬 FILENAME / SOURCE OVERLAP  [{label}]")
    print("─" * 70)

    split_stems = {}
    split_sources = {}

    def extract_source(stem):
        s = stem
        s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[-_]?\d{3,}$", "", s)
        s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
        return s

    for split in SPLITS:
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        label_files = get_label_files(labels_dir)
        stems = [Path(lf).stem for lf in label_files]
        split_stems[split] = set(stems)
        split_sources[split] = Counter(extract_source(s) for s in stems)

    print("  Exact filename overlap:")
    any_overlap = False
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 >= s2 or s1 not in split_stems or s2 not in split_stems:
                continue
            overlap = split_stems[s1] & split_stems[s2]
            tag = "⚠️" if overlap else "✓"
            print(f"     {tag} {s1} ∩ {s2}: {len(overlap)} shared")
            if overlap:
                any_overlap = True
    if not any_overlap:
        print("     (no exact overlaps — good)")

    print("\n  Source-prefix overlap:")
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 >= s2 or s1 not in split_sources or s2 not in split_sources:
                continue
            common = set(split_sources[s1]) & set(split_sources[s2])
            tag = "⚠️" if len(common) > 5 else ("ℹ️" if common else "✓")
            print(f"     {tag} {s1} ∩ {s2}: {len(common)} shared sources")


# ─────────────────────────────────────────────────────────────────
# Perceptual hash leakage check
# ─────────────────────────────────────────────────────────────────
def perceptual_hash_analysis(dataset_root, label, max_per_split=2000, hash_distance=4):
    print("\n" + "─" * 70)
    print(f"  🖼️  PERCEPTUAL HASH LEAKAGE  [{label}]")
    print("─" * 70)

    if not HASH_AVAILABLE:
        print("  ⚠️ Skipped: pip install pillow imagehash")
        return

    split_hashes = {}
    for split in SPLITS:
        images_dir = os.path.join(dataset_root, split, "images")
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(images_dir):
            continue
        label_files = get_label_files(labels_dir)
        hashes = {}
        n = 0
        for lf in label_files:
            if max_per_split and n >= max_per_split:
                break
            img_path = get_image_for_label(lf, images_dir)
            if not img_path:
                continue
            try:
                with Image.open(img_path) as img:
                    h = imagehash.phash(img.convert("RGB"))
                hashes[Path(lf).stem] = h
                n += 1
            except Exception:
                continue
        split_hashes[split] = hashes
        print(f"  Hashed {len(hashes)} images in {split}")

    print("\n  Cross-split near-duplicates (Hamming ≤ "
          f"{hash_distance}) — LEAKAGE CHECK:")
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 >= s2 or s1 not in split_hashes or s2 not in split_hashes:
                continue
            h1 = list(split_hashes[s1].items())[:2000]
            h2 = list(split_hashes[s2].items())[:2000]
            leaks = 0
            for stem1, hash1 in h1:
                for stem2, hash2 in h2:
                    if hash1 - hash2 <= hash_distance:
                        leaks += 1
                        break
            tag = "⚠️ LEAKAGE" if leaks > 5 else ("ℹ️" if leaks else "✓")
            print(f"     {tag} {s1} ↔ {s2}: {leaks} near-duplicate matches")


# ─────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────
def run_full_analysis(datasets_config, run_hash=False):
    """
    datasets_config: list of (label, path) tuples
    """
    all_stats = {}

    # ── Phase 1: per-dataset analysis ──
    for label, path in datasets_config:
        print("\n" + "█" * 70)
        print(f"  🔶 DATASET: {label}")
        print(f"     Path: {path}")
        print("█" * 70)

        if not os.path.isdir(path):
            print(f"  ⚠️ Directory not found: {path}")
            continue

        stats = compute_dataset_stats(path)
        all_stats[label] = stats

        print_dataset_summary(stats, label)
        print_shift_report(stats, label)
        filename_source_analysis(path, label)
        if run_hash:
            perceptual_hash_analysis(path, label)

    # ── Phase 2: cross-dataset comparison ──
    if len(all_stats) >= 2:
        cross_dataset_comparison(all_stats)

    # ── Phase 3: fidelity report (vs FULL) ──
    full_label = datasets_config[0][0]  # first is treated as reference
    if len(all_stats) >= 2 and full_label in all_stats:
        fidelity_report(all_stats, full_label)

    print("\n" + "=" * 70)
    print("  ✅ Analysis complete.")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Order matters: first entry is treated as the REFERENCE for fidelity check
    datasets = [
        ("FULL",
         r"/home/constantin/Doctorat/GunDatasetHistogram"),
        # ("ABLATION_OLD",
        #  r"/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubset"),
        ("ABLATION",
         r"/home/constantin/Doctorat/GunDatasetHistogram17percentage"),
    ]

    # Set run_hash=True if pillow+imagehash are installed and you want leakage detection
    run_full_analysis(datasets, run_hash=False)