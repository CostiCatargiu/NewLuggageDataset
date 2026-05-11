"""
Extended dataset analysis for the luggage detection project.

Adds to the original instance-distribution analysis:
  1. Per-split SIZE distribution (small/medium/large) — to detect train/test shift
  2. Per-split CLASS distribution — to detect class imbalance shift
  3. Cross-split FILENAME / source overlap detection (video-frame leakage)
  4. Near-duplicate detection via perceptual hashing (image leakage)
  5. Side-by-side comparison of FULL vs ABLATION dataset

Usage:
    python dataset_analysis.py

Requires: Pillow (PIL)  — install with `pip install pillow`
          imagehash      — install with `pip install imagehash`  (optional, for dup detection)
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
IMAGE_SIZE = 640                # used for COCO-style size thresholds
SMALL_AREA_NORM = (32 * 32) / (IMAGE_SIZE * IMAGE_SIZE)  # 0.0025
MEDIUM_AREA_NORM = (96 * 96) / (IMAGE_SIZE * IMAGE_SIZE)  # 0.0225
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Optional dependencies — script degrades gracefully if not installed
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
    """Yields (class_id, x_center, y_center, width, height) tuples."""
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                yield cid, x, y, w, h
            except ValueError:
                continue


def classify_size(w_norm, h_norm):
    """COCO-style: small <32², medium <96², large >=96² (at 640px)."""
    area = w_norm * h_norm
    if area < SMALL_AREA_NORM:
        return "small"
    elif area < MEDIUM_AREA_NORM:
        return "medium"
    else:
        return "large"


def get_image_for_label(label_path, images_dir):
    """Find the matching image file for a given label .txt."""
    stem = Path(label_path).stem
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────
# Original analysis (kept intact, lightly cleaned)
# ─────────────────────────────────────────────────────────────────
def detailed_instance_analysis(dataset_root):
    """Original instance-per-image analysis."""

    splits = ["train", "valid", "test"]
    print("=" * 70)
    print("         DETAILED INSTANCES-PER-IMAGE ANALYSIS")
    print("=" * 70)

    all_counts = []

    for split in splits:
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue

        label_files = get_label_files(labels_dir)
        counts = []
        per_class_counts = {}

        for lf in label_files:
            ann = list(parse_label_file(lf))
            counts.append(len(ann))
            class_in_image = Counter(cid for cid, *_ in ann)
            for cid, cnt in class_in_image.items():
                per_class_counts.setdefault(cid, []).append(cnt)

        if not counts:
            continue
        all_counts.extend(counts)

        print(f"\n{'─' * 70}")
        print(f"  📂 {split.upper()} ({len(counts)} images)")
        print(f"{'─' * 70}")

        print(f"\n  ── Basic Statistics ──")
        print(f"     Mean   : {statistics.mean(counts):.2f}")
        print(f"     Median : {statistics.median(counts):.2f}")
        print(f"     Mode   : {statistics.mode(counts)}")
        print(f"     Min    : {min(counts)}")
        print(f"     Max    : {max(counts)}")
        print(f"     Std    : {statistics.stdev(counts) if len(counts) > 1 else 0:.2f}")
        print(f"     Total  : {sum(counts)}")

        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        print(f"\n  ── Percentiles ──")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = min(int(n * p / 100), n - 1)
            print(f"     P{p:>2d}  : {sorted_counts[idx]}")


# ─────────────────────────────────────────────────────────────────
# NEW: per-split size + class distribution
# ─────────────────────────────────────────────────────────────────
def split_distribution_analysis(dataset_root):
    """Compute per-split size and class distributions for shift detection."""
    splits = ["train", "valid", "test"]
    summary = {}

    print("\n" + "=" * 70)
    print("         📊 PER-SPLIT SIZE & CLASS DISTRIBUTION")
    print("=" * 70)

    for split in splits:
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        label_files = get_label_files(labels_dir)

        size_counts = Counter()
        class_counts = Counter()
        total = 0
        per_image_small_ratio = []

        for lf in label_files:
            ann = list(parse_label_file(lf))
            if not ann:
                per_image_small_ratio.append(0.0)
                continue
            small_in_img = 0
            for cid, x, y, w, h in ann:
                size = classify_size(w, h)
                size_counts[size] += 1
                class_counts[cid] += 1
                total += 1
                if size == "small":
                    small_in_img += 1
            per_image_small_ratio.append(small_in_img / len(ann))

        summary[split] = {
            "size_counts": size_counts,
            "class_counts": class_counts,
            "total": total,
            "per_image_small_ratio": per_image_small_ratio,
        }

        print(f"\n  📂 {split.upper()}")
        print(f"     Total annotations: {total}")
        print(f"     ── Size distribution ──")
        for sz in ["small", "medium", "large"]:
            c = size_counts[sz]
            pct = 100 * c / total if total else 0
            print(f"       {sz:>6s} : {c:>6d} ({pct:5.1f}%)")
        print(f"     ── Class distribution ──")
        for cid in sorted(class_counts):
            c = class_counts[cid]
            pct = 100 * c / total if total else 0
            print(f"       Class {cid} : {c:>6d} ({pct:5.1f}%)")
        if per_image_small_ratio:
            avg_small = statistics.mean(per_image_small_ratio)
            print(f"     Avg per-image small-object ratio: {avg_small:.3f}")

    # ── Shift detection ──
    if "train" in summary and "test" in summary:
        print("\n" + "─" * 70)
        print("  🔍 TRAIN ↔ TEST SHIFT DETECTION")
        print("─" * 70)
        train_total = summary["train"]["total"]
        test_total = summary["test"]["total"]
        if train_total and test_total:
            for sz in ["small", "medium", "large"]:
                tr = 100 * summary["train"]["size_counts"][sz] / train_total
                te = 100 * summary["test"]["size_counts"][sz] / test_total
                shift = te - tr
                rel = (shift / tr * 100) if tr else 0
                tag = "⚠️ shift" if abs(rel) > 15 else "✓ ok"
                print(f"     {sz:>6s}: train {tr:5.1f}% → test {te:5.1f}%  "
                      f"(Δ {shift:+5.1f}pp / {rel:+5.1f}% rel)  {tag}")
            for cid in sorted(set(summary["train"]["class_counts"]) |
                              set(summary["test"]["class_counts"])):
                tr = 100 * summary["train"]["class_counts"][cid] / train_total
                te = 100 * summary["test"]["class_counts"][cid] / test_total
                shift = te - tr
                rel = (shift / tr * 100) if tr else 0
                tag = "⚠️ shift" if abs(rel) > 15 else "✓ ok"
                print(f"     Class{cid:>2d}: train {tr:5.1f}% → test {te:5.1f}%  "
                      f"(Δ {shift:+5.1f}pp / {rel:+5.1f}% rel)  {tag}")

    return summary


# ─────────────────────────────────────────────────────────────────
# NEW: cross-split filename / source overlap (video-frame leakage)
# ─────────────────────────────────────────────────────────────────
def filename_source_analysis(dataset_root):
    """Detect possible video-frame leakage via filename patterns."""
    print("\n" + "=" * 70)
    print("         🎬 FILENAME / SOURCE OVERLAP DETECTION")
    print("=" * 70)

    splits = ["train", "valid", "test"]
    split_stems = {}
    split_sources = {}

    # Heuristic: extract the "source" prefix from a filename.
    # Roboflow / video-derived datasets typically have names like:
    #   videoName_frame_00012_jpg.rf.<hash>.jpg
    #   scene01_001.jpg
    # We strip frame numbers, hashes, and roboflow suffixes to recover the source.
    def extract_source(stem):
        s = stem
        s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)  # roboflow hash
        s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)  # frame indicator
        s = re.sub(r"[-_]?\d{3,}$", "", s)                            # trailing frame number
        s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
        return s

    for split in splits:
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        label_files = get_label_files(labels_dir)
        stems = [Path(lf).stem for lf in label_files]
        split_stems[split] = set(stems)
        split_sources[split] = Counter(extract_source(s) for s in stems)

    # ── Exact filename overlap (the worst case) ──
    print("\n  ── Exact filename overlap across splits ──")
    found_any = False
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2 or s1 not in split_stems or s2 not in split_stems:
                continue
            overlap = split_stems[s1] & split_stems[s2]
            if overlap:
                found_any = True
                print(f"     ⚠️ {s1} ∩ {s2}: {len(overlap)} shared filenames")
            else:
                print(f"     ✓ {s1} ∩ {s2}: no shared filenames")
    if not found_any:
        print("     (no exact overlaps — good)")

    # ── Source overlap (video-frame leakage) ──
    print("\n  ── Source-prefix overlap (video-frame leakage check) ──")
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2 or s1 not in split_sources or s2 not in split_sources:
                continue
            common = set(split_sources[s1]) & set(split_sources[s2])
            if common:
                # how many frames each side has from the overlapping sources
                n1 = sum(split_sources[s1][src] for src in common)
                n2 = sum(split_sources[s2][src] for src in common)
                tag = "⚠️" if len(common) > 5 else "ℹ️"
                print(f"     {tag} {s1} ∩ {s2}: {len(common)} shared sources "
                      f"({n1} frames in {s1}, {n2} in {s2})")
                if len(common) <= 10:
                    for src in list(common)[:10]:
                        print(f"        - '{src}': {split_sources[s1][src]}/{split_sources[s2][src]}")
            else:
                print(f"     ✓ {s1} ∩ {s2}: no shared sources")

    print("\n  Note: source extraction is heuristic. If your filenames don't follow")
    print("  a video-frame pattern, this section may be unreliable.")


# ─────────────────────────────────────────────────────────────────
# NEW: perceptual-hash near-duplicate detection
# ─────────────────────────────────────────────────────────────────
def perceptual_hash_analysis(dataset_root, max_per_split=None, hash_distance=4):
    """Detect near-duplicate images within and across splits using pHash."""
    print("\n" + "=" * 70)
    print("         🖼️  PERCEPTUAL HASH NEAR-DUPLICATE DETECTION")
    print("=" * 70)

    if not HASH_AVAILABLE:
        print("\n  ⚠️ Skipped: install Pillow + imagehash to enable this check.")
        print("     pip install pillow imagehash")
        return

    splits = ["train", "valid", "test"]
    split_hashes = {}

    for split in splits:
        images_dir = os.path.join(dataset_root, split, "images")
        labels_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(images_dir):
            continue
        label_files = get_label_files(labels_dir)
        hashes = {}  # stem -> phash
        n_processed = 0
        for lf in label_files:
            if max_per_split and n_processed >= max_per_split:
                break
            img_path = get_image_for_label(lf, images_dir)
            if not img_path:
                continue
            try:
                with Image.open(img_path) as img:
                    h = imagehash.phash(img.convert("RGB"))
                hashes[Path(lf).stem] = h
                n_processed += 1
            except Exception:
                continue
        split_hashes[split] = hashes
        print(f"  Hashed {len(hashes)} images in {split}")

    # ── Within-split near-duplicates ──
    print("\n  ── Within-split near-duplicates (Hamming distance ≤ "
          f"{hash_distance}) ──")
    for split, hashes in split_hashes.items():
        items = list(hashes.items())
        dup_pairs = 0
        # Sample to keep this O(n^2) tractable on large splits
        sample = items if len(items) <= 2000 else items[:2000]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                if sample[i][1] - sample[j][1] <= hash_distance:
                    dup_pairs += 1
        note = "" if len(items) <= 2000 else f"  (sampled first 2000 of {len(items)})"
        tag = "⚠️" if dup_pairs > 50 else "✓"
        print(f"     {tag} {split}: {dup_pairs} near-duplicate pairs{note}")

    # ── Cross-split near-duplicates (this is the LEAKAGE check) ──
    print("\n  ── Cross-split near-duplicates (LEAKAGE check) ──")
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2 or s1 not in split_hashes or s2 not in split_hashes:
                continue
            h1 = list(split_hashes[s1].items())
            h2 = list(split_hashes[s2].items())
            # cap to keep tractable
            h1 = h1[:2000]
            h2 = h2[:2000]
            leaks = 0
            examples = []
            for stem1, hash1 in h1:
                for stem2, hash2 in h2:
                    if hash1 - hash2 <= hash_distance:
                        leaks += 1
                        if len(examples) < 5:
                            examples.append((stem1, stem2, hash1 - hash2))
                        break  # one match per s1 image is enough to count
            tag = "⚠️ LEAKAGE" if leaks > 5 else ("ℹ️" if leaks > 0 else "✓")
            print(f"     {tag} {s1} ↔ {s2}: {leaks} near-duplicate pairs")
            for ex in examples:
                print(f"        - '{ex[0]}' ↔ '{ex[1]}' (dist={ex[2]})")


# ─────────────────────────────────────────────────────────────────
# Top-level: run everything for a given dataset root
# ─────────────────────────────────────────────────────────────────
def full_analysis(dataset_root, label, run_hash=True):
    print("\n\n" + "=" * 70)
    print(f"  ANALYZING: {label}")
    print(f"  Path: {dataset_root}")
    print("=" * 70)
    detailed_instance_analysis(dataset_root)
    split_distribution_analysis(dataset_root)
    filename_source_analysis(dataset_root)
    if run_hash:
        perceptual_hash_analysis(dataset_root, max_per_split=2000, hash_distance=4)


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ORIGINAL = r"/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12"
    ABLATION1 = r"/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubset"
    ABLATION2 = r"/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubsetNEW"

    print("\n" + "🔶" * 35)
    print("  ORIGINAL (FULL) DATASET")
    print("🔶" * 35)
    full_analysis(ORIGINAL, "ORIGINAL (FULL)", run_hash=True)

    print("\n\n" + "🔷" * 35)
    print("  ABLATION (30%) DATASET")
    print("🔷" * 35)
    full_analysis(ABLATION, "ABLATION (30%)", run_hash=True)

    print("\n\n" + "=" * 70)
    print("  ✅ Full analysis complete.")
    print("=" * 70)