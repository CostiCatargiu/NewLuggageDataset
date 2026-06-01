#!/usr/bin/env python3
"""
Deep Dataset Analysis — comprehensive quality audit for a merged image dataset.

Analyzes:
  1. BASIC STATS: total images, labels, orphans (img without label, label without img)
  2. IMAGE PROPERTIES: resolution distribution, aspect ratios, file sizes, formats
  3. ANNOTATION STATS: class distribution, instances/image, bbox sizes, aspect ratios
  4. SIZE ANALYSIS: S/M/L at multiple thresholds (24/72, 32/96, 48/144, 64/192)
  5. DUPLICATE DETECTION:
     - Exact duplicates (MD5 hash)
     - Near-duplicates (perceptual hash — similar frames from same video)
  6. SIMILAR FRAME DETECTION: finds clusters of visually similar images
  7. SOURCE GROUP ANALYSIS: groups by filename prefix, detects video sequences
  8. ANNOTATION QUALITY: empty labels, invalid bboxes, out-of-range values
  9. CLASS CO-OCCURRENCE: which classes appear together
  10. PER-CLASS SIZE DISTRIBUTION: size breakdown per class

Output: detailed report to console + saved to txt file.

Requirements:
  pip install Pillow imagehash

Usage:
  python dataset_deep_analysis.py
"""

import os
import re
import sys
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not installed. Image property analysis disabled.")
    print("  pip install Pillow")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("WARNING: imagehash not installed. Near-duplicate detection disabled.")
    print("  pip install imagehash")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_PATH = r"c:\DISK\GunDataset4/train"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
LABELS_DIR = os.path.join(DATASET_PATH, "labels")
OUTPUT_FILE = os.path.join(os.path.dirname(DATASET_PATH), "dataset_deep_analysis.txt")

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
IMAGE_SIZE = 640  # training resolution

# Size thresholds for analysis
SIZE_THRESHOLDS = [
    ("24/72",  24,  72),
    ("32/96",  32,  96),   # COCO default
    ("48/144", 48, 144),
    ("64/192", 64, 192),
]

# Near-duplicate detection: images with hash distance <= this are "similar"
HASH_DISTANCE_THRESHOLD = 8  # lower = stricter (0 = identical)
PHASH_SIZE = 16  # hash resolution (higher = more sensitive)

# Maximum images to process for perceptual hashing (slow on huge datasets)
MAX_HASH_IMAGES = 50000


# =============================================================================
# OUTPUT HELPER
# =============================================================================
class ReportWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []

    def write(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"\nReport saved to: {self.filepath}")


# =============================================================================
# ANNOTATION PARSING
# =============================================================================
def parse_label(label_path):
    """Parse a YOLO label file. Returns list of (class_id, x, y, w, h) and issues."""
    annotations = []
    issues = []

    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    issues.append(f"line {line_num}: too few values ({len(parts)})")
                    continue
                try:
                    cid = int(float(parts[0]))
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Validate ranges
                    if cid < 0 or cid >= len(CLASS_NAMES):
                        issues.append(f"line {line_num}: invalid class_id={cid}")
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        issues.append(f"line {line_num}: center ({x:.4f}, {y:.4f}) out of [0,1]")
                    if not (0 < w <= 1 and 0 < h <= 1):
                        issues.append(f"line {line_num}: size ({w:.4f}, {h:.4f}) out of (0,1]")

                    annotations.append((cid, x, y, w, h))
                except ValueError as e:
                    issues.append(f"line {line_num}: parse error: {e}")
    except Exception as e:
        issues.append(f"file error: {e}")

    return annotations, issues


def classify_size(w_norm, h_norm, small_px, medium_px):
    """Classify object size at given threshold."""
    area = w_norm * h_norm
    small_area = (small_px ** 2) / (IMAGE_SIZE ** 2)
    medium_area = (medium_px ** 2) / (IMAGE_SIZE ** 2)
    if area < small_area:
        return 'small'
    elif area < medium_area:
        return 'medium'
    return 'large'


def extract_source(stem):
    """Extract source prefix from filename (groups video frames, augmentations)."""
    s = stem
    s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]?\d{3,}$", "", s)
    s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
    return s


def detect_augmentation(stem):
    """Check if filename suggests it's an augmented image."""
    # Roboflow augmentations typically have .rf.<hex> suffix
    if re.search(r"\.rf\.[a-f0-9]+$", stem, flags=re.IGNORECASE):
        return True
    # Common augmentation suffixes
    if re.search(r"_(flip|rot|bright|blur|crop|aug|mosaic|cutout)\d*$", stem, flags=re.IGNORECASE):
        return True
    return False


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    report = ReportWriter(OUTPUT_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report.write("=" * 80)
    report.write(f"  DEEP DATASET ANALYSIS")
    report.write(f"  {timestamp}")
    report.write(f"  Dataset: {DATASET_PATH}")
    report.write("=" * 80)

    # =========================================================================
    # 1. BASIC STATS
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  1. BASIC STATS")
    report.write(f"{'=' * 80}")

    all_images = {}
    all_labels = {}

    if os.path.isdir(IMAGES_DIR):
        for f in os.listdir(IMAGES_DIR):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                stem = os.path.splitext(f)[0]
                all_images[stem] = f

    if os.path.isdir(LABELS_DIR):
        for f in os.listdir(LABELS_DIR):
            if f.endswith('.txt'):
                stem = os.path.splitext(f)[0]
                all_labels[stem] = f

    matched = set(all_images.keys()) & set(all_labels.keys())
    orphan_images = set(all_images.keys()) - set(all_labels.keys())
    orphan_labels = set(all_labels.keys()) - set(all_images.keys())

    report.write(f"  Total images:           {len(all_images)}")
    report.write(f"  Total labels:           {len(all_labels)}")
    report.write(f"  Matched pairs:          {len(matched)}")
    report.write(f"  Orphan images (no lbl): {len(orphan_images)}")
    report.write(f"  Orphan labels (no img): {len(orphan_labels)}")

    if orphan_images and len(orphan_images) <= 20:
        report.write(f"    Orphan images: {sorted(orphan_images)}")
    if orphan_labels and len(orphan_labels) <= 20:
        report.write(f"    Orphan labels: {sorted(orphan_labels)}")

    # =========================================================================
    # 2. IMAGE PROPERTIES
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  2. IMAGE PROPERTIES")
    report.write(f"{'=' * 80}")

    widths = []
    heights = []
    file_sizes = []
    formats = Counter()
    broken_images = []

    if HAS_PIL:
        report.write(f"  Scanning {len(matched)} images...")
        for i, stem in enumerate(sorted(matched)):
            if i % 5000 == 0 and i > 0:
                report.write(f"    ...{i}/{len(matched)}")

            img_path = os.path.join(IMAGES_DIR, all_images[stem])
            try:
                file_sizes.append(os.path.getsize(img_path))
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    formats[img.format or 'UNKNOWN'] += 1
            except Exception as e:
                broken_images.append((stem, str(e)))

        if widths:
            report.write(f"\n  Resolution distribution:")
            report.write(f"    Width:  min={min(widths)}, max={max(widths)}, "
                        f"median={statistics.median(widths):.0f}, mean={statistics.mean(widths):.0f}")
            report.write(f"    Height: min={min(heights)}, max={max(heights)}, "
                        f"median={statistics.median(heights):.0f}, mean={statistics.mean(heights):.0f}")

            # Resolution buckets
            res_buckets = Counter()
            for w, h in zip(widths, heights):
                res_buckets[f"{w}x{h}"] += 1
            report.write(f"\n  Top 10 resolutions:")
            for res, count in res_buckets.most_common(10):
                pct = 100.0 * count / len(widths)
                report.write(f"    {res:<20s} {count:>6d} ({pct:>5.1f}%)")

            # Aspect ratios
            ars = [w / h for w, h in zip(widths, heights)]
            report.write(f"\n  Aspect ratios (W/H):")
            report.write(f"    min={min(ars):.3f}, max={max(ars):.3f}, "
                        f"median={statistics.median(ars):.3f}, mean={statistics.mean(ars):.3f}")

            # File sizes
            report.write(f"\n  File sizes:")
            report.write(f"    min={min(file_sizes)/1024:.1f}KB, max={max(file_sizes)/1024:.1f}KB, "
                        f"median={statistics.median(file_sizes)/1024:.1f}KB, "
                        f"total={sum(file_sizes)/1024/1024/1024:.2f}GB")

            # Formats
            report.write(f"\n  Image formats:")
            for fmt, count in formats.most_common():
                report.write(f"    {fmt:<10s} {count:>6d}")

        if broken_images:
            report.write(f"\n  BROKEN IMAGES ({len(broken_images)}):")
            for stem, err in broken_images[:20]:
                report.write(f"    {stem}: {err}")
    else:
        report.write(f"  SKIPPED — install Pillow")

    # =========================================================================
    # 3. ANNOTATION STATS
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  3. ANNOTATION STATS")
    report.write(f"{'=' * 80}")

    class_counts = Counter()
    instances_per_image = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_ars = []  # aspect ratios
    annotation_issues = {}
    empty_labels = []
    per_class_sizes = {t: {cid: Counter() for cid in range(len(CLASS_NAMES))} for t, _, _ in SIZE_THRESHOLDS}
    size_counts = {t: Counter() for t, _, _ in SIZE_THRESHOLDS}
    class_cooccurrence = Counter()  # pairs of classes in same image

    for stem in sorted(matched):
        label_path = os.path.join(LABELS_DIR, all_labels[stem])
        annotations, issues = parse_label(label_path)

        if issues:
            annotation_issues[stem] = issues

        if len(annotations) == 0:
            empty_labels.append(stem)

        instances_per_image.append(len(annotations))

        classes_in_image = set()
        for cid, x, y, w, h in annotations:
            class_counts[cid] += 1
            classes_in_image.add(cid)
            bbox_widths.append(w)
            bbox_heights.append(h)
            area = w * h
            bbox_areas.append(area)
            ar = max(w, h) / max(min(w, h), 1e-9)
            bbox_ars.append(ar)

            for t_label, small_px, medium_px in SIZE_THRESHOLDS:
                sz = classify_size(w, h, small_px, medium_px)
                size_counts[t_label][sz] += 1
                per_class_sizes[t_label][cid][sz] += 1

        # Co-occurrence
        classes_list = sorted(classes_in_image)
        for i in range(len(classes_list)):
            for j in range(i + 1, len(classes_list)):
                pair = (CLASS_NAMES[classes_list[i]], CLASS_NAMES[classes_list[j]])
                class_cooccurrence[pair] += 1

    total_instances = sum(class_counts.values())

    report.write(f"  Total instances:        {total_instances}")
    report.write(f"  Images with labels:     {len(matched)}")
    report.write(f"  Empty labels:           {len(empty_labels)}")

    report.write(f"\n  Instances per image:")
    report.write(f"    min={min(instances_per_image)}, max={max(instances_per_image)}, "
                f"median={statistics.median(instances_per_image):.1f}, "
                f"mean={statistics.mean(instances_per_image):.2f}")
    ipi_buckets = Counter()
    for n in instances_per_image:
        if n == 0:
            ipi_buckets['0'] += 1
        elif n == 1:
            ipi_buckets['1'] += 1
        elif n <= 3:
            ipi_buckets['2-3'] += 1
        elif n <= 5:
            ipi_buckets['4-5'] += 1
        elif n <= 10:
            ipi_buckets['6-10'] += 1
        else:
            ipi_buckets['11+'] += 1
    report.write(f"    distribution:")
    for bucket in ['0', '1', '2-3', '4-5', '6-10', '11+']:
        count = ipi_buckets.get(bucket, 0)
        pct = 100.0 * count / len(instances_per_image) if instances_per_image else 0
        report.write(f"      {bucket:>5s} objects: {count:>6d} images ({pct:>5.1f}%)")

    # Class distribution
    report.write(f"\n  Class distribution:")
    report.write(f"  {'Class':<12} {'Count':>8} {'%':>8}")
    for cid, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cid, 0)
        pct = 100.0 * count / total_instances if total_instances else 0
        report.write(f"  {name:<12} {count:>8} {pct:>7.1f}%")
    report.write(f"  {'TOTAL':<12} {total_instances:>8}")

    # Bbox dimensions
    if bbox_widths:
        report.write(f"\n  Bounding box dimensions (normalized 0-1):")
        report.write(f"    Width:  min={min(bbox_widths):.4f}, max={max(bbox_widths):.4f}, "
                    f"median={statistics.median(bbox_widths):.4f}")
        report.write(f"    Height: min={min(bbox_heights):.4f}, max={max(bbox_heights):.4f}, "
                    f"median={statistics.median(bbox_heights):.4f}")
        report.write(f"    Area:   min={min(bbox_areas):.6f}, max={max(bbox_areas):.6f}, "
                    f"median={statistics.median(bbox_areas):.6f}")

        report.write(f"\n  Bounding box dimensions (pixels at {IMAGE_SIZE}px):")
        px_w = [w * IMAGE_SIZE for w in bbox_widths]
        px_h = [h * IMAGE_SIZE for h in bbox_heights]
        px_a = [a * IMAGE_SIZE * IMAGE_SIZE for a in bbox_areas]
        report.write(f"    Width:  min={min(px_w):.1f}px, max={max(px_w):.1f}px, "
                    f"median={statistics.median(px_w):.1f}px")
        report.write(f"    Height: min={min(px_h):.1f}px, max={max(px_h):.1f}px, "
                    f"median={statistics.median(px_h):.1f}px")
        report.write(f"    Area:   min={min(px_a):.1f}px², max={max(px_a):.1f}px², "
                    f"median={statistics.median(px_a):.1f}px²")

        report.write(f"\n  Aspect ratios (max/min side):")
        report.write(f"    min={min(bbox_ars):.2f}, max={max(bbox_ars):.2f}, "
                    f"median={statistics.median(bbox_ars):.2f}, mean={statistics.mean(bbox_ars):.2f}")

    # =========================================================================
    # 4. SIZE ANALYSIS
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  4. SIZE ANALYSIS (at {IMAGE_SIZE}px training resolution)")
    report.write(f"{'=' * 80}")

    for t_label, small_px, medium_px in SIZE_THRESHOLDS:
        sc = size_counts[t_label]
        total = sum(sc.values())
        report.write(f"\n  Threshold [{t_label}] (small<{small_px}px, med<{medium_px}px):")
        for sz in ['small', 'medium', 'large']:
            count = sc.get(sz, 0)
            pct = 100.0 * count / total if total else 0
            report.write(f"    {sz:<8s} {count:>8d} ({pct:>5.1f}%)")

    # =========================================================================
    # 5. PER-CLASS SIZE DISTRIBUTION
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  5. PER-CLASS SIZE DISTRIBUTION [32/96] (COCO)")
    report.write(f"{'=' * 80}")

    t_label = "32/96"
    report.write(f"\n  {'Class':<12} {'Small':>8} {'Medium':>8} {'Large':>8} {'Total':>8} {'%Small':>8}")
    for cid, name in enumerate(CLASS_NAMES):
        sc = per_class_sizes[t_label][cid]
        s = sc.get('small', 0)
        m = sc.get('medium', 0)
        l = sc.get('large', 0)
        total = s + m + l
        pct_s = 100.0 * s / total if total else 0
        report.write(f"  {name:<12} {s:>8} {m:>8} {l:>8} {total:>8} {pct_s:>7.1f}%")

    # =========================================================================
    # 6. CLASS CO-OCCURRENCE
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  6. CLASS CO-OCCURRENCE (classes in same image)")
    report.write(f"{'=' * 80}")

    if class_cooccurrence:
        for pair, count in class_cooccurrence.most_common():
            report.write(f"    {pair[0]} + {pair[1]}: {count} images")
    else:
        report.write(f"    No co-occurrence found (single-class images only)")

    # =========================================================================
    # 7. SOURCE GROUP ANALYSIS
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  7. SOURCE GROUP ANALYSIS")
    report.write(f"{'=' * 80}")

    source_groups = defaultdict(list)
    augmented_count = 0
    for stem in sorted(matched):
        source = extract_source(stem)
        source_groups[source].append(stem)
        if detect_augmentation(stem):
            augmented_count += 1

    group_sizes = [len(v) for v in source_groups.values()]

    report.write(f"  Total source groups:    {len(source_groups)}")
    report.write(f"  Augmented filenames:    {augmented_count} ({100.0*augmented_count/len(matched):.1f}%)")
    report.write(f"\n  Source group sizes:")
    report.write(f"    min={min(group_sizes)}, max={max(group_sizes)}, "
                f"median={statistics.median(group_sizes):.0f}, mean={statistics.mean(group_sizes):.1f}")

    # Size buckets
    sg_buckets = Counter()
    for s in group_sizes:
        if s == 1:
            sg_buckets['1 (unique)'] += 1
        elif s <= 3:
            sg_buckets['2-3'] += 1
        elif s <= 5:
            sg_buckets['4-5'] += 1
        elif s <= 10:
            sg_buckets['6-10'] += 1
        elif s <= 50:
            sg_buckets['11-50'] += 1
        else:
            sg_buckets['51+'] += 1
    report.write(f"    distribution:")
    for bucket in ['1 (unique)', '2-3', '4-5', '6-10', '11-50', '51+']:
        count = sg_buckets.get(bucket, 0)
        pct = 100.0 * count / len(source_groups) if source_groups else 0
        report.write(f"      {bucket:>12s}: {count:>6d} groups ({pct:>5.1f}%)")

    # Largest groups
    largest = sorted(source_groups.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    report.write(f"\n  Top 15 largest source groups:")
    for source, stems in largest:
        report.write(f"    {source:<50s} {len(stems):>5d} images")

    # =========================================================================
    # 8. EXACT DUPLICATE DETECTION (MD5)
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  8. EXACT DUPLICATE DETECTION (MD5 hash)")
    report.write(f"{'=' * 80}")

    report.write(f"  Hashing {len(matched)} images...")
    md5_hashes = defaultdict(list)
    for i, stem in enumerate(sorted(matched)):
        if i % 5000 == 0 and i > 0:
            report.write(f"    ...{i}/{len(matched)}")
        img_path = os.path.join(IMAGES_DIR, all_images[stem])
        try:
            h = hashlib.md5(open(img_path, 'rb').read()).hexdigest()
            md5_hashes[h].append(stem)
        except:
            pass

    exact_dupes = {h: stems for h, stems in md5_hashes.items() if len(stems) > 1}
    total_exact_dupes = sum(len(v) - 1 for v in exact_dupes.values())

    report.write(f"\n  Unique images (by content): {len(md5_hashes)}")
    report.write(f"  Exact duplicate groups:     {len(exact_dupes)}")
    report.write(f"  Total exact duplicates:     {total_exact_dupes}")

    if exact_dupes:
        report.write(f"\n  Exact duplicate groups (showing first 20):")
        for i, (h, stems) in enumerate(sorted(exact_dupes.items(), key=lambda x: -len(x[1]))):
            if i >= 20:
                report.write(f"    ... and {len(exact_dupes) - 20} more groups")
                break
            report.write(f"    [{len(stems)} copies] {stems[:5]}{'...' if len(stems) > 5 else ''}")

    # =========================================================================
    # 9. NEAR-DUPLICATE DETECTION (perceptual hash)
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  9. NEAR-DUPLICATE DETECTION (perceptual hash)")
    report.write(f"{'=' * 80}")

    if HAS_PIL and HAS_IMAGEHASH:
        stems_to_hash = sorted(matched)[:MAX_HASH_IMAGES]
        report.write(f"  Computing perceptual hashes for {len(stems_to_hash)} images...")
        report.write(f"  Hash size: {PHASH_SIZE}, distance threshold: {HASH_DISTANCE_THRESHOLD}")

        phashes = {}
        for i, stem in enumerate(stems_to_hash):
            if i % 5000 == 0 and i > 0:
                report.write(f"    ...{i}/{len(stems_to_hash)}")
            img_path = os.path.join(IMAGES_DIR, all_images[stem])
            try:
                with Image.open(img_path) as img:
                    h = imagehash.phash(img, hash_size=PHASH_SIZE)
                    phashes[stem] = h
            except:
                pass

        # Group by similar hash (bucket by hash → find collisions)
        # For large datasets, use hash buckets instead of O(n²) comparison
        hash_buckets = defaultdict(list)
        for stem, h in phashes.items():
            # Use truncated hash as bucket key for approximate grouping
            bucket_key = str(h)[:8]
            hash_buckets[bucket_key].append((stem, h))

        near_dupe_clusters = []
        seen = set()

        for bucket_key, items in hash_buckets.items():
            if len(items) < 2:
                continue
            # Within each bucket, do pairwise comparison
            for i in range(len(items)):
                if items[i][0] in seen:
                    continue
                cluster = [items[i][0]]
                for j in range(i + 1, len(items)):
                    if items[j][0] in seen:
                        continue
                    dist = items[i][1] - items[j][1]
                    if dist <= HASH_DISTANCE_THRESHOLD:
                        cluster.append(items[j][0])
                        seen.add(items[j][0])
                if len(cluster) > 1:
                    near_dupe_clusters.append(cluster)
                    seen.add(items[i][0])

        total_near_dupes = sum(len(c) - 1 for c in near_dupe_clusters)

        report.write(f"\n  Near-duplicate clusters:    {len(near_dupe_clusters)}")
        report.write(f"  Total near-duplicates:      {total_near_dupes}")
        report.write(f"  Images in clusters:         {sum(len(c) for c in near_dupe_clusters)}")

        if near_dupe_clusters:
            # Sort by cluster size
            near_dupe_clusters.sort(key=len, reverse=True)
            report.write(f"\n  Cluster size distribution:")
            cluster_sizes = Counter(len(c) for c in near_dupe_clusters)
            for size, count in sorted(cluster_sizes.items(), reverse=True):
                report.write(f"    {size} images: {count} clusters")

            report.write(f"\n  Top 20 largest near-duplicate clusters:")
            for i, cluster in enumerate(near_dupe_clusters[:20]):
                report.write(f"    [{len(cluster)} similar] {cluster[:5]}{'...' if len(cluster) > 5 else ''}")
    else:
        report.write(f"  SKIPPED — install Pillow and imagehash")

    # =========================================================================
    # 10. ANNOTATION QUALITY ISSUES
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  10. ANNOTATION QUALITY")
    report.write(f"{'=' * 80}")

    report.write(f"  Empty labels:           {len(empty_labels)}")
    report.write(f"  Labels with issues:     {len(annotation_issues)}")

    if annotation_issues:
        report.write(f"\n  Issue details (first 30):")
        for i, (stem, issues) in enumerate(sorted(annotation_issues.items())):
            if i >= 30:
                report.write(f"    ... and {len(annotation_issues) - 30} more files with issues")
                break
            for issue in issues:
                report.write(f"    {stem}: {issue}")

    # Tiny bboxes (< 4px at 640)
    tiny_threshold = 4.0 / IMAGE_SIZE
    tiny_boxes = sum(1 for a in bbox_areas if a < (tiny_threshold ** 2))
    report.write(f"\n  Tiny bboxes (<{4}px at {IMAGE_SIZE}): {tiny_boxes} "
                f"({100.0*tiny_boxes/total_instances:.2f}%)" if total_instances else "")

    # Huge bboxes (>80% of image)
    huge_boxes = sum(1 for a in bbox_areas if a > 0.64)
    report.write(f"  Huge bboxes (>80% area):     {huge_boxes} "
                f"({100.0*huge_boxes/total_instances:.2f}%)" if total_instances else "")

    # Extreme aspect ratios
    extreme_ar = sum(1 for ar in bbox_ars if ar > 10)
    report.write(f"  Extreme aspect ratio (>10:1): {extreme_ar} "
                f"({100.0*extreme_ar/total_instances:.2f}%)" if total_instances else "")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    report.write(f"\n{'=' * 80}")
    report.write(f"  SUMMARY")
    report.write(f"{'=' * 80}")
    report.write(f"  Images:          {len(matched)}")
    report.write(f"  Instances:       {total_instances}")
    report.write(f"  Classes:         {len(CLASS_NAMES)}")
    report.write(f"  Inst/image:      {statistics.mean(instances_per_image):.2f} avg")
    report.write(f"  Exact dupes:     {total_exact_dupes}")
    if HAS_PIL and HAS_IMAGEHASH:
        report.write(f"  Near dupes:      {total_near_dupes}")
    report.write(f"  Source groups:   {len(source_groups)}")
    report.write(f"  Augmented:       {augmented_count} ({100.0*augmented_count/len(matched):.1f}%)")
    report.write(f"  Empty labels:    {len(empty_labels)}")
    report.write(f"  Quality issues:  {len(annotation_issues)}")

    # Dataset health rating
    issues_total = total_exact_dupes + len(annotation_issues) + len(empty_labels)
    if issues_total == 0:
        health = "EXCELLENT"
    elif issues_total < len(matched) * 0.01:
        health = "GOOD"
    elif issues_total < len(matched) * 0.05:
        health = "ACCEPTABLE"
    else:
        health = "NEEDS ATTENTION"
    report.write(f"\n  Dataset health:  {health}")
    report.write(f"{'=' * 80}")

    report.save()


if __name__ == "__main__":
    main()
