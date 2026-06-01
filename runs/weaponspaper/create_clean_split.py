#!/usr/bin/env python3
"""
Create Clean Train/Valid/Test Split — source-aware, stratified, balanced.

Takes ALL images from a single folder and splits them into train/valid/test:
  - Source-aware: images from same source stay in same split (no leakage)
  - Stratified by (dominant_class, dominant_size): preserves distributions
  - Multi-seed: tests N candidates, picks the split with lowest deviation
  - No images left behind: every image goes into exactly one split

Split ratios (configurable):
  train: 70%
  valid: 15%
  test:  15%

Usage:
  python create_clean_split.py
"""

import os
import re
import random
import shutil
import statistics
from collections import defaultdict, Counter
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PATH = r"c:\DISK\GunDataset4\train"  # merged folder with all images
OUTPUT_PATH = r"c:\DISK\GunDatasetClean"    # output with train/valid/test

IMAGES_DIR = os.path.join(INPUT_PATH, "images")
LABELS_DIR = os.path.join(INPUT_PATH, "labels")

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
IMAGE_SIZE = 640

# Split ratios (must sum to 1.0)
SPLIT_RATIOS = {
    'train': 0.70,
    'valid': 0.15,
    'test':  0.15,
}

# Size thresholds for stratification (COCO default)
SMALL_PX = 32
MEDIUM_PX = 96

# Optimization
N_CANDIDATES = 500     # number of random seeds to try
BASE_SEED = 0

# Scoring weights
CLASS_WEIGHT = 2.0     # penalty for class distribution deviation
SIZE_WEIGHT = 1.0      # penalty for size distribution deviation
SHIFT_WEIGHT = 3.0     # penalty for train↔test shift divergence


# =============================================================================
# HELPERS
# =============================================================================
def parse_label(label_path):
    """Parse YOLO label file."""
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(float(parts[0]))
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        annotations.append((cid, x, y, w, h))
                    except ValueError:
                        pass
    except:
        pass
    return annotations


def classify_size(w, h):
    """COCO-style size classification."""
    area = w * h
    small_area = (SMALL_PX ** 2) / (IMAGE_SIZE ** 2)
    medium_area = (MEDIUM_PX ** 2) / (IMAGE_SIZE ** 2)
    if area < small_area:
        return 'small'
    elif area < medium_area:
        return 'medium'
    return 'large'


def extract_source(stem):
    """Extract source prefix from filename."""
    s = stem
    s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]?\d{3,}$", "", s)
    s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
    return s


# =============================================================================
# SOURCE GROUP
# =============================================================================
class SourceGroup:
    def __init__(self, source):
        self.source = source
        self.images = []         # (img_filename, label_filename)
        self.class_counts = Counter()
        self.size_counts = Counter()
        self.n_instances = 0

    @property
    def n_images(self):
        return len(self.images)

    @property
    def dominant_class(self):
        if not self.class_counts:
            return -1
        return self.class_counts.most_common(1)[0][0]

    @property
    def dominant_size(self):
        if not self.size_counts:
            return 'none'
        return self.size_counts.most_common(1)[0][0]

    @property
    def bucket_key(self):
        return (self.dominant_class, self.dominant_size)


def build_source_groups():
    """Build source groups from all images."""
    groups = {}

    all_images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])

    skipped = 0
    for img_file in all_images:
        stem = os.path.splitext(img_file)[0]
        label_file = stem + '.txt'
        label_path = os.path.join(LABELS_DIR, label_file)

        if not os.path.isfile(label_path):
            skipped += 1
            continue

        source = extract_source(stem)
        if source not in groups:
            groups[source] = SourceGroup(source)

        g = groups[source]
        g.images.append((img_file, label_file))

        annotations = parse_label(label_path)
        g.n_instances += len(annotations)
        for cid, x, y, w, h in annotations:
            g.class_counts[cid] += 1
            g.size_counts[classify_size(w, h)] += 1

    print(f"  Built {len(groups)} source groups from {len(all_images)} images "
          f"({skipped} skipped — no label)")
    return groups


# =============================================================================
# DISTRIBUTION HELPERS
# =============================================================================
def compute_distributions(groups_list):
    """Compute class and size distributions for a list of source groups."""
    class_counts = Counter()
    size_counts = Counter()
    n_images = 0
    n_instances = 0

    for g in groups_list:
        n_images += g.n_images
        n_instances += g.n_instances
        class_counts += g.class_counts
        size_counts += g.size_counts

    return {
        'n_images': n_images,
        'n_instances': n_instances,
        'class_counts': class_counts,
        'size_counts': size_counts,
    }


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def compute_score(split_groups, total_stats):
    """Score a candidate split. Lower = better."""
    score = 0.0

    split_stats = {}
    for split_name, groups in split_groups.items():
        split_stats[split_name] = compute_distributions(groups)

    # 1. Per-split class deviation from overall distribution
    for split_name, stats in split_stats.items():
        if not stats['n_instances']:
            score += 1000  # penalize empty splits
            continue
        for cid in range(len(CLASS_NAMES)):
            split_pct = pct(stats['class_counts'].get(cid, 0), stats['n_instances'])
            total_pct = pct(total_stats['class_counts'].get(cid, 0), total_stats['n_instances'])
            score += CLASS_WEIGHT * abs(split_pct - total_pct)

    # 2. Per-split size deviation from overall distribution
    for split_name, stats in split_stats.items():
        if not stats['n_instances']:
            continue
        for sz in ['small', 'medium', 'large']:
            split_pct = pct(stats['size_counts'].get(sz, 0), stats['n_instances'])
            total_pct = pct(total_stats['size_counts'].get(sz, 0), total_stats['n_instances'])
            score += SIZE_WEIGHT * abs(split_pct - total_pct)

    # 3. Train↔test shift penalty
    if 'train' in split_stats and 'test' in split_stats:
        tr = split_stats['train']
        te = split_stats['test']
        if tr['n_instances'] and te['n_instances']:
            for cid in range(len(CLASS_NAMES)):
                tr_pct = pct(tr['class_counts'].get(cid, 0), tr['n_instances'])
                te_pct = pct(te['class_counts'].get(cid, 0), te['n_instances'])
                score += SHIFT_WEIGHT * abs(tr_pct - te_pct)
            for sz in ['small', 'medium', 'large']:
                tr_pct = pct(tr['size_counts'].get(sz, 0), tr['n_instances'])
                te_pct = pct(te['size_counts'].get(sz, 0), te['n_instances'])
                score += SHIFT_WEIGHT * abs(tr_pct - te_pct)

    # 4. Split ratio penalty (actual vs target)
    total_images = sum(s['n_images'] for s in split_stats.values())
    for split_name, stats in split_stats.items():
        actual_ratio = stats['n_images'] / total_images if total_images else 0
        target_ratio = SPLIT_RATIOS[split_name]
        score += 10.0 * abs(actual_ratio - target_ratio) * 100  # 1% off = 10 penalty

    return score


# =============================================================================
# STRATIFIED SPLITTING
# =============================================================================
def stratified_split(source_groups, seed):
    """Split source groups into train/valid/test using stratified sampling."""
    rng = random.Random(seed)

    # Bucket groups by (dominant_class, dominant_size)
    buckets = defaultdict(list)
    for g in source_groups.values():
        buckets[g.bucket_key].append(g)

    split_groups = {'train': [], 'valid': [], 'test': []}

    for bucket_key, groups in buckets.items():
        rng.shuffle(groups)

        # Count total images in this bucket
        total_imgs = sum(g.n_images for g in groups)

        # Target counts per split
        targets = {
            'train': int(round(total_imgs * SPLIT_RATIOS['train'])),
            'valid': int(round(total_imgs * SPLIT_RATIOS['valid'])),
            'test':  int(round(total_imgs * SPLIT_RATIOS['test'])),
        }

        # Assign groups greedily
        assigned = {'train': 0, 'valid': 0, 'test': 0}
        for g in groups:
            # Find the split that's most under its target
            best_split = None
            best_deficit = -float('inf')
            for split_name in ['train', 'valid', 'test']:
                deficit = targets[split_name] - assigned[split_name]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_split = split_name

            split_groups[best_split].append(g)
            assigned[best_split] += g.n_images

    return split_groups


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'=' * 70}")
    print(f"  CREATE CLEAN SPLIT")
    print(f"{'=' * 70}")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Ratios: train={SPLIT_RATIOS['train']:.0%}, "
          f"valid={SPLIT_RATIOS['valid']:.0%}, test={SPLIT_RATIOS['test']:.0%}")
    print(f"{'=' * 70}")

    # Phase 1: Build source groups
    print(f"\n  PHASE 1: Building source groups")
    source_groups = build_source_groups()

    total_stats = compute_distributions(list(source_groups.values()))
    print(f"  Total: {total_stats['n_images']} images, {total_stats['n_instances']} instances")
    print(f"  Source groups: {len(source_groups)}")

    group_sizes = [g.n_images for g in source_groups.values()]
    print(f"  Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
          f"median={statistics.median(group_sizes):.0f}, mean={statistics.mean(group_sizes):.1f}")

    # Phase 2: Find best split
    print(f"\n  PHASE 2: Testing {N_CANDIDATES} candidate splits")
    best_seed = BASE_SEED
    best_score = float('inf')
    best_split = None
    scores = []

    for i in range(N_CANDIDATES):
        seed = BASE_SEED + i
        candidate = stratified_split(source_groups, seed)
        score = compute_score(candidate, total_stats)
        scores.append((seed, score))

        if score < best_score:
            best_score = score
            best_seed = seed
            best_split = candidate

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_CANDIDATES} | best: seed={best_seed}, score={best_score:.3f}")

    scores.sort(key=lambda x: x[1])
    print(f"\n  Score distribution:")
    print(f"    Best:   {scores[0][1]:.3f} (seed {scores[0][0]})")
    print(f"    Median: {scores[len(scores)//2][1]:.3f}")
    print(f"    Worst:  {scores[-1][1]:.3f}")

    # Phase 3: Verification
    print(f"\n  PHASE 3: Verification (seed={best_seed}, score={best_score:.3f})")
    print(f"  {'─' * 60}")

    # Check source overlap
    split_sources = {}
    for split_name, groups in best_split.items():
        split_sources[split_name] = set(g.source for g in groups)

    for s1 in ['train', 'valid', 'test']:
        for s2 in ['train', 'valid', 'test']:
            if s1 >= s2:
                continue
            overlap = split_sources[s1] & split_sources[s2]
            tag = "LEAK" if overlap else "OK"
            print(f"  Source overlap {s1}∩{s2}: {len(overlap)} [{tag}]")

    # Per-split stats
    for split_name in ['train', 'valid', 'test']:
        groups = best_split[split_name]
        stats = compute_distributions(groups)
        ratio = stats['n_images'] / total_stats['n_images']
        target = SPLIT_RATIOS[split_name]

        print(f"\n  ── {split_name.upper()} ──")
        print(f"  Images:    {stats['n_images']:>6d} ({ratio*100:.1f}%, target {target*100:.0f}%)")
        print(f"  Instances: {stats['n_instances']:>6d}")

        print(f"  Class distribution:")
        print(f"  {'Class':<12} {'Count':>7} {'Split%':>8} {'Total%':>8} {'Delta':>8}")
        for cid, name in enumerate(CLASS_NAMES):
            s_pct = pct(stats['class_counts'].get(cid, 0), stats['n_instances'])
            t_pct = pct(total_stats['class_counts'].get(cid, 0), total_stats['n_instances'])
            delta = s_pct - t_pct
            tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {name:<12} {stats['class_counts'].get(cid,0):>7} {s_pct:>7.1f}% {t_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

        print(f"  Size distribution [32/96]:")
        print(f"  {'Size':<12} {'Count':>7} {'Split%':>8} {'Total%':>8} {'Delta':>8}")
        for sz in ['small', 'medium', 'large']:
            s_pct = pct(stats['size_counts'].get(sz, 0), stats['n_instances'])
            t_pct = pct(total_stats['size_counts'].get(sz, 0), total_stats['n_instances'])
            delta = s_pct - t_pct
            tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {sz:<12} {stats['size_counts'].get(sz,0):>7} {s_pct:>7.1f}% {t_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

    # Train↔test shift
    tr_stats = compute_distributions(best_split['train'])
    te_stats = compute_distributions(best_split['test'])
    print(f"\n  ── Train ↔ Test SHIFT ──")
    print(f"  {'Metric':<16} {'Train%':>8} {'Test%':>8} {'Shift':>8}")
    for cid, name in enumerate(CLASS_NAMES):
        tr_pct = pct(tr_stats['class_counts'].get(cid, 0), tr_stats['n_instances'])
        te_pct = pct(te_stats['class_counts'].get(cid, 0), te_stats['n_instances'])
        shift = te_pct - tr_pct
        print(f"  cls:{name:<11} {tr_pct:>7.1f}% {te_pct:>7.1f}% {shift:>+7.2f}pp")
    for sz in ['small', 'medium', 'large']:
        tr_pct = pct(tr_stats['size_counts'].get(sz, 0), tr_stats['n_instances'])
        te_pct = pct(te_stats['size_counts'].get(sz, 0), te_stats['n_instances'])
        shift = te_pct - tr_pct
        print(f"  sz:{sz:<12} {tr_pct:>7.1f}% {te_pct:>7.1f}% {shift:>+7.2f}pp")

    # Phase 4: Copy files
    print(f"\n  PHASE 4: Copying files to {OUTPUT_PATH}")

    for split_name, groups in best_split.items():
        out_img = os.path.join(OUTPUT_PATH, split_name, "images")
        out_lbl = os.path.join(OUTPUT_PATH, split_name, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        copied = 0
        for g in groups:
            for img_file, label_file in g.images:
                shutil.copy2(
                    os.path.join(IMAGES_DIR, img_file),
                    os.path.join(out_img, img_file)
                )
                shutil.copy2(
                    os.path.join(LABELS_DIR, label_file),
                    os.path.join(out_lbl, label_file)
                )
                copied += 1

        print(f"  {split_name}: {copied} image-label pairs")

    # Create data.yaml
    yaml_path = os.path.join(OUTPUT_PATH, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {OUTPUT_PATH}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"test: test/images\n")
        f.write(f"\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"\n  data.yaml created at {yaml_path}")

    # Final summary
    total_copied = sum(
        compute_distributions(groups)['n_images']
        for groups in best_split.values()
    )
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"  Seed: {best_seed}, Score: {best_score:.3f}")
    print(f"  Total: {total_copied} images split into {OUTPUT_PATH}")
    print(f"  Source overlap: 0 (zero leakage)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
