#!/usr/bin/env python3
"""
Create Ablation Dataset — source-aware stratified subset from split dataset.

Takes an already-split dataset (train/valid/test) and samples ~30% from each
split independently, preserving:
  - Source grouping (no leakage)
  - Class distribution
  - Size distribution
  - Train↔test shift

Input:
  GunDatasetClean/
    train/images/ train/labels/
    valid/images/ valid/labels/
    test/images/  test/labels/

Output:
  GunDatasetAblation/
    train/images/ train/labels/   (~30% of original train ≈ 5,676 images)
    valid/images/ valid/labels/   (~30% of original valid  ≈ 1,216 images)
    test/images/  test/labels/    (~30% of original test   ≈ 1,216 images)
                                   Total ≈ 8,108 images

Usage:
  python create_ablation.py
"""

import os
import re
import random
import shutil
import statistics
from collections import defaultdict, Counter

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PATH = r"c:\DISK\GunDatasetClean"
OUTPUT_PATH = r"c:\DISK\GunDatasetAblation"

SAMPLE_PERCENT = 42  # % to keep from each split (~8K training images)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
IMAGE_SIZE = 640
SPLITS = ['train', 'valid', 'test']

SMALL_PX = 32
MEDIUM_PX = 96

N_CANDIDATES = 300
BASE_SEED = 0


# =============================================================================
# HELPERS
# =============================================================================
def parse_label(label_path):
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
    area = w * h
    small_area = (SMALL_PX ** 2) / (IMAGE_SIZE ** 2)
    medium_area = (MEDIUM_PX ** 2) / (IMAGE_SIZE ** 2)
    if area < small_area:
        return 'small'
    elif area < medium_area:
        return 'medium'
    return 'large'


def extract_source(stem):
    s = stem
    s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]?\d{3,}$", "", s)
    s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
    return s


def pct(count, total):
    return 100.0 * count / total if total else 0.0


# =============================================================================
# SOURCE GROUP
# =============================================================================
class SourceGroup:
    def __init__(self, source):
        self.source = source
        self.images = []
        self.class_counts = Counter()
        self.size_counts = Counter()
        self.n_instances = 0

    @property
    def n_images(self):
        return len(self.images)

    @property
    def bucket_key(self):
        dom_class = self.class_counts.most_common(1)[0][0] if self.class_counts else -1
        dom_size = self.size_counts.most_common(1)[0][0] if self.size_counts else 'none'
        return (dom_class, dom_size)


def build_source_groups(img_dir, lbl_dir):
    groups = {}
    images = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        label_file = stem + '.txt'
        label_path = os.path.join(lbl_dir, label_file)
        if not os.path.isfile(label_path):
            continue

        source = extract_source(stem)
        if source not in groups:
            groups[source] = SourceGroup(source)

        g = groups[source]
        g.images.append((img_file, label_file))

        for cid, x, y, w, h in parse_label(label_path):
            g.class_counts[cid] += 1
            g.size_counts[classify_size(w, h)] += 1
            g.n_instances += 1

    return groups


def compute_stats(groups_list, lbl_dir):
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


# =============================================================================
# SAMPLING
# =============================================================================
def sample_groups(source_groups, target_images, seed):
    """Sample source groups to reach target_images using stratified sampling."""
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for g in source_groups.values():
        buckets[g.bucket_key].append(g)

    total_images = sum(g.n_images for g in source_groups.values())

    selected = []
    for key, groups in buckets.items():
        bucket_total = sum(g.n_images for g in groups)
        bucket_target = target_images * bucket_total / total_images

        rng.shuffle(groups)
        bucket_count = 0
        for g in groups:
            if bucket_count >= bucket_target:
                break
            selected.append(g)
            bucket_count += g.n_images

    return selected


def score_candidate(selected_groups, full_stats):
    """Score how well sampled groups match full distribution. Lower = better."""
    sel_class = Counter()
    sel_size = Counter()
    sel_instances = 0

    for g in selected_groups:
        sel_class += g.class_counts
        sel_size += g.size_counts
        sel_instances += g.n_instances

    if sel_instances == 0:
        return float('inf')

    score = 0.0
    for cid in range(len(CLASS_NAMES)):
        s_pct = pct(sel_class.get(cid, 0), sel_instances)
        f_pct = pct(full_stats['class_counts'].get(cid, 0), full_stats['n_instances'])
        score += 2.0 * abs(s_pct - f_pct)

    for sz in ['small', 'medium', 'large']:
        s_pct = pct(sel_size.get(sz, 0), sel_instances)
        f_pct = pct(full_stats['size_counts'].get(sz, 0), full_stats['n_instances'])
        score += 1.0 * abs(s_pct - f_pct)

    return score


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'=' * 70}")
    print(f"  CREATE ABLATION DATASET ({SAMPLE_PERCENT}%)")
    print(f"{'=' * 70}")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Sample: {SAMPLE_PERCENT}% per split")
    print(f"  Seeds:  {N_CANDIDATES}")
    print(f"{'=' * 70}")

    total_selected = 0
    total_full = 0

    for split in SPLITS:
        img_dir = os.path.join(INPUT_PATH, split, "images")
        lbl_dir = os.path.join(INPUT_PATH, split, "labels")

        if not os.path.isdir(img_dir):
            print(f"\n  {split}: NOT FOUND, skipping")
            continue

        print(f"\n  {'─' * 60}")
        print(f"  SPLIT: {split.upper()}")
        print(f"  {'─' * 60}")

        # Build source groups
        groups = build_source_groups(img_dir, lbl_dir)
        full_stats = compute_stats(list(groups.values()), lbl_dir)

        n_full = full_stats['n_images']
        target = max(1, int(round(n_full * SAMPLE_PERCENT / 100)))
        total_full += n_full

        print(f"  Full: {n_full} images, {full_stats['n_instances']} instances, "
              f"{len(groups)} source groups")
        print(f"  Target: ~{target} images ({SAMPLE_PERCENT}%)")

        # Find best sampling
        print(f"  Testing {N_CANDIDATES} candidates...")
        best_seed = BASE_SEED
        best_score = float('inf')
        best_selected = None

        for i in range(N_CANDIDATES):
            seed = BASE_SEED + i
            selected = sample_groups(groups, target, seed)
            score = score_candidate(selected, full_stats)

            if score < best_score:
                best_score = score
                best_seed = seed
                best_selected = selected

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{N_CANDIDATES} | best: seed={best_seed}, score={best_score:.3f}")

        # Stats for best selection
        sel_stats = compute_stats(best_selected, lbl_dir)
        sel_sources = set(g.source for g in best_selected)
        full_sources = set(g.source for g in groups.values())
        remaining_sources = full_sources - sel_sources

        print(f"\n  Best: seed={best_seed}, score={best_score:.3f}")
        print(f"  Selected: {sel_stats['n_images']} images ({100*sel_stats['n_images']/n_full:.1f}%), "
              f"{sel_stats['n_instances']} instances")

        # Class comparison
        print(f"\n  Class distribution:")
        print(f"  {'Class':<12} {'Full%':>8} {'Abl%':>8} {'Delta':>8}")
        for cid, name in enumerate(CLASS_NAMES):
            f_pct = pct(full_stats['class_counts'].get(cid, 0), full_stats['n_instances'])
            s_pct = pct(sel_stats['class_counts'].get(cid, 0), sel_stats['n_instances'])
            delta = s_pct - f_pct
            tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {name:<12} {f_pct:>7.1f}% {s_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

        # Size comparison
        print(f"\n  Size distribution [32/96]:")
        print(f"  {'Size':<12} {'Full%':>8} {'Abl%':>8} {'Delta':>8}")
        for sz in ['small', 'medium', 'large']:
            f_pct = pct(full_stats['size_counts'].get(sz, 0), full_stats['n_instances'])
            s_pct = pct(sel_stats['size_counts'].get(sz, 0), sel_stats['n_instances'])
            delta = s_pct - f_pct
            tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {sz:<12} {f_pct:>7.1f}% {s_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

        # Copy files
        out_img = os.path.join(OUTPUT_PATH, split, "images")
        out_lbl = os.path.join(OUTPUT_PATH, split, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        copied = 0
        for g in best_selected:
            for img_file, label_file in g.images:
                shutil.copy2(
                    os.path.join(img_dir, img_file),
                    os.path.join(out_img, img_file)
                )
                shutil.copy2(
                    os.path.join(lbl_dir, label_file),
                    os.path.join(out_lbl, label_file)
                )
                copied += 1

        total_selected += copied
        print(f"\n  Copied: {copied} image-label pairs to {OUTPUT_PATH}/{split}/")

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

    # Source overlap check between ablation splits
    print(f"\n{'=' * 70}")
    print(f"  SOURCE OVERLAP CHECK (ablation)")
    print(f"{'=' * 70}")

    abl_sources = {}
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if not os.path.isdir(img_dir):
            continue
        sources = set()
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                sources.add(extract_source(os.path.splitext(f)[0]))
        abl_sources[split] = sources

    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 >= s2 or s1 not in abl_sources or s2 not in abl_sources:
                continue
            overlap = abl_sources[s1] & abl_sources[s2]
            tag = "LEAK" if overlap else "OK"
            print(f"  {s1} ∩ {s2}: {len(overlap)} shared sources [{tag}]")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
    print(f"  Full dataset:     {total_full} images")
    print(f"  Ablation dataset: {total_selected} images ({100*total_selected/total_full:.1f}%)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  data.yaml: {yaml_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
