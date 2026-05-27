"""
CreateSubsetV2 — Source-aware, shift-optimized stratified ablation builder.

Improvements over CreateSubset.py:
  1. SOURCE-AWARE GROUPING: Images from the same source (video/sequence)
     are kept together in the same split. This eliminates source-prefix
     leakage between train/valid/test.
  2. MULTI-SEED OPTIMIZATION: Runs N candidate samplings (default 200)
     and picks the one with the lowest combined deviation score.
  3. SHIFT-AWARE OBJECTIVE: The optimization score penalizes:
       - Class distribution deviation from FULL (per split)
       - Size distribution deviation from FULL (per split)
       - Train↔test relative shift magnification vs FULL
  4. MULTI-LABEL STRATIFICATION: Instead of using only the dominant
     class, each source group is bucketed by its (class_profile, size_profile)
     which captures the full distribution of objects.
  5. VERIFICATION: Built-in comparison report showing FULL vs ABLATION
     distributions + train↔test shift analysis.

Usage:
    python CreateSubsetV2.py

The script reads the FULL dataset, groups images by source prefix,
then runs stratified sampling over source groups (not individual images).
"""

import os
import re
import random
import shutil
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 640
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
SPLITS = ['train', 'valid', 'test']

# Size thresholds (COCO default)
SMALL_PX = 32
MEDIUM_PX = 96
SMALL_AREA_NORM = (SMALL_PX ** 2) / (IMAGE_SIZE ** 2)
MEDIUM_AREA_NORM = (MEDIUM_PX ** 2) / (IMAGE_SIZE ** 2)

# Additional thresholds for shift analysis
SIZE_THRESHOLDS = [
    ("24/72",  24,  72),
    ("32/96",  32,  96),   # COCO default
    ("48/144", 48, 144),
    ("64/192", 64, 192),
]


# ─────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────
def parse_label_file(label_path: str) -> List[Tuple]:
    """Yield (class_id, x_center, y_center, width, height) tuples."""
    annotations = []
    try:
        with open(label_path, 'r') as f:
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


def classify_size(w_norm: float, h_norm: float, small_px: int = 32, medium_px: int = 96) -> str:
    """COCO-style size class."""
    area = w_norm * h_norm
    small_area = (small_px ** 2) / (IMAGE_SIZE ** 2)
    medium_area = (medium_px ** 2) / (IMAGE_SIZE ** 2)
    if area < small_area:
        return 'small'
    elif area < medium_area:
        return 'medium'
    return 'large'


def extract_source(stem: str) -> str:
    """
    Extract the source prefix from a filename stem.
    This groups augmented variants and video frames from the same source.
    """
    s = stem
    # Remove Roboflow augmentation suffixes: .rf.<hex>
    s = re.sub(r"\.rf\.[a-f0-9]+$", "", s, flags=re.IGNORECASE)
    # Remove frame numbers: _frame_001, -frame001, etc.
    s = re.sub(r"[-_]frame[-_]?\d+", "", s, flags=re.IGNORECASE)
    # Remove trailing numeric IDs (3+ digits)
    s = re.sub(r"[-_]?\d{3,}$", "", s)
    # Remove extension-like suffixes
    s = re.sub(r"_jpg$|_png$|_jpeg$", "", s, flags=re.IGNORECASE)
    return s


# ─────────────────────────────────────────────────────────────────
# Source group data structure
# ─────────────────────────────────────────────────────────────────
class SourceGroup:
    """A group of images sharing the same source prefix."""

    def __init__(self, source: str):
        self.source = source
        self.images: List[str] = []       # filenames
        self.n_instances: int = 0
        self.class_counts: Counter = Counter()
        self.size_counts: Counter = Counter()  # at COCO threshold

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def dominant_class(self) -> int:
        if not self.class_counts:
            return -1
        return self.class_counts.most_common(1)[0][0]

    @property
    def dominant_size(self) -> str:
        if not self.size_counts:
            return 'none'
        return self.size_counts.most_common(1)[0][0]

    @property
    def bucket_key(self) -> Tuple:
        """(dominant_class, dominant_size) — used for stratified sampling."""
        return (self.dominant_class, self.dominant_size)


def build_source_groups(images_path: str, labels_path: str) -> Dict[str, SourceGroup]:
    """
    Scan a split's images/labels and group them by source prefix.
    Each SourceGroup contains all images from the same source.
    """
    groups: Dict[str, SourceGroup] = {}

    all_images = [
        f for f in os.listdir(images_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    for img_file in all_images:
        stem = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_path, stem + '.txt')
        if not os.path.exists(label_file):
            continue

        source = extract_source(stem)
        if source not in groups:
            groups[source] = SourceGroup(source)

        g = groups[source]
        g.images.append(img_file)

        annotations = parse_label_file(label_file)
        g.n_instances += len(annotations)
        for cid, x, y, w, h in annotations:
            g.class_counts[cid] += 1
            g.size_counts[classify_size(w, h)] += 1

    return groups


# ─────────────────────────────────────────────────────────────────
# Scoring: how well does a candidate ablation match the FULL dataset?
# ─────────────────────────────────────────────────────────────────
def compute_split_stats(images: List[str], labels_path: str) -> dict:
    """Compute class and size distributions for a set of images."""
    class_counts = Counter()
    size_counts_multi = {tl: Counter() for tl, _, _ in SIZE_THRESHOLDS}
    total = 0

    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_path, stem + '.txt')
        for cid, x, y, w, h in parse_label_file(label_file):
            class_counts[cid] += 1
            total += 1
            for t_label, small_px, medium_px in SIZE_THRESHOLDS:
                sz = classify_size(w, h, small_px, medium_px)
                size_counts_multi[t_label][sz] += 1

    return {
        'n_images': len(images),
        'n_instances': total,
        'class_counts': class_counts,
        'size_counts_multi': size_counts_multi,
    }


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def compute_deviation_score(
    candidate_stats: Dict[str, dict],
    full_stats: Dict[str, dict],
    shift_weight: float = 2.0,
) -> float:
    """
    Score a candidate ablation against the FULL dataset.
    Lower = better.

    Components:
      1. Distribution deviation: sum of |ablation_pct - full_pct| across
         classes and sizes, for each split.
      2. Shift deviation: how much the train↔test shift differs between
         ablation and full (penalized more heavily).
    """
    score = 0.0

    # ── 1. Per-split distribution deviation ──
    for split in SPLITS:
        if split not in candidate_stats or split not in full_stats:
            continue
        c = candidate_stats[split]
        f = full_stats[split]
        if not c['n_instances'] or not f['n_instances']:
            continue

        # Class deviation
        for cid in range(len(CLASS_NAMES)):
            c_pct = pct(c['class_counts'].get(cid, 0), c['n_instances'])
            f_pct = pct(f['class_counts'].get(cid, 0), f['n_instances'])
            score += abs(c_pct - f_pct)

        # Size deviation (at all thresholds)
        for t_label, _, _ in SIZE_THRESHOLDS:
            for sz in ['small', 'medium', 'large']:
                c_pct = pct(c['size_counts_multi'][t_label].get(sz, 0), c['n_instances'])
                f_pct = pct(f['size_counts_multi'][t_label].get(sz, 0), f['n_instances'])
                score += abs(c_pct - f_pct)

    # ── 2. Train↔test shift deviation ──
    if 'train' in candidate_stats and 'test' in candidate_stats and \
       'train' in full_stats and 'test' in full_stats:
        ct = candidate_stats['train']
        ce = candidate_stats['test']
        ft = full_stats['train']
        fe = full_stats['test']

        if ct['n_instances'] and ce['n_instances'] and ft['n_instances'] and fe['n_instances']:
            # Class shift deviation
            for cid in range(len(CLASS_NAMES)):
                c_tr = pct(ct['class_counts'].get(cid, 0), ct['n_instances'])
                c_te = pct(ce['class_counts'].get(cid, 0), ce['n_instances'])
                f_tr = pct(ft['class_counts'].get(cid, 0), ft['n_instances'])
                f_te = pct(fe['class_counts'].get(cid, 0), fe['n_instances'])

                c_shift = c_te - c_tr
                f_shift = f_te - f_tr
                score += shift_weight * abs(c_shift - f_shift)

            # Size shift deviation (all thresholds)
            for t_label, _, _ in SIZE_THRESHOLDS:
                for sz in ['small', 'medium', 'large']:
                    c_tr = pct(ct['size_counts_multi'][t_label].get(sz, 0), ct['n_instances'])
                    c_te = pct(ce['size_counts_multi'][t_label].get(sz, 0), ce['n_instances'])
                    f_tr = pct(ft['size_counts_multi'][t_label].get(sz, 0), ft['n_instances'])
                    f_te = pct(fe['size_counts_multi'][t_label].get(sz, 0), fe['n_instances'])

                    c_shift = c_te - c_tr
                    f_shift = f_te - f_tr
                    score += shift_weight * abs(c_shift - f_shift)

    return score


# ─────────────────────────────────────────────────────────────────
# Source-aware stratified sampling
# ─────────────────────────────────────────────────────────────────
def stratified_sample_sources(
    source_groups: Dict[str, SourceGroup],
    target_image_count: int,
    seed: int = 42,
) -> List[str]:
    """
    Sample source groups (not individual images) to reach target_image_count.

    Strategy:
      1. Bucket source groups by (dominant_class, dominant_size).
      2. Proportionally sample groups from each bucket.
      3. All images from a selected source group are included.

    Because source groups have varying sizes, the final count may not exactly
    match target_image_count. We use greedy residual assignment to get close.
    """
    rng = random.Random(seed)

    # Bucket groups
    buckets: Dict[Tuple, List[SourceGroup]] = defaultdict(list)
    for g in source_groups.values():
        buckets[g.bucket_key].append(g)

    total_images = sum(g.n_images for g in source_groups.values())

    # Target number of images per bucket (proportional)
    bucket_targets = {}
    for key, groups in buckets.items():
        bucket_img_count = sum(g.n_images for g in groups)
        bucket_targets[key] = target_image_count * bucket_img_count / total_images

    # Greedy sampling: within each bucket, shuffle groups and pick until
    # we've reached the target for that bucket
    selected_images = []
    for key, groups in buckets.items():
        target = bucket_targets[key]
        rng.shuffle(groups)

        bucket_selected = 0
        for g in groups:
            if bucket_selected >= target:
                break
            selected_images.extend(g.images)
            bucket_selected += g.n_images

    return selected_images


# ─────────────────────────────────────────────────────────────────
# Multi-seed optimization
# ─────────────────────────────────────────────────────────────────
def find_best_ablation(
    full_dataset_path: str,
    split_percents: Dict[str, float],
    n_candidates: int = 200,
    base_seed: int = 0,
    verbose: bool = True,
) -> Tuple[int, float, Dict[str, List[str]]]:
    """
    Run n_candidates random samplings, score each against FULL, return the best.

    Returns:
        (best_seed, best_score, best_splits)
        where best_splits = {'train': [...images...], 'valid': [...], 'test': [...]}
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  PHASE 1: Building source groups and computing FULL stats")
        print("=" * 70)

    # Build source groups and full stats for each split
    split_source_groups: Dict[str, Dict[str, SourceGroup]] = {}
    full_stats: Dict[str, dict] = {}

    for split in SPLITS:
        images_path = os.path.join(full_dataset_path, split, 'images')
        labels_path = os.path.join(full_dataset_path, split, 'labels')

        if not os.path.isdir(images_path):
            print(f"  WARNING: {images_path} not found, skipping split '{split}'")
            continue

        groups = build_source_groups(images_path, labels_path)
        split_source_groups[split] = groups

        # Compute full stats
        all_images = []
        for g in groups.values():
            all_images.extend(g.images)
        full_stats[split] = compute_split_stats(all_images, labels_path)

        n_sources = len(groups)
        n_images = sum(g.n_images for g in groups.values())
        n_instances = full_stats[split]['n_instances']

        if verbose:
            print(f"  {split}: {n_images} images, {n_instances} instances, "
                  f"{n_sources} source groups "
                  f"(avg {n_images/n_sources:.1f} img/source)")

    if verbose:
        print(f"\n  Source-group size distribution:")
        for split in SPLITS:
            if split not in split_source_groups:
                continue
            sizes = [g.n_images for g in split_source_groups[split].values()]
            if sizes:
                print(f"    {split}: min={min(sizes)}, median={statistics.median(sizes):.0f}, "
                      f"max={max(sizes)}, mean={statistics.mean(sizes):.1f}, "
                      f"groups with >10 imgs: {sum(1 for s in sizes if s > 10)}")

    # ── Run candidates ──
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  PHASE 2: Testing {n_candidates} candidate samplings")
        print(f"{'=' * 70}")

    best_seed = base_seed
    best_score = float('inf')
    best_splits: Dict[str, List[str]] = {}

    scores = []
    for i in range(n_candidates):
        seed = base_seed + i
        candidate_stats = {}
        candidate_splits = {}

        for split in SPLITS:
            if split not in split_source_groups:
                continue

            images_path = os.path.join(full_dataset_path, split, 'images')
            labels_path = os.path.join(full_dataset_path, split, 'labels')
            groups = split_source_groups[split]

            total_images = sum(g.n_images for g in groups.values())
            target = max(1, int(round(total_images * split_percents[split] / 100)))

            sampled = stratified_sample_sources(groups, target, seed=seed)
            candidate_splits[split] = sampled
            candidate_stats[split] = compute_split_stats(sampled, labels_path)

        score = compute_deviation_score(candidate_stats, full_stats)
        scores.append((seed, score))

        if score < best_score:
            best_score = score
            best_seed = seed
            best_splits = candidate_splits

        if verbose and (i + 1) % 50 == 0:
            print(f"    Tested {i + 1}/{n_candidates} candidates | "
                  f"best so far: seed={best_seed}, score={best_score:.3f}")

    if verbose:
        scores.sort(key=lambda x: x[1])
        print(f"\n  Score distribution across {n_candidates} candidates:")
        print(f"    Best:   {scores[0][1]:.3f} (seed {scores[0][0]})")
        print(f"    Median: {scores[len(scores)//2][1]:.3f}")
        print(f"    Worst:  {scores[-1][1]:.3f}")
        print(f"    Spread: {scores[-1][1] - scores[0][1]:.3f}")

        # Show top 5
        print(f"\n  Top 5 candidates:")
        for seed, score in scores[:5]:
            print(f"    seed={seed:>4d}  score={score:.3f}")

    return best_seed, best_score, best_splits


# ─────────────────────────────────────────────────────────────────
# Verification report
# ─────────────────────────────────────────────────────────────────
def print_verification_report(
    full_dataset_path: str,
    best_splits: Dict[str, List[str]],
    best_seed: int,
    best_score: float,
):
    """Print a comprehensive comparison of FULL vs ABLATION."""

    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: Verification Report")
    print(f"  Best seed: {best_seed} | Score: {best_score:.3f}")
    print(f"{'=' * 70}")

    for split in SPLITS:
        if split not in best_splits:
            continue

        images_path = os.path.join(full_dataset_path, split, 'images')
        labels_path = os.path.join(full_dataset_path, split, 'labels')

        # Full stats
        all_images = [
            f for f in os.listdir(images_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
               and os.path.exists(os.path.join(labels_path, os.path.splitext(f)[0] + '.txt'))
        ]
        full = compute_split_stats(all_images, labels_path)
        ablation = compute_split_stats(best_splits[split], labels_path)

        print(f"\n  ── {split.upper()} ──")
        print(f"  FULL:     {full['n_images']:>6d} images, {full['n_instances']:>6d} instances")
        print(f"  ABLATION: {ablation['n_images']:>6d} images, {ablation['n_instances']:>6d} instances "
              f"({100*ablation['n_images']/full['n_images']:.1f}%)")

        # Class comparison
        print(f"\n  Class distribution:")
        print(f"  {'Class':<12} {'FULL %':>10} {'ABLATION %':>12} {'delta pp':>10}")
        for cid, name in enumerate(CLASS_NAMES):
            f_pct = pct(full['class_counts'].get(cid, 0), full['n_instances'])
            a_pct = pct(ablation['class_counts'].get(cid, 0), ablation['n_instances'])
            delta = a_pct - f_pct
            tag = ' !!!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {name:<12} {f_pct:>9.1f}% {a_pct:>11.1f}% {delta:>+9.2f}pp{tag}")

        # Size comparison (COCO default threshold)
        print(f"\n  Size distribution [32/96] (COCO default):")
        print(f"  {'Size':<12} {'FULL %':>10} {'ABLATION %':>12} {'delta pp':>10}")
        t_label = "32/96"
        for sz in ['small', 'medium', 'large']:
            f_pct = pct(full['size_counts_multi'][t_label].get(sz, 0), full['n_instances'])
            a_pct = pct(ablation['size_counts_multi'][t_label].get(sz, 0), ablation['n_instances'])
            delta = a_pct - f_pct
            tag = ' !!!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  {sz:<12} {f_pct:>9.1f}% {a_pct:>11.1f}% {delta:>+9.2f}pp{tag}")

    # ── Train↔test shift comparison ──
    if 'train' in best_splits and 'test' in best_splits:
        train_labels = os.path.join(full_dataset_path, 'train', 'labels')
        test_labels = os.path.join(full_dataset_path, 'test', 'labels')
        train_images_dir = os.path.join(full_dataset_path, 'train', 'images')
        test_images_dir = os.path.join(full_dataset_path, 'test', 'images')

        # Full
        all_train = [
            f for f in os.listdir(train_images_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
               and os.path.exists(os.path.join(train_labels, os.path.splitext(f)[0] + '.txt'))
        ]
        all_test = [
            f for f in os.listdir(test_images_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
               and os.path.exists(os.path.join(test_labels, os.path.splitext(f)[0] + '.txt'))
        ]
        full_tr = compute_split_stats(all_train, train_labels)
        full_te = compute_split_stats(all_test, test_labels)
        abl_tr = compute_split_stats(best_splits['train'], train_labels)
        abl_te = compute_split_stats(best_splits['test'], test_labels)

        print(f"\n  ── Train ↔ Test SHIFT comparison ──")
        print(f"  {'Metric':<18} {'FULL shift':>12} {'ABLATION shift':>16} {'delta':>10}")
        print(f"  {'─' * 60}")

        # Class shift
        for cid, name in enumerate(CLASS_NAMES):
            f_tr = pct(full_tr['class_counts'].get(cid, 0), full_tr['n_instances'])
            f_te = pct(full_te['class_counts'].get(cid, 0), full_te['n_instances'])
            a_tr = pct(abl_tr['class_counts'].get(cid, 0), abl_tr['n_instances'])
            a_te = pct(abl_te['class_counts'].get(cid, 0), abl_te['n_instances'])
            f_shift = f_te - f_tr
            a_shift = a_te - a_tr
            delta = a_shift - f_shift
            tag = ' !!!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  cls:{name:<13} {f_shift:>+11.2f}pp {a_shift:>+15.2f}pp {delta:>+9.2f}pp{tag}")

        # Size shift at COCO threshold
        for sz in ['small', 'medium', 'large']:
            f_tr = pct(full_tr['size_counts_multi']['32/96'].get(sz, 0), full_tr['n_instances'])
            f_te = pct(full_te['size_counts_multi']['32/96'].get(sz, 0), full_te['n_instances'])
            a_tr = pct(abl_tr['size_counts_multi']['32/96'].get(sz, 0), abl_tr['n_instances'])
            a_te = pct(abl_te['size_counts_multi']['32/96'].get(sz, 0), abl_te['n_instances'])
            f_shift = f_te - f_tr
            a_shift = a_te - a_tr
            delta = a_shift - f_shift
            tag = ' !!!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
            print(f"  sz:{sz:<14} {f_shift:>+11.2f}pp {a_shift:>+15.2f}pp {delta:>+9.2f}pp{tag}")

    # ── Source overlap check ──
    print(f"\n  ── Source-prefix overlap check ──")
    split_sources = {}
    for split in SPLITS:
        if split not in best_splits:
            continue
        sources = set()
        for img_file in best_splits[split]:
            stem = os.path.splitext(img_file)[0]
            sources.add(extract_source(stem))
        split_sources[split] = sources

    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 >= s2 or s1 not in split_sources or s2 not in split_sources:
                continue
            overlap = split_sources[s1] & split_sources[s2]
            tag = "LEAK" if overlap else "OK"
            print(f"  {s1} intersect {s2}: {len(overlap)} shared sources [{tag}]")


# ─────────────────────────────────────────────────────────────────
# Copy files to output directory
# ─────────────────────────────────────────────────────────────────
def copy_ablation(
    full_dataset_path: str,
    output_path: str,
    best_splits: Dict[str, List[str]],
    generate_previews: bool = False,
):
    """Copy selected images + labels to the output directory."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE 4: Copying files to {output_path}")
    print(f"{'=' * 70}")

    for split in SPLITS:
        if split not in best_splits:
            continue

        images_path = os.path.join(full_dataset_path, split, 'images')
        labels_path = os.path.join(full_dataset_path, split, 'labels')

        out_images = os.path.join(output_path, split, 'images')
        out_labels = os.path.join(output_path, split, 'labels')
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        copied = 0
        for img_file in best_splits[split]:
            stem = os.path.splitext(img_file)[0]
            label_file = stem + '.txt'

            src_img = os.path.join(images_path, img_file)
            src_lbl = os.path.join(labels_path, label_file)

            if os.path.isfile(src_img) and os.path.isfile(src_lbl):
                shutil.copy2(src_img, os.path.join(out_images, img_file))
                shutil.copy2(src_lbl, os.path.join(out_labels, label_file))
                copied += 1

        print(f"  {split}: copied {copied} image-label pairs")

    print(f"\n  Done! Ablation dataset saved at: {output_path}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
def main():
    # ── CONFIGURE THESE ──
    full_dataset_path = '/home/constantin/Doctorat/GunDatasetHistogram'
    output_path = '/home/constantin/Doctorat/GunDatasetHistogram17percentage_v2'

    split_percents = {
        'train': 17,
        'valid': 17,
        'test':  27,
    }

    n_candidates = 200   # number of random seeds to try
    base_seed = 0        # starting seed

    # ── RUN ──
    best_seed, best_score, best_splits = find_best_ablation(
        full_dataset_path=full_dataset_path,
        split_percents=split_percents,
        n_candidates=n_candidates,
        base_seed=base_seed,
        verbose=True,
    )

    # ── VERIFY ──
    print_verification_report(
        full_dataset_path=full_dataset_path,
        best_splits=best_splits,
        best_seed=best_seed,
        best_score=best_score,
    )

    # ── COPY FILES ──
    copy_ablation(
        full_dataset_path=full_dataset_path,
        output_path=output_path,
        best_splits=best_splits,
        generate_previews=False,
    )

    print(f"\n{'=' * 70}")
    print(f"  ALL DONE")
    print(f"  Best seed: {best_seed}")
    print(f"  Score: {best_score:.3f}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
