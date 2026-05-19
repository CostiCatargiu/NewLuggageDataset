"""
Create a stratified ablation subset of the luggage detection dataset.

Improvements over the previous version:
  1. Correct class names for the luggage dataset (bag, backpack, trolley).
  2. Fixed quota math — buckets now sum to the target size, not >target due
     to multi-class double-counting.
  3. Stratification by (class × size) — preserves both class balance AND
     small/medium/large object distribution from the source split.
  4. Verification report — prints source vs subset distributions side by
     side so you can confirm faithfulness.

Stratification strategy:
  Each image is assigned to a bucket based on:
    - dominant class (most frequent class in the image)
    - dominant size (most frequent COCO-size of objects in the image)
  This produces (n_classes × 3) buckets, each sampled proportionally.
  This is more robust than per-class sampling alone because it preserves
  the size distribution of the source split.
"""

import os
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import cv2

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 640
SMALL_AREA_NORM = (32 * 32) / (IMAGE_SIZE * IMAGE_SIZE)   # 0.0025
MEDIUM_AREA_NORM = (96 * 96) / (IMAGE_SIZE * IMAGE_SIZE)  # 0.0225
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


# ─────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────
def parse_label_file(label_path):
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


def classify_size(w_norm, h_norm):
    """COCO-style size class at 640px."""
    area = w_norm * h_norm
    if area < SMALL_AREA_NORM:
        return 'small'
    elif area < MEDIUM_AREA_NORM:
        return 'medium'
    else:
        return 'large'


def get_image_signature(label_path):
    """
    Compute the (dominant_class, dominant_size) signature for an image.

    Returns (cls_id, size) where:
      - cls_id is the most frequent class in the image (-1 if no annotations)
      - size is the most frequent size class ('small'/'medium'/'large', or 'none')
    """
    ann = parse_label_file(label_path)
    if not ann:
        return (-1, 'none')

    class_counter = Counter(cid for cid, *_ in ann)
    size_counter = Counter(classify_size(w, h) for _, _, _, w, h in ann)

    dominant_class = class_counter.most_common(1)[0][0]
    dominant_size = size_counter.most_common(1)[0][0]
    return (dominant_class, dominant_size)


def get_split_distribution(images, labels_path, class_names):
    """Compute per-class instance counts and per-size instance counts."""
    class_counts = Counter()
    size_counts = Counter()
    total_instances = 0
    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_path, stem + '.txt')
        for cid, x, y, w, h in parse_label_file(label_file):
            class_counts[cid] += 1
            size_counts[classify_size(w, h)] += 1
            total_instances += 1
    return class_counts, size_counts, total_instances


# ─────────────────────────────────────────────────────────────────
# Stratified sampling (class × size)
# ─────────────────────────────────────────────────────────────────
def stratified_sample(valid_images, labels_path, percent, seed=42):
    """
    Stratified sampling on (dominant_class, dominant_size) signatures.

    Each image is assigned to exactly ONE bucket (no double-counting),
    so bucket quotas sum cleanly to the target total.
    """
    rng = random.Random(seed)

    # Bucket each image by (dominant_class, dominant_size)
    buckets = defaultdict(list)
    for img_file in valid_images:
        stem = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_path, stem + '.txt')
        signature = get_image_signature(label_file)
        buckets[signature].append(img_file)

    target_size = max(1, int(round(len(valid_images) * percent / 100)))
    sampled = []

    # Sample proportionally from each bucket using floor + remainder allocation
    # to ensure we hit target_size exactly.
    n_total = len(valid_images)
    quotas = {}
    fractional = {}
    for sig, bucket in buckets.items():
        exact = len(bucket) / n_total * target_size
        quotas[sig] = int(exact)
        fractional[sig] = exact - quotas[sig]

    # Distribute leftover slots to buckets with largest fractional remainders
    leftover = target_size - sum(quotas.values())
    if leftover > 0:
        sorted_sigs = sorted(fractional, key=fractional.get, reverse=True)
        for sig in sorted_sigs[:leftover]:
            quotas[sig] += 1

    # Sample from each bucket
    for sig, bucket in buckets.items():
        quota = min(quotas[sig], len(bucket))
        if quota > 0:
            sampled.extend(rng.sample(bucket, quota))

    # Final size guard: if buckets were too small to fill quotas, top up randomly
    if len(sampled) < target_size:
        sampled_set = set(sampled)
        remaining = [img for img in valid_images if img not in sampled_set]
        rng.shuffle(remaining)
        sampled.extend(remaining[:target_size - len(sampled)])

    # Trim if rounding pushed us over (shouldn't happen with floor+remainder, but safe)
    if len(sampled) > target_size:
        sampled = rng.sample(sampled, target_size)

    return sampled


# ─────────────────────────────────────────────────────────────────
# Preview / annotation drawing
# ─────────────────────────────────────────────────────────────────
def draw_annotations(img_path, label_path, class_names):
    image = cv2.imread(img_path)
    if image is None:
        return None
    h, w = image.shape[:2]

    for cid, x_c, y_c, bw, bh in parse_label_file(label_path):
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Color per class for clearer previews
        colors = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128)]
        color = colors[cid % len(colors)]
        label = class_names[cid] if 0 <= cid < len(class_names) else f'cls{cid}'

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return image


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────
def print_distribution_comparison(split, source_images, sampled_images, labels_path, class_names):
    """Compare source vs sampled distributions to verify faithfulness."""
    src_cls, src_size, src_total = get_split_distribution(source_images, labels_path, class_names)
    sub_cls, sub_size, sub_total = get_split_distribution(sampled_images, labels_path, class_names)

    print(f"\n   ── {split.upper()} distribution comparison ──")
    print(f"   Source: {len(source_images)} images, {src_total} instances")
    print(f"   Subset: {len(sampled_images)} images, {sub_total} instances")

    print(f"\n   Class distribution:")
    print(f"   {'Class':<14} {'Source %':>10} {'Subset %':>10} {'Δ pp':>8}")
    for cid, name in enumerate(class_names):
        src_pct = 100 * src_cls.get(cid, 0) / src_total if src_total else 0
        sub_pct = 100 * sub_cls.get(cid, 0) / sub_total if sub_total else 0
        delta = sub_pct - src_pct
        tag = ' ⚠️' if abs(delta) > 2.0 else ''
        print(f"   [{cid}] {name:<10} {src_pct:>9.1f}% {sub_pct:>9.1f}% {delta:>+7.2f}pp{tag}")

    print(f"\n   Size distribution:")
    print(f"   {'Size':<14} {'Source %':>10} {'Subset %':>10} {'Δ pp':>8}")
    for size in ['small', 'medium', 'large']:
        src_pct = 100 * src_size.get(size, 0) / src_total if src_total else 0
        sub_pct = 100 * sub_size.get(size, 0) / sub_total if sub_total else 0
        delta = sub_pct - src_pct
        tag = ' ⚠️' if abs(delta) > 2.0 else ''
        print(f"   {size:<14} {src_pct:>9.1f}% {sub_pct:>9.1f}% {delta:>+7.2f}pp{tag}")


# ─────────────────────────────────────────────────────────────────
# Main split processor
# ─────────────────────────────────────────────────────────────────
def create_subset(original_dataset_path, output_dataset_path, split_percents,
                  class_names, seed=42, generate_previews=True):
    """
    Create a stratified subset of the dataset.

    Args:
        original_dataset_path: path with train/valid/test/{images,labels}
        output_dataset_path:   where to write the subset
        split_percents:        {'train': 30, 'valid': 40, 'test': 50}
        class_names:           list of class names matching label IDs
        seed:                  RNG seed for reproducibility
        generate_previews:     if True, write annotated preview images
    """
    for split, percent in split_percents.items():
        print(f"\n{'='*60}")
        print(f"Processing split: {split} ({percent}% subset)")
        print(f"{'='*60}")

        images_path = os.path.join(original_dataset_path, split, 'images')
        labels_path = os.path.join(original_dataset_path, split, 'labels')

        out_images_path = os.path.join(output_dataset_path, split, 'images')
        out_labels_path = os.path.join(output_dataset_path, split, 'labels')
        preview_path = os.path.join(output_dataset_path, 'preview', split)

        os.makedirs(out_images_path, exist_ok=True)
        os.makedirs(out_labels_path, exist_ok=True)
        if generate_previews:
            os.makedirs(preview_path, exist_ok=True)

        # Collect valid image-label pairs
        if not os.path.isdir(images_path):
            print(f"  ⚠️  Images directory missing: {images_path}")
            continue

        all_images = [
            f for f in os.listdir(images_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        valid_images = [
            f for f in all_images
            if os.path.exists(os.path.join(labels_path, os.path.splitext(f)[0] + '.txt'))
        ]

        if not valid_images:
            print(f"  ⚠️  No valid image-label pairs found in {split}. Skipping.")
            continue

        print(f"  Found {len(valid_images)} valid image-label pairs.")

        # Stratified sampling on (class × size)
        sampled_images = stratified_sample(valid_images, labels_path, percent, seed=seed)
        print(f"  ✅ Sampled {len(sampled_images)} / {len(valid_images)} images "
              f"(target: {int(round(len(valid_images) * percent / 100))})")

        # Copy files (and optionally write previews)
        for img_file in sampled_images:
            stem = os.path.splitext(img_file)[0]
            label_file = stem + '.txt'

            src_img = os.path.join(images_path, img_file)
            src_lbl = os.path.join(labels_path, label_file)

            shutil.copy2(src_img, os.path.join(out_images_path, img_file))
            shutil.copy2(src_lbl, os.path.join(out_labels_path, label_file))

            if generate_previews:
                preview = draw_annotations(src_img, src_lbl, class_names)
                if preview is not None:
                    cv2.imwrite(os.path.join(preview_path, img_file), preview)

        # Verification: compare source vs subset distributions
        print_distribution_comparison(
            split, valid_images, sampled_images, labels_path, class_names
        )

    print(f"\n{'='*60}")
    print(f"📁 Subset dataset saved at: {output_dataset_path}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
def main():
    original_dataset_path = '/home/constantin/Doctorat/GunDatasetHistogram'
    output_dataset_path = '/home/constantin/Doctorat/GunDatasetHistogram17percentage'

    # CORRECTED for luggage dataset
    class_names = ['knife', 'long_gun', 'other', 'pistol']

    split_percents = {
        'train':17,
        'valid':17,
        'test':  27,
    }

    create_subset(
        original_dataset_path,
        output_dataset_path,
        split_percents,
        class_names=class_names,
        seed=42,
        generate_previews=True,   # set False to skip preview generation (faster)
    )


if __name__ == "__main__":
    main()