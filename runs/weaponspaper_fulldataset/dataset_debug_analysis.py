"""
Dataset Debug Analysis — Why does test performance differ from validation?

Compares box distributions, spatial patterns, and image characteristics
across train/valid/test splits to identify distribution shifts.

No model inference needed — purely annotation + image analysis.

Outputs:
  1. Box position heatmaps (where objects tend to appear per split)
  2. Box size distributions per split per class
  3. Aspect ratio distributions per split per class
  4. Spatial center distributions (are objects centered or at edges?)
  5. Per-image object count distributions
  6. Image brightness/contrast analysis (crude lighting comparison)
  7. Near-duplicate detection via perceptual hashing
  8. Class co-occurrence patterns (which classes appear together)
  9. Sample grid visualizations per split
"""

import os
import sys
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Optional: perceptual hashing
try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[WARN] imagehash not installed — skipping perceptual duplicate detection")


# =============================================================================
# Configuration
# =============================================================================

DATASET_ROOT = "/home/constantin/Doctorat/GunDatasetHistogram"
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
SPLITS = ['train', 'valid', 'test']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
IMAGE_SIZE = 640  # model input size

# Output directory for plots and grids
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")

# How many images to sample for brightness analysis (per split)
BRIGHTNESS_SAMPLE_SIZE = 2000

# Grid visualization settings
GRID_SAMPLES_PER_SPLIT = 16  # 4x4 grid
GRID_THUMB_SIZE = 320

# Perceptual hash settings
PHASH_SAMPLE_SIZE = 5000  # per split, for speed
PHASH_THRESHOLD = 6  # hamming distance ≤ this = near-duplicate


# =============================================================================
# Annotation Parsing
# =============================================================================

def parse_label_file(label_path):
    """Parse YOLO label file, return list of (class_id, cx, cy, w, h) normalized."""
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append((cid, cx, cy, w, h))
    except Exception:
        pass
    return annotations


def get_image_label_pairs(split_dir):
    """Get list of (image_path, label_path) pairs."""
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    
    if not os.path.isdir(images_dir):
        return []
    
    pairs = []
    for f in sorted(os.listdir(images_dir)):
        if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(f)[0]
        label_path = os.path.join(labels_dir, stem + '.txt')
        if os.path.exists(label_path):
            pairs.append((os.path.join(images_dir, f), label_path))
    return pairs


# =============================================================================
# 1. Box Statistics Collection
# =============================================================================

def collect_box_stats(dataset_root):
    """Collect all box statistics per split."""
    stats = {}
    
    for split in SPLITS:
        split_dir = os.path.join(dataset_root, split)
        pairs = get_image_label_pairs(split_dir)
        
        if not pairs:
            print(f"  [WARN] No pairs found for split: {split}")
            continue
        
        boxes = []  # (class_id, cx, cy, w, h, area)
        per_image_counts = []
        empty_images = 0
        class_cooccurrence = Counter()  # frozenset of classes in same image
        
        for img_path, label_path in pairs:
            anns = parse_label_file(label_path)
            per_image_counts.append(len(anns))
            
            if len(anns) == 0:
                empty_images += 1
                continue
            
            classes_in_image = set()
            for cid, cx, cy, w, h in anns:
                area = w * h
                boxes.append((cid, cx, cy, w, h, area))
                classes_in_image.add(cid)
            
            if len(classes_in_image) > 1:
                class_cooccurrence[frozenset(classes_in_image)] += 1
        
        stats[split] = {
            'n_images': len(pairs),
            'n_boxes': len(boxes),
            'empty_images': empty_images,
            'boxes': boxes,
            'per_image_counts': per_image_counts,
            'class_cooccurrence': class_cooccurrence,
            'pairs': pairs,
        }
        
        print(f"  {split}: {len(pairs)} images, {len(boxes)} boxes, {empty_images} empty")
    
    return stats


# =============================================================================
# 2. Position Heatmap
# =============================================================================

def compute_position_heatmap(boxes, grid_size=20):
    """Compute 2D histogram of box centers."""
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    for _, cx, cy, _, _, _ in boxes:
        gx = min(int(cx * grid_size), grid_size - 1)
        gy = min(int(cy * grid_size), grid_size - 1)
        heatmap[gy, gx] += 1
    
    if heatmap.sum() > 0:
        heatmap /= heatmap.sum()
    
    return heatmap


def save_heatmap(heatmap, title, filepath, grid_size=20):
    """Save heatmap as image using OpenCV."""
    # Scale up for visibility
    scale = 30
    img = np.zeros((grid_size * scale, grid_size * scale, 3), dtype=np.uint8)
    
    max_val = heatmap.max() if heatmap.max() > 0 else 1
    
    for y in range(grid_size):
        for x in range(grid_size):
            val = heatmap[y, x] / max_val
            # Blue (cold) to Red (hot)
            b = int(255 * max(0, 1 - 2 * val))
            g = int(255 * max(0, 1 - abs(2 * val - 1)))
            r = int(255 * min(1, 2 * val))
            cv2.rectangle(img,
                          (x * scale, y * scale),
                          ((x + 1) * scale - 1, (y + 1) * scale - 1),
                          (b, g, r), -1)
    
    # Add title
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(filepath, img)


# =============================================================================
# 3. Distribution Analysis
# =============================================================================

def compute_distributions(stats):
    """Compute per-split, per-class distributions."""
    report = {}
    
    for split, s in stats.items():
        boxes = s['boxes']
        
        # Overall
        areas = [b[5] for b in boxes]
        widths = [b[3] for b in boxes]
        heights = [b[4] for b in boxes]
        aspect_ratios = [b[4] / b[3] if b[3] > 0 else 0 for b in boxes]
        centers_x = [b[1] for b in boxes]
        centers_y = [b[2] for b in boxes]
        
        # Per class
        per_class = defaultdict(lambda: {'areas': [], 'widths': [], 'heights': [],
                                          'aspect_ratios': [], 'cx': [], 'cy': []})
        for cid, cx, cy, w, h, area in boxes:
            per_class[cid]['areas'].append(area)
            per_class[cid]['widths'].append(w)
            per_class[cid]['heights'].append(h)
            per_class[cid]['aspect_ratios'].append(h / w if w > 0 else 0)
            per_class[cid]['cx'].append(cx)
            per_class[cid]['cy'].append(cy)
        
        report[split] = {
            'overall': {
                'areas': areas,
                'widths': widths,
                'heights': heights,
                'aspect_ratios': aspect_ratios,
                'centers_x': centers_x,
                'centers_y': centers_y,
            },
            'per_class': dict(per_class),
        }
    
    return report


def percentile_stats(values, name=""):
    """Compute percentile statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'max': float(np.max(arr)),
    }


# =============================================================================
# 4. Brightness / Contrast Analysis
# =============================================================================

def analyze_brightness(pairs, sample_size=2000, seed=42):
    """Sample images and compute mean brightness and contrast."""
    rng = random.Random(seed)
    sampled = rng.sample(pairs, min(sample_size, len(pairs)))
    
    brightness_values = []
    contrast_values = []
    
    for img_path, _ in sampled:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        brightness_values.append(float(np.mean(img)))
        contrast_values.append(float(np.std(img)))
    
    return {
        'n_sampled': len(brightness_values),
        'brightness': percentile_stats(brightness_values),
        'contrast': percentile_stats(contrast_values),
    }


# =============================================================================
# 5. Perceptual Hash Duplicate Detection
# =============================================================================

def compute_phashes(pairs, sample_size=5000, seed=42):
    """Compute perceptual hashes for a sample of images."""
    if not HAS_IMAGEHASH:
        return {}
    
    rng = random.Random(seed)
    sampled = rng.sample(pairs, min(sample_size, len(pairs)))
    
    hashes = {}
    for img_path, _ in sampled:
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img)
            hashes[img_path] = h
        except Exception:
            pass
    
    return hashes


def find_cross_split_duplicates(hash_dict_a, hash_dict_b, threshold=6):
    """Find near-duplicate images between two splits."""
    duplicates = []
    
    paths_a = list(hash_dict_a.keys())
    paths_b = list(hash_dict_b.keys())
    
    for pa in paths_a:
        ha = hash_dict_a[pa]
        for pb in paths_b:
            hb = hash_dict_b[pb]
            dist = ha - hb
            if dist <= threshold:
                duplicates.append((pa, pb, dist))
    
    return duplicates


# =============================================================================
# 6. Sample Grid Visualization
# =============================================================================

def create_sample_grid(pairs, class_names, split_name, output_path,
                       n_samples=16, thumb_size=320, seed=42):
    """Create a grid of sample images with annotations drawn."""
    rng = random.Random(seed)
    sampled = rng.sample(pairs, min(n_samples, len(pairs)))
    
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    
    grid = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    
    colors = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128),
              (128, 255, 0), (0, 255, 255)]
    
    for idx, (img_path, label_path) in enumerate(sampled):
        row = idx // cols
        col = idx % cols
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Draw annotations
        anns = parse_label_file(label_path)
        for cid, cx, cy, bw, bh in anns:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = colors[cid % len(colors)]
            label = class_names[cid] if 0 <= cid < len(class_names) else f'c{cid}'
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Resize to thumbnail
        thumb = cv2.resize(img, (thumb_size, thumb_size))
        
        # Place in grid
        y_off = row * thumb_size
        x_off = col * thumb_size
        grid[y_off:y_off + thumb_size, x_off:x_off + thumb_size] = thumb
    
    # Add split label
    cv2.putText(grid, f"{split_name.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, grid)
    print(f"    Saved grid: {output_path}")


# =============================================================================
# 7. Spatial Quadrant Analysis
# =============================================================================

def quadrant_analysis(boxes):
    """Analyze which quadrant of the image objects appear in."""
    quadrants = Counter()  # TL, TR, BL, BR, Center
    edge_count = 0
    
    for _, cx, cy, w, h, _ in boxes:
        # Quadrant
        if cx < 0.33:
            col = 'L'
        elif cx > 0.67:
            col = 'R'
        else:
            col = 'C'
        
        if cy < 0.33:
            row = 'T'
        elif cy > 0.67:
            row = 'B'
        else:
            row = 'M'
        
        quadrants[row + col] += 1
        
        # Edge detection: box extends to image border
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        if x1 < 0.02 or y1 < 0.02 or x2 > 0.98 or y2 > 0.98:
            edge_count += 1
    
    total = len(boxes)
    return {
        'quadrants': {k: (v, f"{v / total * 100:.1f}%") for k, v in sorted(quadrants.items())},
        'edge_touching': (edge_count, f"{edge_count / total * 100:.1f}%" if total > 0 else "0%"),
        'total': total,
    }


# =============================================================================
# 8. Box Overlap Analysis
# =============================================================================

def box_overlap_stats(boxes):
    """Analyze how much boxes overlap within images (grouped by image implicitly via position clusters)."""
    if len(boxes) < 2:
        return {'mean_iou_with_nearest': 0, 'n_high_overlap': 0}
    
    # This is approximate — we don't have image grouping, so we just
    # check if any boxes are suspiciously similar (near-identical annotations)
    identical_count = 0
    for i in range(len(boxes)):
        for j in range(i + 1, min(i + 5, len(boxes))):  # just check neighbors
            _, cx1, cy1, w1, h1, _ = boxes[i]
            _, cx2, cy2, w2, h2, _ = boxes[j]
            if (abs(cx1 - cx2) < 0.01 and abs(cy1 - cy2) < 0.01 and
                    abs(w1 - w2) < 0.01 and abs(h1 - h2) < 0.01):
                identical_count += 1
    
    return {'near_identical_pairs': identical_count}


# =============================================================================
# MAIN REPORT
# =============================================================================

def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print_separator("DATASET DEBUG ANALYSIS")
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  Output:  {OUTPUT_DIR}")
    
    # ── Step 1: Collect all box statistics ──
    print_separator("1. COLLECTING BOX STATISTICS")
    stats = collect_box_stats(DATASET_ROOT)
    
    if not stats:
        print("[ERROR] No data found. Check DATASET_ROOT path.")
        return
    
    # ── Step 2: Position heatmaps ──
    print_separator("2. POSITION HEATMAPS")
    for split, s in stats.items():
        # Overall heatmap
        heatmap = compute_position_heatmap(s['boxes'])
        save_heatmap(heatmap, f"{split} - all classes",
                     os.path.join(OUTPUT_DIR, f"heatmap_{split}_all.png"))
        
        # Per-class heatmaps
        for cid, name in enumerate(CLASS_NAMES):
            class_boxes = [b for b in s['boxes'] if b[0] == cid]
            if class_boxes:
                heatmap = compute_position_heatmap(class_boxes)
                save_heatmap(heatmap, f"{split} - {name}",
                             os.path.join(OUTPUT_DIR, f"heatmap_{split}_{name}.png"))
        
        print(f"  Saved heatmaps for {split}")
    
    # ── Step 3: Distribution comparison ──
    print_separator("3. BOX DISTRIBUTION COMPARISON")
    distributions = compute_distributions(stats)
    
    for split, dist in distributions.items():
        print(f"\n  ── {split.upper()} ──")
        ovr = dist['overall']
        
        area_stats = percentile_stats(ovr['areas'])
        ar_stats = percentile_stats(ovr['aspect_ratios'])
        cx_stats = percentile_stats(ovr['centers_x'])
        cy_stats = percentile_stats(ovr['centers_y'])
        w_stats = percentile_stats(ovr['widths'])
        h_stats = percentile_stats(ovr['heights'])
        
        print(f"    Area (norm):  mean={area_stats['mean']:.5f}  med={area_stats['p50']:.5f}  "
              f"p10={area_stats['p10']:.5f}  p90={area_stats['p90']:.5f}")
        print(f"    Width (norm): mean={w_stats['mean']:.4f}  med={w_stats['p50']:.4f}  "
              f"p10={w_stats['p10']:.4f}  p90={w_stats['p90']:.4f}")
        print(f"    Height(norm): mean={h_stats['mean']:.4f}  med={h_stats['p50']:.4f}  "
              f"p10={h_stats['p10']:.4f}  p90={h_stats['p90']:.4f}")
        print(f"    Asp. ratio:   mean={ar_stats['mean']:.3f}  med={ar_stats['p50']:.3f}  "
              f"p10={ar_stats['p10']:.3f}  p90={ar_stats['p90']:.3f}")
        print(f"    Center X:     mean={cx_stats['mean']:.3f}  med={cx_stats['p50']:.3f}  "
              f"std={cx_stats['std']:.3f}")
        print(f"    Center Y:     mean={cy_stats['mean']:.3f}  med={cy_stats['p50']:.3f}  "
              f"std={cy_stats['std']:.3f}")
        
        # Per class
        for cid, name in enumerate(CLASS_NAMES):
            if cid in dist['per_class']:
                pc = dist['per_class'][cid]
                a = percentile_stats(pc['areas'])
                ar = percentile_stats(pc['aspect_ratios'])
                cxs = percentile_stats(pc['cx'])
                cys = percentile_stats(pc['cy'])
                print(f"    [{name:>8s}] area med={a['p50']:.5f}  AR med={ar['p50']:.3f}  "
                      f"cx={cxs['mean']:.3f}±{cxs['std']:.3f}  cy={cys['mean']:.3f}±{cys['std']:.3f}  "
                      f"n={len(pc['areas'])}")
    
    # ── Step 4: Cross-split comparison table ──
    print_separator("4. CROSS-SPLIT DISTRIBUTION COMPARISON")
    
    # Compare key metrics between splits
    print(f"\n  {'Metric':<25} ", end="")
    for split in SPLITS:
        if split in distributions:
            print(f"{'  ' + split.upper():>12}", end="")
    print()
    print(f"  {'─' * 60}")
    
    metrics_to_compare = [
        ('Area mean', lambda d: np.mean(d['overall']['areas'])),
        ('Area median', lambda d: np.median(d['overall']['areas'])),
        ('Area P10', lambda d: np.percentile(d['overall']['areas'], 10)),
        ('Area P90', lambda d: np.percentile(d['overall']['areas'], 90)),
        ('Width mean', lambda d: np.mean(d['overall']['widths'])),
        ('Height mean', lambda d: np.mean(d['overall']['heights'])),
        ('AR mean', lambda d: np.mean(d['overall']['aspect_ratios'])),
        ('AR std', lambda d: np.std(d['overall']['aspect_ratios'])),
        ('Center X mean', lambda d: np.mean(d['overall']['centers_x'])),
        ('Center X std', lambda d: np.std(d['overall']['centers_x'])),
        ('Center Y mean', lambda d: np.mean(d['overall']['centers_y'])),
        ('Center Y std', lambda d: np.std(d['overall']['centers_y'])),
    ]
    
    for metric_name, metric_fn in metrics_to_compare:
        print(f"  {metric_name:<25} ", end="")
        values = []
        for split in SPLITS:
            if split in distributions:
                val = metric_fn(distributions[split])
                values.append(val)
                print(f"{val:>12.5f}", end="")
            else:
                print(f"{'—':>12}", end="")
        print()
        
        # Flag large deviations between train and test
        if len(values) >= 3:
            train_val = values[0]
            test_val = values[2]
            if train_val > 0:
                rel_diff = abs(test_val - train_val) / train_val * 100
                if rel_diff > 10:
                    print(f"  {'':25} ⚠️  train→test shift: {rel_diff:.1f}%")
    
    # Per-class area comparison
    print(f"\n  Per-class median area (normalized):")
    print(f"  {'Class':<12}", end="")
    for split in SPLITS:
        if split in distributions:
            print(f"{'  ' + split.upper():>12}", end="")
    print(f"{'  Δ train→test':>15}")
    print(f"  {'─' * 55}")
    
    for cid, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}", end="")
        values = []
        for split in SPLITS:
            if split in distributions and cid in distributions[split]['per_class']:
                val = np.median(distributions[split]['per_class'][cid]['areas'])
                values.append(val)
                print(f"{val:>12.5f}", end="")
            else:
                values.append(None)
                print(f"{'—':>12}", end="")
        
        if values[0] and values[2] and values[0] > 0:
            rel = (values[2] - values[0]) / values[0] * 100
            tag = " ⚠️" if abs(rel) > 10 else ""
            print(f"{rel:>+13.1f}%{tag}", end="")
        print()
    
    # ── Step 5: Quadrant analysis ──
    print_separator("5. SPATIAL QUADRANT ANALYSIS")
    print(f"  (3x3 grid: T=top, M=mid, B=bottom, L=left, C=center, R=right)")
    
    for split, s in stats.items():
        qa = quadrant_analysis(s['boxes'])
        print(f"\n  ── {split.upper()} ({qa['total']} boxes) ──")
        print(f"    Edge-touching: {qa['edge_touching'][0]} ({qa['edge_touching'][1]})")
        
        # Print as 3x3 grid
        for row in ['T', 'M', 'B']:
            row_str = "    "
            for col in ['L', 'C', 'R']:
                key = row + col
                count, pct = qa['quadrants'].get(key, (0, "0.0%"))
                row_str += f"  {key}:{pct:>6s}"
            print(row_str)
    
    # ── Step 6: Per-image count distribution ──
    print_separator("6. PER-IMAGE OBJECT COUNT DISTRIBUTION")
    
    for split, s in stats.items():
        counts = s['per_image_counts']
        c = Counter(counts)
        print(f"\n  ── {split.upper()} ──")
        print(f"    Images: {s['n_images']}  Empty: {s['empty_images']} ({s['empty_images']/s['n_images']*100:.1f}%)")
        print(f"    Mean: {np.mean(counts):.2f}  Med: {np.median(counts):.0f}  "
              f"Max: {max(counts)}  Std: {np.std(counts):.2f}")
        
        # Distribution
        dist_str = "    Distribution: "
        for n_obj in sorted(c.keys())[:10]:
            dist_str += f"{n_obj}obj={c[n_obj]}({c[n_obj]/len(counts)*100:.1f}%) "
        print(dist_str)
    
    # ── Step 7: Class co-occurrence ──
    print_separator("7. CLASS CO-OCCURRENCE")
    
    for split, s in stats.items():
        cooc = s['class_cooccurrence']
        if not cooc:
            print(f"\n  ── {split.upper()} ── No multi-class images")
            continue
        
        print(f"\n  ── {split.upper()} ──")
        total_multi = sum(cooc.values())
        print(f"    Multi-class images: {total_multi}")
        for classes, count in cooc.most_common(10):
            names = sorted([CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"c{c}" for c in classes])
            print(f"      {' + '.join(names)}: {count} images")
    
    # ── Step 8: Brightness analysis ──
    print_separator("8. IMAGE BRIGHTNESS & CONTRAST")
    
    for split, s in stats.items():
        print(f"\n  ── {split.upper()} ──")
        brightness = analyze_brightness(s['pairs'], BRIGHTNESS_SAMPLE_SIZE)
        b = brightness['brightness']
        c = brightness['contrast']
        print(f"    Sampled: {brightness['n_sampled']} images")
        print(f"    Brightness: mean={b['mean']:.1f}  med={b['p50']:.1f}  "
              f"p10={b['p10']:.1f}  p90={b['p90']:.1f}  std={b['std']:.1f}")
        print(f"    Contrast:   mean={c['mean']:.1f}  med={c['p50']:.1f}  "
              f"p10={c['p10']:.1f}  p90={c['p90']:.1f}  std={c['std']:.1f}")
    
    # ── Step 9: Perceptual hash duplicates ──
    if HAS_IMAGEHASH:
        print_separator("9. PERCEPTUAL HASH NEAR-DUPLICATES")
        
        phashes = {}
        for split, s in stats.items():
            print(f"  Computing hashes for {split}...")
            phashes[split] = compute_phashes(s['pairs'], PHASH_SAMPLE_SIZE)
            print(f"    {len(phashes[split])} hashes computed")
        
        # Cross-split duplicates
        for i, split_a in enumerate(SPLITS):
            for split_b in SPLITS[i + 1:]:
                if split_a in phashes and split_b in phashes:
                    dups = find_cross_split_duplicates(
                        phashes[split_a], phashes[split_b], PHASH_THRESHOLD
                    )
                    print(f"\n  {split_a} ↔ {split_b}: {len(dups)} near-duplicates "
                          f"(threshold={PHASH_THRESHOLD})")
                    for pa, pb, dist in dups[:5]:
                        print(f"    dist={dist}: {Path(pa).name} ↔ {Path(pb).name}")
                    if len(dups) > 5:
                        print(f"    ... and {len(dups) - 5} more")
    
    # ── Step 10: Sample grids ──
    print_separator("10. SAMPLE GRID VISUALIZATIONS")
    
    for split, s in stats.items():
        grid_path = os.path.join(OUTPUT_DIR, f"grid_{split}.png")
        create_sample_grid(s['pairs'], CLASS_NAMES, split, grid_path,
                           GRID_SAMPLES_PER_SPLIT, GRID_THUMB_SIZE)
    
    # ── Final Summary ──
    print_separator("SUMMARY: KEY DIFFERENCES TO INVESTIGATE")
    
    if 'train' in distributions and 'test' in distributions:
        train_d = distributions['train']
        test_d = distributions['test']
        valid_d = distributions.get('valid')
        
        # Check area shift
        train_area = np.median(train_d['overall']['areas'])
        test_area = np.median(test_d['overall']['areas'])
        area_shift = (test_area - train_area) / train_area * 100
        
        # Check center shift
        train_cx_std = np.std(train_d['overall']['centers_x'])
        test_cx_std = np.std(test_d['overall']['centers_x'])
        
        train_cy_std = np.std(train_d['overall']['centers_y'])
        test_cy_std = np.std(test_d['overall']['centers_y'])
        
        # Check brightness (if computed)
        train_brightness = analyze_brightness(stats['train']['pairs'], 500)
        test_brightness = analyze_brightness(stats['test']['pairs'], 500)
        bright_shift = (test_brightness['brightness']['mean'] - train_brightness['brightness']['mean'])
        
        print(f"\n  Area shift (train→test):       {area_shift:+.1f}%")
        print(f"  Center X spread (train/test):  {train_cx_std:.3f} / {test_cx_std:.3f}")
        print(f"  Center Y spread (train/test):  {train_cy_std:.3f} / {test_cy_std:.3f}")
        print(f"  Brightness shift:              {bright_shift:+.1f} (mean pixel value)")
        
        if abs(area_shift) > 10:
            print(f"\n  ⚠️  SIGNIFICANT area shift detected — test objects are "
                  f"{'larger' if area_shift > 0 else 'smaller'} than training")
        
        if abs(bright_shift) > 10:
            print(f"\n  ⚠️  SIGNIFICANT brightness shift — test images are "
                  f"{'brighter' if bright_shift > 0 else 'darker'} than training")
        
        # Check edge-touching difference
        train_edge = quadrant_analysis(stats['train']['boxes'])['edge_touching']
        test_edge = quadrant_analysis(stats['test']['boxes'])['edge_touching']
        train_edge_pct = float(train_edge[1].replace('%', ''))
        test_edge_pct = float(test_edge[1].replace('%', ''))
        
        if abs(test_edge_pct - train_edge_pct) > 3:
            print(f"\n  ⚠️  Edge-touching difference: train={train_edge_pct:.1f}% vs test={test_edge_pct:.1f}%")
    
    print(f"\n  Output saved to: {OUTPUT_DIR}")
    print(f"  Check heatmaps and grids for visual inspection!")
    print(f"\n{'=' * 70}")
    print(f"  Analysis complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
