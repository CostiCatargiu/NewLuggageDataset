#!/usr/bin/env python3
"""
Inspect Split Dataset — verify quality of train/valid/test split.

Checks:
  1. Basic counts per split
  2. Class distribution per split + deviation from overall
  3. Size distribution per split at multiple thresholds
  4. Per-class size distribution per split
  5. Instances/image stats per split
  6. Source-prefix overlap between splits (leakage check)
  7. Filename overlap between splits (exact leakage)
  8. Train↔Valid↔Test shift analysis
  9. Annotation quality per split
  10. Overall verdict

Usage:
  python inspect_split.py
"""

import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASETS = {
    "FULL": r"c:\DISK\GunDatasetClean",
    "ABLATION": r"c:\DISK\GunDatasetAblation",
}
OUTPUT_FILE = os.path.join(list(DATASETS.values())[0], "split_inspection_comparison.txt")

SPLITS = ['train', 'valid', 'test']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']
IMAGE_SIZE = 640

SIZE_THRESHOLDS = [
    ("24/72",  24,  72),
    ("32/96",  32,  96),
    ("48/144", 48, 144),
    ("64/192", 64, 192),
]


# =============================================================================
# HELPERS
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


def parse_label(label_path):
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
                    issues.append(f"line {line_num}: too few values")
                    continue
                try:
                    cid = int(float(parts[0]))
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if cid < 0 or cid >= len(CLASS_NAMES):
                        issues.append(f"line {line_num}: invalid class {cid}")
                    annotations.append((cid, x, y, w, h))
                except ValueError as e:
                    issues.append(f"line {line_num}: parse error")
    except:
        issues.append("file read error")
    return annotations, issues


def classify_size(w, h, small_px, medium_px):
    area = w * h
    small_area = (small_px ** 2) / (IMAGE_SIZE ** 2)
    medium_area = (medium_px ** 2) / (IMAGE_SIZE ** 2)
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
# ANALYZE ONE SPLIT
# =============================================================================
def analyze_split(split_name, dataset_path):
    img_dir = os.path.join(dataset_path, split_name, "images")
    lbl_dir = os.path.join(dataset_path, split_name, "labels")

    if not os.path.isdir(img_dir):
        return None

    images = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    labels = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')) if os.path.isdir(lbl_dir) else set()

    stats = {
        'split': split_name,
        'n_images': len(images),
        'stems': set(),
        'sources': set(),
        'class_counts': Counter(),
        'size_counts': {t: Counter() for t, _, _ in SIZE_THRESHOLDS},
        'per_class_sizes': {t: {cid: Counter() for cid in range(len(CLASS_NAMES))} for t, _, _ in SIZE_THRESHOLDS},
        'instances_per_image': [],
        'n_instances': 0,
        'empty_labels': 0,
        'orphan_images': 0,
        'annotation_issues': 0,
        'bbox_areas': [],
        'bbox_ars': [],
    }

    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        stats['stems'].add(stem)
        stats['sources'].add(extract_source(stem))

        label_path = os.path.join(lbl_dir, stem + '.txt')
        if not os.path.isfile(label_path):
            stats['orphan_images'] += 1
            stats['instances_per_image'].append(0)
            continue

        annotations, issues = parse_label(label_path)
        if issues:
            stats['annotation_issues'] += len(issues)

        if len(annotations) == 0:
            stats['empty_labels'] += 1

        stats['instances_per_image'].append(len(annotations))
        stats['n_instances'] += len(annotations)

        for cid, x, y, w, h in annotations:
            stats['class_counts'][cid] += 1
            area = w * h
            stats['bbox_areas'].append(area)
            ar = max(w, h) / max(min(w, h), 1e-9)
            stats['bbox_ars'].append(ar)

            for t_label, small_px, medium_px in SIZE_THRESHOLDS:
                sz = classify_size(w, h, small_px, medium_px)
                stats['size_counts'][t_label][sz] += 1
                stats['per_class_sizes'][t_label][cid][sz] += 1

    return stats


# =============================================================================
# ANALYZE ONE FULL DATASET
# =============================================================================
def analyze_dataset(ds_name, ds_path):
    """Analyze all splits of a dataset, return dict of stats + totals."""
    all_stats = {}
    for split in SPLITS:
        stats = analyze_split(split, ds_path)
        if stats:
            all_stats[split] = stats

    # Compute totals
    total_images = sum(s['n_images'] for s in all_stats.values())
    total_instances = sum(s['n_instances'] for s in all_stats.values())
    total_class = Counter()
    total_size = {t: Counter() for t, _, _ in SIZE_THRESHOLDS}
    for s in all_stats.values():
        total_class += s['class_counts']
        for t, _, _ in SIZE_THRESHOLDS:
            total_size[t] += s['size_counts'][t]

    return {
        'name': ds_name,
        'path': ds_path,
        'splits': all_stats,
        'total_images': total_images,
        'total_instances': total_instances,
        'total_class': total_class,
        'total_size': total_size,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    report = ReportWriter(OUTPUT_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report.write("=" * 100)
    report.write(f"  SPLIT INSPECTION — PARALLEL COMPARISON")
    report.write(f"  {timestamp}")
    for ds_name, ds_path in DATASETS.items():
        report.write(f"  {ds_name}: {ds_path}")
    report.write("=" * 100)

    # Analyze all datasets
    all_datasets = {}
    for ds_name, ds_path in DATASETS.items():
        report.write(f"\n  Analyzing {ds_name}...")
        all_datasets[ds_name] = analyze_dataset(ds_name, ds_path)

    ds_names = list(all_datasets.keys())

    # =========================================================================
    # 1. BASIC COUNTS — SIDE BY SIDE
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  1. BASIC COUNTS — SIDE BY SIDE")
    report.write(f"{'=' * 100}")

    for ds_name, ds in all_datasets.items():
        report.write(f"\n  ── {ds_name} ──")
        report.write(f"  {'Split':<10} {'Images':>8} {'%':>7} {'Instances':>10} {'Inst/Img':>10} {'Empty':>7}")
        report.write(f"  {'─' * 60}")
        for split in SPLITS:
            if split not in ds['splits']:
                continue
            s = ds['splits'][split]
            ratio = pct(s['n_images'], ds['total_images'])
            ipi = statistics.mean(s['instances_per_image']) if s['instances_per_image'] else 0
            report.write(f"  {split:<10} {s['n_images']:>8} {ratio:>6.1f}% {s['n_instances']:>10} {ipi:>10.2f} {s['empty_labels']:>7}")
        report.write(f"  {'TOTAL':<10} {ds['total_images']:>8} {'100.0':>6}% {ds['total_instances']:>10}")

    # =========================================================================
    # 2. CLASS DISTRIBUTION — COMPARISON
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  2. CLASS DISTRIBUTION — COMPARISON")
    report.write(f"{'=' * 100}")

    max_class_devs = {}
    for ds_name, ds in all_datasets.items():
        max_class_devs[ds_name] = 0

    for cid, name in enumerate(CLASS_NAMES):
        report.write(f"\n  {name}:")
        # Header
        header = f"  {'Split':<10}"
        for ds_name in ds_names:
            header += f" {'['+ds_name+'] %':>14} {'delta':>8}"
        report.write(header)
        report.write(f"  {'─' * (10 + len(ds_names) * 24)}")

        for split in SPLITS:
            line = f"  {split:<10}"
            for ds_name, ds in all_datasets.items():
                if split not in ds['splits']:
                    line += f" {'N/A':>14} {'':>8}"
                    continue
                s = ds['splits'][split]
                t_pct = pct(ds['total_class'].get(cid, 0), ds['total_instances'])
                s_pct = pct(s['class_counts'].get(cid, 0), s['n_instances'])
                delta = s_pct - t_pct
                max_class_devs[ds_name] = max(max_class_devs[ds_name], abs(delta))
                tag = '!!' if abs(delta) > 2.0 else ('!' if abs(delta) > 1.0 else '')
                line += f" {s_pct:>13.1f}% {delta:>+7.2f}pp{tag}"
            report.write(line)

    report.write(f"\n  Max class deviation:")
    for ds_name, dev in max_class_devs.items():
        report.write(f"    {ds_name}: {dev:.2f}pp")

    # =========================================================================
    # 3. SIZE DISTRIBUTION — COMPARISON
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  3. SIZE DISTRIBUTION — COMPARISON [32/96] (COCO)")
    report.write(f"{'=' * 100}")

    max_size_devs = {}
    for ds_name in ds_names:
        max_size_devs[ds_name] = 0

    t_label = "32/96"
    header = f"  {'Split':<10}"
    for ds_name in ds_names:
        header += f"  {'['+ds_name+']':>10} {'S%':>6} {'M%':>6} {'L%':>6}"
    report.write(header)
    report.write(f"  {'─' * (10 + len(ds_names) * 30)}")

    for split in SPLITS:
        line = f"  {split:<10}"
        for ds_name, ds in all_datasets.items():
            if split not in ds['splits']:
                line += f"  {'N/A':>10} {'':>6} {'':>6} {'':>6}"
                continue
            s = ds['splits'][split]
            sc = s['size_counts'][t_label]
            total_s = sum(sc.values())
            ts = ds['total_size'][t_label]
            total_t = sum(ts.values())

            for sz in ['small', 'medium', 'large']:
                s_pct = pct(sc.get(sz, 0), total_s)
                t_pct = pct(ts.get(sz, 0), total_t)
                max_size_devs[ds_name] = max(max_size_devs[ds_name], abs(s_pct - t_pct))

            sm_pct = pct(sc.get('small', 0), total_s)
            md_pct = pct(sc.get('medium', 0), total_s)
            lg_pct = pct(sc.get('large', 0), total_s)
            line += f"  {total_s:>10} {sm_pct:>5.1f}% {md_pct:>5.1f}% {lg_pct:>5.1f}%"
        report.write(line)

    # Overall
    line = f"  {'OVERALL':<10}"
    for ds_name, ds in all_datasets.items():
        ts = ds['total_size'][t_label]
        total_t = sum(ts.values())
        sm_pct = pct(ts.get('small', 0), total_t)
        md_pct = pct(ts.get('medium', 0), total_t)
        lg_pct = pct(ts.get('large', 0), total_t)
        line += f"  {total_t:>10} {sm_pct:>5.1f}% {md_pct:>5.1f}% {lg_pct:>5.1f}%"
    report.write(line)

    report.write(f"\n  Max size deviation:")
    for ds_name, dev in max_size_devs.items():
        report.write(f"    {ds_name}: {dev:.2f}pp")

    # =========================================================================
    # 4. SIZE DISTRIBUTION — ALL THRESHOLDS
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  4. SIZE DISTRIBUTION — ALL THRESHOLDS (overall totals)")
    report.write(f"{'=' * 100}")

    for t_label_iter, small_px, medium_px in SIZE_THRESHOLDS:
        report.write(f"\n  Threshold [{t_label_iter}] (small<{small_px}px, med<{medium_px}px):")
        header = f"  {'Dataset':<12}"
        header += f" {'Small':>8} {'%':>7} {'Medium':>8} {'%':>7} {'Large':>8} {'%':>7}"
        report.write(header)
        for ds_name, ds in all_datasets.items():
            ts = ds['total_size'][t_label_iter]
            total_t = sum(ts.values())
            sm = ts.get('small', 0)
            md = ts.get('medium', 0)
            lg = ts.get('large', 0)
            report.write(f"  {ds_name:<12} {sm:>8} {pct(sm,total_t):>6.1f}% {md:>8} {pct(md,total_t):>6.1f}% {lg:>8} {pct(lg,total_t):>6.1f}%")

    # =========================================================================
    # 5. PER-CLASS SIZE DISTRIBUTION
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  5. PER-CLASS SIZE DISTRIBUTION [32/96] — COMPARISON")
    report.write(f"{'=' * 100}")

    t_label = "32/96"
    for cid, name in enumerate(CLASS_NAMES):
        report.write(f"\n  {name}:")
        for ds_name, ds in all_datasets.items():
            report.write(f"    [{ds_name}]")
            report.write(f"    {'Split':<10} {'Small':>7} {'Medium':>8} {'Large':>8} {'Total':>8} {'%Small':>8}")
            for split in SPLITS:
                if split not in ds['splits']:
                    continue
                s = ds['splits'][split]
                sc = s['per_class_sizes'][t_label][cid]
                sm = sc.get('small', 0)
                md = sc.get('medium', 0)
                lg = sc.get('large', 0)
                total = sm + md + lg
                report.write(f"    {split:<10} {sm:>7} {md:>8} {lg:>8} {total:>8} {pct(sm,total):>7.1f}%")

    # =========================================================================
    # 6. SOURCE & FILENAME OVERLAP
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  6. SOURCE & FILENAME OVERLAP (leakage check)")
    report.write(f"{'=' * 100}")

    has_leaks = {}
    for ds_name, ds in all_datasets.items():
        has_leaks[ds_name] = False
        report.write(f"\n  ── {ds_name} ──")

        report.write(f"  Source-prefix overlap:")
        for s1 in SPLITS:
            for s2 in SPLITS:
                if s1 >= s2 or s1 not in ds['splits'] or s2 not in ds['splits']:
                    continue
                overlap = ds['splits'][s1]['sources'] & ds['splits'][s2]['sources']
                tag = "LEAK" if overlap else "OK"
                if overlap:
                    has_leaks[ds_name] = True
                report.write(f"    {s1} ∩ {s2}: {len(overlap)} [{tag}]")

        report.write(f"  Filename overlap:")
        for s1 in SPLITS:
            for s2 in SPLITS:
                if s1 >= s2 or s1 not in ds['splits'] or s2 not in ds['splits']:
                    continue
                overlap = ds['splits'][s1]['stems'] & ds['splits'][s2]['stems']
                tag = "LEAK" if overlap else "OK"
                if overlap:
                    has_leaks[ds_name] = True
                report.write(f"    {s1} ∩ {s2}: {len(overlap)} [{tag}]")

    # =========================================================================
    # 7. ABLATION FIDELITY — how well ablation matches full
    # =========================================================================
    if len(all_datasets) > 1:
        report.write(f"\n{'=' * 100}")
        report.write(f"  7. ABLATION FIDELITY — how well ABLATION matches FULL")
        report.write(f"{'=' * 100}")

        full_ds = all_datasets.get(ds_names[0])
        abl_ds = all_datasets.get(ds_names[1])

        if full_ds and abl_ds:
            max_abl_class_dev = 0
            max_abl_size_dev = 0

            report.write(f"\n  Class distribution comparison:")
            report.write(f"  {'Class':<12} {'FULL%':>8} {'ABL%':>8} {'Delta':>8}")
            for cid, name in enumerate(CLASS_NAMES):
                f_pct = pct(full_ds['total_class'].get(cid, 0), full_ds['total_instances'])
                a_pct = pct(abl_ds['total_class'].get(cid, 0), abl_ds['total_instances'])
                delta = a_pct - f_pct
                max_abl_class_dev = max(max_abl_class_dev, abs(delta))
                tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
                report.write(f"  {name:<12} {f_pct:>7.1f}% {a_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

            t_label = "32/96"
            report.write(f"\n  Size distribution comparison [32/96]:")
            report.write(f"  {'Size':<12} {'FULL%':>8} {'ABL%':>8} {'Delta':>8}")
            for sz in ['small', 'medium', 'large']:
                f_total = sum(full_ds['total_size'][t_label].values())
                a_total = sum(abl_ds['total_size'][t_label].values())
                f_pct = pct(full_ds['total_size'][t_label].get(sz, 0), f_total)
                a_pct = pct(abl_ds['total_size'][t_label].get(sz, 0), a_total)
                delta = a_pct - f_pct
                max_abl_size_dev = max(max_abl_size_dev, abs(delta))
                tag = ' !!' if abs(delta) > 2.0 else (' !' if abs(delta) > 1.0 else '')
                report.write(f"  {sz:<12} {f_pct:>7.1f}% {a_pct:>7.1f}% {delta:>+7.2f}pp{tag}")

            # Per-split comparison
            report.write(f"\n  Per-split class deviation (ABLATION vs FULL):")
            report.write(f"  {'Split':<10} {'Max class dev':>15} {'Max size dev':>15}")
            for split in SPLITS:
                if split not in full_ds['splits'] or split not in abl_ds['splits']:
                    continue
                fs = full_ds['splits'][split]
                als = abl_ds['splits'][split]
                max_cd = 0
                max_sd = 0
                for cid in range(len(CLASS_NAMES)):
                    fp = pct(fs['class_counts'].get(cid, 0), fs['n_instances'])
                    ap = pct(als['class_counts'].get(cid, 0), als['n_instances'])
                    max_cd = max(max_cd, abs(fp - ap))
                for sz in ['small', 'medium', 'large']:
                    fsc = fs['size_counts'][t_label]
                    asc = als['size_counts'][t_label]
                    ft = sum(fsc.values())
                    at = sum(asc.values())
                    fp = pct(fsc.get(sz, 0), ft)
                    ap = pct(asc.get(sz, 0), at)
                    max_sd = max(max_sd, abs(fp - ap))
                report.write(f"  {split:<10} {max_cd:>14.2f}pp {max_sd:>14.2f}pp")

            report.write(f"\n  Overall ablation fidelity:")
            report.write(f"    Max class deviation: {max_abl_class_dev:.2f}pp")
            report.write(f"    Max size deviation:  {max_abl_size_dev:.2f}pp")

            if max_abl_class_dev < 0.5 and max_abl_size_dev < 1.0:
                abl_fidelity = "EXCELLENT"
            elif max_abl_class_dev < 1.0 and max_abl_size_dev < 2.0:
                abl_fidelity = "GOOD"
            elif max_abl_class_dev < 2.0 and max_abl_size_dev < 3.0:
                abl_fidelity = "ACCEPTABLE"
            else:
                abl_fidelity = "NEEDS ATTENTION"
            report.write(f"    Ablation fidelity:   {abl_fidelity}")

    # =========================================================================
    # 8. CROSS-SPLIT SHIFT
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  8. TRAIN ↔ TEST SHIFT — COMPARISON")
    report.write(f"{'=' * 100}")

    for ds_name, ds in all_datasets.items():
        if 'train' not in ds['splits'] or 'test' not in ds['splits']:
            continue
        tr = ds['splits']['train']
        te = ds['splits']['test']

        report.write(f"\n  ── {ds_name} ──")
        report.write(f"  {'Metric':<16} {'Train%':>8} {'Test%':>8} {'Shift':>8}")
        report.write(f"  {'─' * 45}")

        for cid, name in enumerate(CLASS_NAMES):
            p1 = pct(tr['class_counts'].get(cid, 0), tr['n_instances'])
            p2 = pct(te['class_counts'].get(cid, 0), te['n_instances'])
            shift = p2 - p1
            tag = ' !!' if abs(shift) > 2.0 else (' !' if abs(shift) > 1.0 else '')
            report.write(f"  cls:{name:<11} {p1:>7.1f}% {p2:>7.1f}% {shift:>+7.2f}pp{tag}")

        t_label = "32/96"
        for sz in ['small', 'medium', 'large']:
            sc1 = tr['size_counts'][t_label]
            sc2 = te['size_counts'][t_label]
            t1 = sum(sc1.values())
            t2 = sum(sc2.values())
            p1 = pct(sc1.get(sz, 0), t1)
            p2 = pct(sc2.get(sz, 0), t2)
            shift = p2 - p1
            tag = ' !!' if abs(shift) > 2.0 else (' !' if abs(shift) > 1.0 else '')
            report.write(f"  sz:{sz:<12} {p1:>7.1f}% {p2:>7.1f}% {shift:>+7.2f}pp{tag}")

    # =========================================================================
    # 9. OVERALL VERDICT
    # =========================================================================
    report.write(f"\n{'=' * 100}")
    report.write(f"  9. OVERALL VERDICT")
    report.write(f"{'=' * 100}")

    for ds_name, ds in all_datasets.items():
        max_cd = max_class_devs.get(ds_name, 0)
        max_sd = max_size_devs.get(ds_name, 0)
        leak = has_leaks.get(ds_name, False)

        if max_cd < 0.5 and max_sd < 1.0 and not leak:
            fidelity = "EXCELLENT"
        elif max_cd < 1.0 and max_sd < 2.0 and not leak:
            fidelity = "GOOD"
        elif max_cd < 2.0 and max_sd < 3.0:
            fidelity = "ACCEPTABLE"
        else:
            fidelity = "NEEDS ATTENTION"

        report.write(f"\n  ── {ds_name} ──")
        report.write(f"    Images:            {ds['total_images']}")
        report.write(f"    Instances:         {ds['total_instances']}")
        report.write(f"    Max class dev:     {max_cd:.2f}pp")
        report.write(f"    Max size dev:      {max_sd:.2f}pp")
        report.write(f"    Source leakage:    {'YES' if leak else 'NONE'}")
        report.write(f"    Split fidelity:    {fidelity}")

    report.write(f"\n{'=' * 100}")

    report.save()


if __name__ == "__main__":
    main()
