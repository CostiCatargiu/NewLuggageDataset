#!/usr/bin/env python3
"""
Merge 2 dataset folders (each with train/valid splits) into a single
folder with all images and labels together in one train split.

Input:
  dataset1/
    train/images/  train/labels/
    valid/images/  valid/labels/
  dataset2/
    train/images/  train/labels/
    valid/images/  valid/labels/

Output:
  merged/
    train/images/   (all images from both datasets, all splits)
    train/labels/   (all labels from both datasets, all splits)

Also reports:
  - Total images/labels copied
  - Duplicate filenames (if any)
  - Class distribution

Usage:
  python merge_datasets.py
"""

import os
import shutil
from collections import Counter
from pathlib import Path

# =============================================================================
# CONFIGURATION — update these paths
# =============================================================================
# Each dataset entry has:
#   "path": folder path
#   "class_remap": dict mapping old_class_id -> new_class_id (optional)
#                  If not set, class IDs are kept as-is
DATASETS = [
    {
        "path": r"c:\Users\uids9378\Downloads\WeaponDataset.v11i.yolov12",
        "class_remap": None,  # classes already correct: 0=knife, 1=long_gun, 2=other, 3=pistol
    },
    {
        "path": r"c:\Users\uids9378\Downloads\NoGUN.v5i.yolov12",
        "class_remap": {0: 2},  # dataset 2 has only "other" as class 0 → remap to 2
    },
]

OUTPUT_PATH = r"c:\DISK\GunDataset4"

# Splits to merge from each dataset (skips if not found)
SPLITS = ["train", "valid", "test"]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
CLASS_NAMES = ['knife', 'long_gun', 'other', 'pistol']


# =============================================================================
# MAIN
# =============================================================================
def main():
    out_images = os.path.join(OUTPUT_PATH, "train", "images")
    out_labels = os.path.join(OUTPUT_PATH, "train", "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    seen_filenames = {}  # filename -> source path (for duplicate detection)
    total_copied = 0
    total_skipped = 0
    class_counts = Counter()

    print(f"\n{'=' * 70}")
    print(f"  MERGE DATASETS")
    print(f"{'=' * 70}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Sources: {len(DATASETS)} datasets x {len(SPLITS)} splits")
    print(f"{'=' * 70}\n")

    for ds_entry in DATASETS:
        ds_path = ds_entry["path"]
        class_remap = ds_entry.get("class_remap", None)

        print(f"\n  Dataset: {ds_path}")
        if class_remap:
            remap_str = ", ".join(f"{old}->{new} ({CLASS_NAMES[new]})" for old, new in class_remap.items())
            print(f"    Class remap: {remap_str}")
        else:
            print(f"    Class remap: none (keep as-is)")

        for split in SPLITS:
            img_dir = os.path.join(ds_path, split, "images")
            lbl_dir = os.path.join(ds_path, split, "labels")

            if not os.path.isdir(img_dir):
                print(f"    {split}/images — NOT FOUND, skipping")
                continue

            images = [
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]

            copied = 0
            skipped = 0

            for img_file in images:
                stem = os.path.splitext(img_file)[0]
                label_file = stem + ".txt"
                label_path = os.path.join(lbl_dir, label_file)

                # Skip if no label
                if not os.path.isfile(label_path):
                    continue

                # Check for duplicate filename
                if img_file in seen_filenames:
                    skipped += 1
                    total_skipped += 1
                    continue

                # Copy image
                shutil.copy2(
                    os.path.join(img_dir, img_file),
                    os.path.join(out_images, img_file)
                )

                # Read label, remap classes if needed, write to output
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(float(parts[0]))
                            if class_remap and cid in class_remap:
                                cid = class_remap[cid]
                            new_lines.append(f"{cid} {' '.join(parts[1:])}\n")
                            class_counts[cid] += 1
                        except ValueError:
                            new_lines.append(line)
                    elif line.strip():
                        new_lines.append(line)

                with open(os.path.join(out_labels, label_file), 'w') as f:
                    f.writelines(new_lines)

                seen_filenames[img_file] = os.path.join(ds_path, split)
                copied += 1
                total_copied += 1

            print(f"    {split}: {copied} copied, {skipped} duplicates skipped")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
    print(f"  Total images copied:  {total_copied}")
    print(f"  Duplicates skipped:   {total_skipped}")
    print(f"  Output: {OUTPUT_PATH}/train/")

    print(f"\n  Class distribution:")
    total_instances = sum(class_counts.values())
    for cid, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cid, 0)
        pct = 100.0 * count / total_instances if total_instances else 0
        print(f"    {name:<12} {count:>8} ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<12} {total_instances:>8}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
