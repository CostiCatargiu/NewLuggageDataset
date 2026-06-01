#!/usr/bin/env python3
"""
Remove images with empty label files from dataset splits.

Scans all splits (train/valid/test), finds label files that are empty
(0 bytes or only whitespace), and deletes both the label and its
corresponding image.

Usage:
  python remove_empty_labels.py
"""

import os

# =============================================================================
# CONFIGURATION — add all dataset paths to clean
# =============================================================================
DATASET_PATHS = [
    "/home/constantin/Doctorat/GunDatasetNoAugSplit",          # full
    "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation",  # ablation
]

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


# =============================================================================
# MAIN
# =============================================================================
def main():
    total_removed = 0

    for ds_path in DATASET_PATHS:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {ds_path}")
        print(f"{'=' * 60}")

        for split in SPLITS:
            img_dir = os.path.join(ds_path, split, "images")
            lbl_dir = os.path.join(ds_path, split, "labels")

            if not os.path.isdir(lbl_dir):
                print(f"  {split}: NOT FOUND, skipping")
                continue

            removed = 0
            label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

            for lbl_file in label_files:
                lbl_path = os.path.join(lbl_dir, lbl_file)

                # Check if empty
                is_empty = False
                with open(lbl_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        is_empty = True

                if is_empty:
                    # Delete label
                    os.remove(lbl_path)

                    # Delete matching image
                    stem = os.path.splitext(lbl_file)[0]
                    for ext in IMAGE_EXTS:
                        img_path = os.path.join(img_dir, stem + ext)
                        if os.path.isfile(img_path):
                            os.remove(img_path)
                            break

                    removed += 1

            total_removed += removed
            print(f"  {split}: removed {removed} empty-label pairs")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL REMOVED: {total_removed}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
