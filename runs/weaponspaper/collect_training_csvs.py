#!/usr/bin/env python3
"""
Collect all CSV files from training run subfolders.

Scans all subfolders in INPUT_DIR, finds any .csv file,
renames it with the folder name prefix, and copies to OUTPUT_DIR.

Example:
  runs_new_weapon_dataset/
    swa_09_04_ds2/
      results.csv
      some_other.csv
    weapon_tuned_all72/
      results.csv

  Output:
    training_csvs/
      swa_09_04_ds2__results.csv
      swa_09_04_ds2__some_other.csv
      weapon_tuned_all72__results.csv
"""

import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION — change these paths
# =============================================================================
INPUT_DIR = "runs_new_weapon_dataset"
OUTPUT_DIR = "training_csvs"

# =============================================================================

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    found = 0
    skipped = 0

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue

        # Find ALL csv files in this folder and subfolders
        csv_files = list(folder.rglob("*.csv"))

        if not csv_files:
            skipped += 1
            continue

        for csv_file in csv_files:
            # Build name: foldername__csvname.csv
            new_name = f"{folder.name}__{csv_file.name}"
            dest = output_path / new_name

            if dest.exists():
                print(f"  [SKIP] {new_name} — already exists")
                continue

            shutil.copy2(csv_file, dest)
            found += 1
            print(f"  [OK] {folder.name}/{csv_file.name} → {new_name}")

    print(f"\n{'=' * 50}")
    print(f"  Collected: {found} CSV files")
    print(f"  Skipped:   {skipped} folders (no CSV)")
    print(f"  Output:    {output_path.resolve()}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
