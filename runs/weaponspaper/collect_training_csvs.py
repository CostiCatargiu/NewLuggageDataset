#!/usr/bin/env python3
"""
Collect training CSV files from run folders.

Goes through all subfolders in a given directory, finds results.csv files,
renames them with the folder name, and copies them to an output folder.

Example:
  Input structure:
    runs_new_weapon_dataset/
      swa_09_04_ds2/
        results.csv
      weapon_tuned_all72/
        results.csv
      wt_boost15/
        results.csv

  Output:
    training_csvs/
      swa_09_04_ds2_results.csv
      weapon_tuned_all72_results.csv
      wt_boost15_results.csv

Usage:
  python collect_training_csvs.py
  python collect_training_csvs.py --input /path/to/runs --output /path/to/output
  python collect_training_csvs.py --csv-name results.csv  (default)
"""

import argparse
import shutil
from pathlib import Path


def collect_csvs(input_dir: str, output_dir: str, csv_name: str = "results.csv"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    found = 0
    skipped = 0

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue

        csv_file = folder / csv_name
        if not csv_file.exists():
            # Also check in subfolders (e.g., folder/weights/results.csv)
            csv_candidates = list(folder.rglob(csv_name))
            if csv_candidates:
                csv_file = csv_candidates[0]
            else:
                skipped += 1
                continue

        # Rename: folder_name + _results.csv
        new_name = f"{folder.name}_results.csv"
        dest = output_path / new_name

        if dest.exists():
            print(f"  [SKIP] {new_name} — already exists in output")
            skipped += 1
            continue

        shutil.copy2(csv_file, dest)
        found += 1
        print(f"  [OK] {folder.name}/ → {new_name}")

    print(f"\n{'=' * 50}")
    print(f"  Collected: {found} CSV files")
    print(f"  Skipped:   {skipped} folders (no CSV or already copied)")
    print(f"  Output:    {output_path}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Collect training CSVs from run folders")
    parser.add_argument("--input", "-i", type=str, default="runs_new_weapon_dataset",
                        help="Input directory with run folders (default: runs_new_weapon_dataset)")
    parser.add_argument("--output", "-o", type=str, default="training_csvs",
                        help="Output directory for collected CSVs (default: training_csvs)")
    parser.add_argument("--csv-name", type=str, default="results.csv",
                        help="Name of CSV file to look for (default: results.csv)")
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print(f"  Collecting training CSVs")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  CSV:    {args.csv_name}")
    print(f"{'=' * 50}\n")

    collect_csvs(args.input, args.output, args.csv_name)


if __name__ == "__main__":
    main()
