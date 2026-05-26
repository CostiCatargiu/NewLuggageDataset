#!/usr/bin/env python3
"""
Rename folders based on a mapping dictionary.
Run this script from the directory containing the folders to rename.
"""

import os
from pathlib import Path

# === CONFIGURE THIS ===
# Add your folder renames here: "old_name": "new_name"
RENAMES = {
    # Session 1 renames (SWA+TAL naming was wrong)
    "swa_08_5_tal_07_4": "tal_07_4_ds1_run2",
    "swa_09_5_tal_06_5": "tal_06_5_ds1_run2",
    "swa_09_4_tal_06_5": "tal_06_5_ds1_run3",
    "swa_08_5_tal_06_5ds2": "tal_06_5_ds2_run2",
    
    # Session 2 renames (DS1 -> DS2 corrections)
    "swa_09_04_ds12": "swa_09_04_a07_b4_ds2",
    "swa_09_04_ds1": "swa_09_04_a06_b5_ds2",
    "swa_08_05_ds1": "swa_08_05_a05_b6",
    "swa_09_05_ds1": "swa_09_05_a07_b4_ds2",
    "tal_07_5_ds1": "tal_07_5_ds2",
    "tal_08_3_ds1": "tal_08_3_ds2",
}

# === OPTIONAL: Set target directory (leave None to use current directory) ===
TARGET_DIR = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_new_weapon_dataset"


def rename_folders(target_dir=None, dry_run=True):
    """
    Rename folders based on RENAMES mapping.
    
    Args:
        target_dir: Directory to search for folders. Defaults to current directory.
        dry_run: If True, only print what would be renamed without actually renaming.
    """
    if target_dir is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir)
    
    if not target_dir.exists():
        print(f"ERROR: Directory not found: {target_dir}")
        return
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Scanning: {target_dir}")
    print("=" * 60)
    
    renamed_count = 0
    skipped_count = 0
    not_found = []
    
    for old_name, new_name in RENAMES.items():
        old_path = target_dir / old_name
        new_path = target_dir / new_name
        
        if not old_path.exists():
            not_found.append(old_name)
            continue
        
        if new_path.exists():
            print(f"SKIP: '{old_name}' -> '{new_name}' (target already exists!)")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"WOULD RENAME: '{old_name}' -> '{new_name}'")
        else:
            try:
                old_path.rename(new_path)
                print(f"RENAMED: '{old_name}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR renaming '{old_name}': {e}")
                skipped_count += 1
    
    print("=" * 60)
    if dry_run:
        print(f"DRY RUN COMPLETE - No changes made")
        print(f"Would rename: {len(RENAMES) - len(not_found) - skipped_count} folders")
    else:
        print(f"Renamed: {renamed_count} folders")
    
    if skipped_count:
        print(f"Skipped: {skipped_count} folders")
    
    if not_found:
        print(f"\nNot found ({len(not_found)}):")
        for name in not_found:
            print(f"  - {name}")


if __name__ == "__main__":
    import sys
    
    # Check for --run flag to actually rename (default is dry run)
    dry_run = "--run" not in sys.argv
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("Run with --run flag to actually rename folders")
        print("=" * 60)
        print()
    
    rename_folders(target_dir=TARGET_DIR, dry_run=dry_run)
    
    if dry_run:
        print()
        print("To actually rename, run:")
        print("  python rename_folders.py --run")
