#!/usr/bin/env python3
"""
Push REVIEWED labels back into the ORIGINAL dataset, using name_map.csv to resolve
the Roboflow round-trip renaming exactly.

Roboflow round-trip: your export was  train__<base>.jpg  (recorded in name_map.csv
as 'exported_image'). After upload/review/re-download Roboflow renames it to
  train__<base>_jpg.rf.<NEWHASH>.jpg   (appends an ext token + a NEW .rf hash).
So we map a reviewed file back by:
  reviewed name -> strip ext -> cut at '.rf' -> strip ONE trailing _jpg/_jpeg/_png
  -> match to the 'exported_image' stem in name_map.csv
  -> get that row's original_path -> overwrite that original label (in its split).

The ORIGINAL split is preserved (labels overwritten in place). Per-split backup
made once. DRY_RUN=True first to see the match report, then set False.
"""
import os
import csv
import glob
import shutil

# =============================================================================
# CONFIG
# =============================================================================
REVIEW_DS  = "/home/constantin/Doctorat/GunDatasetNoAugSplit/review_candidates2_reviewed"  # reviewed export root
NAME_MAP   = "/home/constantin/Doctorat/GunDatasetNoAugSplit/review_candidates2/name_map.csv"
BACKUP_SUB = "labels_prereview_backup"
EXT_TOKENS = ("_jpg", "_jpeg", "_png", "_bmp", "_webp")
DRY_RUN    = True     # True = report only; False = overwrite original labels


def core_key(name):
    """basename -> drop ext -> cut at '.rf'  (lowercased)."""
    base = os.path.splitext(os.path.basename(name))[0]
    return base.split(".rf", 1)[0].lower()


def strip_one_ext_token(s):
    for t in EXT_TOKENS:
        if s.endswith(t):
            return s[: -len(t)]
    return s


def orig_label_from_image(img_path):
    parts = img_path.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img_path)[0]
    return os.path.splitext(lbl)[0] + ".txt", (parts[0] if len(parts) == 2 else os.path.dirname(img_path))


def load_map():
    """exported_image stem (lower) -> dict(original_path, original_stem, split)."""
    m = {}
    with open(NAME_MAP) as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(row["exported_image"]))[0].lower()
            m[stem] = row
    return m


def main():
    name_map = load_map()
    print(f"  name_map entries: {len(name_map)}")

    reviewed = sorted(glob.glob(os.path.join(REVIEW_DS, "**", "*.txt"), recursive=True))
    reviewed = [p for p in reviewed if os.path.basename(p).lower() not in ("classes.txt", "readme.txt")]
    print(f"  reviewed label files: {len(reviewed)}\n")

    backed_up = set()
    matched = unmatched = written = 0
    for rl in reviewed:
        core = core_key(rl)
        row = name_map.get(core) or name_map.get(strip_one_ext_token(core))
        if row is None:
            unmatched += 1
            print(f"    UNMATCHED  {os.path.basename(rl)}  (key='{core}')")
            continue

        matched += 1
        orig_img = row["original_path"]
        orig_lbl, split_dir = orig_label_from_image(orig_img)
        if DRY_RUN:
            continue

        # back up this split's labels once
        bak = os.path.join(split_dir, BACKUP_SUB)
        src = os.path.join(split_dir, "labels")
        if src not in backed_up:
            if os.path.isdir(src) and not os.path.isdir(bak):
                shutil.copytree(src, bak)
                print(f"    [backup] {src} -> {bak}")
            backed_up.add(src)

        os.makedirs(os.path.dirname(orig_lbl), exist_ok=True)
        shutil.copyfile(rl, orig_lbl)
        written += 1

    print(f"\n  matched: {matched}   unmatched: {unmatched}")
    if DRY_RUN:
        print("  DRY_RUN=True — nothing written. Check matches, then set DRY_RUN=False.")
    else:
        print(f"  original labels overwritten: {written}  (per-split backups in '{BACKUP_SUB}')")


if __name__ == "__main__":
    main()
