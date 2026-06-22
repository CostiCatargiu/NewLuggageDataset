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
import re
import csv
import glob
import shutil

# =============================================================================
# CONFIG
# =============================================================================
REVIEW_DS  = "/home/constantin/Doctorat/GunDatasetNoAugSplit/review_candidates2_reviewed"  # reviewed export root
NAME_MAP   = "/home/constantin/Doctorat/GunDatasetNoAugSplit/review_candidates2/name_map.csv"
BACKUP_SUB = "labels_prereview_backup"
DRY_RUN    = True     # True = report only; False = overwrite original labels

# trailing tokens Roboflow appends: '_<ext>' optionally followed by an aug index '_<n>'
_EXT_TAIL = re.compile(r"_(jpg|jpeg|png|bmp|webp)(_\d+)?$")


def canon(name):
    """Canonical key: drop ext -> cut at '.rf' -> lowercase -> repeatedly strip any
    trailing '_<ext>' (optionally followed by '_<digit>' aug index). Applied to BOTH
    the reviewed names and the map's exported_image so the round-trip stacking of
    _jpg/_jpeg/_png tokens (and _1/_2 copies) collapses to the same stable base."""
    b = os.path.splitext(os.path.basename(name))[0]
    b = b.split(".rf", 1)[0].lower()
    while True:
        nb = _EXT_TAIL.sub("", b)
        if nb == b:
            return b
        b = nb


def orig_label_from_image(img_path):
    parts = img_path.replace("\\", "/").rsplit("/images/", 1)
    lbl = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img_path)[0]
    return os.path.splitext(lbl)[0] + ".txt", (parts[0] if len(parts) == 2 else os.path.dirname(img_path))


def load_map():
    """canon(exported_image) -> row.  Collisions (same canon from >1 export) -> None."""
    m = {}
    with open(NAME_MAP) as f:
        for row in csv.DictReader(f):
            k = canon(row["exported_image"])
            m[k] = None if k in m else row     # mark ambiguous keys as None
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
        key = canon(rl)
        row = name_map.get(key)
        if row is None:
            unmatched += 1
            print(f"    UNMATCHED  {os.path.basename(rl)}  (key='{key}')")
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
