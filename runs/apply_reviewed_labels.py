#!/usr/bin/env python3
"""
Push REVIEWED labels back into the ORIGINAL dataset.

Walks the reviewed dataset (all splits), and for each reviewed label finds the
matching image in the ORIGINAL dataset BY NAME, then overwrites that original
label with the reviewed one — keeping each image in its ORIGINAL split.

Why match by a normalized KEY instead of exact filename:
  Roboflow names are  <base>.rf.<hash>.<ext>.  When you re-upload/re-download,
  the hash changes and the split may be re-shuffled. The stable identity is the
  part BEFORE '.rf' (the original base). So we match on:
      strip extension -> cut at '.rf' -> strip any 'split__' prefix the export added
  The ORIGINAL split is preserved (we only replace label *content* at the original
  location, never move images), so a Roboflow re-split can't scramble your eval set.

SAFETY: backs up each original split's labels once (labels -> labels_prereview_backup)
before overwriting. DRY_RUN=True first to preview the match report; then set False.
"""
import os
import glob
import shutil

# =============================================================================
# CONFIG
# =============================================================================
ORIG_DS      = "/home/constantin/Doctorat/GunDatasetNoAugSplit"
ORIG_SPLITS  = ["train", "valid", "test"]
LABELS_SUB   = "labels"
IMAGES_SUB   = "images"

REVIEW_DS    = "/home/constantin/Doctorat/GunDatasetNoAugSplit/review_candidates2_reviewed"  # reviewed export root
BACKUP_SUB   = "labels_prereview_backup"
STRIP_PREFIXES = ["train__", "valid__", "val__", "test__"]
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DRY_RUN      = True     # True = only report matches; False = actually overwrite labels


def norm_key(name, strip_prefix=False):
    """basename -> drop ext -> cut at '.rf' -> optionally strip split prefix."""
    base = os.path.splitext(os.path.basename(name))[0]
    base = base.split(".rf", 1)[0]
    if strip_prefix:
        for p in STRIP_PREFIXES:
            if base.startswith(p):
                base = base[len(p):]
                break
    return base


def build_original_index():
    """key -> list of (split, original_label_path).  Lists catch ambiguous keys."""
    index = {}
    for split in ORIG_SPLITS:
        img_dir = os.path.join(ORIG_DS, split, IMAGES_SUB)
        lbl_dir = os.path.join(ORIG_DS, split, LABELS_SUB)
        if not os.path.isdir(img_dir):
            continue
        for ext in IMG_EXTS:
            for img in glob.glob(os.path.join(img_dir, "**", f"*{ext}"), recursive=True):
                key = norm_key(img)
                lbl = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
                index.setdefault(key, []).append((split, lbl))
    return index


def find_reviewed_labels():
    """All label .txt files anywhere under the reviewed dataset."""
    return sorted(glob.glob(os.path.join(REVIEW_DS, "**", "*.txt"), recursive=True))


def main():
    index = build_original_index()
    print(f"  original images indexed: {sum(len(v) for v in index.values())} "
          f"({len(index)} unique keys)")

    reviewed = find_reviewed_labels()
    reviewed = [p for p in reviewed if os.path.basename(p).lower() not in
                ("classes.txt", "readme.txt")]
    print(f"  reviewed label files:    {len(reviewed)}\n")

    backed_up = set()
    matched = unmatched = ambiguous = written = 0
    for rl in reviewed:
        key = norm_key(rl, strip_prefix=True)
        hits = index.get(key)
        if not hits:
            unmatched += 1
            print(f"    UNMATCHED  {os.path.basename(rl)}  (key='{key}')")
            continue
        if len(hits) > 1:
            ambiguous += 1
            print(f"    AMBIGUOUS  {os.path.basename(rl)}  (key='{key}' -> "
                  f"{len(hits)} originals: {[h[0] for h in hits]}) — skipped")
            continue

        matched += 1
        split, orig_lbl = hits[0]
        if DRY_RUN:
            continue

        # back up this split's labels once
        if split not in backed_up:
            src = os.path.join(ORIG_DS, split, LABELS_SUB)
            bak = os.path.join(ORIG_DS, split, BACKUP_SUB)
            if os.path.isdir(src) and not os.path.isdir(bak):
                shutil.copytree(src, bak)
                print(f"    [backup] {src} -> {bak}")
            backed_up.add(split)

        os.makedirs(os.path.dirname(orig_lbl), exist_ok=True)
        shutil.copyfile(rl, orig_lbl)     # reviewed label overwrites the original (same split)
        written += 1

    print(f"\n  matched: {matched}   unmatched: {unmatched}   ambiguous: {ambiguous}")
    if DRY_RUN:
        print("  DRY_RUN=True — nothing written. Review the matches above, then set DRY_RUN=False.")
    else:
        print(f"  labels overwritten in original dataset: {written}")
        print(f"  (originals backed up per split to '{BACKUP_SUB}'; revert by restoring it.)")


if __name__ == "__main__":
    main()
