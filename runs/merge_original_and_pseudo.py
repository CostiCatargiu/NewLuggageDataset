#!/usr/bin/env python3
"""
Merge ORIGINAL label .txt files with the PSEUDO candidate .txt files into a new,
complete label set — originals never touched.

For every image:
    merged = original lines  +  pseudo lines that aren't already present (dedup)
Images with no candidates just get a copy of their original labels, so the output
is a COMPLETE label set you can swap in wholesale for training.

Handles both cases of the pseudo files (originals+new, or new-only) — it always
starts from the dataset originals and unions in the candidate lines.

SAFETY: writes to OUT_ROOT (a new dir). Originals are read-only here. Defaults to
TRAIN ONLY — do not auto-merge val/test (review those first).
"""
import os, glob, yaml

DS          = "/home/constantin/Doctorat/GunDatasetNoAugSplit"
DATA_YAML   = f"{DS}/data.yaml"
PSEUDO_ROOT = f"{DS}/pseudo_labels_other"     # candidate labels (per split)
OUT_ROOT    = f"{DS}/labels_merged"           # merged output (per split) — NEW dir
SPLITS      = ["train"]                        # TRAIN ONLY by default
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def split_images(data, key):
    if key not in data or not data[key]:
        return []
    root = data.get("path", "")
    entries = data[key] if isinstance(data[key], list) else [data[key]]
    imgs = []
    for e in entries:
        p = e if os.path.isabs(e) else os.path.join(root, e)
        if os.path.isdir(p):
            for ext in IMG_EXTS:
                imgs += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
        elif p.endswith(".txt") and os.path.isfile(p):
            imgs += [l.strip() if os.path.isabs(l.strip()) else os.path.join(root, l.strip())
                     for l in open(p) if l.strip()]
    return sorted(set(imgs))


def orig_label(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    p = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(p)[0] + ".txt"


def read_lines(path):
    return [l.strip() for l in open(path) if l.strip()] if os.path.isfile(path) else []


def main():
    data = yaml.safe_load(open(DATA_YAML))
    for split in SPLITS:
        out_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)

        n_files = n_added_boxes = n_imgs_added = 0
        for img in split_images(data, split):
            stem = os.path.splitext(os.path.basename(img))[0]
            orig_lines = read_lines(orig_label(img))
            pseudo_lines = read_lines(os.path.join(PSEUDO_ROOT, split, stem + ".txt"))

            new_lines = [l for l in pseudo_lines if l not in orig_lines]   # dedup against originals
            merged = orig_lines + new_lines

            with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
                if merged:
                    f.write("\n".join(merged) + "\n")
                # empty original + no candidate -> empty file (valid: image with no objects)

            n_files += 1
            if new_lines:
                n_imgs_added += 1
                n_added_boxes += len(new_lines)

        print(f"  [{split}] wrote {n_files} merged files -> {out_dir}")
        print(f"           added {n_added_boxes} new boxes across {n_imgs_added} images")

    print("\n  done. To train on the merged labels (originals stay safe):")
    print(f"    cp -r {DS}/labels/<split> {DS}/labels/<split>_backup")
    print(f"    cp -r {OUT_ROOT}/<split>/*  {DS}/labels/<split>/")
    print("  Expect train: ~566 boxes across ~501 images. If 0 -> path mismatch, tell me your layout.")


if __name__ == "__main__":
    main()
