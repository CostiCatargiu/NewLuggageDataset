#!/usr/bin/env python3
"""
Append the NEW candidate boxes into the dataset's original label .txt files.

For each image, reads the augmented label (pseudo_labels_other/<split>/) and the
original dataset label, finds the lines that are NEW, and appends ONLY those to
the original .txt. Backs up each split's labels dir once before touching it.

SAFETY: defaults to TRAIN ONLY. Do not auto-merge val/test (review them first).
"""
import os, glob, shutil, yaml

DS          = "/home/constantin/Doctorat/GunDatasetNoAugSplit"
DATA_YAML   = f"{DS}/data.yaml"
PSEUDO_ROOT = f"{DS}/pseudo_labels_other"
LABELS_ROOT = f"{DS}/labels"          # original label dirs: labels/<split>/
SPLITS      = ["train"]               # TRAIN ONLY by default
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


def main():
    data = yaml.safe_load(open(DATA_YAML))
    for split in SPLITS:
        backup = os.path.join(LABELS_ROOT, f"{split}_backup")
        src = os.path.join(LABELS_ROOT, split)
        if not os.path.exists(backup):
            shutil.copytree(src, backup)
            print(f"  backed up {src} -> {backup}")
        else:
            print(f"  backup already exists: {backup} (skipping backup)")

        added_boxes = added_imgs = 0
        for img in split_images(data, split):
            stem = os.path.splitext(os.path.basename(img))[0]
            aug_path = os.path.join(PSEUDO_ROOT, split, stem + ".txt")
            orig_path = orig_label(img)
            if not os.path.isfile(aug_path):
                continue
            orig_lines = [l.strip() for l in open(orig_path)] if os.path.isfile(orig_path) else []
            aug_lines = [l.strip() for l in open(aug_path) if l.strip()]
            new_lines = [l for l in aug_lines if l and l not in orig_lines]
            if not new_lines:
                continue
            os.makedirs(os.path.dirname(orig_path), exist_ok=True)
            with open(orig_path, "a") as f:
                # ensure newline before appending
                if orig_lines and not open(orig_path).read().endswith("\n"):
                    f.write("\n")
                f.write("\n".join(new_lines) + "\n")
            added_boxes += len(new_lines)
            added_imgs += 1
        print(f"  [{split}] appended {added_boxes} boxes into {added_imgs} label files")
    print("\n  done. Revert any split with: rm -r labels/<split> && mv labels/<split>_backup labels/<split>")


if __name__ == "__main__":
    main()
