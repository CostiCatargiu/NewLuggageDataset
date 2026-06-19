#!/usr/bin/env python3
"""
Count existing "other" GT instances per split, and express the pseudo-label
candidates as a TRUE incompleteness rate (relative to actual "other" annotations
and to images that contain "other" — not relative to the whole dataset).

Fill CANDIDATES with the numbers the pseudo-label run printed.
"""
import os, glob, yaml

DATA_YAML  = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"
OTHER_NAME = "other"
SPLITS     = ["train", "val", "test"]
IMG_EXTS   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# from the pseudo-label generation log (candidate boxes / images-with-candidates):
CANDIDATES = {
    "train": {"boxes": 566, "images": 501},
    "val":   {"boxes": 246, "images": 208},
    "test":  {"boxes": 255, "images": 205},
}


def other_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for i, n in items:
        if str(n).strip().lower() == OTHER_NAME.lower():
            return int(i)
    raise ValueError("other not in names")


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


def lbl(img):
    parts = img.replace("\\", "/").rsplit("/images/", 1)
    p = (parts[0] + "/labels/" + parts[1]) if len(parts) == 2 else os.path.splitext(img)[0]
    return os.path.splitext(p)[0] + ".txt"


def main():
    data = yaml.safe_load(open(DATA_YAML))
    oid = other_id(data["names"])
    print(f"other id = {oid}\n")
    print(f"{'split':<6} {'imgs':>7} {'imgs_w_other':>13} {'other_GT':>9} "
          f"{'cand':>6} {'%boxes_missing':>15} {'%imgs_w_other_missing':>22}")
    for s in SPLITS:
        imgs = split_images(data, s)
        n_other_gt = 0
        n_imgs_other = 0
        for im in imgs:
            lp = lbl(im)
            if not os.path.isfile(lp):
                continue
            cnt = sum(1 for l in open(lp) if l.strip() and int(l.split()[0]) == oid)
            if cnt:
                n_imgs_other += 1
                n_other_gt += cnt
        c = CANDIDATES.get(s, {"boxes": 0, "images": 0})
        pct_boxes = 100 * c["boxes"] / max(n_other_gt + c["boxes"], 1)
        pct_imgs  = 100 * c["images"] / max(n_imgs_other, 1)
        print(f"{s:<6} {len(imgs):>7} {n_imgs_other:>13} {n_other_gt:>9} "
              f"{c['boxes']:>6} {pct_boxes:>14.1f}% {pct_imgs:>21.1f}%")
    print("\n  %boxes_missing = candidates / (existing_other_GT + candidates)  [LOWER BOUND -- model")
    print("  suppresses the unlabeled objects, so true incompleteness is higher].")
    print("  %imgs_w_other_missing = flagged images / images that actually contain 'other'.")


if __name__ == "__main__":
    main()
