#!/usr/bin/env python3
"""
Model soup — average the weights of the top TAL runs (ZERO training cost).

All TAL runs share the IDENTICAL stock YOLOv12s architecture, so their
weights can be averaged directly ("uniform soup", Wortsman et al. 2022).
Soups of fine-tuned variants of the same init typically gain +0.3-1.0 mAP
for free. Greedy mode: add runs one at a time, keep only if val mAP50
improves.

Usage:
  python soup_top_runs.py            # greedy soup (recommended)
  python soup_top_runs.py --uniform  # plain average of all RUNS

Output: runs_noaug_weapon70/soup/weights/soup_best.pt
        -> evaluate it through your usual test_full_dataset pipeline.
"""

import sys
import copy
import os
import torch
from ultralytics import YOLO

PROJECT_DIR = "runs_noaug_weapon70"
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
IMG_SIZE = 640

# Top TAL runs by test mAP50 — identical architecture, different recipes.
# Order matters for greedy mode (best first).
RUNS = [
    "v5_topk15_beta3_70",        # 80.45
    "v5_topk15_tal07_beta35_70", # 80.28
    "v5_tal07_swa09_05_70",      # 80.11
    "v5_topk17_beta3_702",       # 80.09
    "v5_topk15_beta35_70",       # 80.09
    "v5_tal07_70",               # 80.02
]


def load_sd(run):
    p = os.path.join(PROJECT_DIR, run, "weights", "best.pt")
    ckpt = torch.load(p, map_location="cpu")
    return ckpt, ckpt["model"].float().state_dict()


def avg_sds(sds):
    """Average float tensors; keep first value for ints (BN counters)."""
    out = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            out[k] = sum(sd[k] for sd in sds) / len(sds)
        else:
            out[k] = sds[0][k]
    return out


def save_soup(base_ckpt, sd, members):
    ckpt = copy.deepcopy(base_ckpt)
    ckpt["model"].load_state_dict(sd)
    out_dir = os.path.join(PROJECT_DIR, "soup", "weights")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "soup_best.pt")
    torch.save(ckpt, out)
    print(f"\nSaved soup of {len(members)} runs -> {out}")
    print("Members:", ", ".join(members))
    return out


def val_map50(weights):
    m = YOLO(weights)
    r = m.val(data=DATA_YAML, split="val", imgsz=IMG_SIZE, verbose=False)
    return float(r.box.map50)


def main():
    uniform = "--uniform" in sys.argv
    print(f"Loading {len(RUNS)} checkpoints...")
    ckpts = {r: load_sd(r) for r in RUNS}
    base_ckpt = ckpts[RUNS[0]][0]

    if uniform:
        sd = avg_sds([ckpts[r][1] for r in RUNS])
        out = save_soup(base_ckpt, sd, RUNS)
        print(f"val mAP50 (soup): {val_map50(out):.4f}")
        return

    # Greedy soup: start from best run, add members only if val mAP50 improves
    members = [RUNS[0]]
    best_sd = ckpts[RUNS[0]][1]
    out = save_soup(base_ckpt, best_sd, members)
    best_score = val_map50(out)
    print(f"  start: {RUNS[0]}  val mAP50={best_score:.4f}")

    for r in RUNS[1:]:
        trial_sd = avg_sds([ckpts[m][1] for m in members + [r]])
        out = save_soup(base_ckpt, trial_sd, members + [r])
        score = val_map50(out)
        keep = score >= best_score
        print(f"  +{r}: val mAP50={score:.4f}  {'KEEP' if keep else 'drop'}")
        if keep:
            members.append(r)
            best_score = score
        else:
            save_soup(base_ckpt, avg_sds([ckpts[m][1] for m in members]), members)

    print(f"\nFinal greedy soup: {members}  val mAP50={best_score:.4f}")
    print("Now run soup_best.pt through your test_full_dataset evaluation.")


if __name__ == "__main__":
    main()
