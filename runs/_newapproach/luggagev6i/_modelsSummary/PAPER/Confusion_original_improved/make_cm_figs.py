#!/usr/bin/env python3
"""
make_cm_figs.py — redraw the size-resolved confusion matrices in the paper's style.

Reads cm_values.json (row-normalised, rows = ground truth, cols = prediction)
and writes one PNG + PDF per variant into this folder.

Style decisions, all deliberate:
  * palette matches the paper's figures (teal for correct, coral for missed)
  * the DIAGONAL is bold and boxed        -> the correct-class rate
  * the BACKGROUND column is bold+coral   -> the miss rate, the failure mode
    that matters for an alarm system
  * off-diagonal inter-class confusions stay light grey so they recede
  * a real title per panel, not a placeholder

    python make_cm_figs.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["backpack", "bag", "trolley", "background"]
SIZES = ["All", "Small", "Medium", "Large"]

# paper palette
TEAL, TEAL_BG = "#3D9A8B", "#E4F2EF"
COR,  COR_BG  = "#C85439", "#FAE7E1"
GRY,  GRY_BG  = "#6E737A", "#F0F1F3"
cmap_ok   = LinearSegmentedColormap.from_list("ok",   ["#FFFFFF", TEAL_BG, TEAL])
cmap_miss = LinearSegmentedColormap.from_list("miss", ["#FFFFFF", COR_BG,  COR])
cmap_off  = LinearSegmentedColormap.from_list("off",  ["#FFFFFF", GRY_BG,  GRY])

TITLES = {
 "v26_default": ("YOLO26s", "baseline (stock loss, stock 3-level head)"),
 "v26_custom":  ("YOLO26s", r"recommended: p2dys-p234rich + $\beta$=1"),
 "v12_default": ("YOLOv12s", "baseline (stock loss, stock 3-level head)"),
 "v12_custom":  ("YOLOv12s", "recommended: ls_shift (4-level head)"),
}


def draw(ax, M, title, show_ylab):
    n = len(CLASSES)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    for i in range(n):
        for j in range(n):
            v = M[i][j]
            diag   = (i == j and i < n - 1)          # correct class
            missed = (j == n - 1 and i < n - 1)      # GT predicted as background
            ghost  = (i == n - 1 or (i == j == n - 1))
            cm = cmap_ok if diag else (cmap_miss if missed else cmap_off)
            shade = 0.0 if ghost and i == n - 1 else v
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=cm(min(shade * 0.78, 0.82)),
                                   edgecolor="white", linewidth=1.6))
            if diag:
                ax.add_patch(Rectangle((j - .46, i - .46), .92, .92, fill=False,
                                       edgecolor=TEAL, linewidth=1.8, zorder=3))
            # all cell text is BLACK for legibility; emphasis is carried by
            # weight and size, and the semantics by the cell fill colour.
            if diag or missed:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=13.5, fontweight="bold", color="black", zorder=4)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10.5, color="black", zorder=4)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, fontsize=9.5, rotation=28, ha="right", color="black")
    ax.set_yticklabels(CLASSES if show_ylab else [""] * n, fontsize=9.5, color="black")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=7, color="black")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)


def main():
    data = json.load(open(os.path.join(HERE, "cm_values.json")))
    for key, (model, variant) in TITLES.items():
        fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.15))
        for k, sz in enumerate(SIZES):
            lab = sz if sz == "All" else {
                "Small": "Small  (area $<32^2$)",
                "Medium": "Medium  ($32^2$–$96^2$)",
                "Large": "Large  ($>96^2$)"}[sz]
            draw(axes[k], data[key][sz], lab, show_ylab=(k == 0))
        fig.suptitle(f"{model} — {variant}", fontsize=14.5, fontweight="bold",
                     y=1.005, color="black")
        fig.text(0.5, -0.055,
                 "rows = ground truth, columns = prediction · row-normalised · IoU $\\geq$ 0.5 · test split\n"
                 "bold teal = correct class    bold coral = missed (predicted background)    grey = inter-class confusion",
                 ha="center", fontsize=9.5, color="black")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(HERE, f"{key}_cm.{ext}"), dpi=200,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {key}_cm.png / .pdf")


if __name__ == "__main__":
    main()
