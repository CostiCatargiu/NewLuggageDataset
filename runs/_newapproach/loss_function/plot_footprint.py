#!/usr/bin/env python3
"""
Figure generator for diag_anchor_footprint.py.

Reads footprint_stats.json (already written by the diagnostic) and produces a
three-panel publication figure. No model, no dataset, no GPU — it only needs
the JSON, so it runs in seconds on a laptop.

PANELS
------
A  Candidate supply vs selected positives, per pyramid level.
   The gap between the two bars IS the selection bias. This is the panel that
   shows TAL over-selecting P4 and abandoning P5.

B  Can a level even supply topk? Stacked composition of GTs per level:
   zero candidates / below topk / at-or-above topk. This is the panel that
   shows P5 is geometrically unreachable rather than unfairly ranked.

C  Per-class selection bias by level, against the neutral line at 1.0.

USAGE
-----
    python plot_footprint.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURATION
# =============================================================================
STATS_JSON = "diag_fp_out/footprint_stats.json"
OUT_DIR = "diag_fp_out"
BASENAME = "footprint_figure"      # writes .png and .pdf
DPI = 200
FIGSIZE = (14.5, 4.4)
# =============================================================================

C_CAND = "#8FA8C8"     # candidate supply — muted blue
C_SEL = "#C85A54"      # selected positives — muted red
C_OK = "#5B8C5A"       # pool >= topk
C_LOW = "#E0A458"      # 0 < pool < topk
C_ZERO = "#B3443E"     # pool == 0
NEUTRAL = "#444444"


def main():
    if not os.path.exists(STATS_JSON):
        sys.exit(f"not found: {STATS_JSON}\nRun diag_anchor_footprint.py first, "
                 f"or edit STATS_JSON at the top of this file.")
    with open(STATS_JSON) as f:
        S = json.load(f)

    per_level = S["per_level"]
    per_class = S.get("per_class", {})
    topk = S.get("topk", 10)
    strides = sorted(per_level.keys(), key=lambda k: int(k))
    labels = [f"P{i+3}\ns{s}" for i, s in enumerate(strides)]
    x = np.arange(len(strides))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=FIGSIZE)

    # ---------------------------------------------------------------- panel A
    cand = np.array([per_level[s]["cand_share"] * 100 for s in strides])
    sel = np.array([per_level[s]["sel_share"] * 100 for s in strides])
    bias = np.array([per_level[s]["selection_bias"] for s in strides])
    w = 0.38

    axA.bar(x - w / 2, cand, w, label="candidate supply", color=C_CAND, edgecolor="white")
    axA.bar(x + w / 2, sel, w, label="selected positives", color=C_SEL, edgecolor="white")

    for xi, (c, s_, b) in enumerate(zip(cand, sel, bias)):
        axA.text(xi - w / 2, c + 1.6, f"{c:.1f}%", ha="center", va="bottom", fontsize=8.5)
        axA.text(xi + w / 2, s_ + 1.6, f"{s_:.1f}%", ha="center", va="bottom", fontsize=8.5)
        col = C_SEL if b < 0.9 else (NEUTRAL if b <= 1.15 else C_OK)
        axA.text(xi, max(c, s_) + 8.5, f"bias {b:.2f}", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color=col)

    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("share of total (%)")
    axA.set_title("A — supply vs selection", fontsize=11, fontweight="bold", loc="left")
    axA.set_ylim(0, max(cand.max(), sel.max()) * 1.32)
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", alpha=0.25, linewidth=0.6)
    axA.set_axisbelow(True)

    # ---------------------------------------------------------------- panel B
    zero = np.array([per_level[s]["frac_gt_pool_zero"] * 100 for s in strides])
    lt = np.array([per_level[s]["frac_gt_pool_lt_topk"] * 100 for s in strides])
    low = lt - zero                       # 0 < pool < topk
    ok = 100.0 - lt                       # pool >= topk

    axB.bar(x, ok, 0.55, label=f"pool $\\geq$ topk ({topk})", color=C_OK, edgecolor="white")
    axB.bar(x, low, 0.55, bottom=ok, label=f"0 < pool < topk", color=C_LOW, edgecolor="white")
    axB.bar(x, zero, 0.55, bottom=ok + low, label="pool = 0 (unreachable)",
            color=C_ZERO, edgecolor="white")

    for xi, s in enumerate(strides):
        if zero[xi] > 3:
            axB.text(xi, 100 - zero[xi] / 2, f"{zero[xi]:.1f}%", ha="center",
                     va="center", fontsize=9, color="white", fontweight="bold")
        if ok[xi] > 6:
            axB.text(xi, ok[xi] / 2, f"{ok[xi]:.1f}%", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")

    # median pool size folded into the tick label so it cannot collide with it
    labels_b = [f"{lb}\nmedian {per_level[s]['pool_p50']:.0f}"
                for lb, s in zip(labels, strides)]
    axB.set_xticks(x); axB.set_xticklabels(labels_b)
    axB.set_ylabel("share of ground-truth boxes (%)")
    axB.set_title("B — can the level supply topk?", fontsize=11, fontweight="bold", loc="left")
    axB.set_ylim(0, 100); axB.set_xlim(-0.6, len(strides) - 0.4)
    # legend below the panel: every in-bar region is occupied at some level, so
    # any in-axes placement clips a label
    axB.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=3, frameon=False, handlelength=1.3, columnspacing=1.2)
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="y", alpha=0.25, linewidth=0.6)
    axB.set_axisbelow(True)

    # ---------------------------------------------------------------- panel C
    if per_class:
        cls = list(per_class.keys())
        wc = 0.8 / max(len(cls), 1)
        palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        for i, c in enumerate(cls):
            vals = [per_class[c].get(s, {}).get("selection_bias", np.nan) for s in strides]
            off = (i - (len(cls) - 1) / 2) * wc
            axC.bar(x + off, vals, wc * 0.9, label=str(c),
                    color=palette[i % len(palette)], edgecolor="white")
        axC.axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=1.2, zorder=0)
        axC.text(len(strides) - 0.45, 1.06, "level-neutral", fontsize=8,
                 color=NEUTRAL, ha="right")
        axC.set_xticks(x); axC.set_xticklabels(labels)
        axC.set_ylabel("selection bias  (sel share / cand share)")
        axC.set_title("C — selection bias per class", fontsize=11,
                      fontweight="bold", loc="left")
        axC.legend(frameon=False, fontsize=9, ncol=1, loc="upper left")
        axC.spines[["top", "right"]].set_visible(False)
        axC.grid(axis="y", alpha=0.25, linewidth=0.6)
        axC.set_axisbelow(True)
    else:
        axC.axis("off")

    n_gt, n_fg = S.get("n_gt", "?"), S.get("n_fg", "?")
    fig.suptitle(
        f"Anchor-footprint decomposition — {S.get('assigner','?')} "
        f"(topk={topk}, {S.get('split','?')} split, {S.get('imgsz','?')}px, "
        f"{n_gt} GTs / {n_fg} foreground anchors)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, BASENAME + ".png")
    pdf = os.path.join(OUT_DIR, BASENAME + ".pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")      # vector, for the paper
    print(f"saved -> {png}")
    print(f"saved -> {pdf}")


if __name__ == "__main__":
    main()
