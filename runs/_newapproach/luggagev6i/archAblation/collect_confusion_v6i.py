#!/usr/bin/env python3
r"""
Walk a tree of run folders; give each run its OWN folder with everything it
produced, plus one combined report. Edit the CONFIG block below and run it.

    python collect_confusion_v6i.py


WHY
===
Per-class metrics on the best config say this:

    class      inst      P       R    AP50   AR50
    trolley    3194   85.4    80.7    88.3   96.6
    backpack   1648   80.7    73.3    82.6   96.4
    bag        1330   76.3    61.7    71.7   95.6

Bag's AR50 is 95.6 while its recall at the operating point is 61.7. The detector
FINDS ~96% of bags and KEEPS ~62%. That is not a detection failure — and every
mechanism tried on this project (SCB, SBB, SNL1, NWD, TSH, SNT, SWA, LB-TAL,
cls_pw) touches assignment, localisation, or box regression. None of them decides
which CLASS LABEL a well-localised box receives.

The confusion matrix separates two causes that need opposite fixes:

    bag -> background    box exists, scores too low       confidence / calibration
    bag -> backpack      box exists, wrong label          classification

Two hints it may be the second: bag loses precision AND recall together
(-9.0 / -19.0 vs trolley), and cls_pw made bag WORSE (48.2 -> 47.4), which argues
the problem is not sample count.


MATRIX INDEXING (verified against ultralytics/utils/metrics.py)
===============================================================
    matrix[predicted, true]              index nc == background

    matrix[nc, gc]        FN  — ground truth of class gc predicted as background
    matrix[dc, nc]        FP  — class dc predicted where no ground truth exists
    matrix[dc, gc], d!=g      — misclassification: gc's box labelled dc

The leak table normalises each TRUE-class COLUMN: of all ground truths of class X,
what fraction was labelled correctly, mislabelled as each other class, or missed.


OUTPUT
======
    <OUT_DIR>/
      <run_name>/
        confusion_matrix.png              copied if present
        confusion_matrix_normalized.png   copied if present
        results.csv                       copied if present
        confusion.csv                     raw counts, labelled  (RUN_MATRIX only)
        REPORT.txt                        this run's leak table + final metrics
      ALL_RUNS_REPORT.txt                 every run side by side
      leak_table.csv                      one row per run x true class
      INDEX.txt                           what was found where

RUN_MATRIX = True re-runs validation to pull validator.confusion_matrix.matrix out
as an array — the PNG is a render, so the numbers have to be regenerated. Inference
only, ~10s per model, and it writes to <OUT_DIR>/_val_<run>/ so nothing in your runs
tree is touched.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys

# ============================== CONFIG ======================================= #
# The folder that CONTAINS your run folders. Walked recursively, so pointing at
# the parent of several project dirs is fine.
RUNS_ROOT = "/home/constantin/Doctorat/YoloLib/runs/detect"

OUT_DIR = "confusion_collected"

# False -> only copy what already exists on disk (no GPU, no torch needed).
# True  -> also re-run val and write the confusion matrix AS NUMBERS. This is the
#          mode that answers the bag question; the PNGs only let you eyeball it.
RUN_MATRIX = True

# Substring filter on run names. Empty list = every run found.
# e.g. ONLY = ["y26_base_rep", "y26_scb3_sbb50", "y26_nwd50"]
ONLY: list[str] = []

# Only used when RUN_MATRIX is True.
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
SPLIT = "test"
IMG_SIZE = 640
BATCH = 32
# ============================================================================= #

# results.csv column -> short label. Ultralytics has renamed these across versions,
# so match on a substring rather than an exact key.
FINAL_COLS = [
    ("metrics/precision", "P"),
    ("metrics/recall", "R"),
    ("metrics/mAP50(", "mAP50"),
    ("metrics/mAP50-95", "mAP50-95"),
]


def find_runs(root: str) -> dict[str, str]:
    """Map run name -> directory for anything that looks like a run."""
    out: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if (
            "results.csv" in filenames
            or any(f.startswith("confusion_matrix") for f in filenames)
            or os.path.isfile(os.path.join(dirpath, "weights", "best.pt"))
        ):
            name = os.path.basename(dirpath.rstrip("/\\"))
            if name.startswith("_val_"):
                continue  # our own output from a previous RUN_MATRIX pass
            if name in out:  # same leaf name under two projects: disambiguate
                name = f"{os.path.basename(os.path.dirname(dirpath))}__{name}"
            out[name] = dirpath
    return dict(sorted(out.items()))


def final_metrics(results_csv: str) -> dict:
    """Last-epoch row of results.csv, reduced to the four numbers that matter."""
    try:
        with open(results_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last, out = rows[-1], {"epochs": len(rows)}
        for needle, label in FINAL_COLS:
            for k, v in last.items():
                if k and needle in k:
                    try:
                        out[label] = round(float(v) * 100, 2)
                    except (TypeError, ValueError):
                        pass
                    break
        return out
    except Exception:
        return {}


def leak_rows(name: str, matrix, names: dict) -> list[dict]:
    """Column-normalise: for each TRUE class, where did its ground truths go?"""
    nc = len(names)
    rows = []
    for gc in range(nc):
        col = matrix[:, gc]
        total = float(col.sum())
        if total <= 0:
            continue
        r = {"run": name, "true_class": names[gc], "gt_total": int(total),
             "correct": round(100.0 * col[gc] / total, 2)}
        for dc in range(nc):
            if dc != gc:
                r[f"as_{names[dc]}"] = round(100.0 * col[dc] / total, 2)
        r["as_background"] = round(100.0 * col[nc] / total, 2)
        r["misclassified_total"] = round(
            sum(100.0 * col[dc] / total for dc in range(nc) if dc != gc), 2
        )
        rows.append(r)
    return rows


def leak_block(rows: list[dict], width: int = 26) -> list[str]:
    if not rows:
        return ["  (no confusion matrix — set RUN_MATRIX = True)"]
    other = sorted({k for r in rows for k in r if k.startswith("as_") and k != "as_background"})
    hdr = (f"{'run':<{width}}{'true':<10}{'n':>6}{'correct':>9}"
           + "".join(f"{k:>14}" for k in other) + f"{'missed':>9}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        run = r["run"] if len(r["run"]) <= width - 1 else r["run"][: width - 4] + "..."
        out.append(
            f"{run:<{width}}{r['true_class']:<10}{r['gt_total']:>6}{r['correct']:>9.1f}"
            + "".join(f"{r.get(k, 0.0):>14.1f}" for k in other)
            + f"{r['as_background']:>9.1f}"
        )
    return out


HOW_TO_READ = [
    "",
    "HOW TO READ IT",
    "  missed high, misclassified low   -> a DETECTION / CONFIDENCE problem. The",
    "     campaign was aimed correctly and you are near the ceiling.",
    "  misclassified high               -> a CLASSIFICATION problem. Every mechanism",
    "     tried so far (assignment, localisation, box regression) is orthogonal to",
    "     it, which would explain twelve flat results.",
    "",
    "  Compare the SAME class ACROSS runs. If the misclassified column is identical",
    "  for baseline / best-config / NWD, no loss mechanism ever moved the real",
    "  failure — a cleaner statement than any of the mAP deltas.",
]


def write_run_report(path, name, src, fm, rows, files) -> None:
    L = [f"RUN: {name}", f"source: {src}", ""]
    if fm:
        L += ["FINAL EPOCH (from results.csv)",
              f"  epochs   {fm.get('epochs', '?')}",
              *[f"  {k:<9}{fm[k]:.2f}" for k in ("P", "R", "mAP50", "mAP50-95") if k in fm], ""]
    else:
        L += ["FINAL EPOCH: results.csv not found or unparseable", ""]
    L += ["CONFUSION LEAK — where did each TRUE class end up? (% of its ground truths)",
          "matrix[predicted, true]; 'missed' = predicted as background", ""]
    L += leak_block(rows, width=len(name) + 2 if rows else 26)
    L += HOW_TO_READ + ["", "files collected:"] + [f"  {f}" for f in files]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def collect_one(name, src, out):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    got = []
    for fn in ("confusion_matrix.png", "confusion_matrix_normalized.png", "results.csv"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(d, fn))
            got.append(fn)
    fm = final_metrics(os.path.join(d, "results.csv")) if "results.csv" in got else {}
    return fm, got


def build_matrix(name, src, out):
    """Re-run val, return (matrix, names) or (None, None). Inference only."""
    import numpy as np
    from ultralytics import YOLO

    w = os.path.join(src, "weights", "best.pt")
    if not os.path.isfile(w):
        return None, None
    m = YOLO(w)
    v = m.val(data=DATA_YAML, split=SPLIT, imgsz=IMG_SIZE, batch=BATCH, plots=True,
              verbose=False, project=out, name=f"_val_{name}", exist_ok=True)
    cm = getattr(getattr(m, "validator", None), "confusion_matrix", None)
    if cm is None:
        return None, None
    names = v.names if isinstance(v.names, dict) else dict(enumerate(v.names))
    return np.asarray(cm.matrix), {int(k): str(x) for k, x in names.items()}


if __name__ == "__main__":
    root = os.path.expanduser(RUNS_ROOT)
    if not os.path.isdir(root):
        sys.exit(f"[ABORT] RUNS_ROOT is not a directory: {root}\n"
                 f"        Edit the CONFIG block at the top of this file.")
    runs = find_runs(root)
    if ONLY:
        runs = {k: v for k, v in runs.items() if any(o in k for o in ONLY)}
    if not runs:
        sys.exit(f"[ABORT] no run directories under {root}"
                 + (f" matching {ONLY}" if ONLY else ""))
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n  RUNS_ROOT   {root}")
    print(f"  OUT_DIR     {OUT_DIR}")
    print(f"  RUN_MATRIX  {RUN_MATRIX}" + ("" if RUN_MATRIX else "   <- no numbers, only file copying"))
    print(f"  ONLY        {ONLY or 'all'}")
    print(f"\n  {len(runs)} runs found\n")
    print(f"  {'run':<34}{'files':>7}{'mAP50-95':>10}{'matrix':>8}")
    print("  " + "-" * 59)

    all_rows, index = [], []
    for name, src in runs.items():
        fm, got = collect_one(name, src, OUT_DIR)
        rows = []
        if RUN_MATRIX:
            try:
                matrix, names = build_matrix(name, src, OUT_DIR)
                if matrix is not None:
                    labels = [names[i] for i in range(len(names))] + ["background"]
                    with open(os.path.join(OUT_DIR, name, "confusion.csv"), "w", newline="") as f:
                        wr = csv.writer(f)
                        wr.writerow([r"pred\true"] + labels)
                        for i, lab in enumerate(labels):
                            wr.writerow([lab] + [int(x) for x in matrix[i]])
                    got.append("confusion.csv")
                    rows = leak_rows(name, matrix, names)
                    all_rows += rows
            except Exception as e:
                print(f"    [skip matrix] {name}: {e}")
        write_run_report(os.path.join(OUT_DIR, name, "REPORT.txt"), name, src, fm, rows, got)
        got.append("REPORT.txt")
        mp = f"{fm['mAP50-95']:.2f}" if "mAP50-95" in fm else "-"
        print(f"  {name:<34}{len(got):>7}{mp:>10}{('yes' if rows else '-'):>8}")
        index.append(f"{name}\n    src    {src}\n    files  {', '.join(got)}")

    with open(os.path.join(OUT_DIR, "INDEX.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    L = ["ALL RUNS — confusion leak",
         "matrix[predicted, true]; 'missed' = predicted as background", ""]
    L += leak_block(all_rows) + HOW_TO_READ
    with open(os.path.join(OUT_DIR, "ALL_RUNS_REPORT.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    if all_rows:
        keys = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(os.path.join(OUT_DIR, "leak_table.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(all_rows)
        print("\n" + "\n".join(L))
    else:
        print("\n  no matrices built — set RUN_MATRIX = True to get the leak table")

    print(f"\n  one folder per run under {OUT_DIR}/  +  ALL_RUNS_REPORT.txt, INDEX.txt")
