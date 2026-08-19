# DATASET_v6i — LuggageDataset.v6i.yolov12

Everything measured about the dataset itself, separate from the model folders
(`MODEL_v12/`, `MODEL_v26/`) and the model diagnostics (`DIAGNOSTICS/`).

**Start with `DATASET_ANALYSIS_v6i.md`.** It is the write-up: the numbers, what
they explain, and the property→mechanism→outcome map.

**Then read `SIZE_THRESHOLDS.md`** before quoting any per-size number. Four different
size taxonomies exist in this project and **two are in use simultaneously**: the dataset
analysis and all diagnostics bucket by **max bbox side** (48/96 px), while every
`mAP*_small/medium/large` in the results JSONs buckets by **COCO area** (1024/9216 px²).
Those are different partitions of the same boxes — a 25×100 px box is `large` by max side
and `medium` by area, and 70.6% of this dataset is taller than wide. That document lists
every definition, what uses it, and the complete results under each.

## Headline numbers

- 12 184 images / 57 814 instances / 3 classes — 9138 / 1827 / 1219 image split
- **60.0%** of instances are small (max side <48 px); mean box **39×55 px @ 640**
- **70.6%** of boxes are taller than wide (mean h/w 1.55)
- trolley 50.7% · backpack 27.1% · bag 22.2%
- 80.4% of train images are **640×360** — fixed-camera framing
- Data quality: zero background images, zero missing labels, zero malformed boxes

## The two findings that shaped the campaign

1. **A large object has 51× the candidate-anchor supply of a small one** (657.7 vs
   12.9 in-box anchors), yet `topk=10` applies to both. Stock TAL therefore already
   converts **59.9%** of a small object's entire pool into positives, against **1.49%**
   for large. Small objects are limited by geometry, not by the assigner — which is why
   every "give small objects more positives" mechanism returned nothing.

2. **AR50_small ≈ 0.95 vs R50_small ≈ 0.70.** The detections exist; they score too low.
   25 pp of headroom on the ranking axis against the ~0.9 pp the assignment axis was
   contesting — and not recoverable by thresholding at any usable precision.

## One caveat to carry into the paper

The **val split under-represents large objects** (7.7% vs train 11.3%, test 9.8%) and
its boxes are **23% smaller in area** than test's. Val is a fine tuning split — the
threshold sweep transferred to test intact — but do not report a large-object result
from val alone; it rests on 758 instances against a measured seed spread of 2.06 pp on
large mAP50-95.

## Contents

| path | what it is |
|---|---|
| `DATASET_ANALYSIS_v6i.md` | the write-up — 9 sections, paper-ready |
| `SIZE_THRESHOLDS.md` | all four size taxonomies, the dataset distribution under each, and all 119 runs' per-size results |
| `raw/LuggageDatasetSplitv6i.txt` | label-level scan: splits, classes, sizes, shapes, quality |
| `raw/diag_anchor_footprin_results.txt` | anchor geometry, 4 assigner passes on one checkpoint, findings F1–F12 |
| `scripts/analyze_dataset_v6i.py` | regenerates the scan and the CSVs, **all four taxonomies** — no arguments, edit the CONFIG block |
| `scripts/diag_anchor_footprint.py` | regenerates the anchor diagnostic |
| `tables/all_runs_per_size_COCOarea.csv` | 119 runs × 24 per-size metrics (mAP50, mAP50-95, AR50, AR50-95, P50, R50 per bucket) |
| `tables/size_distribution_all_thresholds.csv` | every taxonomy × every split |
| `tables/size_thresholds_crosswalk.csv` | definition, status and consumers of each threshold |
| `tables/*.csv` | machine-readable versions of every other table, for plotting |

## Regenerating

Both scripts are arg-free. Set `DATA_ROOT` at the top of
`scripts/analyze_dataset_v6i.py` (currently
`/home/constantin/Doctorat/LuggageDataset.v6i.yolov12`) and run:

```bash
cd DATASET_v6i/scripts
python analyze_dataset_v6i.py
```

It writes `DATASET_ANALYSIS_v6i_regenerated.txt` next to itself and refreshes
`../tables/split_summary.csv` and `../tables/class_size_shape.csv`.

The anchor diagnostic needs a checkpoint; it was run against
`runs_newl_luggagev6i/yolov12s_default/weights/best.pt` on the val split, 115 batches,
imgsz 640, holding weights fixed across all four assigner schemes so the comparison
isolates geometry rather than confounding it with each config's own predictions.

## Note on the CSVs

`tables/*.csv` were transcribed from the two raw reports so they are available without
the dataset mounted. `analyze_dataset_v6i.py` regenerates `split_summary.csv` and a
fuller `class_size_shape.csv` directly from the labels when the dataset is present —
the transcribed files are for offline plotting, the script is the source of truth.
