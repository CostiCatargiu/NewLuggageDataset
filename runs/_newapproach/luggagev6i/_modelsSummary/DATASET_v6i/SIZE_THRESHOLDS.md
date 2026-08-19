# Object-size thresholds — every definition used in this project, and the results under each

Four different size taxonomies appear across this project. They are **not
interchangeable**, and two of them are in active use at the same time: the dataset
analysis and the diagnostics use max-side bins, while every `mAP*_small/medium/large`
in the results JSONs uses COCO **area** bins. This document lists all of them, where
each is used, and the results computed under each.

---

## 1. The four taxonomies

| id | definition | measure | small | medium | large | used by |
|---|---|---|---|---|---|---|
| **A** | 48 / 96 px | **max bbox side** @640 | <48 | 48–96 | >96 | dataset report §3A/§4A, anchor-footprint diagnostic, `diag_miss_vs_score.py`, `diag_threshold_sweep.py` |
| **B** | 32 / 64 px | **max bbox side** @640 | <32 | 32–64 | >64 | dataset report §3B/§4B only — comparability reference, never used for results |
| **C** | 32² / 96² px² = 1024 / 9216 | **bbox area** (COCO) | <1024 | 1024–9216 | >9216 | **every `__test_full_dataset.json`** — i.e. all 119 runs' per-size mAP/AR/P/R |
| **D** | 60 / 120 px | max bbox side | <60 | 60–120 | >120 | **v5i only, superseded.** Do not use for v6i. |

**Taxonomy D — the one that was wrong.** The v5i script scaled the report's 48/96
thresholds by 1.25 because v5i labels were 512 px while training ran at 640. v6i labels
are *already* 640-wide, so the factor is 1.00. Running 60/120 on v6i pushed the entire
48–60 px band into "small" and produced **small = 75.1% of val GTs where the correct
figure is 62.2%**. Sections 4, 5 and 6 of the first anchor-footprint pass were computed
on that inflated bucket and were discarded. `SIZE_BINS = (48.0, 96.0)` in
`scripts/diag_anchor_footprint.py` is the corrected value.

### The trap

The four taxonomies give **different bucket membership for the same box**. Worked
examples, computed:

| box (W×H) | max side | area px² | **A** 48/96 | **B** 32/64 | **C** area | **D** 60/120 |
|---|---|---|---|---|---|---|
| 39×55 *(dataset mean)* | 55 | 2145 | medium | medium | medium | **small** |
| **25×100** *(tall, typical trolley)* | 100 | 2500 | **large** | large | **medium** | medium |
| 20×34 | 34 | 680 | **small** | **medium** | small | small |
| 32×32 | 32 | 1024 | **small** | **medium** | **medium** | small |
| 30×48 | 48 | 1440 | medium | medium | medium | **small** |
| 8×12 | 12 | 96 | small | small | small | small |
| 120×200 | 200 | 24 000 | large | large | large | large |

The divergence bites hardest on **tall thin boxes**, which is what this dataset is made
of: a 25×100 trolley is **large under A** (max side 100 > 96) and **medium under C**
(area 2500 < 9216). With 70.6% of boxes taller than wide and mean h/w 1.55, that is not
a corner case. Row 1 also shows why D was wrong: it calls the *average* box in the
dataset "small".

`diag_miss_vs_score.py` prints the warning in its own header:

> `size edges (px, MAX SIDE): small<48  medium<= 96  large>`
> `NOT the COCO area buckets used in the results JSONs — do not cross-compare.`

**Rule: never put a number from a results JSON in the same table as a number from the
dataset report or a diagnostic without saying which taxonomy each came from.**

---

## 2. Dataset distribution under each taxonomy

### Taxonomy A — max side 48 / 96 *(the campaign's analysis standard)*

| split | small | % | medium | % | large | % |
|---|---|---|---|---|---|---|
| train | 25 076 | **60.0%** | 12 006 | 28.7% | 4 741 | 11.3% |
| valid | 6 103 | **62.2%** | 2 958 | 30.1% | 758 | 7.7% |
| test | 3 717 | **60.2%** | 1 852 | 30.0% | 603 | 9.8% |

Per class, train:

| class | small | medium | large | small % | large % |
|---|---|---|---|---|---|
| backpack | 6 531 | 3 331 | 1 476 | 57.6% | 13.0% |
| bag | 5 664 | 2 070 | 1 534 | **61.1%** | **16.6%** |
| trolley | 12 881 | 6 605 | 1 731 | 60.7% | 8.2% |

### Taxonomy B — max side 32 / 64 *(COCO-like edges, on max side)*

| split | small | % | medium | % | large | % |
|---|---|---|---|---|---|---|
| train | 14 959 | **35.8%** | 16 398 | 39.2% | 10 466 | 25.0% |
| valid | 3 669 | **37.4%** | 4 034 | 41.1% | 2 116 | 21.6% |
| test | 2 175 | **35.2%** | 2 510 | 40.7% | 1 487 | 24.1% |

Per class, train:

| class | small | medium | large | small % | large % |
|---|---|---|---|---|---|
| backpack | 3 699 | 4 600 | 3 039 | 32.6% | 26.8% |
| bag | 3 499 | 3 267 | 2 502 | **37.8%** | 27.0% |
| trolley | 7 761 | 8 531 | 4 925 | 36.6% | 23.2% |

**A vs B on the same data:** "small" moves from 60.0% to 35.8% of train purely by
changing the edge from 48 px to 32 px. Any claim of the form "this dataset is N% small
objects" is meaningless without the threshold attached.

### Taxonomy C — COCO area 1024 / 9216 px²

**Not yet computed on the labels.** Every per-size *result* in this project is under C,
but the *dataset distribution* under C was never scanned. `scripts/analyze_dataset_v6i.py`
now emits it (`SIZE_C`) — run it against the dataset to fill this table in.

Expected placement, from the geometry: with mean box 39×55 = 2145 px² and 70.6% of boxes
tall, C's small bucket (<1024 px², e.g. anything under ~26×39) should land **between B
and A** — plausibly 40–50% of instances. Do not quote a number until it is measured.

### Taxonomy D — max side 60 / 120 *(v5i, superseded)*

Produced small = **75.1%** of val GTs against the correct 62.2%. Recorded here only so
the discrepancy is traceable if an old figure resurfaces.

---

## 3. Results under taxonomy C — all 119 runs

Full table: **`tables/all_runs_per_size_COCOarea.csv`** — 119 rows × 24 columns
(mAP50 and mAP50-95 per bucket, plus AR50, AR50-95, P50, R50 per bucket), sorted by
model then by mAP50-95.

Every number in `MODEL_v26/RESULTS_TABLES_v26.txt`, `MODEL_v12/RESULTS_TABLES_v12.txt`
and both `SUMMARY_*.md` files is from this taxonomy.

### YOLO26 — reference points (mAP50 per bucket unless marked 95)

| run | mAP50-95 | mAP50 | S | M | L | S95 | M95 | L95 |
|---|---|---|---|---|---|---|---|---|
| `y26_base_rep` *(baseline)* | 55.24 | 80.18 | 77.30 | 86.45 | 81.75 | 51.00 | 65.98 | 60.87 |
| `y26_identity` *(port control)* | 55.24 | 80.18 | 77.30 | 86.45 | 81.75 | 51.00 | 65.98 | 60.87 |
| `y26_stock_b32` *(batch control)* | 55.76 | 80.96 | 77.68 | 87.46 | 82.80 | 50.96 | 67.31 | 59.28 |
| `y26_scb_b3` | 55.66 | 80.75 | 78.02 | 86.80 | 81.11 | 51.24 | 66.85 | 59.43 |
| `y26_scb2_sbb50` | **55.70** | 80.95 | 78.25 | 87.09 | 77.43 | 51.33 | 66.72 | 58.55 |
| `y26_scb3_sbb50` | 55.65 | 80.86 | 77.92 | 87.25 | **83.36** | 51.30 | 66.69 | 60.82 |
| `y26_nwd50` | 54.78 | 80.62 | 78.19 | 87.12 | 78.13 | 50.85 | 66.52 | 55.00 |
| `y26_nwd50_scb3_sbb` | 54.56 | 80.73 | **78.26** | 87.12 | 78.70 | 50.50 | 66.64 | 57.02 |
| `y26_eiou` | 55.15 | 79.31 | 76.63 | 86.94 | **73.95** | 51.24 | 66.70 | 53.99 |
| `y26_p2k2_hi` *(arch, b48)* | **56.46** | 81.81 | 78.77 | 88.38 | 80.92 | 52.37 | 67.03 | 57.53 |

Bucket champions across all 73 v26 runs:

| bucket | best run | value |
|---|---|---|
| mAP50 small | `y26_dys_swa0603` | **79.49** |
| mAP50 medium | `y26_p2k2_hi` | **88.38** |
| mAP50 large | `y26_swa_a06_b15` | **85.55** |
| mAP50-95 small | `y26_s10_p45` | **52.40** |
| mAP50-95 large | `yolo26_custom-9` *(= baseline)* | **60.87** |

**No run beat the baseline on mAP50-95 large.** 73 attempts, and the stock configuration
is still the best large-object localiser.

### YOLOv12 — reference points

| run | mAP50-95 | mAP50 | S | M | L | S95 | M95 | L95 |
|---|---|---|---|---|---|---|---|---|
| `yolov12s_default` *(baseline)* | 54.77 | 79.75 | 76.65 | 86.59 | 81.87 | 49.98 | 65.07 | 57.73 |
| `yolov12s_sqrt0703` | 55.64 | 80.70 | 77.37 | 87.01 | 82.12 | 50.63 | 65.61 | 59.99 |
| `lb_uniform` | 55.57 | 80.46 | 77.39 | 86.36 | 81.04 | 50.85 | 65.32 | 57.75 |
| `arch_ls_shift` | 55.98 | 81.70 | 78.34 | 87.31 | **84.32** | 51.22 | 65.08 | 60.09 |
| `ls_shift_gctxP3` | **56.02** | 81.35 | 78.37 | 86.59 | 80.78 | **51.58** | 64.92 | 56.14 |
| `nwd_small` | 54.38 | 79.84 | 76.47 | 86.38 | **85.41** | 49.37 | 64.57 | 60.11 |
| `snt` | 54.79 | 80.60 | 77.18 | 86.42 | 81.87 | 49.78 | 65.46 | 59.10 |
| `qfl` | **47.25** | 69.71 | 65.67 | 80.46 | 63.49 | 42.75 | 59.51 | 46.49 |

Bucket champions across all 46 v12 runs:

| bucket | best run | value |
|---|---|---|
| mAP50 small | `ls_shift_sqrt` | **78.73** |
| mAP50 medium | `clsw_sqrt` | **87.41** |
| mAP50 large | `nwd_small` | **85.41** |
| mAP50-95 small | `ls_shift_gctxP3` | **51.58** |
| mAP50-95 large | `ms_s_sqrt_a0703_b15` | **61.27** |

Note `qfl`: AR50_small **97.23** — the *highest* small-object recall of any run in the
project — with R50_small at 58.33 and mAP50-95 at 47.25. Recall up, precision and AP
down: the same ranking-collapse signature as SNT on YOLO26.

---

## 4. Results under taxonomy A — the diagnostics

### `diag_miss_vs_score.py` — test split, 3 runs, max-side 48/96

GT counts per bucket match the dataset report exactly (3717 / 1852 / 603), confirming
the taxonomy.

`y26_identity` (= baseline), all classes:

| bucket | GT | hit@0.25 | recovered *(scoring loss)* | true_miss *(proposal loss)* |
|---|---|---|---|---|
| small | 3717 | 76.3% | **20.1%** | 3.6% |
| medium | 1852 | 84.9% | 11.7% | 3.3% |
| large | 603 | 80.8% | 15.6% | 3.6% |

Per class and size, same run:

| class | size | GT | hit@0.25 | recovered | true_miss |
|---|---|---|---|---|---|
| backpack | small | 957 | 74.6 | 21.2 | 4.2 |
| backpack | medium | 501 | 82.4 | 12.8 | 4.8 |
| backpack | large | 190 | 83.2 | 14.7 | 2.1 |
| bag | small | 833 | 66.0 | **29.5** | 4.4 |
| bag | medium | 331 | 73.4 | 20.8 | 5.7 |
| bag | large | 166 | 69.3 | 25.3 | 5.4 |
| trolley | small | 1927 | 81.6 | 15.4 | 3.0 |
| trolley | medium | 1020 | 89.9 | 8.2 | 1.9 |
| trolley | large | 247 | 86.6 | 9.7 | 3.6 |

Raw files for all three runs (`y26_identity`, `y26_scb3_sbb50`, `y26_p2k2_hi`) in
`DIAGNOSTICS/miss_Scoreresults/`.

### Anchor-footprint diagnostic — val split, max-side 48/96

GT counts 6102 / 2958 / 758 — again matching the dataset report's valid row exactly.
Candidate pools, selection cross-tab and positive-set quality under this taxonomy are in
`tables/anchor_candidate_pool.csv`, `tables/assigner_crosstab.csv` and
`tables/positive_set_quality.csv`; the narrative is `raw/diag_anchor_footprin_results.txt`
findings F1–F12.

---

## 5. Known gaps

- **Taxonomy C dataset distribution is unmeasured.** All results are reported under it;
  the label distribution under it has never been scanned. `analyze_dataset_v6i.py` now
  computes it — one run against the dataset closes this.
- **`y26_snt_t25` / `y26_snt_t50` have no `__test_full_dataset.json` in the repo.**
  The runs exist (confusion matrices are in `DIAGNOSTICS/confusion_collected/`) but their
  per-size metrics were only ever read off the console. The numbers quoted in
  `SUMMARY_v26.md` (−3.93 / −12.00 overall, large −10.58 / −16.60) come from that
  console output, not from a file here. Re-export if they go into the paper.
- **Taxonomy B has no results attached** — it exists only as a dataset-distribution
  reference. No model was ever evaluated with 32/64 max-side bins.

---

## 6. What to write in the paper

State the threshold explicitly next to every size-conditioned number, and pick one for
the headline. The defensible choice is:

- **Report model results under C (COCO area 32²/96²)** — it is what all 119 runs already
  use, and it is the convention reviewers expect.
- **Report the dataset characterisation under A (max side 48/96)**, and say so — max side
  is the right measure when the argument is about anchor stride coverage, because it is
  the side length, not the area, that determines how many stride-8 cells a box spans.
- **Never mix them in one table.** Where both are needed, label the columns.
