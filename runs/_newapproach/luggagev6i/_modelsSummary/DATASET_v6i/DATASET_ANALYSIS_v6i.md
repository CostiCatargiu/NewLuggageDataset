# LuggageDataset v6i — dataset characterisation and what it explains

Everything here is measured, not assumed. Raw outputs in `raw/`, machine-readable
tables in `tables/`, regeneration scripts in `scripts/`.

Sources: `raw/LuggageDatasetSplitv6i.txt` (label-level scan) and
`raw/diag_anchor_footprin_results.txt` (anchor-geometry diagnostic, four assigner
passes on one fixed checkpoint).

---

## 1. The dataset at a glance

| split | images | img % | instances | inst % | boxes/img | mean W | mean H | mean h/w |
|---|---|---|---|---|---|---|---|---|
| train | 9138 | 75.0% | 41 823 | 72.3% | 4.58 | 39 px | 55 px | 1.55 |
| valid | 1827 | 15.0% | 9 819 | 17.0% | 5.37 | 35 px | 49 px | 1.55 |
| test | 1219 | 10.0% | 6 172 | 10.7% | 5.06 | 40 px | 56 px | 1.54 |

12 184 images, 57 814 instances, 3 classes. Zero background images, zero missing
label files, zero malformed boxes across all three splits. **80.4% of train images
are 640×360** — a fixed-camera CCTV aspect, not a photographic mix.

Classes: **trolley 50.7% / backpack 27.1% / bag 22.2%** (train). Class proportions
hold to within 1.9 pp across splits.

---

## 2. This is a small-object dataset, and that is not a figure of speech

| taxonomy | small | medium | large |
|---|---|---|---|
| **A** — max side <48 / 48–96 / >96 px | **60.0%** | 28.7% | 11.3% |
| **B** — max side <32 / 32–64 / >64 px | 35.8% | 39.2% | 25.0% |

The mean box is **39×55 px in a 640-px frame** — 0.53% of the image area. Under the
COCO area convention (<32²=1024 px²) the mean box at 2145 px² is *medium*, but its
**max side of 55 px is only 6.9 stride-8 cells**, and that is the number the assigner
actually sees.

> ### ⚠ Two taxonomies are in use at once — see `SIZE_THRESHOLDS.md`
>
> The tables above (and everything else in this document) use **max side**. But every
> `mAP50_small / medium / large` in the results JSONs — i.e. every per-size number in
> `MODEL_v26/`, `MODEL_v12/` and both `SUMMARY_*.md` files — uses the **COCO area**
> convention (1024 / 9216 px²), which is a *different bucketing of the same boxes*.
> A 25×100 px box is `large` under max-side and `medium` under area, and with 70.6% of
> this dataset taller than wide that divergence is common. **Do not put a dataset number
> and a results number in the same table without labelling which taxonomy each uses.**
> `SIZE_THRESHOLDS.md` lists all four definitions, where each is used, and the full
> results under each.

**On sample size:** under A, "large" is only 11.3% of train and **7.7% of val** — so
every large-object number in this project rests on a small sample, and swings of 2 pp on
large mAP are within noise (measured seed spread on large mAP50-95: **2.06 pp**, vs
0.12 pp overall).

---

## 3. Boxes are tall, and the classes differ sharply in how tall

| split | wide (<0.8) | square (0.8–1.25) | tall (>1.25) | mean h/w |
|---|---|---|---|---|
| train | 6.0% | 23.4% | **70.6%** | 1.55 |
| valid | 5.8% | 22.8% | 71.4% | 1.55 |
| test | 5.7% | 23.0% | 71.3% | 1.54 |

Per class (train):

| class | n | mean h/w | wide | square | tall |
|---|---|---|---|---|---|
| backpack | 11 338 | 1.47 | 1.2% | 23.8% | **74.9%** |
| bag | 9 268 | **1.33** | **13.2%** | **36.0%** | 50.8% |
| trolley | 21 217 | **1.68** | 5.5% | 17.6% | **76.9%** |

**Bag is the shape outlier.** It holds **48.5% of every wide box in the dataset**
while being 22.2% of instances, and it is the only class whose three shape bins are
anywhere near balanced. Trolley is the opposite — 76.9% tall, visually canonical.

This 70.6%-tall figure is what motivated the **EIoU** experiment (an aspect-ratio-aware
penalty ought to help a tall-box dataset). It did not: EIoU scored **−0.09 overall and
−7.80 on large**. The dataset property was real; the inference from it was wrong.

---

## 4. Bag is bimodal in size as well as in shape

Train, taxonomy A, as a share of each class:

| class | small | medium | large |
|---|---|---|---|
| backpack | 57.6% | 29.4% | 13.0% |
| **bag** | **61.1%** | **22.3%** | **16.6%** |
| trolley | 60.7% | 31.1% | 8.2% |

Bag has both the **highest large share** (16.6%) and the **lowest medium share**
(22.3%) — the emptiest middle of the three. Combined with §3, bag is the class with
the widest intra-class variance on *both* axes.

**Why this matters.** Bag is the worst class by a wide margin (AP50-95 **0.4666** vs
trolley 0.6185, a 15.2 pp gap). Three independent lines rule out the obvious causes:

- **Not frequency.** `cls_pw` inverse-frequency reweighting made bag *worse* at both
  settings (47.37 and 47.50, from 48.2).
- **Not assignment geometry.** Stock selection bias by class is
  backpack 0.92/1.44/0.45, bag 0.96/1.19/0.85, trolley 0.94/1.49/0.09 at s8/s16/s32 —
  bag is the *most* evenly served of the three (finding F12).
- **Not proposal failure.** Bag's AR50_small is 0.9352 against R50_small 0.62 — a
  **31 pp** recall gap that is pure confidence ranking.

What is left is exactly what §3 and §4 describe: **"bag" is a semantically loose
category** — handbags, duffels, shopping bags, plastic bags — with genuinely
heterogeneous appearance, size and aspect ratio, and the model's confidence reflects
that ambiguity honestly. This is a *labelling-taxonomy* limitation, not an
optimisation one, and no loss or assignment mechanism in 119 runs moved it.

---

## 5. The val split is not an unbiased proxy for test

| quantity | train | valid | test | valid vs test |
|---|---|---|---|---|
| large share (taxonomy A) | 11.3% | **7.7%** | 9.8% | **−2.1 pp (−21% relative)** |
| large count | 4741 | **758** | 603 | — |
| mean W × mean H | 39×55 | **35×49** | 40×56 | **−23% in area** |
| boxes / image | 4.58 | 5.37 | 5.06 | +6% denser |

Valid objects are **~23% smaller in area than test objects** and the split carries
proportionally fewer large instances. Two consequences that were acted on:

1. **Confidence thresholds were tuned on val and reported on test** (`diag_threshold_sweep.py`,
   `SPLIT_TUNE="val"` / `SPLIT_REPORT="test"`) precisely to avoid leakage — but the
   size drift means a val-optimal threshold is tuned on a slightly harder, smaller-object
   distribution. The +1.9 micro-F1 gain survived the transfer, so the drift is tolerable
   at that magnitude.
2. **Any val-only claim about large objects rests on 758 instances.** Do not report a
   large-object result from val alone.

Otherwise the splits are clean: class proportions within 1.9 pp, shape distributions
within 0.8 pp, no leakage indicators, no missing classes.

---

## 6. The geometric fact that explains the whole campaign

From the anchor-footprint diagnostic — mean in-box candidate anchors per GT, and how
many stock TAL (`topk=10`) actually selects:

| size | s8 pool | s16 pool | s32 pool | total pool | selected | **selection rate** |
|---|---|---|---|---|---|---|
| small | 9.82 | 2.46 | 0.62 | **12.90** | 7.73 | **59.9%** |
| medium | 42.38 | 10.63 | 2.71 | 55.72 | 9.87 | 17.7% |
| large | 501.02 | 125.28 | 31.37 | **657.67** | 9.79 | **1.49%** |

A large object has **51× the candidate supply** of a small one, and `topk=10` is applied
to both. The consequence, stated plainly:

> **Stock TAL already converts 60% of a small object's entire candidate pool into
> positives, against 1.5% for a large object. At stride 8 specifically it takes 6.64 of
> small's 9.82 candidates (68%) and 0.16 of large's 501 (0.03%).**

Small objects are **not starved by the assigner**. They are starved by *geometry* —
36.2% of GTs have fewer than 10 stride-8 candidates in the first place, and 31.1% have
zero candidates at stride 32.

**This is why the entire "give small objects more positives" family failed.** There is
almost nothing left to give:

- `lb_prop` raises the P3 budget to 8 (above stock's effective 6.64) → **54.82, +0.05.** Nothing.
- The outcome curve in the P3 budget is **single-peaked at 4**, i.e. *fewer* positives
  than stock: 8→0.5482, ~6.6 (stock)→0.5477, 5→0.5542, **4→0.5557**, 2→0.5534.
- The one mechanism that clearly worked on v12 (`lb_uniform`, +0.79) works by **cutting**
  small's stride-8 positives from 6.64 to 3.64, not by adding any.

And the P5 budget is fiction for 92% of the dataset: small draws 0.01 positives from
s32 and medium 0.00.

---

## 7. Where the actual headroom is — and it is not in the dataset's geometry

Measured across all runs: **AR50_small ≈ 0.95 while R50_small ≈ 0.70.**

The detections for nearly all small objects **exist**. They score below the F1-optimal
threshold. Confirmed independently by `diag_miss_vs_score.py`: true proposal failure is
only **2.7–5.5%**, i.e. the model is ~95% capable and read at ~76%.

That is **25 pp of headroom** against the ~0.9 pp the entire assignment axis has been
contesting. But it is not reachable by thresholding either — recovering bag's remaining
26.8% costs 20 786 false positives (precision 5.8%). The binding constraint is
**confidence ranking**, and the misclassification rate held invariant at 3.94–4.90%
across 81 runs regardless of mechanism.

The localisation half of the same story: the mAP50→mAP50-95 ratio is **0.65 for small**
vs **0.75 for medium**, and it moved only 0.649→0.657 across 18 v12 runs. Small-box
*localisation* was the least-attacked axis, and NWD (the one mechanism aimed at it)
traded correctly — mAP50 and small AP rose monotonically with the blend weight while
mAP50-95 fell.

---

## 8. Dataset property → mechanism outcome, the full map

| dataset property | mechanism it motivated | outcome |
|---|---|---|
| 60% of instances small (max side <48 px) | SCB, SNL1, LB-TAL, NWD | SCB **+0.42**, SNL1 +0.25, LB-TAL +0.79 (v12) / −0.43 (v26), NWD −0.47 mAP50-95 but **+0.44 mAP50** |
| 51× candidate-supply gap small↔large | LB-TAL per-level budgets | single-peaked at P3=4, i.e. **cut** small's supply; adding supply gained nothing |
| 70.6% tall boxes, mean h/w 1.55 | **EIoU** | **falsified** — −0.09 overall, −7.80 on large |
| trolley 50.7% vs bag 22.2% | **cls_pw** inverse-frequency | **falsified** — bag got worse at both settings |
| large only 11.3% train / 7.7% val | batch-size and seed controls | large seed spread **2.06 pp**; large claims need 4 correlated metrics, not one |
| bag bimodal in size *and* shape | — | explains the 15.2 pp class gap that no mechanism moved |
| AR50_small 0.95 vs R50_small 0.70 | posboost, QFL, SNT, TSH, threshold sweep | all falsified or unreachable; **QFL −7.53**, **SNT −3.93/−12.00** on v26 |
| 640×360 fixed-camera framing | — | why 896-px runs were never justified on this dataset |

---

## 9. What a reader of the paper should take from this section

1. It is a **small-object, tall-box, fixed-camera** dataset — 60% of instances under 48 px,
   70.6% taller than wide, 80% of frames at 640×360.
2. The splits are **clean and well matched** on class and shape, with one documented
   drift: **val under-represents large objects and its boxes are 23% smaller in area
   than test's**.
3. The assigner is **not** the bottleneck. Stock TAL already assigns 60% of a small
   object's candidate anchors; the supply ceiling is geometric and cannot be raised by
   any budgeting scheme.
4. The bottleneck is **confidence ranking**: ~95% of small objects are proposed, ~76%
   are read, and the gap is not recoverable by thresholding at any usable precision.
5. The worst class (**bag**, 15.2 pp behind trolley) is limited by **category
   heterogeneity**, not by frequency, geometry, or proposal quality — a labelling
   question, not an optimisation one.

---

## Files

```
DATASET_v6i/
├── DATASET_ANALYSIS_v6i.md          this document
├── SIZE_THRESHOLDS.md               all four size taxonomies + results under each
├── README.md
├── raw/
│   ├── LuggageDatasetSplitv6i.txt         label-level scan, 8 sections
│   └── diag_anchor_footprin_results.txt   anchor geometry, 4 assigner passes, F1–F12
├── scripts/
│   ├── analyze_dataset_v6i.py             regenerates the scan + CSVs (arg-free)
│   └── diag_anchor_footprint.py           regenerates the anchor diagnostic
└── tables/
    ├── split_summary.csv
    ├── class_counts.csv
    ├── size_by_split.csv                  taxonomies A and B
    ├── size_distribution_all_thresholds.csv   all four taxonomies, all splits
    ├── size_thresholds_crosswalk.csv      definition, status, what uses it
    ├── class_size_all_thresholds_train.csv
    ├── class_size_shape_train.csv
    ├── shape_by_split.csv
    ├── all_runs_per_size_COCOarea.csv     all 119 runs x 24 per-size metrics
    ├── anchor_candidate_pool.csv          the 51× table
    ├── assigner_crosstab.csv              size × level × scheme
    └── positive_set_quality.csv           mean/p10 IoU, %<0.3, centre distance
```
