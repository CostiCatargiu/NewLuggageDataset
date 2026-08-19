# YOLO26 (yolo26s) — LuggageDataset v6i — what was tried, what worked

**Baseline** `y26_base_rep` = `yolo26_custom-9` — 55.24 mAP50-95 / 80.18 mAP50, stock
yolo26s, 640 px, 70 epochs, seed 0, **batch 82**.

**73 runs** across 16 result files. All numbers are the v6i **test** split
(1219 images, 6172 instances).

**Training on this box is deterministic** — `y26_base_rep` came back bit-identical to
`yolo26_custom-9` across all 118 metric values, and `y26_identity` reproduced it again
through a rebuilt `metrics.py`. So every delta below is *exact*, not an average. It is
also **single-seed**: exact does not mean general, and that belongs in the limitations.

---

## The headline

| | config | mAP50-95 | Δ | Δ% |
|---|---|---|---|---|
| **Best loss** | `y26_scb3_sbb50` | 55.65 | +0.41 | +0.74% |
| Best loss (raw mAP) | `y26_scb2_sbb50` | 55.70 | +0.46 | +0.83% |
| **Best arch** | `y26_p2k2_hi` | 56.46 | +0.70 vs matched b32 | +1.26% |

`scb3_sbb50` is the config to report even though `scb2_sbb50` has the higher mAP: it is
the only configuration in the campaign that gains overall **without giving up the large
bucket** (mAP50 large 83.36 vs baseline 81.75, +1.97%). `scb2_sbb50` pays −5.28% there.

---

## What worked

### SCB — Size-Conditioned Beta *(the only mechanism with a real effect)*
`align_metric = score^alpha · IoU^beta` selects positives, and in the one2one branch
`topk2 = 1` means it picks **the single anchor** that produces every prediction. IoU is a
high-variance ranking signal on small boxes and a stable one on large boxes, so a single
global `beta` over-trusts IoU exactly where it is least reliable. SCB interpolates beta
by GT size.

`tal_beta_small` 3.0 → **+0.42**. But its own sweep was a single-point spike
(2.0 → −0.19, 3.0 → **+0.42**, 4.0 → −0.07), which made it look like an artifact for two
days until SBB rescued it.

### SBB — Size-conditioned Branch Blending *(only meaningful paired)*
Alone: +0.15, and it costs 4.03 points of large. Its value is entirely as a
**counterweight**.

### The one real finding: opposing size-biases recover large
Every single-direction "help small objects" mechanism buys overall mAP by giving up
large. Pairing it with an opposing large-side push recovers it — **confirmed three times,
with three different mechanisms at three different levels of the pipeline**:

| pair | large alone | large paired |
|---|---|---|
| SCB + SBB | 59.43 (−1.44) | **60.82 (−0.05)** |
| SNL1 + SBB | 56.50 (−4.37) | 60.48 (−0.39) |
| NWD + SBB | 78.13 mAP50 (−3.62) | 80.81 (−0.94) |

SCB+SBB is **super-additive**: SCB alone 59.43, SBB alone 56.84, the pair 60.82 — 1.39
above the better single and 3.98 above the worse. Neither single predicts it.

### `scb2_sbb50` — a failed setting rescued by pairing
`beta_small = 2.0` was the *worst* point of the SCB sweep (55.05, below baseline, the
only SCB setting that lost). Paired with SBB it becomes the campaign's mAP maximum
(55.70). A below-baseline config turned above-baseline is stronger evidence for the
principle than another point near an optimum.

### Architecture — P2 head + DySample
One `DySample` at P3→P2, groups=4. Every deviation lost: count 0/1/2/3 →
55.03/55.94/55.57/54.49; groups 2/4/8 → 55.34/55.94/55.52. Four module additions
(`ZGGlobalContext2`, `ZGDSConv`, others) all lost.

**+0.70 against a matched b32 control** (not the +1.22 against b82 — see the confound
section). Recall +1.78%, small +1.41%, paid for on large −2.28%.

---

## What did not work

| mechanism | result | why it is still worth reporting |
|---|---|---|
| **SNT** — soft negative targets | **−3.93** (τ=0.25) / **−12.00** (τ=0.50), monotone | The largest effect in the campaign, and the most informative. Recall went **UP** (AR50_95_small 71.68 → 76.00) while AP collapsed, and large fell hardest (−16.60). AR up + AP down is a **ranking** failure: in an NMS-free head the confidence gap between the selected anchor and its neighbours *is* the duplicate suppression, and SNT closed it. |
| **TSH** — target sharpening | falsified on a **pre-registered** criterion | The inverse of SNT: widen the same gap. Both rho points lowered large (−3.54, −3.88). Together with SNT this shows the gap sits at an **interior optimum** — closing it costs 10.6 large, widening it costs 3.5. |
| **SNL1** — scale-normalised L1 | +0.25 alone, **−0.37** on top of SCB, large −4.37 | Does not stack. `p` is flat (0.25 vs 0.50 identical in three separate contexts). Identical to the `dfl_obj_norm` proposal in `small-object-loss-mods` — do not re-run it. |
| **NWD** — Gaussian Wasserstein blend | mAP50 **+0.44**, mAP50-95 **−0.46**, both monotone in `nwd` | Not a failure — a *trade*. A forgiving similarity buys detection and loses localisation. Small mAP50 78.26, best of any config. On the confusion metrics `nwd50_sbb` is the best pure-loss config in the campaign. |
| **EIoU** | −0.09, large **−7.80** | The tall-box argument (70.6% of boxes have h/w > 1.25) was reasonable. The data said no. |
| **cls_pw** — class-frequency weighting | bag 48.2 → **47.4** | Made the worst class worse. Argues bag's deficit is not sample count. |
| **SWA** (ported from v12) | −0.48 | |
| **LB-TAL** (ported from v12) | −0.43 to −0.82 | Per-level budgets; every variant lost on the P2 arch. |
| **`sbb_q` sweep** | 0.25 → 55.28, 0.50 → 55.65, 0.75 → 55.20 | Knife-edge, not a plateau. Report it as such. |
| gain sweeps (`box`/`cls`/`dfl`, `tal_alpha`, `tal_beta`) | ≈ +0.13 | Axis answered. |

**Ported v12 mechanisms lost across the board.** That is the paper's comparative point:
improvements are architecture-specific, and the reason is structural — DFL-free
regression (`reg_max = 1`) and an NMS-free one2one head.

---

## Two confounds that cost real time

### Batch size is worth more than architecture
```
stock @ b82   55.24        stock @ b32   55.76        batch alone: +0.52
```
Every arch run was b32/b48 against a **b82** baseline. Between **42% and 62%** of each
published architecture gain is batch, not architecture:

| config | vs b82 | vs b32 | batch share |
|---|---|---|---|
| `y26_p2k2_hi` | +1.22 | **+0.70** | 42% |
| `y26_p2k1_lo` | +1.01 | +0.50 | 51% |
| `y26_dys_p2rich` | +0.83 | +0.32 | 62% |

It distorts the size columns too: against b82, `p2k2_hi` reads large at −1.02%; against
the matched control it is −2.28%. The confounded version *understates* the cost.

**Still open:** `p2k2_hi`/`p2k1_lo` ran at b48 against a b32 control. The b48 stock run
makes this final.

### The loss axis is the productive one, once batch is controlled
```
batch alone (b82 -> b32)     -1.83 missed detections
LOSS axis at fixed b82       -1.62
ARCH b48 vs a b32 control    -0.42
ARCH + loss, b32 vs b32      -0.06   (clean)
```

---

## The diagnosis that reframes everything

Confusion matrices over **81 runs** plus a low-confidence re-scoring pass:

**1. Classification is a constant.** Misclassification spans **3.94–4.90%** across all 81
runs; missed spans 15.03–18.96%. Twelve loss mechanisms, three architectures, four
assignment schemes — *nothing* moved classification. It is a property of the data.

**2. Everything is a recall problem.** `correct ≈ 100 − missed − 4.4`.

**3. The misses are SCORING, not PROPOSAL.** Re-running at conf=0.001:

| class | now | ceiling | headroom | true_miss |
|---|---|---|---|---|
| bag | 68.3 | 95.1 | +26.8 | **4.9** |
| backpack | 78.0 | 95.9 | +17.9 | 4.1 |
| trolley | 84.7 | 97.3 | +12.7 | 2.7 |

Only ~4% of ground truths are genuinely undetectable. **The model is ~95% capable and is
being read at ~76%.**

**4. But the ceiling is unreachable by thresholding.** Recovering bag's 26.8% drags in
**20,786 false positives** — sixteen per true positive, precision 5.8%. The correct boxes
exist but are ranked *below junk*.

```
bag       thr      TP      FP       P       R      F1
        0.001    1268   20786     5.8    95.3    10.8
        0.250     909     466    66.1    68.3    67.2
        0.400     816     260    75.8    61.4    67.8   <- best
```

**The binding constraint is confidence ranking** — not detection, not localisation, not
classification. Every mechanism tried operated on assignment, localisation or box
regression. None touched score *ordering*. That explains the entire flat campaign in one
sentence, and it is measured rather than argued.

**5. Free win: per-class thresholds** (tuned on val, applied to test, transfer near-exact):
```
micro @ 0.25 (current)   P 73.4   R 80.0   F1 76.6
micro @ per-class tuned  P 83.4   R 74.0   F1 78.5     +1.9
```
The optimum is *higher* than 0.25 (0.40–0.50), not lower.

---

## Folder layout

```
training/   every run_yolo26_*.py + verify_patch_v6i.py
results/    every runs_yolo26_*__test_full_dataset.json  (73 runs, 16 files)
patch/      the patched loss.py, tal.py, metrics.py, default.yaml
```

## Provenance for the headline numbers

```
y26_base_rep        runs_yolo26_sbb_overnight_v6i__test_full_dataset.json
y26_stock_b32       runs_yolo26_round10_v6i__test_full_dataset.json
y26_scb2_sbb50      runs_yolo26_round10_v6i__test_full_dataset.json
y26_scb_b3          runs_yolo26_loss_study_v6i__test_full_dataset.json
y26_scb3_sbb50      runs_yolo26_combo_v6i__test_full_dataset.json
y26_p2k2_hi         runs_yolo26_round5_v6i__test_full_dataset.json
y26_p2k1_lo         runs_yolo26_round5_v6i__test_full_dataset.json
y26_dys_p2rich      runs_yolo26_overnight4_v6i__test_full_dataset.json
```

## Known-bad runs — exclude from any table

- `y26_sqrt0703-4` — 55.65 correct / 39.50 missed / bag 28.1. Failed, not weak.
- `y26_lsshift`, `y26_gctxp3` — built on a C2PSA skeleton mismatch (round-1 YAMLs used
  `reps=1 args=[1024,1]`, later rounds the shipped `reps=2 args=[1024]`). Uninterpretable.
- `y26_s10_*` — these are round-6 **LB-TAL budget** configs on the P2 architecture at b32,
  *not* loss runs. They rank near the top if you classify by name; classify by source file.

## APPENDIX — every config, what it was meant to do, what it did

All 73 runs. Δ is vs `y26_base_rep` 55.24 **regardless of batch**, so arch rows
(b32/b48) are inflated by ~+0.52 — see the confound section. S/M/L are mAP50.

### SCB — Size-Conditioned Beta  *(assignment; the one that worked)*
Intent: `align_metric = score^α · IoU^β`. IoU is noisy on small boxes and stable on
large ones, so one global β over-trusts IoU exactly where it is least reliable.
Interpolate β by GT size — β_small for objects ≤ `beta_ref_px`, β for larger.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_scb_b3` | β_small 3.0 | 55.66 | **+0.42** | best single mechanism in the campaign |
| `y26_scb_s4` | β_small 4.0 | 55.17 | −0.07 | weaker push, lost |
| `y26_scb_s2` | β_small 2.0 | 55.05 | −0.20 | stronger push, **worst SCB point** — later rescued by SBB |
| `y26_scb_r32` | ref 32 px | 55.22 | −0.02 | reference-size axis is flat |
| `y26_scb_r128` | ref 128 px | 55.21 | −0.04 | " |
| `y26_scb_b4s2` | β 4→2 | 55.49 | +0.25 | |
| `y26_scb_b3_arch` | β 3.0 + P2 arch | 55.95 | +0.71 | b32 — arch-confounded |

**Verdict:** works, but the β sweep is a single-point spike (2.0 −0.20, 3.0 **+0.42**,
4.0 −0.07). Looked like an artifact until SBB rescued the failing settings.

### SBB — Size-conditioned Branch Blending  *(regression weight, per branch)*
Intent: the two branches differ in supervision density (one2many topk=10 vs one2one
topk2=1). Give them **opposite** size preferences so each specialises — a question only
a dual-branch head can pose.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_sbb_inv50` | q 0.5, invert | 55.39 | +0.15 | alone: weak, and costs 4.03 on large |
| `y26_sbb_q50` | q 0.5, no invert | 55.16 | −0.08 | the losing sign (one2one → small) |
| `y26_scb3_sbb50` | + SCB 3.0 | 55.65 | +0.41 | **large 83.36 — the only config that gains without losing large** |
| `y26_scb2_sbb50` | + SCB 2.0 | 55.70 | **+0.46** | a below-baseline setting turned best-of-campaign |
| `y26_scb3_sbb25` | q 0.25 | 55.28 | +0.04 | |
| `y26_scb3_sbb75` | q 0.75 | 55.20 | −0.04 | q is a **knife-edge**, not a plateau |
| `y26_snl1_sbb` | + SNL1, no SCB | 55.16 | −0.08 | large recovered 56.50→60.48 — principle holds |
| `y26_scb3_snl25_sbb` | all three | 55.59 | +0.35 | |

**Verdict:** near-worthless alone, essential as a counterweight. Its whole value is the
opposing-bias principle.

### SNL1 — Scale-Normalised L1  *(regression normalisation)*
Intent: YOLO26 is DFL-free, so the L1 target is normalised by **image** size — an 8 px
box yields 0.0063 and a 256 px box 0.20, a 32× gradient difference for the same relative
error. Divide by the GT's own extent instead.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_snl1_p25` | p 0.25 | 55.49 | +0.25 | works alone, large **79.37** (−2.4) |
| `y26_snl1_p50` | p 0.50 | 55.48 | +0.24 | p axis is flat |
| `y26_scb3_snl25` | + SCB | 55.29 | +0.05 | **does not stack** (−0.37 on SCB) |
| `y26_scb3_snl50` | + SCB | 55.36 | +0.11 | |
| `y26_snl1_p25_arch` | + P2 arch | 55.85 | +0.61 | b32 |

**Verdict:** the diagnosis is right, the fix buys +0.25 and costs large. Identical to the
`dfl_obj_norm` proposal in `small-object-loss-mods` — **do not re-run it**.

### SNT — Soft Negative Targets  *(FALSIFIED, and the most informative failure)*
Intent: `topk2=1` makes every non-selected anchor a hard negative, and the count of
well-fitting discarded anchors scales with object size. Give near-misses a soft target.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_snt_t25` | τ 0.25 | 51.31 | **−3.93** | small −3.54, **large −10.58** |
| `y26_snt_t50` | τ 0.50 | 43.24 | **−12.00** | small −12.67, **large −16.60** |

Recall went **UP** (AR50_95_small 71.68 → 76.00) while AP collapsed. AR up + AP down is a
**ranking** failure: in an NMS-free head the winner/runner-up confidence gap *is* the
duplicate suppression, and SNT closed it. Large fell hardest because large objects span
the most anchors and so emit the most duplicates — the mechanism's fingerprint.

### TSH — Target SHarpening  *(FALSIFIED on a pre-registered criterion)*
Intent: the inverse of SNT. If closing the gap costs 10.6 large, **widen** it —
`target_scores ** ρ`, ρ<1 pushes the winner toward 1.0.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_sharp_r75` | ρ 0.75 | 55.35 | +0.11 | large 79.90 (−1.85) |
| `y26_sharp_r50` | ρ 0.50 | 55.38 | +0.14 | large 79.11 (−2.64), **medium 87.17 best of the block** |

Both ρ points lowered large. Written before the runs: *"both LOWER large → stock is
already at the optimum."* Combined with SNT this is a **two-sided** result — closing the
gap costs 10.6 large, widening it costs 2.6. The gap sits at an **interior optimum**,
which is why the whole soft-target family (VFL, label smoothing, quality-aware targets)
cannot work on this head.

### NWD / EIoU  *(the overlap metric itself — the only upstream change)*
Intent: every other mechanism reweights `bbox_iou`'s **output**; these change what it
**computes**. IoU is a ratio, so a 3 px error costs ~0.5 CIoU on a 15×25 box and ~0.05 on
a 100 px one — raised to β=6 that is a ~47× difference in assignment weight from the same
absolute error. NWD models boxes as Gaussians: an absolute measure, scale-invariant.

| config | setting | mAP50-95 | Δ | mAP50 | small | what happened |
|---|---|---|---|---|---|---|
| `y26_identity` | nwd 0, ciou | 55.24 | +0.00 | 80.18 | 77.30 | **port verified inert** (bit-identical to baseline) |
| `y26_nwd25` | nwd 0.25 | 55.05 | −0.20 | 80.37 | 77.94 | |
| `y26_nwd50` | nwd 0.50 | 54.78 | −0.47 | **80.62** | **78.19** | |
| `y26_nwd50_sbb` | + SBB | 54.69 | −0.55 | 80.56 | 78.05 | large 78.13→80.81, principle 3rd time |
| `y26_nwd50_scb3_sbb` | + SCB + SBB | 54.56 | −0.68 | 80.73 | **78.26** | best small of any config |
| `y26_eiou` | iou_type eiou | 55.15 | −0.09 | 79.31 | 76.63 | **large 73.95 (−7.80)** — falsified |

**Verdict:** NWD is a *trade*, not a failure — mAP50 and small rise monotonically with
`nwd` while mAP50-95 falls monotonically. A forgiving similarity buys detection and loses
localisation. On the **confusion metrics** `y26_nwd50_sbb` is the best pure-loss config in
the campaign (80.12 correct vs 79.77), which mAP50-95 completely hides. EIoU's tall-box
argument (70.6% of boxes h/w>1.25) was reasonable and the data said no.

### cls_pw — class-frequency reweighting  *(FALSIFIED)*
Intent: dataset is 53% trolley / 20% bag, and bag is the worst class by 13.8 pp. Weight
the classification BCE by inverse frequency.

| config | setting | mAP50-95 | Δ | bag AP50-95 |
|---|---|---|---|---|
| `y26_scb3_sbb50_pw25` | p 0.25 | 55.11 | −0.13 | 47.37 (from 48.2) |
| `y26_scb3_sbb50_pw50` | p 0.50 | 55.63 | +0.39 | 47.50 (from 48.2) |

Bag got **worse** in both. Strong evidence its deficit is not sample count — later
confirmed: bag is scoring-limited, not frequency-limited.

### SWA — ported from YOLOv12  *(all lost)*
`y26_swa_a06_03` +0.35 · `y26_swa_a09_04` +0.13 · `y26_swa_a06_b25` +0.11 ·
`y26_swa_a06_b15` −0.06 · `y26_swa_a08_04` −0.18 · `y26_swa_a07_04` −0.49 ·
`y26_swa_a09_03` −0.63 · `y26_dys_swa0603` +0.36 (b32) · `y26_dys_swa_lb` +0.02 (b32)
· `y26_sqrt0703` +0.03

On v12 this family gave **+0.86 with 29/32 replications**. Here the best is +0.35, inside
the band, and the mean is negative. **The clearest cross-architecture failure in the
project.**

### LB-TAL — per-level top-k budgets, ported from YOLOv12  *(all lost)*
`y26_lb_uniform` +0.28 · `y26_lb_coarse244` +0.10 · `y26_lb_p4wide` +0.08 ·
`y26_lb_p3_3` −0.08 · `y26_cmb_p4wide` +0.02 · `y26_cmb_uniform` −0.38 ·
`y26_dys_lbuni` +0.41 (b32) · `y26_dys_lbp2k2` +0.02 (b32)

The `y26_s10_*` runs (`p45` +1.02, `hi` +0.86, `p5` +0.65, `bal` +0.65) are **also
LB-TAL**, on the P2 architecture at b32 — arch-confounded, *not* loss results.

### Gains and exponents  *(axis answered)*
`y26_dfl3` (dfl 1.5→3.0) +0.13 · `y26_beta4` (β 6→4) +0.13 · `y26_alpha075` (α 0.5→0.75)
+0.35. Gain changes on this model are worth ~0.1–0.3. Not a mechanism.

### Architecture — P2 head + DySample  *(all b32/b48, Δ inflated by ~+0.52)*
| config | mAP50-95 | Δ | note |
|---|---|---|---|
| `y26_p2k2_hi` | 56.46 | +1.22 | **best arch**; +0.70 vs matched b32 control |
| `y26_p2k1_lo` | 56.25 | +1.01 | +0.50 matched |
| `y26_dys_p2rich` | 56.07 | +0.83 | +0.32 matched |
| `y26_dys_p2starve` | 56.03 | +0.79 | |
| `y26_p2k4_hi` | 55.91 | +0.67 | |
| `y26_arch_scb3_sbb50` | 55.57 | +0.33 | arch + best loss — **−0.19 vs stock b32; the axes do not compose** |
| `y26_p2_wide` | 55.53 | +0.29 | |
| `y26_wide_starve` | 55.46 | +0.22 | |
| `y26_stock_b32` | 55.76 | +0.52 | **the control — batch alone** |
| `y26_p2_dys_gctx` | 54.79 | −0.45 | +ZGGlobalContext2 |
| `y26_p2_dys3` | 54.49 | −0.75 | 3 DySamples |

DySample count 0/1/2/3 → 55.03/55.94/55.57/54.49; groups 2/4/8 → 55.34/55.94/55.52.
**One DySample at P3→P2 with groups=4 is the peak; every deviation loses.** Four module
additions all lost.

### Excluded — do not use
`y26_lsshift` +0.34 and `y26_gctxp3` −0.15 were built on a **C2PSA skeleton mismatch**
(round-1 YAMLs used `reps=1 args=[1024,1]`, later rounds the shipped `reps=2 args=[1024]`).
Uninterpretable. `y26_sqrt0703-4` is a failed run (39.50 missed).

---

## Never run

`run_yolo26_overnight_v6i.py` defines five runs at **batch 16 held constant**
(`y26_3lvl_640_b16`, `y26_3lvl_896_b16`, `y26_p2_896_b16`, `y26_m_640_b16`,
`y26_m_p2_640_b16`). No results exist for any of them. It is the best-designed runner in
the project — batch pinned so the deltas are attributable — and it would answer the
**resolution** question, which is the largest untested effect. Note the source images are
natively 640×360, so 896 is *upsampling*: more pixels, not more information.
