# YOLO26 (yolo26s) — LuggageDataset v6i — what was tried, what worked

**Baseline** `y26_base_rep` = `yolo26_custom-9` — 55.24 mAP50-95 / 80.18 mAP50, stock
yolo26s, 640 px, 70 epochs, seed 0, **batch 82**.

**92 runs** across 17 result files (73 through round 11, plus 19 from rounds 12–15).
All numbers are the v6i **test** split (1219 images, 6172 instances).
A further **8 runs exist only as diagnostics** — see the last section.

**Training on this box is deterministic** — `y26_base_rep` came back bit-identical to
`yolo26_custom-9` across all 118 metric values, and `y26_identity` reproduced it again
through a rebuilt `metrics.py`. So every delta below is *exact*, not an average. It is
also **single-seed**: exact does not mean general, and that belongs in the limitations.

---

## BASELINE AND CONTROL RUNS — the reference points every delta is taken against

Every run below uses **stock loss** (`_ALL_OFF`, no mechanism live), 640 px, 70 epochs,
seed 0. They differ only in batch size and in graph. Test split, COCO-area size buckets.

| run | batch | graph | mAP50-95 | mAP50 | P | R | S50 | M50 | L50 | S95 | M95 | L95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `yolo26_custom-9` | 82 | 3-lvl stock | **55.24** | 80.18 | 80.32 | 71.43 | 77.30 | 86.45 | 81.75 | 51.00 | 65.98 | **60.87** |
| `y26_base_rep` | 82 | 3-lvl stock | 55.24 | 80.18 | 80.32 | 71.43 | 77.30 | 86.45 | 81.75 | 51.00 | 65.98 | 60.87 |
| `y26_identity` | 82 | 3-lvl stock | 55.24 | 80.18 | 80.32 | 71.43 | 77.30 | 86.45 | 81.75 | 51.00 | 65.98 | 60.87 |
| `y26_stock_b48` | **48** | 3-lvl stock | **55.76** | 80.96 | 79.59 | 73.18 | 77.68 | **87.46** | 82.80 | 50.96 | **67.31** | 59.28 |
| `y26_3lvl_head64` | 32 | 3-lvl, c3=64 | 55.25 | 79.86 | 79.02 | 73.83 | 77.26 | 85.96 | 81.19 | 51.22 | 65.83 | 57.15 |
| `y26_p2_headref0` | 32 | P2, c3=64 | 55.76 | 81.05 | 80.16 | 73.34 | 78.57 | 86.71 | 78.60 | **52.17** | 65.59 | 54.87 |
| `y26_p2_headref128` | 32 | P2, c3=128 | 54.61 | 80.18 | 80.20 | 73.29 | 77.81 | 85.87 | 80.31 | 51.19 | 64.11 | 58.18 |
| `y26_p2_remap` | 32 | P2 + **remap** | **55.84** | 80.78 | **80.69** | 72.82 | 77.63 | 87.19 | 81.10 | 51.54 | 66.47 | 58.53 |
| `y26_p2add_h0` | 32 | P2add + remap | 55.83 | 80.83 | 79.10 | 74.48 | 77.98 | 86.78 | 81.02 | 51.48 | 66.46 | 58.51 |
| `y26_remap_dys_stock` | 32 | P2+DySample+remap | 55.70 | 80.69 | 79.84 | **74.59** | **78.39** | 85.82 | 79.07 | 51.93 | 65.44 | 57.05 |

Per-class AP50-95, same runs:

| run | backpack | bag | trolley | AR50 small | R50 small |
|---|---|---|---|---|---|
| `y26_base_rep` (=custom-9, =identity) | 56.16 | 47.38 | **62.18** | **96.31** | 70.67 |
| `y26_stock_b48` | 57.07 | 48.27 | 61.93 | **96.66** | 71.67 |
| `y26_3lvl_head64` | 56.84 | 47.52 | 61.40 | 94.85 | 71.33 |
| `y26_p2_headref0` | 57.13 | 47.93 | 62.22 | 96.17 | 72.33 |
| `y26_p2_headref128` | 55.98 | 46.44 | 61.42 | 95.17 | 72.67 |
| `y26_p2_remap` | 57.24 | 47.80 | **62.47** | 95.92 | 70.67 |
| `y26_p2add_h0` | 57.40 | **48.39** | 61.69 | 94.96 | 71.67 |
| `y26_remap_dys_stock` | **57.45** | 47.73 | 61.92 | 94.53 | **74.33** |

For reference, YOLOv12: **`yolov12s_default` = 54.77 / 79.75**, P 80.37 R 72.16,
S50 76.65 M50 86.59 L50 81.87, S95 49.98 M95 65.07 L95 57.73,
backpack 55.81 / bag 46.66 / trolley 61.85.

### How many baselines actually exist

Ten rows above, but far fewer independent measurements:

- **b82 stock: one.** `yolo26_custom-9`, `y26_base_rep` and `y26_identity` are
  **bit-identical to 10 decimal places** — same mAP50, same S95, same L95, run from three
  different scripts on three different dates. That is a *determinism proof* (and it is what
  certifies the round-11 `metrics.py` rebuild as inert), **not a replication**. Three
  identical draws of a deterministic computation carry the information of one.
- **b48 stock: one.** `y26_stock_b48`.
- **b32 stock 3-level: none.** `y26_3lvl_head64` is 3-level but with the head shrunk to
  c3=64, so it is not a stock control.
- **Seed-varied baselines: zero.** No baseline has ever been run at a second seed, on
  either model.

So the reference point for 73 loss deltas is one number with no error bar, and the
reference for the arch deltas is a different one number at a different batch.

### ⚠ `y26_stock_b48` was called `y26_stock_b32` until now

The run was **batch 48**, changed on the training box; the runner script still said 32 and
the name followed the script. Renamed everywhere on 2026-08-21. Consequences:

1. **`y26_p2k2_hi` / `y26_p2k1_lo` (b48) now HAVE a matched control.** The "still open"
   item below — *"p2k2_hi/p2k1_lo ran at b48 against a b32 control, the b48 stock run does
   not exist"* — is **closed**. It does exist; it is this run. Those two arch numbers are
   batch-matched after all.
2. **Rounds 13/14/15 are NOT matched.** They ran at b32 against 55.76 believing it was b32.
   `BATCH = 32 # matches ... y26_stock_b48` in `run_yolo26_round13_v6i.py` and
   `BATCH_ARCH` in `run_yolo26_overnight_r1213_v6i.py` were wrong; both now carry a warning.
3. **It may explain the control failure.** `y26_p2_headref0` (b32) returned 55.76 where
   `y26_p2_b32` = 55.03 was expected. If the earlier P2 run was also b48, that 0.73 gap is
   a *batch* effect, not the tree change it was read as — which would partly un-void the
   earlier arch deltas rather than voiding them.
4. **"batch alone: +0.52" is now b82→b48, not b82→b32.** Every statement in the confound
   section below that says "b32" for the control means **b48**.

**Unresolved and only you can answer:** the actual batch used for `y26_p2_b32`,
`y26_p2k2_hi`, `y26_p2k1_lo` and the DySample family. `params` and `run_meta` are empty
dicts in every results JSON and no `args.yaml` was captured, so **the repo has no record of
the batch size for any run** — only what the scripts claim, and the scripts have now been
wrong once. Worth capturing `args.yaml` alongside results from here on.

---

## The headline

| | config | mAP50-95 | Δ | Δ% |
|---|---|---|---|---|
| **Best loss** | `y26_scb3_sbb50` | 55.65 | +0.41 | +0.74% |
| Best loss (raw mAP) | `y26_scb2_sbb50` | 55.70 | +0.46 | +0.83% |
| **Best arch** | P2 + DySample, **n=10** | **56.08 ± 0.19** | **+0.32** vs `y26_stock_b48` | +0.57% |

`scb3_sbb50` is the config to report even though `scb2_sbb50` has the higher mAP: it is
the only configuration in the campaign that gains overall **without giving up the large
bucket** (mAP50 large 83.36 vs baseline 81.75, +1.97%). `scb2_sbb50` pays −5.28% there.

**Do not report `y26_p2k2_hi` 56.46 as the architecture number.** Rounds 4-6 passed
LB-TAL budgets while the **stock** `loss.py` was installed: the config system accepted
`use_lbtal=True`, the header printed it, the preflight validated the budget dict — and
nothing read the flag, so the assigner was never built. The ten differently-labelled
budget runs are **ten replicates of one configuration** (P2 + DySample, stock loss,
b32/640/seed 0):

```
overall  55.89 .. 56.46    mean 56.08   sd 0.19
small                      mean 52.14   sd 0.23
large    53.72 .. 60.66    mean 57.21   sd 2.11
```

So 56.46 is **the luckiest of ten draws of doing nothing**, and the architecture figure is
the mean, 56.08. Large is unusable at sd 2.11 — differences under ~4 pp there are not
readable, which retires several large-bucket stories told earlier in this project.

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
One `DySample` at P3→P2, groups=4. Deviations: count 0/1/2/3 →
55.03/55.94/55.57/54.49; groups 2/4/8 → 55.34/55.94/55.52. Four module additions
(`ZGGlobalContext2`, `ZGDSConv`, others) lost.

The architecture figure is the **mean of the ten collapsed budget runs, 56.08 ± 0.19**,
not the best of them — **+0.32** against the `y26_stock_b48` control (55.76). Recall +1.78%,
small +1.41%, paid for on large −2.28%.

**Which of those claims actually survive the noise.** Every architecture run is n=1, and
the replicate distribution is sd **0.19** (the ten collapsed runs, one configuration) to
**0.46** (the eight heterogeneous b32 arch runs). Measuring each deviation against the
arch mean 56.08:

| claim | value | Δ | vs sd 0.46 | verdict |
|---|---|---|---|---|
| DySample count 0 is worse | 55.03 | −1.05 | 2.3 sd | **holds** |
| DySample count 3 is worse | 54.49 | −1.59 | 3.5 sd | **holds** |
| `ZGGlobalContext2` is worse | 54.79 | −1.29 | 2.8 sd | **holds** |
| DySample count 2 is worse | 55.57 | −0.51 | 1.1 sd | **not established** |
| groups 8 is worse | 55.52 | −0.56 | 1.2 sd | **not established** |
| groups 2 is worse | 55.34 | −0.74 | 1.6 sd | **weak** |
| `p2_wide` is worse | 55.53 | −0.55 | 1.2 sd | **not established** |

So **"one DySample at groups=4 is the peak, every deviation loses" holds only for the
extremes.** The groups sweep establishes nothing at all — 55.34 / 55.94 / 55.52 is one
draw each from a distribution whose sd is comparable to the whole spread. Report the
module choice as "1 DySample, groups=4, chosen from an underpowered sweep", not as an
optimum.

**The control is also n=1.** The +0.32 is `56.08 ± 0.19` against a single `y26_stock_b48`
run with no error bar of its own. The stock model is deterministic, so three seeds
(~5 GPU-h) would put an interval on both sides of the comparison. That is the cheapest
run in the project that strengthens an existing headline instead of chasing a new one.

**Why replicates vary at all when this box is otherwise deterministic:** `DySample` calls
`F.grid_sample`, whose CUDA backward uses atomic adds and is nondeterministic. The stock
3-level model reproduces bit-identically (`y26_base_rep` matched `yolo26_custom-9` across
all 118 values); the P2 + DySample graph does not, which is exactly why it has a spread
to measure and the stock model does not.

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
| **LB-TAL** (ported from v12) | −0.43 to −0.82 | Per-level budgets. **Rounds 4-6 never actually ran it** — stock `loss.py` ignored the flag, so ten budget labels were one config. Genuinely tested only in round 7, where both variants landed inside the null band. |
| **`sbb_q` sweep** | 0.25 → 55.28, 0.50 → 55.65, 0.75 → 55.20 | Knife-edge, not a plateau. Report it as such. |
| gain sweeps (`box`/`cls`/`dfl`, `tal_alpha`, `tal_beta`) | ≈ +0.13 | Axis answered. |

**Ported v12 mechanisms lost across the board.** That is the paper's comparative point:
improvements are architecture-specific, and the reason is structural — DFL-free
regression (`reg_max = 1`) and an NMS-free one2one head.

---

## Two confounds that cost real time

### Batch size is worth more than architecture
```
stock @ b82   55.24        stock @ b48   55.76        batch alone: +0.52  (b82 -> b48)
```
Every arch run was b32/b48 against a **b82** baseline. Between **42% and 62%** of each
published architecture gain is batch, not architecture:

| config | vs b82 | vs b48 | batch share |
|---|---|---|---|
| `y26_p2k2_hi` | +1.22 | **+0.70** | 42% |
| `y26_p2k1_lo` | +1.01 | +0.50 | 51% |
| `y26_dys_p2rich` | +0.83 | +0.32 | 62% |

It distorts the size columns too: against b82, `p2k2_hi` reads large at −1.02%; against
the matched control it is −2.28%. The confounded version *understates* the cost.

**RESOLVED 2026-08-21.** This section previously read *"still open: p2k2_hi/p2k1_lo ran at
b48 against a b32 control; the b48 stock run makes this final."* That b48 stock run
**exists** — it is `y26_stock_b48`, which was mislabelled `y26_stock_b32` because the
runner script said 32 while 48 was used on the training box. So the `vs b48` column above
is batch-matched for `p2k2_hi`/`p2k1_lo` after all.

What this breaks instead: **rounds 13/14/15 ran at b32 against this b48 control.** Every
delta in the round-13 head-width square and the round-14/15 remap decomposition is
cross-batch. And the batch used for `y26_p2_b32` and the DySample family is **unrecorded** —
see the baseline section above.

### A second confound stacked on the first: 56.46 is the max of ten replicates
After removing batch, what is left of `y26_p2k2_hi`'s +0.70 is still not a single
measurement. Every `p2k*` / `dys_p2*` run in rounds 4-6 requested an `_lb(...)` per-level
budget against a **stock** `loss.py` that never read the flag — so the assigner was never
built and all ten are the SAME configuration:

```
stock b48 control (y26_stock_b48)          55.76
P2 + DySample, ten replicates    mean      56.08   sd 0.19    -> architecture  +0.32
                                 best      56.46              -> the reported number
```

Reporting the best of ten draws as the effect adds roughly **+0.38 of selection** on top
of the batch inflation. `run_yolo26_dysample_sweep_v6i.py` sets no `use_lbtal` and varies
the YAML, so its count/groups sweeps are genuine architecture comparisons — but they must
be read against sd 0.19, not against zero.

**The lesson the project already learned the hard way:** a config key can be accepted by
`default.yaml`, echoed in the run header, and validated by a preflight, while the file
that has to consume it ignores it. Three different files, and only the third matters.
Round 7 added the epoch-1 liveness guard for exactly this, and every runner since carries
it.

### The loss axis is the productive one, once batch is controlled
```
batch alone (b82 -> b48)     -1.83 missed detections
LOSS axis at fixed b82       -1.62
ARCH b48 vs the b48 control  -0.42   (now batch-matched)
ARCH + loss, b32 vs b48      -0.06   (NOT batch-matched)
```

The "ARCH" rows inherit the replicate problem above: if they were computed from
`y26_p2k2_hi` they use the best of ten draws. Recomputing against the ten-run mean would
make the loss axis look stronger, not weaker.

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
y26_stock_b48       runs_yolo26_round10_v6i__test_full_dataset.json
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
- `y26_s10_*` — these are round-6 **LB-TAL budget** configs on the P2 architecture at b32.
  They are assignment results on a fixed graph, not architecture results, and they belong
  in the LB-TAL section rather than the architecture table. Classify by source file, not
  by name.

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

**Which way round.** `E2ELoss` assigns `one2many sign=-1, one2one sign=+1`, and
`sbb_weight` computes `w = (sqrt(area_px)/ref) ** (sign * q)`, so a **negative** sign
favours SMALL and a **positive** sign favours LARGE. Therefore:

| `sbb_invert` | one2many | one2one | result |
|---|---|---|---|
| `False` (default) | small | large | `y26_sbb_q50` 55.16, **−0.08** |
| `True` | large | **small** | `y26_sbb_inv50` 55.39, **+0.15** |

**The winning arm is `invert=True`: one2one leans SMALL, one2many leans LARGE** — and
`y26_scb3_sbb50`, the headline config, uses it (`run_yolo26_combo_v6i.py:235`). That is
the *opposite* of the intuition in the code comments ("one2one carries the large ones,
its single pick is reliable there"), which describes the arrangement that LOST. The
account that fits the data instead: one2one is the output branch and specialises on the
dominant, hardest population (small), while one2many is auxiliary and discarded at
inference, so it can absorb the large objects. Consistent with the winning arm costing
4.03 points of large.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `y26_sbb_inv50` | q 0.5, invert | 55.39 | +0.15 | **the winning sign** (one2one → small); alone it is weak and costs 4.03 on large |
| `y26_sbb_q50` | q 0.5, no invert | 55.16 | −0.08 | the losing sign (one2one → large) |
| `y26_scb3_sbb50` | + SCB 3.0 | 55.65 | +0.41 | **large 83.36 — the only config that gains without losing large** |
| `y26_scb2_sbb50` | + SCB 2.0 | 55.70 | **+0.46** | a below-baseline setting turned best-of-campaign |
| `y26_scb3_sbb25` | q 0.25 | 55.28 | +0.04 | |
| `y26_scb3_sbb75` | q 0.75 | 55.20 | −0.04 | q is a **knife-edge**, not a plateau |
| `y26_snl1_sbb` | + SNL1, no SCB | 55.16 | −0.08 | large recovered 56.50→60.48 — principle holds |
| `y26_scb3_snl25_sbb` | all three | 55.59 | +0.35 | |

**Verdict:** near-worthless alone, essential as a counterweight. Its whole value is the
opposing-bias principle.

**Open caveat — the effect is time-varying.** `E2ELoss` decays `o2m` 0.8 → 0.1, so the
LARGE-leaning branch carries ~80% of the loss early and the SMALL-leaning one ~90% late.
SBB therefore implements a large→small *curriculum*, not a static specialisation, which
is a better explanation for why `sbb_q` is a knife-edge (0.25 → 55.28, 0.50 → 55.65,
0.75 → 55.20) than a narrow optimum: `q` is integrated against a moving branch weight and
is not a single-axis knob. One run with `o2m` pinned would separate the two.

**Open caveat — SCB is not branch-isolated.** SBB, SNT and TSH are all scoped to one
branch inside `E2ELoss`. SCB is set in `v8DetectionLoss.__init__`, so `tal_beta_small`
applies to **both** branches — even though its justification ("topk2=1 makes the metric
pick a single anchor") is a one2one argument. The campaign's one working mechanism has
never been attributed to a branch. Two runs (SCB on one2one only, SCB on one2many only)
would settle it.

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

### LB-TAL — per-level top-k budgets, ported from YOLOv12  *(all lost; most were never run)*
On the stock 3-level model:
`y26_lb_uniform` +0.28 · `y26_lb_coarse244` +0.10 · `y26_lb_p4wide` +0.08 ·
`y26_lb_p3_3` −0.08 · `y26_cmb_p4wide` +0.02 · `y26_cmb_uniform` −0.38 ·
`y26_dys_lbuni` +0.41 (b32) · `y26_dys_lbp2k2` +0.02 (b32)

**Rounds 4-6 never actually ran the mechanism.** Those runs passed budgets against a
stock `loss.py` that does not read `use_lbtal`, so no `LevelBalancedTaskAlignedAssigner`
was ever constructed. That covers the whole `p2k*` family in
`run_yolo26_round5_v6i.py` and the `y26_s10_*` runs — ten different budget labels, one
configuration, spread 55.89..56.46 (sd 0.19) purely from `grid_sample` nondeterminism.
They are neither loss results nor architecture results; they are **replicates**, and that
is their only value — a free n=10 control distribution for the P2 graph.

The genuine test is round 7, which added the epoch-1 liveness guard: `lbuni` and `lbp2k2`
both landed inside the pre-registered null band (55.70..56.46). **LB-TAL does nothing on
YOLO26**, on either three levels or four.

### Gains and exponents  *(axis answered)*
`y26_dfl3` (dfl 1.5→3.0) +0.13 · `y26_beta4` (β 6→4) +0.13 · `y26_alpha075` (α 0.5→0.75)
+0.35. Gain changes on this model are worth ~0.1–0.3. Not a mechanism.

### Architecture — P2 head + DySample  *(all b32/b48, Δ inflated by ~+0.52 of batch)*
**rep** marks the ten runs whose LB-TAL budget was a silent no-op. They are replicates of
one configuration, so their individual Δ values are draws from **56.08 ± 0.19**, not
separate results. Read the whole block against that sd.

| config | mAP50-95 | Δ | note |
|---|---|---|---|
| `y26_p2k2_hi` | 56.46 | +1.22 | **rep** — the luckiest of the ten; do not report as an effect |
| `y26_p2k1_lo` | 56.25 | +1.01 | **rep** |
| `y26_dys_p2rich` | 56.07 | +0.83 | **rep** — essentially the mean |
| `y26_dys_p2starve` | 56.03 | +0.79 | **rep** |
| `y26_p2_dysample` | 55.94 | +0.70 | **rep** |
| `y26_p2k4_hi` | 55.91 | +0.67 | **rep** — the unluckiest, and 0.03 below `p2_dysample` |
| `y26_arch_scb3_sbb50` | 55.57 | +0.33 | arch + best loss — **−0.19 vs stock b32; the axes do not compose** |
| `y26_p2_wide` | 55.53 | +0.29 | 1.2 sd below the arch mean — **not established as worse** |
| `y26_wide_starve` | 55.46 | +0.22 | 1.3 sd below — not established |
| `y26_stock_b48` | 55.76 | +0.52 | **the control — batch alone, and itself n=1** |
| `y26_p2_dys_gctx` | 54.79 | −0.45 | +ZGGlobalContext2 — 2.8 sd, real |
| `y26_p2_dys3` | 54.49 | −0.75 | 3 DySamples — 3.5 sd, real |

The six **rep** rows span 55.91..56.46 while being the same configuration. Any
architecture claim smaller than ~0.4 in this table is unreadable — see the significance
table in the architecture section for which deviations survive and which do not.

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

---

## APPENDIX — 8 runs with NO results JSON, and their reconstruction

**Completeness audit, 2026-08-21.** Every YOLO26 run that has a
`__test_full_dataset.json` anywhere in the repo (92 distinct runs, checked across
`MODEL_v26/results/`, `archAblation/`, `round8_deploy/`, `round11_deploy/` and the
top-level scratch copies) **is** in `MODEL_v26/results/`. Nothing missing, nothing
orphaned, no run present in one tree and absent from the other.

But `DIAGNOSTICS/confusion_collected/` holds **80** YOLO26 run folders, and only 72 of
them have a results JSON. These 8 trained and were collected, but were never evaluated
into a results file — so any figure quoted for them elsewhere in this document came from
**console output**, not from a file in this repo.

### Reconstruction

`results.csv` survives for all 8, giving per-epoch **val** mAP. Calibrating val→test on
the 72 runs where both exist:

```
test mAP50-95  =  val(best epoch)  +  0.69      mean over n=72
                                     sd 0.30,  range +0.06 .. +1.30
```

| run | epochs | val best | @ep | **est. test** | quoted elsewhere | verdict |
|---|---|---|---|---|---|---|
| `y26_p2_b32` | 70 | 54.43 | 50 | **55.12** ± 0.30 | **55.03** | **corroborated** |
| `y26_snt_t25` | 70 | 50.55 | 58 | **51.24** ± 0.30 | **51.31** | **corroborated** |
| `y26_snt_t50` | 70 | 41.92 | 68 | **42.61** ± 0.30 | **43.24** | within ~2 sd |
| `y26_p2_dysample` | 70 | 55.10 | 48 | 55.79 ± 0.30 | — | new |
| `y26_p2_dys_snake` | 70 | 54.70 | 50 | 55.39 ± 0.30 | — | new |
| `y26_p2_snake_p3p4` | 70 | 54.39 | 57 | 55.08 ± 0.30 | — | new |
| `y26_levelspec` | **25** | 51.02 | 24 | — | — | **incomplete run — do not use** |
| `y26_sqrt0703-4` | **1** | 36.35 | 1 | — | — | **failed run — do not use** |

### What this settles

1. **The console numbers were real.** `y26_p2_b32` = 55.03 and the two SNT figures are
   independently corroborated from `results.csv`, which is a different artifact produced
   by a different code path. The −3.93 / −12.00 SNT result and the P2-costs-0.73 story
   both rest on genuine measurements.
2. **Therefore `y26_p2_headref0` really did fail to reproduce `y26_p2_b32`.** 55.76 vs a
   corroborated 55.03 is a real 0.73 discrepancy, not a mis-transcription — so it needs
   the batch or tree-change explanation, and cannot be waved away.
3. **Two runs in the confusion set are not results at all.** `y26_levelspec` stopped at
   epoch 25 and `y26_sqrt0703-4` at epoch 1. Both appear in
   `confusion_collected/ALL_RUNS_REPORT.txt` alongside completed runs with no marker
   distinguishing them. Anything read off that report should skip these two.

### Incidental finding — the val→test offset

`test = val_best + 0.69 ± 0.30` over 72 runs. Two consequences worth carrying:

- The **sd of 0.30** is a third independent estimate of this project's noise floor,
  matching the round-12 cls scatter (~0.3) and the last-10-epoch val swing (0.49–0.93),
  and contradicting the 0.12 figure taken from the single v12 `lb_uniform` seed pair.
- The offset is a **selection artifact**: `val best` is the max of ~70 correlated noisy
  evaluations, so it is biased high on val, yet test still comes in 0.69 *above* it. Test
  being systematically easier than val is consistent with the dataset finding that the val
  split carries smaller objects (mean box area 23% below test) and fewer large instances
  (7.7% vs 9.8%). See `DATASET_v6i/DATASET_ANALYSIS_v6i.md` §5.

### Known non-repo gap

`round8_deploy/run_yolo26_round10_v6i.py` is the real round-10 runner, but the copies at
`MODEL_v26/training/run_yolo26_round10_v6i.py` and at the repo root are **a Windows
disk-space scanner** that was saved over the filename. The genuine runner is only in
`round8_deploy/`. Fix before archiving.

### Script-side audit — configs defined but never produced a result

Cross-checking every `{"name": ...}` in `MODEL_v26/training/*.py` against the 92 captured
runs. Two distinct categories:

**Ran, but never evaluated into a results JSON (7)** — all have a
`DIAGNOSTICS/confusion_collected/` folder with 70 epochs of `results.csv`, and all are
reconstructed in the table above:

| script | configs |
|---|---|
| `run_yolo26_arch2_v6i.py` | `y26_p2_b32`, `y26_p2_dysample`, `y26_p2_dys_snake`, `y26_p2_snake_p3p4`, `y26_levelspec` |
| `run_yolo26_snt_v6i.py` | `y26_snt_t25`, `y26_snt_t50` |

`run_yolo26_arch2_v6i.py` is the single largest evaluation gap in the project — an entire
5-run architecture round whose numbers were only ever read off the console.

**Never ran at all (7)** — no result, no confusion folder, no `results.csv`:

| script | configs | note |
|---|---|---|
| `run_yolo26_overnight_v6i.py` | `y26_3lvl_640_b16`, `y26_3lvl_896_b16`, `y26_p2_896_b16`, `y26_m_640_b16`, `y26_m_p2_640_b16` | **the resolution + capacity grid.** Fully written, requires nothing custom, never executed. The two axes with the largest evidence behind them (896 gave +1.41 mean, 4/4 on v5i) and zero v6i measurements. |
| `run_yolo26_dysample_sweep_v6i.py` | `y26_dys_g16` | groups=16 arm of the DySample sweep — so that sweep is 3 points, not 4 |
| `run_yolo26_loss_isolated_v6i.py`, `run_yolo26_port_v6i.py` | `y26_anchor` | defined in two scripts, run in neither |

Everything else in `MODEL_v26/training/` maps to a captured result.
`run_yolo26_round12_v6i.py` and `run_yolo26_round13_v6i.py` report into the shared
`runs_yolo26_overnight_r1213_v6i` file rather than their own, which is why a filename-based
check flags them; their 7 configs are all present.
