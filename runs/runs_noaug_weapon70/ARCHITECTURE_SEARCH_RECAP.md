# Architecture Search — Complete Recap

**Task:** weapon detection, 4 classes (knife, long_gun, pistol, **other**)
**Dataset:** 70% ablation split (~13k train / ~2.8k val images)
**Base model:** YOLOv12s (width 0.5), 640×640, 80 epochs (120–150 for runs with large fresh structure), seed 0
**Protocol:** append-only modules, identity-at-init (zero-gated, γ=0) where possible, pretrained transfer via Detect-remap, **default TAL** on architecture runs to isolate the architecture effect.

---

## 1. The headline result

**Architecture is not the lever on this dataset. Loss/assignment design is.**

| Lever | Best result (test mAP50) | Notes |
|---|---|---|
| **Loss / TAL tuning** (architecture frozen) | **80.45%** (`v5_topk15_beta3`) | The only lever that beat baseline; +2.0 pts |
| **Architecture** (TAL frozen at default) | 79.40% (`r11_widefuse`) | +0.95 over *its* baseline; never beat loss tuning |
| Baseline (`original_loss_70`) | 78.45% | — |

Across **~50 architecture variants over ~28 rounds**, not one cleanly beat plain loss tuning, and the differences between architectures are the same size as the run-to-run noise.

---

## 2. The diagnosis (the most important finding)

The entire dataset ceiling is the **"other"** class.

- The three **weapon classes are saturated** at ~86–88% AP50. No architecture moves them — there's no room.
- **"other" sits at ~51–52% AP50** — the whole headline number is essentially `(≈87×3 + other)/4`, so every architecture difference is "other" moving and nothing else.
- **"other" has high recall (~0.84) but low precision/AP (~0.51).** The detector *finds* the objects and *mis-scores* them. The recall-vs-precision gap is ~0.33 for "other" vs ~0.09 for the weapon classes.

**Implication:** the bottleneck is **classification / ranking / confidence**, not localization or "finding." This is the signature of a **label-quality problem** in the heterogeneous catch-all "other" class — and it explains why feature-side architecture changes never helped.

Every architecture also *hurt* small-object detection, concentrated in small-"other" (baseline small-"other" AP50 ≈ 38.6; most architectures dropped it to 23–33).

---

## 3. Every architectural lever tested — and the result

| Round(s) | Lever | Mechanism | Result |
|---|---|---|---|
| 1–16 | **Receptive field / context** | zero-gated large-kernel attention at P4-BU (ZGLSKA family, ~20 variants) | Best arch (`r11_widefuse` 79.40), but plateaued; flat vs loss tuning |
| 1–16 | **P3 head mods** | attention / SE / SPP / LSKA / dual-path at P3 | All hurt small objects |
| 1–16 | **Backbone depth, P5 attention** | more reps; P5→A2C2f | Flat to worse |
| 17 | **Spatial routing** | per-pixel softmax over multi-RF branches (SelectFuse) | No edge over static fusion |
| 18 | **Classifier capacity** | deeper / wider cls tower | No effect on "other" |
| 18 | **Detection scale (P2)** | stride-4 head | Recall ↑, AP flat, overall regressed |
| 19 | **Neck topology** | BiFPN-style weighted fusion (WeightedConcat) | Flat / negative |
| — | **Input resolution** | 768px | Did not help (tested separately) |
| 20–22 | **Train-only aux head** | deep supervision, dropped at inference | Nominally best on val once (78.77), but noise-band; "dose-response" dissolved on a 3rd point + validation |
| 24 | **Decoupled cls head** | box & cls read separate features | **Reliable precision gain; top-2 on validation** ← the one consistent signal |
| 24 | **Objectness branch** | FG/BG head, score=σ(cls+obj) | Best small-"other" AP in project (41.23) but over-suppressed → overall regressed |
| 26 | **Softened objectness** | β<1 reweighting | Fixed over-suppression, but lands ≈ baseline |
| 25–27 | **Decoupled on widefuse / dual-neck** | best backbone + decoupling, separate cls pyramid | `r25` best-on-validation (78.98); dual-neck (120 ep) |
| 28 | **Cosine / prototype classifier** | angular scoring vs learnable prototypes | **Underperformed** — unstable early, plateaued ~77.3 val, below linear |

---

## 4. The variance problem (why no result is trustworthy yet)

The run-to-run noise is **as large as the entire architecture signal (~±1 point)**, coming from three sources:

1. **Seed** (random init of fresh layers, data shuffle, augmentation) — never measured (no multi-seed runs).
2. **Test-vs-val split luck** — gaps of ±1pp. The flashiest test numbers are the luckiest splits:
   - `r11_widefuse`: val 78.38 → test 79.40 (**+1.02**, lucky)
   - `r21_widefuse_aux_w50`: val 78.32 → test 79.57 (**+1.25**, lucky)
   - `r25_widefuse_decoupled`: val 78.98 → test 78.40 (**−0.58**, unlucky)
3. **Batch size** drifted between runs (16–58), driven by memory — a BatchNorm confound that systematically penalized the heavy/low-batch runs.

**Consequence:** the candidates swap rank between validation and test. That pattern *is* what "it's all noise" looks like. **No architecture difference has been confirmed with a seed-check**, so none is statistically established.

---

## 5. Best models

- **Overall best (any method):** `v5_topk15_beta3` = 80.45 test (but split-inflated). The most *robust* (val+test agree, no lucky split): **`v5_tal07`** = 80.02 test / 79.54 val. **Report this one.**
- **Best architecture (test mAP50):** `r11_widefuse` = 79.40 (lucky split).
- **Best architecture (validation, the fair metric):** `r25_widefuse_decoupled` = 78.98.
- **Most consistent architectural effect:** the **decoupled cls head** — reliably higher precision on both splits; the only positive that didn't depend on which split you looked at.

---

## 6. What the search established (the real, defensible conclusions)

1. **Loss-function / label-assignment (TAL) design dominates architecture on this dataset.** TAL tuning gave +2.0 pts with zero architecture change; nothing architectural matched it.
2. **The "other" class is the ceiling, and it behaves like a data/label problem** — found but mis-scored (high recall, low precision). Every architectural axis was tested — receptive field, routing, classifier capacity, detection scale, resolution, neck topology, deep supervision, ranking heads (decoupled / objectness), and the **scoring mechanism itself** (cosine) — and none moved it past noise. Changing *how scores are computed* (cosine) made it *worse*, which is strong evidence the classifier mechanism is not the bottleneck.
3. **The one reproducible architectural effect is precision** (decoupled cls head). A defensible, useful sub-result even though it doesn't raise overall mAP.

This is a complete, publishable negative/ablation result: *a systematic architecture search showed loss-function design dominates architecture for this task, with the residual ceiling located in the "other"-class annotations.*

---

## 7. Recommended next steps (none are more architecture)

1. **Seed-check the finalists** — `r25_widefuse_decoupled`, `r11_widefuse`, baseline, and the best TAL config, **3 seeds each, same batch & split.** This is the only way to confirm whether any architecture difference (or the decoupled precision effect) is real. It will most likely show the candidates overlap → confirming the noise-band conclusion.
2. **"other"-class error analysis** — one hour examining the high-confidence false positives and missed/low-scored boxes. This determines whether the ceiling is fixable at all. If the labels are noisy/ambiguous (very likely given the recall-vs-precision signature), **no architecture or loss change will fix it.**
3. **Fix the batch confound** — for any comparison that matters, hold the physical batch constant.
4. **(Optional) arch × best-TAL** — `r11_widefuse` / `r25_decoupled` × the `topk15_beta3` TAL has never been combined; it's the only untested path that could plausibly clear 80.45, but expect partial-or-no stacking.

---

*Bottom line: the architecture is at its ceiling. The remaining gain — if any exists — is in the loss/assignment side (proven) and the "other"-class labels (the diagnosed wall), not in the network structure.*
