# Methods tried so far — complete inventory

Every mechanism implemented across the 19 loss files, with the config key that
activates it, the file(s) that actually read that key, and the measured result.

**Δ is computed against that run's OWN round anchor**, not against a global
baseline. This differs from `LOSS_ABLATION_STATUS.md`, which sorted all rounds
against 57.43 and therefore mislabelled several in-round wins as losses.
Rounds with no anchor run are marked `no anchor` — those numbers are not
comparable to anything.

Round anchors: `r3_swa_anchor` 56.44 · `r4_baseline_clean` 56.39 ·
`r8_anchor` 56.25 · `r9_anchor2` / `r9b_anchor` / `r10_anchor` 57.43 (one run,
reported three times).

---

## 1. Assigner (which anchors get assigned to which GT)

| Method | What it changes | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|---|
| Stock TAL | `score^α · iou^β`, topk=10 | `tal_topk/alpha/beta` | all | `r10_anchor` 57.43 | — | baseline |
| TAL retuning | topk 13, α 0.7, β 4 | same | all | `v6_tal_best` 55.52 | no anchor | worst of sweep |
| TAL strict | β 7 | same | all | `r2_tal_strict` 56.78 | no anchor | — |
| **SATAL** | per-scale α/β/topk by GT area | `use_satal` | `loss.py`, `loss2`, `v1/loss_satal*` (external `utils/satal.py`) | `r3_satal_r3` 54.57 | **−1.87** | ❌ strongly negative |
| SATAL + MPDIoU | " | " | " | `r4_satal_mpdiou` 54.01 | **−2.38** | ❌ |
| SATAL + WIoU | " | " | " | `r4_satal_wiou` 53.78 | **−2.61** | ❌ |
| **AR-aware TAL** | β relaxed per-GT as h/w rises | `use_artal` | `loss2`, `v1/..._v3` | `r8_artal` 56.52 | **+0.27** | ⚠️ best mAP50 (83.49); no mAP50-95 gain |
| Shape-aware TAL | external assigner | `use_shape_tal` | `loss_custom_git`, `losscustomorig`, `loss_satal3` | — | — | 🔴 module absent; **silently falls back to stock TAL** |
| NWD-in-TAL | NWD replaces IoU in the align metric | `tal_nwd*` | `v1/loss_v1updated` only | r11 runs | not in CSV | — |

> ⚠️ Every SATAL run also moved `tal_alpha` 0.5→0.6, `tal_beta` 6.0→5.0,
> `tal_topk` 10→12 and two `satal_*` params — the same 6-variable bundle in both
> r3 and r4. The −2.4 pt is reproducible across rounds, but which of the six
> causes it is unresolved. `tal_beta` 6→5 is an equally plausible culprit.

---

## 2. Sample weighting (which samples matter) — the largest family

| Method | What it changes | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|---|
| SWA (alpha blend) | `w = α·area_w + (1−α)·score_w` | `alpha_start/end/min/max` | 14 files | `swa1_09_03` 57.48 | no anchor | flat |
| SWA boost sweep | boost 1.2–3.0, px 36/48 | `small_obj_boost`, `small_obj_px` | 15 files | `swa1c_b200_px48` 57.22 | no anchor | flat |
| SWA v2 boost | " | " | " | `swa2_b150` 57.47 | no anchor | flat |
| Area weight `1/area` (legacy) | batch-max normalized, ~400:1 spread | `area_mode=legacy` | 9 files | — | — | 🔴 carries bug **B1** |
| Area weight fixed-ref | deterministic, batch-independent | `area_mode=fixed` | `v1/loss_v1updated/v2updated` | `r9_area_fixed` 57.03 | −0.40 | ❌ |
| **Area weight sqrt** | `sqrt` area mode | `area_weight_mode=sqrt` | `loss2`, `v1/..._v3` | `r8_area_sqrt` 56.71 | **+0.46** | ⭐ largest single-variable in-round gain |
| Width-adaptive weight | weight keyed on box **width**, not area | `small_obj_width_thresh_px` | `loss3` only | `run4_metric_widthboost` 57.19 | no anchor | — |
| **IARW** | `1 + γ(1−IoU)` regression boost | `iarw_gamma` | `v1/loss_v1updated/v2updated` | `r9b_iarw_lo` 57.10 | −0.33 | ❌ |
| Per-class boost | boost per class id | `small_obj_boost_bag` etc. | `loss2`, `v1/..._v3` | `r8_boost_bag` 56.35 | +0.10 | ➖ flat |

---

## 3. Box regression metric (which IoU flavour)

| Method | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|
| CIoU (stock) | `box_loss_type=ciou` | all | `r4_baseline_clean` 56.39 | — | baseline |
| **MPDIoU** | `=mpdiou` | `loss.py`, `loss2`, `v1/..._v2/_v3` | `r4_mpdiou` 56.43 | **+0.04** | ➖ flat |
| **WIoU v3** | `=wiou` | " | `r4_wiou` 56.43 | **+0.04** | ➖ flat |
| **Focaler-CIoU** | `=focaler` | " | `r4_focaler` 56.37 | **−0.02** | ➖ flat |
| EIoU | `=eiou` / `_bbox_eiou_loss` | `loss.py`, `loss3`, `loss_satal3`, `v1/loss_v1updated` | `run1_eiou` 57.27 | no anchor | — |
| SIoU | `=siou` / `_bbox_siou_loss` | same | `run2_siou` 57.29 | no anchor | — |
| α-IoU | `alpha_iou` | `v1/loss_v1updated` | `r10_alpha_iou` 56.90 | −0.53 | ❌ |
| Inner-IoU | — | early ablation | `run5_inner_iou` 56.90 | no anchor | — |
| Focal-IoU / Wise-IoU | `wise_iou` | `v1/loss_v2updated` | — | — | — |

> 🔴 **Correction to the status doc.** "Every IoU variant loses (56.4–56.5)" is an
> artifact of comparing r4 runs against the r10 anchor (57.43). Against their own
> anchor (`r4_baseline_clean` 56.39) MPDIoU, WIoU and Focaler are **+0.04, +0.04,
> −0.02** — flat, not losing. They are also 2-variable (each also flips
> `use_class_weighting` on).

---

## 4. NWD family — four incompatible implementations

| Method | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|
| NWD blend (namespace A) | `nwd_C`, `nwd_mode`, `nwd_weight` | `loss2`, `v1/loss_nwd`, `v1/..._swa_plus{,_v2,_v3}` | `run2_nwd_blend` 56.91 | no anchor | — |
| NWD small-only (A) | `use_nwd` + `nwd_small_threshold` | same | `r4_nwd_small` 55.71 | **−0.68** | ❌ |
| NWD ratio + adaptive C (**B**) | `nwd_ratio`, `nwd_c_adaptive`, `nwd_anneal` | `v1/loss_v1updated/v2updated` | **`r10_nwd_fixedc` 57.75** | **+0.32** | ⭐ best measured |
| NWD ratio, fixed C (B) | `nwd_ratio` only | same | `r9b_nwd_adapt` 57.45 | +0.02 | ➖ |
| NWD gated (C) | `nwd_const`, `nwd_gate_px` | `loss_satal3` | — | — | — |
| NWD pixel-space (D) | `nwd_c_px`, `nwd_small_width_px` | `loss.py`, `loss_ardfl` | **never run** | — | this is what `_BASE` uses |
| NWD width-gated (E) | `nwd_width_gate_px` | `loss3` | — | — | — |

> 🔴 **The labels on the two best runs are swapped.** `r10_nwd_fixedc` has
> `nwd_c_adaptive: 1` — adaptive C. `r9b_nwd_adapt` has it at 0 — fixed C. They
> differ by that key alone.
>
> 🔴 **`_BASE` in `run_ardfl_ablation.py` is namespace D**, not B. What actually
> won was: small-objects-only (< 48²px), `c = 0.5·√area` per anchor, ratio
> `0.3 · smallness · anneal(1.0→0.1)`. `_BASE` uses fixed C = 12 px at constant
> weight 0.5 on **every** box — a different mechanism, 10–50× stronger.

---

## 5. DFL / box representation

| Method | What it changes | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|---|
| Stock DFL | 16 bins, all 4 edges identical | `reg_max` | **all 19 files, one implementation** | — | — | never modified |
| DFL decode | `softmax · arange(16)` expectation | — | **all 19 files, identical** | — | — | never modified |
| DFL small boost | multiply DFL for small objects | `dfl_small_boost` | `v1/loss_v1updated/v2updated` | `r9_dflboost` 57.10 | −0.33 | ❌ |
| DFL IoU-gated boost | boost only low-IoU anchors | `dfl_iou_gated` | same | `r9b_dfl_gated` 57.25 | −0.18 | ❌ (3-var) |
| **DFL entropy** | sharpen the bin distribution | `dfl_entropy_weight` | `v1/loss_v1updated` | **`r10_dfl_entropy` 57.71** | **+0.28** | ⭐ best bag AP (49.96) |
| **AR-DFL** | per-edge DFL weights (h vs w) | `use_ardfl`, `ardfl_h_weight` … | `loss.py`, `loss_ardfl` | **never run** | — | only genuinely new mechanism |
| AR-DFL entropy | entropy on height edges only | `ardfl_entropy` | same | never run | — | 🔴 not the same maths as `dfl_entropy_weight` (bug B5) |

---

## 6. Classification

| Method | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|
| BCE (stock) | — | all | — | — | baseline |
| VFL | `use_vfl` | `loss.py`, `loss3`, `v1/..._v2/_v3` | — | — | reported failed |
| QFL | `cls_mode=qfl` | `v1/..._v2/_v3` | — | — | reported failed |
| Class weighting (linear) | `use_class_weighting` | `loss.py`, `loss2`, `v1/..._v2/_v3` | `run5_class_balance` 57.50 | no anchor | — |
| Class weighting (modes) | `class_weight_mode` | `loss2`, `v1/..._v3` | `run1_class_balance` 57.25 | no anchor | — |
| cls-SWA | `use_cls_swa` | `loss2`, `v1/..._v3` | `r8_cls_swa` 56.13 | −0.12 | ❌ |
| Small-object cls boost | `small_obj_cls_boost` | `loss3` | — | — | — |

---

## 7. Extra loss terms / regularizers

| Method | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|
| Center loss | `center_loss_weight_init` | 12 files, 4 versions | `r8_center_crowd` 56.39 | +0.14 (4-var) | ➖ |
| Repulsion | `use_repulsion` | `loss2`, `v1/..._v2/_v3` | `r7_wiou_rep` 56.21 | no anchor | reported inert |
| Bag asymmetric penalty | `use_bag_penalty` | `loss2`, `v1/..._v3` | `r8_bag_penalty` 56.35 | +0.10 | ➖ flat |
| Tightness penalty | `tightness_gamma` | `v1/loss_v1updated` | `r10_tightness` 57.22 | −0.21 | ❌ |
| L1 aux | `l1_aux_weight` | `v1/loss_v1updated` | `r10_l1_smooth` 57.23 | −0.20 | ❌ |
| Box-edge jitter | `box_jitter` | `v1/loss_v2updated` | — | — | — |
| AR penalty | — | early ablation | `run4_ar_penalty` 56.84 | no anchor | — |
| FSUS aux hook | — | `v1/loss_satal` | — | — | — |
| DetectAux / DetectObj heads | `aux_weight`, `obj_weight` | 10 files | — | — | — |

---

## 8. Optimization hygiene

| Method | Config key | Implemented in | Best run | Δ | Verdict |
|---|---|---|---|---|---|
| Per-sample clip | `iou_clip`, `dfl_clip` | `loss_cliping`, `loss_custom` | `v6_clip_loose` 56.72 | no anchor | flat |
| Aggregate clip | same | `loss_nocliping` | `v6_clip_medium` 56.60 | no anchor | flat |
| Soft clip | `_soft_clip` | `loss_satal3` | — | — | — |
| Stacked everything | — | `loss2` | `r7_stack` 51.95 | no anchor | ❌ collapse |

---

## 9. Summary — what the corrected numbers say

Ranked by Δ against own-round anchor, single-variable runs only:

| rank | run | Δ | mechanism |
|---|---|---|---|
| 1 | `r8_area_sqrt` | **+0.46** | sqrt area weighting |
| 2 | `r10_nwd_fixedc` | +0.32 | NWD, adaptive C (4-var) |
| 3 | `r10_dfl_entropy` | **+0.28** | DFL entropy sharpening |
| 4 | `r8_artal` | **+0.27** | AR-aware TAL |
| 5 | `r8_boost_bag` / `r8_bag_penalty` | +0.10 | per-class |
| … | `r4_mpdiou` / `r4_wiou` | +0.04 | IoU variants (flat, not losing) |
| ↓ | `r9b_iarw`, `r10_alpha_iou` | −0.5 | ❌ |
| ↓↓ | SATAL bundle | −1.9 … −2.6 | ❌ reproducible across rounds |

**Caveat that dominates all of the above:** there are no seed replicates anywhere
(`r9_anchor2`, `r9b_anchor`, `r10_anchor` are one run reported three times), and
the same run under two eval sources differs by up to 1.11 pt. Every Δ in the top
half of that table is smaller than the measurement uncertainty. None of the
"wins" are established.
