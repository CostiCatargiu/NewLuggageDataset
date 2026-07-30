# Luggage Detection (YOLOv12) — LOSS-FUNCTION Ablation Status & Roadmap

> **Scope:** This document covers the **loss / assigner** experiments only.
> Architecture work is tracked separately in
> `_newapproach/arch_best/ARCH_ABLATION_STATUS.md`.
> **Task:** detect `backpack`, `bag`, `trolley` (512×512, 94% tall, mean AR 2.69, 40% small).
> **Honest summary:** ~60 loss configs across 18 files. **None produced a real win.**
> The clean baseline (plain TAL) ≈ 57.4 mAP50-95; the best mod (NWD) = 57.75 (+0.3 ≈ noise).
> **Last updated:** 2026-07-30

---

## 1. The central problem the loss must solve

```
mAP50    ≈ 83     →  objects are FOUND
mAP50-95 ≈ 57.6   →  boxes are NOT TIGHT
─────────────────────────────────────────
25-point gap      =  LOCALIZATION QUALITY
AR50_small ≈ 96%  →  recall ceiling already hit
```
The loss must improve **box tightness**, not detection. Reweighting *which* samples
matter (SWA, class weights, penalties) or *which* IoU flavor is used cannot fix a
gap that lives in the **box representation / quantization** itself.

Dataset facts that constrain any loss idea:
- **94% tall**, mean AR 2.69, per-class AR bag 2.23 < backpack 2.55 < trolley 2.96
- mean object 33 × 72 px → **width is the hard, thin axis**; height has huge range
- **bag is hardest** (AP50-95 ≈ 0.50 vs trolley ≈ 0.63) and the precision bottleneck

---

## 2. File inventory — two lineages + one clean rebuild

### 2.1 Canonical rebuild (source of truth)
| File | Lines | What it is |
|------|-------|-----------|
| `loss.py` | ~1231 | **Clean v2 rebuild.** Single-source config (`SataLSwaConfig`), fixes v1's silent-failure bugs. Neutral config reproduces stock exactly. **Build new work here.** |

### 2.2 Lineage A — "SATAL-plus" (`v1/`)
| File | Lines | Stage / what it added |
|------|-------|----------------------|
| `loss_satal.py` | ~789 | stock v8 + SATAL switch + FSUS aux hook (earliest) |
| `loss_satal_swa_plus.py` | ~1231 | v1: SATAL + SWA + class weighting + NWD + center loss + adaptive clip |
| `loss_nwd.py` | ~1235 | tuned twin of above (NWD C=6.0, debug on) |
| `loss_satal_swa_plus_v2.py` | ~1230 | v2 (R4–7): MPDIoU, WIoU v3, Focaler-CIoU, QFL, per-anchor-stride fix, repulsion, class-weight modes |
| `loss_satal_swa_plus_v3.py` | ~1103 | v3 (R8): area_weight_mode, per-class boost, center-loss FIX, cls-SWA, bag penalty, **AR-aware TAL** |

### 2.3 Lineage B — "custom ablation" (root)
| File | Lines | Stage / what it added |
|------|-------|----------------------|
| `losscustomoriginal.py` / `losscustomorig.py` | ~1068 | root: SWA + center loss + adaptive clip + optional Shape-aware TAL; +DetectAux/DetectObj heads |
| `loss_v1updated.py` | ~1002 | NEW-1..17: fixed-ref area weight, dfl_small_boost, VFL, NWD blend+anneal, **IARW**, α-IoU, L1-aux, **DFL entropy**, tightness penalty, EIoU/SIoU, **NWD-in-TAL assigner** |
| `loss_v2updated.py` | ~1048 | re-scoped sibling: **Focal-IoU**, **Wise-IoU**, **box-edge jitter** (drops v1updated's EIoU/α-IoU/entropy branches) |

### 2.4 Clipping A/B/C study + baseline (root)
| File | Lines | Role |
|------|-------|------|
| `loss_ORIG.py` | ~832 | **stock Ultralytics baseline** (CIoU+DFL+BCE+TAL), +Aux/Obj heads |
| `loss_cliping.py` | ~937 | SWA + **per-sample** hard clamp |
| `loss_nocliping.py` | ~925 | SWA + **aggregate** clamp (A/B counterpart) |
| `loss_custom.py` | ~937 | near-dup of cliping (alpha log every epoch) |
| `loss_custom_git.py` | ~965 | custom + Shape-aware TAL |
| `loss_satal3.py` | ~1056 | bug-fixed SATAL branch: soft-clip, size-gated NWD, AR importance weight, EIoU |
| `loss2.py` | ~1103 | kitchen-sink (= v3 superset: SATAL+SWA+NWD+all IoU+bag+AR-TAL+repulsion) |
| `loss3.py` | ~900 | cleanly-gated EIoU/SIoU + width-gated NWD + width-aware boost |

> ⚠️ **Latent bug** shared by `loss_cliping/custom/custom_git/losscustomorig`:
> `calculate_segmentation_loss` returns `loss / fg_mask.sum` (missing `()`).
> `loss_satal3.py` fixed it (BUGFIX 3). Irrelevant for detection-only runs, but fix
> before any seg work.

---

## 3. Measured results (60 runs, `luggagenew_results/*.json`)

All on `test_full_dataset`. Sorted by mAP50-95. **Clean baseline (`r*_anchor`) ≈ 57.43.**

### 3.1 Top of the table
| Run | mAP50 | **mAP50-95** | bag AP50-95 | Mechanism |
|-----|-------|--------------|-------------|-----------|
| `r10_nwd_fixedc` | 82.96 | **57.75** | 49.84 | **NWD, fixed C — best overall (+0.3)** |
| `r10_dfl_entropy` | 82.61 | 57.71 | **49.96** | **DFL entropy — best bag** |
| `r9b_nwd_adapt` | 82.75 | 57.45 | 49.65 | NWD, adaptive C |
| `r9b_anchor`/`r10_anchor` | 82.70 | 57.43 | 49.29 | **clean baseline (plain TAL)** |
| `r9b_dfl_gated` | 82.52 | 57.25 | 49.24 | gated DFL |

### 3.2 The plateau body (representative)
| Run | mAP50-95 | Mechanism |
|-----|----------|-----------|
| `r8_area_sqrt` | 56.71 | sqrt area-weighting |
| `r8_artal` | 56.52 | AR-aware TAL (**highest mAP50 = 83.49**, but not tighter) |
| `r4_wiou` | 56.43 | WIoU v3 |
| `r4_mpdiou` | 56.43 | MPDIoU |
| `r4_focaler` | 56.37 | Focaler-CIoU |
| `r8_bag_penalty` | 56.35 | bag asymmetric penalty |
| `r8_cls_swa` | 56.13 | cls-SWA ranking boost |
| `v6_tal_best` | 55.52 | TAL topk13/α0.7/β4 |

### 3.3 The clear LOSERS
| Run | mAP50-95 | Mechanism |
|-----|----------|-----------|
| `r3_satal_r3` | **54.57** | **SATAL — −2.8pt vs plain TAL** |
| `r4_satal_mpdiou` | 54.01 | SATAL + MPDIoU |
| `r7_stack` | 51.95 | stacked mods — collapsed |

---

## 4. What we learned — why NOTHING worked

1. **The loss space plateaued, just like the arch space.** Best mod (57.75) beats the
   clean baseline (57.43) by ~+0.3 — within seed noise.
2. **🔴 SATAL actively HURTS on luggage** (−2.8pt). It is the assigner locked ON in the
   arch experiments — a cross-project inconsistency to be aware of (arch doc used
   SATAL-on; the loss data says SATAL-off is better).
3. **NWD is the only mildly useful mod** — tops the table, purpose-built for small
   objects (40% of data). Keep it; it is the one thing to carry forward.
4. **DFL-entropy gives the best bag AP** — a hint that sharpening the box *distribution*
   (not reweighting it) is the productive direction.
5. **Every IoU variant loses** (EIoU/SIoU/MPDIoU/WIoU/Focaler/α-IoU/Focal-IoU: 56.4–56.5).
   CIoU is already fine; swapping the IoU flavor is a dead end.
6. **AR-aware TAL raises mAP50 but not mAP50-95** — it *finds* more (assignment) but
   doesn't *tighten* boxes. Confirms the bottleneck is regression precision, not
   assignment.
7. **Class-specific tricks are inert** — bag penalty, repulsion, cls-SWA, class weighting
   all flat/negative. Your own comments: "QFL/VFL failed", "repulsion inert",
   "linear class weighting failed".
8. **Reweighting is exhausted.** SWA, area-weight modes, per-class boosts, IARW, clipping,
   center loss, box jitter — all reweight/reshape the *same scalar IoU/DFL signal* and all
   plateau.

**Root cause (the one-line diagnosis):**
> Every one of the 60 configs changes *which samples matter* or *which IoU flavor* is used.
> **None changes the box REPRESENTATION.** The 25-pt gap is a box-tightness /
> quantization problem, and stock DFL quantizes all 4 edges into **16 identical bins** —
> too coarse for tall-object height, wasteful for thin-object width. That representation
> was never touched.

---

## 5. The genuinely UNTRIED axis — box representation

The only mechanism no file modifies is the **DFL bin structure itself**.

### 5.1 AR-DFL (Aspect-Ratio-aware DFL) — the primary candidate
- **Idea:** asymmetric bins per edge. Height (top+bottom) gets **more/wider bins**
  (24, higher `reg_max`), width (left+right) gets **fewer** (8). Or per-edge
  stride-normalized range so each edge's bins match its true value distribution.
- **Why it can work when everything else failed:** it attacks *quantization granularity*
  — orthogonal to every reweighting scheme tried. It changes the resolution at which the
  box can be expressed, directly targeting tall-box height precision (the 25-pt gap).
- **Confirmed novel:** across all 18 files the closest are `r10_dfl_entropy` (sharpen the
  distribution) and `r8_artal` (AR in the *assigner*). **Asymmetric-bin DFL is untested.**
- **Stack on NWD:** build on top of `r10_nwd_fixedc` (the best mod), not instead of it.
- **Where:** `loss.py::DFLoss` (line ~445) + `block.py::DFL` integral (arch side must
  match `reg_max`, since Detect builds `reg_max*4` regression channels).

### 5.2 Secondary untried ideas (lower priority)
- **Per-edge DFL range** (asymmetric range instead of asymmetric bin count) — cheaper,
  no arch change, worth a quick control.
- **Explicit height/width decoupled regression head** — architectural, larger change.

---

## 6. What to test next — priority order

### Priority 0 — Diagnostic: prove where the gap lives  *[cheap, do first]*
- Measure per-edge DFL residual (left/right vs top/bottom) and IoU-vs-AR on the current
  best model. If height error ≫ width error, AR-DFL is justified with evidence (and it
  becomes a paper figure).

### Priority 1 — Build & test **AR-DFL** (§5.1)  *[the one real idea]*
- Implement asymmetric-bin DFL as a toggle in `loss.py` (keep neutral default).
- Ablation: baseline → NWD → NWD+AR-DFL(24/8) → NWD+AR-DFL(range-norm).
- Success = NWD+AR-DFL > NWD (57.75) with the gain concentrated in mAP75/mAP50-95.

### Priority 2 — Lock the canonical config  *[methodology]*
- Adopt **plain TAL + NWD(fixed C)** as the single baseline for ALL future runs
  (loss AND arch), retiring SATAL. Removes the arch/loss inconsistency.

### Priority 3 — Multi-seed the AR-DFL result  *[for the paper]*
- Seeds [0,42,123], mean ± std, paired test — the plateau is a noise-width band, so any
  claimed win must clear significance.

### Priority 4 — Pair with the arch novelty
- Combine AR-DFL (regression axis) with ARSC (geometry axis, arch doc). Orthogonal
  bottlenecks → the two-legged contribution for the paper.

---

## 7. Do-NOT-retry list (already exhausted)

SATAL · SWA · every IoU variant (EIoU/SIoU/MPDIoU/WIoU/Focaler/α-IoU/Focal-IoU) ·
VFL/QFL · class weighting · bag penalty · repulsion · cls-SWA · center loss ·
loss clipping (any granularity) · box jitter · IARW · area-weight modes · AR-aware TAL
(for mAP50-95). All measured, all flat-or-negative. Keep only **NWD** and the hint from
**DFL-entropy**.

---

## 8. Key file map

| Purpose | Path |
|---------|------|
| Canonical loss (build here) | `_newapproach/loss_function/loss.py` |
| SATAL-plus lineage | `_newapproach/loss_function/v1/loss_satal_swa_plus_v{1,2,3}.py` |
| Custom-ablation lineage | `_newapproach/loss_function/v1/loss_v{1,2}updated.py` |
| Results (60 runs) | `runs/luggagenew_results/runs_luggage_round*__test_full_dataset.json`, `runs_newluggage_r{9,9b,10}__test_full_dataset.json` |
| Arch counterpart doc | `_newapproach/arch_best/ARCH_ABLATION_STATUS.md` |
