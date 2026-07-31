# Complete mechanism inventory — code-verified

Every mechanism implemented across all 20 loss files, root and `v1/`, identified by
the config key the code actually reads (`getattr(hyp, ...)` / `g("...")` extraction,
not docstrings). Status column says whether it was ever measured.

**Method:** config-key extraction per file + reading each distinct implementation
body. Where this contradicts `LOSS_ABLATION_STATUS.md` / `METHODS_TABLE.md` /
`LOSS_CODE_AUDIT.md`, the discrepancy is flagged in §5.

Legend: ✅ run · ⚪ never run · 🔴 inert or silently disabled · ⚠️ ambiguous provenance

---

## 1. File inventory (actual line counts + config surface)

### Root

| file | lines | keys | role |
|---|---|---|---|
| `loss_ORIG.py` | 831 | 0 | stock Ultralytics baseline + Aux/Obj heads |
| `loss_nocliping.py` | 925 | 17 | SWA + **aggregate** clamp |
| `loss_cliping.py` | 936 | 17 | SWA + **per-sample** clamp |
| `loss_custom.py` | 936 | 17 | near-dup of cliping |
| `loss_custom_git.py` | 964 | 20 | custom + shape-TAL |
| `losscustomorig.py` | 1068 | 20 | root SWA/center/clip + shape-TAL |
| `loss_satal3.py` | 1056 | 28 | orphan branch: soft-clip, gated NWD (ns C), AR weight, EIoU |
| `loss3.py` | 899 | 12 | orphan branch: width-gated NWD (ns E), width-aware boost |
| `loss_ardfl.py` | 1238 | 44 | **stale fork of `loss.py`** frozen at the AR-DFL stage |
| `loss2.py` | 1722 | 58 | kitchen sink — byte-identical to `v1/loss_satal_swa_plus_v3.py` |
| `loss.py` | **1985** | **65** | canonical rebuild — A-DFL, PEU, LBA, EDGEW live here |

### v1/

| file | lines | keys | role |
|---|---|---|---|
| `loss_satal.py` | 789 | 11 | earliest: stock + SATAL switch + FSUS hook only |
| `loss_satal_swa_plus.py` | 1231 | 28 | +NWD ns A, SWA, center loss, clipping |
| `loss_nwd.py` | 1235 | 28 | **identical key set** to the above; differs in hardcoded C |
| `loss_satal_swa_plus_v2.py` | 1521 | 43 | +IoU variants, QFL, class weighting, repulsion |
| `loss_satal_swa_plus_v3.py` | 1722 | 58 | +area modes, AR-TAL, per-class, bag penalty, cls-SWA |
| `loss_v1updated.py` | 1765 | 53 | separate lineage: NWD ns B, DFL entropy, IARW, L1 family |
| `loss_v2updated.py` | 1572 | 36 | fork of v1updated that **removes** entropy/α-IoU/tightness/L1 |
| `losscustomoriginal.py` | 1068 | 20 | byte-identical to `losscustomorig.py` |

### ⚠️ Config-surface collisions — files no `args_yaml` can distinguish

1. `loss_cliping.py` ≡ `loss_nocliping.py` ≡ `loss_custom.py` — **identical 17 keys**,
   three different clamp behaviours. The whole clipping A/B study is unidentifiable
   from saved configs.
2. `loss_custom_git.py` ≡ `losscustomorig.py` — identical 20 keys.
3. `v1/loss_nwd.py` ≡ `v1/loss_satal_swa_plus.py` — identical 28 keys, different C.
4. `loss2.py` ≡ `v1/loss_satal_swa_plus_v3.py` — md5 `ef48df86`.
5. `losscustomorig.py` ≡ `v1/losscustomoriginal.py` — md5 `f672417c`.

---

## 2. Mechanisms by axis

### 2.1 Assigner

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 1 | Stock TAL | `tal_topk`, `tal_alpha`, `tal_beta` | all | ✅ baseline |
| 2 | SATAL (area-gated α/β/topk) | `use_satal`, `satal_alpha_small/large`, `satal_beta_small/large`, `satal_small_area`, `satal_large_area`, `satal_topk_factor` | `loss.py`, `loss2`, `loss_ardfl`, `v1/loss_satal*`, `v1/loss_nwd` | ✅ −1.9…−2.6 |
| 3 | Shape-aware TAL | `use_shape_tal`, `shape_gamma`, `shape_min` | `loss_custom_git`, `losscustomorig`, `loss_satal3`, `v1/losscustomoriginal` | 🔴 class exists at `_newapproach/loss_v3_luggage_fixed.py:314` but **not on the import path** these files use → silent stock TAL |
| 4 | AR-aware TAL | `use_artal`, `artal_ar_thresh`, `artal_ar_scale`, `artal_beta_relax` | `loss2.py:712`, `v1/..._v3` | ✅ +0.27 |
| 5 | NWD-in-TAL | `tal_nwd`, `tal_nwd_c`, `tal_nwd_area`, `tal_nwd_ratio` | `v1/loss_v1updated.py:179` | ⚠️ r11, not in results CSV |
| 6 | **LBA** (level prior) | `use_lba`, `lba_strength`, `lba_ref_cells`, `lba_sigma`, `lba_log` | `loss.py:857,920` | ⚪ never run |

### 2.2 Sample weighting

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 7 | SWA additive blend | `alpha_start/end/min/max` | 14 files | ✅ flat |
| 8 | SWA multiplicative | `swa_mode="scale"` | `loss.py`, `loss_ardfl` | ⚪ never run |
| 9 | Small-object boost | `small_obj_boost`, `small_obj_px` | most | ✅ flat |
| 10 | Bounded size weight | `swa_boost`, `swa_size_axis`, `swa_width_thresh_px`, `swa_area_thresh_px2` | `loss.py`, `loss_ardfl` | ⚪ never run — replaces the 400:1 legacy weight |
| 11 | Boost shaping | `swa_boost_power`, `swa_smooth` | `loss2`, `v1/..._v2/_v3` | ✅ |
| 12 | Area mode legacy/fixed | `area_mode`, `area_ref_px`, `area_gamma`, `area_w_cap` | `v1/loss_v{1,2}updated` | ✅ −0.40 |
| 13 | Area weight inv/sqrt/log | `area_weight_mode` | `loss2`, `v1/..._v3` | ✅ **+0.46** best |
| 14 | Width-adaptive weight | `small_obj_width_thresh_px` | `loss3` only | ⚠️ no anchor |
| 15 | IARW | `iarw_gamma` | `v1/loss_v{1,2}updated` | ✅ −0.33 |
| 16 | Per-class boost | `small_obj_boost_bag/backpack/trolley` | `loss2`, `v1/..._v3` | ✅ +0.10 |
| 17 | cls-SWA | `use_cls_swa`, `cls_swa_boost` | `loss2`, `v1/..._v3` | ✅ −0.12 |
| 18 | Weight renormalisation | `weight_renorm` | `v1/loss_v{1,2}updated` | ✅ |
| 19 | AR importance weight | `ar_lambda`, `ar_ref`, `ar_cap` | `loss_satal3` only | ⚪ orphan |

### 2.3 Box regression metric

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 20 | CIoU | `box_loss_type=ciou` | all | ✅ baseline |
| 21 | MPDIoU | `=mpdiou` | `loss.py`, `loss2`, `loss_ardfl`, `v1/..._v2/_v3` | ✅ +0.04 |
| 22 | WIoU v3 | `=wiou`, `wiou_alpha`, `wiou_delta`, `wiou_momentum` | same | ✅ +0.04 |
| 23 | Focaler-CIoU | `=focaler`, `focaler_d`, `focaler_u` | same | ✅ −0.02 |
| 24 | EIoU | `box_metric=eiou` / `iou_type` | `loss3`, `loss_satal3`, `v1/loss_v1updated` | ⚠️ no anchor |
| 25 | SIoU | same | same | ⚠️ no anchor |
| 26 | α-IoU | `alpha_iou` | `v1/loss_v1updated` | ✅ −0.53 |
| 27 | Focal-IoU | `focal_iou_gamma` | `v1/loss_v2updated` | ⚪ never run |
| 28 | Wise-IoU | `wise_iou` | `v1/loss_v2updated` | ⚪ never run |
| 29 | Inner-IoU | `iou_ratio` | `loss3` | ⚠️ no anchor |
| 30 | NWD **ns A** | `use_nwd`, `nwd_mode`, `nwd_weight`, `nwd_small_threshold` | `loss2`, `v1/loss_nwd`, `v1/..._swa_plus{,_v2,_v3}` | ✅ −0.68 |
| 31 | NWD **ns B** | `nwd_ratio`, `nwd_c`, `nwd_adaptive`, `nwd_anneal`, `nwd_anneal_min`, `nwd_c_adaptive`, `nwd_c_k` | `v1/loss_v{1,2}updated` | ✅ **+0.32** best |
| 32 | NWD **ns C** | `nwd_const`, `nwd_gate_px` | `loss_satal3` | ⚪ orphan |
| 33 | NWD **ns D** | `nwd_c_px`, `nwd_small_width_px`, `nwd_debug` | `loss.py`, `loss_ardfl` | ⚪ never run — this is what `_BASE` uses |
| 34 | NWD **ns E** | `nwd_c`, `nwd_width_gate_px` | `loss3` | ⚪ orphan |

### 2.4 DFL / box representation

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 35 | Stock DFL | `reg_max` | all 20 files, **one implementation** | never modified |
| 36 | DFL small boost | `dfl_small_boost` | `v1/loss_v{1,2}updated` | ✅ −0.33 |
| 37 | DFL IoU-gated boost | `dfl_iou_gated` | same | ✅ −0.18 |
| 38 | **DFL entropy** | `dfl_entropy_weight`, `dfl_entropy_small_only` | `v1/loss_v1updated.py:803` | ✅ **+0.28**, best bag AP |
| 39 | AR-DFL per-edge weights | `use_ardfl`, `ardfl_h_weight`, `ardfl_w_weight`, `ardfl_ar_gate`, `ardfl_ar_thresh` | `loss.py`, `loss_ardfl` | ⚪ never run — wrong-signed, see §5 |
| 40 | AR-DFL entropy | `ardfl_entropy`, `ardfl_entropy_w` | same | ⚪ never run — bug B5 |
| 41 | **A-DFL** range scale | `use_adfl`, `adfl_w_scale`, `adfl_h_scale`, `adfl_log_clamp` | `loss.py:796-840` + `adfl_patch_dfl.py` | ⚪ never run |
| 42 | **PEU** uncertainty | `use_peu`, `peu_beta`, `peu_lambda`, `peu_detach`, `peu_warmup_epochs`, `peu_min_var`, `peu_w_clip`, `peu_log` | `loss.py:984-1017` | ⚪ never run |
| 43 | **EDGEW** fixed per-edge | `use_edgew`, `edgew_l/t/r/b` | `loss.py` | ⚪ never run — PEU's control |

### 2.5 Classification

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 44 | BCE | — | all | ✅ baseline |
| 45 | VFL | `use_vfl`, `vfl_alpha`, `vfl_gamma` | `loss.py`, `loss3`, `loss_ardfl`, `v1/..._v2/_v3` | ✅ reported failed |
| 46 | QFL | `cls_mode=qfl`, `qfl_beta` | `loss2`, `v1/..._v2/_v3` | ✅ reported failed |
| 47 | Class weighting | `use_class_weighting` / `use_class_weights` / `class_weights` | `loss.py`, `loss2`, `loss3`, `loss_satal3`, `v1/..._v2/_v3` | ✅ flat |
| 48 | Class weight modes | `class_weight_mode` | `loss2`, `v1/..._v3` | ✅ flat |
| 49 | Class weight normalisation | `normalize_class_weights` | `loss3` | ⚪ orphan |
| 50 | Small-object cls boost | `small_obj_cls_boost` | `loss3` | ⚪ orphan |
| 51 | cls loss selector | `cls_loss` | `v1/loss_v{1,2}updated` | ✅ |

### 2.6 Extra loss terms

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 52 | Center loss | `center_loss_weight_init/min`, `center_loss_decay_epochs` | 12 files, 4 versions | ✅ +0.14 (4-var) |
| 53 | Center loss modes | `center_loss_mode`, `center_crowd_iou` | `loss2`, `v1/..._v3` | ✅ |
| 54 | Repulsion | `use_repulsion`, `repulsion_weight` | `loss2`, `v1/..._v2/_v3` | ✅ inert |
| 55 | Bag asymmetric penalty | `use_bag_penalty`, `bag_penalty_weight`, `bag_class_id` | `loss2`, `v1/..._v3` | ✅ +0.10 |
| 56 | Tightness penalty | `tightness_gamma`, `tightness_small_only` | `v1/loss_v1updated` | ✅ −0.21 |
| 57 | L1 aux | `l1_aux_weight`, `l1_aux_beta`, `l1_aux_small_only` | `v1/loss_v1updated` | ✅ −0.20 |
| 58 | L1 balanced | `l1_balanced`, `l1_balanced_alpha`, `l1_balanced_gamma` | `v1/loss_v1updated` | ⚪ never run |
| 59 | Relative L1 | `rel_l1_weight`, `rel_l1_small_only` | `v1/loss_v1updated` | ⚪ never run |
| 60 | Box-edge jitter | `box_jitter`, `box_jitter_anneal` | `v1/loss_v2updated` | ⚪ never run |
| 61 | FSUS aux hook | — | `v1/loss_satal.py:293` | ⚪ never run |
| 62 | DetectAux / DetectObj heads | `aux_weight`, `obj_weight` | 10 files | ⚠️ |

### 2.7 Optimisation hygiene

| # | mechanism | config key(s) | implemented in | status |
|---|---|---|---|---|
| 63 | Per-sample clip | `iou_clip_start/end`, `dfl_clip_start/end` | `loss_cliping`, `loss_custom`, `v1/loss_v1updated` | 🔴 see §5 |
| 64 | Aggregate clip | same keys | `loss_nocliping` | 🔴 never binds |
| 65 | Soft clip | `_soft_clip` | `loss_satal3` | ⚪ orphan |
| 66 | Clip master switch | `use_loss_clip` | `loss.py`, `loss2`, `v1/..._v2/_v3` | 🔴 `loss2.py:20` marks it DEPRECATED/inert |
| 67 | Clip-rate telemetry | `[FIX-2]` | `v1/loss_v1updated:687` | ✅ |

---

## 3. Never run (⚪) — the untested surface

**Whole mechanisms:** LBA, PEU, EDGEW, A-DFL, AR-DFL, AR-DFL entropy, NWD ns D,
NWD ns C, NWD ns E, AR importance weight, soft clip, Focal-IoU, Wise-IoU,
box jitter, L1-balanced, relative L1, FSUS, small-object cls boost, class-weight
normalisation, bounded size weight, SWA multiplicative mode.

**Note:** every mechanism in `loss.py` that is not also in the v1 lineage has zero
measurements. The canonical rebuild has never produced a number.

---

## 4. Known inert / silently disabled (🔴)

| what | evidence |
|---|---|
| Clipping, all granularities | `loss2.py:20` — *"[C] DEPRECATED — inert in Rounds 1-3; keep off"*; runtime banner at `loss2.py:989` repeats it |
| Aggregate clip specifically | `loss_nocliping.py:222` clamps a normalised aggregate (~1–2) at 20→10 — threshold is ~10× the value, never binds |
| Shape-aware TAL | class not on the import path; guarded by `and ShapeAwareTaskAlignedAssigner is not None` → prints True, runs stock TAL |
| `r9b_nwd_adapt` "fixed C" | `v1/loss_v1updated.py:357` defaults `nwd_c=64.0`; line 395 states this saturates to `nwd≈1` for small boxes → the run was near-inert |
| B1 small-object gate | `(small_obj_px / min_stride)**2` vs grid-unit areas; `min_stride=8` for all anchors → 4×/16× wrong at stride 16/32, in 9 files |

---

## 5. Discrepancies against the existing docs

1. **Line counts in `LOSS_ABLATION_STATUS.md` §2 are wrong for exactly the four
   files that carry the R4–R11 mechanisms:** v2 1230→**1521**, v3 1103→**1722**,
   `loss_v1updated` 1002→**1765** (+76%), `loss_v2updated` 1048→**1572**. The four
   that match are the old untouched files. The doc describes earlier versions of
   the code your results came from.

2. **`loss.py` is 1985 lines, not ~1231.** The audit's *"`loss.py` ↔ `loss_ardfl.py`,
   48/53 identical bodies, `loss_ardfl` carries no mechanism `loss.py` lacks"* is
   now stale in the other direction: `loss.py` gained ~750 lines (PEU/LBA/A-DFL/
   EDGEW) that `loss_ardfl` lacks. They are no longer near-duplicates; `loss_ardfl`
   is a frozen fork.

3. **Section letters collide between lineages.** `loss2`/`losscustomorig` use
   E=SATAL, F=class weighting. `loss_v1updated`/`loss_v2updated` use
   E=classification, F=IARW. Any `[E]`/`[F]` log line or "Section F" note is
   ambiguous without knowing the installed file.

4. **`METHODS_TABLE.md` §8 lists the clipping A/B/C as a measured study.** The code
   marks it deprecated and inert. Those rows should be struck, not scored.

5. **AR-DFL is wrong-signed.** Moving an edge by `e` px costs `e/w` on a width edge,
   `e/h` on a height edge — ratio `h/w` = 2.69 here. Configured
   `ardfl_h_weight=1.5 / ardfl_w_weight=0.75` moves capacity *away* from the more
   IoU-sensitive edges, and the 1.25× mean changes total DFL magnitude.

6. **`dfl_entropy_small_only` defaults to 1** (`v1/loss_v1updated.py:392`), so the
   best-bag-AP run scoped entropy to small objects. `loss.py:670`'s `ardfl_entropy`
   gates on height edges — a different mechanism, not just the B5 scaling bug.

7. **AR-TAL has no fixed-β control.** `loss2.py:745-750` gives
   `β_eff = clamp(6 − 2·clamp((AR−2)/2,0,1), min=1)`; at the dataset mean AR 2.69
   that is **β_eff = 5.31**, near-identical to a flat β=5. `r8_artal`'s +0.27 may be
   a global β nudge. `adfl_iso050` and `peu_fixed_top` exist as controls; this has none.
   `ar = max(w/h, h/w)` is also symmetric — it cannot tell tall from wide.

8. **The project's own noise tolerance is ±0.35** (`loss2.py:7-8`: *"must land within
   ±0.35 of r2_swa_const06"*). NWD +0.32, DFL-entropy +0.28, AR-TAL +0.27 and every
   IoU variant fall inside it. Only `r8_area_sqrt` (+0.46) clears it.

9. **SATAL gates on area on a 94%-tall dataset.** Thresholds 0.0025/0.0225 normalised
   = 25×25 / 77×77 px at 512²; the mean object (33×72 = 2376 px² → 0.0091) sits in the
   interpolation band. `loss.py:673-696` already argues this exact point for the
   *weighting* (*"a thin trolley has medium AREA but small WIDTH"*) and defaults
   `swa_size_axis="width"`. `loss2.py:29-31` says the same of the assigner
   (*"assignment starvation that area-based SATAL misses"*). A width-gated SATAL was
   implied in the code headers and never built.
