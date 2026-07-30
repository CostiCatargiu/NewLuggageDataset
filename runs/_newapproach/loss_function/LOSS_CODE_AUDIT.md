# Loss-Function Code Audit — what is actually implemented in `loss_function/`

Method: AST parse of all 19 `.py` files, normalized body-hashing of every function
to separate copy-paste from real forks, `getattr(hyp, ...)` key extraction to find
which file reads which config key, plus manual reading of every distinct
implementation. Claims below cite file:line and come from code, not docstrings.

**Scope:** 19 loss files, 22,825 lines, 90 distinct qualnames.

---

## 1. There are not 18 loss functions. There are ~11.

Byte-identical files (`md5sum`):

| A | B |
|---|---|
| `loss2.py` | `v1/loss_satal_swa_plus_v3.py` |
| `losscustomorig.py` | `v1/losscustomoriginal.py` |

Near-identical (fraction of shared function bodies that hash equal):

| pair | identical bodies |
|---|---|
| `loss_custom_git.py` ↔ `losscustomorig.py` | 39/39 (differs only by 4 extra defs) |
| `loss_cliping.py` ↔ `loss_custom.py` | 38/39 |
| `v1/loss_nwd.py` ↔ `v1/loss_satal_swa_plus.py` | 42/45 |
| `loss_ORIG.py` ↔ `v1/loss_satal.py` | 29/31 |
| `loss.py` ↔ `loss_ardfl.py` | 48/53 (**all 5 differences are docstrings + 1 added assert**) |
| `v1/loss_v1updated.py` ↔ `v1/loss_v2updated.py` | 39/45 |

`loss_ardfl.py` carries no mechanism `loss.py` lacks. Keeping two 1,200-line
copies in sync by hand is the main maintenance risk in the folder.

---

## 2. What is common to every file

Every file is a fork of the same stock Ultralytics `v8DetectionLoss` + `BboxLoss`.
Functions with exactly **one** implementation across all 17 distinct files:

- `DFLoss.__call__` — **the DFL loss itself was never modified in any file.**
- `v8DetectionLoss.bbox_decode` — 2 hashes, but the only difference is a variable
  rename (`c` → `ch`) and deleted comments. **The DFL decode was never modified.**
  `self.proj = torch.arange(reg_max)` is identical everywhere.
- `VarifocalLoss`, `KeypointLoss`, `v8PoseLoss.kpts_decode`, `E2EDetectLoss.__call__`,
  `v8SegmentationLoss.__init__`, `RotatedBboxLoss.__init__`.

This confirms the "box representation is untouched" claim in `LOSS_ABLATION_STATUS.md`
§5 — mechanically, not just by assertion. Note however that DFL decodes as an
*expectation* over bins (`softmax(3).matmul(proj)`), i.e. a continuous value. The
premise that "DFL quantizes edges into 16 bins, too coarse" is not correct as
stated; there is no quantization floor. Range is not binding either (at stride 8
the cap is a 240 px box; mean object height is 72 px at 512).

Mechanisms present in nearly every non-stock file (same or near-same code):
SWA alpha-blended area/score weighting, `small_obj_boost`, center loss,
epoch-scheduled clipping, and the `DetectAuxLoss` / `DetectObjLoss` heads.

---

## 3. Where the files actually diverge

| function | files | distinct implementations |
|---|---|---|
| `BboxLoss.forward` | 17 | **12** |
| `v8DetectionLoss.__init__` | 17 | 14 |
| `v8DetectionLoss.__call__` | 17 | 11 |
| `v8DetectionLoss._print_config` | 14 | 11 |
| `BboxLoss._compute_weights` | 12 | 5 |
| `v8DetectionLoss._compute_center_loss` | 12 | 4 |
| `BboxLoss._get_dynamic_alpha` | 12 | 3 |

Mechanisms unique to exactly one file:

| file | unique |
|---|---|
| `loss2.py` / `v1/..._v3.py` | `ARAwareTaskAlignedAssigner` (AR-relaxed TAL beta) |
| `loss_satal3.py` | `_eiou`, `_nwd`, `_soft_clip`, `_fg_stride`, `attach_epoch_sync` |
| `v1/loss_v1updated.py` | `NWDTaskAlignedAssigner`, `_bbox_eiou_loss`, `_bbox_siou_loss` |
| `loss3.py` | `_iou_metric`, `_width_adaptive_weight` |
| `loss.py` / `loss_ardfl.py` | `DFLoss.per_edge`, `_dfl_edge_entropy`, `SataLSwaConfig` |
| `v1/loss_satal.py` | `_collect_fsus_loss` |

---

## 4. 🔴 Four mutually incompatible NWD config namespaces

There are **four separate NWD implementations** and **five key namespaces**. No file
reads another's keys.

| namespace | keys | read by |
|---|---|---|
| A | `nwd_C`, `nwd_mode`, `nwd_weight`, `nwd_small_threshold` | `loss2`, `v1/loss_nwd`, `v1/..._swa_plus{,_v2,_v3}` |
| B | `nwd_ratio`, `nwd_c`, `nwd_adaptive`, `nwd_anneal`, `nwd_anneal_min`, `nwd_c_adaptive`, `nwd_c_k` | `v1/loss_v1updated`, `v1/loss_v2updated` |
| C | `nwd_const`, `nwd_gate_px` | `loss_satal3` |
| D | `nwd_c_px`, `nwd_small_width_px`, `nwd_debug` | `loss.py`, `loss_ardfl.py` |
| E | `nwd_c`, `nwd_width_gate_px` | `loss3` |

**`use_nwd` is not read by `loss_v1updated.py` at all.** The r9/r10 rounds ran on
that file, so `use_nwd: false` in `r10_nwd_fixedc.yaml` was inert text; the live
switch was `nwd_ratio: 0.3`.

**Consequence:** an `args_yaml` alone cannot tell you what ran. The saved configs are
a union of every generation's namespace, and which subset was live depends on which
file was installed at `ultralytics/utils/loss.py` at that moment — which is recorded
nowhere. This is the single biggest threat to the validity of the 60-run table.

### 4.1 What `r10_nwd_fixedc` actually computed

Decoding `r10_nwd_fixedc.yaml` against `loss_v1updated.py:635-672`:

```
nwd_ratio=0.3  nwd_adaptive=1  nwd_anneal=1  nwd_c_adaptive=1  nwd_c_k=0.5
small_obj_px=48  nwd_anneal_min=0.1
```

- applied **only** to `small_mask` (area < 48² px) — not to all boxes
- `c = 0.5 * sqrt(area_px)`, clamped ≥ 4 → **per-anchor adaptive temperature**, ~2–24 px
- per-anchor ratio `= 0.3 * smallness * anneal`, where `anneal` decays 1.0 → 0.1
- ⇒ effective NWD weight ≤ 0.3 early, ≤ 0.03 by the final epoch, on a subset of boxes

The run named "fixed C" used **adaptive** C. `r9b_nwd_adapt` differs from it by
`nwd_c_adaptive` alone — the two labels are swapped.

Compare `run_ardfl_ablation.py::_BASE`, which claims to reproduce it:

```python
use_nwd=True, nwd_mode="blend", nwd_weight=0.5, nwd_c_px=12.0
```

In `loss.py` that is a **constant 0.5 blend, fixed 12 px C, on every foreground box,
no annealing, no smallness ramp**. Different scope, different temperature, roughly
10–50× the strength. `ardfl_anchor` will not reproduce 57.75, and every AR-DFL delta
would be measured against a baseline that has never been run.

---

## 5. Bugs and silent-failure paths

| # | issue | files | status |
|---|---|---|---|
| B1 | `small_threshold = (small_obj_px / min_stride)**2` compared against **grid-unit** areas. `min_stride` is 8 for all anchors, so for stride-16/32 anchors the small-object gate is off by 4×/16×. | `loss_cliping`, `loss_custom`, `loss_custom_git`, `loss_nocliping`, `losscustomorig`, `v1/loss_nwd`, `v1/..._swa_plus`, `v1/..._swa_plus_v2`, `v1/losscustomoriginal` (**9 files**) | fixed only in `v1/loss_v1updated` `[FIX-1]` and `loss.py` |
| B2 | `return loss / fg_mask.sum` — divides by a bound method, not a number. | `loss_cliping:937`, `loss_custom:937`, `loss_custom_git:965`, `losscustomorig:965`, `v1/losscustomoriginal:965` | detection-only runs unaffected |
| B3 | `use_shape_tal` guarded by `if ... and ShapeAwareTaskAlignedAssigner is not None` — prints `True` and silently runs stock TAL if `shape_tal.py` is not importable. It is not present in this folder. | `loss_custom_git:395`, `losscustomorig:395`, `loss_satal3:489`, `v1/losscustomoriginal` | **verify against run logs** |
| B4 | `class E2EDetectLoss` defined twice; the second (line 1335) shadows the first (line 1239). | `loss.py` | harmless, remove |
| B5 | `ardfl_entropy` adds a bare unweighted scalar outside the `/norm` division, unlike `v1/loss_v1updated.py:808` which uses `(ent * weight).sum() / target_scores_sum`. Different mechanism and scale from the `r10_dfl_entropy` run it is based on. Also gates on height edges, where the original gated on small objects. | `loss.py:670`, `loss_ardfl.py` | fix before use |
| — | SATAL raises `ImportError` loudly if `ultralytics.utils.satal` is missing. | `loss.py:763` | **not** a silent-failure risk |

---

## 6. Experiment-design issues in `collected_results/`

Independent of the code, from diffing each `args_yaml` against its own round anchor:

- **No seed replicates exist.** `r9_anchor2`, `r9b_anchor`, `r10_anchor` carry
  byte-identical metrics (82.70 / 57.43) — one run reported three times. The noise
  band was never measured, so "+0.3 is noise" and "SATAL costs 2.8" are both unsupported.
- **Round anchors span 1.18 pt** (r8 56.25, r3_swa 56.44, r9b/r10 57.43) and are
  genuinely different configs. Sorting all rounds against 57.43 mislabels in-round
  wins as losses: `r8_artal` is **+0.27 over its own anchor**.
- **Cross-source offset up to 1.11 pt** for the *same* run (`run2_siou_seed42`:
  56.18 under `per_class_seed42` vs 57.29 under `runs_loss_ablation`). The master
  table mixes at least six sources.
- **Multi-variable diffs.** `r3_satal_r3` moved **12** knobs at once (incl.
  `tal_alpha` 0.5→0.6, `tal_beta` 6.0→5.0, `tal_topk` 10→12, SWA 0.6→0.0,
  `small_obj_boost` 1.75→1.0). Its −2.8 is unattributable; in-round it is −1.87.
  `r10_nwd_fixedc` moved 4. `r9b_dfl_gated` 3. `r9b_iarw_nwd` 4.
- **Clean single-variable runs:** `r10_dfl_entropy`, `r10_alpha_iou`, `r10_l1_smooth`,
  `r10_tightness`, `r8_artal`, `r8_area_sqrt`, `r8_bag_penalty`, `r8_boost_bag`,
  `r8_cls_swa`, `r9b_iarw`, `r9b_iarw_lo`. Everything else moved 3–14.

---

## 7. Recommended order of work

1. **Record which loss file is installed** with every run (hash it into the params
   json). Without this the config archive is not interpretable.
2. **Rebuild the results table** with per-round deltas and one eval source.
3. **Three seed replicates of one config** to establish the noise band, before
   ranking anything separated by < 1 pt.
4. **Fix `_BASE`** in `run_ardfl_ablation.py` to actually reproduce `r10_nwd_fixedc`
   (small-only, adaptive C, annealed ratio 0.3) — or drop the claim and treat
   `ardfl_anchor` as a new baseline in its own right.
5. **Fix B5** so `ardfl_entropy` reproduces the mechanism that gave the best bag AP.
6. **Delete** `loss_ardfl.py`, `loss2.py`, `losscustomorig.py` as duplicates; keep
   `loss.py` and the `v1/` originals.
7. Only then run AR-DFL — and see the aspect-ratio note below first.

### Note on AR-DFL direction

For a box `w × h`, moving one edge by `e` px costs `e/w` IoU on a width edge and
`e/h` on a height edge — ratio exactly `h/w`, i.e. **2.69× on this dataset**
(bag 2.23, backpack 2.55, trolley 2.96). DFL bins are stride-uniform, so both edge
pairs get the same absolute precision while width errors cost ~2.7× more IoU. The
configured `ardfl_h_weight=1.5 / ardfl_w_weight=0.75` moves capacity *away* from the
width edges. A mean-normalized, per-box form (`w_edge = 2h/(w+h)`, `h_edge = 2w/(w+h)`)
has the right sign, needs no AR gate, and avoids the current 1.25× DFL-magnitude
confound.

Caveat, from this audit: `_bbox_eiou_loss` (`v1/loss_v1updated.py:123`) already
penalizes `(Δw)²/cw²` and `(Δh)²/ch²` — approximately per-axis *relative* error,
the same idea in the IoU branch — and measured 56.4–56.5. Expect the corrected
AR-DFL to be small too.
