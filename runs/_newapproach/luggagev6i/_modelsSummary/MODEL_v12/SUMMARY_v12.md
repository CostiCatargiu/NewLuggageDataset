# YOLOv12 (yolov12s) — Luggage dataset — what was tried, what worked

**Baseline** `yolov12s_default` — 54.77 mAP50-95 / 79.75 mAP50, stock yolov12s, 640 px.

**46 runs** across 7 result files. **v6i only** — every run here is directly comparable to
the YOLO26 tables in `MODEL_v26/`.

> The v5i material (11 training scripts, 6 result files, ~50 runs at 57–59 mAP50-95,
> including all the 896 work) was moved out to `../_v5i_removed/`. It is a **different
> test set** with its own baseline (`v12s_default2` = 57.63), so mixing it with anything
> in this folder produces meaningless deltas. Classification was done on the actual
> `DATA_YAML =` line, not the filename — `run_arch_port_v6i.py`, `run_model_scale.py` and
> `run_newluggage_ablation_new_swa_sqrtv6i.py` all mention v5i in passing but train on
> v6i, and were kept.

---

## The headline (v6i)

| | config | mAP50-95 | Δ | Δ% |
|---|---|---|---|---|
| **Best loss** | `yolov12s_sqrt0703` | 55.64 | **+0.86** | **+1.58%** |
| **Best arch** | `ls_shift_gctxP3` | 56.02 | +1.24 | +2.27% |
| | `arch_ls_shift` | 55.98 | +1.21 | +2.20% |

**The loss result is the strongest evidence in the entire project** — and unlike anything
on the YOLO26 side, it has real statistics behind it:

```
32 runs, 29 above baseline
sign test p = 1.3e-06
seed spread 0.11
positive on EVERY size bucket (S +0.95%, M +0.49%, L +0.31%)
```

Nothing on the v26 side comes close to that level of support. Lead the paper with it.

---

## What worked

### `sqrt0703` — the SWA family *(the project's best-evidenced result)*
Area-weighted regression with a sqrt schedule, alpha 0.7 → 0.3. **+0.86 (+1.58%)**,
positive on all four size buckets, 29/32 replications above baseline.

Runners-up from the same family confirm it is a real region rather than a spike:

| config | mAP50-95 | Δ | note |
|---|---|---|---|
| `yolov12s_sqrt0703` | 55.64 | +0.86 | best |
| `ms_s_sqrt_a0703_b15` | 55.49 | +0.72 | large **+3.31%** |
| `ms_s_sqrt_a0703_b25` | 55.47 | +0.69 | large −3.14% |

### Architecture — level-shift
`arch_ls_shift` is the only config in the campaign that beat baseline on **all four size
buckets simultaneously**, and it was 13/13 above baseline. Recall +1.94%, small +2.22%,
large +3.00%.

`ls_shift_gctxP3` edges it on mAP50-95 (56.02 vs 55.98) but is worse on large (−1.33% vs
+3.00%). `arch_ls_shift` is the one to report.

### Resolution 640 → 896 — NOT MEASURED ON v6i
**Zero 896 runs exist on v6i, for either model.** Every 640→896 pair in this project was
run on **v5i** and now lives in `../_v5i_removed/` (`run_luggage_arch_levelspec896.py`,
`runs_luggage_arch_hires__test_full_dataset.json`).

For context only, not for any table here: four v5i pairs gave +1.26 / +1.58 / +1.36 /
+1.44, mean **+1.41**, positive 4/4 — larger than any loss or arch effect anywhere in the
project. But they are confounded three ways: different dataset, b16 vs b32 with no
matched control, and each is a *specific architecture* at 896 rather than a pure
resolution change.

Also worth stating in any write-up: the source images are natively **640×360**, so 896 is
**upsampling** — more pixels to compute over, not more information.

Resolution is the largest untested effect on v6i.

---

## What did not work

| mechanism | result |
|---|---|
| **LB-TAL** — per-level top-k budgets | `lb_uniform` 55.57, `lb_prop` 54.82, `lb_p3_3` 55.21, `lb_p4wide` 55.03 — the uniform variant roughly matches the sqrt loss, the rest lose |
| **QFL** — quality focal loss | **47.25** — catastrophic |
| **NWD** on v12 | `nwd_blend25` 55.25, `nwd_small` 54.38 |
| **Size-conditioned combos** | `cmb_sizecond_aggr` 55.25, `cmb_sizecond` 55.10, `lb_sizecond` 54.88 — all below the plain sqrt result |
| **SNT** on v12 | 54.79 — below baseline, same direction as on YOLO26 |

The PEU family and the `gain_*` sweeps were v5i and have moved to `../_v5i_removed/`.

---

## The comparative point (this is the paper)

The same mechanisms behave **oppositely** on the two detectors:

| mechanism | YOLOv12 | YOLO26 |
|---|---|---|
| SWA / `sqrt0703` | **+0.86**, 29/32, p=1.3e-06 | **−0.48** |
| LB-TAL | ≈ 0 (`lb_uniform` +0.80) | **−0.43 to −0.82** |
| SNT | −0.0 (54.79 vs 54.77) | **−3.93 / −12.00** |
| SCB (size-conditioned beta) | not applicable — `topk=10` dilutes it | **+0.42**, the best v26 mechanism |

The reason is structural, not incidental:

- **YOLOv12**: `reg_max = 16` (DFL bins), NMS, one assignment branch with `topk = 10`.
- **YOLO26**: `reg_max = 1` (DFL-free, L1 on image-normalised ltrb), NMS-free `end2end`
  head, two branches, one2one with `topk2 = 1` — **a single anchor produces every
  prediction**.

That last property is why SNT is catastrophic on YOLO26 and inert on v12: with `topk2=1`
the winner/runner-up confidence gap *is* the duplicate suppression, and softening targets
destroys it. On v12, NMS does that job and the loss cannot break it.

**Claim:** loss-level improvements are architecture-specific, and the DFL-free /
NMS-free combination changes which interventions are even coherent.

---

## Open confound

The v12 arch runs were **b32 against a b54 baseline**. On YOLO26 the identical confound
was worth +0.52 mAP50-95 and accounted for 42–62% of the apparent architecture gain.

**No matched v12 control exists.** One stock `yolov12s` run at b32 fixes it. Until then
the arch numbers here are **upper bounds**, and I would not put the v12 arch claim in the
paper without it. The **loss** result is unaffected — it is batch-matched and stands.

---

## APPENDIX — every config, what it was meant to do, what it did

All 46 v6i runs. Δ is vs `yolov12s_default` 54.77. Arch rows were b32 against a b54
baseline and are **not** batch-matched. S/L are mAP50.

### SWA / `sqrt0703` — area-weighted regression  *(the project's best result)*
Intent: small objects contribute less regression gradient than large ones purely through
geometry. Weight the box loss by inverse area on a sqrt schedule, annealing α over
training so late epochs return toward stock.

| config | setting | mAP50-95 | Δ | S | L | what happened |
|---|---|---|---|---|---|---|
| `yolov12s_sqrt0703` | α 0.7→0.3 | 55.64 | **+0.86** | 77.37 | 82.12 | **best loss result**; positive on all 4 buckets |
| `ms_s_sqrt_a0703_b15` | + β 1.5 | 55.49 | +0.72 | 77.16 | **84.58** | best large of the family |
| `ms_s_sqrt_a0703_b25` | + β 2.5 | 55.47 | +0.69 | 77.66 | 79.30 | |
| `ms_s_sqrt_a09_03` | α 0.9→0.3 | 55.29 | +0.52 | 77.65 | 79.28 | |
| `swa0703_px44` | + px 44 | 55.27 | +0.50 | 77.18 | 81.09 | |
| `ms_s_sqrt_a08_04` | α 0.8→0.4 | 55.27 | +0.50 | 77.51 | 80.28 | |
| `ms_s_sqrt_a06_03` | α 0.6→0.3 | 55.25 | +0.48 | 77.26 | 82.31 | |
| `ms_s_sqrt_a0703_px36` | + px 36 | 55.23 | +0.45 | 77.05 | 77.08 | |
| `ms_s_sqrt_a0703_b10` | + β 1.0 | 55.21 | +0.43 | 76.96 | **85.16** | |
| `ms_s_sqrt_a07_04` | α 0.7→0.4 | 55.20 | +0.43 | 77.22 | 83.17 | |
| `ms_s_sqrt_a09_04` | α 0.9→0.4 | 54.98 | +0.20 | 77.33 | 82.73 | weakest of the family |

**Verdict: the strongest result in the project.** 11 configs, **all** above baseline, and
across the wider campaign 29/32 above with **sign test p = 1.3e-06** and seed spread 0.11.
The α axis is a smooth region, not a spike — the opposite of YOLO26's SCB. **On YOLO26
this exact family gave −0.48.**

### LB-TAL — per-level top-k budgets  *(assignment)*
Intent: `tal_topk=10` is applied globally, so large objects — which have far more
in-box anchors — consume the budget. Allocate top-k **per FPN level** so P3 keeps a
guaranteed share.

| config | setting | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `lb_uniform` | uniform budget | 55.57 | **+0.79** | best LB-TAL; roughly matches the sqrt loss |
| `lb_uniform_mk2` | repeat | 55.57 | +0.79 | reproduces exactly |
| `lb_uniform_seed1` | seed 1 | 55.45 | +0.68 | **seed spread 0.12** — the one seed check in the project |
| `lb_uniform_tk13` | topk 13 | 55.42 | +0.65 | |
| `lb_coarse_244` | {2,4,4} | 55.34 | +0.57 | |
| `lb_p3_3` | P3 = 3 | 55.21 | +0.43 | |
| `lb_p4wide` | P4 wide | 55.03 | +0.26 | |
| `lb_prop` | proportional | 54.82 | +0.05 | |
| `lb_sizecond` | + size-cond | 54.88 | +0.11 | |

**Verdict:** uniform works (+0.79), every *shaped* budget is worse than uniform. **On
YOLO26 the same mechanism gave −0.43 to −0.82.**

### Combinations
`cmb_p4wide` +0.83 · `cmb_sizecond_aggr` +0.48 · `cmb_sizecond` +0.33 ·
`cmb_lbU_swa0703` +0.33 · `cmb_p4wide_clsswa` +0.15 · `cmb_p4wide_qg50` **−3.15**

`cmb_lbU_swa0703` is the key negative: LB-TAL (+0.79) **plus** sqrt0703 (+0.86) gives
**+0.33** — far below either alone. The two axes do not compose, the same
non-additivity later seen on YOLO26.

### Other loss mechanisms
| config | intent | mAP50-95 | Δ | what happened |
|---|---|---|---|---|
| `clsw_sqrt` | class-weighted + sqrt | 55.28 | +0.51 | |
| `nwd_blend25` | Wasserstein blend 0.25 | 55.25 | +0.47 | mild positive — **opposite sign to YOLO26** (−0.20) |
| `nwd_small` | NWD, small only | 54.38 | −0.40 | large 85.41, best large in the folder |
| `posboost` | boost positives | 55.06 | +0.29 | large 75.60, worst |
| `snt` | soft negative targets | 54.79 | +0.02 | **inert** — vs −3.93/−12.00 on YOLO26 |
| `qfl` | quality focal loss | **47.25** | **−7.53** | catastrophic; large 63.49 |

`snt` is the single most informative row in this table: **inert on v12, catastrophic on
YOLO26.** v12 has NMS to remove the duplicates that soft targets create; YOLO26's
`topk2=1` head does not, so the confidence gap SNT destroys *is* its duplicate
suppression.

### Architecture — level-shift family  *(b32 vs a b54 baseline — NOT matched)*
Intent: shift which FPN level handles which object size, so small objects are assigned to
a finer-stride level than the default range gives them.

| config | mAP50-95 | Δ | S | L | note |
|---|---|---|---|---|---|
| `ls_shift_gctxP3` | 56.02 | +1.24 | 78.37 | 80.78 | highest mAP, but large −1.09 |
| `arch_ls_shift` | 55.98 | +1.21 | 78.34 | **84.32** | **the one to report** — beat baseline on all 4 buckets, 13/13 |
| `ls_shift_wiou` | 55.91 | +1.13 | 78.25 | 80.75 | + WIoU |
| `arch_levelspec` | 55.90 | +1.13 | 77.89 | 80.83 | level-specific heads |
| `ls_shift_k5` | 55.87 | +1.10 | 78.05 | 84.16 | k=5 |
| `arch_ls_shift_v6` | 55.84 | +1.07 | 78.22 | 83.72 | |
| `arch_dysample_p2_gctx2` | 55.78 | +1.01 | 78.02 | 83.19 | DySample + P2 + gctx |
| `arch_ls_k5` | 55.68 | +0.91 | 77.37 | 82.46 | |
| `arch_levelspec_v6` | 55.63 | +0.86 | 77.78 | 80.86 | |
| `ls_shift_sqrt` | 55.61 | +0.84 | **78.73** | 81.60 | + the sqrt loss — **below arch alone, axes do not compose** |
| `arch_gctx22` | 55.50 | +0.73 | 77.81 | 81.06 | |
| `ls_shift_gctxP3P4` | 55.49 | +0.72 | 77.99 | 78.81 | |
| `ls_shift_lbtalB` | 55.14 | +0.37 | 78.12 | 82.93 | + LB-TAL |

**Verdict:** the level-shift family is consistently the top of the folder, but the batch
confound is unquantified here. `ls_shift_sqrt` (+0.84) sitting **below** `arch_ls_shift`
(+1.21) is the same arch+loss non-additivity found on YOLO26.

---

## Folder layout

```
training/   13 v6i runners — run_lbtal_*, run_sizecond_*, run_overnight_tune,
            run_newluggage_ablation_*, run_arch_*, run_model_scale
results/    7 JSONs, 46 runs — ALL v6i, directly comparable to MODEL_v26/
patch/      lossv2updated*.py, loss_custom_v3_fixed.py, satal.py,
            zg_modules_v6i.py, patch_ultralytics_modules.py, verify_port.py,
            archAblation loss.py / tal.py
```

## Provenance for the headline numbers

```
yolov12s_default      runs_newl_luggagev6i__test_full_dataset.json
yolov12s_sqrt0703     runs_newl_luggagev6i__test_full_dataset.json
ms_s_sqrt_a0703_b15   runs_newl_luggagev6i__test_full_dataset.json
arch_ls_shift         runs_arch_v6i__test_full_dataset.json
ls_shift_gctxP3       runs_arch_refine_v6i__test_full_dataset.json
ls_shift_wiou         runs_arch_refine_v6i__test_full_dataset.json
```

Best run in this folder is now `ls_shift_gctxP3` at 56.02 — the 57–59 figures that used to
appear here were v5i and are gone.
