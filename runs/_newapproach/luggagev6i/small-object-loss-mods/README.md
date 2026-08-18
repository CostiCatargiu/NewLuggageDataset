# Small-Object Loss Modifications — Ultralytics YOLO

Modifications to the loss, assignment, and box-metric code to better fit a **luggage detection dataset**
(backpack / bag / trolley) dominated by small, tall objects.

> **Status: written and reviewed, but NEVER EXECUTED.** No PyTorch-capable interpreter was available on the
> machine where these changes were made. Everything below is reasoned from the code, not measured.
> Validate before trusting it. See [Validation](#validation-do-this-first).

---

## Contents

This folder mirrors the repo-relative paths of every modified file:

```
ultralytics/utils/metrics.py     # bbox_iou: NWD blending, EIoU, IOU_FLAGS map
ultralytics/utils/tal.py         # TaskAlignedAssigner: scale equalization, iou_type, NWD
ultralytics/utils/loss.py        # BboxLoss: NWD, iou_type, object-normalized L1
ultralytics/cfg/default.yaml     # 5 new hyperparameters
ultralytics/cfg/__init__.py      # type validation for the new keys
docs/macros/train-args.md        # documentation rows
```

Copy them over the corresponding paths in an Ultralytics checkout to apply.

---

## Dataset that motivated this

| Property | Value |
|---|---|
| Images | 9138 train / 1827 valid / 1219 test |
| Instances | 41823 / 9819 / 6172 (~4.6–5.4 boxes per image) |
| Classes | backpack 27%, bag 22%, trolley 51% |
| Size (max side < 32 / 32–64 / > 64px) | **35.8% / 39.2% / 25.0%** |
| Mean box | **39 × 55 px** |
| Shape | **70.6% tall** (h/w > 1.25), mean h/w 1.55 |
| Per-class h/w | trolley 1.68, backpack 1.47, bag 1.33 (stable across all splits) |
| Image size | ~90% are 640 × 360 (16:9) |

Two properties drive everything below: **objects are small**, and **boxes are consistently tall** with a
class-specific aspect ratio.

---

## Mechanisms

All four default to stock behaviour. **With default settings the code should be numerically identical to
upstream** — that is the single most important thing to verify.

### 1. NWD blending — `nwd`, `nwd_c`

**Files:** `metrics.py` (`bbox_iou`), consumed by `BboxLoss` and `TaskAlignedAssigner.iou_calculation`

IoU degrades non-linearly for a few pixels of error on a tiny box. A 3px offset on a 15×25px backpack drops
CIoU to ~0.5; the same *relative* error on a 100px trolley barely moves it. The assigner then raises overlap
to `beta=6.0`, amplifying that noise.

Each box is modelled as a Gaussian; the 2nd Wasserstein distance is mapped to a similarity and blended with
the IoU variant:

```
similarity = (1 - nwd) * CIoU + nwd * exp(-sqrt(W2^2) / nwd_c)
```

`nwd_c` is a normalization constant **in pixels**, set to the dataset's mean object size.

Implementation notes:

- `BboxLoss` receives boxes in per-level *grid units* (`target_bboxes / stride_tensor`). NWD is an
  absolute-scale metric, so positives are converted back to pixels using the `stride` tensor already in the
  signature. CIoU is scale-invariant, so this is a no-op for the existing path, and it is skipped entirely
  when `nwd=0.0`.
- The Wasserstein term is accumulated in fp32 and halved before squaring: pixel distances squared
  (640² / 4 ≈ 102,000) exceed the fp16 max of 65504 and would produce `inf` under AMP.
- An offset of `(1e-3 * nwd_c)²` is added before the `sqrt` so its gradient stays finite as a prediction
  converges onto its target.

**Secondary benefit:** `exp(-d/c)` is strictly positive, so the alignment metric never collapses to exactly
zero (see [IoU clipping](#iou-clipping-not-implemented) below).

Reference: Wang et al., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*,
[arXiv:2110.13389](https://arxiv.org/abs/2110.13389).

### 2. Scale-equalized task alignment — `scale_balance`

**File:** `tal.py` (new `TaskAlignedAssigner.instance_gain`)

`select_candidates_in_gts` only accepts anchors whose **centre falls inside** the ground-truth box, so the
candidate pool grows with the object's **area**. `select_topk_candidates` then caps at `tal_topk=10`:

| Object | P3 candidates | P4 | P5 | Positives after top-10 |
|---|---|---|---|---|
| 12 × 18 px | ~3 | ~0 | 0 | **~3** |
| 39 × 55 px (dataset mean) | ~34 | ~8 | ~2 | **10** (capped) |
| 100 × 160 px | ~250 | ~60 | ~15 | **10** (capped) |

A small object therefore contributes roughly **3–5× less gradient than a mean-sized one, purely through
geometry** — before any consideration of detection difficulty. This affects 35.8% of the training instances.

Each ground truth's contribution is rescaled by

```
gain_i = (mean_positives / n_positives_i) ^ scale_balance
```

Normalizing against the batch **mean** (rather than a constant) keeps the overall loss magnitude stable, so
the `box`/`cls` balance and `target_scores_sum` do not drift.

**Known nonlinearity — disclose this in any write-up.** `target_scores` is a BCE target, so boosting can push
it above 1 and it is clamped to 1.0. In practice the clamp binds for small objects that are *already* well
fitted, meaning the boost concentrates on small objects that are still poorly fitted. That is arguably the
desired behaviour, but it is a real nonlinearity, not a clean reweighting.

### 3. Configurable IoU type — `iou_type`

**Files:** `metrics.py` (`EIoU` flag + `IOU_FLAGS` map), `loss.py`, `tal.py`

`BboxLoss` and `TaskAlignedAssigner.iou_calculation` both had `CIoU=True` hardcoded. They now share one
setting, so loss and assignment cannot silently diverge.

CIoU's aspect-ratio term has a documented structural flaw:

```
dv/dw = -(h/w) * dv/dh
```

The gradients are **always opposite in sign**, so CIoU can trade width against height but can never scale both
together — it regresses aspect *ratio*, never aspect *magnitude*. With 70.6% tall boxes clustered around
class-stable ratios, that is the wrong inductive bias. **EIoU** replaces the term with independent width and
height penalties.

Values: `iou`, `giou`, `diou`, `ciou` (default), `eiou`.

Reference: Zhang et al., *Focal and Efficient IOU Loss*, [arXiv:2101.08158](https://arxiv.org/abs/2101.08158).

### 4. Object-normalized L1 — `dfl_obj_norm`

**File:** `loss.py` (`BboxLoss.forward`, `reg_max == 1` branch)

In the DFL-free branch, edge distances were normalized by **image** size:

```python
target_ltrb[..., 0::2] /= imgsz[1]
```

So a 12px object yields `ltrb ≈ 6/640 = 0.009` while a 200px trolley yields `0.31` — a **~35× difference in
loss magnitude and gradient for the same relative error**. Unlike CIoU, this term is *linearly* biased toward
large objects. Setting `dfl_obj_norm=True` normalizes by the target box's own extent (`w,h,w,h` matched
against `l,t,r,b`) instead, making it scale-invariant.

> **This is a no-op unless `reg_max == 1`.** YOLOv8 / YOLO11 use `reg_max=16`, so it only applies to YOLO26 or
> an explicitly DFL-free config. Confirm which model you are training before relying on it.

---

## Hyperparameters

Added to `ultralytics/cfg/default.yaml` with type validation in `ultralytics/cfg/__init__.py`:

| Key | Type | Default | Validation set |
|---|---|---|---|
| `nwd` | float | `0.0` | `CFG_FRACTION_KEYS` ([0, 1]) |
| `nwd_c` | float | `24.0` | `CFG_FLOAT_KEYS` |
| `scale_balance` | float | `0.0` | `CFG_FRACTION_KEYS` ([0, 1]) |
| `iou_type` | str | `ciou` | `CFG_STR_KEYS` |
| `dfl_obj_norm` | bool | `False` | `CFG_BOOL_KEYS` |

### Suggested starting point for this dataset

```bash
yolo train model=yolo11n.pt data=luggage.yaml \
  nwd=0.5 nwd_c=46 scale_balance=0.5 rect=True
```

`nwd_c = 46` comes from the mean object size, `sqrt(39 × 55) ≈ 46`. `rect=True` matters because ~90% of the
images are 640×360 and square letterboxing wastes ~44% of the canvas on padding.

---

## Validation (do this first)

Nothing here has been run. In priority order:

```bash
# 1. Defaults must reproduce the baseline EXACTLY.
#    Mechanisms 1-3 touch bbox_iou's return structure and the assigner's normalization,
#    which every detection model depends on.
pytest tests/test_python.py -k "loss or train" -x
yolo train model=yolo11n.pt data=coco8.yaml epochs=3

# 2. Then exercise the new paths and watch for NaN in the first iterations.
#    If box loss goes NaN immediately, suspect the fp16 branch in the NWD block.
yolo train model=yolo11n.pt data=coco8.yaml epochs=3 nwd=0.5 nwd_c=46
yolo train model=yolo11n.pt data=coco8.yaml epochs=3 scale_balance=0.5
yolo train model=yolo11n.pt data=coco8.yaml epochs=3 iou_type=eiou
```

Three defects were caught by review rather than by running the code — an in-place `clamp_` on an
autograd-tracked tensor, an infinite `sqrt` gradient at zero distance, and fp16 overflow under AMP. Others may
remain.

Also note: the Ultralytics `AGENTS.md` requires work to be done in a `git worktree` on a feature branch rather
than the primary checkout. These edits were made in place.

---

## Not implemented — discussed and deliberately deferred

### Config-only changes, likely higher value than any of the above

- **`rect=True` / `imgsz=(640, 384)`** — ~90% of images are 640×360; square letterboxing wastes ~44% of the
  canvas. Raising `imgsz` to 960/1280 is the most reliable small-object fix in practice.
- **Reduce `scale` / `mosaic`** — mosaic tiles 4 images into one canvas, halving apparent object size, and
  `scale: 0.5` samples 0.5–1.5×. On a dataset where a third of instances are already < 32px, the downscale
  tail pushes objects below what P3 can resolve.

### Architecture

- **Add a P2 head (stride 4) and drop P5.** `ultralytics/cfg/models/v8/yolov8-p2.yaml` already exists. P5
  targets 128–640px objects, of which this dataset has almost none. No loss-code change needed — strides are
  read dynamically. **Probably a larger gain than any loss modification here.**

### IoU clipping (analysis only)

`iou_calculation` ends with `.clamp_(0)`. CIoU ranges over (-1, 1] and goes negative for disjoint boxes, so
the clamp collapses every bad candidate to exactly `0`. With `align_metric = score^alpha * overlaps^beta` and
`beta=6.0`, the metric is then zero, and `torch.topk` on a vector of ties **breaks them by index order** —
selecting the spatially first anchors rather than the best ones. Small objects hit this far more often, so
they disproportionately receive arbitrary, spatially-biased assignment early in training.

Precedent: `RotatedBboxLoss.floor = 0.01` is passed to `probiou` and documented as *"bound gradients for
sub-stride boxes"* — the maintainers already added a floor for the tiny-object case, but only in the rotated
path.

Proposed fix: **decouple ranking from normalization** — use the unclamped signed metric for `topk` selection
and the clamped one for `pos_overlaps` / target-score scaling. A naive shift such as `(x+1)/2` would be wrong,
since `pos_overlaps` scales `target_scores` and a terrible match would receive a 0.5 BCE target.

Measure before fixing: log the fraction of in-GT candidates with CIoU ≤ 0, bucketed by object size, over the
first few epochs.

### DFL bin utilization (the strongest paper idea)

DFL bins are **1 grid unit wide = 8px at P3**. For an object of size `s` at stride `r`, each edge distance is
`≈ s/2r`, so the 39×55px mean lands at `d ≈ 2.4` and `3.4`. **Targets occupy bins 0–4; bins 5–15 receive
essentially no mass** — roughly 70% of the box-regression capacity is dead, and quantization where it matters
is 8px on a 39px object (~20% of its width).

Proposed: keep the bins, move them. Monotone centres, denser near zero:

```
c_k = (n-1) * (exp(alpha * k / (n-1)) - 1) / (exp(alpha) - 1)
```

`alpha -> 0` recovers standard DFL exactly, making it a strict generalization. Encode with `torch.searchsorted`
and linear interpolation between neighbouring centres; decode as `sum(p_k * c_k)`.

Extensions: **per-level alpha** (P3 needs fine resolution, P5 does not), **learnable alpha**, or **per-axis
bins** — with h/w = 1.55 the t/b distances are systematically ~1.55× larger than l/r, so horizontal edges
crowd into even lower bins.

Requires keeping three places consistent: `DFLoss` (encode), `v8DetectionLoss.proj` (training decode), and
`nn/modules/block.py::DFL.conv.weight` (inference/export integral).

The diagnostic figure — bin-occupancy histogram, split by axis — is what sells this.

### Other deferred ideas

- **Anisotropic NWD** — replace the scalar `nwd_c` with per-axis `sigma_x=39`, `sigma_y=55`, giving a
  Mahalanobis distance under a dataset-derived covariance prior. The isotropic version implemented here is the
  special case `sigma_x == sigma_y`.
- **Class-conditional shape prior** — per-class h/w is strongly class-dependent and stable across splits
  (trolley 1.68/1.66/1.66). Track an EMA of per-class `log(h/w)` from ground truth and regularize predicted
  log-aspect toward it, weighted by inverse object size so it vanishes for large objects. Rationale: at 15px
  the appearance evidence for exact extent is weak, so the class prior is the best remaining information.
- **`beta` is hardcoded at 6.0** in `v8DetectionLoss.__init__` — a 6th power on a metric that is noisy for
  small boxes. Worth exposing and sweeping 4.0–6.0.
- **SWA — deliberately rejected.** Ultralytics already ships `ModelEMA`, which covers most of the benefit; SWA
  is a generic optimization technique with no connection to small objects or this dataset's statistics.

---

## Notes for a write-up

- **Only mechanisms 1 and 2 are candidate contributions.** #3 is EIoU (2021) — a baseline row, not novelty.
  #4 is closer to a defect fix.
- **NWD itself is applied work** (Wang et al. 2021). The more original angle is mechanism 2, whose strength is
  the diagnostic: plot positives-per-ground-truth against object area and show the curve rising to the `topk`
  cap. That quantifies a task-aligned-assignment bias that does not appear to be widely reported.
- **Novelty is unverified.** No literature search was performed. Before investing, search specifically for:
  GFLv2, "adaptive/dynamic DFL", "non-uniform bin regression detection", "learned quantization bounding box
  regression".
- **One dataset will not carry a paper.** These mechanisms are motivated by this size histogram; validate on
  VisDrone, AI-TOD, or TinyPerson.
- **Report mAP per size bucket**, not just overall mAP. With 60% of instances in the small bin, overall mAP
  moves for reasons that cannot be attributed.
- **Five knobs is 32 combinations** — too many to sweep honestly. Fix `iou_type` with a quick two-way test,
  treat `dfl_obj_norm` as model-dependent, and reserve the real ablation for `nwd` × `scale_balance`.
