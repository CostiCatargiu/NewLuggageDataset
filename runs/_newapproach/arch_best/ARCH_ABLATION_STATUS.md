# Luggage Detection (YOLOv12) — Research Status & Roadmap

> **Target venue:** IEEE Access
> **Task:** Detect `backpack`, `bag`, `trolley` in 512×512 images.
> **Base model:** YOLOv12s.
> **Baseline run:** `v12s_default2` — mAP50 = **82.77**, mAP50-95 = **57.63** (test_full_dataset).
> **Last updated:** 2026-07-30

---

## 1. Dataset — the facts that drive every decision

Source: `LuggageDatasetSplit.txt` (train split, 42,538 instances).

| Property | Value | Implication |
|----------|-------|-------------|
| Classes | backpack 27%, bag 22%, trolley 51% | trolley dominant; bag rarest |
| **Tall objects (h/w > 1.25)** | **94%** | the single defining property |
| Wide objects (w/h > 1.25) | 1.1% | horizontal context is ~wasted |
| Mean aspect ratio (h/w) | **2.69** | strongly vertical |
| Per-class AR | bag **2.23** < backpack **2.55** < trolley **2.96** | subtle AR variation *within* tallness |
| **Small objects (<48 px)** | **40.3%** | high-res detail matters |
| Mean object size | 33 × 72 px | small + tall |
| Image size | 512 × 512 (all) | — |
| Hardest class | **bag** | most shape variance (241 wide + 1078 square + 8171 tall) |

**Design law derived from this:** the dataset has **ONE dominant geometry (tall)** with **subtle per-class AR differences**. Any dataset-specific module must *commit to the tall prior* and *adapt within tallness* — NOT hedge across wide/square/tall shapes.

---

## 2. The central diagnostic finding

```
mAP50    ≈ 83     →  objects are FOUND
mAP50-95 ≈ 57.6   →  boxes are NOT TIGHT
─────────────────────────────────────────
25-point gap      =  LOCALIZATION QUALITY problem
AR50_small ≈ 96%  →  proves recall ceiling is already hit
```

**Interpretation:** the model reliably *finds* objects; it fails to *localize them precisely*. This gap lives in **box regression / convolution geometry**, NOT in feature detection. Feature-attention modules (which improve *what* is seen) therefore cannot close it — and empirically, they don't (see plateau below).

---

## 3. All experiments run so far (40 runs, all @640 unless "hires")

**Comparability:** every run below shares the identical loss/assigner config
`use_satal=true, tal_topk=12, tal_alpha=0.6, tal_beta=5.0`. Only the architecture changes. Metric = **mAP50-95** on `test_full_dataset`. Baseline `v12s_default2 = 57.63`.

### 3.1 Top tier — the winners

| Rank | Run | mAP50 | **mAP50-95** | Δ vs base | Notes |
|------|-----|-------|--------------|-----------|-------|
| 1 | `arch_levelspec_hires2` | 83.91 | **59.28** | **+1.65** | **BEST OVERALL** — level-specific head @**896px** |
| 2 | `arch_dysample_hires4` | 83.83 | 59.19 | +1.56 | DySample @896px |
| 3 | `arch_decoupled2_hires3` | 83.82 | 59.09 | +1.46 | @896px |
| 4 | `arch_p2head_gctx_hires10` | 84.02 | 59.03 | +1.40 | P2 head + gctx @896px |
| 5 | `arch_levelspec` (@640) | 83.34 | **58.02** | **+0.39** | **best @640** |
| 6 | `arch_dysample_p2_gctx2` (@640) | 83.00 | 57.91 | +0.28 | best @640 *combo* |
| 7 | `arch_gctx22` (@640) | 83.15 | 57.84 | +0.21 | best @640 *single module* |
| 8 | `arch_p2head_coordatt3` (@640) | 82.79 | 57.78 | +0.15 | CoordAtt |

### 3.2 The 640px plateau (the key negative result)

Excluding hires and broken runs, **all 640px architectures fall within a 0.90-point band: 56.94 → 58.02.** Representative slice:

| Run | mAP50-95 | Δ | Module family |
|-----|----------|---|---------------|
| `arch_levelspec` | 58.02 | +0.39 | level-specific head |
| `arch_gctx22` | 57.84 | +0.21 | global context (avg+max) |
| `arch_p2head_coordatt3` | 57.78 | +0.15 | coordinate attention |
| `arch_dsconv` | 57.77 | +0.14 | dynamic snake conv |
| `arch_detail_aux4` | 57.64 | +0.01 | small-detail + aux (best on large objs) |
| **`v12s_default2`** | **57.63** | **0.00** | **baseline** |
| `arch_dysample` | 57.61 | −0.02 | content-aware upsampling |
| `arch_deepcls` | 57.49 | −0.14 | deeper cls tower |
| `arch_shapecbam` | 56.91 | −0.72 | H/V/S shape attention (**wrong premise**) |
| `arch_star` | 56.82 | −0.81 | StarNet multiplicative mixing |

### 3.3 Broken / failed runs (exclude from analysis)

| Run | mAP50-95 | Cause |
|-----|----------|-------|
| `arch_gctx2_dysample` | 41.87 | catastrophic — broken training run |
| `arch_ls_nodys2` | 34.03 | catastrophic — broken training run |

---

## 4. What we learned (the story so far)

1. **Resolution beats architecture 3-to-1.** Going 640→896px gives **+1.5 mAP50-95**; the best *architecture* change @640 gives only **+0.4**. Every top-4 run is a hires run.
2. **The 640px feature-space is saturated.** 30+ modules land in a 0.9-pt band. Stacking winners shows **zero additivity** (e.g. `dysample_p2_gctx5` = 57.63 did *not* beat `gctx22` = 57.84).
3. **The bottleneck is localization, not detection** (Section 2). This is *why* feature-attention plateaus.
4. **Every module tested so far is a KNOWN published block** — CBAM/BAM, Coordinate Attention (CVPR'21), DySample (ICCV'23), StarNet (CVPR'24), DSConv (ICCV'23), SE (2018). "We added X to YOLOv12" is an *application*, not a *contribution*.
5. **`ShapeCBAM` failed on a wrong premise** — it assumed bags are wide and split capacity across H/V/S kernels. The data says 94% tall. It wasted 2/3 of its capacity → −0.72.
6. **bag is the hardest class** (AP50-95 ≈ 0.50 vs trolley ≈ 0.63) — most shape variance.

---

## 5. Architecture vs Loss — where each change lives

| ARCHITECTURE changes (network graph) | LOSS / TRAINING changes (optimization) |
|--------------------------------------|----------------------------------------|
| gctx2, CoordAtt, DySample, ZGStar, ZGDSConv, P2 head, levelspec, resolution | **SATAL assigner** (locked baseline: topk=12/α=0.6/β=5.0) |
| DetectDeepCls / DetectWideCls | DetectAuxDual (aux loss — partly architectural) |
| **ARSC / ARSPP / ARGate (NEW — see §6)** | **AR-DFL (proposed, unbuilt — see §7)** |

**Insight:** we have exhausted the architecture/feature space but **barely touched the loss/regression space** — which is exactly where the 25-pt localization gap lives.

---

## 6. NEW novel contribution — implemented, ready to train

**Theme: aspect-ratio-STEERED convolution geometry.** Instead of attention on square
features, steer the *convolution geometry itself* toward each object's vertical extent.
All three are in `nn_modules/block.py`, zero-gated (identity at init), channel-preserving.

| Block | Mechanism | Role |
|-------|-----------|------|
| **ARSC** *(flagship)* | Per-**location** scalar `r(p)∈[0,1]` blends a tall `(k×1)` kernel with a square 3×3 kernel: `y=(1−r)·sq + r·tall`. `r` predicted by 3×3 conv→sigmoid. | Learns *how tall* the kernel should be at every pixel. |
| **ARSPP** | Multi-scale pyramid of **vertical** strip convs (k∈{3,7,11}) — the tall-object analogue of ASPP. | Multi-scale vertical context, no wasted horizontal RF. |
| **ARGate** | Global per-image verticality gate on a tall branch. | Cheap **control**: isolates the value of ARSC's *per-location* adaptivity. |

**Why this is genuinely novel (not another off-the-shelf block):**
- ZGStrip / ZGDSConv use a **fixed** elongated kernel everywhere; ARSC learns per-pixel geometry.
- CoordAtt / gctx2 keep **square** convolution geometry + attention; ARSC steers geometry.
- ShapeCBAM hedged across wide/square/tall (wrong premise); ARSC commits to the tall prior.
- It attacks **localization** (the real bottleneck), which attention provably cannot.
- The `r` map gives a ready-made **"verticality vs class-AR" paper figure**.

### 6.1 The planned ablation — `run_luggage_arch_novel.py` (7 runs @640)

| # | Run | Config | Purpose |
|---|-----|--------|---------|
| 1 | `arch_strip_baseline` | ZGStrip @P3P4P5 | **fixed** geometry control |
| 2 | `arch_arsc` | ARSC @P3P4P5 | **flagship** (per-location) |
| 3 | `arch_arsc_p3p4` | ARSC @P3P4 | placement ablation |
| 4 | `arch_argate` | ARGate @P3P4P5 | **global** adaptivity control |
| 5 | `arch_arspp` | ARSPP @P3P4P5 | vertical pyramid |
| 6 | `arch_arspp_p3p4` | ARSPP @P3P4 | placement ablation |
| 7 | `arch_arsc_gctx2` | ARSC + gctx2 | additivity with best-known module |

**Expected narrative (to confirm empirically):**
`fixed strip (1) < global gate (4) < per-location ARSC (2)`
→ proves that *adaptivity*, and specifically *per-location* adaptivity, is what helps.
Run (7) tests whether ARSC is additive with the #1 known module (it should be, if it truly attacks a different bottleneck).

### 6.2 Status of the code
- ✅ `nn_modules/block.py` — ARSC/ARSPP/ARGate classes + `__all__`
- ✅ `nn_modules/tasks.py` — import + width-scaling registration
- ✅ `nn_modules/__init__.py` — re-export hub (mirrors live package)
- ✅ `run_luggage_arch_novel.py` — 7-run ablation script
- ⏳ **NOT yet built/trained** — needs copy to `ultralytics/nn/modules/` then `--build-only`

---

## 7. What to test next — and WHY (priority order)

### Priority 1 — Run the novel-block ablation (§6.1)  *[code ready]*
- **Why:** it's the current architecture novelty, motivated directly by dataset stats.
- **How:** copy `nn_modules/` → `ultralytics/nn/modules/`, then
  `python run_luggage_arch_novel.py --build-only` (verify), then full run.
- **Success = ** `arch_arsc` > `arch_strip_baseline` > baseline, and ordering in §6.1 holds.

### Priority 2 — Build **AR-DFL** (Aspect-Ratio-aware DFL)  *[unbuilt — the strongest idea]*
- **Why:** attacks the **25-pt localization gap** *directly* — the real bottleneck.
  Stock DFL uses **16 identical bins for all 4 box edges**. For 94%-tall objects, the
  height edges need more/wider bins (high variance) and width edges need fewer (low
  variance). Symmetric bins for asymmetric objects **waste capacity on width and starve
  height** → loose boxes. This is a **loss-space contribution nobody has published for
  elongated-object detection.**
- **Where:** DFL lives in `nn_modules/block.py::DFL` (integral) + the loss/assigner path.
- **Why it pairs with ARSC:** ARSC = geometry (arch axis); AR-DFL = regression (loss axis).
  Orthogonal bottlenecks → should be **additive** → a two-legged contribution.

### Priority 3 — Multi-seed validation (statistical significance)  *[for the paper]*
- **Why:** the 640 plateau is a 0.9-pt band; a +0.2 "win" could be noise. IEEE Access
  reviewers increasingly demand this.
- **How:** re-run the top 2–3 configs with seeds `[0, 42, 123]`, report **mean ± std**,
  run a paired significance test.

### Priority 4 — Combine best recipe (deployable model)  *[practical value]*
- **Why:** the paper needs a single "best model we recommend."
- **How:** `levelspec` (best head) + **896px** (biggest lever) + winning novel block.
  Current best deployable = `levelspec_hires2 = 59.28`; goal is to beat it with ARSC/AR-DFL.

### Priority 5 — Cross-detector comparison  *[paper context]*
- **Why:** reviewers want to know how YOLOv12s+our-method compares to alternatives.
- **How:** add RT-DETR, YOLOv11 baselines to the comparison table (same data/splits).

### Priority 6 (optional) — Knowledge Distillation (v12l → v12s)
- **Why:** reliable +1–2%, orthogonal to everything else. Practical, less novel.
- **How:** train a v12l teacher, distill to the v12s student.

---

## 8. Recommended paper narrative (3-legged story)

| Leg | Content | Status | Role |
|-----|---------|--------|------|
| **1. Diagnosis** | 40-run ablation proving the feature-space is saturated and localization is the true bottleneck (the 25-pt gap analysis) | ✅ done | methodological rigor |
| **2. Novelty** | **ARSC** (geometry) and/or **AR-DFL** (regression), both motivated by the 94%-tall / AR-2.69 statistics | ⏳ ARSC ready, AR-DFL to build | the core contribution |
| **3. Practical recipe** | resolution scaling (896px, +1.5) + best head (levelspec) as the deployable model | ✅ done | practical value |

The elegance: **Leg 1 justifies Leg 2.** We don't propose ARSC/AR-DFL out of nowhere — we *prove* feature engineering is exhausted, then show the untapped gain lives in geometry/regression.

---

## 9. Key file map

| Purpose | Path |
|---------|------|
| Dataset stats | `arch_best/LuggageDatasetSplit.txt` |
| Eval results (640) | `arch_best/arch_luggage_eval/runs_luggage_arch__test_full_dataset.json` (+ `arch4`, `arch5`) |
| Eval results (896) | `arch_best/arch_luggage_eval/runs_luggage_arch_hires__test_full_dataset.json` |
| Novel blocks | `_newapproach/nn_modules/block.py` (ARSC/ARSPP/ARGate) |
| Registration | `_newapproach/nn_modules/tasks.py`, `_newapproach/nn_modules/__init__.py` |
| Novel ablation script | `arch_best/training_arch_luggage/run_luggage_arch_novel.py` |
| Combo/speculative script | `arch_best/training_arch_luggage/run_luggage_arch_overnight2.py` |

## 10. Reproduction checklist (training machine, Linux)

1. Copy `nn_modules/{block.py, tasks.py, __init__.py}` → `ultralytics/nn/modules/`.
2. `python run_luggage_arch_novel.py --build-only`  ← verify all 7 construct (~1 min).
3. `python run_luggage_arch_novel.py`  ← full ablation (~8–10 h).
4. Evaluate on `test_full_dataset`, append to the eval JSONs, update §3 of this file.
