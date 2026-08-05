#!/usr/bin/env python3
"""
LB-TAL ISOLATION STUDY — level-balanced per-level top-k, everything else OFF.

=============================================================================
WHY LB-TAL (and not SATAL / SNATAL)
=============================================================================
diag_anchor_footprint.py (v6i val) decomposed the small-object assignment
pathology into SUPPLY vs METRIC and found:

  * SEL BIAS ~= 1.0 at P3/P4  -> the alignment METRIC is already level-neutral.
    Re-weighting it (SATAL alpha/beta, LBA-on-metric) has little to fix.
  * Small GTs are FORCED to accept ~60% of a tiny candidate pool (12.9 -> 7.73)
    while large GTs cream the best ~1.5% of 657 -> global pooled top-k lets
    large objects dominate small GTs' positive slots.
  * BUT the forced small-object extras are GOOD (mean IoU 0.806, 1.3% marginal)
    -> CUTTING supply (SNA-TAL) removes useful signal. That is why SNA-TAL and
    SATAL are PREDICTED to underperform, matching their v5i failures.

The diagnostic's own recommendation (Section 1/2): "Fix by allocating topk PER
LEVEL, not globally." LB-TAL does exactly that: it splits the topk budget per
pyramid level and selects within each level, so coarse levels get a guaranteed
share and the budget can be biased toward the fine levels where small objects
live. It RE-ALLOCATES the same budget (total capped at topk) — no supply cut,
no metric change.

=============================================================================
DESIGN — isolate LB-TAL on the pure-stock base
=============================================================================
Base = _ALL_OFF (stock CIoU+DFL+BCE, stock TAL topk10/a0.5/b6.0, gains
7.5/0.5/1.5, class weighting OFF). Only the assigner's top-k ALLOCATION changes,
so the delta vs the v6i anchor is LB-TAL's own effect.

Runs (PRIMARY — LB-TAL, the contribution):
  lb_prop     proportional  — k_level ~ live per-level candidate share (adaptive)
  lb_uniform  uniform       — equal budget per level (per-level-vs-pooled control)
  lb_fixed_632 fixed {8:6,16:3,32:2} — hand P3 six / P4 three / P5 two
  lb_fixed_442 fixed {8:4,16:4,32:2} — flatter, push more budget up to P4/P5

Runs (COMPARISON ARMS — SATAL / SNA-TAL, v6i-refitted, NOT the contribution):
  cmp_satal_mild   SATAL a1.5/b3.0 topkx1.5   — the "scale-adaptive metric" alt
  cmp_snatal_r025  SNA-TAL rho0.25 k_min2     — the "supply-cut" alt
  These populate the ablation table so the paper can show LB-TAL beats the
  obvious alternatives. The diagnostic predicts both are ~flat/negative on v6i;
  running them once confirms it. They are mutually exclusive with LB-TAL (the
  loss raises if >1 assigner is on) so they run as their OWN isolated jobs.

READ THE SIZE BUCKETS, not just overall mAP. The mechanism targets AP_small /
AR_small; run CocoEvalAllFolders_luggage.py (or diagnostic) on best.pt for the
per-size numbers — overall mAP can be flat while small moves.

VERIFY FIRST: run  python selftest_lbtal.py  — all checks must PASS.
Then, to confirm the mechanism actually moved the allocation, re-run
diag_anchor_footprint.py pointing WEIGHTS at an LB-TAL best.pt and check that
P5 sel-share rose and small-object selectivity fell.

REQUIRES lossv2updated.py installed as ultralytics/utils/loss.py (it has
LevelBalancedTaskAlignedAssigner + the use_lbtal / lbtal_* keys), and those keys
whitelisted in cfg/default.yaml.

Usage:
    python run_lbtal_isolated.py            # all runs in RUNS
    python run_lbtal_isolated.py lb_prop    # a subset
"""

import sys
import time
import gc
import copy
import json
import os

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_lbtal_isolated"

EPOCHS = 70
IMG_SIZE = 640            # eval MUST also be 640 (the 896 lesson)
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100

# v6i anchor (yolov12s pure stock) — the reference to beat. Set to the ms_s
# test mAP50-95 from run_model_scale.py. The baseline JSON in this folder shows
# yolov12s_default = 0.5477 test mAP50-95; use your in-house anchor if different.
BASELINE_TEST_MAP5095 = 0.5477

# =============================================================================
# Everything-off base — pure stock TAL. LB-TAL sits on top by only changing the
# top-k ALLOCATION; nothing else here evaluates True.
# =============================================================================
_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
    # LB-TAL off by default in the base
    use_lbtal=False,
)


def _lbtal(mode, level_topk=None, min_level_k=1):
    return dict(
        _ALL_OFF,
        use_lbtal=True,
        lbtal_mode=mode,
        lbtal_level_topk=level_topk,
        lbtal_min_level_k=min_level_k,
    )


def _satal(alpha_s, beta_s, topk_factor):
    """SATAL comparison arm, with v6i-refitted constants.

    alpha_large / beta_large match STOCK (0.5 / 6.0) so the large-object branch
    is a true control (t=1 reduces SATAL to stock). area thresholds are the v6i
    re-fit (small 0.005 / large 0.030) — object_area / letterboxed-640^2. These
    match the defaults now baked into lossv2updated.py / satal.py.
    """
    return dict(
        _ALL_OFF,
        use_satal=True,
        satal_alpha_small=alpha_s, satal_beta_small=beta_s,
        satal_alpha_large=0.5, satal_beta_large=6.0,
        satal_small_area=0.005, satal_large_area=0.030,
        satal_topk_factor=topk_factor,
    )


def _snatal(rho):
    """SNA-TAL comparison arm (scale-free; no v6i re-fit needed)."""
    return dict(
        _ALL_OFF,
        use_snatal=True, snatal_rho=rho, snatal_kmin=2,
    )


def _cls_swa(boost):
    """cls-SWA: boost the CLASSIFICATION bce for small-object fg anchors.

    Section K of loss2. Unlike SWA (which reweights BOX loss and only moved
    small +0.64, within noise, because small-object LOCALIZATION is already good
    — AR50_small 0.95, IoU 0.806), cls-SWA attacks the actual small-object
    bottleneck: SCORING. The baseline shows small objects are FOUND but DROPPED
    (AR50_small 0.951 vs R50_small 0.703). cls-SWA raises the cls-loss pressure
    on small positives so their confidence rises and more survive the operating
    threshold. Small = area_feat < (small_obj_px/stride)^2, with small_obj_px=36
    (v6i-refit), so 'small' is the v6i small minority, not the majority.

    WATCH R50_small (best-F1 recall), not just mAP — if it climbs from 0.703
    toward AR50_small 0.951, the loss closed the scoring gap. mAP may move less
    than recall because this is a ranking/threshold fix, not a localization one.

    CRITICAL: _ALL_OFF sets small_obj_px=0, which makes cls-SWA's small test
    (area < (0/stride)^2) NEVER fire -> the run would be a silent no-op == the
    anchor. So we MUST set small_obj_px here. 36 is the v6i-refit (36^2=1296px^2
    ~0.6x mean object area) — the v6i small minority, matching Section A's scope.
    """
    return dict(_ALL_OFF, use_cls_swa=True, cls_swa_boost=boost,
                small_obj_px=36)


def _swa(a0, a1, boost, px=48, mode="sqrt"):
    """SWA (Section A) box-loss weighting — the sqrt0703 winner is _swa(0.7,0.3,2.0).

    NOTE: SWA and cls-SWA share the small_obj_px key. When STACKING them below,
    both use the SAME px, so 'small' means one consistent thing across box+cls.
    sqrt0703 used px=48; the isolated cls-SWA runs use px=36. For a clean combo,
    pick ONE px for both (px48 keeps the sqrt0703 winner unchanged).
    """
    return dict(
        _ALL_OFF,
        alpha_start=a0, alpha_end=a1, alpha_min=a1, alpha_max=a0,
        small_obj_px=px, small_obj_boost=boost, area_weight_mode=mode,
    )


def _combo(**overrides):
    """Merge extra keys onto _ALL_OFF for stacked mechanisms."""
    return dict(_ALL_OFF, **overrides)


RUNS = [
    # =========================================================================
    # PRIMARY — LB-TAL (the contribution). Per-level top-k allocation, the
    # mechanism the anchor-footprint diagnostic pointed to.
    # =========================================================================
    {"name": "lb_prop", "batch": BATCH,
     "label": "LB-TAL proportional (k_level ~ live candidate share) — adaptive",
     "params": _lbtal("proportional")},

    {"name": "lb_uniform", "batch": BATCH,
     "label": "LB-TAL uniform (equal budget/level) — per-level-vs-pooled control",
     "params": _lbtal("uniform")},

    {"name": "lb_fixed_632", "batch": BATCH,
     "label": "LB-TAL fixed {8:6,16:3,32:2} — bias budget to fine levels",
     "params": _lbtal("fixed", level_topk={8: 6, 16: 3, 32: 2})},

    {"name": "lb_fixed_442", "batch": BATCH,
     "label": "LB-TAL fixed {8:4,16:4,32:2} — flatter, more to P4/P5",
     "params": _lbtal("fixed", level_topk={8: 4, 16: 4, 32: 2})},

    # =========================================================================
    # SECOND SMALL-OBJECT CANDIDATE — cls-SWA (Section K). SCORING-side, not
    # assignment. Directly targets the diagnosed 0.95->0.70 "found but not
    # scored" gap by boosting the classification loss on small-object positives.
    # Complementary to LB-TAL (assignment) and independent of SWA (box weight,
    # which the dose-response showed does NOT move small). READ R50_small.
    # =========================================================================
    {"name": "clsswa_b175", "batch": BATCH,
     "label": "cls-SWA boost 1.75 (px36) — scoring-side small-object fix",
     "params": _cls_swa(1.75)},

    {"name": "clsswa_b20", "batch": BATCH,
     "label": "cls-SWA boost 2.0 (px36) — stronger scoring pressure",
     "params": _cls_swa(2.0)},

    # =========================================================================
    # COMPARISON ARMS — SATAL & SNA-TAL, v6i re-fitted. NOT the contribution;
    # these are the "obvious alternatives" the paper must show LB-TAL beats.
    # The anchor-footprint diagnostic PREDICTS both underperform (metric already
    # level-neutral -> SATAL has little to fix; forced extras are good IoU 0.806
    # -> SNA-TAL cuts useful signal). One run each confirms the prediction and
    # populates the ablation table. Separate runs — they are mutually exclusive
    # with LB-TAL and cannot be stacked (the loss raises on >1 assigner).
    # =========================================================================
    {"name": "cmp_satal_mild", "batch": BATCH,
     "label": "[cmp] SATAL small a1.5/b3.0 topkx1.5 — v6i-refit gentle (predicted ~flat)",
     "params": _satal(1.5, 3.0, 1.5)},

    {"name": "cmp_satal_strong", "batch": BATCH,
     "label": "[cmp] SATAL small a2.0/b2.0 topkx2.0 — v6i-refit aggressive (predicted ~flat)",
     "params": _satal(2.0, 2.0, 2.0)},

    {"name": "cmp_snatal_r025", "batch": BATCH,
     "label": "[cmp] SNA-TAL rho=0.25 k_min=2 geometric pool — balanced (predicted ~flat/neg)",
     "params": _snatal(0.25)},

    {"name": "cmp_snatal_r040", "batch": BATCH,
     "label": "[cmp] SNA-TAL rho=0.40 k_min=2 geometric pool — mild cut (predicted ~flat/neg)",
     "params": _snatal(0.40)},

    # =========================================================================
    # COMBINED RUNS — DO NOT RUN UNTIL THE ISOLATED NUMBERS ARE IN.
    # =========================================================================
    # Uncomment these ONLY after you have each mechanism's isolated v6i delta,
    # so a combined result is attributable (e.g. "SWA +0.87, cls-SWA +X, LB-TAL
    # +Y, together +Z"). Stacking before isolating gives a number with no
    # attribution — the confound this project's methodology exists to avoid.
    #
    # COMPATIBILITY:
    #   * SWA (box) + cls-SWA (cls) + LB-TAL (assignment) touch THREE DIFFERENT
    #     loss stages, so all three stack cleanly — no mutual-exclusion.
    #   * Only ONE assigner may be on: LB-TAL here excludes SATAL/SNA-TAL/AR-TAL.
    #   * px CONSISTENCY: SWA and cls-SWA share small_obj_px. The combos below
    #     fix BOTH to px48 (keeps the sqrt0703 winner intact) OR px36 — pick one
    #     and keep it identical across the two SWA-family terms, else "small"
    #     means two different things in the same run.
    #   * WHICH LB-TAL variant: fill in the winning lbtal_mode/level_topk from
    #     the isolated LB-TAL runs before enabling the LB-TAL combos.
    #
    # ---- Tier 1: the two proven-so-far small boosters (box + cls) ----------
    # {"name": "cmb_swa0703_clsswa175", "batch": BATCH,
    #  "label": "COMBO SWA sqrt0.7->0.3 b2.0 px48 + cls-SWA 1.75 (px48) — box+cls small boost",
    #  "params": _combo(
    #      alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    #      area_weight_mode="sqrt", small_obj_boost=2.0,
    #      use_cls_swa=True, cls_swa_boost=1.75,
    #      small_obj_px=48)},   # ONE px for both SWA and cls-SWA
    #
    # {"name": "cmb_swa0703_clsswa20", "batch": BATCH,
    #  "label": "COMBO SWA sqrt0.7->0.3 b2.0 px48 + cls-SWA 2.0 (px48)",
    #  "params": _combo(
    #      alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    #      area_weight_mode="sqrt", small_obj_boost=2.0,
    #      use_cls_swa=True, cls_swa_boost=2.0,
    #      small_obj_px=48)},
    #
    # ---- Tier 2: add the winning LB-TAL assigner on top --------------------
    # REPLACE lbtal_mode / lbtal_level_topk with the isolated LB-TAL winner.
    # {"name": "cmb_lbtal_clsswa175", "batch": BATCH,
    #  "label": "COMBO LB-TAL(winner) + cls-SWA 1.75 (px36) — assignment + scoring",
    #  "params": _combo(
    #      use_lbtal=True, lbtal_mode="proportional", lbtal_level_topk=None,
    #      use_cls_swa=True, cls_swa_boost=1.75, small_obj_px=36)},
    #
    # {"name": "cmb_lbtal_swa0703", "batch": BATCH,
    #  "label": "COMBO LB-TAL(winner) + SWA sqrt0.7->0.3 b2.0 px48 — assignment + box",
    #  "params": _combo(
    #      use_lbtal=True, lbtal_mode="proportional", lbtal_level_topk=None,
    #      alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    #      area_weight_mode="sqrt", small_obj_boost=2.0, small_obj_px=48)},
    #
    # ---- Tier 3: the full small-object stack (box + cls + assignment) -------
    # The paper's "all combined" row — only after each part is validated.
    # {"name": "cmb_all_small", "batch": BATCH,
    #  "label": "COMBO SWA + cls-SWA + LB-TAL(winner) — full small-object stack",
    #  "params": _combo(
    #      use_lbtal=True, lbtal_mode="proportional", lbtal_level_topk=None,
    #      alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    #      area_weight_mode="sqrt", small_obj_boost=2.0,
    #      use_cls_swa=True, cls_swa_boost=1.75,
    #      small_obj_px=48)},   # ONE px shared by SWA + cls-SWA
]


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            try:
                crit.epoch = epoch
                if hasattr(crit, "_sync_bbox_loss_state"):
                    crit._sync_bbox_loss_state()
            except Exception:
                pass


def run_one(rc):
    name, params, batch = rc["name"], rc["params"], rc["batch"]
    print(f"\n{'=' * 76}\n  RUN {name}\n  {rc['label']}\n"
          f"  model={MODEL_WEIGHTS}  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    # write ablation_params.json so CocoEvalAllFolders_luggage.py can read the config
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch,
                       "seed": SEED}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "batch": batch, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    try:
        tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
            data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
        # NOTE: per-size buckets need CocoEvalAllFolders_luggage.py on best.pt —
        # that is where the small-object effect this mechanism targets shows up.
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    key, key50 = "test_map5095", "test_map50"
    ref = BASELINE_TEST_MAP5095
    print(f"\n{'=' * 76}\n  LB-TAL ISOLATION RESULTS (test split)\n{'=' * 76}")
    print(f"{'run':<14}{'batch':>6}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x[key] if x[key] == x[key] else -9)):
        d = ("%+10.2f" % ((r[key] - ref) * 100)) if ref else "%10s" % "—"
        print(f"{r['name']:<14}{r['batch']:>6}{r[key50] * 100:>9.2f}"
              f"{r[key] * 100:>11.2f}{d}{r['hours']:>6.1f}")
    print(f"\n  v6i anchor (all off) = {ref * 100:.2f}")
    print("  d_anchor > 0 -> per-level top-k helps ON ITS OWN.")
    print("  Then read per-SIZE buckets (CocoEvalAllFolders_luggage.py) — the")
    print("  small-object effect is the point, and can move while overall is flat.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]

    print(f"\n{'=' * 76}")
    print(f"  LB-TAL ISOLATION  @{IMG_SIZE}px, {EPOCHS}ep, stock base (all else OFF)")
    print(f"  anchor to beat: {BASELINE_TEST_MAP5095 * 100:.2f}")
    print(f"  runs: {', '.join(r['name'] for r in todo)}")
    print(f"{'=' * 76}\n")
    print("  REMINDER: run  python selftest_lbtal.py  first — all checks must PASS.\n")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")

    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "batch": r["batch"], "hours": float("nan"),
                        "val_map50": float("nan"), "val_map5095": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

    summarise(res, out_path)
