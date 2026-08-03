#!/usr/bin/env python3
"""
Round 9 — 4 mechanism runs + mandatory fresh anchor (5 trainings).

WHY THESE FOUR, given v6/r0/r7/r8 (~30 runs):
  * Every reweighting/assignment variant landed within ~±0.5 mAP50-95 of its
    anchor; the anchor itself drifted 1.18 between rounds (r0_default2 57.43
    vs r8_anchor 56.25). So: (a) a fresh anchor is NON-NEGOTIABLE, (b) only
    mechanisms that are categorically different from reweighting are worth
    GPU time.
  * Diagnosed deficit, stable across all rounds: AR50_small ~0.96 but
    AP50-95_small ~0.51 -> small objects are FOUND but badly RANKED and
    loosely BOXED. The four runs attack exactly that, one lever each:
      r9_vfl        RANKING   — IoU-aware classification (cls_mode='qfl')
      r9_dflboost   EDGES     — DFL-only boost for small objects (originally
                                dfl_small_boost). NOT in loss2.py -> removed;
                                run now equals the anchor at px36.
      r9_nwd        SURFACE   — NWD blend: changes the SHAPE of the loss for
                                tiny tall boxes where IoU is cliff-like; the
                                only run that isn't a gradient rescale
      r9_area_fixed WEIGHTING — originally fixed-ref area weighting (area_mode/
                                area_ref_px/area_gamma/area_w_cap). Those knobs
                                are NOT in loss2.py -> removed; what remains is
                                the constant SWA alpha=0.5 schedule.
  * All four are SINGLE-mechanism runs (r7_stack: composition without
    renormalization cost -4.3 — nothing is stacked here).
  * Clips OFF everywhere: r0/v6 showed clipping is null standalone, and a cap
    would silently absorb the DFL boost (watch clip-rate log = 0.00%).

RECONCILED TO loss2.py (the only loss implementation present in this repo):
  - NWD keys renamed: nwd_ratio -> nwd_weight, nwd_c -> nwd_C; NWD is gated by
    use_nwd (default False).
  - cls_loss -> cls_mode ('bce' | 'qfl'); loss2.py has no VFL, so 'vfl' maps to
    'qfl' and vfl_alpha/vfl_gamma are removed (QFL strength is qfl_beta).
  - Removed (no loss2.py impl): weight_renorm, area_mode, area_ref_px,
    area_gamma, area_w_cap, dfl_small_boost.
  CAUTION: loss2.py's nwd_C is STRIDE-NORMALIZED (default ~4.0), not pixels —
  the legacy 64.0 saturates NWD (~inert). Retune nwd_C to ~2-6 for r9_nwd.

VERIFY AT LAUNCH (config banner, first lines of console):
  r9_anchor     : cls_mode: bce | use_nwd: False | nwd_weight/C: 0.0 / 64.0
  r9_vfl        : cls_mode: qfl  (qfl_beta=2.0)
  r9_dflboost   : small_obj_px: 36  (== anchor; dfl_small_boost removed)
  r9_nwd        : use_nwd: True nwd_mode: blend nwd_weight/C: 0.5 / 64.0, px48
  r9_area_fixed : alpha 0.5/0.5 const SWA (area_* removed), small_obj_px: 48
If a banner shows a default instead, the key was dropped by cfg validation —
ABORT, do not burn the run (r0a2_dflboost lesson).

ANCHOR REPRODUCIBILITY CHECK: r9_anchor uses settings at which every optional
path is inert (alpha=0; use_nwd False; cls_mode bce). With the same
seed/data/imgsz as r8_anchor it should land within normal determinism
tolerance of it; a large gap means the environment changed again.

DECISION RULE (fixed in advance, val-split, before any test eval):
  A mechanism is a CANDIDATE only if val mAP50-95 > anchor + 0.5 OR
  val AP50-95_small > anchor + 0.8. Candidates (and the anchor) then get
  seeds 1 and 2 before any conclusion; test eval happens once, at the end,
  on the multi-seed survivors only (pass --with-test to force earlier).

Usage:
  python run_r9_sweep.py                 # all runs not yet completed
  python run_r9_sweep.py r9_vfl          # only named run(s)
  python run_r9_sweep.py --with-test     # also eval test split per run (discouraged)
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
except ImportError:  # very old ultralytics
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"

PROJECT_DIR = "runs_newluggage_r9"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Shared OFF blocks — every run states EVERY custom param explicitly, so the
# saved ablation_params.json is ground truth (the eval JSONs never were).
# =============================================================================
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_boost=1.0,
    # NOTE: weight_renorm / area_mode / area_ref_px / area_gamma / area_w_cap
    # are NOT implemented in loss2.py and have been removed.
)
# NOTE: dfl_small_boost is NOT in loss2.py (removed). NWD keys renamed to the
# loss2.py names: nwd_ratio -> nwd_weight, nwd_c -> nwd_C. loss2.py gates NWD on
# use_nwd (default False), so the OFF block sets use_nwd=False.
_TARGETED_OFF = dict(use_nwd=False, nwd_weight=0.0, nwd_C=64.0)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
# loss2.py uses cls_mode ('bce' | 'qfl'); it has no VFL, so vfl_* are removed.
_CLS_BCE = dict(cls_mode="bce")

# =============================================================================
# RUN CONFIGS
# =============================================================================

# Fresh anchor — every NEW path inert; the comparison basis for this round
# and the reproducibility check against r8_anchor.
R9_ANCHOR = dict(
    **_SWA_OFF, small_obj_px=48,  # px inert here (boost=1, dfl=1, nwd=0)
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# 1) RANKING — IoU-aware classification, everything else stock.
#    Positives weighted by their soft TAL score (localization quality), so the
#    classifier learns to rank by IoU. Targets the AP50-95 vs AR50 gap without
#    touching regression at all.
#    NOTE: loss2.py has no Varifocal loss; cls_mode='qfl' (Quality Focal Loss)
#    is its IoU-aware option, so cls_loss='vfl' -> cls_mode='qfl' and vfl_alpha/
#    vfl_gamma are removed (loss2.py's QFL strength is qfl_beta, default 2.0).
R9_VFL = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK,
    cls_mode="qfl",
)

# 2) EDGES — DFL-only boost for genuinely small objects.
#    px=36 (area < 36^2 at 640): with tall h/w~2.7 boxes this is roughly the
#    max-side<~47px @640 tail (~ the <38px tail @512, between overview bins A
#    and B ≈ bottom ~25-30% of instances) — targeted, unlike the px48 runs
#    that boosted ~57%. Boost 2.5 on DFL only, alpha=0 so no area blending:
#    isolates "sharper edge distributions for small boxes" from everything
#    Section A already falsified. Clips OFF so the boost cannot be absorbed.
# NOTE: dfl_small_boost is NOT implemented in loss2.py and was removed. With it
# gone this config equals the anchor and no longer tests the DFL-only boost.
R9_DFLBOOST = dict(
    **_SWA_OFF, small_obj_px=36,
    use_nwd=False, nwd_weight=0.0, nwd_C=64.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# 3) SURFACE — NWD blend for small objects (Wang et al. 2021).
#    For area < 48^2 the regression loss is 0.5*(1-CIoU) + 0.5*(1-NWD),
#    NWD in PIXEL space with c=64 (~dataset mean box sqrt(w*h)~61 @640).
#    The only intervention that changes the loss LANDSCAPE where IoU is
#    cliff-like for 20-40px-wide tall boxes, rather than rescaling gradients.
#    px=48 deliberately broader than r9_dflboost: NWD is a smoothing, not a
#    dose, so covering the whole small bin is the honest test.
# NOTE: NWD IS implemented in loss2.py (use_nwd + nwd_mode='blend' + nwd_weight
# + nwd_C). Renamed: nwd_ratio -> nwd_weight, nwd_c -> nwd_C; use_nwd=True turns
# it on. dfl_small_boost removed (not in loss2.py). CAUTION: loss2.py's nwd_C is
# in STRIDE-NORMALIZED coords (default 4.0), not pixels — the old 64.0 pixel
# constant saturates NWD (~1 = inert). Retune nwd_C to ~2-6 for a real test.
R9_NWD = dict(
    **_SWA_OFF, small_obj_px=48,
    use_nwd=True, nwd_mode="blend", nwd_weight=0.5, nwd_C=64.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# 4) WEIGHTING — the r8_area_sqrt signal, re-tested clean.
#    r8_area_sqrt was the ONLY run in ~30 to move small AP50-95 up (+0.59)
#    without a size trade-off — but on the legacy batch-relative weight (noisy,
#    batch-dependent) and single-seed. This is the deterministic version:
#    constant alpha 0.5 blending a fixed-ref sqrt weight (ref=64px, cap 3.0),
#    renormalized (NEW-1) so total magnitude is preserved — a pure
#    redistribution toward smaller boxes. boost=1: one mechanism only.
#    If THIS fails to replicate the r8 gain, area weighting closes for good;
#    if it holds, it is finally attributable to a defined, reproducible rule.
# NOTE: the fixed-ref area-weighting knobs (weight_renorm, area_mode,
# area_ref_px, area_gamma, area_w_cap) are NOT implemented in loss2.py and were
# removed. loss2.py's related knob is area_weight_mode ('inv'|'sqrt'|'log') on
# BboxLoss; the closest surviving mechanism here is the constant SWA alpha
# schedule (alpha 0.5) which still redistributes weight toward smaller boxes.
R9_AREA_FIXED = dict(
    alpha_start=0.5, alpha_end=0.5, alpha_min=0.5, alpha_max=0.5,
    small_obj_px=48, small_obj_boost=1.0,
    **_TARGETED_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER (anchor first — everything is judged against it)
# =============================================================================
RUNS = [
    {"name": "r9_anchor",     "phase": "-",  "label": "Fresh anchor — all optional paths inert; repro check vs r8_anchor",   "params": R9_ANCHOR},
    # {"name": "r9_vfl",        "phase": "E",  "label": "QFL cls (qfl_beta=2.0) — IoU-aware ranking, regression stock",        "params": R9_VFL},
    {"name": "r9_dflboost",   "phase": "A'", "label": "px36 (dfl_small_boost removed -> == anchor in loss2.py)",             "params": R9_DFLBOOST},
    {"name": "r9_nwd",        "phase": "A''","label": "NWD blend weight=0.5 C=64 @px48 — loss-surface change for tiny boxes","params": R9_NWD},
    {"name": "r9_area_fixed", "phase": "A",  "label": "Const SWA alpha=0.5 (area_* removed in loss2.py) — size reweighting", "params": R9_AREA_FIXED},
    # After this round: seeds 1,2 for the anchor + any candidate passing the
    # decision rule in the header. Add entries like:
    # {"name": "r9_anchor_s1", "phase": "-", "label": "anchor seed 1", "params": R9_ANCHOR, "seed": 1},
]


# =============================================================================
# Epoch sync — drives alpha / clip / center-decay schedules in the custom loss
# =============================================================================
def on_train_epoch_start(trainer):
    """Push trainer.epoch into the custom loss, DDP-safe."""
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


def run_one(run_cfg, with_test=False):
    name = run_cfg["name"]
    label = run_cfg["label"]
    params = run_cfg["params"]
    seed = run_cfg.get("seed", SEED)

    print(f"\n{'=' * 70}")
    print(f"  RUN: {name}  (phase {run_cfg.get('phase', '?')}, seed {seed})")
    print(f"  {label}")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    train_kwargs = {
        "data": DATA_YAML,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": PROJECT_DIR,
        "name": name,
        "patience": 100,
        "close_mosaic": 10,
        "seed": seed,
        "deterministic": True,
        "exist_ok": False,
    }
    train_kwargs.update(copy.deepcopy(params))

    results = model.train(**train_kwargs)
    elapsed = (time.time() - start_time) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    # ---- persist ground-truth config next to the run results ----
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "phase": run_cfg.get("phase"), "label": label,
                       "params": params, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                       "batch": BATCH, "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    # ---- VAL-split metrics from training results (selection basis) ----
    val_map50, val_map5095 = float("nan"), float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key])
                break
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if key in rd:
                val_map5095 = float(rd[key])
                break
    except Exception:
        pass

    # ---- optional TEST eval (default OFF: selection must stay on val) ----
    test_map50, test_map5095 = float("nan"), float("nan")
    if with_test:
        try:
            best_pt = os.path.join(save_dir, "weights", "best.pt")
            test_model = YOLO(best_pt)
            tm = test_model.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                                batch=BATCH, device=DEVICE, workers=WORKERS,
                                project=PROJECT_DIR, name=f"{name}_test")
            test_map50 = float(tm.box.map50)
            test_map5095 = float(tm.box.map)
            del test_model, tm
        except Exception as e:
            print(f"  [WARN] test eval failed: {e}")

    # Free GPU memory before the next run
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "phase": run_cfg.get("phase"), "label": label,
            "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "val_map5095": val_map5095,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
    """A run counts as done if its summary entry exists with a val score."""
    path = os.path.join(PROJECT_DIR, "summary.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for r in json.load(f):
                if r.get("name") == name and r.get("val_map5095") == r.get("val_map5095"):
                    return True
    except Exception:
        pass
    return False


def load_summary():
    path = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def main():
    args = [a for a in sys.argv[1:]]
    with_test = "--with-test" in args
    only = {a for a in args if not a.startswith("--")}
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 70}")
    print("  ROUND 9 — anchor + VFL / DFL-boost / NWD / fixed-area (one lever each)")
    print(f"  Runs: {', '.join(r['name'] for r in todo)}")
    if with_test:
        print("  [!] --with-test: test-split eval per run (leaks test into selection)")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in todo:
        if not only and already_done(run_cfg["name"]):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (found in summary.json)")
            continue

        try:
            result = run_one(run_cfg, with_test=with_test)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "phase": run_cfg.get("phase"),
                      "label": run_cfg["label"], "seed": run_cfg.get("seed", SEED),
                      "elapsed_h": float("nan"),
                      "val_map50": float("nan"), "val_map5095": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}

        if result["name"] in done_names:
            summary = [r for r in summary if r["name"] != result["name"]]
        summary.append(result)
        done_names.add(result["name"])

        try:
            os.makedirs(PROJECT_DIR, exist_ok=True)
            with open(os.path.join(PROJECT_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

    total_elapsed = (time.time() - overall_start) / 3600

    print(f"\n{'=' * 70}")
    print(f"  ALL RUNS COMPLETE ({total_elapsed:.2f}h total)")
    print(f"{'=' * 70}")
    print(f"  {'Run':<18}{'Ph':>4}{'Time(h)':>9}{'val mAP50':>11}{'val 50-95':>11}"
          f"{'tst mAP50':>11}{'tst 50-95':>11}")
    print(f"  {'-' * 75}")

    def fmt(v, pct=True):
        if v != v:  # NaN
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    anchor = next((r for r in summary if r["name"] == "r9_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>4}"
                f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
                f"{fmt(r.get('val_map5095', float('nan'))):>11}"
                f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if (anchor and r["name"] != "r9_anchor"
                and r.get("val_map5095") == r.get("val_map5095")
                and anchor.get("val_map5095") == anchor.get("val_map5095")):
            d = (r["val_map5095"] - anchor["val_map5095"]) * 100
            line += f"   ({'+' if d >= 0 else ''}{d:.2f} vs anchor)"
        print(line)
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    print("\n  DECISION RULE: candidate iff val mAP50-95 > anchor+0.5"
          " (or small AP50-95 > anchor+0.8, from the run's own val logs).")
    print("  Candidates AND the anchor -> seeds 1,2 before any conclusion;"
          " test eval once, at the end, on multi-seed survivors only.")


if __name__ == "__main__":
    main()