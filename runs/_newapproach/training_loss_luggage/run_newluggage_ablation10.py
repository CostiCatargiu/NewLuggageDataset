#!/usr/bin/env python3
"""
Round 9b — 3 NEW mechanisms + improved NWD/DFL + anchor (8 trainings).

CONTEXT (30+ prior runs):
  Every reweighting/assignment variant landed within ±0.5 mAP50-95 of anchor.
  Diagnosed deficit (stable across ALL rounds):
    AR50_small ~0.96 but AP50-95_small ~0.51
    → small objects are FOUND but loosely BOXED at strict IoU.

  VFL (cls-side IoU-aware ranking) confirmed null — the gap is REGRESSION
  quality, not classification calibration. This round attacks only the
  regression path, with three new ideas plus improved versions of NWD/DFL.

NEW MECHANISMS IN THIS ROUND:

  r9b_iarw       [Section F, NEW-9] IoU-Aware Regression Weighting.
                  Per-anchor regression boost proportional to (1-IoU).detach():
                  loose boxes get amplified, tight boxes left alone. Unlike area
                  weighting (assumes small=hard), this MEASURES which predictions
                  need work. Self-correcting: as the box improves, boost fades.
                  The STRONGEST candidate — novel, directly targets the deficit.

  r9b_nwd_adapt  [NEW-5b/5c] Adaptive NWD with annealing.
                  Fixes from R9: (a) continuous smallness ramp instead of flat
                  on/off, (b) lower ratio 0.3 (was 0.5), (c) anneal from full
                  NWD early to 10% at end. NWD smooths the loss landscape where
                  IoU is cliff-like for tiny boxes; annealing lets CIoU's tight
                  optimum dominate for final convergence.

  r9b_dfl_gated  [NEW-3b] IoU-gated DFL boost.
                  Fixes from R9: the flat ×2.5 DFL boost is modulated by (1-IoU)
                  per anchor. Small objects with IoU=0.9 get boost ×1.15; those
                  with IoU=0.3 get ×2.05. Focuses edge sharpening where needed.

  r9b_iarw_nwd   IARW + adaptive NWD. The two mechanisms target orthogonal
                  aspects: IARW redistributes regression effort by quality gap;
                  NWD smooths the loss surface shape. They compound safely
                  because NWD changes the loss, IARW rescales it. WITH weight
                  renormalization so composition doesn't shift magnitude.

  r9b_iarw_dfl   IARW + IoU-gated DFL. IARW boosts both IoU and DFL loss on
                  loose predictions; IoU-gated DFL additionally sharpens edge
                  distributions for small objects specifically. They compound
                  because IARW is per-prediction quality-based, DFL boost is
                  per-prediction quality×size-based.

REQUIRES: loss.py with NEW-3b (dfl_iou_gated), NEW-5b/5c (nwd_adaptive,
  nwd_anneal, nwd_anneal_min), and NEW-9 (iarw_gamma) params whitelisted
  in the cfg patch.

VERIFY AT LAUNCH (config banner):
  r9b_anchor     : iarw_gamma=0 | nwd=0 | dfl_boost=1 | all NEW paths inert
  r9b_iarw       : iarw_gamma=2.0, everything else stock
  r9b_iarw_lo    : iarw_gamma=1.0, conservative version
  r9b_nwd_adapt  : nwd_ratio=0.3 adaptive+anneal->0.1, iarw=0
  r9b_dfl_gated  : dfl_small_boost=2.5 iou_gated=1, iarw=0
  r9b_iarw_nwd   : iarw_gamma=2.0 + nwd_ratio=0.3 adaptive+anneal
  r9b_iarw_dfl   : iarw_gamma=2.0 + dfl_small_boost=2.0 iou_gated=1

DECISION RULE (same as R9, fixed before any eval):
  A mechanism is a CANDIDATE only if val mAP50-95 > anchor + 0.5 OR
  val AP50-95_small > anchor + 0.8. Candidates (and the anchor) then get
  seeds 1 and 2 before any conclusion; test eval happens once, at the end,
  on the multi-seed survivors only (pass --with-test to force earlier).

Usage:
  python run_newluggage_ablation9.py                 # all runs not yet completed
  python run_newluggage_ablation9.py r9b_iarw        # only named run(s)
  python run_newluggage_ablation9.py --with-test     # also eval test split per run (discouraged)
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

PROJECT_DIR = "runs_newluggage_r10"

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
    # NEW-1/NEW-2: inert at alpha=0 (renorm is identity, area weight unused)
    weight_renorm=1, area_mode="fixed", area_ref_px=64.0, area_gamma=0.5,
    area_w_cap=3.0,
)
_TARGETED_OFF = dict(
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
)
_IARW_OFF = dict(iarw_gamma=0.0)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
_CLS_BCE = dict(cls_loss="bce", vfl_alpha=0.75, vfl_gamma=2.0)

# =============================================================================
# RUN CONFIGS
# =============================================================================

# Fresh anchor — every NEW path inert; the comparison basis for this round.
R9B_ANCHOR = dict(
    **_SWA_OFF, small_obj_px=48,  # px inert here (boost=1, dfl=1, nwd=0, iarw=0)
    **_TARGETED_OFF, **_IARW_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 1) IARW — IoU-Aware Regression Weighting (Section F, NEW-9)
#    THE primary candidate. Per-anchor regression boost = 1 + gamma*(1-IoU):
#      IoU=0.3 -> boost 2.4x   (loose box: try harder)
#      IoU=0.7 -> boost 1.6x   (OK box: moderate push)
#      IoU=0.9 -> boost 1.2x   (tight box: leave alone)
#    Applied to BOTH IoU loss and DFL loss. Self-correcting: as the box
#    tightens, the boost fades. Doesn't assume small=hard — measures directly.
#    gamma=2.0 is a moderate value; higher values (3.0) focus more aggressively.
# ---------------------------------------------------------------------------
R9B_IARW = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, iarw_gamma=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# Conservative IARW: gamma=1.0 (weaker boost for comparison)
R9B_IARW_LO = dict(
    **_SWA_OFF, small_obj_px=48,
    **_TARGETED_OFF, iarw_gamma=1.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 2) ADAPTIVE NWD WITH ANNEALING (Section A'', NEW-5b/5c)
#    Fixes from R9's flat r=0.5:
#      - ratio 0.3 (gentler: NWD is a stabilizer, not a replacement)
#      - adaptive: tiny objects get full ratio, near-threshold get ~0 (continuous)
#      - anneal: full NWD early -> 10% of ratio at end of training
#    NWD smooths the cliff-like CIoU landscape for 20-40px-wide tall boxes;
#    annealing ensures CIoU's tighter optimum dominates for final convergence.
# ---------------------------------------------------------------------------
R9B_NWD_ADAPT = dict(
    **_SWA_OFF, small_obj_px=48,
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.3, nwd_c=64.0, nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
    **_IARW_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 3) IoU-GATED DFL BOOST (Section A', NEW-3b)
#    Fixes from R9's flat x2.5:
#      - IoU-gated: boost modulated by (1-IoU), so well-localized small objects
#        are left alone. IoU=0.3 -> ×2.05; IoU=0.9 -> ×1.15.
#      - px=36 (genuinely small only, ~bottom 25-30% of instances)
# ---------------------------------------------------------------------------
R9B_DFL_GATED = dict(
    **_SWA_OFF, small_obj_px=36,
    dfl_small_boost=2.5, dfl_iou_gated=1,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
    **_IARW_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 4) COMPOSITION: IARW + Adaptive NWD
#    IARW redistributes regression effort by per-prediction quality gap.
#    NWD smooths the loss surface shape for small objects.
#    Orthogonal: NWD changes WHAT is optimized; IARW changes HOW MUCH.
#    weight_renorm=1 ensures composition doesn't shift total magnitude.
# ---------------------------------------------------------------------------
R9B_IARW_NWD = dict(
    **_SWA_OFF, small_obj_px=48,
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.3, nwd_c=64.0, nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
    iarw_gamma=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# ---------------------------------------------------------------------------
# 5) COMPOSITION: IARW + IoU-gated DFL
#    IARW boosts both IoU and DFL for loose predictions globally.
#    IoU-gated DFL additionally sharpens edge distributions for SMALL objects.
#    IARW is quality-only; DFL gated is quality×size. They compound cleanly.
#    Lower DFL boost (2.0) since IARW already amplifies loose-box DFL.
# ---------------------------------------------------------------------------
R9B_IARW_DFL = dict(
    **_SWA_OFF, small_obj_px=36,
    dfl_small_boost=2.0, dfl_iou_gated=1,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
    iarw_gamma=2.0,
    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK, **_CLS_BCE,
)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER (anchor first — everything is judged against it)
# =============================================================================
RUNS = [
    {"name": "r9b_anchor",    "phase": "-",    "label": "Fresh anchor — all NEW paths inert",                                 "params": R9B_ANCHOR},
    {"name": "r9b_iarw",      "phase": "F",    "label": "IARW gamma=2.0 — IoU-aware regression weighting (primary candidate)","params": R9B_IARW},
    {"name": "r9b_iarw_lo",   "phase": "F",    "label": "IARW gamma=1.0 — conservative variant",                              "params": R9B_IARW_LO},
    {"name": "r9b_nwd_adapt", "phase": "A''",  "label": "Adaptive NWD r=0.3 + anneal — improved loss surface for small",       "params": R9B_NWD_ADAPT},
    {"name": "r9b_dfl_gated", "phase": "A'",   "label": "IoU-gated DFL boost 2.5 @px36 — targeted edge sharpening",           "params": R9B_DFL_GATED},
    {"name": "r9b_iarw_nwd",  "phase": "F+A''","label": "IARW 2.0 + adaptive NWD 0.3 — quality redistribution + surface fix", "params": R9B_IARW_NWD},
    {"name": "r9b_iarw_dfl",  "phase": "F+A'", "label": "IARW 2.0 + IoU-gated DFL 2.0 @px36 — quality + edge compound",      "params": R9B_IARW_DFL},
    # After this round: seeds 1,2 for the anchor + any candidate passing the
    # decision rule in the header. Add entries like:
    # {"name": "r9b_anchor_s1", "phase": "-", "label": "anchor seed 1", "params": R9B_ANCHOR, "seed": 1},
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
    print("  ROUND 9b — IARW + adaptive NWD + IoU-gated DFL (regression-only attacks)")
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

    anchor = next((r for r in summary if r["name"] == "r9b_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>4}"
                f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
                f"{fmt(r.get('val_map5095', float('nan'))):>11}"
                f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if (anchor and r["name"] != "r9b_anchor"
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