#!/usr/bin/env python3
"""
ASSIGNER ISOLATION STUDY — SATAL & SNATAL, everything else OFF.

=============================================================================
WHY
=============================================================================
SATAL and SNATAL are ASSIGNERS (they change which anchors become positives),
never tested in isolation. Every prior SATAL/SNATAL run was stacked on the
SWA-high base (alpha 0.9->0.4, boost 2.0), so their effect was confounded with
SWA. This study runs each assigner on a PURE STOCK-TAL base — SWA off, center
off, clip off, area=inv — so the delta vs the baseline anchor (57.43) is the
assigner's OWN effect, nothing else.

  baseline anchor (all off)         = 57.43 test mAP50-95 (small 52.19)
  -> a run here "helps" iff it clears 57.43 on its own.

If an assigner clears the anchor here, the follow-up is to stack it on the
sqrt winner (57.86). Isolation FIRST — do not stack a piece that fails alone.

=============================================================================
DESIGN — 4 runs, all other phases OFF
=============================================================================
  as_satal_mild     SATAL small a1.5/b3.0, topk x1.5   (gentle scale split)
  as_satal_strong   SATAL small a2.0/b2.0, topk x2.0   (aggressive)
  as_snatal_r025    SNATAL rho 0.25, k_min 2           (balanced supply cut)
  as_snatal_r040    SNATAL rho 0.40, k_min 2           (mild supply cut)

SATAL small/large area thresholds are stride-normalized (0.0025 / 0.0225),
matching loss defaults. SNATAL uses the GEOMETRIC candidate pool + k_min=2
(the reviewed/fixed implementation in lossv2updated.py) so k_eff is stable
from epoch 0 and single-anchor GTs are avoided.

REQUIRES lossv2updated.py installed as ultralytics/utils/loss.py (it has both
SATAL and SNATAL, with the SNATAL geometric-pool fix + k_min=2 default), and
the use_satal / satal_* / use_snatal / snatal_* keys whitelisted in
cfg/default.yaml.

Usage:
  python run_assigner_isolated.py                  # all 4
  python run_assigner_isolated.py as_satal_mild    # subset
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
# CONFIGURATION — must match the baseline/SWA recipe (57.43 / 57.86)
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_assigner_isolated"

EPOCHS = 70
IMG_SIZE = 640            # eval MUST also be 640 (the 896 lesson)
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

BASELINE_TEST_MAP5095 = 0.5743   # anchor (all off), the reference to beat

# =============================================================================
# Everything-off base — pure stock TAL, no SWA / center / clip.
# =============================================================================
_ALL_OFF = dict(
    # SWA off: alpha 0 -> area weight multiplied by 0 -> pure score weighting
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    # center off
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    # clip off
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    # NWD off, DFL-entropy off
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    # stock TAL base params (SATAL/SNATAL sit on these)
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
)


def _satal(alpha_s, beta_s, topk_factor):
    return dict(
        **_ALL_OFF,
        use_satal=True,
        satal_alpha_small=alpha_s, satal_beta_small=beta_s,
        satal_alpha_large=1.0, satal_beta_large=6.0,
        satal_small_area=0.0025, satal_large_area=0.0225,
        satal_topk_factor=topk_factor,
    )


def _snatal(rho):
    return dict(
        **_ALL_OFF,
        use_snatal=True, snatal_rho=rho, snatal_kmin=2,
    )


def _clip(iou_s, iou_e, dfl_s, dfl_e):
    """PHASE C — per-sample loss clipping on the PURE STOCK base.

    lossv2updated.py applies  clamp(max = value / 10.0), so the EFFECTIVE cap
    is one tenth of what is passed here. _ALL_OFF ships 999.0 (eff 99.9, never
    binds), so all four keys are overridden.

    Calibration: at alpha=0 the per-sample weight is just the TAL score weight
    (<= ~1), so per-sample losses live in ~0.1-1.2. Caps MUST sit inside that
    range to fire — the original R0C_CLIP_LOOSE/MID/TIGHT were sized for the
    SWA-high base (peaks 2.5-3.0) and are inert here. Only R0C_CLIP_SOLO was
    scaled correctly; it is reproduced below as _CLIP_MID.

    Schedule: _get_gradient_clip_values() anneals start -> end across training
    (progress = epoch/total_epochs). The epoch is pushed in by the
    on_train_epoch_start callback below, so the anneal is live.
    """
    c = dict(_ALL_OFF)
    c.update(
        use_loss_clip=True,
        iou_clip_start=iou_s, iou_clip_end=iou_e,
        dfl_clip_start=dfl_s, dfl_clip_end=dfl_e,
    )
    return c


# C-1 VERY LOOSE — trims only the extreme tail. THE VALIDITY CHECK: if the
#     [CLIP] rate prints ~0% after one epoch, every config below is inert too
#     and the whole phase is void. Run this one first.
_CLIP_VLOOSE = _clip(12.0, 10.0, 15.0, 12.0)   # eff iou 1.2->1.0 | dfl 1.5->1.2
# C-2 LOOSE — cuts the outlier shoulder.
_CLIP_LOOSE  = _clip(10.0,  7.0, 12.0,  9.0)   # eff iou 1.0->0.7 | dfl 1.2->0.9
# C-3 MID — this is R0C_CLIP_SOLO, the only original config scaled for no-SWA.
_CLIP_MID    = _clip( 8.0,  5.0, 10.0,  7.0)   # eff iou 0.8->0.5 | dfl 1.0->0.7
# C-4 TIGHT — bites into the body of the distribution. Upper dose, not a
#     candidate; included so the dose-response has a known-bad end point.
_CLIP_TIGHT  = _clip( 5.0,  3.0,  7.0,  5.0)   # eff iou 0.5->0.3 | dfl 0.7->0.5


RUNS = [
    # --- Phase 0: in-session anchor. Without this every delta is measured
    #     against an external number from a different script execution.
    {"name": "as_anchor",       "phase": "0",
     "label": "_ALL_OFF, pure stock TAL — in-session reference (must land ~57.43)",
     "params": dict(_ALL_OFF)},
    {"name": "as_satal_mild",   "phase": "E",
     "label": "SATAL small a1.5/b3.0 topkx1.5 — gentle scale split (SWA OFF)",
     "params": _satal(1.5, 3.0, 1.5)},
    {"name": "as_satal_strong", "phase": "E",
     "label": "SATAL small a2.0/b2.0 topkx2.0 — aggressive (SWA OFF)",
     "params": _satal(2.0, 2.0, 2.0)},
    {"name": "as_snatal_r025",  "phase": "N",
     "label": "SNATAL rho=0.25 k_min=2 geometric pool — balanced (SWA OFF)",
     "params": _snatal(0.25)},
    {"name": "as_snatal_r040",  "phase": "N",
     "label": "SNATAL rho=0.40 k_min=2 geometric pool — mild (SWA OFF)",
     "params": _snatal(0.40)},
    # --- Phase C: per-sample loss-clipping dose-response, isolated ---------
    # Never executed in any prior round: the four R0C_* configs were defined
    # in run_newluggage_ablationfirst_tal_clip.py but left commented out of
    # RUNS, and their caps were sized for SWA-high anyway.
    # Doubles as the robust-loss / loss-truncation arm: clipping caps the
    # influence of outlier or mislabelled boxes (label audit: 0.97% class
    # swaps, box jitter unmeasured).
    {"name": "as_clip_vloose",  "phase": "C",
     "label": "clip eff iou 1.2->1.0 / dfl 1.5->1.2 — tail only (VALIDITY CHECK, run first)",
     "params": _CLIP_VLOOSE},
    {"name": "as_clip_loose",   "phase": "C",
     "label": "clip eff iou 1.0->0.7 / dfl 1.2->0.9 — outlier shoulder",
     "params": _CLIP_LOOSE},
    {"name": "as_clip_mid",     "phase": "C",
     "label": "clip eff iou 0.8->0.5 / dfl 1.0->0.7 — R0C_CLIP_SOLO reproduced",
     "params": _CLIP_MID},
    {"name": "as_clip_tight",   "phase": "C",
     "label": "clip eff iou 0.5->0.3 / dfl 0.7->0.5 — upper dose, expect a loss",
     "params": _CLIP_TIGHT},
]


# =============================================================================
# Epoch sync (harmless here — SWA is off — but keeps the loss state consistent)
# =============================================================================
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
    name, params = rc["name"], rc["params"]
    print(f"\n{'=' * 76}\n  RUN {name}  [{rc['phase']}]\n  {rc['label']}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=100, close_mosaic=10, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(params))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "assigner_params.json"), "w") as f:
            json.dump({"name": name, "phase": rc["phase"], "label": rc["label"],
                       "params": params, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                       "batch": BATCH, "seed": SEED}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "phase": rc["phase"], "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    # TEST-split eval at 640 (NEVER 896)
    try:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        tm = YOLO(best_pt).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                               batch=BATCH, device=DEVICE, workers=WORKERS,
                               project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def already_done(name, summary):
    for r in summary:
        if r.get("name") == name and r.get("test_map5095") == r.get("test_map5095"):
            return True
    return False


def load_summary():
    p = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def summarise(summary):
    ref = BASELINE_TEST_MAP5095

    def pc(v):
        return f"{v * 100:.2f}" if v == v else "n/a"

    print(f"\n{'=' * 64}\n  ASSIGNER ISOLATION RESULTS (test split)\n{'=' * 64}")
    print(f"  {'run':<18}{'ph':>3}{'mAP50':>8}{'mAP50-95':>10}{'d_anchor':>10}{'h':>6}")
    print("  " + "-" * 55)
    for r in sorted(summary, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        d = "—"
        if r["test_map5095"] == r["test_map5095"]:
            d = f"{(r['test_map5095'] - ref) * 100:+.2f}"
        hrs = f"{r['hours']:.1f}" if r.get("hours") == r.get("hours") else "n/a"
        print(f"  {r['name']:<18}{str(r.get('phase','?')):>3}{pc(r['test_map50']):>8}"
              f"{pc(r['test_map5095']):>10}{d:>10}{hrs:>6}")
        if r.get("error"):
            print(f"      FAILED: {r['error']}")
    print(f"\n  baseline anchor (all off) = {ref * 100:.2f}")
    print(f"  d_anchor > 0 -> the assigner helps ON ITS OWN (then stack on sqrt 57.86).")


def main():
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 76}")
    print(f"  ASSIGNER ISOLATION (SATAL/SNATAL, all else OFF)  @{IMG_SIZE}px, {EPOCHS}ep")
    print(f"  runs: {', '.join(r['name'] for r in todo)}")
    print(f"  baseline to beat: {BASELINE_TEST_MAP5095 * 100:.2f}")
    print(f"{'=' * 76}")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    summary = load_summary()

    for rc in todo:
        if not only and already_done(rc["name"], summary):
            print(f"\n  [SKIP] {rc['name']} already completed.")
            continue
        try:
            res = run_one(rc)
        except Exception as e:
            print(f"\n  [ERROR] {rc['name']} failed: {e}")
            res = {"name": rc["name"], "phase": rc.get("phase"), "hours": float("nan"),
                   "val_map50": float("nan"), "val_map5095": float("nan"),
                   "test_map50": float("nan"), "test_map5095": float("nan"),
                   "error": str(e)}
        summary = [r for r in summary if r["name"] != res["name"]]
        summary.append(res)
        with open(out_path, "w") as f:   # incremental, crash-safe
            json.dump(summary, f, indent=2)

    summarise(summary)


if __name__ == "__main__":
    main()
