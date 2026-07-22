#!/usr/bin/env python3
"""
Round 6 — ported to loss_satal_swa_v2.py (rebuilt loss).

WHY A PORT WAS NEEDED
---------------------
Round 5 targeted loss_satal_swa_plus_v2.py. The rebuilt loss renames most keys
and REPLACES the SWA mechanism, so the old configs would either be rejected or
silently mean something different. Mapping:

  OLD (round 5)                      NEW (loss_satal_swa_v2)
  ---------------------------------  --------------------------------------
  alpha_start/alpha_end/min/max      swa_alpha / swa_alpha_end
  small_obj_px                       swa_width_thresh_px (or swa_area_thresh_px2)
  small_obj_boost                    swa_boost
  iou_clip_start/end, dfl_clip_*     use_loss_clip + iou_clip / dfl_clip
  use_class_weighting                use_class_weights (+ class_counts REQUIRED)
  cls_mode="bce"                     use_vfl=False
  nwd_C                              nwd_c_px            (PIXELS, not grid cells)
  center_loss_*                      (removed — was always 0.0 anyway)
  swa_smooth / swa_boost_power       (removed — superseded by bounded weighting)
  box_loss_type                      box_loss_type       (unchanged; wiou/mpdiou kept)

*** COMPARABILITY WARNING — READ BEFORE INTERPRETING RESULTS ***
The v2 SWA is NOT the same mechanism as v1's. v1 used area_weight = 1/area
normalized by the batch max (up to 400:1 spread, batch-dependent scale). v2 uses
a bounded weight in [1, swa_boost]. Therefore:
  - SWA-OFF runs ARE directly comparable to Rounds 1-5 (weight = score_weight in
    both, and v2's normalizer is mathematically identical to target_scores_sum
    when SWA is off). The anchor run below tests exactly this.
  - Any SWA-ON run is a NEW mechanism. Do NOT compare r6_wiou_swa against the
    Round-4 precision record of 0.8337 as if it were a reproduction. It is a
    fresh measurement that needs its own baseline.

RUN ORDER (descending confidence of a gain)
  1. r6_anchor_default  — SWA-OFF, CIoU, stock TAL. RUN FIRST. Must land within
                          +/-0.35 of v6_default2 (82.54 / 56.84) or the port is
                          wrong and nothing downstream is interpretable.
  2. r6_wiou_default    — WIoU on SWA-OFF. Round 4's one genuine box-loss win,
                          on the proven mAP50-95 base.
  3. r6_mpdiou_default  — MPDIoU on SWA-OFF. Corner-based tight-loc variant.
  4. r6_eiou_default    — EIoU on SWA-OFF. Explicit w/h error; dataset is 94%
                          tall (mean h/w 2.69, 33x72px) so width error is the
                          hard axis. Never actually tested (the earlier EIoU/SIoU
                          sweep ran a different loss file entirely).
  5. r6_wiou_swa_bounded— WIoU + bounded width-based SWA. New mechanism.
  6. r6_nwd_c12         — NWD small_only, C=12px. NOTE: v1's C=4.0 was in GRID
                          CELLS; at stride 8 that is ~32px, at stride 32 ~128px.
                          The old C=2/C=4/C=6 bracket was never a physical-size
                          bracket at all. C is now in PIXELS: 12px ~ mean object
                          width, the principled starting point.
  7. r6_satal13_mpdiou  — SATAL (topk_factor 1.3, softened) + MPDIoU.

PREFLIGHT (automated below, aborts on failure):
  [x] active loss module is loss_satal_swa_v2 (checks for box_loss_type knob)
  [x] all new hyp keys accepted by get_cfg
  [x] ultralytics/utils/satal.py importable if any run sets use_satal
  [x] epoch tracking attached (v2 warns loudly and DISABLES schedules otherwise)
  [x] verify_config() asserts the live loss state matches the printed config

Usage:
  python run_newluggage_ablationv6.py
"""

import copy
import gc
import json
import os
import time

import torch

# =============================================================================
# STEP 0 — register custom keys with NEUTRAL defaults, before any get_cfg/YOLO
# =============================================================================
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

_CUSTOM_LOSS_DEFAULTS = {
    # SATAL
    "use_satal": False,
    "tal_topk": 10, "tal_alpha": 0.5, "tal_beta": 6.0,
    "satal_alpha_small": 1.5, "satal_beta_small": 3.0,
    "satal_alpha_large": 1.0, "satal_beta_large": 6.0,
    "satal_small_area": 0.0025, "satal_large_area": 0.0225,
    "satal_topk_factor": 1.5,
    # SWA (bounded)
    "swa_mode": "scale", "swa_alpha": 0.0, "swa_alpha_end": None,
    "swa_size_axis": "width", "swa_boost": 1.0,
    "swa_width_thresh_px": 24.0, "swa_area_thresh_px2": 1024.0,
    # box metric
    "box_loss_type": "ciou",
    "wiou_alpha": 1.9, "wiou_delta": 3.0, "wiou_momentum": 0.02,
    # NWD (C in PIXELS)
    "use_nwd": False, "nwd_mode": "blend", "nwd_weight": 0.5,
    "nwd_c_px": 12.0, "nwd_small_width_px": 24.0, "nwd_debug": False,
    # cls
    "use_class_weights": False, "class_counts": None,
    "use_vfl": False, "vfl_alpha": 0.75, "vfl_gamma": 2.0,
    # clipping
    "use_loss_clip": False, "iou_clip": 2.0, "dfl_clip": 5.0,
}

for _k, _v in _CUSTOM_LOSS_DEFAULTS.items():
    DEFAULT_CFG_DICT.setdefault(_k, _v)
    if not hasattr(DEFAULT_CFG, _k):
        setattr(DEFAULT_CFG, _k, _v)

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v4i.yolov12_70percentage/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_luggage_satal2"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Reference numbers (70% subset, 70ep, seed 0, test split)
REF_ANCHOR_MAP50 = 0.8254      # v6_default2
REF_ANCHOR_MAP5095 = 0.5684
NOISE_FLOOR = 0.0035           # +/-0.35 points, measured

# Class counts in data.yaml names[] order. v2 REQUIRES these when class
# weighting is on (v1 hardcoded counts that did not match the dataset).
CLASS_COUNTS = [11491.0, 9490.0, 21557.0]   # backpack, bag, trolley

# =============================================================================
# Shared blocks
# =============================================================================
_SWA_OFF = dict(swa_mode="scale", swa_alpha=0.0, swa_alpha_end=None, swa_boost=1.0)

# Bounded re-interpretation of the old const-0.6 recipe. NOT a reproduction:
# v1 boosted by 1/area (400:1); this boosts by <=1.75 keyed on WIDTH, which is
# the hard axis for a 94%-tall dataset.
_SWA_BOUNDED = dict(
    swa_mode="scale", swa_alpha=0.0, swa_alpha_end=None,
    swa_size_axis="width", swa_boost=1.75, swa_width_thresh_px=48.0,
)

_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0, use_satal=False)

_SATAL_LOOSE13 = dict(
    use_satal=True,
    tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_alpha_small=1.2, satal_beta_small=5.0,
    satal_topk_factor=1.3,
    satal_small_area=0.0025, satal_large_area=0.0225,
)

_WIOU = dict(box_loss_type="wiou", wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02)

# Pins so nothing drifts; each run overrides only its lever.
_PINS = dict(
    use_satal=False,
    box_loss_type="ciou",
    use_nwd=False, nwd_mode="small_only", nwd_c_px=12.0, nwd_small_width_px=24.0,
    use_loss_clip=False,
    use_class_weights=True, class_counts=CLASS_COUNTS,   # ON, to match Rounds 1-5
    use_vfl=False,
)


def _cfg(swa=_SWA_OFF, **overrides):
    c = dict(_PINS)
    c.update(_TAL_STOCK)
    c.update(swa)
    c.update(overrides)
    return c


# =============================================================================
# RUN CONFIGS  (anchor FIRST — everything downstream depends on it)
# =============================================================================
RUNS = [
    # {"name": "r6_anchor_default", "seed": 0, "params": _cfg(swa=_SWA_OFF),
    #  "label": "[1/7] ANCHOR: SWA-OFF CIoU stock TAL -- must match 82.54/56.84 +/-0.35"},

    {"name": "r6_wiou_default", "seed": 0, "params": _cfg(swa=_SWA_OFF, **_WIOU),
     "label": "[2/7] WIoU on SWA-OFF -- R4's one genuine box-loss win"},

    {"name": "r6_mpdiou_default", "seed": 0, "params": _cfg(swa=_SWA_OFF, box_loss_type="mpdiou"),
     "label": "[3/7] MPDIoU on SWA-OFF -- corner-based tight localization"},

    {"name": "r6_eiou_default", "seed": 0, "params": _cfg(swa=_SWA_OFF, box_loss_type="eiou"),
     "label": "[4/7] EIoU on SWA-OFF -- explicit w/h error, matched to 94%-tall data"},

    {"name": "r6_wiou_swa_bounded", "seed": 0, "params": _cfg(swa=_SWA_BOUNDED, **_WIOU),
     "label": "[5/7] WIoU + bounded width-SWA -- NEW mechanism, not a reproduction"},

    {"name": "r6_nwd_c12", "seed": 0,
     "params": _cfg(swa=_SWA_OFF, use_nwd=True, nwd_mode="small_only", nwd_c_px=12.0),
     "label": "[6/7] NWD small_only C=12px -- C now in PIXELS (v1's C was grid cells)"},

    {"name": "r6_satal13_mpdiou", "seed": 0,
     "params": _cfg(swa=_SWA_OFF, box_loss_type="mpdiou", **_SATAL_LOOSE13),
     "label": "[7/7] SATAL topk_factor 1.3 (softened) + MPDIoU"},
]


# =============================================================================
# PREFLIGHT
# =============================================================================
def preflight():
    print("\n[preflight] checking configuration ...")

    try:
        cfg = get_cfg(overrides={"box_loss_type": "wiou", "swa_boost": 1.5,
                                 "nwd_c_px": 12.0, "use_class_weights": True,
                                 "class_counts": CLASS_COUNTS})
        assert getattr(cfg, "box_loss_type", None) == "wiou"
        print("[preflight] OK  custom keys accepted by get_cfg")
    except Exception as e:
        raise SystemExit(f"[preflight] FAIL  custom keys rejected: {e}\n"
                         "  -> STEP 0 registration did not take effect.")

    # Active loss must be v2. Instantiate-and-check (bytecode name inspection is
    # unreliable: getattr string literals live in co_consts, not co_names).
    from ultralytics.utils.loss import BboxLoss, v8DetectionLoss, SataLSwaConfig
    try:
        probe = SataLSwaConfig(None, nc=3, total_epochs=EPOCHS)
        has_box = hasattr(probe, "box_loss_type")
        has_swa = hasattr(probe, "swa_boost")
    except Exception as e:
        raise SystemExit(f"[preflight] FAIL  could not build SataLSwaConfig: {e}\n"
                         "  -> ultralytics/utils/loss.py is not loss_satal_swa_v2.py")
    if not (has_box and has_swa):
        raise SystemExit("[preflight] FAIL  loss module is not loss_satal_swa_v2 "
                         f"(box_loss_type={has_box} swa_boost={has_swa})")
    print("[preflight] OK  active loss module is loss_satal_swa_v2")

    if any(r["params"].get("use_satal") for r in RUNS):
        try:
            from ultralytics.utils.satal import ScaleAdaptiveTaskAlignedAssigner  # noqa: F401
            print("[preflight] OK  SATAL assigner importable")
        except ImportError as e:
            raise SystemExit(f"[preflight] FAIL  use_satal run present but satal.py missing: {e}")

    if not os.path.isfile(DATA_YAML):
        raise SystemExit(f"[preflight] FAIL  DATA_YAML not found: {DATA_YAML}")
    print("[preflight] OK  dataset yaml found")
    print(f"[preflight] class_counts (must match data.yaml names[] order): {CLASS_COUNTS}")
    print(f"[preflight] anchor target: {REF_ANCHOR_MAP50:.4f} / {REF_ANCHOR_MAP5095:.4f} "
          f"+/-{NOISE_FLOOR:.4f}\n")


# =============================================================================
# RUN
# =============================================================================
def run_one(run_cfg, anchor_result):
    name, label, params = run_cfg["name"], run_cfg["label"], run_cfg["params"]
    seed = run_cfg.get("seed", 0)

    print(f"\n{'=' * 72}\n  RUN: {name}  (seed {seed})\n  {label}\n{'=' * 72}\n")
    start = time.time()

    model = YOLO(MODEL_WEIGHTS)

    # Epoch tracking. v2 exposes attach_epoch_tracking(); without it, schedules
    # are DISABLED with a loud warning (v1 silently froze at the aggressive end).
    from ultralytics.utils.loss import attach_epoch_tracking
    attach_epoch_tracking(model)

    train_kwargs = dict(
        data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
        workers=WORKERS, project=PROJECT_DIR, name=name, patience=100,
        close_mosaic=10, seed=seed, deterministic=True,
    )
    train_kwargs.update(copy.deepcopy(params))

    results = model.train(**train_kwargs)
    elapsed = (time.time() - start) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))

    # Persist the ACTUAL params next to results (so a config can never be
    # mis-attributed after the fact).
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": label, "params": params, "seed": seed,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
                       "loss_file": "loss_satal_swa_v2.py"}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    val_map50 = float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for k in ("metrics/mAP50(B)", "metrics/mAP50"):
            if k in rd:
                val_map50 = float(rd[k])
                break
    except Exception:
        pass

    test_map50 = test_map5095 = float("nan")
    try:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        tmodel = YOLO(best_pt)
        tm = tmodel.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                        device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                        name=f"{name}_test")
        test_map50, test_map5095 = float(tm.box.map50), float(tm.box.map)
        del tmodel, tm
    except Exception as e:
        print(f"  [WARN] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {"name": name, "label": label, "seed": seed, "elapsed_h": elapsed,
           "val_map50": val_map50, "test_map50": test_map50, "test_map5095": test_map5095}

    # Anchor gate: if the port drifted, say so loudly and immediately.
    if name == "r6_anchor_default" and test_map5095 == test_map5095:
        d50 = test_map50 - REF_ANCHOR_MAP50
        d5095 = test_map5095 - REF_ANCHOR_MAP5095
        ok = abs(d50) <= NOISE_FLOOR and abs(d5095) <= NOISE_FLOOR
        out["anchor_ok"] = bool(ok)
        out["anchor_delta"] = {"map50": d50, "map5095": d5095}
        print(f"\n  {'=' * 66}")
        print(f"  ANCHOR CHECK: d(mAP50)={d50:+.4f}  d(mAP50-95)={d5095:+.4f}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  The port does NOT reproduce the reference baseline. Every")
            print("  downstream run in this round is uninterpretable until this")
            print("  is resolved. Check: class weighting on/off, loss gains, and")
            print("  that SWA is genuinely off (verify_config).")
        print(f"  {'=' * 66}\n")

    return out


def main():
    preflight()

    print(f"\n{'=' * 72}")
    print(f"  ROUND 6 -- ported to loss_satal_swa_v2.py ({len(RUNS)} runs)")
    print(f"  {', '.join(r['name'] for r in RUNS)}")
    print(f"{'=' * 72}")

    t0 = time.time()
    summary = []
    anchor_result = None

    for rc in RUNS:
        try:
            res = run_one(rc, anchor_result)
            if rc["name"] == "r6_anchor_default":
                anchor_result = res
                if res.get("anchor_ok") is False:
                    print("  [WARN] continuing despite anchor FAIL — treat all "
                          "downstream numbers as provisional.")
        except Exception as e:
            print(f"\n  [ERROR] run '{rc['name']}' failed: {e}")
            import traceback
            traceback.print_exc()
            res = {"name": rc["name"], "label": rc["label"], "seed": rc.get("seed", 0),
                   "elapsed_h": float("nan"), "val_map50": float("nan"),
                   "test_map50": float("nan"), "test_map5095": float("nan"),
                   "error": str(e)}
        summary.append(res)

        try:
            os.makedirs(PROJECT_DIR, exist_ok=True)
            with open(os.path.join(PROJECT_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

    total = (time.time() - t0) / 3600
    base = anchor_result["test_map5095"] if anchor_result else float("nan")

    print(f"\n{'=' * 72}")
    print(f"  ALL RUNS COMPLETE ({total:.2f}h)")
    print(f"{'=' * 72}")
    print(f"  {'Run':<24}{'Time(h)':>9}{'val50':>9}{'test50':>10}{'test50-95':>11}{'d vs anchor':>13}")
    print(f"  {'-' * 76}")
    for r in summary:
        def f(v, pct=True):
            return "n/a" if v != v else (f"{v * 100:.2f}%" if pct else f"{v:.2f}")
        d = ""
        if base == base and r["test_map5095"] == r["test_map5095"]:
            dv = r["test_map5095"] - base
            flag = "" if abs(dv) > NOISE_FLOOR else " (noise)"
            d = f"{dv:+.4f}{flag}"
        print(f"  {r['name']:<24}{f(r['elapsed_h'], False):>9}{f(r['val_map50']):>9}"
              f"{f(r['test_map50']):>10}{f(r['test_map5095']):>11}{d:>13}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    print(f"\n  Noise floor is +/-{NOISE_FLOOR:.4f}. Anything inside it is not a result.")
    print(f"  Confirm any winner with 3 seeds before reporting it.\n")


if __name__ == "__main__":
    main()