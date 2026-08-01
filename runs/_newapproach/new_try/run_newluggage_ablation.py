

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
PROJECT_DIR = "runs_newluggage5_r0"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Shared blocks
# =============================================================================
# SWA fully off, px=0 (safe ONLY when the center loss is also off)
_SWA_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0,
)
# SWA weighting off but px kept alive at 48 — REQUIRED for Phase B runs,
# because the center loss reads the same small_obj_px threshold.
_SWA_OFF_KEEP_PX48 = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=48, small_obj_boost=1.0,
)
_CENTER_OFF = dict(
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
)
_CLIP_OFF = dict(
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
)
_TAL_STOCK = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# High-alpha SWA base reused by Phase C (run as r0a_swa_a09_04 = C control)
_SWA_HIGH = dict(alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
                 small_obj_px=48, small_obj_boost=2.0)

# =============================================================================
# DEFAULT — all four sections off / stock
# =============================================================================
R0_DEFAULT = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# PHASE A — SWA alpha schedules (px=48, boost 2.0; min/max inert per schedule)
# =============================================================================
R0A_SWA_A09_04 = dict(**_SWA_HIGH,
                      **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)   # also Phase-C control
R0A_SWA_A07_03 = dict(alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
                      small_obj_px=48, small_obj_boost=2.0,
                      **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
R0A_SWA_A05_025 = dict(alpha_start=0.5, alpha_end=0.25, alpha_min=0.25, alpha_max=0.5,
                       small_obj_px=48, small_obj_boost=2.0,
                       **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# Same two SWA schedules, but reshape the area weight from 1/area ('inv') to
# sqrt(1/area). 'inv' concentrates weight on the tiniest boxes; 'sqrt' spreads
# emphasis over small+medium (dataset areas span ~28x, p10..p90).
R0A_SWA_A09_04_SQRT = dict(alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
                           small_obj_px=48, small_obj_boost=2.0, area_weight_mode='sqrt',
                           **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
R0A_SWA_A07_03_SQRT = dict(alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
                           small_obj_px=48, small_obj_boost=2.0, area_weight_mode='sqrt',
                           **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)


# =============================================================================
# PHASE A2 — SWA follow-up: four DISTINCT mechanisms (not more of the same).
# Success criterion, fixed in advance: AP50-95_small must improve by >= +0.5
# over the default; otherwise Section A closes with a negative result.
# =============================================================================
# A2-1 Targeted scope: boost only the truly tiny (~18% of instances)
R0A2_PX24_TGT = dict(alpha_start=0.9, alpha_end=0.4, alpha_min=0.4, alpha_max=0.9,
                     small_obj_px=24, small_obj_boost=2.0, dfl_small_boost=1.0,
                     **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-2 High contrast, no decay: max dose on the tiny bin only, held all run
R0A2_PX24_HI = dict(alpha_start=0.7, alpha_end=0.7, alpha_min=0.7, alpha_max=0.7,
                    small_obj_px=24, small_obj_boost=4.0, dfl_small_boost=1.0,
                    **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-3 Inverted (rising) schedule: size emphasis during the LATE refinement
# phase (incl. the close_mosaic window) instead of the early chaotic phase
R0A2_RISE = dict(alpha_start=0.2, alpha_end=0.8, alpha_min=0.2, alpha_max=0.8,
                 small_obj_px=36, small_obj_boost=2.0, dfl_small_boost=1.0,
                 **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)
# A2-4 DFL-only boost: alpha=0 (no area blending), boost ONLY the DFL term for
# small objects -> targets box-edge precision, the diagnosed deficit
# (AP50_small 0.79 vs AP50-95_small 0.52 with AR50_small ~0.96).
# REQUIRES the dfl_small_boost patch in loss.py.
R0A2_DFLBOOST = dict(alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
                     small_obj_px=36, small_obj_boost=1.0, dfl_small_boost=2.5,
                     **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK)

# =============================================================================
# PHASE B — center loss ISOLATED (SWA weighting off, px kept at 48)
# =============================================================================
R0B_CENTER_W05 = dict(
    **_SWA_OFF_KEEP_PX48, **_CLIP_OFF, **_TAL_STOCK,
    center_loss_weight_init=0.5, center_loss_weight_min=0.01, center_loss_decay_epochs=35,
)
R0B_CENTER_W10 = dict(
    **_SWA_OFF_KEEP_PX48, **_CLIP_OFF, **_TAL_STOCK,
    center_loss_weight_init=1.0, center_loss_weight_min=0.01, center_loss_decay_epochs=35,
)

# =============================================================================
# PHASE C — clipping dose-response on the HIGH-alpha SWA base
#           (control = r0a_swa_a09_04; effective cap = value/10)
# =============================================================================
R0C_CLIP_LOOSE = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=30.0, iou_clip_end=20.0,   # eff. 3.0 -> 2.0
    dfl_clip_start=35.0, dfl_clip_end=25.0,   # eff. 3.5 -> 2.5
)
R0C_CLIP_MID = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=20.0, iou_clip_end=12.0,   # eff. 2.0 -> 1.2
    dfl_clip_start=25.0, dfl_clip_end=15.0,   # eff. 2.5 -> 1.5
)
R0C_CLIP_TIGHT = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=10.0, iou_clip_end=6.0,    # eff. 1.0 -> 0.6
    dfl_clip_start=12.0, dfl_clip_end=8.0,    # eff. 1.2 -> 0.8
)
# ISOLATED C — clipping WITHOUT SWA (compare vs r0_default).
# At alpha=0 the weight is just the TAL score weight (<= ~1), so per-sample
# losses live in ~0.1-1.2: caps must sit INSIDE that range to ever fire
# (the C-on-A cap values would be inert here, as Round 1 showed).
# Distinguishes "clipping helps in general" from "clipping only counteracts
# the spikes Section A introduces".
R0C_CLIP_SOLO = dict(
    **_SWA_OFF, **_CENTER_OFF, **_TAL_STOCK,
    iou_clip_start=8.0, iou_clip_end=5.0,     # eff. 0.8 -> 0.5
    dfl_clip_start=10.0, dfl_clip_end=7.0,    # eff. 1.0 -> 0.7
)

# =============================================================================
# PHASE D — TAL: quantity axis vs composition axis (everything else off)
# =============================================================================
R0D_TAL_TOPK6 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=6, tal_alpha=0.5, tal_beta=6.0)
R0D_TAL_BETA4 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=10, tal_alpha=0.5, tal_beta=4.0)
R0D_TAL_TOOD = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                    tal_topk=10, tal_alpha=1.0, tal_beta=6.0)
# Harden IoU emphasis (ratio 12->16): select positives more by overlap
# quality -> train regression on well-localized anchors. Mirror of beta4;
# together they give both directions of the composition axis.
R0D_TAL_BETA8 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                     tal_topk=10, tal_alpha=0.5, tal_beta=8.0)
# Loose direction of the quantity axis: with topk6 and stock 10 this makes a
# 3-point dose-response curve (6/10/13). Geometry note: tiny boxes (~20x45 @640)
# only have ~11 in-box candidates, so topk13 adds positives mostly to
# medium/large objects (lower-aligned ones). Expected at-or-below stock -- run
# to close the axis, not because it is favored.
R0D_TAL_TOPK13 = dict(**_SWA_OFF, **_CENTER_OFF, **_CLIP_OFF,
                      tal_topk=13, tal_alpha=0.7, tal_beta=4.0)

# =============================================================================
# PHASE E — SA-TAL (Scale-Adaptive Task-Aligned Assigner)
#           Different assigner params for small vs large objects:
#           small -> higher alpha (lean on cls), lower beta (soft IoU),
#                    more positives (topk x factor); large -> stock-like.
#           Built on the SWA-high base so SWA weighting stays active.
#           NOTE: requires ultralytics/utils/satal.py (ScaleAdaptiveTaskAlignedAssigner).
# =============================================================================
# E-1 Moderate scale split: gentle small-object relaxation
R0E_SATAL_MILD = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,     # large/base behavior
    use_satal=True,
    satal_alpha_small=1.5, satal_beta_small=3.0,  # small: cls-heavy, soft IoU
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_small_area=0.0025, satal_large_area=0.0225,
    satal_topk_factor=1.5,                        # small objects get 1.5x positives
)
# E-2 Aggressive scale split: stronger relaxation + more small-object positives
R0E_SATAL_STRONG = dict(
    **_SWA_HIGH, **_CENTER_OFF, **_CLIP_OFF,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    use_satal=True,
    satal_alpha_small=2.0, satal_beta_small=2.0,  # even more cls-driven, very soft IoU
    satal_alpha_large=1.0, satal_beta_large=6.0,
    satal_small_area=0.0025, satal_large_area=0.0225,
    satal_topk_factor=2.0,                        # small objects get 2x positives
)

# =============================================================================
# PHASE N — SUPPLY-NORMALIZED TAL (SNA-TAL)
#           Per-GT budget k_eff = clamp(round(rho * pool), k_min, topk).
#           Directly targets the anchor-footprint diagnosis: small objects are
#           supply-limited and forced onto a diluted positive set (54.3%
#           selectivity vs 4.3% for large). rho sweep mirrors report table 6:
#           small-object taken/GT should drop 8.64 -> ~2.5 / 3.9 / 5.8.
#           Built on SWA-high base to match the Phase-E SATAL comparison.
#           NOTE: requires use_snatal support in loss.py (Section N).
# =============================================================================
_TAL_STOCK_ABG = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
R0N_SNATAL_R015 = dict(**_SWA_HIGH, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK_ABG,
                       use_snatal=True, snatal_rho=0.15, snatal_kmin=1)
R0N_SNATAL_R025 = dict(**_SWA_HIGH, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK_ABG,
                       use_snatal=True, snatal_rho=0.25, snatal_kmin=1)
R0N_SNATAL_R040 = dict(**_SWA_HIGH, **_CENTER_OFF, **_CLIP_OFF, **_TAL_STOCK_ABG,
                       use_snatal=True, snatal_rho=0.40, snatal_kmin=1)

# =============================================================================
# RUNS TO EXECUTE, IN ORDER
# =============================================================================
RUNS = [
    # --- Phase A: SWA alpha schedules (2 configs) ---
    {"name": "r0a_swa_a09_04",   "phase": "A", "label": "SWA 0.9->0.4, px48, boost 2.0 — strong/long",                     "params": R0A_SWA_A09_04},
    {"name": "r0a_swa_a07_03",   "phase": "A", "label": "SWA 0.7->0.3, px48, boost 2.0 — medium",                          "params": R0A_SWA_A07_03},
    # --- Phase A (area-weight shape = sqrt): same schedules, sqrt(1/area) ---
    {"name": "r0a_swa_a09_04_sqrt", "phase": "A", "label": "SWA 0.9->0.4, px48, boost 2.0, area=sqrt — small+medium spread", "params": R0A_SWA_A09_04_SQRT},
    {"name": "r0a_swa_a07_03_sqrt", "phase": "A", "label": "SWA 0.7->0.3, px48, boost 2.0, area=sqrt — small+medium spread", "params": R0A_SWA_A07_03_SQRT},
    # --- Phase E: SA-TAL scale-adaptive assigner (2 configs) ---
    {"name": "r0e_satal_mild",   "phase": "E", "label": "SA-TAL small a1.5/b3.0 topkx1.5 (on SWA-high) — mild scale split",   "params": R0E_SATAL_MILD},
    {"name": "r0e_satal_strong", "phase": "E", "label": "SA-TAL small a2.0/b2.0 topkx2.0 (on SWA-high) — strong scale split", "params": R0E_SATAL_STRONG},
    # --- Phase N: supply-normalized TAL (targets diagnosed dilution) ---
    {"name": "r0n_snatal_r015",  "phase": "N", "label": "SNA-TAL rho=0.15 (on SWA-high) — aggressive supply cut",             "params": R0N_SNATAL_R015},
    {"name": "r0n_snatal_r025",  "phase": "N", "label": "SNA-TAL rho=0.25 (on SWA-high) — balanced supply cut",               "params": R0N_SNATAL_R025},
    {"name": "r0n_snatal_r040",  "phase": "N", "label": "SNA-TAL rho=0.40 (on SWA-high) — mild supply cut",                   "params": R0N_SNATAL_R040},
]


# =============================================================================
# Epoch sync — drives alpha / clip / center-decay schedules in the custom loss
# =============================================================================
def on_train_epoch_start(trainer):
    """Push trainer.epoch into the custom loss, DDP-safe.

    The loss reads self._model.current_epoch on the *unwrapped* DetectionModel,
    so set attributes via de_parallel(trainer.model), never on a DDP wrapper.
    criterion.epoch is set directly as a second path (criterion lives on the
    model in current ultralytics; older versions kept it on the trainer).
    """
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


def run_one(run_cfg):
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

    # ---- val-split mAP50 from training results ----
    val_map50 = float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key])
                break
    except Exception:
        pass

    # ---- explicit TEST-split evaluation on best.pt ----
    test_map50, test_map5095 = float("nan"), float("nan")
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
            "seed": seed, "elapsed_h": elapsed, "val_map50": val_map50,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
    """A run counts as done if its summary entry exists with a test score."""
    path = os.path.join(PROJECT_DIR, "summary.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for r in json.load(f):
                if r.get("name") == name and r.get("test_map50") == r.get("test_map50"):
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
    only = set(sys.argv[1:])  # optional: run only named runs
    todo = [r for r in RUNS if (not only or r["name"] in only)]

    print(f"\n{'=' * 70}")
    print("  ROUND 0 SWEEP — default + A (SWA alpha) + B (center) + C (clips) + D (TAL)")
    print(f"  Runs: {', '.join(r['name'] for r in todo)}")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in todo:
        if not only and already_done(run_cfg["name"]):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (found in summary.json)")
            continue

        try:
            result = run_one(run_cfg)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "phase": run_cfg.get("phase"),
                      "label": run_cfg["label"], "seed": run_cfg.get("seed", SEED),
                      "elapsed_h": float("nan"), "val_map50": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}

        # replace stale entry if re-running, else append
        if result["name"] in done_names:
            summary = [r for r in summary if r["name"] != result["name"]]
        summary.append(result)
        done_names.add(result["name"])

        # incremental summary dump — survives a crash mid-study
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
    print(f"  {'Run':<24}{'Ph':>3}{'Time(h)':>9}{'val mAP50':>11}{'test mAP50':>12}{'test 50-95':>12}")
    print(f"  {'-' * 71}")

    def fmt(v, pct=True):
        if v != v:  # NaN
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    for r in sorted(summary, key=lambda x: x["name"]):
        print(f"  {r['name']:<24}{str(r.get('phase', '?')):>3}"
              f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
              f"{fmt(r['test_map50']):>12}{fmt(r['test_map5095']):>12}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")


if __name__ == "__main__":
    main()