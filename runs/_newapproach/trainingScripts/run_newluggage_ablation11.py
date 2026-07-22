#!/usr/bin/env python3
"""
Round 11 — GEOMETRY + ASSIGNMENT (change WHAT is optimized, not the weight).

WHY THIS ROUND
  R2-R10 (~50 runs) all reweighted the SAME CIoU+DFL signal and all landed in
  the noise band. The one thing that ever nudged the needle was R10's
  nwd_fixedc -> NWD (a scale-robust distance) beat CIoU's small-box cliff.
  R11 pulls that thread: stop reweighting CIoU, and change the geometry / the
  assignment metric themselves.

MECHANISMS (all in loss_v1updated.py, off by default):
  [NEW-15] box_metric = 'eiou' | 'siou' — replace CIoU's degenerate aspect
           v-term with DIRECT w/h penalties (EIoU) or an angle-aligned
           distance+shape cost (SIoU). Targets tall/narrow luggage (AR~2.58)
           whose width/height CIoU regresses weakly.
  [NEW-16] rel_l1_weight — scale-NORMALIZED residual |pred-gt|/gt_dim, so small
           boxes get proportionally larger gradient (unlike R10's absolute L1).
  [NEW-17] tal_nwd — scale-aware TAL assignment: blend NWD into the alignment
           overlap for small GTs so tiny/tall boxes get better-aligned anchors.
           Changes WHICH predictions are supervised toward each small object.

  Plus two NWD re-tunings that reuse the existing adaptive-NWD machinery
  (the R10 winner), pushing ratio / temperature-k further.

RUNS (ordered by expected value; anchor first — everything is judged vs it):
  r11_anchor        all inert (box_metric=ciou, rel_l1=0, tal_nwd=0)
  r11_talnwd        NWD-aware assignment (small GTs), c=8 area=48^2 ratio=1.0
  r11_eiou          base metric = EIoU
  r11_siou          base metric = SIoU
  r11_nwd_k03       R10 nwd_fixedc but temperature k=0.3 (sharper small regime)
  r11_nwd_full      near-full NWD for the smallest bin (ratio 0.7, no anneal)
  r11_relL1         scale-normalized residual, small-only, w=0.05
  r11_eiou_talnwd   best geometry x best assignment (compose)

REQUIRES — whitelist these NEW keys in your cfg patch (box_metric is a STRING,
  like cls_loss):
    box_metric, rel_l1_weight, rel_l1_small_only,
    tal_nwd, tal_nwd_c, tal_nwd_area, tal_nwd_ratio
  (the R10 keys must already be whitelisted.)

VERIFY AT LAUNCH — epoch-0 banner:
    [R11] box_metric: <ciou|eiou|siou>
    [R11] rel_l1_weight: <..>
    [R11] tal_nwd: <0|1 ...>
  For r11_talnwd, also confirm NO "[WARN] ... no iou_calculation hook" line —
  that warning means your ultralytics version won't apply NWD assignment.

DECISION RULE (fixed before eval): candidate iff val mAP50-95 > anchor+0.5 OR
  val AP50-95_small > anchor+0.8. Candidates + anchor -> seeds 1,2; test once.

Usage:
  python run_newluggage_ablation11.py                # all runs
  python run_newluggage_ablation11.py r11_eiou       # only named run(s)
  python run_newluggage_ablation11.py --with-test    # also eval test (discouraged)
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
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_newluggage_r11"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Full OFF baseline — every custom key stated so the anchor is bit-inert.
# =============================================================================
_ALL_OFF = dict(
    # Section A (SWA / area)
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=48, small_obj_boost=1.0,
    weight_renorm=1, area_mode="fixed", area_ref_px=64.0, area_gamma=0.5, area_w_cap=3.0,
    # Section A'/A'' (targeted DFL / NWD blend)
    dfl_small_boost=1.0, dfl_iou_gated=0,
    nwd_ratio=0.0, nwd_c=64.0, nwd_adaptive=0, nwd_anneal=0, nwd_anneal_min=0.1,
    # Section F (IARW)
    iarw_gamma=0.0,
    # Round 10
    alpha_iou=1.0,
    l1_aux_weight=0.0, l1_aux_beta=2.0, l1_aux_small_only=1,
    l1_balanced=0, l1_balanced_alpha=0.5, l1_balanced_gamma=1.5,
    dfl_entropy_weight=0.0, dfl_entropy_small_only=1,
    nwd_c_adaptive=0, nwd_c_k=0.5,
    tightness_gamma=0.0, tightness_small_only=1,
    # Round 11
    box_metric="ciou",
    rel_l1_weight=0.0, rel_l1_small_only=1,
    tal_nwd=0, tal_nwd_c=8.0, tal_nwd_area=2304.0, tal_nwd_ratio=1.0,
    # Section B (center)
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    # Section C (clips)
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    # Section D (TAL) / E (cls)
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_loss="bce", vfl_alpha=0.75, vfl_gamma=2.0,
)


def _base(**overrides):
    cfg = copy.deepcopy(_ALL_OFF)
    cfg.update(overrides)
    return cfg


# =============================================================================
# RUN CONFIGS
# =============================================================================
R11_ANCHOR      = _base()
R11_TALNWD      = _base(tal_nwd=1, tal_nwd_c=8.0, tal_nwd_area=2304.0, tal_nwd_ratio=1.0)
R11_EIOU        = _base(box_metric="eiou")
R11_SIOU        = _base(box_metric="siou")
# reuse adaptive-NWD machinery (R10 winner) with sharper temperature
R11_NWD_K03     = _base(nwd_ratio=0.3, nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
                        nwd_c_adaptive=1, nwd_c_k=0.3)
# near-full NWD for the smallest bin, no anneal
R11_NWD_FULL    = _base(nwd_ratio=0.7, nwd_adaptive=1, nwd_anneal=0,
                        nwd_c_adaptive=1, nwd_c_k=0.5)
R11_RELL1       = _base(rel_l1_weight=0.05, rel_l1_small_only=1)
R11_EIOU_TALNWD = _base(box_metric="eiou", tal_nwd=1, tal_nwd_c=8.0,
                        tal_nwd_area=2304.0, tal_nwd_ratio=1.0)

RUNS = [
    # {"name": "r11_anchor",      "phase": "-",     "label": "Fresh anchor — all NEW-15..17 inert",                  "params": R11_ANCHOR},
    {"name": "r11_talnwd",      "phase": "R11-D'","label": "NWD-aware TAL assignment (small GTs)",                 "params": R11_TALNWD},
    {"name": "r11_eiou",        "phase": "R11-G", "label": "EIoU base loss — direct w/h penalty",                 "params": R11_EIOU},
    {"name": "r11_siou",        "phase": "R11-G", "label": "SIoU base loss — angle+shape cost",                   "params": R11_SIOU},
    {"name": "r11_nwd_k03",     "phase": "R11-A''","label": "Adaptive NWD, temperature k=0.3 (sharper)",          "params": R11_NWD_K03},
    {"name": "r11_nwd_full",    "phase": "R11-A''","label": "Near-full NWD smallest bin (r=0.7, no anneal)",       "params": R11_NWD_FULL},
    {"name": "r11_relL1",       "phase": "R11-I", "label": "Scale-normalized residual (small), w=0.05",           "params": R11_RELL1},
    {"name": "r11_eiou_talnwd", "phase": "R11-GD'","label": "EIoU x NWD-assignment — geometry + assignment",       "params": R11_EIOU_TALNWD},
]


# =============================================================================
# Epoch sync
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


def run_one(run_cfg, with_test=False):
    name = run_cfg["name"]; label = run_cfg["label"]; params = run_cfg["params"]
    seed = run_cfg.get("seed", SEED)

    print(f"\n{'=' * 70}\n  RUN: {name}  (phase {run_cfg.get('phase', '?')}, seed {seed})\n  {label}\n{'=' * 70}\n")
    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    train_kwargs = {
        "data": DATA_YAML, "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
        "device": DEVICE, "workers": WORKERS, "project": PROJECT_DIR, "name": name,
        "patience": 100, "close_mosaic": 10, "seed": seed,
        "deterministic": True, "exist_ok": False,
    }
    train_kwargs.update(copy.deepcopy(params))

    results = model.train(**train_kwargs)
    elapsed = (time.time() - start_time) / 3600
    print(f"\n  TRAIN DONE: {name} ({elapsed:.2f}h)")

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "phase": run_cfg.get("phase"), "label": label,
                       "params": params, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                       "batch": BATCH, "seed": seed}, f, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save params json: {e}")

    val_map50, val_map5095 = float("nan"), float("nan")
    try:
        rd = getattr(results, "results_dict", {}) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in rd:
                val_map50 = float(rd[key]); break
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if key in rd:
                val_map5095 = float(rd[key]); break
    except Exception:
        pass

    test_map50, test_map5095 = float("nan"), float("nan")
    if with_test:
        try:
            best_pt = os.path.join(save_dir, "weights", "best.pt")
            test_model = YOLO(best_pt)
            tm = test_model.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                                batch=BATCH, device=DEVICE, workers=WORKERS,
                                project=PROJECT_DIR, name=f"{name}_test")
            test_map50 = float(tm.box.map50); test_map5095 = float(tm.box.map)
            del test_model, tm
        except Exception as e:
            print(f"  [WARN] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "phase": run_cfg.get("phase"), "label": label,
            "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "val_map5095": val_map5095,
            "test_map50": test_map50, "test_map5095": test_map5095}


def already_done(name):
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

    print(f"\n{'=' * 70}\n  ROUND 11 — geometry (EIoU/SIoU) + scale-aware assignment + NWD re-tune")
    print(f"  Runs: {', '.join(r['name'] for r in todo)}")
    if with_test:
        print("  [!] --with-test: test-split eval per run (leaks test into selection)")
    print(f"{'=' * 70}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in todo:
        if not only and already_done(run_cfg["name"]):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (in summary.json)")
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
    print(f"\n{'=' * 70}\n  ALL RUNS COMPLETE ({total_elapsed:.2f}h total)\n{'=' * 70}")
    print(f"  {'Run':<18}{'Ph':>7}{'Time(h)':>9}{'val mAP50':>11}{'val 50-95':>11}"
          f"{'tst mAP50':>11}{'tst 50-95':>11}")
    print(f"  {'-' * 78}")

    def fmt(v, pct=True):
        if v != v:
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    anchor = next((r for r in summary if r["name"] == "r11_anchor"), None)
    for r in sorted(summary, key=lambda x: x["name"]):
        line = (f"  {r['name']:<18}{str(r.get('phase', '?')):>7}"
                f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
                f"{fmt(r.get('val_map5095', float('nan'))):>11}"
                f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if (anchor and r["name"] != "r11_anchor"
                and r.get("val_map5095") == r.get("val_map5095")
                and anchor.get("val_map5095") == anchor.get("val_map5095")):
            d = (r["val_map5095"] - anchor["val_map5095"]) * 100
            line += f"   ({'+' if d >= 0 else ''}{d:.2f} vs anchor)"
        print(line)
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    print("\n  DECISION RULE: candidate iff val mAP50-95 > anchor+0.5"
          " (or small AP50-95 > anchor+0.8). Candidates + anchor -> seeds 1,2.")


if __name__ == "__main__":
    main()
