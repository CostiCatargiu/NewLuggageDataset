#!/usr/bin/env python3
"""
AR-DFL ablation — the one genuinely untried loss axis (box REPRESENTATION).

=============================================================================
WHY THIS EXISTS
=============================================================================
~60 prior loss configs (SATAL, SWA, NWD, every IoU variant, VFL/QFL, bag
penalty, repulsion, cls-SWA, center loss, clipping, box jitter, IARW,
area-weight modes, AR-aware TAL) ALL plateaued on mAP50-95. The R9/R10
changelogs concluded gradient RESCALING is null: the model converges to the
same optimum regardless of the weighting path.

Every one of those changed *which samples matter* or *which IoU flavour* is
used. NONE changed the box REPRESENTATION. The 25pt mAP50->mAP50-95 gap is a
box-tightness problem, and stock DFL quantizes all 4 box edges into the SAME
16-bin grid. For 94%-tall objects (mean AR 2.69) the HEIGHT edges (top,bottom)
carry the large, hard-to-localize range while WIDTH edges (left,right) are
short and easy. Symmetric per-edge DFL therefore under-serves exactly the axis
that dominates the residual error.

AR-DFL reweights the per-edge DFL loss toward the HEIGHT edges (and, optionally,
sharpens only the height-edge distributions). It is:
  - orthogonal to NWD (the one prior mod that helped) -> built ON TOP of NWD,
  - a representation change, NOT another rescaling of the same signal,
  - zero architecture change (same reg_max, same head).

bbox2dist edge order = (left, top, right, bottom):
    width  edges = columns [0, 2]
    height edges = columns [1, 3]

=============================================================================
CONFIG KNOBS (whitelisted in loss.py::SataLSwaConfig)
=============================================================================
  use_ardfl        master switch
  ardfl_h_weight   multiplier on height-edge (top,bottom) DFL   (>1)
  ardfl_w_weight   multiplier on width-edge  (left,right) DFL   (<=1)
  ardfl_ar_gate    apply only to boxes with GT h/w > ardfl_ar_thresh
  ardfl_ar_thresh  aspect-ratio gate threshold
  ardfl_entropy    sharpen ONLY height-edge distributions (r10_dfl_entropy idea,
                   targeted at the axis that needs it)
  ardfl_entropy_w  weight of that entropy term

=============================================================================
RUNS (7) — anchor + dose-response + gate + entropy + best-guess combo
=============================================================================
  ardfl_anchor       AR-DFL OFF, NWD ON  -> the beat-target (= r10_nwd_fixedc)
  ardfl_h15          h=1.5 / w=1.0        (mild height emphasis)
  ardfl_h20          h=2.0 / w=1.0        (stronger)
  ardfl_h15_w075     h=1.5 / w=0.75       (also DE-emphasise width)
  ardfl_gate         h=1.5 / w=1.0, AR-gated (only tall boxes, h/w>1.5)
  ardfl_entropy      h=1.5 + height-edge entropy sharpening (w=0.05)
  ardfl_full         h=1.5 / w=0.75 + gate + height entropy (best-guess combo)

DECISION RULE (fixed before eval): candidate iff
  val mAP50-95 > ardfl_anchor + 0.5  OR  val AP75/mAP50-95 gain concentrated
  in the tighter-IoU bins (the whole point).

REQUIRES: the AR-DFL-enabled loss.py copied to ultralytics/utils/loss.py.

Usage:
  python run_ardfl_ablation.py                 # all runs
  python run_ardfl_ablation.py ardfl_h15       # single run(s)
  python run_ardfl_ablation.py --with-test     # also eval test split
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
PROJECT_DIR = "runs_ardfl"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Baseline block — plain TAL + NWD(fixed C), the best prior recipe
# (= r10_nwd_fixedc = 57.75). AR-DFL is layered on top of THIS, so the
# comparison isolates AR-DFL's contribution, not NWD's.
# Every custom key is stated explicitly so the saved params json is ground truth.
# =============================================================================
_BASE = dict(
    # assigner: plain TAL (SATAL is off — it HURTS on luggage, -2.8pt)
    use_satal=False, tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    # SWA off (rescaling proven null)
    swa_mode="scale", swa_alpha=0.0, swa_boost=1.0, swa_size_axis="width",
    # regression metric: stock CIoU (every variant lost)
    box_loss_type="ciou",
    # NWD ON, fixed C in pixels — the one prior mod that helped
    use_nwd=True, nwd_mode="blend", nwd_weight=0.5, nwd_c_px=12.0,
    # cls / clip off
    use_class_weights=False, use_vfl=False, use_loss_clip=False,
)

_ARDFL_OFF = dict(
    use_ardfl=False, ardfl_h_weight=1.0, ardfl_w_weight=1.0,
    ardfl_ar_gate=False, ardfl_ar_thresh=1.5,
    ardfl_entropy=False, ardfl_entropy_w=0.0,
)


def cfg(**overrides):
    d = dict(_BASE)
    d.update(_ARDFL_OFF)
    d.update(overrides)
    return d


RUNS = [
    {
        "name": "ardfl_anchor", "phase": "-",
        "label": "AR-DFL OFF + NWD (= r10_nwd_fixedc beat-target 57.75)",
        "params": cfg(),
    },
    {
        "name": "ardfl_h15", "phase": "A",
        "label": "height-edge DFL x1.5 (mild)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=1.0),
    },
    {
        "name": "ardfl_h20", "phase": "A",
        "label": "height-edge DFL x2.0 (stronger)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=2.0, ardfl_w_weight=1.0),
    },
    {
        "name": "ardfl_h15_w075", "phase": "A",
        "label": "height x1.5 + width x0.75 (shift capacity to height)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=0.75),
    },
    {
        "name": "ardfl_gate", "phase": "B",
        "label": "height x1.5, AR-gated (only boxes with h/w>1.5)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=1.0,
                      ardfl_ar_gate=True, ardfl_ar_thresh=1.5),
    },
    {
        "name": "ardfl_entropy", "phase": "C",
        "label": "height x1.5 + height-edge entropy sharpening (w=0.05)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=1.0,
                      ardfl_entropy=True, ardfl_entropy_w=0.05),
    },
    {
        "name": "ardfl_full", "phase": "ABC",
        "label": "height x1.5 + width x0.75 + gate + height entropy (combo)",
        "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=0.75,
                      ardfl_ar_gate=True, ardfl_ar_thresh=1.5,
                      ardfl_entropy=True, ardfl_entropy_w=0.05),
    },
]


# =============================================================================
# Epoch sync — drives any epoch-dependent schedule in the custom loss (DDP-safe)
# =============================================================================
def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    # loss.py exposes a module-level set_epoch(); call it if importable
    try:
        from ultralytics.utils.loss import set_epoch
        set_epoch(epoch, getattr(trainer, "epochs", EPOCHS))
    except Exception:
        pass
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            try:
                crit.epoch = epoch
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
            test_map50 = float(tm.box.map50)
            test_map5095 = float(tm.box.map)
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


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = set(sys.argv[1:])
    with_test = "--with-test" in args
    args.discard("--with-test")

    if args:
        unknown = args - {r["name"] for r in RUNS}
        if unknown:
            print(f"Unknown run(s): {sorted(unknown)}")
            print(f"Available: {[r['name'] for r in RUNS]}")
            sys.exit(1)
        runs = [r for r in RUNS if r["name"] in args]
    else:
        runs = RUNS

    print(f"\n{'=' * 70}")
    print(f"  AR-DFL ABLATION ({len(runs)} runs @{IMG_SIZE}, {EPOCHS} epochs, seed {SEED})")
    print(f"  Beat-target: ardfl_anchor (NWD, no AR-DFL) ~= 57.75 mAP50-95")
    print(f"  Baseline (plain TAL, no NWD)               ~= 57.43")
    print(f"{'=' * 70}")
    for r in runs:
        print(f"  {r['name']:<18s} {r['label']}")
    print(f"{'=' * 70}\n")

    all_results = []
    for r in runs:
        all_results.append(run_one(r, with_test=with_test))

    os.makedirs(PROJECT_DIR, exist_ok=True)
    summary_path = os.path.join(PROJECT_DIR, "ardfl_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  RESULTS (val split — selection basis)")
    print(f"{'=' * 70}")
    print(f"{'name':<18s} {'val mAP50':>10s} {'val mAP50-95':>13s} {'time':>6s}")
    print("-" * 52)
    anchor = next((r for r in all_results if r["name"] == "ardfl_anchor"), None)
    for r in all_results:
        delta = ""
        if anchor and r["name"] != "ardfl_anchor":
            try:
                delta = f"  ({(r['val_map5095'] - anchor['val_map5095']) * 100:+.2f})"
            except Exception:
                delta = ""
        print(f"{r['name']:<18s} {r['val_map50']:>10.4f} "
              f"{r['val_map5095']:>13.4f} {r['elapsed_h']:>5.1f}h{delta}")
    print(f"\nSummary saved: {summary_path}")
