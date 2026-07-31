#!/usr/bin/env python3
"""
loss_custom_v3 ablation — neutral / LBA / AR-DFL / PEU.

=============================================================================
WHAT THIS TESTS
=============================================================================
Four runs, one variable each, all against the SAME baseline:

  v3_anchor    no keys set -> loss_custom_v3.py IS loss_original_stock.py.
               Must reproduce v12s_default2 (57.63 mAP50-95, test split).
               If it does not, STOP — nothing below is interpretable.

  v3_lba       use_lba=True, strength 1.0, size axis 'max'.
               Prior round measured large +3.99 / medium -1.42 / net -0.07,
               but through the box+dfl normalisation bug. The question is
               whether that split reproduces on a clean baseline.

  v3_ardfl     use_ardfl=True, w=1.5 h=0.75 — the CORRECTED direction.
               An e-px error costs e/w on a width edge and e/h on a height
               edge, ratio h/w = 2.69 here, so width edges are ~2.7x more
               IoU-sensitive. Every previous run used h>w, i.e. backwards,
               and measured -0.46. Nobody has run it this way.

  v3_peu       use_peu=True, beta 0.5, lambda 0, detach=True, norm_by_mu=True.
               detach=True/lambda=0 is one of the two non-degenerate configs
               (detach=True with lambda>0 collapsed at -4.33 and -7.31).
               norm_by_mu removes the size confound that made raw variance
               attenuate large objects rather than uncertain edges.

=============================================================================
READ FIRST
=============================================================================
* RECIPE MUST MATCH v12s_default2. The 57.63 comparison is only valid if
  epochs / imgsz / batch / model / close_mosaic / seed are identical. Check
  the values below against that run's args.yaml before starting.
* The baseline number is on the TEST split, so WITH_TEST is on by default.
* Every run writes the md5 of the INSTALLED loss file into its params json.
  The previous ablations did not, and their configs became uninterpretable.

REQUIRES loss_custom_v3.py copied to ultralytics/utils/loss.py, and the
use_ardfl / use_peu / use_lba keys whitelisted in cfg/default.yaml.

Usage:
    python run_custom_v3_ablation.py
"""

import copy
import gc
import hashlib
import json
import os
import time

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION  — must match v12s_default2
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_custom_v3"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100

WITH_TEST = True          # baseline 57.63 is a TEST-split number
BASELINE_TEST_MAP5095 = 0.5763   # v12s_default2, for the sanity check

ONLY = []                 # e.g. ["v3_anchor", "v3_lba"] — empty = all
# =============================================================================

RUNS = [
    {"name": "v3_anchor", "kind": "baseline",
     "label": "no keys set — must reproduce v12s_default2 (57.63)",
     "params": {}},

    # ref_cells=6.0 on the MAX axis -> nominal 48 / 96 / 192 px at 640. Against
    # this dataset that puts a 45px object on P3 (prior 0.996 vs P4 0.55), the
    # mean 90px object on P4, and 150px on P4/P5. The FCOS-style ref_cells=4
    # would drop the mean object's P3 prior to 0.33 and push nearly everything
    # onto P4 — already over-selected at 2.12x per the footprint diagnostic.
    # strength 0.8 because P5 cannot absorb what gets pushed up (median pool 2,
    # 26.3% of GTs have zero reachable anchors there).
    {"name": "v3_lba", "kind": "assigner",
     "label": "LBA strength 0.8, axis 'max', ref_cells 6.0 (nominal 48/96/192px)",
     "params": dict(use_lba=True, lba_strength=0.8, lba_ref_cells=6.0,
                    lba_sigma=1.0, lba_size_axis="max", lba_size_gate_px=0.0)},

    # Ratio derived, not tuned: 1px on a width edge buys 1/41 = 2.44% IoU, on a
    # height edge 1/90 = 1.11%. Mean-normalised weights with w/h = 2.69 are
    # w+h=2, w/h=2.69 -> 1.458 / 0.542.
    {"name": "v3_ardfl", "kind": "dfl",
     "label": "AR-DFL w=1.46 h=0.54 — width edges up, ratio = h/w = 2.69",
     "params": dict(use_ardfl=True, ardfl_w_weight=1.458, ardfl_h_weight=0.542)},

    # beta=1.0: with centred log-variance a 2x variance ratio gives a weight
    # ratio exp(-beta*ln2), so beta=1 reproduces exactly the 2.0x top:bottom
    # relative-residual asymmetry measured (6.67% vs 3.40%). beta=0.5 would give
    # only 1.4x. warmup 10 because bin distributions are near-uniform early.
    {"name": "v3_peu", "kind": "dfl",
     "label": "PEU beta=1.0 lambda=0 detach=True norm_by_mu=True (matches 2x measured asymmetry)",
     "params": dict(use_peu=True, peu_beta=1.0, peu_lambda=0.0, peu_detach=True,
                    peu_norm_by_mu=True, peu_warmup_epochs=10,
                    peu_min_var=0.05, peu_w_clip=3.0)},
]


# =============================================================================
def _loss_fingerprint():
    """Record WHICH loss file actually ran — the habit the earlier rounds lacked."""
    try:
        import ultralytics.utils.loss as L
        p = L.__file__
        return {"path": p,
                "md5": hashlib.md5(open(p, "rb").read()).hexdigest()[:12],
                "has_custom_v3": hasattr(L, "CustomLossCfg")}
    except Exception as e:
        return {"error": str(e)}


def _on_epoch_start(trainer):
    """Feed the epoch to PEU's warmup gate."""
    try:
        from ultralytics.utils.loss import set_epoch
        set_epoch(trainer.epoch, getattr(trainer, "epochs", EPOCHS))
    except Exception:
        pass


def _on_epoch_end(trainer):
    """Per-epoch telemetry: level occupancy (always) and PEU weights (if on)."""
    try:
        from ultralytics.utils.loss import lba_report
        r = lba_report(reset=True)
        if r:
            print("  [level] " + "  ".join(
                f"s{k}={v['share'] * 100:.1f}%" for k, v in sorted(r.items())))
    except Exception:
        pass
    try:
        from ultralytics.utils.loss import peu_report
        r = peu_report(reset=True)
        if r:
            print("  [PEU] w  " + "  ".join(f"{k}={v:.3f}" for k, v in r["weight"].items()))
            print("  [PEU] var" + "  ".join(f" {k}={v:.3f}" for k, v in r["var"].items()))
    except Exception:
        pass


def run_one(rc):
    name, p = rc["name"], rc["params"]
    print(f"\n{'=' * 76}\n  RUN {name}   [{rc['kind']}]\n  {rc['label']}\n{'=' * 76}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_train_epoch_end", _on_epoch_end)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=False)
    kw.update(copy.deepcopy(p))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    meta = {"name": name, "kind": rc["kind"], "label": rc["label"], "params": p,
            "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH, "seed": SEED,
            "model": MODEL_WEIGHTS, "close_mosaic": CLOSE_MOSAIC,
            "loss_file": _loss_fingerprint()}
    try:
        with open(os.path.join(save_dir, "v3_params.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "kind": rc["kind"], "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    if WITH_TEST:
        try:
            tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
            out["test_map50"] = float(tm.box.map50)
            out["test_map5095"] = float(tm.box.map)
            # size buckets — every real effect so far lived in one bucket and
            # got diluted in the headline average
            try:
                out["test_map5095_small"] = float(tm.box.maps[0]) if hasattr(tm.box, "maps") else float("nan")
            except Exception:
                pass
        except Exception as e:
            print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    anchor = next((r for r in res if r["name"] == "v3_anchor"), None)
    a = anchor["test_map5095"] if anchor and WITH_TEST else (
        anchor["val_map5095"] if anchor else None)
    key = "test_map5095" if WITH_TEST else "val_map5095"
    key50 = "test_map50" if WITH_TEST else "val_map50"

    print(f"\n{'=' * 76}\n  RESULTS ({'test' if WITH_TEST else 'val'} split)\n{'=' * 76}")
    print(f"{'run':<14}{'kind':<11}{'mAP50':>9}{'mAP50-95':>11}{'delta':>9}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x[key] if x[key] == x[key] else -9)):
        d = f"{(r[key] - a) * 100:+.2f}" if a and r["name"] != "v3_anchor" else "—"
        print(f"{r['name']:<14}{r['kind']:<11}{r[key50] * 100:>9.2f}"
              f"{r[key] * 100:>11.2f}{d:>9}{r['hours']:>5.1f}")

    if anchor and WITH_TEST:
        gap = (anchor["test_map5095"] - BASELINE_TEST_MAP5095) * 100
        print(f"\n  SANITY: v3_anchor {anchor['test_map5095'] * 100:.2f} vs "
              f"v12s_default2 {BASELINE_TEST_MAP5095 * 100:.2f}  ->  {gap:+.2f}")
        if abs(gap) > 0.5:
            print("  !! NEUTRAL DOES NOT REPRODUCE THE BASELINE. Fix that before")
            print("     reading any mechanism result — the previous rebuild lost")
            print("     1.68 pt exactly this way.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    todo = [r for r in RUNS if not ONLY or r["name"] in set(ONLY)]

    print(f"\n{'=' * 76}")
    print(f"  loss_custom_v3 ABLATION   @{IMG_SIZE}px, {EPOCHS} epochs, batch {BATCH}")
    print(f"  loss file: {_loss_fingerprint()}")
    print(f"  baseline to beat: {BASELINE_TEST_MAP5095 * 100:.2f} (v12s_default2, test)")
    print(f"{'=' * 76}")
    for r in todo:
        print(f"  {r['name']:<14} {r['label']}")
    print(f"{'=' * 76}\n")

    res = [run_one(r) for r in todo]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "v3_summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    summarise(res, out)
