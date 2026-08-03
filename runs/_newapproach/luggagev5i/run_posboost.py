#!/usr/bin/env python3
"""
loss_custom_v3 — ROUND 2: positive-confidence boost + its control.

=============================================================================
WHAT THIS TESTS
=============================================================================
Four runs, one variable each, all against the SAME baseline:

  v3_anchor    no keys set -> loss_custom_v3.py IS loss_original_stock.py.
               Must reproduce v12s_default2 (57.63 mAP50-95, test split).
               If it does not, STOP — nothing below is interpretable.

  v3_lba       use_lba=True, strength 0.8, axis 'max', ref_cells 6.0.
               Prior round measured large +3.99 / medium -1.42 / net -0.07,
               but through the box+dfl normalisation bug. The question is
               whether that split reproduces on a clean baseline. strength 0.8
               (not 1.0) because P5 cannot absorb what gets pushed up.

  v3_ardfl     use_ardfl=True, w=1.458 h=0.542 — the CORRECTED direction.
               An e-px error costs e/w on a width edge and e/h on a height
               edge, ratio h/w = 2.69 here, so width edges are ~2.7x more
               IoU-sensitive. Every previous run used h>w, i.e. backwards,
               and measured -0.46. Weights mean-normalised to sum 2 at w/h=2.69.

  v3_peu       use_peu=True, beta 1.0, lambda 0, detach=True, norm_by_mu=True.
               detach=True/lambda=0 is one of the two non-degenerate configs
               (detach=True with lambda>0 collapsed at -4.33 and -7.31).
               beta=1.0 reproduces the measured 2.0x top:bottom residual
               asymmetry. norm_by_mu removes the size confound that made raw
               variance attenuate large objects rather than uncertain edges.

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
BATCH = 58            # !! VERIFY against v3_anchor2's args.yaml. Your runner
                      # versions disagreed (58 here, 54 in the telemetry-era
                      # one) — the gated run MUST use whatever the anchor and
                      # v3_lba2 actually trained with, or the comparison
                      # carries a batch confound.
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
    # Round 1 (closed): anchor 57.63 | lba2 57.57 (large +1.37, small -0.37) |
    #   lba_gated 57.33 | ardfl_w 57.41 | ardfl_h 57.02 | dflgain17 56.91 |
    #   peu 56.53. DFL channel triple-eliminated; LBA's gain and cost proved
    #   coupled (gating removed both). Nothing beat baseline.
    #
    # Round 2 — the CLASSIFICATION channel, untouched so far. Evidence:
    #   confusion matrix (anchor, test): bag row -> backpack 0.03, trolley 0.02,
    #   background 0.25;  bag column -> correct 0.68, MISSED 0.24.
    #   => bag is NOT confused with other classes, it is UNDER-SCORED.
    #   Same signature globally: AR50_small 0.962 vs R50_small 0.727.
    #   Cross-class penalties (SATAL Section L) would fire on 3% of the problem
    #   and were dropped on this evidence.

    # ---- MECHANISM: positive-confidence boost ----------------------------
    # Scales ONLY the positive bce term at fg anchors, per class x size.
    # Weights from the per-cell recall gaps, NOT from class counts:
    # trolley 0.87 -> 1.0, backpack 0.80 -> 1.0, bag 0.68 -> 1.5; small x1.5.
    # small_px = 60 MODEL px == the dataset analysis' 48px small bin (labels
    # are 512px, training at 640 -> 1.25x). Not zero-sum: raising a logit
    # takes nothing from other anchors (unlike LBA's anchor reallocation).
    {"name": "v3_posboost", "kind": "cls",
     "label": "POS-BOOST bag 1.5 / small 1.5 (<60 model-px), positive term only",
     "params": dict(use_pos_boost=True, pos_boost_bag=1.5, pos_boost_backpack=1.0,
                    pos_boost_trolley=1.0, pos_boost_small=1.5,
                    pos_boost_small_px=60.0, pos_boost_clip=3.0)},

    # ---- CONTROL: plain inverse-frequency class weighting ----------------
    # Rare classes ARE often the under-scored ones, so a posboost gain must be
    # shown to exceed what ordinary imbalance weighting buys. Scales BOTH bce
    # terms by sqrt-damped inverse frequency -> ~[1.06, 1.17, 0.77].
    # Same role dflgain17 played for PEU. If this matches v3_posboost, the
    # "aimed by the confusion matrix" claim is dead and the finding is just
    # "weight the rare class".
    {"name": "v3_clsweight", "kind": "control",
     "label": "CONTROL inverse-frequency class weighting, use_freq_weight (both bce terms)",
     "params": dict(use_freq_weight=True)},
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
    """Per-epoch telemetry: level occupancy, gate pass-rate, PEU weights.

    Printed AND appended to <save_dir>/telemetry.csv — otherwise the most
    informative signal in the run exists only in stdout and is lost.
    """
    row = {"epoch": int(getattr(trainer, "epoch", -1))}

    try:
        from ultralytics.utils.loss import lba_report
        r = lba_report(reset=True)
        if r:
            print("  [level] " + "  ".join(
                f"s{k}={v['share'] * 100:.1f}%" for k, v in sorted(r.items())))
            for k, v in sorted(r.items()):
                row[f"fg_share_s{k}"] = round(v["share"], 6)
                row[f"fg_count_s{k}"] = v["fg"]
    except Exception:
        pass

    try:
        from ultralytics.utils.loss import lba_gate_report
        g = lba_gate_report(reset=True)
        if g:
            print(f"  [gate]  pass_rate={g['pass_rate'] * 100:.1f}%  ({g['gts']} GTs)")
            row["gate_pass_rate"] = round(g["pass_rate"], 6)
            row["gate_gts"] = g["gts"]
    except Exception:
        pass  # older loss file without lba_gate_report -> column simply absent

    try:
        from ultralytics.utils.loss import posboost_report
        pb = posboost_report(reset=True)
        if pb:
            names = ("backpack", "bag", "trolley")
            print("  [posb]  fg_score " + "  ".join(
                f"{n}={s:.3f}" for n, s in zip(names, pb["fg_score"]))
                + f"   mean_boost={pb['mean_boost']:.2f}")
            for n, s, c in zip(names, pb["fg_score"], pb["fg_count"]):
                row[f"fg_score_{n}"] = round(s, 6)
                row[f"fg_count_{n}"] = int(c)
            row["mean_boost"] = round(pb["mean_boost"], 4)
    except Exception:
        pass

    try:
        from ultralytics.utils.loss import peu_report
        r = peu_report(reset=True)
        if r:
            print("  [PEU] w  " + "  ".join(f"{k}={v:.3f}" for k, v in r["weight"].items()))
            print("  [PEU] var" + "  ".join(f" {k}={v:.3f}" for k, v in r["var"].items()))
            for e in ("left", "top", "right", "bottom"):
                row[f"peu_w_{e}"] = round(r["weight"][e], 6)
                row[f"peu_var_{e}"] = round(r["var"][e], 6)
    except Exception:
        pass

    if len(row) > 1:
        try:
            import csv
            p = os.path.join(str(trainer.save_dir), "telemetry.csv")
            new = not os.path.exists(p)
            with open(p, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"  [warn] telemetry not saved: {e}")


def run_one(rc):
    name, p = rc["name"], rc["params"]
    print(f"\n{'=' * 76}\n  RUN {name}   [{rc['kind']}]\n  {rc['label']}\n{'=' * 76}\n")

    t0 = time.time()

    # Fresh PEU warmup state per run: _EPOCH is module-global in the loss file
    # and every run's on_train_epoch_start callback updates it, so without this
    # reset a later run (v3_peu is last) constructs its criterion with
    # epoch=69/set=True left over from the previous run — and the fail-closed
    # warmup gate would treat any loss forward before this run's first epoch
    # callback as fully warmed.
    try:
        import ultralytics.utils.loss as _L
        _L._EPOCH.update({"epoch": 0, "total": 0, "set": False})
    except Exception:
        pass

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
            # Per-CLASS AP50-95 (tm.box.maps is per-class, NOT per-size-bucket).
            # Ultralytics' .val() does not expose small/medium/large mAP; use
            # CocoEvalAllFolders_luggage.py on best.pt for the size buckets —
            # that is where every real effect so far lived (diluted in the mean).
            try:
                if hasattr(tm.box, "maps") and tm.box.maps is not None:
                    out["test_ap_per_class"] = [float(v) for v in tm.box.maps]
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

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "v3_summary.json")

    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "kind": r["kind"], "hours": float("nan"),
                        "val_map50": float("nan"), "val_map5095": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        # incremental dump — survives a crash mid-study
        with open(out, "w") as f:
            json.dump(res, f, indent=2)

    summarise(res, out)
