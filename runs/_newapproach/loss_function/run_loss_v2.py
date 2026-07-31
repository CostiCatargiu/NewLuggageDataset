#!/usr/bin/env python3
"""
Loss ablation v2 — on the CORRECT baseline this time.

=============================================================================
THE BASELINE (this is the thing every previous ablation got wrong)
=============================================================================
Your real default is v12s_default2 = 82.77 / 57.63 on test_full_dataset:

    stock TaskAlignedAssigner,  tal_topk=12,  tal_alpha=0.6,  tal_beta=5.0

NOT SATAL. The saved config for that run lists use_satal: True, but that key was
INERT in the build that produced it — the training dynamics prove it: epoch-1
box_loss is 1.335, matching stock TAL (peu_anchor = 1.337). A genuinely active
SATAL run starts at 1.66-1.69. This is the same failure mode documented in
LOSS_CODE_AUDIT.md: the saved configs are a union of every generation's key
namespace, and most keys are dead for whichever loss.py was installed.

Consequences of getting this wrong, both of which happened:
  * the PEU ablation used stock TAL with topk=10/a=0.5/b=6.0  -> 55.95,
    i.e. 1.68 BELOW the real default, so every PEU delta was measured against
    a baseline that was never competitive;
  * the first LBA ablation set use_satal=True and SATAL *is* importable on this
    install, so it genuinely activated -> that run differed from the default by
    TWO things (SATAL + LBA) and was unattributable.

Incidentally this also gives a clean, purely loss-side result that was hiding in
the archive all along:

    stock TAL  topk=10  a=0.5  b=6.0   ->  55.95
    stock TAL  topk=12  a=0.6  b=5.0   ->  57.63     (+1.68)

TAL hyperparameter retuning alone is worth +1.68. No SATAL, no new mechanism.

=============================================================================
RUNS
=============================================================================
  base_anchor    the real default, nothing else on  -> beat-target 57.63
  ardfl_h15      AR-DFL, height edges x1.5 (the loss doc's Priority 1, never run)
  ardfl_h15_w075 AR-DFL, height x1.5 + width x0.75
  lba_s10        Level-Balanced Assignment, strength 1.0
  lba_s05        Level-Balanced Assignment, strength 0.5
  gain_box10     box gain 7.5 -> 10.0   (never varied in 78 runs)
  gain_dfl30     dfl gain 1.5 -> 3.0    (never varied in 78 runs)

Every run differs from base_anchor by exactly ONE mechanism.

Usage:
  python run_loss_v2.py                          # all, anchor first
  python run_loss_v2.py base_anchor ardfl_h15    # subset
  python run_loss_v2.py --with-test              # also eval test (needed for 57.63)
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_lossv2"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

BEAT_TARGET = 57.63          # v12s_default2 on test_full_dataset

# ---- the REAL default: stock TAL, retuned. use_satal MUST stay False. -------
_BASE = dict(
    use_satal=False, tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    swa_mode="scale", swa_alpha=0.0, swa_boost=1.0,
    box_loss_type="ciou",
    use_nwd=False, use_class_weights=False, use_vfl=False, use_loss_clip=False,
    use_ardfl=False, use_adfl=False, use_peu=False, use_edgew=False, use_lba=False,
    lba_log=True,      # log level occupancy even when LBA is off -> real baseline share
)


def cfg(**over):
    d = dict(_BASE)
    d.update(over)
    return d


RUNS = [
    {"name": "base_anchor", "rank": 0, "kind": "baseline",
     "label": f"real default: stock TAL topk=12 a=0.6 b=5.0 — beat-target {BEAT_TARGET}",
     "params": cfg()},

    {"name": "ardfl_h15", "rank": 1, "kind": "AR-DFL",
     "label": "height-edge DFL x1.5 — the loss doc's Priority 1, never actually run",
     "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=1.0)},

    {"name": "lba_s10", "rank": 2, "kind": "LBA",
     "label": "level-balanced assignment, strength 1.0 sigma 1.0",
     "params": cfg(use_lba=True, lba_strength=1.0, lba_ref_cells=4.5, lba_sigma=1.0)},

    {"name": "gain_box10", "rank": 3, "kind": "gain",
     "label": "box gain 7.5 -> 10.0 (never varied in 78 runs)",
     "params": cfg(), "train_over": dict(box=10.0)},

    {"name": "gain_dfl30", "rank": 4, "kind": "gain",
     "label": "dfl gain 1.5 -> 3.0 (never varied in 78 runs)",
     "params": cfg(), "train_over": dict(dfl=3.0)},

    {"name": "ardfl_h15_w075", "rank": 5, "kind": "AR-DFL",
     "label": "height x1.5 + width x0.75",
     "params": cfg(use_ardfl=True, ardfl_h_weight=1.5, ardfl_w_weight=0.75)},

    {"name": "lba_s05", "rank": 6, "kind": "LBA",
     "label": "level-balanced assignment, strength 0.5 (milder)",
     "params": cfg(use_lba=True, lba_strength=0.5, lba_ref_cells=4.5, lba_sigma=1.0)},
]


def _loss_fingerprint():
    try:
        import ultralytics.utils.loss as L
        return {"path": L.__file__,
                "md5": hashlib.md5(open(L.__file__, "rb").read()).hexdigest()[:12],
                "has_lba": hasattr(L, "lba_report")}
    except Exception as e:
        return {"error": str(e)}


def on_train_epoch_start(trainer):
    try:
        from ultralytics.utils.loss import set_epoch
        set_epoch(trainer.epoch, getattr(trainer, "epochs", EPOCHS))
    except Exception:
        pass
    try:
        de_parallel(trainer.model).current_epoch = trainer.epoch
    except Exception:
        pass


def on_train_epoch_end(trainer):
    try:
        from ultralytics.utils.loss import lba_report
        r = lba_report(reset=True)
        if r:
            print("  [level] fg share: " + "  ".join(f"s{k}={v['share']*100:.1f}%"
                                                     for k, v in sorted(r.items())))
    except Exception:
        pass


def run_one(rc, seed=SEED, with_test=False):
    name = rc["name"] if seed == SEED else f"{rc['name']}_s{seed}"
    print(f"\n{'=' * 78}\n  RUN {name}   [{rc['kind']}]  seed={seed}\n  {rc['label']}\n{'=' * 78}\n")
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=100, close_mosaic=10, seed=seed, deterministic=True,
              exist_ok=False)
    kw.update(copy.deepcopy(rc["params"]))
    kw.update(copy.deepcopy(rc.get("train_over", {})))     # box/cls/dfl gains
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump({"name": name, "kind": rc["kind"], "label": rc["label"],
                       "params": rc["params"], "train_over": rc.get("train_over", {}),
                       "seed": seed, "epochs": EPOCHS, "beat_target": BEAT_TARGET,
                       "loss_file": _loss_fingerprint()}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json: {e}")

    def _m(rd, *ks):
        for k in ks:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "kind": rc["kind"], "seed": seed, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    if with_test:
        try:
            tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
            out["test_map50"], out["test_map5095"] = float(tm.box.map50), float(tm.box.map)
        except Exception as e:
            print(f"  [warn] test eval: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--with-test", action="store_true",
                    help="also evaluate the test split — REQUIRED to compare against 57.63")
    args = ap.parse_args()

    print(f"\n{'=' * 78}")
    print(f"  LOSS ABLATION v2  @{IMG_SIZE}px  {EPOCHS} epochs")
    print(f"  baseline: stock TAL, topk=12, alpha=0.6, beta=5.0  (use_satal=False)")
    print(f"  beat-target: {BEAT_TARGET} mAP50-95 on test_full_dataset")
    print(f"  loss file: {_loss_fingerprint()}")
    print(f"{'=' * 78}")

    unknown = set(args.runs) - {r["name"] for r in RUNS}
    if unknown:
        sys.exit(f"unknown: {sorted(unknown)}\navailable: {[r['name'] for r in RUNS]}")
    sel = RUNS if not args.runs else [r for r in RUNS if r["name"] in set(args.runs)]
    todo = [(r, SEED) for r in sel]

    if not args.with_test:
        print("\n  NOTE: --with-test not set. Val numbers are NOT comparable to the")
        print("        57.63 target, which is a test_full_dataset figure.\n")

    for r, s in todo:
        print(f"  {r['rank']:>2}  {r['name']:<16s} {r['kind']:<9s} {r['label']}")
    print(f"{'=' * 78}\n")

    res = [run_one(r, seed=s, with_test=args.with_test) for r, s in todo]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    a = next((r["val_map5095"] for r in res if r["name"] == "base_anchor"), None)
    print(f"\n{'=' * 78}\n  RESULTS\n{'=' * 78}")
    print(f"{'run':<18s}{'kind':<10s}{'val50-95':>10s}{'vs anchor':>11s}"
          f"{'test50-95':>11s}{'vs 57.63':>10s}{'h':>6s}")
    print("-" * 78)
    for r in sorted(res, key=lambda x: -(x["val_map5095"] if x["val_map5095"] == x["val_map5095"] else -9)):
        d = f"{(r['val_map5095']-a)*100:+.2f}" if a and r["name"] != "base_anchor" else "—"
        t = r["test_map5095"] * 100
        dt = f"{t-BEAT_TARGET:+.2f}" if t == t else "—"
        ts = f"{t:.2f}" if t == t else "—"
        print(f"{r['name']:<18s}{r['kind']:<10s}{r['val_map5095']*100:>10.2f}{d:>11s}"
              f"{ts:>11s}{dt:>10s}{r['hours']:>5.1f}")
    print(f"\nsaved -> {out}")
