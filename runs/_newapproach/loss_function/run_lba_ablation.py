#!/usr/bin/env python3
"""
Level-Balanced Assignment (LBA) ablation — the assigner axis.

=============================================================================
THE MEASURED PATHOLOGY
=============================================================================
diag_per_edge_dfl.py on v12s_default2, 67,685 foreground anchors / 4,600 images:

    stride   % of anchor grid   % of foreground   ratio
       8          76.2%              59.3%        0.78
      16          19.0%              39.4%        2.07
      32           4.8%               1.3%        0.28

P5 receives 887 foreground anchors in total. An entire pyramid level gets
essentially no box/DFL gradient while P4 is over-subscribed 7x relative to it.

Meanwhile LARGE objects (max side > 96px) are 29.5% of all foreground, carry the
worst top-edge residual (8.17px vs 3.43px for small), and the worst AP: 44.41 on
stock, recoverable to 59.85 with a tuned assigner. So it is NOT a capacity limit
— it is a supervision-allocation problem, and the assigner is what allocates.

=============================================================================
WHY IT HAPPENS, AND WHY THIS IS NEW
=============================================================================
TAL selects topk candidates per GT by score^alpha * iou^beta. That metric is
completely LEVEL-AGNOSTIC — nothing in it knows which pyramid level a candidate
came from. Levels with more anchors and easier early IoU win the topk, and
coarse levels starve.

OTA, SimOTA, TAL, TOOD and DW all balance assignment across GTs. None of them
balance across LEVELS. FCOS and ATSS impose HARD size ranges per level, which
cannot express partial preference and are hand-set rather than derived.

LBA multiplies the alignment metric by a soft scale-matching prior:

    octaves = log2( sqrt(w*h) / (stride * lba_ref_cells) )
    prior   = exp( -octaves^2 / (2 * lba_sigma^2) )
    align  <- align * prior^lba_strength

Soft, parameter-free, no architecture change, and lba_strength=0 reproduces
stock TAL exactly.

Measured justification for ref_cells=8: objects currently assigned to each level
sit at a geometric size of 35 / 71 / 172 px against nominal 64 / 128 / 256 px —
i.e. every level is being fed objects SMALLER than its resolution suits, worst
at P5. The prior corrects exactly that.

=============================================================================
RUNS
=============================================================================
  lba_anchor     v12s_default2 config, LBA off  -> the beat-target (57.63)
  lba_s05        strength 0.5  (mild prior)
  lba_s10        strength 1.0  (full prior)
  lba_s10_sig15  strength 1.0, sigma 1.5 octaves (broader, gentler)
  lba_s20        strength 2.0  (sharp prior — risks starving P3)

BASELINE NOTE: this ablation runs on the ASSIGNER-MATCHED baseline
(use_satal=True, topk=12, alpha=0.6, beta=5.0) because that is your real working
config at 57.63. The earlier PEU ablation used plain stock TAL and sat 1.68
points lower — a baseline that was never competitive.

Usage:
  python run_lba_ablation.py                       # all
  python run_lba_ablation.py lba_anchor lba_s10    # the decisive two
  python run_lba_ablation.py --seeds
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
PROJECT_DIR = "runs_lba"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# measured foreground share per level on the baseline — the thing LBA must move
MEASURED_SHARE = {8: 0.593, 16: 0.394, 32: 0.013}
GRID_SHARE = {8: 0.762, 16: 0.190, 32: 0.048}

# ---- the assigner-matched baseline (v12s_default2 = 57.63) ------------------
_BASE = dict(
    use_satal=True, tal_topk=12, tal_alpha=0.6, tal_beta=5.0,
    satal_alpha_small=1.2, satal_beta_small=4.5, satal_topk_factor=1.3,
    swa_mode="scale", swa_alpha=0.0, swa_boost=1.0,
    box_loss_type="ciou",
    use_nwd=False, use_class_weights=False, use_vfl=False, use_loss_clip=False,
    use_ardfl=False, use_adfl=False, use_peu=False, use_edgew=False,
)
# ref_cells=4.5 is MEASURED, not guessed: objects currently assigned to each
# level have geometric size 4.43 / 4.43 / 5.37 cells. The first attempt used 8.0
# (derived from object HEIGHT in cells, which is wrong because the prior keys on
# sqrt(w*h) and these boxes are 2.79x taller than wide). That pushed everything
# down the pyramid — P5 peaked at 6.9% then decayed to 2.7% while P3 rose to 68%.
_LBA_OFF = dict(use_lba=False, lba_strength=1.0, lba_ref_cells=4.5,
                lba_sigma=1.0, lba_log=True)


def cfg(**over):
    d = dict(_BASE); d.update(_LBA_OFF); d.update(over); return d


RUNS = [
    {"name": "lba_anchor", "rank": 0, "kind": "baseline",
     "label": "v12s_default2 config, LBA off — beat-target 57.63",
     "params": cfg()},
    {"name": "lba_s10", "rank": 1, "kind": "LBA",
     "label": "strength 1.0, sigma 1.0 — the full prior (default)",
     "params": cfg(use_lba=True, lba_strength=1.0, lba_sigma=1.0)},
    {"name": "lba_s05", "rank": 2, "kind": "LBA",
     "label": "strength 0.5 — mild prior, lowest risk of starving P3",
     "params": cfg(use_lba=True, lba_strength=0.5, lba_sigma=1.0)},
    {"name": "lba_s10_sig15", "rank": 3, "kind": "LBA",
     "label": "strength 1.0, sigma 1.5 octaves — broader, gentler falloff",
     "params": cfg(use_lba=True, lba_strength=1.0, lba_sigma=1.5)},
    {"name": "lba_s20", "rank": 4, "kind": "LBA",
     "label": "strength 2.0 — sharp prior; may over-commit and starve P3",
     "params": cfg(use_lba=True, lba_strength=2.0, lba_sigma=1.0)},
]
SEED_RUNS = ["lba_anchor", "lba_s10"]
SEEDS_LIST = [0, 42, 123]


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
    """Per-level foreground share — this IS the hypothesis test.

    If LBA works mechanically, P5's share must rise from the measured 1.3%
    toward its 4.8% grid share, and P4's 39.4% must fall toward 19.0%.
    If the shares do not move, the prior is not reaching the assigner.
    """
    try:
        from ultralytics.utils.loss import lba_report
        r = lba_report(reset=True)
        if not r:
            return
        parts = []
        for s in sorted(r):
            base = MEASURED_SHARE.get(s)
            d = f" ({(r[s]['share']-base)*100:+.1f})" if base is not None else ""
            parts.append(f"s{s}={r[s]['share']*100:.1f}%{d}")
        print("  [LBA] fg share per level: " + "  ".join(parts)
              + f"   | grid share {[f's{k}={v*100:.1f}%' for k, v in GRID_SHARE.items()]}")
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
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    try:
        with open(os.path.join(save_dir, "lba_params.json"), "w") as f:
            json.dump({"name": name, "kind": rc["kind"], "label": rc["label"],
                       "params": rc["params"], "seed": seed, "epochs": EPOCHS,
                       "measured_fg_share": MEASURED_SHARE, "grid_share": GRID_SHARE,
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
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--seeds", action="store_true")
    args = ap.parse_args()

    print(f"\n{'=' * 78}")
    print(f"  LEVEL-BALANCED ASSIGNMENT ABLATION  @{IMG_SIZE}px  {EPOCHS} epochs")
    print(f"  baseline = v12s_default2 (satal=True topk=12 a=0.6 b=5.0) = 57.63")
    print(f"  measured fg share  s8={MEASURED_SHARE[8]*100:.1f}%  "
          f"s16={MEASURED_SHARE[16]*100:.1f}%  s32={MEASURED_SHARE[32]*100:.1f}%")
    print(f"  grid share         s8={GRID_SHARE[8]*100:.1f}%  "
          f"s16={GRID_SHARE[16]*100:.1f}%  s32={GRID_SHARE[32]*100:.1f}%")
    print(f"  loss file: {_loss_fingerprint()}")
    print(f"{'=' * 78}")

    if args.seeds:
        todo = [(r, s) for r in RUNS if r["name"] in SEED_RUNS for s in SEEDS_LIST]
    else:
        unknown = set(args.runs) - {r["name"] for r in RUNS}
        if unknown:
            sys.exit(f"unknown: {sorted(unknown)}\navailable: {[r['name'] for r in RUNS]}")
        sel = RUNS if not args.runs else [r for r in RUNS if r["name"] in set(args.runs)]
        todo = [(r, SEED) for r in sel]

    # ---- refuse to run LBA variants without a matched anchor ---------------
    # This has now been launched three times without lba_anchor. Every LBA run
    # is uninterpretable without a baseline trained under IDENTICAL conditions:
    # the only other reference is v12s_default2, trained in a different session,
    # and this project has already produced 1.5-point gaps between supposedly
    # identical baselines. Six GPU-hours of unattributable numbers is worse than
    # an error message.
    wants_lba = any(r["kind"] == "LBA" for r, _ in todo)
    has_anchor_queued = any(r["name"] == "lba_anchor" for r, _ in todo)
    anchor_done = os.path.isdir(os.path.join(PROJECT_DIR, "lba_anchor"))
    if wants_lba and not has_anchor_queued and not anchor_done:
        sys.exit(
            "\nREFUSING TO RUN.\n"
            "  LBA variants were requested but 'lba_anchor' is neither queued nor\n"
            f"  already present in {PROJECT_DIR}/.\n\n"
            "  Without a baseline trained under identical conditions the results\n"
            "  cannot be attributed to LBA. Run one of:\n\n"
            "      python run_lba_ablation.py lba_anchor lba_s10\n"
            "      python run_lba_ablation.py                  # full list, anchor first\n"
        )

    for r, s in todo:
        print(f"  {r['rank']:>2}  {r['name']:<16s} seed={s}  {r['label']}")
    print(f"{'=' * 78}\n")

    res = [run_one(r, seed=s, with_test=args.with_test) for r, s in todo]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "lba_summary_seeds.json" if args.seeds else "lba_summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    a = next((r["val_map5095"] for r in res if r["name"] == "lba_anchor"), None)
    print(f"\n{'=' * 78}\n  RESULTS (val)\n{'=' * 78}")
    print(f"{'run':<18s}{'kind':<10s}{'mAP50':>8s}{'mAP50-95':>11s}{'delta':>9s}{'h':>6s}")
    print("-" * 78)
    for r in sorted(res, key=lambda x: -(x["val_map5095"] if x["val_map5095"] == x["val_map5095"] else -9)):
        d = f"{(r['val_map5095']-a)*100:+.2f}" if a and r["name"] != "lba_anchor" else "—"
        print(f"{r['name']:<18s}{r['kind']:<10s}{r['val_map50']*100:>8.2f}"
              f"{r['val_map5095']*100:>11.2f}{d:>9s}{r['hours']:>5.1f}")
    print("\nCHECK THE [LBA] TELEMETRY FIRST: if P5's foreground share did not rise")
    print("from 1.3%, the prior never reached the assigner and the mAP is meaningless.")
    print(f"\nsaved -> {out}")
