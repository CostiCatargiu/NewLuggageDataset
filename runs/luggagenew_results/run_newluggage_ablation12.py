#!/usr/bin/env python3
"""
Round 12 — IMPROVE THE BEST CONFIG (NWD winner) + measure it honestly.

WHY THIS ROUND
  Across R2-R11 (~55 runs) the ONLY mechanism that produced a repeatable gain was
  R10's nwd_fixedc: adaptive+annealed NWD regression -> +1.36 mAP50-95 and the best
  small-object score (small AP50-95 +1.43). Everything else reweighted the same
  CIoU+DFL signal and landed inside the noise band.

  But that +1.36 is single-seed, and the "all-levers-off" anchors themselves span
  82.54-82.80 mAP50 / 56.25-57.43 mAP50-95 across files. That anchor spread (~0.3 /
  ~1.2) IS the noise floor -- it is the same size as the best "win". So Round 12 has
  two jobs:
    (1) CONFIRM the NWD winner is real (>2*std over a 3-seed anchor), not a lucky draw.
    (2) IMPROVE it by (a) pushing the one lever that moved (nwd_ratio), and
        (b) composing it with the ORTHOGONAL winners -- geometry (EIoU) and
        assignment (NWD-aware TAL). These touch different stages than NWD
        regression, so unlike the r7_stack collapse they should not conflict.

  All mechanisms live in loss_v1updated.py (the file the R10 winner ran on), so this
  round needs NO new loss code and NO cross-file porting.

THE WINNER (baseline for this round, = r10_nwd_fixedc):
    nwd_ratio=0.3, nwd_c=64.0, nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
    nwd_c_adaptive=1, nwd_c_k=0.5, small_obj_px=48

RUNS (ordered by expected value; anchor + winner run at 3 seeds first):
  r12_anchor       all inert -- the 3-seed noise floor (THE reference)
  r12_nwd_win      = r10_nwd_fixedc -- CONFIRM the win at 3 seeds before building on it
  r12_nwd_r045     winner but nwd_ratio 0.30 -> 0.45 -- push the lever that moved
  r12_nwd_eiou     winner + box_metric=eiou -- NWD (metric) + EIoU (aspect):
                   two orthogonal localization fixes for tall-narrow small boxes
  r12_nwd_talnwd   winner + NWD-aware TAL assignment -- compose the two winners'
                   territory (regression-side NWD + assignment-side NWD; the
                   principled replacement for r8_artal, which lives in a different file)
  r12_full         winner + eiou + tal_nwd -- ONLY meaningful if the two above each
                   clear the seed gate; otherwise expect an r7-style stacking loss

SEED PROTOCOL (the point of this round)
  Every config carries a `seeds` list. Default: anchor + nwd_win at seeds {0,1,2};
  the probe candidates at seed {0} only. Workflow:
    1. Run everything -> get seed-0 numbers for all + the 3-seed anchor/winner band.
    2. Promote any probe with seed-0 val mAP50-95 > anchor_mean + 0.5 to 3 seeds:
         python run_newluggage_ablation12.py --promote r12_nwd_eiou,r12_nwd_talnwd
       (or edit its `seeds` to [0,1,2]). already-done seed runs are skipped.
    3. ACCEPTANCE GATE (fixed before eval): a config is a real win iff
         mean(val mAP50-95) > mean(anchor) + 2 * std(anchor)   [primary]
       Ties/borderline broken by small AP50-95 on the finalist test eval
       (run your existing full-dataset eval script on best.pt of the finalists;
       val does not expose per-size AP).

VERIFY AT LAUNCH -- epoch-0 banner must show (loss_v1updated.py prints these):
    nwd_ratio, nwd_adaptive, nwd_c_adaptive, box_metric, tal_nwd
  If tal_nwd=1, confirm NO "[WARN] ... no iou_calculation hook" line -- that means
  your ultralytics version will not apply NWD-aware assignment.

REQUIRES -- these keys must be whitelisted in your cfg patch (already are, since
  R10/R11 used them): nwd_ratio, nwd_c, nwd_adaptive, nwd_anneal, nwd_anneal_min,
  nwd_c_adaptive, nwd_c_k, box_metric, tal_nwd, tal_nwd_c, tal_nwd_area, tal_nwd_ratio.

Usage:
  python run_newluggage_ablation12.py                       # all runs, default seeds
  python run_newluggage_ablation12.py r12_nwd_eiou          # only named base(s)
  python run_newluggage_ablation12.py --promote r12_nwd_eiou,r12_full   # add seeds 1,2
  python run_newluggage_ablation12.py --with-test           # also eval test (discouraged)
"""

import sys
import time
import gc
import copy
import json
import os
import math
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
PROJECT_DIR = "runs_newluggage_r12"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

# =============================================================================
# Full OFF baseline — every custom key stated so the anchor is bit-inert.
# (Identical to the Round-11 _ALL_OFF: same loss_v1updated.py key set.)
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


# The R10 winner (r10_nwd_fixedc), as a reusable override block.
_NWD_WIN = dict(
    nwd_ratio=0.3, nwd_c=64.0, nwd_adaptive=1, nwd_anneal=1, nwd_anneal_min=0.1,
    nwd_c_adaptive=1, nwd_c_k=0.5, small_obj_px=48,
)
_TALNWD = dict(tal_nwd=1, tal_nwd_c=8.0, tal_nwd_area=2304.0, tal_nwd_ratio=1.0)


def _win(**overrides):
    """Winner config with additional overrides composed on top."""
    return _base(**{**_NWD_WIN, **overrides})


# =============================================================================
# RUN CONFIGS  (base configs; expanded across seeds below)
# =============================================================================
R12_ANCHOR     = _base()
R12_NWD_WIN    = _win()
R12_NWD_R045   = _win(nwd_ratio=0.45)
R12_NWD_EIOU   = _win(box_metric="eiou")
R12_NWD_TALNWD = _win(**_TALNWD)
R12_FULL       = _win(box_metric="eiou", **_TALNWD)

# seeds per base: the reference pair gets the full 3-seed band up front; probes
# start at seed 0 and are promoted with --promote once they clear anchor+0.5.
CANDIDATES = [
    {"name": "r12_anchor",     "phase": "-",       "seeds": [0, 1, 2], "params": R12_ANCHOR,
     "label": "Fresh anchor — all levers inert (3-seed noise floor)"},
    {"name": "r12_nwd_win",    "phase": "R12-conf","seeds": [0, 1, 2], "params": R12_NWD_WIN,
     "label": "= r10_nwd_fixedc — CONFIRM the winner at 3 seeds"},
    {"name": "r12_nwd_r045",   "phase": "R12-push","seeds": [0],       "params": R12_NWD_R045,
     "label": "Winner + nwd_ratio 0.30->0.45 — push the lever that moved"},
    {"name": "r12_nwd_eiou",   "phase": "R12-geom","seeds": [0],       "params": R12_NWD_EIOU,
     "label": "Winner + EIoU base — NWD metric + direct w/h penalty (tall-narrow)"},
    {"name": "r12_nwd_talnwd", "phase": "R12-asgn","seeds": [0],       "params": R12_NWD_TALNWD,
     "label": "Winner + NWD-aware TAL — compose regression-NWD & assignment-NWD"},
    {"name": "r12_full",       "phase": "R12-comp","seeds": [0],       "params": R12_FULL,
     "label": "Winner + EIoU + NWD-assignment — compose (only if both clear the gate)"},
]

PROMOTE_SEEDS = [1, 2]   # seeds added to a base when passed via --promote


def build_runs(only_bases=None, promote=None):
    """Expand CANDIDATES into concrete per-seed run dicts (name suffixed _s<seed>)."""
    promote = set(promote or [])
    runs = []
    for c in CANDIDATES:
        if only_bases and c["name"] not in only_bases:
            continue
        seeds = list(c["seeds"])
        if c["name"] in promote:
            seeds = sorted(set(seeds) | set(PROMOTE_SEEDS))
        for s in seeds:
            runs.append({
                "base": c["name"], "name": f"{c['name']}_s{s}",
                "phase": c["phase"], "label": c["label"],
                "params": c["params"], "seed": s,
            })
    return runs


# =============================================================================
# Epoch sync (identical to R11)
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
            json.dump({"name": name, "base": run_cfg.get("base"),
                       "phase": run_cfg.get("phase"), "label": label,
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

    return {"name": name, "base": run_cfg.get("base"), "phase": run_cfg.get("phase"),
            "label": label, "seed": seed, "elapsed_h": elapsed,
            "val_map50": val_map50, "val_map5095": val_map5095,
            "test_map50": test_map50, "test_map5095": test_map5095}


def load_summary():
    path = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def already_done(name, summary):
    for r in summary:
        if r.get("name") == name and r.get("val_map5095") == r.get("val_map5095"):
            return True
    return False


def _mean_std(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan"), 0
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, 0.0, len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var), len(vals)


def report(summary):
    """Per-seed table + per-base mean±std aggregation + 2*std acceptance gate."""
    def fmt(v, pct=True):
        if v != v:
            return "n/a"
        return f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    print(f"\n{'=' * 84}\n  PER-SEED RESULTS\n{'=' * 84}")
    print(f"  {'Run':<20}{'Ph':>9}{'Time(h)':>9}{'val mAP50':>11}{'val 50-95':>11}"
          f"{'tst mAP50':>11}{'tst 50-95':>11}")
    print(f"  {'-' * 82}")
    for r in sorted(summary, key=lambda x: x["name"]):
        print(f"  {r['name']:<20}{str(r.get('phase', '?')):>9}"
              f"{fmt(r['elapsed_h'], pct=False):>9}{fmt(r['val_map50']):>11}"
              f"{fmt(r.get('val_map5095', float('nan'))):>11}"
              f"{fmt(r['test_map50']):>11}{fmt(r['test_map5095']):>11}")
        if r.get("error"):
            print(f"      -> failed: {r['error']}")

    # aggregate by base
    bases = {}
    for r in summary:
        b = r.get("base") or r["name"].rsplit("_s", 1)[0]
        bases.setdefault(b, []).append(r)

    a50 = [r["val_map50"] for r in bases.get("r12_anchor", [])]
    a5095 = [r["val_map5095"] for r in bases.get("r12_anchor", [])]
    amu50, astd50, an = _mean_std(a50)
    amu, astd, an95 = _mean_std(a5095)
    gate = amu + 2 * astd if amu == amu else float("nan")

    print(f"\n{'=' * 84}\n  PER-BASE MEAN +/- STD  (val split)\n{'=' * 84}")
    print(f"  {'Base':<20}{'n':>3}{'mAP50 mean':>13}{'+/-std':>9}"
          f"{'50-95 mean':>13}{'+/-std':>9}{'d50-95 vs anchor':>19}")
    print(f"  {'-' * 82}")
    for b in ["r12_anchor", "r12_nwd_win", "r12_nwd_r045", "r12_nwd_eiou",
              "r12_nwd_talnwd", "r12_full"]:
        rs = bases.get(b)
        if not rs:
            continue
        m50, s50, n50 = _mean_std([r["val_map50"] for r in rs])
        m95, s95, n95 = _mean_std([r["val_map5095"] for r in rs])
        d = (m95 - amu) * 100 if (m95 == m95 and amu == amu) else float("nan")
        dstr = "n/a" if d != d else f"{'+' if d >= 0 else ''}{d:.2f}"
        print(f"  {b:<20}{n95:>3}{fmt(m50):>13}{fmt(s50):>9}"
              f"{fmt(m95):>13}{fmt(s95):>9}{dstr:>19}")

    print(f"\n  Anchor noise floor (val mAP50-95): mean={fmt(amu)} std={fmt(astd)} "
          f"(n={an95}); ACCEPTANCE GATE = mean > {fmt(gate)} (anchor + 2*std)")
    if an95 < 3:
        print("  [!] anchor has <3 seeds -- std is unreliable; run r12_anchor at seeds 0,1,2"
              " before trusting the gate.")
    print("  DECISION: promote seed-0 probes with val 50-95 > anchor_mean + 0.5 to 3 seeds")
    print("  (--promote name1,name2), then accept only bases whose 3-seed mean clears the gate.")
    print("  Tie-break / final: run your full-dataset eval script on the finalists' best.pt")
    print("  and compare small AP50-95 (val does not expose per-size AP).")


def main():
    args = [a for a in sys.argv[1:]]
    with_test = "--with-test" in args
    promote = []
    if "--promote" in args:
        i = args.index("--promote")
        if i + 1 < len(args):
            promote = [x for x in args[i + 1].split(",") if x]
    only = {a for a in args if not a.startswith("--") and a not in promote}

    runs = build_runs(only_bases=only or None, promote=promote)

    print(f"\n{'=' * 84}")
    print("  ROUND 12 — improve & CONFIRM the NWD winner (3-seed mean+/-std protocol)")
    print(f"  Runs: {', '.join(r['name'] for r in runs)}")
    if promote:
        print(f"  Promoting to seeds {PROMOTE_SEEDS}: {', '.join(promote)}")
    if with_test:
        print("  [!] --with-test: test-split eval per run (leaks test into selection)")
    print(f"{'=' * 84}")

    overall_start = time.time()
    summary = load_summary()
    done_names = {r["name"] for r in summary}

    for run_cfg in runs:
        if already_done(run_cfg["name"], summary):
            print(f"\n  [SKIP] {run_cfg['name']} already completed (in summary.json)")
            continue
        try:
            result = run_one(run_cfg, with_test=with_test)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_cfg['name']}' failed: {e}")
            result = {"name": run_cfg["name"], "base": run_cfg.get("base"),
                      "phase": run_cfg.get("phase"), "label": run_cfg["label"],
                      "seed": run_cfg.get("seed", SEED), "elapsed_h": float("nan"),
                      "val_map50": float("nan"), "val_map5095": float("nan"),
                      "test_map50": float("nan"), "test_map5095": float("nan"),
                      "error": str(e)}

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
    print(f"\n{'=' * 84}\n  ALL RUNS COMPLETE ({total_elapsed:.2f}h total)\n{'=' * 84}")
    report(summary)


if __name__ == "__main__":
    main()
