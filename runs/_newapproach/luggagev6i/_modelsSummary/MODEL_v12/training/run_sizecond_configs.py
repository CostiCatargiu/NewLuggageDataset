#!/usr/bin/env python3
"""
SIZE-CONDITIONAL LB-TAL (the F9 mechanism) + Section S, on measured budgets.

=============================================================================
WHY — straight from the 4-pass footprint diagnostic
=============================================================================
ONE GLOBAL BUDGET CANNOT SERVE ALL THREE SIZES:
  * small GTs peak around P3=4 and are P4-SUPPLY-limited (2.46 candidates),
  * large GTs need a MINIMAL P3 — they have 501 stride-8 candidates, so any P3
    budget forces junk positives that collapsed large IoU 0.876 -> 0.720 and
    killed the original cmb.
p4wide {8:4,16:7,32:1} is the best single global 3-tuple (cmb_p4wide: 0.5560
overall, small 0.5115 — 8/8 small-object metrics and the best mAP50 of all 28
runs). size_cond gives EACH GT a budget chosen by its own max-side size, which
is what F9 said "would dominate any single global 3-tuple".

BUDGETS ARE _F9 IN THIS FILE, NOT THE LOSS DEFAULT. Predicted allocation from
the measured metric-valid ceilings (diag_fp_out_v6i):

    scheme                      small tot   large P3   large tot
    stock         (measured)        7.73       0.16       9.80  -> 0.5477 ANCHOR
    p4wide        (measured)        5.91       2.17       9.54  -> best small cfg
    loss default  {5/4/1,1/7/2}     6.86       0.68       8.76
    _F9           {4/4/0,0/7/2}     5.72       0.00       8.08

The loss default puts small at 6.86 total positives. The two best small-object
configs sat at 5.73 and 5.91; stock's 7.73 is the ANCHOR. More small supply
measured WORSE, twice — so _F9 pulls small back to 5.72 and takes large's P3 to
0, matching stock's 0.16 instead of forcing 0.68 junk picks. See the comment
block above _F9 for the full derivation.

=============================================================================
THE RUNS — 4 pending, 2 already executed (commented out below)
=============================================================================
  1. cmb_sizecond    sqrt0703 + size_cond(_F9) — THE headline candidate.
  2. lb_sizecond     size_cond on all-off — ATTRIBUTION for #1 (allocation, or
                     the sqrt combination?) and vs lb_p4wide / lb_uniform.
  3. cmb_sizecond_aggr  probes the OTHER direction on small (P3=6, ~8.0 total,
                     stock territory). A probe of the supply axis, not a
                     candidate.
  4. snt             sqrt0703 + Section S, the scale-normalised confidence
                     target. SINGLE AXIS, deliberately NOT stacked with
                     size_cond — see below.

ALREADY EXECUTED, kept commented for provenance:
  cmb_p4wide_clsswa  cmb_p4wide + cls_swa 1.75
  cmb_p4wide_seed1   seed-1 confirm of cmb_p4wide

  NOTE on the seed run: cmb_p4wide (0.5560) sits 0.04 pp from sqrt0703
  (0.5564) against 0.0012 measured seed noise, so "beats cmb_p4wide" in the
  summary below is a comparison against a number that needs its seed-1 result
  read alongside it. Pull that run's test_map5095 out of its summary.json and
  put it in CURRENT_BEST_SEED1 so the comparison is honest.

WHY snt IS NOT STACKED WITH size_cond. Both target small-object confidence by
different routes. Effects here are ~0.005 against 0.0012 measured seed noise,
so a stacked result cannot be attributed, and a null cannot be told apart from
two effects cancelling. Stack them only if BOTH show signal alone.

WHAT SECTION S DOES. TAL trains each GT toward a confidence ceiling equal to
its own best achievable IoU (pos_overlaps). Measured peak target per GT:
small 0.8365 / medium 0.8933 / large 0.9028 — small objects are explicitly
taught to be 6.4% less confident, which is the AR50_small 0.95 -> R50_small
0.70 gap. That ceiling is INVARIANT to allocation (under p4wide it reads
0.8366, identical to stock), so no assigner — including size_cond — can move
it. SNT rescales by ema_global/ema_size. RISK: it may cost precision; WATCH
P50_small AS CLOSELY AS R50_small.

Baselines: cmb_p4wide 0.5560 / small 0.5115 (current best), lb_uniform 0.5557,
sqrt0703 0.5564 (still champion), anchor 0.5477. Seed noise 0.0012 overall,
0.0206 on large. READ small mAP + P50_small (CocoEvalAllFolders_luggage.py).

VERIFY FIRST (5 min, no training): diag_anchor_footprint.py with the size_cond
assigner — confirm small GTs draw ~4/2.1/0 and large ~0/6.8/1.3, i.e. large's
stride-8 count hits 0 while small stays near 5.7 total.

REQUIRES lossv2updated.py (mode='size_cond' + set_gt_sizes + Section S)
installed as ultralytics/utils/loss.py; lbtal_size_* and snt_* whitelisted in
default.yaml. preflight() scans the installed loss SOURCE for every key each
selected config activates and aborts on a no-op, and aborts on run-dir
collisions before burning hours (see OVERWRITE_EXISTING to bypass).

Usage:
    python run_sizecond_configs.py
    python run_sizecond_configs.py cmb_sizecond snt
"""

import sys
import time
import gc
import copy
import json
import os
import hashlib

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_sizecond"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 54
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
BASELINE_TEST_MAP5095 = 0.5477
CURRENT_BEST = 0.5560   # cmb_p4wide, seed 0

# Fill in from runs_sizecond/summary.json once cmb_p4wide_seed1 is read back.
# Leave None to have summarise() say so rather than imply the seed-0 number is
# confirmed: 0.5560 vs sqrt0703's 0.5564 is a 0.04 pp gap against 0.12 pp seed
# noise, so the ranking between them is not established by seed 0 alone.
CURRENT_BEST_SEED1 = None

# Set True to reuse existing run directories instead of aborting on them.
# This sets exist_ok=True on model.train(), so ultralytics WRITES INTO the
# existing folder — weights, results.csv and ablation_params.json are
# overwritten. Bypassing ONLY the preflight check is not enough: exist_ok would
# still be False, every colliding run would raise at launch, get swallowed by
# the per-run exception handler, and be recorded as an error AFTER the
# non-colliding runs had already spent their hours. So this flag controls both.
#
# The budgets changed (_F9: small 5/4/1 -> 4/4/0, large 1/7/2 -> 0/7/2), so any
# results already sitting in those folders came from a DIFFERENT config. Check
# ablation_params.json before treating overwritten numbers as comparable.
OVERWRITE_EXISTING = False

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
    use_lbtal=False,
)

_SQRT0703 = dict(
    _ALL_OFF,
    area_weight_mode="sqrt",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)


def _lb_sizecond(base, size_budgets=None, quality_gate=0.0, cls_swa=None):
    cfg = dict(base, use_lbtal=True, lbtal_mode="size_cond",
               lbtal_quality_gate=quality_gate)
    if size_budgets is not None:
        cfg["lbtal_size_budgets"] = size_budgets
    if cls_swa is not None:
        cfg["use_cls_swa"] = True
        cfg["cls_swa_boost"] = cls_swa
        cfg["small_obj_px"] = cfg.get("small_obj_px", 48) or 48
    return cfg


def _sq_p4wide(cls_swa=None):
    cfg = dict(_SQRT0703, use_lbtal=True, lbtal_mode="fixed",
               lbtal_level_topk={8: 4, 16: 7, 32: 1})
    if cls_swa is not None:
        cfg["use_cls_swa"] = True
        cfg["cls_swa_boost"] = cls_swa
    return cfg


# =============================================================================
# SIZE-CONDITIONAL BUDGETS — tuned against the measured footprint, not guessed
# =============================================================================
# Predicted allocation from the four footprint passes (metric-valid ceilings in
# diag_fp_out_v6i), small and large totals per GT:
#
#   scheme                       small tot   large P3   large tot
#   stock          (measured)        7.73       0.16       9.80   -> 0.5477 ANCHOR
#   p4wide         (measured)        5.91       2.17       9.54   -> best small cfg
#   loss default   {5/4/1,·,1/7/2}   6.86       0.68       8.76
#   _F9 below      {4/4/0,·,0/7/2}   5.72       0.00       8.08
#
# Two changes from the loss-file default, both measured rather than assumed:
#
#   small P3 5->4, P5 1->0.  The two best small-object configs sat at 5.73
#     (lb_uniform) and 5.91 (cmb_p4wide) total positives per small GT. Stock's
#     7.73 is the ANCHOR at 0.5477 — MORE small supply measured WORSE, twice.
#     The loss default lands at 6.86, i.e. on the losing side of that. P5 goes
#     to 0 because small selects 0.00-0.01 there in every pass: those anchors
#     exist geometrically but carry a ~0 alignment metric.
#
#   large P3 1->0.  Stock takes 0.16 stride-8 positives per large GT; a budget
#     of 1 forces ~0.68. Those are the junk picks that collapsed large quality
#     under uniform (mean IoU 0.876 -> 0.720) and killed the original cmb.
#     Zero is legal — the assigner does `if k <= 0: continue` — and large keeps
#     125 P4 candidates, so it cannot be starved.
#
# _AGGR deliberately probes the OTHER direction on small (P3=6 -> ~8.0 total,
# stock territory). Kept as a probe, not a candidate; its large row is fixed to
# 0 as well since that change has no downside.
_F9 = {"small":  {8: 4, 16: 4, 32: 0},
       "medium": {8: 4, 16: 6, 32: 1},
       "large":  {8: 0, 16: 7, 32: 2}}

_AGGR = {"small": {8: 6, 16: 3, 32: 0},
         "medium": {8: 4, 16: 6, 32: 1},
         "large": {8: 0, 16: 8, 32: 1}}


def _snt(base, strength=1.0):
    """Section S: scale-normalised confidence target (in lossv2updated.py)."""
    return dict(base, use_snt=True, snt_strength=strength, snt_momentum=0.02,
                snt_max_scale=1.15, snt_warmup_epochs=3,
                snt_small_px=48.0, snt_medium_px=96.0)


RUNS = [
    {"name": "cmb_sizecond", "batch": BATCH,
     "label": "sqrt0703 + size_cond (_F9 tuned budgets) — the headline candidate",
     "params": _lb_sizecond(_SQRT0703, size_budgets=_F9)},

    {"name": "lb_sizecond", "batch": BATCH,
     "label": "size_cond on all-off base — attribution + vs lb_p4wide/lb_uniform",
     "params": _lb_sizecond(_ALL_OFF, size_budgets=_F9)},

    {"name": "cmb_sizecond_aggr", "batch": BATCH,
     "label": "sqrt0703 + size_cond aggressive {s:6/3/0} — probes MORE small supply",
     "params": _lb_sizecond(_SQRT0703, size_budgets=_AGGR)},

    # Section S — kept as a SINGLE-AXIS test, deliberately NOT stacked with
    # size_cond. Both target small-object confidence by different routes; with
    # effects of ~0.005 against 0.0012 seed noise, a stacked result cannot be
    # attributed and a null cannot be told apart from two effects cancelling.
    # Stack them only if BOTH show signal on their own.
    # {"name": "snt", "batch": BATCH,
    #  "label": "sqrt0703 + scale-normalised confidence target — the one axis no assigner can reach",
    #  "params": _snt(_SQRT0703)},

    # ---------------------------------------------------------------------
    # ALREADY EXECUTED — left commented for provenance, not deleted.
    #
    # cls_swa is pos_boost's structural twin: both scale the CLS LOSS WEIGHT
    # on small/selected positives. pos_boost measured -0.58 pp and moved
    # R50_small by 0.34 pp, because loss weighting cannot push a prediction
    # past its own target ceiling (pos_overlaps) — which is exactly what
    # Section S changes instead. Read this run's result against that
    # prediction: if it also lands ~-0.5 pp, the ceiling argument holds and
    # the whole cls-weighting family is closed.
    # {"name": "cmb_p4wide_clsswa", "batch": BATCH,
    #  "label": "cmb_p4wide + cls_swa 1.75 — third axis (scoring) on the current best",
    #  "params": _sq_p4wide(cls_swa=1.75)},
    #
    # The seed-1 confirm of the current best. Put its test_map5095 into
    # CURRENT_BEST_SEED1 above — without it, every "vs_best" number below is
    # measured against a single-seed 0.5560 that sits 0.04 pp from sqrt0703.
    # {"name": "cmb_p4wide_seed1", "batch": BATCH, "seed": 1,
    #  "label": "cmb_p4wide SEED 1 — confirm the current best before comparing",
    #  "params": _sq_p4wide()},
]


def loss_provenance():
    info = {"path": None, "md5": None, "has_size_cond": False,
            "has_set_gt_sizes": False, "has_snt": False, "_body": ""}
    try:
        import ultralytics.utils.loss as _lm
        path = getattr(_lm, "__file__", None)
        info["path"] = path
        if path and os.path.exists(path):
            src = open(path, "rb").read()
            info["md5"] = hashlib.md5(src).hexdigest()[:12]
            txt = src.decode("utf-8", "ignore")
            info["has_size_cond"] = "size_cond" in txt
            info["has_set_gt_sizes"] = "set_gt_sizes" in txt
            info["_body"] = "\n".join(
                l for l in txt.split("\n") if not l.strip().startswith("#"))
            info["has_snt"] = "use_snt" in info["_body"]
    except Exception as e:
        info["error"] = str(e)
    return info


LOSS_INFO = loss_provenance()


def unimplemented_params(params):
    """Keys a config tries to ACTIVATE that the installed loss never reads.

    default.yaml whitelists the union of every mechanism ever written, so the
    cfg checker happily accepts keys the installed loss ignores — the run then
    trains a bit-identical copy of its base config and burns 1.5 GPU-h
    reproducing a number you already have. That is exactly how `posboost` was
    ranked HIGH for a whole sweep before anyone noticed use_pos_boost appeared
    nowhere in lossv2updated.py. Only keys whose value DIFFERS from the all-off
    default are checked.
    """
    body = LOSS_INFO.get("_body") or ""
    if not body:
        return []
    return sorted(k for k, v in params.items()
                  if not (k in _ALL_OFF and _ALL_OFF[k] == v) and k not in body)


def preflight(todo):
    print(f"  loss.py: {LOSS_INFO.get('path')}")
    print(f"  md5={LOSS_INFO.get('md5')}  size_cond={LOSS_INFO.get('has_size_cond')} "
          f"set_gt_sizes={LOSS_INFO.get('has_set_gt_sizes')} "
          f"snt={LOSS_INFO.get('has_snt')}")
    missing = [k for k in ("has_size_cond", "has_set_gt_sizes") if not LOSS_INFO.get(k)]
    if missing:
        print(f"\n  [ABORT] installed loss.py missing: {', '.join(missing)}")
        print("  Install the updated lossv2updated.py as ultralytics/utils/loss.py.")
        return False

    need_snt = any(r["params"].get("use_snt") for r in todo)
    if need_snt and not LOSS_INFO.get("has_snt"):
        print("\n  [ABORT] a selected config sets use_snt but the installed "
              "loss.py has no Section S.")
        return False

    bad = {r["name"]: unimplemented_params(r["params"]) for r in todo}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print("\n  [ABORT] these configs set params the installed loss NEVER READS.")
        print("          Each would train an exact copy of its base config:")
        for name, ks in bad.items():
            print(f"            {name:<22} {', '.join(ks)}")
        return False

    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))]
    if clash and OVERWRITE_EXISTING:
        print(f"\n  [warn] OVERWRITE_EXISTING=True — writing into existing run "
              f"dirs: {', '.join(clash)}")
        print("         Their weights/results.csv will be replaced. If any were")
        print("         trained before the _F9 budget change they are not")
        print("         comparable anyway; check ablation_params.json.")
        clash = []
    if clash:
        print(f"\n  [ABORT] run dirs already exist (exist_ok=False): {', '.join(clash)}")
        print("  Delete or rename them, or set OVERWRITE_EXISTING=True at the top.")
        print("  Otherwise those runs fail after the earlier ones have already")
        print("  burned their hours.")
        return False
    return True


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
    name, params, batch = rc["name"], rc["params"], rc["batch"]
    seed = rc.get("seed", 0)
    print(f"\n{'=' * 76}\n  RUN {name}  (seed {seed})\n  {rc['label']}\n"
          f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(params))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    try:
        with open(os.path.join(save_dir, "ablation_params.json"), "w") as f:
            json.dump({"name": name, "label": rc["label"], "params": params,
                       "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": batch,
                       "seed": seed, "loss_file": {k: v for k, v in LOSS_INFO.items()
                                                   if k != "_body"}}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    out = {"name": name, "seed": seed, "batch": batch, "hours": hours,
           "save_dir": save_dir, "weights": weights,
           "params": copy.deepcopy(params),
           "loss_file": {k: v for k, v in LOSS_INFO.items() if k != "_body"},
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        tm = YOLO(weights).val(
            data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
            device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    print(f"\n{'=' * 76}\n  SIZE-CONDITIONAL RESULTS (test)\n{'=' * 76}")
    print(f"{'run':<22}{'seed':>5}{'mAP50':>9}{'mAP50-95':>11}{'d_anchor':>10}{'vs_best':>9}{'h':>6}")
    print("-" * 76)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        d = ("%+9.2f" % ((r["test_map5095"] - BASELINE_TEST_MAP5095) * 100))
        vb = ("%+8.2f" % ((r["test_map5095"] - CURRENT_BEST) * 100))
        print(f"{r['name']:<22}{r['seed']:>5}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{d}{vb}{r['hours']:>6.1f}")

    print(f"\n  anchor {BASELINE_TEST_MAP5095*100:.2f} | "
          f"current best cmb_p4wide {CURRENT_BEST*100:.2f} (seed 0)")
    if CURRENT_BEST_SEED1 is None:
        print("  [!] CURRENT_BEST_SEED1 is unset. cmb_p4wide's 0.5560 sits 0.04 pp")
        print("      from sqrt0703's 0.5564 against 0.12 pp seed noise, so vs_best")
        print("      is measured against an unconfirmed number. Read the executed")
        print("      cmb_p4wide_seed1 run and fill it in.")
    else:
        spread = abs(CURRENT_BEST - CURRENT_BEST_SEED1) * 100
        print(f"  cmb_p4wide seed 1 = {CURRENT_BEST_SEED1*100:.2f} "
              f"(seed spread {spread:.2f} pp)")
        print(f"  Treat any vs_best below {max(spread, 0.12):.2f} pp as unresolved.")

    print("\n  vs_best > 0 -> beats cmb_p4wide on OVERALL. But the target is SMALL mAP,")
    print("  and cmb_p4wide won 8/8 small metrics while LOSING 7/8 large ones —")
    print("  the overall number averages that trade away. Run")
    print("  CocoEvalAllFolders_luggage.py on the weights below (cmb_p4wide small")
    print("  = 0.5115) and read P50_small too, especially for snt.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 76}\n  SIZE-CONDITIONAL LB-TAL (F9) + combos — {len(todo)} runs")
    print(f"  target: beat cmb_p4wide (overall 55.60, small 51.15)")
    print(f"  runs: {', '.join(r['name'] for r in todo)}  (~{1.5*len(todo):.0f} GPU-h)")
    print(f"{'=' * 76}\n")
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "seed": r.get("seed", 0), "batch": r["batch"],
                        "hours": float("nan"), "test_map50": float("nan"),
                        "test_map5095": float("nan"), "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
