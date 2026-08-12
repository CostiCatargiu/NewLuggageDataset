#!/usr/bin/env python3
"""
YOLO26 PHASE 2 — finish tuning SWA on the boost axis, THEN combine with LB-TAL.

=============================================================================
WHY THIS ORDER
=============================================================================
Phase 1 swept the ALPHA schedule and found a06_03 (alpha 0.6->0.3) best at
55.59. It did NOT touch small_obj_boost: all nine YOLO26 runs to date used
boost 2.0, inherited from v12. Combining a half-tuned SWA with LB-TAL would
confound "the combination failed" with "the SWA half was mis-set", which is
exactly the ambiguity that makes the v12 combination results hard to read.

So: finish the SWA axis first (PHASE A), then combine the winner (PHASE B).

BOOST IS THE STRONGER DOSE LEVER. alpha only blends area weight against score
weight; small_obj_boost is a hard multiplier applied to every box below
small_obj_px. On v12 at alpha 0.7->0.3 it had a clean interior peak:

        boost   1.0     1.5     2.0     2.5
        v12    55.21   55.49   55.64   55.47      (anchor 54.77)

That peak was found at v12's optimal alpha. YOLO26's optimal alpha is
different (0.6 not 0.7), so the boost optimum may well have moved with it.

=============================================================================
WHAT IS ALREADY KNOWN — do not re-run
=============================================================================
    YOLO26 @640 b82 seed0                mAP    small    med   large
      yolo26_custom-9  stock ANCHOR     55.24   51.00  65.98   60.87
      y26_swa_a06_03   a0.6->0.3 b2.0   55.59   51.32  66.44   57.19  <- best
      y26_lb_uniform   LB uniform       55.52   51.75  65.31   58.58
      y26_lb_coarse244 LB {8:2,16:4,32:4} 55.34 50.99  65.98   60.21
      y26_sqrt0703     a0.7->0.3 b2.0   55.27   51.18  65.48   56.70
      y26_lb_p3_3      LB {8:3,16:4,32:4} 55.16 50.88  66.22   56.93

=============================================================================
PHASE A — the boost axis at YOLO26's optimal alpha (run these two first)
=============================================================================
  y26_swa_a06_b15   alpha 0.6->0.3, boost 1.5
  y26_swa_a06_b25   alpha 0.6->0.3, boost 2.5

With a06_03 (boost 2.0, 55.59) already measured that gives a three-point curve
at fixed alpha. A real dose effect shows a peak or a monotone trend; noise
shows a scatter. The alpha sweep gave a scatter, so this is also a second,
independent test of whether SWA has any structure on YOLO26 at all.

=============================================================================
PHASE B — combine the phase-A winner with LB-TAL (run only after A)
=============================================================================
  y26_lb_p4wide      LB {8:4,16:7,32:1} ALONE   <- the missing control
  y26_cmb_p4wide     SWA(winner) + LB {8:4,16:7,32:1}

WHY p4wide. cmb_p4wide was the best combination on v12 (55.60) and the only
one that was SUPER-ADDITIVE on small: 51.15, above BOTH parents (sqrt0703
50.63, lb_p4wide 50.97), with large landing at the budget's own value rather
than compounding. The other v12 combination behaved oppositely — uniform + SWA
gave large 54.37, below both parents. Two combos, two different behaviours on
large, n=2. That is genuinely undecided, which is why it is worth one run.

WHY THE CONTROL IS NOT OPTIONAL. {8:4,16:7,32:1} has never been run on YOLO26.
Without y26_lb_p4wide, a result from y26_cmb_p4wide cannot be attributed to
the SWA half, the budget, or their interaction. a06_03 supplies the other
parent, so the two runs here complete a readable 2x2.

!! SET BEST_BOOST FROM PHASE A BEFORE RUNNING PHASE B. It defaults to 2.0
   (a06_03, the current best). If phase A moves the optimum and this is not
   updated, phase B tests a combination built on a superseded parent.

=============================================================================
READING THE RESULT
=============================================================================
SMALL is the number to watch in phase B. Above 51.32 (a06_03) means the v12
super-additivity reproduced, and above 51.75 would make it the best small-object
config in the project across both detector families. LARGE is the risk: p4wide
gives P5 a budget of 1 out of 10, and every custom YOLO26 config is already
negative on large (the anchor holds the best value at 60.87).

NOISE. Still unmeasured on YOLO26. In the phase-1 sweep, a09_03 and a08_04
have identical mean alpha and differ by 0.45 pp — treat that as a floor. A
three-point boost curve spanning less than ~0.4 pp is not readable.

Usage:
    python run_yolo26_phase2_v6i.py phaseA
    #  ... read results, set BEST_BOOST ...
    python run_yolo26_phase2_v6i.py phaseB
    python run_yolo26_phase2_v6i.py y26_swa_a06_b15
"""

import copy
import gc
import hashlib
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_CFG = "yolo26s.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_phase2_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                 # must match yolo26_custom-9 and the phase-1 sweep
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

ANCHOR = 0.5524            # yolo26_custom-9
BEST_SINGLE = 0.5559       # y26_swa_a06_03
BEST_SMALL = 0.5175        # y26_lb_uniform — the small-object bar to beat

# !! PHASE B PARENT. Set from phase A. 2.0 = a06_03, the current best.
BEST_BOOST = 2.0

KNOWN = {"yolo26 ANCHOR": (0.5524, 0.5100, 0.6087),
         "y26_swa_a06_03": (0.5559, 0.5132, 0.5719),
         "y26_lb_uniform": (0.5552, 0.5175, 0.5858),
         "y26_lb_coarse244": (0.5534, 0.5099, 0.6021)}

_STOCK = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False, lbtal_mode="uniform", lbtal_level_topk=None,
    lbtal_min_level_k=1, lbtal_quality_gate=0.0,
    tal_alpha=0.5, tal_beta=6.0,
)
CUSTOM_KEYS = tuple(_STOCK)
P4WIDE = {8: 4, 16: 7, 32: 1}


def _swa(start, end, boost=2.0, px=48, **extra):
    return dict(_STOCK, alpha_start=start, alpha_end=end,
                alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost, **extra)


def _lb(budget):
    return dict(use_lbtal=True, lbtal_mode="fixed",
                lbtal_level_topk=budget, lbtal_min_level_k=1)


def _alpha_at(p, epoch):
    prog = epoch / max(EPOCHS, 1)
    a = p["alpha_start"] * (1 - prog) + p["alpha_end"] * prog
    return max(p["alpha_min"], min(p["alpha_max"], a))


RUNS = [
    # ---------------------------------------------------------------- PHASE A
    {"name": "y26_swa_a06_b15", "phase": "A", "params": _swa(0.6, 0.3, boost=1.5),
     "label": "SWA a0.6->0.3 px48 boost 1.5  [v12 @a0.7: 55.49]",
     "why": "Halves the dose on the lever alpha could not move. a06_03 lost 3.68 "
            "pp of large; if that is an over-dose it is more likely coming from "
            "the hard 2x small-object multiplier than from the blend weight. "
            "Watch large: recovery toward 59-60 is the signal."},

    {"name": "y26_swa_a06_b25", "phase": "A", "params": _swa(0.6, 0.3, boost=2.5),
     "label": "SWA a0.6->0.3 px48 boost 2.5  [v12 @a0.7: 55.47]",
     "why": "The dose-INCREASE control that makes phase A two-sided. With b15 "
            "and a06_03 (b2.0) it gives 1.5 / 2.0 / 2.5 at fixed alpha. A peak "
            "at 2.0 means the v12 optimum survived the alpha change; a monotone "
            "trend means the optimum has moved and the axis is worth extending."},

    # ---------------------------------------------------------------- PHASE B
    {"name": "y26_lb_p4wide", "phase": "B", "params": dict(_STOCK, **_lb(P4WIDE)),
     "label": "LB-TAL fixed {8:4,16:7,32:1} ALONE — the missing control  [v12 +0.26]",
     "why": "The one budget used by v12's best combination that has never been "
            "run alone on YOLO26. Without it the combination below is "
            "un-attributable. On v12 it was the weakest LB variant (+0.26) and "
            "cost large (55.29 vs the 57.73 anchor), so a low score here is "
            "expected and is still the number that makes the combo readable."},

    {"name": "y26_cmb_p4wide", "phase": "B",
     "params": _swa(0.6, 0.3, boost=BEST_BOOST, **_lb(P4WIDE)),
     "label": "SWA a0.6->0.3 (phase-A boost) + LB {8:4,16:7,32:1} — v12's best combo, re-tuned",
     "why": "cmb_p4wide was v12's best combination (55.60) and the only one that "
            "was super-additive on small (51.15, above both parents). Every v12 "
            "combination used alpha 0.7->0.3 — none was ever re-tuned. This runs "
            "the same recipe with the SWA half set to YOLO26's own optimum. "
            "Success = small above 51.32; failure mode = large near 54 if the "
            "cmb_lbU_swa0703 compounding pattern holds instead."},

    # ---------------------------------------------------------------- PHASE C
    # Added after phase B. Judge these on SMALL, not on overall mAP50-95.
    {"name": "y26_cmb_uniform", "phase": "C",
     "params": _swa(0.6, 0.3, boost=BEST_BOOST, use_lbtal=True,
                    lbtal_mode="uniform", lbtal_level_topk=None, lbtal_min_level_k=1),
     "label": "SWA a0.6->0.3 boost2.0 + LB-TAL uniform — the two best single configs",
     "why": "The pairing of YOLO26's two top runs (a06_03 55.59, lb_uniform 55.52), "
            "and the one combination never tested on either model. I argued against "
            "it because v12's cmb_lbU_swa0703 lost 0.53 pp to both parents — but "
            "that argument only looked at overall mAP.\n"
            "  THE PATTERN IT ACTUALLY TESTS: SWA+LB combinations are SUPER-ADDITIVE "
            "ON SMALL on both detectors so far.\n"
            "      v12 cmb_p4wide     parents 50.63 / 50.97 -> combo 51.15\n"
            "      Y26 cmb_p4wide     parents 51.32 / 50.75 -> combo 51.37\n"
            "  Two for two, above both parents each time. What kills these combos is "
            "always LARGE, never small. Here the parents are 51.32 and 51.75, so a "
            "third repetition lands ABOVE 51.75 — a new best small in the project "
            "across both detector families.\n"
            "  EXPECTATION, stated up front: overall ~55.2-55.4 (a tie or slightly "
            "below a06_03) and large near 54-55 from compounding. Judged on mAP50-95 "
            "this will look like another null. Judged on small it is the single most "
            "likely config in the whole space to beat 51.75, and small is the metric "
            "that matters for unattended luggage. Both parents are already measured, "
            "so this completes a 2x2 with no extra control."},

    {"name": "y26_swa_a06_03_seed1", "phase": "C", "seed": 1,
     "params": _swa(0.6, 0.3, boost=2.0),
     "label": "a06_03 repeated at SEED 1 — the noise floor, finally",
     "why": "13 custom runs, best +0.35, and NO measured noise floor on YOLO26. The "
            "only evidence is indirect: a09_03 and a08_04 share an identical mean "
            "alpha and differ by 0.45 pp, and the phase-A boost curve spans 0.41 pp. "
            "If the floor really is ~0.45 then every result in this campaign, "
            "including the 51.75 small record, is inside it and none of it is "
            "reportable. If it comes back within 0.1 of 55.59, the whole leaderboard "
            "becomes defensible. This is the highest-value single run remaining and "
            "it should have been run first."},
]


# ============================================================== epoch plumbing
def iter_bbox_losses(criterion):
    """Yield every BboxLoss reachable from a criterion. E2ELoss holds .one2many
    and .one2one, each with its own; a plain v8DetectionLoss holds one directly."""
    seen, stack = set(), [criterion]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        bl = getattr(obj, "bbox_loss", None)
        if bl is not None and hasattr(bl, "get_dynamic_alpha"):
            yield bl
        for attr in ("one2many", "one2one"):
            if hasattr(obj, attr):
                stack.append(getattr(obj, attr))


def get_criteria(trainer):
    out = []
    for holder in (getattr(trainer, "model", None), trainer):
        crit = getattr(holder, "criterion", None)
        if crit is not None:
            out.append(crit)
        mod = getattr(holder, "module", None)
        if mod is not None and getattr(mod, "criterion", None) is not None:
            out.append(mod.criterion)
    return out


_ALPHA_SEEN, _WIRED = {}, {"n": 0}


def set_epoch(trainer):
    """Ultralytics builds the criterion lazily on the first forward, so finding
    no BboxLoss at epoch 0 is expected and harmless (BboxLoss.__init__ already
    sets epoch=0 and reads total_epochs from hyp). At epoch >= 1 it means the
    layout changed, alpha would freeze, and the run is void."""
    epoch = int(getattr(trainer, "epoch", 0))
    found = []
    for crit in get_criteria(trainer):
        for bl in iter_bbox_losses(crit):
            bl.epoch, bl.total_epochs = epoch, EPOCHS
            found.append(bl)
    if not found:
        if epoch == 0:
            print("  [callback] criterion not built yet at epoch 0 (lazy init) — wires at epoch 1")
            return
        raise RuntimeError(f"epoch callback reached NO BboxLoss at epoch {epoch} — run is void.")

    bl = found[0]
    alpha = bl.get_dynamic_alpha()
    _ALPHA_SEEN.setdefault(epoch, round(alpha, 6))
    if _WIRED["n"] == 0:
        _WIRED["n"] = len(found)
        print(f"\n  {'=' * 66}\n  LOSS WIRED — {len(found)} BboxLoss at epoch {epoch}\n  {'=' * 66}")
        print(f"    SWA={bl.swa_enabled()}  mode={bl.area_weight_mode}/{bl.area_weight_norm}")
        print(f"    alpha {bl.alpha_start} -> {bl.alpha_end}  clip [{bl.alpha_min}, {bl.alpha_max}]")
        print(f"    small_obj px<{bl.small_obj_px}  boost x{bl.small_obj_boost}")
        print(f"    dfl_loss: {'present' if bl.dfl_loss else 'None (reg_max=1, L1 branch)'}")
        for crit in get_criteria(trainer):
            for br in ("one2many", "one2one"):
                asg = getattr(getattr(crit, br, None), "assigner", None)
                if asg is not None:
                    print(f"    assigner[{br:8}]: {type(asg).__name__} topk={asg.topk} topk2={asg.topk2}")
        print(f"  {'=' * 66}\n")
    if bl.swa_enabled() and (epoch < 2 or epoch % 10 == 0 or epoch == EPOCHS - 1):
        print(f"  [SWA] epoch {epoch:>3}/{EPOCHS}  alpha={alpha:.4f}")


# ================================================================== preflight
def env_provenance():
    info = {"loss_md5": None, "tal_md5": None, "has_swa": False,
            "has_lbtal": False, "loss_path": None, "missing_keys": []}
    try:
        import ultralytics.utils.loss as _lm
        import ultralytics.utils.tal as _tm
        info["loss_path"] = getattr(_lm, "__file__", None)
        for mod, k in ((_lm, "loss_md5"), (_tm, "tal_md5")):
            p = getattr(mod, "__file__", None)
            if p and os.path.exists(p):
                info[k] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
        info["has_swa"] = hasattr(_lm.BboxLoss, "swa_weight")
        info["has_lbtal"] = hasattr(_tm, "LevelBalancedTaskAlignedAssigner")
    except Exception as e:
        info["import_error"] = str(e)
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        info["missing_keys"] = [k for k in CUSTOM_KEYS if k not in DEFAULT_CFG_DICT]
    except Exception as e:
        info["cfg_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  loss.py : {ENV.get('loss_path')}")
    print(f"  md5     : loss={ENV.get('loss_md5')}  tal={ENV.get('tal_md5')}")
    print(f"  swa_weight={ENV['has_swa']}  LB-TAL={ENV['has_lbtal']}  "
          f"missing keys={ENV['missing_keys'] or 'none'}")
    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import ultralytics: {ENV['import_error']}")
        return False
    if not ENV["has_swa"]:
        print("\n  [ABORT] BboxLoss.swa_weight missing — the ultralytics on the import "
              "path is NOT the patched tree; every run would come out stock.")
        return False
    if not ENV["has_lbtal"] and any(r["params"]["use_lbtal"] for r in todo):
        print("\n  [ABORT] LevelBalancedTaskAlignedAssigner missing but an LB-TAL run is queued.")
        return False
    if ENV["missing_keys"]:
        print(f"\n  [ABORT] keys absent from default.yaml, they would be dropped "
              f"silently: {ENV['missing_keys']}")
        return False

    for r in todo:
        p, n = r["params"], r["name"]
        bad = [k for k in p if k not in CUSTOM_KEYS]
        if bad:
            print(f"\n  [ABORT] {n}: keys not in the ported set: {bad}")
            return False
        swa_on = max(p["alpha_start"], p["alpha_end"], p["alpha_max"]) > 0
        if p["area_weight_mode"] != "inv" and not swa_on:
            print(f"\n  [ABORT] {n}: area_weight_mode set but every alpha is 0 — no-op.")
            return False
        if p["small_obj_boost"] != 1.0 and not p["small_obj_px"]:
            print(f"\n  [ABORT] {n}: small_obj_boost set but small_obj_px=0 — boost never applies.")
            return False
        if p["use_lbtal"] and p["lbtal_mode"] == "fixed" and not p["lbtal_level_topk"]:
            print(f"\n  [ABORT] {n}: mode='fixed' with no budget — silently degrades to uniform.")
            return False
        if p["lbtal_level_topk"] and not p["use_lbtal"]:
            print(f"\n  [ABORT] {n}: budget set but use_lbtal=False.")
            return False

    if any(r["phase"] == "B" for r in todo):
        print(f"\n  [!] PHASE B uses BEST_BOOST={BEST_BOOST}. Confirm this is phase A's "
              f"winner,\n      otherwise the combination is built on a superseded parent.")
    print(f"  [!] BATCH={BATCH} must match yolo26_custom-9 ({ANCHOR * 100:.2f}), "
          f"else every delta is confounded.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


# ======================================================================= train
def run_one(rc):
    name, p = rc["name"], rc["params"]
    seed = rc.get("seed", SEED)          # per-run override, for the seed repeat
    active = {k: v for k, v in p.items() if v != _STOCK[k]}
    print(f"\n{'=' * 78}\n  RUN {name}   [phase {rc['phase']}]\n  {rc['label']}\n{'=' * 78}")
    print(f"  model={MODEL_CFG} imgsz={IMG_SIZE} batch={BATCH} epochs={EPOCHS} seed={seed}"
          + ("   <-- SEED OVERRIDE" if seed != SEED else ""))
    for k, v in sorted(active.items()):
        print(f"      {k:<20} = {v!r}")
    if p["alpha_start"] > 0:
        print(f"  expected alpha: ep0={_alpha_at(p, 0):.4f}  ep{EPOCHS - 1}={_alpha_at(p, EPOCHS - 1):.4f}")
    print(f"{'=' * 78}\n")

    _ALPHA_SEEN.clear()
    _WIRED["n"] = 0
    t0 = time.time()
    model = YOLO(MODEL_CFG)
    if MODEL_WEIGHTS:
        try:
            model.load(MODEL_WEIGHTS)
        except Exception as e:
            print(f"  [warn] weight transfer failed: {e}")
    model.add_callback("on_train_epoch_start", set_epoch)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(p))
    results = model.train(**kw)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    alphas = dict(sorted(_ALPHA_SEEN.items()))
    swa_on = p["alpha_start"] > 0
    out = {"name": name, "phase": rc["phase"], "hours": hours, "weights": weights,
           "seed": seed, "batch": BATCH, "hook_wired_to": _WIRED["n"],
           "alpha_first": next(iter(alphas.values()), None),
           "alpha_last": list(alphas.values())[-1] if alphas else None,
           "void": bool(swa_on and (len(set(alphas.values())) < 2 or _WIRED["n"] == 0)),
           "test_map50": float("nan"), "test_map5095": float("nan")}
    if out["void"]:
        print(f"\n  [VOID] {name}: SWA set but alpha never moved ({alphas}).")
    try:
        with open(os.path.join(save_dir, "phase2_params.json"), "w") as f:
            json.dump({**out, "params": p, "why": rc["why"], "label": rc["label"],
                       "alphas": alphas, "env": ENV, "anchor": ANCHOR,
                       "best_boost_used": BEST_BOOST}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
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
    print(f"\n{'=' * 88}\n  YOLO26 PHASE 2 — v6i @{IMG_SIZE}, b{BATCH}, seed {SEED}\n{'=' * 88}")
    print(f"{'run':<22}{'ph':>3}{'mAP50':>8}{'mAP50-95':>10}{'vs anchor':>11}{'alpha':>15}{'h':>6}")
    print('-' * 88)
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        a = (f"{r['alpha_first']:.3f}->{r['alpha_last']:.3f}"
             if r["alpha_first"] is not None else "-")
        print(f"{r['name']:<22}{r['phase']:>3}{r['test_map50'] * 100:>8.2f}"
              f"{r['test_map5095'] * 100:>10.2f}{(r['test_map5095'] - ANCHOR) * 100:>+11.2f}"
              f"{a:>15}{r['hours']:>6.1f}" + ("  VOID" if r.get("void") else ""))
    print('-' * 88)
    for k, (m, s, l) in KNOWN.items():
        print(f"{k + ' (known)':<22}{'':>3}{'':>8}{m * 100:>10.2f}{(m - ANCHOR) * 100:>+11.2f}"
              f"   small {s * 100:.2f}  large {l * 100:.2f}")
    print(f"\n  BOOST CURVE at alpha 0.6->0.3   (1.5 / 2.0 / 2.5; 2.0 = a06_03 55.59)")
    for r in sorted(ok, key=lambda x: x["name"]):
        if r["phase"] == "A":
            print(f"    {r['name']:<22}{r['test_map5095'] * 100:>8.2f}")
    print("    peak or monotone trend -> the dose axis is real; scatter -> noise, "
          "same as the alpha sweep")
    print(f"\n  COMBINATION READOUT — watch SMALL, not overall mAP:")
    print(f"    > 51.32  the super-additivity reproduced (beats a06_03)")
    print(f"    > {BEST_SMALL * 100:.2f}  best small-object config in the project, both families")
    print(f"    large near 54 -> the compounding pattern instead")

    seeds = [r for r in ok if r.get("seed") != SEED]
    if seeds:
        print(f"\n  NOISE FLOOR — the number every other result depends on:")
        for r in seeds:
            base = next((x for x in ok if x["name"] == r["name"].replace("_seed1", "")), None)
            ref = base["test_map5095"] if base else 0.5559   # a06_03 seed 0
            gap = abs(r["test_map5095"] - ref) * 100
            print(f"    {r['name']:<24}{r['test_map5095'] * 100:>7.2f}  vs seed0 "
                  f"{ref * 100:.2f}   |gap| = {gap:.2f} pp")
            print(f"      < 0.15 -> the leaderboard is real and reportable")
            print(f"      > 0.35 -> every result in this campaign is inside noise, "
                  f"including the 51.75 small record")
    else:
        print(f"\n  Noise on YOLO26 is still UNMEASURED; phase-1 implies a floor near 0.45 pp.")
    print(f"  Per-size: CocoEvalAllFolders_luggage.py on best.pt")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    if "phasea" in args:
        todo = [r for r in RUNS if r["phase"] == "A"]
    elif "phaseb" in args:
        todo = [r for r in RUNS if r["phase"] == "B"]
    elif "phasec" in args:
        todo = [r for r in RUNS if r["phase"] == "C"]
    elif args:
        todo = [r for r in RUNS if r["name"] in set(sys.argv[1:])]
    else:
        todo = list(RUNS)
    print(f"\n{'=' * 88}\n  YOLO26 PHASE 2 — {len(todo)} runs (~{1.6 * len(todo):.1f} GPU-h)")
    print(f"  {', '.join(r['name'] for r in todo)}")
    if len(todo) == len(RUNS):
        print("  NOTE: running A and B together uses BEST_BOOST as set now. Prefer")
        print("        `phaseA`, read it, set BEST_BOOST, then `phaseB`.")
    print(f"{'=' * 88}\n")
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
            res.append({"name": r["name"], "phase": r["phase"], "hours": float("nan"),
                        "error": str(e), "test_map50": float("nan"),
                        "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
