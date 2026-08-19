#!/usr/bin/env python3
"""
YOLO26 SWA + LB-TAL SWEEP — every point has a YOLOv12 counterpart.

=============================================================================
THE DESIGN: RANK CORRELATION, NOT A HUNT FOR A BIGGER NUMBER
=============================================================================
All seven configs below were already measured on YOLOv12 / v6i. That is the
point. A sweep that only reports "which YOLO26 run won" cannot distinguish a
mechanism from noise at these effect sizes. Comparing the ORDER against v12
can: if the v12 ranking reproduces, the mechanism transfers and the winner is
believable; if the ranking scrambles or inverts, we learn something specific
about what YOLO26 changed.

WHAT IS ALREADY KNOWN (do NOT re-run these):

    YOLO26 @640, vs its own stock anchor          mAP    small    med   large
      yolo26_custom-9   stock ANCHOR             55.24   51.00  65.98   60.87
      y26_sqrt0703      SWA a0.7->0.3 px48 b2.0  55.27   51.18  65.48   56.70
      y26_lb_uniform    LB-TAL uniform           55.52   51.75  65.31   58.58

    YOLOv12 @640, vs its own stock anchor 54.77
      sqrt0703          +0.86      lb_uniform        +0.79   (seed1 +0.68)
      a09_03            +0.52      lb_coarse_244     +0.57
      a08_04            +0.50      lb_p3_3           +0.43
      a06_03            +0.48      lb_p4wide         +0.26
      a07_04            +0.43
      a09_04            +0.21

MECHANISM TRANSFER SO FAR:
      SWA sqrt0703    v12 +0.86  ->  YOLO26 +0.03   (3% retained)
      LB-TAL uniform  v12 +0.79  ->  YOLO26 +0.28   (35% retained)

=============================================================================
THE TWO HYPOTHESES THE SWA ARM TESTS
=============================================================================
SWA nets only +0.03 on YOLO26, but that is +0.18 on small MINUS a 4.17 pp
collapse on large (60.87 -> 56.70). Two readings:

  OVER-DOSE     the area weighting is too strong for a model whose assigner
                already inflates sub-stride GTs; a milder schedule keeps the
                small gain without wrecking large. Predicts a06_03 (lowest
                dose here) tops the group and large recovers monotonically as
                alpha falls.
  DOESN'T FIT   SWA's area weighting fights YOLO26's DFL-free L1 head at any
                dose. Predicts all five cluster at 55.2-55.3 with large flat
                near 57 and no structure across alpha.

READ `large`, NOT OVERALL mAP. It is the discriminating number. If a06_03
comes back with large still ~57, stop — the remaining alpha runs are dead.

CAVEAT ON THE GRID: every alpha here is >= 0.6, because these are the five
points that exist on v12. If the optimum sits below 0.6 the curve will still
be pointing downhill at the edge of the grid and the sweep will under-shoot.
_swa(0.7, 0.3, boost=1.0) is the cheapest low-dose point to add if so.

=============================================================================
THE LB-TAL ARM
=============================================================================
14 of 14 LB-TAL variants beat the v12 anchor (+0.05 to +0.83). No variant has
ever lost. The two here are the best measured `fixed` budgets WITHOUT an SWA
confound, and together with y26_lb_uniform (already run) they form a P3-budget
axis with known v12 values:

      P3 budget      4 (uniform)      3 (lb_p3_3)      2 (lb_coarse_244)
      v12 gain         +0.79            +0.43              +0.57

NOTE A STORY THAT THE DATA KILLED: coarse-heavy budgets do NOT protect large.
lb_coarse_244 gives P5 the largest share and has the WORST large of any LB-TAL
run (55.09 vs the 57.73 anchor), while lb_uniform is the only variant that
leaves large untouched (57.75). Any budget reasoning of the form "give P5 more
to help large" is contradicted by v12's own numbers. These two are included
because they are MEASURED, not because a mechanism story favours them.

=============================================================================
!! BATCH — THE ONE UNRESOLVED CONFOUND
=============================================================================
BATCH below is set to 82 to match the ported runs (y26_sqrt0703,
y26_lb_uniform). If yolo26_custom-9 was trained at a DIFFERENT batch, then
55.24 is not a valid reference and every delta this script prints is
confounded. Confirm it, then set BATCH and ANCHOR together. preflight() prints
a warning it cannot resolve for you.

REQUIRES the patched YOLO26 tree: BboxLoss.swa_weight,
LevelBalancedTaskAlignedAssigner, and the 15 custom keys in default.yaml.
Run verify_port.py first.

Usage:
    python verify_port.py
    python run_yolo26_sweep_v6i.py
    python run_yolo26_sweep_v6i.py y26_swa_a06_03 y26_lb_coarse244
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
PROJECT_DIR = "runs_yolo26_sweep_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                  # !! must match yolo26_custom-9 — see the docstring
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# YOLO26 stock anchor (yolo26_custom-9). Every delta is measured against this.
ANCHOR = 0.5524
ANCHOR_SMALL, ANCHOR_MED, ANCHOR_LARGE = 0.5100, 0.6598, 0.6087
# Already-run YOLO26 points, printed alongside for context.
KNOWN = {"y26_sqrt0703": 0.5527, "y26_lb_uniform": 0.5552}
# v12 gain over the v12 anchor, per config — the rank-correlation reference.
V12_GAIN = {
    "y26_swa_a06_03": +0.48, "y26_swa_a09_03": +0.52, "y26_swa_a08_04": +0.50,
    "y26_swa_a07_04": +0.43, "y26_swa_a09_04": +0.21,
    "y26_lb_coarse244": +0.57, "y26_lb_p3_3": +0.43,
    "y26_sqrt0703": +0.86, "y26_lb_uniform": +0.79,
}

# -----------------------------------------------------------------------------
# The 15 keys the YOLO26 port accepts. ANY key outside this set is rejected by
# train(): the other ~29 keys the v12 runners set (box_loss_type, use_nwd,
# cls_mode, use_satal, iou_clip_*, ...) do not exist in YOLO26's default.yaml.
# -----------------------------------------------------------------------------
_STOCK = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False, lbtal_mode="uniform", lbtal_level_topk=None,
    lbtal_min_level_k=1, lbtal_quality_gate=0.0,
    tal_alpha=0.5, tal_beta=6.0,
)
CUSTOM_KEYS = tuple(_STOCK)


def _swa(start, end, boost=2.0, px=48):
    """SWA schedule, matching the v12 _swa() helper exactly."""
    return dict(_STOCK, alpha_start=start, alpha_end=end,
                alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost)


def _lb(budget, mode="fixed"):
    """LB-TAL per-level budget, keyed by stride.

    NOTE the sums: coarse244 {8:2,16:4,32:4} = 10 = topk exactly, but p3_3
    {8:3,16:4,32:4} = 11 > topk. That is not a typo — it is the budget v12
    actually ran. LB-TAL caps the union at topk highest-metric-first, so p3_3
    hands out 11 slots and then drops the weakest one per GT. Same behaviour as
    on v12, so the comparison holds; the preflight warns about it rather than
    silently "fixing" a measured config.
    """
    return dict(_STOCK, use_lbtal=True, lbtal_mode=mode,
                lbtal_level_topk=budget, lbtal_min_level_k=1)


def _alpha_at(p, epoch):
    """Expected alpha, so the [SWA] log lines can be checked against it."""
    prog = epoch / max(EPOCHS, 1)
    a = p["alpha_start"] * (1 - prog) + p["alpha_end"] * prog
    return max(p["alpha_min"], min(p["alpha_max"], a))


# =============================================================================
# RUNS — ordered so that stopping after four still leaves a two-sided test on
# BOTH mechanisms: 1 and 2 are the candidates, 3 and 4 their controls.
# =============================================================================
RUNS = [
    {"name": "y26_swa_a06_03", "params": _swa(0.6, 0.3),
     "label": "SWA sqrt 0.6->0.3 px48 boost2.0  [v12 +0.48]",
     "why": "Lowest dose of the five, so under the OVER-DOSE reading this is "
            "the group's expected winner and the run that should recover large "
            "furthest above sqrt0703's 56.70. If large stays ~57 here, dose is "
            "not the variable and the other four alpha runs are dead."},

    {"name": "y26_lb_coarse244", "params": _lb({8: 2, 16: 4, 32: 4}),
     "label": "LB-TAL fixed {8:2,16:4,32:4}  [v12 +0.57]",
     "why": "Best measured `fixed` budget with no SWA confound. With uniform "
            "(P3=4, already run) and lb_p3_3 (P3=3) it gives a three-point "
            "P3-budget axis whose v12 values are all known. Included because it "
            "is measured — NOT because coarse-heavy protects large, which v12 "
            "disproves (its large is 55.09, the worst of all 14 LB-TAL runs)."},

    {"name": "y26_swa_a09_03", "params": _swa(0.9, 0.3),
     "label": "SWA sqrt 0.9->0.3 px48 boost2.0 — widest decay  [v12 +0.52]",
     "why": "The dose-INCREASE control that makes the SWA arm a two-sided test. "
            "Over-dose predicts this is the worst of the five and that large "
            "falls below sqrt0703's 56.70."},

    {"name": "y26_lb_p3_3", "params": _lb({8: 3, 16: 4, 32: 4}),
     "label": "LB-TAL fixed {8:3,16:4,32:4}  [v12 +0.43]",
     "why": "The middle point of the P3-budget axis (4 / 3 / 2). Differs from "
            "coarse244 only in the finest level, so the pair isolates the P3 "
            "budget with everything else held. Its budget sums to 11 > topk=10, "
            "exactly as on v12 — LB-TAL caps the union highest-metric-first, so "
            "the extra slot is handed out and then the weakest pick dropped."},

    {"name": "y26_swa_a08_04", "params": _swa(0.8, 0.4),
     "label": "SWA sqrt 0.8->0.4 px48 boost2.0  [v12 +0.50]",
     "why": "Fills the alpha curve. Shares alpha_end=0.4 with a07_04 and "
            "a09_04, so those three isolate alpha_start at fixed alpha_end."},

    {"name": "y26_swa_a07_04", "params": _swa(0.7, 0.4),
     "label": "SWA sqrt 0.7->0.4 px48 boost2.0 — shallower decay  [v12 +0.43]",
     "why": "Same start as sqrt0703 but a shallower decay, so it isolates "
            "alpha_end (0.3 vs 0.4) against a run already measured on YOLO26."},

    {"name": "y26_swa_a09_04", "params": _swa(0.9, 0.4),
     "label": "SWA sqrt 0.9->0.4 px48 boost2.0  [v12 +0.21, worst on v12]",
     "why": "v12's weakest SWA point. Last in the queue: if the ranking is "
            "reproducing at all, this one carries the least information. "
            "Consider swapping it for _swa(0.7, 0.3, boost=1.0) to buy a "
            "genuine low-dose point instead."},
]


# ============================================================== epoch plumbing
def iter_bbox_losses(criterion):
    """Yield every BboxLoss reachable from a criterion, E2E or not.

    E2ELoss holds .one2many and .one2one, each a full v8DetectionLoss with its
    own .bbox_loss. A plain v8DetectionLoss holds .bbox_loss directly. Missing a
    branch means alpha never advances there.
    """
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
        mod = getattr(holder, "module", None)  # DDP
        if mod is not None and getattr(mod, "criterion", None) is not None:
            out.append(mod.criterion)
    return out


_ALPHA_SEEN = {}
_WIRED = {"n": 0}


def set_epoch(trainer):
    """Push the epoch into every BboxLoss and log alpha.

    TIMING — do not turn the epoch-0 case back into a hard error. Ultralytics
    builds the criterion LAZILY inside BaseModel.loss() on the first forward, so
    at on_train_epoch_start for epoch 0 there is genuinely no BboxLoss yet. That
    is harmless: BboxLoss.__init__ already sets epoch=0 and reads total_epochs
    from hyp.epochs. From epoch 1 the criterion exists, and finding none THEN
    means the layout changed and the run is void.
    """
    epoch = int(getattr(trainer, "epoch", 0))
    found = []
    for crit in get_criteria(trainer):
        for bl in iter_bbox_losses(crit):
            bl.epoch = epoch
            bl.total_epochs = EPOCHS
            found.append(bl)

    if not found:
        if epoch == 0:
            print("  [callback] criterion not built yet at epoch 0 (lazy init) — "
                  "will wire at epoch 1; epoch-0 defaults are already correct")
            return
        raise RuntimeError(
            f"epoch callback reached NO BboxLoss at epoch {epoch}. The criterion "
            "layout changed: alpha would freeze and the run would be void.")

    bl = found[0]
    alpha = bl.get_dynamic_alpha()
    _ALPHA_SEEN.setdefault(epoch, round(alpha, 6))

    if _WIRED["n"] == 0:
        _WIRED["n"] = len(found)
        print(f"\n  {'=' * 66}")
        print(f"  LOSS WIRED — {len(found)} BboxLoss instance(s) at epoch {epoch}")
        print(f"  {'=' * 66}")
        print(f"    SWA enabled      : {bl.swa_enabled()}")
        print(f"    area_weight_mode : {bl.area_weight_mode}   norm: {bl.area_weight_norm}")
        print(f"    alpha            : {bl.alpha_start} -> {bl.alpha_end}  "
              f"clip [{bl.alpha_min}, {bl.alpha_max}]  over {bl.total_epochs} ep")
        print(f"    small_obj        : px < {bl.small_obj_px}   boost x{bl.small_obj_boost}")
        print(f"    dfl_loss         : {'present' if bl.dfl_loss else 'None (reg_max=1, L1 branch)'}")
        for crit in get_criteria(trainer):
            for br in ("one2many", "one2one"):
                asg = getattr(getattr(crit, br, None), "assigner", None)
                if asg is not None:
                    print(f"    assigner[{br:8}]: {type(asg).__name__}  "
                          f"topk={asg.topk} topk2={asg.topk2}")
        print(f"  {'=' * 66}\n")

    if bl.swa_enabled() and (epoch < 2 or epoch % 10 == 0 or epoch == EPOCHS - 1):
        print(f"  [SWA] epoch {epoch:>3}/{EPOCHS}  alpha={alpha:.4f}")


# ================================================================== preflight
def env_provenance():
    info = {"loss_md5": None, "tal_md5": None, "has_swa": False, "has_lbtal": False,
            "loss_path": None, "missing_keys": []}
    try:
        import ultralytics.utils.loss as _lm
        import ultralytics.utils.tal as _tm
        info["loss_path"] = getattr(_lm, "__file__", None)
        for mod, key in ((_lm, "loss_md5"), (_tm, "tal_md5")):
            p = getattr(mod, "__file__", None)
            if p and os.path.exists(p):
                info[key] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
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
    print(f"  loss.py  : {ENV.get('loss_path')}")
    print(f"  md5      : loss={ENV.get('loss_md5')}  tal={ENV.get('tal_md5')}")
    print(f"  swa_weight={ENV['has_swa']}   LB-TAL class={ENV['has_lbtal']}   "
          f"missing keys={ENV['missing_keys'] or 'none'}")

    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import ultralytics: {ENV['import_error']}")
        return False
    if not ENV["has_swa"]:
        print("\n  [ABORT] BboxLoss.swa_weight missing — the ultralytics on the "
              "import path is NOT the patched tree. Every run would come out stock.")
        return False
    if not ENV["has_lbtal"] and any(r["params"]["use_lbtal"] for r in todo):
        print("\n  [ABORT] LevelBalancedTaskAlignedAssigner missing but an LB-TAL run is queued.")
        return False
    if ENV["missing_keys"]:
        print(f"\n  [ABORT] keys absent from default.yaml — they would be dropped "
              f"silently, not applied: {ENV['missing_keys']}")
        return False

    for r in todo:
        p, n = r["params"], r["name"]
        for k in p:
            if k not in CUSTOM_KEYS:
                print(f"\n  [ABORT] {n}: '{k}' is not a ported key.")
                return False
        swa_on = max(p["alpha_start"], p["alpha_end"], p["alpha_max"]) > 0
        if p["area_weight_mode"] != "inv" and not swa_on:
            print(f"\n  [ABORT] {n}: area_weight_mode set but every alpha is 0 — no-op.")
            return False
        if p["small_obj_boost"] != 1.0 and not p["small_obj_px"]:
            print(f"\n  [ABORT] {n}: small_obj_boost set but small_obj_px=0 — boost never applies.")
            return False
        if p["use_lbtal"] and p["lbtal_mode"] == "fixed" and not p["lbtal_level_topk"]:
            print(f"\n  [ABORT] {n}: lbtal_mode='fixed' with no budget. This does NOT "
                  f"crash — it silently degrades to 'uniform'.")
            return False
        if p["lbtal_level_topk"] and not p["use_lbtal"]:
            print(f"\n  [ABORT] {n}: budget set but use_lbtal=False.")
            return False
        b = p["lbtal_level_topk"]
        if isinstance(b, dict) and sum(b.values()) != 10:
            print(f"  [note] {n}: budget sums to {sum(b.values())} vs topk=10 — "
                  f"LB-TAL caps the union highest-metric-first. Matches v12.")

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
        print("  Delete them, or ultralytics appends -2/-3 and summary.json points "
              "at the wrong folder.")
        return False

    print(f"\n  [!] BATCH={BATCH}. The anchor {ANCHOR * 100:.2f} (yolo26_custom-9) must "
          f"have been trained at this SAME batch,\n      otherwise every delta below is "
          f"confounded. This script cannot check that for you.")
    return True


# ======================================================================= train
def run_one(rc):
    name, p = rc["name"], rc["params"]
    active = {k: v for k, v in p.items() if v != _STOCK[k]}
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  model={MODEL_CFG}  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}")
    print(f"  non-stock keys ({len(active)}):")
    for k, v in sorted(active.items()):
        print(f"      {k:<20} = {v!r}")
    if p["alpha_start"] > 0:
        print(f"  expected alpha: epoch0={_alpha_at(p, 0):.4f}  "
              f"epoch{EPOCHS - 1}={_alpha_at(p, EPOCHS - 1):.4f}   "
              f"(check the [SWA] lines against these)")
    print(f"{'=' * 78}\n")

    _ALPHA_SEEN.clear()
    _WIRED["n"] = 0
    t0 = time.time()

    model = YOLO(MODEL_CFG)
    if MODEL_WEIGHTS:
        try:
            model.load(MODEL_WEIGHTS)
        except Exception as e:
            print(f"  [warn] weight transfer failed: {e} — training from scratch")
    model.add_callback("on_train_epoch_start", set_epoch)

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(p))
    results = model.train(**kw)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")

    alphas = dict(sorted(_ALPHA_SEEN.items()))
    swa_on = p["alpha_start"] > 0
    moved = len(set(alphas.values())) > 1
    out = {"name": name, "hours": hours, "weights": weights, "seed": SEED,
           "batch": BATCH, "hook_wired_to": _WIRED["n"],
           "alpha_first": next(iter(alphas.values()), None),
           "alpha_last": list(alphas.values())[-1] if alphas else None,
           "void": bool(swa_on and (not moved or _WIRED["n"] == 0)),
           "test_map50": float("nan"), "test_map5095": float("nan")}
    if out["void"]:
        print(f"\n  [VOID] {name}: SWA configured but alpha never moved ({alphas}). "
              f"The curriculum did not run — do not report this as a negative result.")

    try:
        with open(os.path.join(save_dir, "sweep_params.json"), "w") as f:
            json.dump({**out, "params": p, "why": rc["why"], "label": rc["label"],
                       "alphas": alphas, "env": ENV, "anchor": ANCHOR}, f, indent=2)
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
    print(f"\n{'=' * 92}\n  YOLO26 SWEEP — v6i @{IMG_SIZE}, b{BATCH}, seed {SEED}\n{'=' * 92}")
    print(f"{'run':<20}{'mAP50':>8}{'mAP50-95':>10}{'vs anchor':>11}"
          f"{'v12 gain':>10}{'alpha':>16}{'h':>6}")
    print('-' * 92)
    rows = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    for r in sorted(rows, key=lambda x: -x["test_map5095"]):
        g = V12_GAIN.get(r["name"])
        a = (f"{r['alpha_first']:.3f}->{r['alpha_last']:.3f}"
             if r["alpha_first"] is not None else "-")
        print(f"{r['name']:<20}{r['test_map50'] * 100:>8.2f}{r['test_map5095'] * 100:>10.2f}"
              f"{(r['test_map5095'] - ANCHOR) * 100:>+11.2f}"
              f"{g if g is None else f'{g:+.2f}':>10}{a:>16}{r['hours']:>6.1f}"
              + ("  VOID" if r.get("void") else ""))
    print('-' * 92)
    print(f"{'yolo26 ANCHOR':<20}{'':>8}{ANCHOR * 100:>10.2f}{0.0:>+11.2f}"
          f"{'--':>10}   (small {ANCHOR_SMALL * 100:.2f}  large {ANCHOR_LARGE * 100:.2f})")
    for k, v in KNOWN.items():
        print(f"{k + ' (known)':<20}{'':>8}{v * 100:>10.2f}{(v - ANCHOR) * 100:>+11.2f}"
              f"{V12_GAIN.get(k, 0):>+10.2f}")

    # rank correlation against v12 — the actual point of the sweep
    pairs = [(V12_GAIN[r["name"]], r["test_map5095"]) for r in rows if r["name"] in V12_GAIN]
    if len(pairs) >= 3:
        def rank(xs):
            s = sorted(range(len(xs)), key=lambda i: xs[i])
            out = [0] * len(xs)
            for pos, i in enumerate(s):
                out[i] = pos
            return out
        rv, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
        n = len(pairs)
        rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rv, ry)) / (n * (n * n - 1))
        print(f"\n  Spearman rho (v12 order vs YOLO26 order, n={n}): {rho:+.2f}")
        print("    ~+1  the v12 ranking reproduces -> the mechanism transfers")
        print("    ~ 0  the order scrambled -> at these effect sizes this is noise")
        print("    ~-1  the order inverted -> systematic, and the over-dose reading gains support")

    print("\n  READ `large` PER SIZE, not just overall — anchor 60.87, y26_sqrt0703 56.70.")
    print("  Seed noise: 0.12 pp on v12, UNMEASURED on YOLO26. Treat anything under")
    print("  ~0.25 pp as a tie until a seed-1 repeat exists.")
    print("  Per-size: CocoEvalAllFolders_luggage.py on best.pt")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 92}\n  YOLO26 SWA + LB-TAL SWEEP — {len(todo)} runs (~{1.6 * len(todo):.1f} GPU-h)")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print(f"  Stopping after 4 still leaves a two-sided test on both mechanisms.")
    print(f"{'=' * 92}\n")
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
            res.append({"name": r["name"], "hours": float("nan"), "error": str(e),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
