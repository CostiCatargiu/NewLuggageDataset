#!/usr/bin/env python3
"""
YOLO26 PORT — the three best YOLOv12 loss configs, transferred to YOLO26 on v6i.

=============================================================================
WHAT IS BEING PORTED, AND WHAT IT SCORED ON YOLOv12
=============================================================================
    config              mAP50-95   vs anchor   small    med    large
    sqrt0703 (SWA)        55.64      +0.87     50.63   65.61   59.99
    lb_uniform            55.57      +0.80     50.85   65.32   57.75
    lb_p4wide             55.03      +0.26     50.97   64.67   55.29
    cmb_p4wide (SWA+LB)   55.60      +0.83     51.15   64.78   55.61
    anchor (stock)        54.77        --      49.98   65.07   57.73

NONE OF THOSE NUMBERS PREDICT THE YOLO26 RESULT. Read the block below before
treating a delta here as a replication of anything.

=============================================================================
WHY YOLO26 IS A DIFFERENT OPTIMISATION PROBLEM
=============================================================================
1. reg_max = 1 -> THERE IS NO DFL. The third loss term is an L1 on
   stride-normalised ltrb (loss.py, the `else` branch of BboxLoss.forward). SWA's
   weight, which scaled CIoU+DFL on v12, now scales CIoU+L1. Section O (DFL
   entropy) cannot be ported at all — there is no distribution to sharpen.

2. end2end = True -> the criterion is E2ELoss, not v8DetectionLoss. It builds
   TWO complete detection losses, one2many (topk=10) and one2one (topk=7,
   topk2=1), and blends them with a gain that DECAYS 0.8 -> 0.1 across training.
   Both read the same model.args, so every key set here reaches both branches.

3. The assigner inflates any GT smaller than stride 8 up to 16 px before the
   in-box test (tal.py select_candidates_in_gts). YOLO26 ships its own
   small-object candidate fix. SWA and LB-TAL both compensate for small-object
   starvation, so the SAME redundancy that made SWA score -0.37 on the P2-head
   architecture may apply here. That is the single most likely reason these
   configs fail to replicate.

LB-TAL IS INSTALLED ON one2many ONLY. In the one2one branch topk2=1 makes
select_highest_overlaps re-select a single anchor per GT, which would erase any
per-level budget while still logging as active. See LevelBalancedTaskAlignedAssigner.

=============================================================================
THE SILENT-NO-OP TRAPS THIS SCRIPT GUARDS AGAINST
=============================================================================
  * The v12 epoch callback sets `crit.epoch`. Under E2ELoss the criterion has no
    .epoch and no .bbox_loss — only .one2many and .one2one. Copy it verbatim and
    alpha FREEZES at its epoch-0 value: the 0.7->0.3 curriculum never runs and
    the result reads as "SWA does not help". set_epoch() below walks both.
  * lbtal_level_topk is blank in default.yaml. A blank YAML key parses to None,
    so getattr's default never fires — the same class of bug that crashed every
    size_cond run at epoch 0. Here it would be worse than a crash: mode='fixed'
    with no budget silently degrades to 'uniform'. preflight() aborts on it.
  * A budget dict round-tripping through YAML arrives with STRING keys ('8'),
    while torch.unique(strides).tolist() yields floats (8.0). The assigner
    normalises both; verify_port.py proves it.

Run verify_port.py FIRST (no GPU, ~10 s). It asserts SWA changes the loss, that
alpha moves between epochs, and that LB-TAL changes the per-level positive
distribution. If any of those fail, nothing here is worth GPU time.

Usage:
    python verify_port.py
    python run_yolo26_port_v6i.py                 # anchor + 3 ports
    python run_yolo26_port_v6i.py y26_sqrt0703
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
MODEL_CFG = "yolo26s.yaml"       # topology; P3-P5, strides 8/16/32
MODEL_WEIGHTS = "yolo26s.pt"     # COCO transfer; set to None to train from scratch
PROJECT_DIR = "runs_yolo26_port_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# YOLOv12 references. Kept for context only — the YOLO26 anchor below is the
# only number any of these runs may legitimately be compared against.
V12_ANCHOR = 0.5477
V12_BEST = 0.5564

# Neutral base: every custom key at its stock-equivalent value, so each config
# below differs from the anchor in exactly one mechanism.
_STOCK = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False, lbtal_mode="uniform", lbtal_level_topk=None,
    lbtal_min_level_k=1, lbtal_quality_gate=0.0,
    tal_alpha=0.5, tal_beta=6.0,
)

# Section A/A2 exactly as recorded for yolov12s_sqrt0703.
_SWA = dict(
    area_weight_mode="sqrt", area_weight_norm="max",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)

RUNS = [
    {"name": "y26_anchor", "params": dict(_STOCK),
     "label": "stock YOLO26 — THE ANCHOR",
     "why": "Every v6i number was measured on YOLOv12 with a DFL head and a "
            "single assignment branch. None of them is a valid reference for "
            "YOLO26. Run this first: without it no delta below means anything."},

    {"name": "y26_sqrt0703", "params": dict(_STOCK, **_SWA),
     "label": "SWA sqrt/0.7->0.3 (v12 champion, +0.87)",
     "why": "The only one of the three that ports with no signature surgery and "
            "no assigner change — it lives entirely in BboxLoss.swa_weight. Also "
            "the best of the three on v12. Note its weight now scales CIoU+L1 "
            "rather than CIoU+DFL, because YOLO26 has no DFL."},

    {"name": "y26_lb_uniform", "params": dict(_STOCK, use_lbtal=True, lbtal_mode="uniform"),
     "label": "LB-TAL uniform per-level top-k (v12 +0.80)",
     "why": "The more defensible of the two LB-TAL ports: 'uniform' derives its "
            "budget from the level count alone, so unlike p4wide it carries no "
            "constants fitted to a YOLOv12 candidate-supply footprint. Applies to "
            "one2many only."},

    {"name": "y26_lb_p4wide", "params": dict(_STOCK, use_lbtal=True, lbtal_mode="fixed",
                                             lbtal_level_topk={8: 4, 16: 7, 32: 1}),
     "label": "LB-TAL fixed {8:4,16:7,32:1} (v12 +0.26)",
     "why": "The weakest of the three on v12 (+0.26, and large fell 2.44 below "
            "the anchor). Its budget was fitted to a measured 3-level YOLOv12 "
            "footprint; YOLO26's assigner inflates sub-stride GTs, so the pools "
            "those twelve positives describe no longer exist. Included as the "
            "attribution control for lb_uniform, not as a candidate."},

    # SWA + LB-TAL stacked. Left out of the default set on purpose: on v12 the
    # combination (cmb_p4wide 55.60) did NOT beat SWA alone (55.64), and stacking
    # two small-object mechanisms is what produced -0.84 on the P2 architecture.
    # Run it only if BOTH single axes show signal.
    # {"name": "y26_cmb_p4wide",
    #  "params": dict(_STOCK, use_lbtal=True, lbtal_mode="fixed",
    #                 lbtal_level_topk={8: 4, 16: 7, 32: 1}, **_SWA),
    #  "label": "SWA + LB-TAL p4wide", "why": "stacked; attribution impossible if it wins"},
]

CUSTOM_KEYS = tuple(_STOCK.keys())


# ============================================================== epoch plumbing
def iter_bbox_losses(criterion):
    """Yield every BboxLoss reachable from a criterion, E2E or not.

    E2ELoss / E2EDetectLoss hold .one2many and .one2one, each a full
    v8DetectionLoss with its own .bbox_loss. A plain v8DetectionLoss holds
    .bbox_loss directly. Missing a branch means alpha never advances there.
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
    """Return every criterion object the trainer might be using."""
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
    """Push the current epoch into every BboxLoss, then record alpha.

    TIMING — do not "simplify" this back into a hard assert at epoch 0.
    Ultralytics builds the criterion LAZILY, inside BaseModel.loss() on the first
    forward pass:

        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

    so at on_train_epoch_start for epoch 0 there is genuinely no BboxLoss yet and
    finding none is expected, not a failure. It is harmless: BboxLoss.__init__
    already sets epoch=0 and reads total_epochs from hyp.epochs, so epoch 0 is
    correct without the hook. From epoch 1 onward the criterion exists, and
    finding none THEN means the layout really did change — that is the case worth
    aborting on, because alpha would freeze and the run would be void.

    The recorded alphas are checked again after training: a run that configured
    SWA but whose alpha never moved is reported VOID rather than negative.
    """
    epoch = int(getattr(trainer, "epoch", 0))
    found = []
    for crit in get_criteria(trainer):
        for bl in iter_bbox_losses(crit):
            bl.epoch = epoch
            bl.total_epochs = EPOCHS
            found.append(bl)
    n = len(found)

    if n == 0:
        if epoch == 0:
            print("  [callback] criterion not built yet at epoch 0 (lazy init) — "
                  "will wire at epoch 1; epoch-0 defaults are already correct")
            return
        raise RuntimeError(
            f"epoch callback reached NO BboxLoss at epoch {epoch}, after the criterion "
            "should exist. The criterion layout changed: alpha would freeze and the "
            "run would be void. Check iter_bbox_losses() against E2ELoss."
        )

    bl = found[0]
    alpha = bl.get_dynamic_alpha()
    _ALPHA_SEEN.setdefault(epoch, round(alpha, 6))

    # First contact: dump what the loss OBJECT holds. The engine/trainer line
    # only proves the keys reached the config; this proves they reached the loss.
    if _WIRED["n"] == 0:
        _WIRED["n"] = n
        print(f"\n  {'=' * 66}")
        print(f"  SWA WIRED — {n} BboxLoss instance(s) reached at epoch {epoch}")
        print(f"  {'=' * 66}")
        print(f"    enabled          : {bl.swa_enabled()}")
        print(f"    area_weight_mode : {bl.area_weight_mode}   norm: {bl.area_weight_norm}")
        print(f"    alpha            : {bl.alpha_start} -> {bl.alpha_end} "
              f"clipped to [{bl.alpha_min}, {bl.alpha_max}] over {bl.total_epochs} epochs")
        print(f"    small_obj        : px < {bl.small_obj_px}  boost x{bl.small_obj_boost}")
        print(f"    dfl_loss         : {'present' if bl.dfl_loss else 'None (reg_max=1, L1 branch)'}")
        for crit in get_criteria(trainer):
            for br in ("one2many", "one2one"):
                sub = getattr(crit, br, None)
                asg = getattr(sub, "assigner", None)
                if asg is not None:
                    print(f"    assigner[{br:9}]: {type(asg).__name__} "
                          f"topk={asg.topk} topk2={asg.topk2}")
            asg = getattr(crit, "assigner", None)
            if asg is not None and not hasattr(crit, "one2many"):
                print(f"    assigner         : {type(asg).__name__} "
                      f"topk={asg.topk} topk2={getattr(asg, 'topk2', '-')}")
        if not bl.swa_enabled():
            print("    NOTE: SWA is INERT for this config (all alphas 0) — expected "
                  "for the anchor and the LB-TAL-only runs.")
        print(f"  {'=' * 66}\n")

    # Periodic proof that the curriculum is moving.
    if bl.swa_enabled() and (epoch < 2 or epoch % 5 == 0 or epoch == EPOCHS - 1):
        first = _ALPHA_SEEN.get(min(_ALPHA_SEEN)) if _ALPHA_SEEN else alpha
        print(f"  [SWA] epoch {epoch:>3}/{EPOCHS}  alpha={alpha:.4f}  (started {first:.4f})")


# ================================================================== preflight
def env_provenance():
    info = {"loss_md5": None, "tal_md5": None, "has_swa": False, "has_lbtal": False,
            "cfg_keys_present": [], "cfg_keys_missing": []}
    try:
        import ultralytics.utils.loss as _lm
        import ultralytics.utils.tal as _tm
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
        for k in CUSTOM_KEYS:
            (info["cfg_keys_present"] if k in DEFAULT_CFG_DICT else info["cfg_keys_missing"]).append(k)
    except Exception as e:
        info["cfg_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  loss.py md5={ENV.get('loss_md5')}   tal.py md5={ENV.get('tal_md5')}")
    print(f"  BboxLoss.swa_weight present: {ENV['has_swa']}")
    print(f"  LevelBalancedTaskAlignedAssigner present: {ENV['has_lbtal']}")
    print(f"  custom keys in DEFAULT_CFG_DICT: {len(ENV['cfg_keys_present'])}/{len(CUSTOM_KEYS)}")

    if ENV.get("import_error"):
        print(f"\n  [ABORT] cannot import the loss modules: {ENV['import_error']}")
        return False
    if not ENV["has_swa"]:
        print("\n  [ABORT] BboxLoss.swa_weight missing — the installed ultralytics is not the patched tree.")
        return False
    if not ENV["has_lbtal"] and any(r["params"].get("use_lbtal") for r in todo):
        print("\n  [ABORT] LevelBalancedTaskAlignedAssigner missing but an LB-TAL run was selected.")
        return False
    if ENV["cfg_keys_missing"]:
        print(f"\n  [ABORT] keys absent from default.yaml (they would be dropped "
              f"silently, not applied): {ENV['cfg_keys_missing']}")
        return False

    # Per-config no-op checks.
    for r in todo:
        p = r["params"]
        swa_on = max(p.get("alpha_start", 0.0), p.get("alpha_end", 0.0), p.get("alpha_max", 0.0)) > 0
        if p.get("area_weight_mode", "inv") != "inv" and not swa_on:
            print(f"\n  [ABORT] {r['name']}: area_weight_mode is set but every alpha is 0 — "
                  f"the area term would be multiplied by 0 and the run would be a no-op.")
            return False
        if p.get("small_obj_boost", 1.0) != 1.0 and not p.get("small_obj_px"):
            print(f"\n  [ABORT] {r['name']}: small_obj_boost set but small_obj_px is 0 — boost never applies.")
            return False
        if p.get("use_lbtal") and p.get("lbtal_mode") == "fixed" and not p.get("lbtal_level_topk"):
            print(f"\n  [ABORT] {r['name']}: lbtal_mode='fixed' with no lbtal_level_topk. "
                  f"This does NOT crash — it silently degrades to 'uniform' and would be "
                  f"recorded as a p4wide result. Set the budget explicitly.")
            return False
        if p.get("lbtal_level_topk") and not p.get("use_lbtal"):
            print(f"\n  [ABORT] {r['name']}: a budget is set but use_lbtal is False.")
            return False

    # Resolve where ultralytics will ACTUALLY write. PROJECT_DIR is relative, and
    # ultralytics resolves a relative project under SETTINGS['runs_dir']/<task>/,
    # not under the cwd. Checking the cwd path silently misses every collision —
    # which is how a rerun ended up in y26_sqrt0703-4 instead of aborting.
    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    print(f"  run dirs checked: {bases}")
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] these run directories already exist:")
        for c in clash:
            print(f"      {c}")
        print("  Delete them, or ultralytics will silently append -2/-3/-4 and the "
              "summary will point at the wrong folder.")
        return False
    return True


# ======================================================================= train
def run_one(rc):
    name = rc["name"]
    # Only the keys that DIFFER from stock, so the mechanism under test is
    # readable at a glance instead of buried in the 100-key engine/trainer line.
    active = {k: v for k, v in rc["params"].items() if v != _STOCK[k]}
    print(f"\n{'=' * 78}\n  RUN {name}\n  {rc['label']}\n{'=' * 78}")
    print(f"  imgsz={IMG_SIZE}  batch={BATCH}  epochs={EPOCHS}  seed={SEED}  model={MODEL_CFG}")
    print(f"  non-stock keys ({len(active)}):")
    for k, v in sorted(active.items()):
        print(f"      {k:<20} = {v!r}")
    if not active:
        print("      (none — this is the stock anchor)")
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
    kw.update(copy.deepcopy(rc["params"]))
    results = model.train(**kw)

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")

    alphas = dict(sorted(_ALPHA_SEEN.items()))
    swa_on = max(rc["params"].get("alpha_start", 0.0), rc["params"].get("alpha_end", 0.0)) > 0
    alpha_moved = len(set(alphas.values())) > 1
    if swa_on and _WIRED["n"] == 0:
        print(f"\n  [VOID] {name}: the epoch hook never reached a BboxLoss during the "
              f"whole run. SWA stayed at its epoch-0 alpha throughout.")
    out = {"name": name, "hours": hours, "weights": weights, "seed": SEED,
           "alpha_first": next(iter(alphas.values()), None),
           "alpha_last": list(alphas.values())[-1] if alphas else None,
           "alpha_moved": alpha_moved,
           "hook_wired_to": _WIRED["n"],
           "void": bool(swa_on and (not alpha_moved or _WIRED["n"] == 0)),
           "test_map50": float("nan"), "test_map5095": float("nan")}
    if out["void"]:
        print(f"\n  [VOID] {name} configured SWA but alpha never changed across epochs "
              f"({alphas}). The curriculum did not run — do not report this as a negative result.")

    try:
        with open(os.path.join(save_dir, "port_params.json"), "w") as f:
            json.dump({**out, "params": rc["params"], "why": rc["why"],
                       "alphas": alphas, "env": ENV}, f, indent=2)
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
    anchor = next((r["test_map5095"] for r in res if r["name"] == "y26_anchor"), float("nan"))
    print(f"\n{'=' * 84}\n  YOLO26 PORT — v6i @{IMG_SIZE}, b{BATCH}, seed {SEED}\n{'=' * 84}")
    print(f"{'run':<20}{'mAP50':>9}{'mAP50-95':>11}{'vs anchor':>12}{'alpha':>16}{'h':>6}")
    print("-" * 84)
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        vs = "%+12.2f" % ((r["test_map5095"] - anchor) * 100) if anchor == anchor else f"{'n/a':>12}"
        a = (f"{r['alpha_first']}->{r['alpha_last']}" if r["alpha_first"] is not None else "-")
        flag = "  VOID" if r.get("void") else ""
        print(f"{r['name']:<20}{r['test_map50'] * 100:>9.2f}{r['test_map5095'] * 100:>11.2f}"
              f"{vs}{a:>16}{r['hours']:>6.1f}{flag}")
    print(f"\n  YOLO26 anchor: {anchor * 100:.2f}" if anchor == anchor else "\n  anchor not run")
    print(f"  YOLOv12 for context only: anchor {V12_ANCHOR * 100:.2f}, best {V12_BEST * 100:.2f}.")
    print("  Compare against the YOLO26 anchor ONLY. Seed noise on v12 was 0.12 pp;")
    print("  it has not been measured here, so treat anything under ~0.2 pp as a tie")
    print("  until a seed-1 run exists.")
    print("  Per-size: CocoEvalAllFolders_luggage.py on best.pt.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 84}\n  YOLO26 PORT — {len(todo)} runs (~{1.6 * len(todo):.1f} GPU-h)")
    print(f"  {', '.join(r['name'] for r in todo)}\n{'=' * 84}\n")
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
