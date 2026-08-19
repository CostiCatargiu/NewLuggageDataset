#!/usr/bin/env python3
"""
YOLO26 SNT — the one mechanism derived from a measurement rather than a prediction.

=============================================================================
THE ANOMALY
=============================================================================
    LARGE-object mAP50-95, configs beating their OWN baseline

        YOLOv12    26 / 45
        YOLO26      0 / 52

Fifty-two YOLO26 configurations — SWA, LB-TAL, SNL1, SCB, SBB, every
architecture variant, every hyperparameter nudge — and not one improved large
objects. On YOLOv12 the same interventions improved large in over half the runs.
Nothing tried so far moved that column, and nothing predicted it either.

=============================================================================
THE PROPOSED CAUSE
=============================================================================
one2one uses topk2 = 1: exactly ONE anchor per GT is positive, every other is a
hard negative with target 0. How many WELL-FITTING anchors that discards scales
with object size:

        8 px bag        few anchors overlap at all   ->  few high-IoU negatives
      250 px trolley    hundreds overlap well        ->  hundreds of them

So the branch carrying ~90% of the loss by the last epoch — and producing every
prediction, since the head is NMS-free — is told that hundreds of nearly correct
boxes around a large object are background. The damage scales with size by
construction.

YOLOv12 has no topk2. One head, ten positives, and the runner-ups are POSITIVES
rather than hard negatives. That is the only structural difference which tracks
the 26/45 vs 0/52 split.

=============================================================================
THE MECHANISM
=============================================================================
A non-selected anchor whose best overlap with any GT exceeds snt_min_iou gets a
soft target in that GT's class channel instead of 0:

    target = snt_tau * IoU ** snt_gamma      (elementwise max with the existing
                                              target, so positives are untouched)

    tau=0.25 gamma=2.0 min_iou=0.5
      IoU   0.40  0.50  0.60  0.70  0.80  0.90  0.95
      tgt   0.00  0.06  0.09  0.12  0.16  0.20  0.23

ONE2ONE ONLY. one2many's runner-ups are already positives — there is nothing to
soften there, and installing it would be a different mechanism with no argument
behind it.

NOT A REWEIGHTING. SWA, SNL1 and SBB all multiply an existing term. SNT changes
what the TARGET IS, and makes it depend on a quantity the classification loss
currently ignores: the assigner computes `overlaps` and throws them away.

=============================================================================
WHY THIS ONE IS WORTH A NIGHT WHEN THE LAST EIGHT WERE NOT
=============================================================================
It is RETRODICTIVE. Every previous proposal in this project reasoned forward
from code to an expected gain, and eight of eight were falsified. This one was
derived backwards from an anomaly that is sitting in the results either way.

It has an INDEPENDENT CHECK THAT ALREADY PASSED. A mechanism of this kind was
tried on YOLOv12 (`snt`, 54.79 vs 54.77 baseline = +0.02, a clean null). Under
this account that is the CORRECT outcome: v12 has no topk2, therefore no
runner-up-as-hard-negative problem, therefore nothing to fix. A theory that
explains why the same idea did nothing elsewhere is worth more than one that
only explains where it should work.

It is FALSIFIABLE IN A SPECIFIC DIRECTION:

    LARGE up, small roughly flat      -> the account holds
    small up, LARGE flat              -> refuted, and cleanly
    both flat                         -> the hard-negative asymmetry is not the
                                         cause of 0/52, which is still worth
                                         knowing

Odds: ~30%, same as everything else. The difference is that the FAILURE case
converts "0/52 on large, unexplained" into "0/52 on large, and we showed it is
not the hard-negative asymmetry either" — a stronger sentence than silence.

=============================================================================
READ LARGE FIRST — AND MIND THE NOISE THERE
=============================================================================
    yolo26_custom-9   large 60.87   <- baseline, the ceiling nothing has beaten
    best custom large       60.66   <- y26_s10_bal, still below it
    y26_scb_b3        large 59.43
    control sd on large      2.11 pp across configs

Large is the noisiest column in this project, so a single run showing +1 means
little. What would be persuasive is BOTH tau points moving large in the same
direction while small stays put — a size-specific effect is exactly what the
mechanism predicts and what noise does not produce twice.

Stock yolo26s.pt, b82/640/seed 0 — matches every loss run in this project.
REQUIRES the patched tal.py (SNT) + loss.py + default.yaml.

Usage:
    python run_yolo26_snt_v6i.py            # both tau points, ~3.6 GPU-h
    python run_yolo26_snt_v6i.py y26_snt_t25
"""

import gc
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_snt_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24
BASE_LARGE = 60.87     # nothing in 52 configs has beaten this
BEST_LOSS = 55.66      # y26_scb_b3
BEST_CUSTOM_LARGE = 60.66  # y26_s10_bal — the closest anything has come
LARGE_SD = 2.11

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_snt_t25", "expect": {"tau": 0.25},
     "params": cfg(snt_tau=0.25, snt_gamma=2.0, snt_min_iou=0.5),
     "label": "SNT tau=0.25 gamma=2.0 min_iou=0.5 — conservative",
     "why": "The gentle point: a runner-up at IoU 0.9 gets target 0.20 instead of "
            "0. Chosen first because softening negatives is the direction that "
            "creates duplicate predictions if overdone, and the head has no NMS to "
            "clean them up — precision is the thing to watch. gamma=2.0 keeps the "
            "softening concentrated on anchors that genuinely nearly fit; min_iou "
            "0.5 leaves everything below half-overlap as a hard negative, so the "
            "background signal is untouched for the vast majority of anchors."},

    {"name": "y26_snt_t50", "expect": {"tau": 0.5},
     "params": cfg(snt_tau=0.5, snt_gamma=2.0, snt_min_iou=0.5),
     "label": "SNT tau=0.50 gamma=2.0 min_iou=0.5 — stronger",
     "why": "Doubles the soft target (IoU 0.9 -> 0.41). Two points make tau a "
            "direction rather than a single guess, and the pair is what makes the "
            "result readable: LARGE has sd 2.11 pp on this dataset, so one run "
            "moving it proves nothing, but BOTH points moving large in the same "
            "direction while small stays flat is a size-specific effect that noise "
            "does not produce twice. If t50 is worse than t25 on precision, the "
            "duplicate-prediction failure mode is real and tau belongs below 0.25."},
]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import E2ELoss
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")
    checks = {
        "TaskAlignedAssigner.snt_enabled": hasattr(TAL.TaskAlignedAssigner, "snt_enabled"),
        "TaskAlignedAssigner.snt_soft_targets": hasattr(TAL.TaskAlignedAssigner, "snt_soft_targets"),
        # NOTE: forward() is only an OOM-fallback wrapper that delegates to
        # _forward(); the assignment work — and the SNT call — lives in _forward.
        # Checking forward() gives a false negative.
        "assignment path calls snt_soft_targets": any(
            "snt_soft_targets" in inspect.getsource(getattr(TAL.TaskAlignedAssigner, m))
            for m in ("_forward", "forward")
            if hasattr(TAL.TaskAlignedAssigner, m)
        ),
        "E2ELoss reads snt_tau": "snt_tau" in inspect.getsource(E2ELoss.__init__),
    }
    for k, v in checks.items():
        print(f"  {k:<38}{v}")
    if not all(checks.values()):
        print()
        print("  [ABORT] the SNT patch is not installed.")
        print("  Copy ultralytics26/ultralytics/{utils/tal.py,utils/loss.py,cfg/default.yaml} then:")
        print("  python verify_patch_v6i.py --ref <round8_deploy/patch> --install --runtime")
        return False
    for r in todo:
        a = TAL.TaskAlignedAssigner(topk=7, topk2=1)
        a.snt_tau = r["params"]["snt_tau"]
        if not a.snt_enabled():
            print(f"  [ABORT] {r['name']}: snt_enabled() False at tau={a.snt_tau}")
            return False
        print(f"  {r['name']:<14} tau={a.snt_tau} gamma={r['params']['snt_gamma']} "
              f"min_iou={r['params']['snt_min_iou']}   OK")
    print()
    print(f"  MODEL {MODEL_WEIGHTS} (stock)  batch {BATCH}  imgsz {IMG_SIZE}  seed {SEED}")
    print(f"  baseline {BASELINE:.2f}   best loss config {BEST_LOSS:.2f}")
    print(f"  LARGE baseline {BASE_LARGE:.2f} — unbeaten in 52 configs "
          f"(closest: {BEST_CUSTOM_LARGE:.2f})")
    print(f"  LARGE sd {LARGE_SD:.2f} pp — one run moving it proves nothing.")
    print(f"  The signal to look for: BOTH tau points move large the same way")
    print(f"  while small stays flat. That is size-specific; noise is not.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print()
        print("  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_callbacks(model, rc):
    """Assert at epoch 1 that SNT is live on one2one and OFF on one2many."""
    state = {"verified": False}
    want = float(rc["expect"]["tau"])

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — SNT needs one2one")
        a1, a2 = o2m.assigner, o2o.assigner
        if not (hasattr(a2, "snt_enabled") and a2.snt_enabled()):
            raise RuntimeError(
                f"{rc['name']}: snt_tau={want} requested but snt_enabled() is False on "
                f"one2one (tau={getattr(a2, 'snt_tau', None)}). E2ELoss is not wiring "
                f"SNT — aborting rather than producing a number.")
        if abs(float(a2.snt_tau) - want) > 1e-6:
            raise RuntimeError(f"{rc['name']}: one2one snt_tau is {a2.snt_tau}, expected {want}")
        if hasattr(a1, "snt_enabled") and a1.snt_enabled():
            raise RuntimeError(
                f"{rc['name']}: SNT is live on ONE2MANY (tau={a1.snt_tau}). It must be "
                f"one2one only — one2many's runner-ups are already positives, so this "
                f"would be a different mechanism entirely.")
        if int(getattr(a2, "topk2", 0)) != 1:
            raise RuntimeError(
                f"{rc['name']}: one2one topk2 is {a2.topk2}, expected 1. SNT's whole "
                f"premise is the single-positive selection — without it there is no "
                f"runner-up problem to fix.")
        print(f"  [guard] SNT live on one2one ONLY | tau={a2.snt_tau} gamma={a2.snt_gamma} "
              f"min_iou={a2.snt_min_iou} topk2={a2.topk2}")
        print(f"  [guard] one2many untouched (snt_tau={getattr(a1, 'snt_tau', 0.0)})")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name = rc["name"]
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={SEED}")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran - cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "params": rc["params"], "expect": rc["expect"], "seed": SEED,
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "snt_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    print()
    print("=" * 84)
    print(f"  SNT — stock {MODEL_WEIGHTS}, b{BATCH}/{IMG_SIZE}, seed {SEED}")
    print("=" * 84)
    print(f"{'run':<16}{'tau':>6}{'mAP50':>9}{'mAP50-95':>10}{'vs base':>9}")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<16}{r['params']['snt_tau']:>6.2f}{r['test_map50'] * 100:>9.2f}"
              f"{v:>10.2f}{v - BASELINE:>+9.2f}")
    print("-" * 84)
    print(f"  {'baseline':<16}{'':>6}{'':>9}{BASELINE:>10.2f}")
    print(f"  {'y26_scb_b3':<16}{'':>6}{'':>9}{BEST_LOSS:>10.2f}   best loss config so far")
    print()
    print("  THE TEST IS THE LARGE COLUMN, NOT THIS TABLE.")
    print(f"  Run CocoEvalAllFolders_luggage.py on each best.pt and fill in:")
    print()
    print(f"    {'config':<16}{'large':>8}{'small':>8}")
    print(f"    {'baseline':<16}{BASE_LARGE:>8.2f}{51.00:>8.2f}   <- large unbeaten in 52 configs")
    for r in sorted(ok, key=lambda x: x["params"]["snt_tau"]):
        print(f"    {r['name']:<16}{'____':>8}{'____':>8}")
    print()
    print(f"    BOTH tau up on large, small flat   -> the account holds. Large has")
    print(f"                                          sd {LARGE_SD:.2f}, so the PAIR is the")
    print(f"                                          evidence, not either run alone.")
    print(f"    small up, large flat               -> refuted, cleanly")
    print(f"    both flat                          -> the hard-negative asymmetry is")
    print(f"                                          NOT the cause of 0/52 — still")
    print(f"                                          worth reporting")
    print()
    print("  Also check PRECISION. Softening negatives is how you get duplicate")
    print("  predictions, and this head has no NMS to remove them. If precision")
    print("  drops sharply at tau=0.50 but not 0.25, the useful range is below 0.25.")
    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<16} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print()
    print("=" * 84)
    print(f"  YOLO26 SNT — {len(todo)} runs, ~{1.8 * len(todo):.1f} GPU-h")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 84)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "params": r["params"], "expect": r["expect"],
                        "seed": SEED, "hours": float("nan"), "error": str(e),
                        "mechanism_verified": False,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
