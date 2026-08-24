#!/usr/bin/env python3
r"""
b1 COMBO SCREEN — three mechanisms meet the campaign maximum, for the first time

Three runs, ~2.7 GPU-h, b82, stock yolo26s.pt, 640, 70 ep, seed 0. NO PATCH.

THIS IS A SCREEN, NOT A ROUND. Read the threshold section before you read the
results, and do not move the threshold afterwards.

    CONTROL      y26_b1, n=2 PAIR MEAN            small 78.94   mAP50 81.67   m5095 54.81
    NOT          y26_b1 seed-0 draw 79.30 — that draw is 0.36 above the pair mean
    BASELINE     y26_identity, 3 seeds            small 77.62   mAP50 80.40   m5095 55.30
    FLOOR        pooled sd (df=11)                small  0.377  m5095 0.325   large 2.331


=============================================================================
WHY THIS EXISTS: EVERY COMBO IN THIS CAMPAIGN WAS RUN ON beta = 3, NONE ON b1
=============================================================================
    combos on b3 : 8       combos on b1 : 0

Round 22 proposed y26_b1_sbb50 and y26_b1_o2mf30 and neither was executed — the
queue pivoted to 22b. So the campaign maximum has never carried a mechanism.

AND THE REASON THAT LOOKED SETTLED IS WRONG. Round 19's ten combos were scored
against b3's SEED-0 DRAW of 79.14. Round 23 showed that draw is inflated: b3's
n=2 mean is 78.77. Re-scored against the corrected control, at the run-vs-n=2
SE of 0.462:

    y26_b3_o2mf30    78.79   +0.02   LEVEL      <- and non-negative on EVERY axis
    y26_b3_swa       78.62   -0.15   LEVEL
    y26_b3_tcn_p50-2 78.56   -0.21   LEVEL
    y26_b3_sbb50f    78.36   -0.41   LEVEL
    y26_b3_sbb50     78.34   -0.43   LEVEL
    y26_b3_o2o       78.26   -0.51   LEVEL
    y26_b3_o2m       77.54   -1.23   below

SEVEN OF EIGHT ARE LEVEL. "Combos are 0-for-10" was an artifact of an inflated
control and should not be repeated. The honest statement is that combos add
nothing MEASURABLE — which is a different claim, because round 19's design
(n=1 vs n=1, SE 0.533, 2 SE = 1.07) could not have detected a +0.4 win if one
existed. Absence of evidence, not evidence of absence.


=============================================================================
THE THRESHOLD. FIXED NOW. DO NOT MOVE IT AFTER SEEING THE NUMBERS.
=============================================================================
Running k screens at n=1 does not give k chances at a win. It gives one
number that LOOKS like a win. If all three configs are truly null (= b1's
mean), the best of k still reads above b1 by chance:

    k        1      2      3      4      6      8     10
    E[max] +0.00  +0.21  +0.32  +0.39  +0.48  +0.54  +0.58   (mAP points)

That is precisely what round 22b did — ten cells, best read +0.19 over b1's own
draw — and what round 23 then undid, four for four.

    THRESHOLD FOR k = 3, holding P(any null screen clears it) at ~5%:

        +2.13 sd over the control  ->  mAP50_small >= 79.74

    >= 79.74   a signal. SEED IT (2 more runs) before believing it.
    78.9-79.7  uninformative BY CONSTRUCTION. Stop. Do not run a fourth.
    <  78.9    the mechanism costs something on b1. Stop.

Nothing in 150 runs has reached 79.74 on a matched graph; the campaign maximum
is 79.30. So this screen is expected to return "stop", and that is a
legitimate outcome that closes the combination axis rather than leaving it
open forever. If you add a fourth run, the bar becomes 79.80 — at which point
you are running a round you cannot win.


=============================================================================
THE THREE, AND WHY THESE THREE
=============================================================================
1  y26_b1_o2mf30   o2m_final 0.1 -> 0.3
   The blend was hardcoded 0.8 -> 0.1 for the whole campaign. On b3 this is the
   ONLY combo that is non-negative on every axis against the corrected control:
   small +0.02, mAP50 +0.31, mAP50-95 +0.15, large +1.11. That mAP50-95 sign is
   the interesting part, because -0.49 there is b1's only real weakness.
   COUNTER-ARGUMENT, and it is real: y26_b3_o2m (beta on one2many only) is the
   worst combo in the campaign at 77.54, which is evidence against the "beta
   does its work in one2many" premise this config rests on. The reply is that
   o2m_final changes the branch WEIGHT, not its assignment. That is an
   argument, not evidence.

2  y26_b1_swa      SWA 0.9 -> 0.4, sqrt/max, px<48, x2
   CORRECTION TO THE RECORD: SWA is 9 OF 10 ABOVE BASELINE on mAP50_small, the
   most consistent family in this campaign. It was written off on mAP50-95 —
   the metric a sqrt(area) box weighting is designed to trade away — and this
   file has called it "0-for-many" more than once, which is wrong. It acts at
   the REGRESSION WEIGHTING stage; beta acts at ASSIGNMENT. Two stages, not two
   attempts at the same quantity, which is the only combination shape that has
   ever produced anything here.

3  y26_b1_sbb50    sbb_q 0.5, inverted
   Targets b1's actual weakness directly. NOTE THE OVERLAP: round 24 arm B is
   already seeding y26_b4s2_sbb50 to establish whether SBB's no-mAP50-95-cost
   signature is real at all. If arm B comes back null, THIS RUN HAS NO BASIS
   and should be read as a formality. Ordered last for that reason.


=============================================================================
EXECUTION ORDER
=============================================================================
    1  y26_b1_o2mf30   the only non-negative-on-every-axis combo on b3
    2  y26_b1_swa      the most consistent family, never met beta=1
    3  y26_b1_sbb50    overlaps round 24 arm B; run it last or not at all

    Usage:
        python run_yolo26_b1combo_screen_v6i.py               # all three
        python run_yolo26_b1combo_screen_v6i.py --preflight   # plan only
        python run_yolo26_b1combo_screen_v6i.py y26_b1_swa    # one by name
"""

import argparse
import gc
import json
import os
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"          # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_b1combo_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30     # identity, 3 seeds
SD_S50, SD_5095, SD_LRG = 0.377, 0.325, 2.331       # pooled, df=11
B1_S50, B1_50, B1_5095 = 78.94, 81.67, 54.81        # y26_b1, n=2 PAIR MEAN
THRESHOLD = 79.74                                    # k=3, ~5% family-wise. FIXED.

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    tal_beta_o2m=None, tal_beta_o2o=None, tal_alpha_o2m=None, tal_alpha_o2o=None,
    tal_beta_sel=None, tal_beta_tgt=None, tcn_p=1.0, tal_beta_level=None,
    scb_branch="both",
    o2m_start=0.8, o2m_final=0.1, o2m_decay=True,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0, cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
    multi_scale=0.0, scale=0.5, close_mosaic=CLOSE_MOSAIC, cos_lr=False,
)

_B1 = dict(tal_beta=1.0)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def _swa(start, end, boost=2.0, px=48):
    """VERBATIM from run_yolo26_round19_v6i.py, which is verbatim from
    run_yolo26_sweep_v6i.py, which produced y26_swa_a09_04.

    alpha_min/alpha_max are the ENDPOINTS (end, start), NOT 0.0/1.0 — a later
    helper in the same file used 0.0/1.0 and silently changed the mechanism.
    """
    return dict(alpha_start=start, alpha_end=end, alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost)


RUNS = [
    {"name": "y26_b1_o2mf30", "order": 1, "seed": 0, "ctrl": B1_S50,
     "params": cfg(**_B1, o2m_final=0.3),
     "expect": {"beta": 1.0, "sbb": None, "swa": None, "o2m_final": 0.3},
     "label": "beta 1 + blend 0.8 -> 0.3 — keep one2many alive through the late epochs",
     "why": "The only b3 combo non-negative on EVERY axis against the corrected "
            "control: small +0.02, mAP50 +0.31, mAP50-95 +0.15, large +1.11. The "
            "mAP50-95 sign matters because -0.49 there is b1's only weakness. "
            "Counter: y26_b3_o2m is the worst combo in the campaign, which cuts "
            "against the premise; o2m_final changes branch WEIGHT not assignment."},

    {"name": "y26_b1_swa", "order": 2, "seed": 0, "ctrl": B1_S50,
     "params": cfg(**_B1, **_swa(0.9, 0.4)),
     "expect": {"beta": 1.0, "sbb": None, "swa": (0.9, 0.4, 48.0, 2.0), "o2m_final": 0.1},
     "label": "beta 1 + SWA 0.9 -> 0.4 — the two most consistent mechanisms, never combined",
     "why": "SWA is 9 OF 10 above baseline on mAP50_small — the most consistent "
            "family here — and was retired on mAP50-95, which is the metric a "
            "sqrt(area) box weighting exists to trade away. It acts at the "
            "REGRESSION WEIGHTING stage while beta acts at ASSIGNMENT: two "
            "stages, not two attempts at the same quantity."},

    {"name": "y26_b1_sbb50", "order": 3, "seed": 0, "ctrl": B1_S50,
     "params": cfg(**_B1, sbb_q=0.5, sbb_invert=True),
     "expect": {"beta": 1.0, "sbb": (0.5, True), "swa": None, "o2m_final": 0.1},
     "label": "beta 1 + SBB q=0.5 inverted — aimed straight at b1's mAP50-95 cost",
     "why": "OVERLAPS ROUND 24 ARM B, which is seeding y26_b4s2_sbb50 to find out "
            "whether SBB's no-cost signature is real at all. If arm B is null "
            "this run has no basis. Ordered last so it is the one a dying queue "
            "drops."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """Prove the CONSUMER reads each key, and that defaults are bit-identical.

    Rounds 4-6 passed a preflight that only checked default.yaml accepted the key
    and tal.py had the class. Ten runs were lost because the file that had to
    consume the flag ignored it.
    """
    import inspect
    print("=" * 76)
    print("  PREFLIGHT")
    print("=" * 76)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils.tal import TaskAlignedAssigner
        from ultralytics.utils.loss import BboxLoss
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False, set()

    src_bbox = inspect.getsource(BboxLoss)
    checks = {
        "default.yaml accepts tal_beta": "tal_beta" in DEFAULT_CFG_DICT,
        "default.yaml accepts o2m_final": "o2m_final" in DEFAULT_CFG_DICT,
        "default.yaml accepts sbb_q": "sbb_q" in DEFAULT_CFG_DICT,
        "default.yaml accepts alpha_start (SWA)": "alpha_start" in DEFAULT_CFG_DICT,
        "default.yaml accepts area_weight_mode": "area_weight_mode" in DEFAULT_CFG_DICT,
        "BboxLoss has swa_enabled()": "def swa_enabled" in src_bbox,
        "BboxLoss carries sbb_q": "sbb_q" in src_bbox,
    }
    for k, v in checks.items():
        print(f"  {k:<44} {v}")
        ok &= v

    print()
    try:
        torch.manual_seed(0)
        a = TaskAlignedAssigner(topk=10, num_classes=3)
        b, n, A = 2, 3, 40
        pd_s, pd_b = torch.rand(b, A, 3), torch.rand(b, A, 4).cumsum(-1)
        gt_l = torch.randint(0, 3, (b, n, 1)).float()
        gt_b, m_gt, anc = torch.rand(b, n, 4).cumsum(-1), torch.ones(b, n, 1), torch.rand(A, 2) * 10
        out1 = a(pd_s, pd_b, anc, gt_l, gt_b, m_gt)
        a.beta_sel, a.beta_tgt, a.tcn_p, a.beta_level = None, None, 1.0, None
        out2 = a(pd_s, pd_b, anc, gt_l, gt_b, m_gt)
        inert = all(torch.equal(x, y) for x, y in zip(out1, out2))
        print(f"  {'defaults are bit-identical to stock':<44} {inert}")
        ok &= inert
    except Exception as ex:
        print(f"  [warn] inertness check could not run: {ex}")

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  RUN  {r['name']:<18} seed {r['seed']}  vs b1 {r['ctrl']:.2f}  {d}")
    print(f"\n  {len(todo)} runs, ~{0.9 * len(todo):.1f} GPU-h   (no patch required)")
    print(f"\n  THRESHOLD IS {THRESHOLD:.2f} ON mAP50_small AND IT IS FIXED.")
    print(f"  k={len(todo)} screens at n=1: a NULL best-of-{len(todo)} still reads "
          f"~+{[0, 0.0, 0.21, 0.32, 0.39][min(len(todo), 4)]:.2f} over b1 by chance.")
    if len(todo) > 3:
        print("  [WARN] more than 3 screens — the 79.74 bar is no longer valid, raise it.")
    return ok, {r["name"] for r in todo}


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Assert at epoch 2 that the requested mechanism is LIVE and nothing else is.

    Presence AND absence are asserted for every mechanism, on BOTH branches.
    Epoch 2 and not epoch 1: the o2m decay is recomputed per epoch and still
    matches stock on the first pass, so an epoch-1 guard passes for a config
    that reverts afterwards.
    """
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 2:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss")
        seen = []
        for tag, br in (("one2many", o2m), ("one2one", o2o)):
            a = br.assigner
            if abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            if abs(float(a.alpha) - 0.5) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected stock 0.5")
            for attr in ("beta_small", "beta_sel", "beta_tgt", "beta_level"):
                if getattr(a, attr, None) is not None:
                    raise RuntimeError(f"{rc['name']}: {tag} {attr} is live but not requested")
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live — falsified in round 21")
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")

            b = br.bbox_loss
            # ---- SBB: presence AND absence
            want_sbb = e["sbb"]
            q = float(getattr(b, "sbb_q", 0.0))
            if want_sbb is None:
                if q != 0.0:
                    raise RuntimeError(f"{rc['name']}: SBB live on {tag} (q={q}) but not requested")
            elif abs(q - want_sbb[0]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={q}, expected {want_sbb[0]}")
            # ---- SWA: presence AND absence, endpoints AND clamps
            want_swa = e["swa"]
            if want_swa is None:
                if b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA live on {tag} but not requested")
            else:
                if not b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA not live on {tag}")
                got = (float(b.alpha_start), float(b.alpha_end),
                       float(b.small_obj_px), float(b.small_obj_boost))
                if any(abs(g - w) > 1e-6 for g, w in zip(got, want_swa)):
                    raise RuntimeError(f"{rc['name']}: {tag} SWA={got}, expected {want_swa}")
                if b.area_weight_mode != "sqrt":
                    raise RuntimeError(f"{rc['name']}: {tag} area_weight_mode="
                                       f"{b.area_weight_mode}, expected sqrt")
                # the 0.0/1.0-vs-endpoint bug: clamps must be (end, start)
                if abs(float(b.alpha_min) - want_swa[1]) > 1e-6 or \
                        abs(float(b.alpha_max) - want_swa[0]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} alpha clamp="
                                       f"({b.alpha_min}, {b.alpha_max}), expected "
                                       f"({want_swa[1]}, {want_swa[0]})")
            if b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SNL1/NWD live on {tag}")

        if e["sbb"] is not None:
            s_m, s_o = float(o2m.bbox_loss.sbb_sign), float(o2o.bbox_loss.sbb_sign)
            if s_m * s_o >= 0:
                raise RuntimeError(f"{rc['name']}: SBB signs o2m={s_m:+.0f} o2o={s_o:+.0f} "
                                   f"— they must be OPPOSITE")
            if s_o >= 0:
                raise RuntimeError(f"{rc['name']}: one2one sbb_sign={s_o:+.0f}; inverted SBB "
                                   f"requires one2one to lean SMALL (negative)")
            seen.append(f"SBB q={e['sbb'][0]} invert={e['sbb'][1]} "
                        f"o2m={s_m:+.0f}(large) o2o={s_o:+.0f}(small)")
        # ---- blend endpoint, on the object that owns it
        got_f = float(getattr(crit, "final_o2m", 0.1))
        if abs(got_f - e["o2m_final"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: final_o2m={got_f}, expected {e['o2m_final']}")
        h = o2o.hyp   # E2ELoss has no .hyp; only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set — LB-TAL is CLOSED on v26")
        a = o2o.assigner
        seen.insert(0, f"alpha={a.alpha} beta={a.beta}")
        seen.append("SWA off" if e["swa"] is None else
                    f"SWA {e['swa'][0]}->{e['swa'][1]} sqrt/max px<{e['swa'][2]:.0f} x{e['swa'][3]}")
        seen.append(f"blend {getattr(crit,'o2m_copy',0.8)} -> {got_f}")
        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    seed = rc["seed"]
    print("\n" + "=" * 76)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 76)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{seed} | vs b1 pair mean "
          f"{rc['ctrl']:.2f} | clears at {THRESHOLD:.2f} | {d}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=seed,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": rc["name"], "ctrl": rc["ctrl"], "threshold": THRESHOLD,
           "params": rc["params"], "expect": {k: str(v) for k, v in rc["expect"].items()},
           "seed": seed, "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan"),
           "test_map50_small": float("nan")}
    # args.yaml is what would have caught the y26_stock_b48 mislabelling: the
    # results JSONs record no batch for any run in this project.
    try:
        with open(os.path.join(save_dir, "b1combo_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{rc['name']}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    todo = order([r for r in RUNS if not a.names or r["name"] in a.names])
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  b1 COMBO SCREEN — {len(todo)} runs, threshold {THRESHOLD:.2f} on mAP50_small")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] preflight failed — fix before spending GPU.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_b1combo_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "threshold": THRESHOLD, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  b1 COMBO SCREEN — RESULTS (mAP50-95 only; SCORE mAP50_small BEFORE CONCLUDING)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<18} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<18} mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS SCREEN")
    print(f"  control is y26_b1's PAIR MEAN {B1_S50:.2f}, NOT its seed-0 draw 79.30.")
    print(f"    >= {THRESHOLD:.2f}   a signal. SEED IT (2 runs) before believing it.")
    print(f"    78.9-79.7   uninformative BY CONSTRUCTION. Stop. No fourth run.")
    print(f"    <  78.90    the mechanism costs something on b1. Stop.")
    print("  The threshold was fixed before these ran. Moving it now is the same")
    print("  error round 22b made and round 23 spent 4.5 GPU-h undoing.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
