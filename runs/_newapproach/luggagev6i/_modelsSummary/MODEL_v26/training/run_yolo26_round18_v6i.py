#!/usr/bin/env python3
r"""
YOLO26 ROUND 18 — THREE CONFIGS FOR mAP50 / mAP50_small (stock yolo26s, b82)

Three runs, ~2.6 GPU-h, seed 0 only. Seed the winner afterwards; nothing here is
reportable at n=1.

PRIMARY METRIC IS mAP50 AND mAP50_small. That is what this project reports, and
it is not the metric the campaign has been optimising — which is why two of the
three mechanisms below were written off.

    bars (b82, stock graph)      mAP50    mAP50_small
    y26_scb_b4s2                 81.03       78.65     <- best small, n=1
    y26_swa_a09_04               81.08       78.23     <- n=1
    y26_identity                 80.18       77.30
    noise floor                  ~0.30 (RESULTS_TABLES_v26.txt NOTES)


=============================================================================
WHY THESE THREE
=============================================================================
Two families dominate mAP50_small on this graph, and they have never met.

SWA — 7 runs, 7 above baseline on mAP50_small, mean +0.52:
    a09_04 78.23   a06_b15 77.97   a08_04 77.81   a07_04 77.75
    a06_03 77.73   a09_03 77.67    a06_b25 77.58        identity 77.30
Seven for seven is the only multi-run support any mechanism in this campaign
has. SWA was filed as "-0.48" because that number came from y26_sqrt0703 at a
DIFFERENT batch, judged on overall mAP50-95 — the metric SWA is designed to
trade away. It weights the box loss by sqrt(area) with a small-object boost;
scoring it on overall mAP50-95 scores exactly what it gives up.

SCB — the tal_beta LEVEL, not the conditioning:
    beta 4.0 + beta_small 2.0   (scb_b4s2)  81.03 / 78.65
    beta 4.0 alone              (beta4)     80.95 / 78.17
    beta 6.0 + beta_small 3.0   (scb_b3)    80.75 / 78.02
    beta 6.0 + beta_small 2.0   (scb_s2)    80.70 / 77.53
    beta 6.0 + beta_small 4.0   (scb_s4)    80.25 / 77.13
Roughly 100 runs varied beta_small at a FIXED tal_beta=6.0. Only two ever moved
the global beta, and both beat every beta=6 config on mAP50_small. At beta=6 the
beta_small sweep is non-monotone and inside noise; at beta=4 both points sit
above all of it. The campaign swept the conditioning and left the level alone.


  RUN 1  swa09_b4s2      SWA and SCB act at different stages — SCB in tal.py
                         (which anchors become positive), SWA in
                         BboxLoss.swa_weight (how much each positive counts).
                         Two families, each averaging ~+0.5 on mAP50_small,
                         never combined. Highest prior left on this axis.

  RUN 2  b4s2_sbb50      b4s2 is the mAP50_small leader and has never met SBB.
                         Every SBB run so far sat at the stock beta=6.0. SBB is
                         independently worth +0.87 mAP50 at n=3.

  RUN 3  b3s15           One parameter. Holds b4s2's beta/beta_small ratio at
                         2.0 and drops the LEVEL to 3.0. The only run here that
                         cannot be confounded, and the one that makes runs 1-2
                         interpretable if either wins.

CALIBRATION: this project is 1-for-10 on combinations, and of nine directional
predictions made in this campaign, nine were falsified and every one was
optimistic. Runs 1 and 2 stack two mechanisms each and carry that base rate.
Run 3 does not.


=============================================================================
SIZE BUCKETS COME FROM THE COCO PASS
=============================================================================
ultralytics val() returns box.map50 / box.map only. mAP50_small is produced by
CocoEvalAllFolders_luggage.py, and the small bucket is the point of this round.
Two things to fix in that script FIRST:

  1. glob "*_params.json" instead of the hardcoded four filenames, so metrics
     bind to a CONFIG rather than to directory order — round 16 was mis-evaluated
     twice for exactly this reason.
  2. skip runs whose results.csv is short of EPOCHS. y26_scb3_sbb50_scale75
     entered the round-17 table as a 40-epoch model scored against 70-epoch ones.

    Usage:
        python run_yolo26_round18_v6i.py                    # all three
        python run_yolo26_round18_v6i.py y26_b3s15          # one by name
        python run_yolo26_round18_v6i.py --seed 1           # seed the winner later
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
MODEL_WEIGHTS = "yolo26s.pt"  # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_round18_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches y26_base_rep and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

CTRL_STOCK_50 = 80.18   # y26_identity     mAP50
CTRL_STOCK_S50 = 77.30  # y26_identity     mAP50_small
CTRL_B4S2_50 = 81.03    # y26_scb_b4s2     THE BAR, n=1
CTRL_B4S2_S50 = 78.65   # y26_scb_b4s2     mAP50_small, n=1
CTRL_SWA_50 = 81.08     # y26_swa_a09_04   n=1
CTRL_SWA_S50 = 78.23    # y26_swa_a09_04   mAP50_small, n=1
NOISE = 0.30            # noise floor, RESULTS_TABLES_v26.txt NOTES

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    scb_branch="both",
    o2m_start=0.8, o2m_final=0.1, o2m_decay=True,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0,
    cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
    multi_scale=0.0, scale=0.5, close_mosaic=CLOSE_MOSAIC, cos_lr=False,
)


def _swa(start, end, boost=2.0, px=48):
    """Copied verbatim from run_yolo26_sweep_v6i.py, which produced y26_swa_a09_04.

    NOTE alpha_min/alpha_max are the ENDPOINTS (end, start) — not 0.0/1.0 as a
    later helper in the same file used. Reconstructing these from memory is how
    this project loses runs.
    """
    return dict(alpha_start=start, alpha_end=end, alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost)


_B4S2 = dict(tal_beta=4.0, tal_beta_small=2.0, tal_beta_ref_px=64.0)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_swa09_b4s2", "ctrl": CTRL_B4S2_50, "seed": 0,
     "params": cfg(**_B4S2, **_swa(0.9, 0.4)),
     "expect": {"beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.0,
                "swa": (0.9, 0.4, 48.0, 2.0)},
     "label": "SWA 0.9->0.4 + SCB beta 4.0 / beta_small 2.0 — two families, different stages",
     "why": "SWA is 7-for-7 above baseline on mAP50_small (mean +0.52) and the beta=4 SCB "
            "points lead the same metric. SWA acts on the box-loss weight, SCB on the "
            "assignment metric, so they are not redundant by construction — and they have "
            "never been combined; every prior SWA pairing was with LB-TAL or the P2 graph. "
            "If the effects are independent this is the config. If it lands at either "
            "single, they are two routes to the same correction and that closes the axis."},

    {"name": "y26_b4s2_sbb50", "ctrl": CTRL_B4S2_50, "seed": 0,
     "params": cfg(**_B4S2, sbb_q=0.5, sbb_invert=True),
     "expect": {"beta": 4.0, "beta_small": (2.0, 64.0), "sbb": 0.5, "swa": None},
     "label": "SCB beta 4.0 / beta_small 2.0 + SBB 0.5 inv — the leader meets SBB",
     "why": "b4s2 is the best mAP50_small config on this graph and has never carried SBB; "
            "every SBB run sat at the stock beta=6.0. SBB is independently worth +0.87 "
            "mAP50 at n=3, which makes this the only pairing here with two multi-run "
            "ingredients rather than two single draws."},

    {"name": "y26_b3s15", "ctrl": CTRL_B4S2_50, "seed": 0,
     "params": cfg(tal_beta=3.0, tal_beta_small=1.5, tal_beta_ref_px=64.0),
     "expect": {"beta": 3.0, "beta_small": (1.5, 64.0), "sbb": 0.0, "swa": None},
     "label": "beta 3.0 / beta_small 1.5 — b4s2's ratio, one step down the LEVEL",
     "why": "tal_beta has exactly two measured values, 6.0 and 4.0, and both beta=4 runs "
            "beat every beta=6 config on mAP50_small. This holds the beta/beta_small ratio "
            "at 2.0 and moves only the level, separating 'the level matters' from 'the "
            "ratio matters' — neither has been isolated, because the level was never "
            "varied. It is also the only run in this round that stacks nothing."},
]


def preflight(todo):
    """No new loss.py keys this round — every parameter already exists upstream.

    Still probes the CONSUMER: rounds 4-6 lost ten runs to a key that default.yaml
    accepted, the run header printed, and loss.py never read.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False
    probe = TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.5, beta=3.0)
    ok = abs(float(probe.beta) - 3.0) < 1e-6
    print(f"  {'TaskAlignedAssigner stores beta':<44} {ok}")
    if not ok:
        print("\n  [ABORT] the assigner is not reading beta — nothing here would run.")
        return False

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  {r['name']:<18} seed{r['seed']}  |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.87 * len(todo):.1f} GPU-h")
    print(f"  bar: {CTRL_B4S2_50:.2f} mAP50 / {CTRL_B4S2_S50:.2f} mAP50_small "
          f"(y26_scb_b4s2, n=1)   noise ~{NOISE}")
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that every requested mechanism is LIVE, and nothing else is."""
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 1:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — not a yolo26 e2e model")
        a1, a2 = o2m.assigner, o2o.assigner
        b1, b2 = o2m.bbox_loss, o2o.bbox_loss
        seen = []

        # ---- beta and beta_small ARE the experiment
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            if not a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: {tag} SCB not live, expected {e['beta_small']}")
            bs, ref = e["beta_small"]
            if abs(float(a.beta_small) - bs) > 1e-6 or abs(float(a.beta_ref_px) - ref) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} SCB=({a.beta_small}, {a.beta_ref_px}), "
                                   f"expected {e['beta_small']}")
        seen.append(f"beta={e['beta']} beta_small={e['beta_small'][0]}@{e['beta_small'][1]}px")

        # ---- SWA: presence AND absence asserted, on both branches
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if e["swa"] is None:
                if b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")
            else:
                want = e["swa"]
                if not b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA not live on {tag}")
                got = (float(b.alpha_start), float(b.alpha_end),
                       float(b.small_obj_px), float(b.small_obj_boost))
                if any(abs(g - w) > 1e-6 for g, w in zip(got, want)):
                    raise RuntimeError(f"{rc['name']}: {tag} SWA={got}, expected {want}")
                if b.area_weight_mode != "sqrt":
                    raise RuntimeError(f"{rc['name']}: {tag} area_weight_mode={b.area_weight_mode}, "
                                       f"expected sqrt")
                if abs(float(b.alpha_min) - want[1]) > 1e-6 or abs(float(b.alpha_max) - want[0]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} alpha clamp=({b.alpha_min}, "
                                       f"{b.alpha_max}), expected ({want[1]}, {want[0]}) — the "
                                       f"endpoints, as in run_yolo26_sweep_v6i.py")
        seen.append("SWA off" if e["swa"] is None else
                    f"SWA {e['swa'][0]}->{e['swa'][1]} sqrt/max px<{e['swa'][2]:.0f} x{e['swa'][3]}")

        # ---- SBB
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if abs(float(b.sbb_q) - e["sbb"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {e['sbb']}")
        if e["sbb"] > 0.0:
            if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} "
                                   f"o2o={b2.sbb_sign:+.0f} — they must be OPPOSITE")
            if float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: one2one sbb_sign={b2.sbb_sign:+.0f}; the arm "
                                   f"that won is invert=True -> one2one leans SMALL (sign<0)")
            seen.append(f"SBB q={e['sbb']} o2m={b1.sbb_sign:+.0f}(large) "
                        f"o2o={b2.sbb_sign:+.0f}(small)")
        else:
            seen.append("SBB off")

        # ---- everything else must be provably off
        for a in (a1, a2):
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            if a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: SBAL is live but was not requested")
            if abs(float(a.alpha) - 0.5) > 1e-6:
                raise RuntimeError(f"{rc['name']}: tal_alpha={a.alpha}, expected the stock 0.5")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.snl1_enabled():
                raise RuntimeError(f"{rc['name']}: SNL1 is live on {tag} but was not requested")
            if float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: NWD is live on {tag} but was not requested")
        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this file is the stock 3-level graph")
        if abs(float(h.cls) - 0.5) > 1e-6 or abs(float(h.box) - 7.5) > 1e-6:
            raise RuntimeError(f"{rc['name']}: gains moved (box={h.box} cls={h.cls}); this round "
                               f"varies the assignment metric and the box-loss weighting only")
        if abs(float(trainer.args.multi_scale)) > 1e-6:
            raise RuntimeError(f"{rc['name']}: multi_scale is live; it measured -1.03 (4.3 sd)")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name, seed = rc["name"], rc["seed"]
    print()
    print("=" * 78)
    print(f"  RUN {name}")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={seed}  bar={rc['ctrl']:.2f} mAP50")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(rc["params"])
    results = model.train(**kw)
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "ctrl": rc["ctrl"], "params": rc["params"], "seed": seed,
           "expect": {k: list(v) if isinstance(v, tuple) else v for k, v in rc["expect"].items()},
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True, "epochs_requested": EPOCHS,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # written as *_params.json so the eval script's glob binds metrics to a CONFIG
    # rather than to directory order — round 16 was mis-evaluated for exactly this.
    try:
        with open(os.path.join(save_dir, "round18_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res):
    ok = [r for r in res if r["test_map50"] == r["test_map50"]]
    if not ok:
        print("\nno completed runs.")
        return
    print("\n" + "=" * 78)
    print("  ROUND 18 — RESULTS  (mAP50; mAP50_small needs the COCO pass)")
    print("=" * 78)
    print(f"{'run':<22}{'mAP50':>9}{'vs bar':>9}{'vs stock':>10}{'mAP50-95':>10}{'hours':>7}")
    print("-" * 67)
    print(f"{'y26_identity':<22}{CTRL_STOCK_50:>9.2f}{CTRL_STOCK_50 - CTRL_B4S2_50:>+9.2f}"
          f"{0.0:>+10.2f}{'-':>10}{'-':>7}")
    print(f"{'y26_swa_a09_04':<22}{CTRL_SWA_50:>9.2f}{CTRL_SWA_50 - CTRL_B4S2_50:>+9.2f}"
          f"{CTRL_SWA_50 - CTRL_STOCK_50:>+10.2f}{'-':>10}{'-':>7}")
    print(f"{'y26_scb_b4s2 (bar)':<22}{CTRL_B4S2_50:>9.2f}{0.0:>+9.2f}"
          f"{CTRL_B4S2_50 - CTRL_STOCK_50:>+10.2f}{'-':>10}{'-':>7}")
    print("-" * 67)
    for r in sorted(ok, key=lambda x: -x["test_map50"]):
        v, v95 = r["test_map50"] * 100, r["test_map5095"] * 100
        print(f"{r['name']:<22}{v:>9.2f}{v - CTRL_B4S2_50:>+9.2f}"
              f"{v - CTRL_STOCK_50:>+10.2f}{v95:>10.2f}{r['hours']:>7.2f}")

    print("\n  READ IT")
    best = max(ok, key=lambda x: x["test_map50"])
    d = best["test_map50"] * 100 - CTRL_B4S2_50
    print(f"    best: {best['name']} {best['test_map50'] * 100:.2f} mAP50  {d:+.2f} vs the bar")
    if d >= NOISE:
        print(f"    Clears the ~{NOISE} noise floor. SEED IT (--seed 1, then --seed 2) before it")
        print("    goes in any table — cls075 led at seed 0 by +0.65 and lost that lead twice.")
    else:
        print(f"    Inside the ~{NOISE} noise floor. Several configs now sit near 81.0 mAP50 by")
        print("    different mechanisms, which is a plateau, not a ranking. Report the plateau")
        print("    and stop tuning the loss.")
    print("\n    mAP50_small is the metric this round exists for and is NOT in the table above.")
    print(f"    Run the COCO pass and read it against {CTRL_B4S2_S50:.2f} (y26_scb_b4s2) and")
    print(f"    {CTRL_STOCK_S50:.2f} (identity) before concluding anything.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    ap.add_argument("--seed", type=int, default=SEED, help="seed the winner later")
    a = ap.parse_args()

    todo = RUNS
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if a.seed != SEED:
        todo = [{**r, "seed": a.seed, "name": f"{r['name']}_s{a.seed}"} for r in todo]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 18 — three configs for mAP50 / mAP50_small ({len(todo)} runs)")
    print("  " + "  ".join(r["name"] for r in todo))
    print("=" * 84)
    if not preflight(todo):
        return

    res, out_path = [], f"{PROJECT_DIR}_results.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"\n  [FAILED] {rc['name']}: {ex}\n")
        # written after EVERY run: if the queue dies at 3am the finished ones survive
        try:
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
        except Exception as ex:
            print(f"  [warn] results not saved: {ex}")
    summarise(res)
    print(f"\n  results -> {out_path}")


if __name__ == "__main__":
    main()
