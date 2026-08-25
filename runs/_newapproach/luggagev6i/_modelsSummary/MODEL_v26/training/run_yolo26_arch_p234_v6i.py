#!/usr/bin/env python3
r"""
PURE ARCH — is the P5 detection level vestigial on v6i?

Four runs, ~4.5 GPU-h, 640, BATCH 48, STOCK LOSS on every run. No loss mechanism
is live anywhere in this file; that is the point. The loss x arch combination is
a SEPARATE round and must not be mixed in here.

    CONTROL   P2+DySample, n=10 replicates      small 78.43 (sd 0.345)
                                                m5095 56.12 (sd 0.200)
    MATCHED   y26_stock_b48, n=1                small 77.68   m5095 55.76
              -> P2+DySample is worth +0.75 small, +0.36 m5095

    NOTE THE n=10 IS SAME-SEED. Rounds 4-6 passed LB-TAL budgets while stock
    loss.py was installed, so nothing read the flag and ten differently-labelled
    runs are ten replicates of ONE config. Their sd measures DySample's
    grid_sample non-determinism, NOT seed variance. Run 4 of this file measures
    seed variance on the arch graph for the first time in the project.


=============================================================================
THE CLAIM: P5 EARNS 0.13 POSITIVES PER GT AND COSTS THE MOST PARAMETERS
=============================================================================
DATASET_v6i/raw/diag_anchor_footprin_results.txt, finding F1, titled
"THE P5 BUDGET IS LARGELY FICTION":

    level                        s8      s16      s32
    pool < topk                36.2%    77.8%    96.2%
    pool == 0                   0.2%     6.3%    31.1%
    positives to small GTs      6.64     1.08     0.01
    positives to medium GTs     6.48     3.39     0.00
    positives to large GTs      0.16     7.96     1.68
    SEL BIAS                    0.94     1.42     0.34   <- METRIC-BIASED

Small GTs draw 0.01 positives from stride 32; medium draw 0.00. Only large uses
it, at 1.68, and large is 7.7% of this dataset -- so P5 earns about 0.13
positives per ground-truth object while rows 7-10 and 28 are the most expensive
stage in the network.

SEL BIAS 0.34 says s32 is METRIC-starved, not merely geometry-starved. A
per-level budget cannot repair that, which is why LB-TAL's six attempts on v26
all landed BELOW baseline (77.40, 77.26, 76.89, 76.79, 76.72).

LARGE OBJECTS DO NOT LIVE ON P5. Finding F5: 7.96 of large's 9.79 positives come
from s16. Removing P5 should cost large very little -- but mAP50_large has a
pooled sd of 2.33 here, so report "no DETECTABLE change", never "no change".

SECONDARY: with P2 present, the top-down path from P5 to P2 is THREE upsamples.
Dropping P5 shortens the dilution chain to the level where 60% of instances are.


=============================================================================
THE REMAP, AND THE ONE THING THAT CHANGES FOR A 3-LEVEL HEAD
=============================================================================
yolo26-p2* pushes stock head rows 17-22 down by 6, to 23-28. Loading yolo26s.pt
naively leaves the whole bottom-up PAN randomly initialised -- that is the
"BROKEN-transfer graph" the p2dys yaml header refers to, and fixing it was worth
+0.81 (y26_p2_b32 55.03 -> y26_p2_remap 55.84).

    p2dys  (4 detect levels)   stock 17-22 -> 23-28   SIX rows
    p234   (3 detect levels)   stock 17-19 -> 23-25   THREE rows

Rows 26-28 DO NOT EXIST in the p234 graphs, so stock rows 20-22 (the P5
bottom-up branch) have no destination. That is correct -- those weights belong
to a branch this graph does not have. But the inherited remap_pan() guards with
`if k2 in sd`, so passing the 6-row range would SILENTLY copy three rows and
skip three, and print a count that looks fine. This file passes the row range
per run and ASSERTS the number of tensors moved, so a mismatch aborts.


=============================================================================
THE FOUR RUNS
=============================================================================
1  y26_p2dys_p234       drop P5 DETECTION, keep the P5 BACKBONE stage.
   prior 0.35           Rows 0-25 are byte-identical to yolo26-p2dys.yaml, so
                        this is a ONE-VARIABLE change against an n=10 control --
                        a better-anchored comparison than anything on the loss
                        axis, where controls are n=2 or n=1.
                        Keeping rows 7-10 (incl. SPPF and C2PSA) matters: it
                        separates "P5 detection is useless" from "P5 context is
                        useless". Truncating the backbone would confound them.

2  y26_p2dys_p234_s1    Seed 1 of run 1. The arch sd on small is 0.345, so a
   ----                 single draw cannot resolve anything under ~1.0. Round 23
                        cost 4.5 GPU-h learning this on the loss axis; do not
                        repeat it here.

3  y26_p2dys_p234rich   Drop P5 AND spend the freed capacity on the fine levels:
   prior 0.25           rows 19 and 22 go from 2 to 4 repeats (P2/4 and P3/8).
                        DEPTH, not width -- width was tried three times
                        (p2_wide 55.53, wide_starve 55.46, p2addw_base 55.06)
                        and all three landed at or below the matched control.
                        Repeat count at the FINE levels has never been touched.
                        NOT a one-variable run. Only interpretable if run 1
                        shows removing P5 is free; if run 1 loses, this
                        confounds two changes and should not be scored as arch.

4  y26_p2dys_ctrl_s1    The CONTROL at seed 1. Every arch number in this project
   ----                 rests on same-seed replicates; seed variance on the arch
                        graph has never been measured. If this lands far from
                        78.43 then the n=10 sd of 0.345 understates the real
                        floor and every arch delta in the campaign is softer
                        than it reads -- including this round's.


=============================================================================
WHAT THIS ROUND IS EXPECTED TO PRODUCE
=============================================================================
Stated in advance because the campaign's directional predictions are 0-for-12
and every miss was optimistic.

    mAP roughly FLAT, at 30-40% fewer parameters.

That is a legitimate result -- "same accuracy, smaller model, and here is the
measurement showing why" -- and it is the honest expectation. This is NOT the
run that produces a bigger number. The bigger number is p2dys + tal_beta=1.0,
which is a separate file and should be run AFTER this one identifies the best
graph. Do not merge them; a two-variable run against an n=10 control wastes the
best-anchored comparison available in the project.

    Usage:
        python run_yolo26_arch_p234_v6i.py                  # all four
        python run_yolo26_arch_p234_v6i.py --preflight      # BUILD the graphs,
                                                            # print params and
                                                            # strides, run nothing
        python run_yolo26_arch_p234_v6i.py y26_p2dys_p234   # one by name
"""

import argparse
import gc
import json
import os
import re
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
WEIGHTS = "yolo26s.pt"                # stock, for the PAN remap source
PROJECT_DIR = "runs_yolo26_arch_p234_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 48                            # matches every arch run in this project
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_DIR = "ultralytics/cfg/models/26"
CFG_P2DYS = f"{CFG_DIR}/yolo26-p2dys.yaml"
CFG_P234 = f"{CFG_DIR}/yolo26-p2dys-p234.yaml"
CFG_P234RICH = f"{CFG_DIR}/yolo26-p2dys-p234rich.yaml"

REMAP_SHIFT = 6
REMAP_4LVL = range(17, 23)            # p2dys: six rows -> 23-28
REMAP_3LVL = range(17, 20)            # p234 : three rows -> 23-25 (26-28 gone)

# Controls. See the header for why the n=10 is same-seed.
CTRL_S50, CTRL_5095 = 78.43, 56.12    # P2+DySample, n=10
CTRL_SD_S50, CTRL_SD_5095 = 0.345, 0.200
STOCK_B48_S50, STOCK_B48_5095 = 77.68, 55.76

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
# =============================================================================


RUNS = [
    {"name": "y26_p2dys_p234", "order": 1, "prior": 0.35, "seed": 0,
     "cfg": CFG_P234, "nl": 3, "strides": [4, 8, 16], "remap": REMAP_3LVL,
     "ctrl": CTRL_S50,
     "label": "drop P5 DETECTION, keep the P5 backbone — one variable vs an n=10 control",
     "why": "P5 earns ~0.13 positives per GT (F1) and large objects live on P4 "
            "(F5, 7.96 of 9.79). Rows 0-25 byte-identical to p2dys."},

    {"name": "y26_p2dys_p234_s1", "order": 2, "prior": 0.00, "seed": 1,
     "cfg": CFG_P234, "nl": 3, "strides": [4, 8, 16], "remap": REMAP_3LVL,
     "ctrl": CTRL_S50,
     "label": "seed 1 of run 1 — arch sd on small is 0.345",
     "why": "A single draw on this graph cannot resolve anything under ~1.0."},

    {"name": "y26_p2dys_p234rich", "order": 3, "prior": 0.25, "seed": 0,
     "cfg": CFG_P234RICH, "nl": 3, "strides": [4, 8, 16], "remap": REMAP_3LVL,
     "ctrl": CTRL_S50,
     "label": "drop P5 + rows 19/22 repeats 2->4 — spend the freed capacity on P2/P3",
     "why": "Depth, not width: width was tried three times and answered. NOT a "
            "one-variable run; only interpretable if run 1 holds."},

    {"name": "y26_p2dys_ctrl_s1", "order": 4, "prior": 0.00, "seed": 1,
     "cfg": CFG_P2DYS, "nl": 4, "strides": [4, 8, 16, 32], "remap": REMAP_4LVL,
     "ctrl": CTRL_S50,
     "label": "CONTROL at seed 1 — arch seed variance, measured for the first time",
     "why": "The n=10 control is ten SAME-SEED replicates. If this lands far from "
            "78.43 every arch delta in the campaign is softer than it reads."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """BUILD every graph. Report levels, strides and parameter counts.

    This is the check that could not be run off-box. It is also the one that
    matters: a 3-level Detect that silently builds with 4 levels, or a graph
    whose parameter count is ABOVE p2dys, invalidates the entire round.
    """
    print("=" * 84)
    print("  PREFLIGHT — building each graph")
    print("=" * 84)
    ok = True
    from ultralytics.nn.tasks import DetectionModel

    base_params = None
    seen = {}
    for rc in todo:
        cfg = rc["cfg"]
        if cfg in seen:
            p, nl, st = seen[cfg]
        else:
            try:
                m = DetectionModel(cfg=cfg, nc=3, verbose=False)
                p = sum(x.numel() for x in m.parameters())
                det = m.model[-1]
                nl = int(getattr(det, "nl", -1))
                st = [int(s) for s in det.stride.tolist()]
                m.eval()
                with torch.no_grad():
                    m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
                seen[cfg] = (p, nl, st)
            except Exception as ex:
                print(f"  [ABORT] {cfg} failed to build: {type(ex).__name__}: {ex}")
                return False, set()
        if cfg == CFG_P2DYS:
            base_params = p
        good_nl = nl == rc["nl"]
        good_st = st == rc["strides"]
        ok &= good_nl and good_st
        print(f"  {rc['name']:<22} {os.path.basename(cfg):<28} params {p:>11,}  "
              f"nl {nl} {'OK' if good_nl else 'WRONG, want ' + str(rc['nl'])}  "
              f"strides {st} {'OK' if good_st else 'WRONG'}")

    if base_params:
        print()
        for cfg, (p, nl, st) in seen.items():
            d = 100.0 * (p - base_params) / base_params
            flag = ""
            if cfg == CFG_P234RICH and p > base_params:
                flag = "  <- ABOVE p2dys: this is NO LONGER capacity-neutral, relabel it"
            print(f"  {os.path.basename(cfg):<30} {p:>11,}  {d:+6.1f}% vs p2dys{flag}")

    print()
    for rc in todo:
        print(f"  RUN  {rc['name']:<22} seed {rc['seed']}  b{BATCH}  "
              f"remap {rc['remap'].start}-{rc['remap'].stop - 1} -> "
              f"{rc['remap'].start + REMAP_SHIFT}-{rc['remap'].stop - 1 + REMAP_SHIFT}  "
              f"vs {rc['ctrl']:.2f}")
    print(f"\n  {len(todo)} runs, ~{1.1 * len(todo):.1f} GPU-h   STOCK LOSS on every run")
    return ok, {r["name"] for r in todo}


# ------------------------------------------------------------------- load --
def load_pretrained(model):
    """Transfer yolo26s.pt into the graph. THIS MUST RUN BEFORE remap_pan().

    YOLO(<yaml>) builds a RANDOMLY INITIALISED model. remap_pan() only copies the
    PAN rows that the P2 graph shifts (120 tensors, 2.97M of 9.67M params). If
    .load() is skipped, the backbone and the whole top-down head start from
    scratch and 69% of the network is random.

    THIS EXACT BUG COST A FULL RUN. Symptoms, for the next person:
        - no "Transferred X/Y items from pretrained weights" in the log
        - epoch-1 mAP50 ~0.015 instead of ~0.4
        - final mAP50 ~0.74 / mAP50-95 ~0.485 against a ~0.80 / ~0.55 baseline
    """
    sd_before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    model.load(WEIGHTS)
    sd_after = model.model.state_dict()
    total = sum(v.numel() for v in sd_after.values())
    moved = sum(sd_after[k].numel() for k in sd_before
                if not torch.equal(sd_before[k].float(), sd_after[k].float()))
    frac = moved / total if total else 0.0
    print(f"  [load ] {moved / 1e6:.2f}M of {total / 1e6:.2f}M params transferred "
          f"from {WEIGHTS}  ({100 * frac:.1f}%)")
    if frac < 0.45:
        raise RuntimeError(
            f"only {100 * frac:.1f}% of parameters came from {WEIGHTS}; the graph is "
            f"mostly RANDOM and this run would train from near-scratch.")
    return moved, frac


# -------------------------------------------------------------------- remap --
def remap_pan(model, rows):
    """Copy stock head rows into the shifted positions this graph puts them in.

    The inherited version hardcoded range(17, 23). For a 3-level head rows 26-28
    do not exist, so three of those six have no destination — and because the
    copy is guarded by `if k2 in sd`, passing the wrong range copies three, skips
    three, and prints a count that looks healthy. The expected count is asserted.
    """
    ckpt = YOLO(WEIGHTS).model.float().state_dict()
    sd = model.model.state_dict()
    moved, skipped = {}, 0
    for k, v in ckpt.items():
        m = re.match(r"model\.(\d+)\.(.+)", k)
        if not m or int(m.group(1)) not in rows:
            continue
        k2 = f"model.{int(m.group(1)) + REMAP_SHIFT}.{m.group(2)}"
        if k2 in sd and sd[k2].shape == v.shape:
            moved[k2] = v
        else:
            skipped += 1
    if not moved:
        raise RuntimeError("remap matched nothing — the row shift or the graph is not "
                           "what this run assumes")
    if skipped:
        raise RuntimeError(f"remap skipped {skipped} tensors inside the requested row "
                           f"range {rows.start}-{rows.stop - 1}. The range does not match "
                           f"this graph — a silent partial copy is exactly the failure "
                           f"this assert exists to prevent.")
    model.model.load_state_dict(moved, strict=False)
    after = model.model.state_dict()
    for k2, v in moved.items():
        if not torch.equal(after[k2].float(), v.float()):
            raise RuntimeError(f"remap did not stick on {k2}")
    print(f"  [remap] {len(moved)} tensors, "
          f"{sum(v.numel() for v in moved.values()) / 1e6:.2f}M params, rows "
          f"{rows.start}-{rows.stop - 1} -> {rows.start + REMAP_SHIFT}-"
          f"{rows.stop - 1 + REMAP_SHIFT}, 0 skipped")
    return len(moved)


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Assert at epoch 2 that the loss is STOCK and the head has the right levels.

    This round's whole claim is that the GRAPH changed and nothing else. A loss
    key left live from a previous session would make every number here a
    two-variable result, and nothing in the logs would say so.
    """
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 2:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss")
        for tag, br in (("one2many", o2m), ("one2one", o2o)):
            a = br.assigner
            if abs(float(a.alpha) - 0.5) > 1e-6 or abs(float(a.beta) - 6.0) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha/beta = {a.alpha}/{a.beta}, "
                                   f"expected STOCK 0.5/6.0 — this is a PURE ARCH round")
            for attr in ("beta_small", "beta_sel", "beta_tgt", "beta_level"):
                if getattr(a, attr, None) is not None:
                    raise RuntimeError(f"{rc['name']}: {tag} {attr} is live — pure arch round")
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live — pure arch round")
            if a.scb_enabled() or a.snt_enabled() or a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: SCB/SNT/TSH live — pure arch round")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0 or b.swa_enabled() \
                    or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB/SWA/SNL1/NWD live — pure arch round")
        h = o2o.hyp
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal set — LB-TAL is CLOSED on v26")
        det = trainer.model.model[-1]
        nl = int(getattr(det, "nl", -1))
        if nl != rc["nl"]:
            raise RuntimeError(f"{rc['name']}: detect has {nl} levels, expected {rc['nl']} "
                               f"— the graph is not the one this run claims")
        st = [int(s) for s in det.stride.tolist()]
        if st != rc["strides"]:
            raise RuntimeError(f"{rc['name']}: strides {st}, expected {rc['strides']}")
        print(f"  [guard] STOCK LOSS confirmed (alpha 0.5, beta 6.0, nothing else live)")
        print(f"  [guard] detect levels {nl}, strides {st}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    print("\n" + "=" * 84)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 84)
    print(f"  {os.path.basename(rc['cfg'])} | b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} "
          f"seed{rc['seed']} | STOCK LOSS | vs {rc['ctrl']:.2f}\n")
    t0 = time.time()
    model = YOLO(rc["cfg"])
    n_load, frac_load = load_pretrained(model)   # MUST precede remap_pan
    n_moved = remap_pan(model, rc["remap"])
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=rc["seed"],
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **_ALL_OFF)
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    n_par = sum(p.numel() for p in model.model.parameters())
    # ---- THE RECORDING FIX. The repo records NO batch for any run: params and
    # run_meta are empty in every results JSON and no args.yaml was ever saved.
    # That single gap is why y26_stock_b48 sat mislabelled as b32 for weeks and
    # why 42 arch runs were, until this week, uninterpretable.
    out = {"name": rc["name"], "cfg": rc["cfg"], "seed": rc["seed"],
           "batch": BATCH, "imgsz": IMG_SIZE, "epochs": EPOCHS,
           "detect_levels": rc["nl"], "strides": rc["strides"],
           "remap_rows": [rc["remap"].start, rc["remap"].stop - 1],
           "remap_tensors": n_moved, "params": n_par, "loss": "STOCK",
           "ctrl": rc["ctrl"], "prior": rc["prior"], "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_p234_params.json"), "w") as f:
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
    print(f"  PURE ARCH — is P5 vestigial?  ({len(todo)} runs, b{BATCH}, STOCK LOSS)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph did not build with the expected levels/strides.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_arch_p234_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "loss": "STOCK",
                       "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  PURE ARCH — RESULTS (mAP50-95 only; SCORE mAP50_small BEFORE CONCLUDING)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<22} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<22} seed {r['seed']}  mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  "
                  f"params {r['params']/1e6:5.2f}M  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print(f"  control is P2+DySample n=10:  small {CTRL_S50:.2f} (sd {CTRL_SD_S50:.3f}), "
          f"m5095 {CTRL_5095:.2f} (sd {CTRL_SD_5095:.3f})")
    print(f"  matched stock_b48:            small {STOCK_B48_S50:.2f}, m5095 {STOCK_B48_5095:.2f}")
    print("  1. run 4 FIRST in your reading. If the control at seed 1 is far from")
    print(f"     {CTRL_S50:.2f}, the 0.345 sd understates the floor and every delta")
    print("     in this round — and in the other 42 arch runs — is softer than it looks.")
    print("  2. REPORT PARAMETERS NEXT TO mAP. Flat mAP at 30-40% fewer parameters is")
    print("     the expected outcome and is a result; do not write it up as a loss.")
    print("  3. mAP50_large has a pooled sd of 2.33. Write 'no detectable change'.")
    print("  4. run 3 is NOT one-variable. Score it only if run 1 holds.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
