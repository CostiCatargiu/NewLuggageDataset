#!/usr/bin/env python3
r"""
P234 FINAL — the missing cell, the recall question, and seeds on both candidates

Five runs, ~6.5 GPU-h, 640, BATCH 48. NO PATCH, NO NEW YAML — every graph and
every key already exists. This is the round that turns the P2 family from a map
into two reportable configurations.

    ESTABLISHED (n=2, the first architecture result in the project with n>1)
        y26_p234rich_b1     79.67 / 78.95   MEAN 79.31   +2.17%   4.3 SE
                            mAP50 82.11 (+2.12%)   m5095 54.92 (-0.69%)
                            AR50_95_small 67.80  (-4.69%)  <- BOTH seeds agree

    n=1, the alternative headline
        y26_p234rich_p2only 78.38   m5095 56.00 (+1.27%)   AR 71.38 (+0.34%)
                            the ONLY config in 169 runs above baseline on BOTH
                            mAP50-95 and the small-object recall ceiling

    BASELINE  y26_identity n=3   small 77.62  mAP50 80.40  m5095 55.30  AR 71.14
    FLOOR     pooled sd (df=12)  small 0.428   ->  n=2 vs n=3 SE 0.391


=============================================================================
WHAT THE LAST ROUND SETTLED
=============================================================================
1. THE MAXIMUM REGRESSED, FOR THE SEVENTH TIME. 79.67 -> 78.95, i.e. -0.72
   against a running average of -0.55 across seven repeated maxima. The pair
   mean 79.31 still clears the best loss-only config (y26_b1, 78.94 at n=2), so
   the architecture axis now beats the loss axis on seeded evidence.

2. P2 DEPTH HAS A MEASURED OPTIMUM AT 4.
        depth (19,22)   small50   m5095   AR50_95_small
        (2, 2)           78.10    55.44       71.24
        (4, 2)           78.38    56.00       71.38   <- peak on ALL THREE
        (6, 2)           78.15    55.66       71.23
   Depth 6 is below depth 4 on every metric. The axis is closed with a bracketed
   peak, which is a better claim than an unbounded "more depth helped".

3. beta=0 DID NOT REPRODUCE ITS p2dys WIN.
        p2dys      b0 - b1 = +0.41
        p234rich   b0 - b1 = -0.34      OPPOSITE SIGN
   Third independent confirmation that the beta surface is FLAT. Stop looking
   for a better beta; there isn't one.

4. THE RECALL COST IS REAL AND REPRODUCIBLE. AR50_95_small 67.92 and 67.67 on
   the two seeds of the winner, against a baseline of 71.14. That is not a draw.
   Every high-mAP configuration in this study buys small-object precision and
   pays for it in the recall ceiling.


=============================================================================
THE FIVE
=============================================================================
1  y26_p2only_b1        0.35  THE MISSING CELL. The graph x loss table has a
                              hole exactly where the answer would be:

                                  graph          stock loss   beta=1
                                  p234  (2,2)      78.10       79.57
                                  p2only(4,2)      78.38    *** THIS RUN ***
                                  rich  (4,4)      78.72       79.31 (n=2)

                              p2only is the ONLY recall-preserving graph and
                              beta=1 is what buys the mAP. If any of p2only's
                              recall survives the loss, this is the config the
                              paper should report. If it lands at 79.3 with
                              AR ~68 it is just another point on the trade curve
                              and the trade is structural, which is also worth
                              knowing.

2  y26_p234rich_b1_s2   ----  THIRD SEED ON THE HEADLINE. n=2 gives a mean; n=3
                              gives an sd you can print. This is the number the
                              paper leads with and it currently has no interval.

3  y26_p234rich_p2only_s1 --  SEED THE ALTERNATIVE HEADLINE. y26_p234rich_p2only
                              is the only configuration in the study that clears
                              the baseline on mAP50-95 AND on recall, and it is
                              ONE DRAW. Given seven consecutive regressions, an
                              unrepeated "gains everywhere" claim is not
                              publishable. 1.3 GPU-h to find out.

4  y26_p234rich_b3      0.30  THE RECALL-PRESERVING BETA. On p2dys the three
                              beta values differ little on mAP but NOT on recall:
                                  b0  AR 67.17     b1  AR 66.79     b3  AR 68.13
                              beta=3 held the best recall ceiling of the three
                              and has never been tried on any p234 graph. Since
                              the beta surface is flat on mAP, choosing beta by
                              its RECALL profile costs nothing and may recover a
                              point of AR.

5  y26_p2only_b1_s1     ----  PRE-EMPTIVE SEED OF RUN 1. Deliberately queued
                              blind rather than conditionally: seven of seven
                              maxima have regressed, so if run 1 lands high it
                              WILL need a repeat, and a second night to get one
                              is worse than one wasted hour if it lands low.
                              Drop this if the queue is tight.


=============================================================================
HOW TO SCORE IT — WRITTEN BEFORE THE RUNS
=============================================================================
RUN 1 IS NOT SCORED ON mAP ALONE. Read it as a pair:
    small50 >= 79.3 AND AR50_95_small >= 70   -> the trade is escapable. Report
                                                 this config; it dominates both
                                                 current candidates.
    small50 >= 79.3 AND AR ~ 68               -> the trade is STRUCTURAL. Say so;
                                                 a reproducible precision/recall
                                                 exchange is a finding, not a
                                                 failure.
    small50 ~ 78.4                            -> beta does nothing on this graph.

RUNS 2-3 ARE THE ONES THAT MAKE THE PAPER. Everything else here is exploration;
these two convert the two candidate configurations into rows with intervals.

EXPECT REGRESSION ON RUN 2. Seven maxima, seven regressions, -0.55 mean. If
seed 2 lands near 78.9 the honest n=3 headline is ~79.2, not 79.67.

DO NOT ADD A SIXTH CONFIGURATION. The beta surface is flat (three independent
confirmations) and the depth axis has a measured peak. There is no third axis
left in this family, and adding cells to a mapped plane only manufactures
another maximum that will regress.

    Usage:
        python run_yolo26_p234_final_v6i.py                    # all five
        python run_yolo26_p234_final_v6i.py --preflight        # build, run nothing
        python run_yolo26_p234_final_v6i.py --seeds            # runs 2, 3, 5 only
        python run_yolo26_p234_final_v6i.py y26_p2only_b1      # one by name
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
WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_p234_final_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 48
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_DIR = "ultralytics/cfg/models/26"
CFG_RICH = f"{CFG_DIR}/yolo26-p2dys-p234rich.yaml"          # depth (4,4)
CFG_P2ONLY = f"{CFG_DIR}/yolo26-p2dys-p234rich-p2only.yaml"  # depth (4,2)

# All p234* graphs are 3-level: stock head rows 17-19 -> 23-25. Rows 26-28 do not
# exist, so the 6-row range used for 4-level graphs would silently copy three and
# skip three while printing a healthy count. Zero skips is asserted below.
REMAP_SHIFT = 6
REMAP_3LVL = range(17, 20)

# YOLO(<yaml>) builds a RANDOM model; .load() is what transfers yolo26s.pt.
# Skipping it left 69% of the graph random and cost a full 6-run night.
MIN_PRETRAINED_FRAC = 0.45

REF_S50, REF_50, REF_5095, REF_AR = 77.62, 80.40, 55.30, 71.14
SD_S50 = 0.428
RICH_B1_MEAN, RICH_B1_AR = 79.31, 67.80      # n=2
P2ONLY_S50, P2ONLY_5095, P2ONLY_AR = 78.38, 56.00, 71.38   # n=1
RICH_S50 = 78.72

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


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_p2only_b1", "order": 1, "prior": 0.35, "seed": 0, "seed_of": None,
     "cfg": CFG_P2ONLY, "depth": (4, 2), "beta": 1.0, "params": cfg(tal_beta=1.0),
     "ctrl": P2ONLY_S50, "ctrl_ar": P2ONLY_AR,
     "label": "depth (4,2) + beta 1.0 — THE MISSING CELL",
     "why": "p2only is the only recall-preserving graph (AR 71.38 vs baseline "
            "71.14) and beta=1 is what buys the mAP. If any recall survives the "
            "loss, this is the config to report. If it lands at 79.3 with AR ~68, "
            "the precision/recall trade is structural — also worth knowing."},

    {"name": "y26_p234rich_b1_s2", "order": 2, "prior": 0.00, "seed": 2, "seed_of": "y26_p234rich_b1",
     "cfg": CFG_RICH, "depth": (4, 4), "beta": 1.0, "params": cfg(tal_beta=1.0),
     "ctrl": RICH_B1_MEAN, "ctrl_ar": RICH_B1_AR,
     "label": "third seed on the headline — n=2 gives a mean, n=3 gives an sd",
     "why": "This is the number the paper leads with and it currently has no "
            "interval. Expect regression: seven maxima, seven regressions."},

    {"name": "y26_p234rich_p2only_s1", "order": 3, "prior": 0.00, "seed": 1, "seed_of": "y26_p234rich_p2only",
     "cfg": CFG_P2ONLY, "depth": (4, 2), "beta": 6.0, "params": cfg(),
     "ctrl": P2ONLY_S50, "ctrl_ar": P2ONLY_AR,
     "label": "seed the ALTERNATIVE headline — the only 'gains everywhere' config",
     "why": "The only configuration in the study above baseline on BOTH mAP50-95 "
            "and the small-object recall ceiling, and it is one draw. After seven "
            "consecutive regressions an unrepeated 'gains everywhere' claim is "
            "not publishable."},

    {"name": "y26_p234rich_b3", "order": 4, "prior": 0.30, "seed": 0, "seed_of": None,
     "cfg": CFG_RICH, "depth": (4, 4), "beta": 3.0, "params": cfg(tal_beta=3.0),
     "ctrl": RICH_B1_MEAN, "ctrl_ar": RICH_B1_AR,
     "label": "depth (4,4) + beta 3.0 — choose beta by its RECALL profile",
     "why": "On p2dys the three beta values barely differ on mAP but do on "
            "recall: b0 AR 67.17, b1 66.79, b3 68.13. beta=3 held the best "
            "ceiling and has never been tried on a p234 graph. Since the beta "
            "surface is flat on mAP, picking beta by recall costs nothing."},

    {"name": "y26_p2only_b1_s1", "order": 5, "prior": 0.00, "seed": 1, "seed_of": "y26_p2only_b1",
     "cfg": CFG_P2ONLY, "depth": (4, 2), "beta": 1.0, "params": cfg(tal_beta=1.0),
     "ctrl": P2ONLY_S50, "ctrl_ar": P2ONLY_AR,
     "label": "pre-emptive seed of run 1 — queued blind on purpose",
     "why": "Seven of seven maxima have regressed. If run 1 lands high it WILL "
            "need a repeat, and waiting a second night for one is worse than one "
            "wasted hour if it lands low. Drop this if the queue is tight."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    print("=" * 92)
    print("  PREFLIGHT — building each graph")
    print("=" * 92)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import DetectionModel
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False, set()
    if "tal_beta" not in DEFAULT_CFG_DICT:
        print("  [ABORT] default.yaml does not accept tal_beta")
        return False, set()

    seen = {}
    for rc in todo:
        c = rc["cfg"]
        if not os.path.exists(c):
            print(f"  [ABORT] {c} not found")
            return False, set()
        if c not in seen:
            try:
                m = DetectionModel(cfg=c, nc=3, verbose=False)
                det = m.model[-1]
                seen[c] = (sum(x.numel() for x in m.parameters()),
                           int(getattr(det, "nl", -1)),
                           [int(s) for s in det.stride.tolist()])
                m.eval()
                with torch.no_grad():
                    m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
            except Exception as ex:
                print(f"  [ABORT] {c} failed to build: {type(ex).__name__}: {ex}")
                return False, set()
        p, nl, st = seen[c]
        good = nl == 3 and st == [4, 8, 16]
        ok &= good
        print(f"  {rc['name']:<24} {os.path.basename(rc['cfg']):<38} params {p:>11,}  "
              f"nl {nl}  {str(st):<14} {'OK' if good else 'WRONG, want nl=3 [4,8,16]'}")

    print()
    print("  THE GRAPH x LOSS TABLE, once this round lands (mAP50_small):")
    print(f"    {'graph':<16}{'stock loss':>12}{'beta=1':>12}{'beta=3':>12}")
    print(f"    {'p234  (2,2)':<16}{78.10:>12.2f}{79.57:>12.2f}{'-':>12}")
    print(f"    {'p2only(4,2)':<16}{P2ONLY_S50:>12.2f}{'RUN 1':>12}{'-':>12}")
    print(f"    {'rich  (4,4)':<16}{RICH_S50:>12.2f}{RICH_B1_MEAN:>12.2f}{'RUN 4':>12}")
    print()
    for rc in todo:
        dd = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        tag = f"p={rc['prior']:.2f}" if rc["prior"] > 0 else f"SEED of {rc['seed_of']}"
        print(f"  RUN  {rc['name']:<24} depth {str(rc['depth']):<7} seed {rc['seed']}  "
              f"{tag:<28} {dd if dd else 'STOCK LOSS'}")
    print(f"\n  {len(todo)} runs, ~{1.3 * len(todo):.1f} GPU-h, b{BATCH}, no patch, no new yaml")
    return ok, {r["name"] for r in todo}


# ------------------------------------------------------------------- load --
def load_pretrained(model):
    """Transfer yolo26s.pt. MUST RUN BEFORE remap_pan().

    YOLO(<yaml>) builds a RANDOMLY INITIALISED model and remap_pan() only copies
    the shifted PAN rows. Skipping .load() leaves ~69% of the network random.
    Symptoms: no "Transferred X/Y items" line, epoch-1 mAP50 ~0.015 instead of
    ~0.4, final mAP50 ~0.74 against a ~0.80 baseline. That cost a full night once.
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
    if frac < MIN_PRETRAINED_FRAC:
        raise RuntimeError(f"only {100 * frac:.1f}% of parameters came from {WEIGHTS}; "
                           f"the graph is mostly RANDOM and would train from near-scratch.")
    return moved, frac


# -------------------------------------------------------------------- remap --
def remap_pan(model, rows):
    """Copy stock head rows into the positions the P2 graph shifts them to.

    Fixing the transfer was worth +0.81 (y26_p2_b32 55.03 -> y26_p2_remap 55.84).
    Zero skips is asserted: a partial copy prints a healthy-looking count.
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
        raise RuntimeError(f"remap skipped {skipped} tensors inside rows "
                           f"{rows.start}-{rows.stop - 1}; a silent partial copy is "
                           f"exactly what this assert prevents")
    model.model.load_state_dict(moved, strict=False)
    after = model.model.state_dict()
    for k2, v in moved.items():
        if not torch.equal(after[k2].float(), v.float()):
            raise RuntimeError(f"remap did not stick on {k2}")
    print(f"  [remap] {len(moved)} tensors, "
          f"{sum(v.numel() for v in moved.values()) / 1e6:.2f}M params, 0 skipped")
    return len(moved)


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """beta and the GRAPH DEPTH are the only things allowed to vary."""
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
            if abs(float(a.beta) - rc["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {rc['beta']}")
            if abs(float(a.alpha) - 0.5) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected stock 0.5")
            for attr in ("beta_small", "beta_sel", "beta_tgt", "beta_level"):
                if getattr(a, attr, None) is not None:
                    raise RuntimeError(f"{rc['name']}: {tag} {attr} live — beta is the only "
                                       f"loss variable in this round")
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live")
            if a.scb_enabled() or a.snt_enabled() or a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: SCB/SNT/TSH live — not requested")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0 or b.swa_enabled() \
                    or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB/SWA/SNL1/NWD live on {tag}")
        if bool(getattr(o2o.hyp, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal set — LB-TAL is CLOSED on v26")
        det = trainer.model.model[-1]
        nl, st = int(getattr(det, "nl", -1)), [int(s) for s in det.stride.tolist()]
        if nl != 3 or st != [4, 8, 16]:
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}, expected "
                               f"3 / [4, 8, 16] — wrong graph")
        print(f"  [guard] beta={rc['beta']}, alpha=0.5, nothing else live")
        print(f"  [guard] detect levels {nl}, strides {st}, head depth {rc['depth']}")
        if rc["seed_of"]:
            print(f"  [guard] certified as seed {rc['seed']} of {rc['seed_of']}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    print("\n" + "=" * 92)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 92)
    dd = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  {os.path.basename(rc['cfg'])} | depth{rc['depth']} | b{BATCH} imgsz{IMG_SIZE} "
          f"ep{EPOCHS} seed{rc['seed']} | vs {rc['ctrl']:.2f} / AR {rc['ctrl_ar']:.2f} | "
          f"{dd if dd else 'STOCK LOSS'}\n")
    t0 = time.time()
    model = YOLO(rc["cfg"])
    n_load, frac_load = load_pretrained(model)      # MUST precede remap_pan
    n_moved = remap_pan(model, REMAP_3LVL)
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=rc["seed"],
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    n_par = sum(p.numel() for p in model.model.parameters())
    out = {"name": rc["name"], "cfg": rc["cfg"], "head_depth": list(rc["depth"]),
           "seed": rc["seed"], "seed_of": rc["seed_of"], "batch": BATCH,
           "imgsz": IMG_SIZE, "epochs": EPOCHS, "tal_beta": rc["beta"],
           "detect_levels": 3, "strides": [4, 8, 16], "remap_tensors": n_moved,
           "pretrained_params": n_load, "pretrained_frac": round(frac_load, 4),
           "n_params": n_par, "loss_params": rc["params"], "ctrl": rc["ctrl"],
           "prior": rc["prior"], "hours": hours, "weights": weights,
           "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "p234_final_params.json"), "w") as f:
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
    ap.add_argument("--seeds", action="store_true", help="only the seed repeats (2, 3, 5)")
    a = ap.parse_args()

    todo = RUNS
    if a.seeds:
        todo = [r for r in todo if r["seed_of"]]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    todo = order(todo)
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 92)
    print(f"  P234 FINAL — the missing cell, the recall question, and seeds ({len(todo)} runs)")
    print("=" * 92)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph did not build with 3 levels / strides [4, 8, 16].")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_p234_final_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "results": res}, f, indent=2)

    print("\n" + "=" * 92)
    print("  P234 FINAL — RESULTS (mAP50-95 only; SCORE mAP50_small AND AR50_95_small)")
    print("=" * 92)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<24} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<24} depth {str(r['head_depth']):<7} beta {r['tal_beta']:<4} "
                  f"seed {r['seed']}  mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print("  RUN 1 IS A PAIR, NOT A NUMBER:")
    print(f"    small >= 79.3 AND AR >= 70   -> the trade is ESCAPABLE. Report this config.")
    print(f"    small >= 79.3 AND AR ~ 68    -> the trade is STRUCTURAL. Say so; a")
    print(f"                                    reproducible exchange is a finding.")
    print(f"    small ~ {P2ONLY_S50:.1f}                -> beta does nothing on this graph.")
    print("  RUNS 2-3 MAKE THE PAPER. They convert the two candidate configurations")
    print("    into rows with intervals; everything else here is exploration.")
    print(f"  EXPECT REGRESSION ON RUN 2. Seven maxima, seven regressions, -0.55 mean.")
    print(f"    Current n=2 mean is {RICH_B1_MEAN:.2f}; if seed 2 lands ~78.9 the honest")
    print(f"    n=3 headline is ~79.2, not 79.67.")
    print(f"  BASELINE small {REF_S50:.2f}  mAP50 {REF_50:.2f}  m5095 {REF_5095:.2f}  AR {REF_AR:.2f}")
    print(f"  FLOOR sd {SD_S50:.3f} (df=12) -> n=2 vs n=3 SE 0.391, n=3 vs n=3 SE 0.349")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
