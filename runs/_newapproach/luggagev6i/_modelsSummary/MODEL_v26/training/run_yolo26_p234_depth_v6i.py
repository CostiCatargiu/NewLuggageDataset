#!/usr/bin/env python3
r"""
P234 DEPTH — the missing corner, and the one axis the p2 grid left open

Four runs, ~5.0 GPU-h, 640, BATCH 48, seed 0. NO PATCH REQUIRED.
Run 4 is a seed repeat; runs 1-3 are one config each.

    CONTROLS, all n=1, all from the p2 grid
        y26_p2dys_p234        p234       stock loss   small 78.10   m5095 55.44
        y26_p2dys_p234_b1     p234       beta=1       small 79.57   m5095 55.11
        y26_p2dys_p234rich    p234rich   stock loss   small 78.72   m5095 56.10
        y26_remap_dys_stock   p2dys      stock loss   small 78.39   m5095 55.70
    BASELINE  y26_identity n=3           small 77.62   mAP50 80.40  m5095 55.30
    FLOOR     arch replicate sd on small 0.345  ->  run-to-run SE 0.488


=============================================================================
WHAT THE P2 GRID ACTUALLY SHOWED, INCLUDING THE PART THAT FAILED
=============================================================================
THE PREDICTION FAILED. The grid was built on "a stride-4 level makes topk bind,
so selection becomes live and beta should do MORE". It does not:

    stock 3-level graph   b1 - b0 = +0.56
    p2dys graph           b1 - b0 = -0.41      <- INVERTED, b0 beats b1

and the beta surface on p2dys is still flat (b0 79.15, b1 78.74, b3 79.19,
spread 0.45 against 0.56 on the stock graph). The mechanism story is dead and
should be written up as refuted, not quietly dropped.

WHAT SURVIVED IS A TRADE, AND IT IS THE SAME ONE EVERYWHERE:

    run                  d P50 small   d AR50 small   d mAP50-95
    p2dys_b0                +4.08         -1.73          -0.85
    p2dys_b1                +2.46         -2.34          -0.42
    p2dys_p234_b1           +3.86         -1.30          -0.19
    p2dys_p234rich          +2.52         -0.71          +0.80    <- the exception
    p2dys_p234              +0.61         -0.30          +0.14

Every configuration buys small-object PRECISION and pays for it in RECALL
CEILING. AR50_95_small: baseline 71.14, p234_b1 down to 67.26. p234rich is the
only cell of six that gains mAP50-95 and the only one that holds recall
(AR50_95_small 70.92 vs 71.14).

AND ONE MONOTONE SIGNAL, THE ONLY ONE IN THE ROUND:

    depth(19,22)   small50   mAP50-95   AR50_95_small
        2  (p234)    78.10     55.44        71.24
        4  (rich)    78.72     56.10        70.92
                    +0.62     +0.66

Both axes up. Nothing else in the grid did that.


=============================================================================
THE FOUR
=============================================================================
1  y26_p234rich_b1      0.40  THE MISSING CORNER. p234rich carries the best
                              mAP50-95 in the round (56.10) and the best recall
                              retention; beta=1 carries the best small (79.57 on
                              p234). Their intersection has never been run:

                                  graph      stock loss   beta=1
                                  p2dys        55.70       54.88
                                  p234         55.44       55.11
                                  p234rich     56.10     *** THIS RUN ***

                              beta=1 costs -0.82 m5095 on p2dys but only -0.33
                              on p234. If that carries, this lands near 55.8
                              m5095 with ~79.5 small -- above baseline on BOTH,
                              which nothing in 162 runs has managed.

2  y26_p234rich_p2only  0.25  ATTRIBUTION, not performance. p234rich moved rows
                              19 AND 22 together, so its gain cannot be assigned.
                              With p234 (2,2) and p234rich (4,4) already
                              measured, (4,2) gives the P2-only contribution
                              directly and (4,4)-(4,2) gives P3-only by
                              subtraction. Expected to land BETWEEN them; that
                              is the point. Attribution is what makes p234rich
                              reportable rather than merely observed.

3  y26_p234rich6        0.25  THE THIRD POINT ON THE DEPTH AXIS. Two points make
                              a line, not a curve. If 6 continues the trend the
                              axis is open; if it turns over, 4 is a MEASURED
                              optimum -- a better statement than an unbounded
                              "more depth helped".

4  y26_p2dys_p234_b1_s1 ----  SEED THE CURRENT MAXIMUM. y26_p2dys_p234_b1 is the
                              best mAP50 (82.06) and best mAP50_small (79.57) of
                              all 162 runs, and it is ONE DRAW. Six selected
                              maxima in this campaign have regressed on repeat:
                              -0.55 mean in round 23, -1.12 for b4s2_sbb50.
                              1.2 GPU-h protecting the number the paper would
                              headline. If the exploration matters more, drop
                              this and fold it into the seed round -- but do not
                              tabulate 79.57 until it exists.


=============================================================================
HOW TO SCORE IT — WRITTEN BEFORE THE RUNS
=============================================================================
EVERY CONTROL HERE IS n=1. The arch run-to-run SE is 0.488, so a difference of
under ~1.0 between any two of these runs is not resolvable. This round MAPS the
depth axis; it does not establish any point on it.

RUN 1: read against y26_p2dys_p234_b1 (79.57) on small AND against
y26_p2dys_p234rich (56.10) on mAP50-95. It only wins if it holds BOTH. Beating
one while losing the other is the trade this whole round is trying to escape.

RUN 2: the number to compute is the decomposition, not the level:
    P2-only  = run2 - 78.10          P3-only = 78.72 - run2
If P2-only carries most of the +0.62, "spend capacity where the objects are" is
supported. If P3-only does, that story is wrong and the reallocation is generic.

RUN 3: turning over is a RESULT. Report the peak, not the disappointment.

WATCH AR50_95_small ON ALL FOUR. Baseline 71.14. Every high-precision cell in
the p2 grid fell to 66-68 and p234rich held 70.92. A config that reaches 79.5
small while holding AR above ~70 is a materially better result than one that
reaches 79.5 by finding fewer objects more confidently.

    Usage:
        python run_yolo26_p234_depth_v6i.py                     # all four
        python run_yolo26_p234_depth_v6i.py --preflight         # build, run nothing
        python run_yolo26_p234_depth_v6i.py y26_p234rich_b1     # one by name
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
PROJECT_DIR = "runs_yolo26_p234_depth_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 48
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_DIR = "ultralytics/cfg/models/26"
CFG_P2DYS = f"{CFG_DIR}/yolo26-p2dys.yaml"
CFG_P234 = f"{CFG_DIR}/yolo26-p2dys-p234.yaml"
CFG_RICH = f"{CFG_DIR}/yolo26-p2dys-p234rich.yaml"
CFG_RICH6 = f"{CFG_DIR}/yolo26-p2dys-p234rich6.yaml"
CFG_P2ONLY = f"{CFG_DIR}/yolo26-p2dys-p234rich-p2only.yaml"
CFG_P2DEEP6 = f"{CFG_DIR}/yolo26-p2dys-p234-p2deep6.yaml"

# All p234* graphs are 3-level: stock head rows 17-19 -> 23-25. Rows 26-28 do not
# exist, so the 6-row range used for p2dys would silently copy three and skip
# three while printing a healthy count. Asserted in remap_pan().
REMAP_SHIFT = 6
REMAP_3LVL = range(17, 20)

# YOLO(<yaml>) builds a RANDOM model; .load() is what transfers yolo26s.pt.
# Skipping it left 69% of the graph random and cost a full 6-run night.
MIN_PRETRAINED_FRAC = 0.45

# p2 grid controls, all n=1
P234_S50, P234_5095 = 78.10, 55.44
P234B1_S50, P234B1_5095 = 79.57, 55.11
RICH_S50, RICH_5095 = 78.72, 56.10
P2DYS_S50, P2DYS_5095 = 78.39, 55.70
REF_S50, REF_50, REF_5095, REF_AR = 77.62, 80.40, 55.30, 71.14
SD_S50_ARCH = 0.345

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
    {"name": "y26_p234rich_b1", "order": 1, "prior": 0.40, "seed": 0,
     "cfg": CFG_RICH, "depth": (4, 4), "beta": 1.0, "params": cfg(tal_beta=1.0),
     "ctrl": P234B1_S50, "ctrl_g": RICH_5095,
     "label": "p234rich + beta 1.0 — THE MISSING CORNER",
     "why": "p234rich holds the round's best mAP50-95 (56.10) and best recall "
            "retention; beta=1 holds the best small (79.57). Their intersection "
            "has never been run. beta=1 costs -0.82 m5095 on p2dys but only "
            "-0.33 on p234; if that carries this is above baseline on BOTH."},

    {"name": "y26_p234rich_p2only", "order": 2, "prior": 0.25, "seed": 0,
     "cfg": CFG_P2ONLY, "depth": (4, 2), "beta": 6.0, "params": cfg(),
     "ctrl": P234_S50, "ctrl_g": P234_5095,
     "label": "depth (4,2) — ATTRIBUTION of p234rich's gain to P2 or P3",
     "why": "p234rich moved rows 19 AND 22. With (2,2) and (4,4) measured, (4,2) "
            "gives P2-only directly and (4,4)-(4,2) gives P3-only by subtraction. "
            "Expected to land BETWEEN them; that is the point."},

    {"name": "y26_p234rich6", "order": 3, "prior": 0.25, "seed": 0,
     "cfg": CFG_RICH6, "depth": (6, 6), "beta": 6.0, "params": cfg(),
     "ctrl": RICH_S50, "ctrl_g": RICH_5095,
     "label": "depth (6,6) — the third point on the only monotone axis in the grid",
     "why": "Two points make a line, not a curve. Continuing the trend opens the "
            "axis; turning over makes 4 a MEASURED optimum, which is the better "
            "statement."},

    # RETARGETED after run 1. This originally seeded y26_p2dys_p234_b1 (79.57),
    # but run 1 produced y26_p234rich_b1 at 79.67 / 82.57 -- the new maximum on
    # BOTH metrics. Seeding the runner-up is the wrong GPU hour. The two differ by
    # 0.10 on small (0.2 SE, statistically identical), so this seeds the one with
    # the better profile: +0.51 mAP50 and +0.66 on AR50_95_small.
    # ---- ADDED after runs 1-2 landed. The depth decomposition split by LEVEL:
    #        P2 depth 2->4 : small +0.29  m5095 +0.57  AR +0.14
    #        P3 depth 2->4 : small +0.34  m5095 +0.10  AR -0.47
    #      P2 depth raises recall; P3 depth costs it. p234rich bundles both.
    {"name": "y26_p234_p2deep6", "order": 5, "prior": 0.30, "seed": 0,
     "cfg": CFG_P2DEEP6, "depth": (6, 2), "beta": 6.0, "params": cfg(),
     "ctrl": 78.38, "ctrl_g": 56.00,
     "label": "depth (6,2) — isolate the good half of the reallocation and push it",
     "why": "(4,2) is the ONLY cell in 164 runs above baseline on BOTH mAP50-95 "
            "(+0.70) and AR50_95_small (+0.25). If the P2-depth effect is real "
            "and roughly linear, (6,2) extends it without paying the P3 recall "
            "cost. For an abandonment alarm a missed bag is a missed alarm, so "
            "the recall ceiling is arguably what decides the system."},

    {"name": "y26_p234rich_b0", "order": 6, "prior": 0.25, "seed": 0,
     "cfg": CFG_RICH, "depth": (4, 4), "beta": 0.0, "params": cfg(tal_beta=0.0),
     "ctrl": 79.67, "ctrl_g": 54.94,
     "label": "p234rich + beta 0 — the loss that won on p2dys, on the better graph",
     "why": "On p2dys, b0 BEAT b1 on both metrics that matter: small +0.41 and "
            "AR50_95_small +0.38. Every p234 cell so far uses b1, so the better "
            "loss on the other P2 graph has never met the better graph. It is "
            "also the simpler claim: beta=0 means the alignment metric ignores "
            "IoU entirely, which reads better than 'we tuned beta to 1'."},

    {"name": "y26_p234rich_b1_s1", "order": 4, "prior": 0.00, "seed": 1,
     "cfg": CFG_RICH, "depth": (4, 4), "beta": 1.0, "params": cfg(tal_beta=1.0),
     "ctrl": 79.67, "ctrl_g": 54.94,
     "label": "SEED the campaign maximum (82.57 mAP50 / 79.67 small, n=1)",
     "why": "Six selected maxima in this campaign have regressed on repeat: -0.55 "
            "mean in round 23, -1.12 for b4s2_sbb50. 1.2 GPU-h protecting the "
            "number the paper would headline."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """BUILD every graph: levels, strides, forward pass, parameter counts."""
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

    seen, base = {}, None
    try:
        m = DetectionModel(cfg=CFG_P2DYS, nc=3, verbose=False)
        base = sum(x.numel() for x in m.parameters())
    except Exception:
        pass

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
        d = f"{100 * (p - base) / base:+.1f}%" if base else "n/a"
        print(f"  {rc['name']:<22} {os.path.basename(rc['cfg']):<36} "
              f"params {p:>11,} ({d:>6} vs p2dys)  nl {nl}  {str(st):<14} "
              f"{'OK' if good else 'WRONG, want nl=3 [4,8,16]'}")

    print()
    print("  the depth grid, once this round lands:")
    print(f"    {'(row19, row22)':<16}{'graph':<34}{'small50':>9}{'m5095':>9}")
    for d, g, s, gg in [("(2, 2)", "p234              MEASURED", f"{P234_S50:.2f}", f"{P234_5095:.2f}"),
                        ("(4, 2)", "p234rich-p2only   run 2", "?", "?"),
                        ("(4, 4)", "p234rich          MEASURED", f"{RICH_S50:.2f}", f"{RICH_5095:.2f}"),
                        ("(6, 6)", "p234rich6         run 3", "?", "?")]:
        print(f"    {d:<16}{g:<34}{s:>9}{gg:>9}")

    print()
    for rc in todo:
        dd = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        tag = f"p={rc['prior']:.2f}" if rc["prior"] > 0 else "SEED"
        print(f"  RUN  {rc['name']:<22} depth {str(rc['depth']):<8} {tag:<7} "
              f"seed {rc['seed']}  vs {rc['ctrl']:.2f}  {dd if dd else 'STOCK LOSS'}")
    print(f"\n  {len(todo)} runs, ~{1.25 * len(todo):.1f} GPU-h, b{BATCH}, no patch required")
    return ok, {r["name"] for r in todo}


# ------------------------------------------------------------------- load --
def load_pretrained(model):
    """Transfer yolo26s.pt into the graph. MUST RUN BEFORE remap_pan().

    YOLO(<yaml>) builds a RANDOMLY INITIALISED model. remap_pan() only copies the
    PAN rows the P2 graph shifts. Skipping .load() leaves the backbone and the
    whole top-down head random — 69% of the network — and the symptoms are:
        - no "Transferred X/Y items from pretrained weights" in the log
        - epoch-1 mAP50 ~0.015 instead of ~0.4
        - final mAP50 ~0.74 against a ~0.80 baseline
    That cost a full 6-run night once. The fraction is asserted so it cannot
    happen silently again.
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
        raise RuntimeError(
            f"only {100 * frac:.1f}% of parameters came from {WEIGHTS}; the graph is "
            f"mostly RANDOM and this run would train from near-scratch.")
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
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}, expected 3 / "
                               f"[4, 8, 16] — wrong graph")
        print(f"  [guard] beta={rc['beta']}, alpha=0.5, nothing else live")
        print(f"  [guard] detect levels {nl}, strides {st}, head depth {rc['depth']}")
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
          f"ep{EPOCHS} seed{rc['seed']} | vs {rc['ctrl']:.2f} / m5095 {rc['ctrl_g']:.2f} | "
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
           "seed": rc["seed"], "batch": BATCH, "imgsz": IMG_SIZE, "epochs": EPOCHS,
           "tal_beta": rc["beta"], "detect_levels": 3, "strides": [4, 8, 16],
           "remap_tensors": n_moved, "pretrained_params": n_load,
           "pretrained_frac": round(frac_load, 4), "n_params": n_par,
           "loss_params": rc["params"], "ctrl": rc["ctrl"], "prior": rc["prior"],
           "hours": hours, "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "p234_depth_params.json"), "w") as f:
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
    print("=" * 92)
    print(f"  P234 DEPTH — the missing corner and the depth axis  ({len(todo)} runs)")
    print("=" * 92)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph did not build with 3 levels / strides [4, 8, 16].")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_p234_depth_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "results": res}, f, indent=2)

    print("\n" + "=" * 92)
    print("  P234 DEPTH — RESULTS (mAP50-95 only; SCORE mAP50_small BEFORE CONCLUDING)")
    print("=" * 92)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<22} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<22} depth {str(r['head_depth']):<8} beta {r['tal_beta']:<4} "
                  f"mAP50 {r['test_map50']*100:6.2f}  mAP50-95 {r['test_map5095']*100:6.2f}  "
                  f"{r['n_params']/1e6:5.2f}M  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print(f"  RUN 1  read against p234_b1 {P234B1_S50:.2f} on small AND against p234rich")
    print(f"         {RICH_5095:.2f} on mAP50-95. It only wins if it holds BOTH; beating one")
    print("         while losing the other is the trade this round exists to escape.")
    print(f"  RUN 2  compute the decomposition, not the level:")
    print(f"           P2-only = run2 - {P234_S50:.2f}     P3-only = {RICH_S50:.2f} - run2")
    print("  RUN 3  turning over is a RESULT. Report the peak, not the disappointment.")
    print(f"  ALL    watch AR50_95_small. Baseline {REF_AR:.2f}; every high-precision cell in")
    print("         the p2 grid fell to 66-68 and only p234rich held 70.92. Reaching 79.5")
    print("         small while holding AR above ~70 is a materially better result than")
    print("         reaching it by finding fewer objects more confidently.")
    print(f"  FLOOR  arch sd on small {SD_S50_ARCH:.3f} -> run-to-run SE 0.488. Every control")
    print("         here is n=1. Under ~1.0 is not resolvable. This round MAPS the axis.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
