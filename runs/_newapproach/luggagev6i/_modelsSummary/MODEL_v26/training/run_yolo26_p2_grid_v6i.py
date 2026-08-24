#!/usr/bin/env python3
r"""
P2 GRID — two graphs x the beta axis, NO SEED RUNS

Six runs, ~7 GPU-h, 640, BATCH 48. One config each, seed 0, nothing repeated.
This is an EXPLORATION round: it maps the cells, it does not establish any of
them. Seed whatever survives afterwards — and it will need seeding, because the
arch replicate sd on mAP50_small is 0.345 and round 23 measured a -0.55
winner's curse on four selected maxima.

Supersedes run_yolo26_arch_p234_v6i.py and run_yolo26_archloss_p2dys_v6i.py,
which are the same experiment split in two with seeds included.


=============================================================================
THE 2x2 THAT RUNS 1-3 COMPLETE
=============================================================================
                    STOCK LOSS              tal_beta = 1.0
    p2dys (4 lvl)   78.39  ALREADY HAVE     run 1
                    y26_remap_dys_stock
    p234  (3 lvl)   run 2                   run 3

One existing number plus three runs closes a full factorial. Runs 4-6 add the
beta shape and the capacity variant. If the queue dies after run 3 you still
have an attributable 2x2, which is why they are ordered this way rather than by
prior.


=============================================================================
WHY THIS ROUND EXISTS: THE SAME LOSS IS WORTH 10x MORE ON A P2 GRAPH
=============================================================================
SCB(beta_small=3) + SBB(q=0.5 inv), identical settings, three graphs:

    graph                  stock loss   + SCB/SBB   contribution
    stock 3-level (n=3)       77.62       77.73       +0.11    <- a NULL
    p2    (remap)             77.63       78.88       +1.25
    p2dys (remap)             78.39       79.23       +0.84

The stock row is n=3, so that null is solid. The P2 rows are n=1 vs n=1 at an
arch run-to-run SE of ~0.488, i.e. 2.6 and 1.7 SE — suggestive, not established.
What makes them worth acting on is that the DIRECTION agrees on two independent
graphs while the stock comparison is a solid null.

THE MECHANISM. probe_topk_binding_v2 measures that on the stock 3-level graph a
small GT holds ~12.90 anchor centres against topk=10. SLACK: every candidate is
already a positive, the ranking cannot change the assignment, and tal_beta's
SELECTION term is INERT. That is why y26_b0 (beta=0, IoU removed from the
alignment metric entirely) cost nothing measurable, and why beta in [0,3] is a
flat plateau spanning just 0.21 across sixteen configurations.

A stride-4 level changes the arithmetic. For a 12x18px GT:

    s4  13.50 centres    s8  3.38    s16  0.84    s32  0.21

so the small pool goes ~12.90 -> ~52 and topk=10 becomes BINDING at ~19%.

    ON A P2 GRAPH SELECTION IS OPERATIVE, so tal_beta acts through SELECTION
    AND TARGETS rather than targets alone, and should be worth MORE than the
    +1.32 it is worth on the stock graph.

Run probe_topk_binding_v2.py first. It now prints the three stride sets and
takes seconds on CPU. If the stride-4 rows do not come back BINDING, the premise
of runs 1, 3, 4 and 5 is wrong and this round should be rethought before it
costs seven hours.


=============================================================================
AND WHY P5 GOES: IT EARNS 0.13 POSITIVES PER GT
=============================================================================
diag_anchor_footprin_results.txt finding F1, "THE P5 BUDGET IS LARGELY FICTION":

    level                       s8      s16      s32
    pool < topk               36.2%    77.8%    96.2%
    pool == 0                  0.2%     6.3%    31.1%
    positives to small GTs     6.64     1.08     0.01
    positives to medium GTs    6.48     3.39     0.00
    positives to large GTs     0.16     7.96     1.68
    SEL BIAS                   0.94     1.42     0.34   <- METRIC-BIASED

Large is 7.7% of the dataset and F5 shows large objects live on P4 (7.96 of
9.79), so P5 earns ~0.13 positives per GT across the set while rows 7-10 and 28
are the most expensive stage in the network. SEL BIAS 0.34 says s32 is METRIC-
starved, which a per-level budget cannot repair — hence LB-TAL's six failures.

The p234 graphs keep the P5 BACKBONE (rows 7-10, incl. SPPF and C2PSA) and drop
only the DETECTION branch, so this separates "P5 detection is useless" from
"P5 context is useless". Rows 0-25 are byte-identical to yolo26-p2dys.yaml.


=============================================================================
THE SIX
=============================================================================
1  y26_p2dys_b1        0.40  beta=1 on p2dys. THE HEADLINE. P2+DySample is +0.75
                             small over stock_b48, beta=1 is +1.32 over identity.
                             Merely additive lands ~79.7; if selection is live
                             here, higher.
2  y26_p2dys_p234      0.35  Drop P5 detection, STOCK loss. One variable against
                             y26_remap_dys_stock. Expected: mAP FLAT at 30-40%
                             fewer parameters — a result, but not a bigger number.
3  y26_p2dys_p234_b1   0.35  Both. The cell that could be the config the paper
                             reports: if it matches run 1 at 30-40% fewer
                             parameters, that is the better result of the two.
                             Two variables, but runs 1 and 2 make it attributable.
4  y26_p2dys_b0        0.05  beta=0 on p2dys. THE FALSIFICATION TEST, and the
                             most informative run here. On the STOCK graph
                             b0 - b1 = -0.56, 1.2 SE, NOT distinguishable —
                             removing IoU from selection was FREE because
                             selection was inert. If selection is live on p2dys
                             THE SAME CHANGE MUST HURT. Its prior is P(improves);
                             its value has nothing to do with its prior.
5  y26_p2dys_b3        0.25  Does the plateau survive a binding topk? On the
                             stock graph b0/b1/b3 span 0.56, inside the floor,
                             because only the target term moved. With selection
                             live the surface should acquire STRUCTURE.
6  y26_p2dys_p234rich  0.25  Drop P5 AND spend the freed capacity on rows 19/22
                             (repeats 2->4, the P2/4 and P3/8 branches). DEPTH,
                             not width: width was tried three times (p2_wide
                             55.53, wide_starve 55.46, p2addw_base 55.06) and
                             all three landed at or below the matched control.
                             Only interpretable if run 2 holds.


=============================================================================
HOW TO SCORE IT — WRITTEN BEFORE THE RUNS
=============================================================================
CONTROL IS y26_remap_dys_stock = 78.39 (p2dys + STOCK loss, n=1).
NOT y26_b1 (78.94, a different graph).
NOT the P2+DySample n=10 mean (78.43, a different remap and graph family).
Getting that wrong flatters or buries every number here by half a point.

THE DIFFERENTIAL IS THE RESULT, not the level:

    stock graph   b1 - b0 = 78.94 - 78.38 = +0.56   (1.2 SE, not distinguishable)
    p2dys graph   b1 - b0 = ?

    clearly wider  -> selection is live on P2 graphs, the beta plateau is a
                      stock-graph artifact, and that is worth more to the paper
                      than the mAP number.
    the same       -> the mechanism story is wrong. Runs 1 and 3 may still be the
                      best numbers in the project; write them up as additive and
                      say the explanation did not survive.

EVERY NUMBER HERE IS n=1. The arch sd on mAP50_small is 0.345, so nothing under
~1.0 is resolvable from a single draw, and the best cell will be inflated by
roughly the winner's curse round 23 measured (-0.55 on four selected maxima).
Do not put any of these in a table before seeding them.

REPORT PARAMETERS NEXT TO mAP for runs 2, 3 and 6. Flat mAP at 30-40% fewer
parameters is the expected outcome for the p234 graphs and is a result in its
own right; do not write it up as a loss.

mAP50_large has a pooled sd of 2.33. Write "no detectable change", never
"no change".

    Usage:
        python run_yolo26_p2_grid_v6i.py                    # all six
        python run_yolo26_p2_grid_v6i.py --preflight        # build graphs, run nothing
        python run_yolo26_p2_grid_v6i.py --arm 2x2          # runs 1-3 only
        python run_yolo26_p2_grid_v6i.py y26_p2dys_b1       # one by name
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
WEIGHTS = "yolo26s.pt"                # stock, the PAN remap source
PROJECT_DIR = "runs_yolo26_p2_grid_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 48                            # matches every arch run in this project
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0                              # ONE SEED. No repeats in this round.
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_DIR = "ultralytics/cfg/models/26"
CFG_P2DYS = f"{CFG_DIR}/yolo26-p2dys.yaml"
CFG_P234 = f"{CFG_DIR}/yolo26-p2dys-p234.yaml"
CFG_P234RICH = f"{CFG_DIR}/yolo26-p2dys-p234rich.yaml"

# yolo26-p2* pushes stock head rows down by 6. A 4-level head has destinations
# for six rows; a 3-level head only for three, because 26-28 do not exist.
# The inherited remap_pan() guarded with `if k2 in sd`, so the wrong range
# copies three, skips three, and prints a healthy-looking count. Asserted below.
REMAP_SHIFT = 6
REMAP_4LVL = range(17, 23)            # -> 23-28
REMAP_3LVL = range(17, 20)            # -> 23-25

CTRL_S50 = 78.39                      # y26_remap_dys_stock: p2dys + STOCK loss, n=1
STOCK_B1, STOCK_B0 = 78.94, 78.38     # stock 3-level graph, for the differential
SD_S50_ARCH = 0.345                   # arch replicate sd on mAP50_small

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

G4 = dict(nl=4, strides=[4, 8, 16, 32], remap=REMAP_4LVL)
G3 = dict(nl=3, strides=[4, 8, 16], remap=REMAP_3LVL)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_p2dys_b1", "order": 1, "prior": 0.40, "arm": "2x2",
     "cfg": CFG_P2DYS, **G4, "beta": 1.0, "params": cfg(tal_beta=1.0),
     "label": "p2dys + beta 1.0 — THE HEADLINE",
     "why": "P2+DySample +0.75 small, beta=1 +1.32. Merely additive lands ~79.7; "
            "if selection is live on this graph, higher."},

    {"name": "y26_p2dys_p234", "order": 2, "prior": 0.35, "arm": "2x2",
     "cfg": CFG_P234, **G3, "beta": 6.0, "params": cfg(),
     "label": "drop P5 detection, STOCK loss — one variable vs 78.39",
     "why": "P5 earns ~0.13 positives per GT (F1) and large objects live on P4 "
            "(F5). Rows 0-25 byte-identical to p2dys. Expected FLAT at 30-40% "
            "fewer parameters — a result, not a bigger number."},

    {"name": "y26_p2dys_p234_b1", "order": 3, "prior": 0.35, "arm": "2x2",
     "cfg": CFG_P234, **G3, "beta": 1.0, "params": cfg(tal_beta=1.0),
     "label": "drop P5 + beta 1.0 — the cell that could be the reported config",
     "why": "If this matches run 1 at 30-40% fewer parameters it is the better "
            "result of the two. Two variables, made attributable by runs 1 and 2."},

    {"name": "y26_p2dys_b0", "order": 4, "prior": 0.05, "arm": "beta",
     "cfg": CFG_P2DYS, **G4, "beta": 0.0, "params": cfg(tal_beta=0.0),
     "label": "p2dys + beta 0 — THE FALSIFICATION TEST",
     "why": "On the stock graph b0 - b1 = -0.56, not distinguishable: removing IoU "
            "from selection was FREE because selection was inert. If selection is "
            "live here the same change MUST hurt. Prior is P(improves); its value "
            "has nothing to do with its prior."},

    {"name": "y26_p2dys_b3", "order": 5, "prior": 0.25, "arm": "beta",
     "cfg": CFG_P2DYS, **G4, "beta": 3.0, "params": cfg(tal_beta=3.0),
     "label": "p2dys + beta 3.0 — does the plateau survive a binding topk?",
     "why": "On the stock graph b0/b1/b3 span 0.56, inside the floor, because only "
            "the target term moved. With selection live the surface should acquire "
            "structure. Three points is the minimum that shows a shape."},

    {"name": "y26_p2dys_p234rich", "order": 6, "prior": 0.25, "arm": "arch",
     "cfg": CFG_P234RICH, **G3, "beta": 6.0, "params": cfg(),
     "label": "drop P5 + rows 19/22 repeats 2->4 — spend the freed capacity on P2/P3",
     "why": "Depth, not width: width was tried three times and answered. Only "
            "interpretable if run 2 holds."},
]


def order(runs):
    """By the explicit order key: the 2x2 first, so a queue that dies early still
    leaves an attributable factorial rather than three unanchored numbers."""
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    """BUILD every graph. Levels, strides, forward pass, parameter counts.

    Rounds 4-6 passed a preflight that only checked the config surface and lost
    ten runs to a flag nothing read. This one builds the thing.
    """
    print("=" * 88)
    print("  PREFLIGHT — building each graph")
    print("=" * 88)
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
        if c == CFG_P2DYS:
            base = p
        good = nl == rc["nl"] and st == rc["strides"]
        ok &= good
        if 4 not in st:
            print(f"  [ABORT] {rc['name']}: no stride-4 level — that is the premise")
            return False, set()
        print(f"  {rc['name']:<22} {os.path.basename(rc['cfg']):<30} "
              f"params {p:>11,}  nl {nl}  {str(st):<18} "
              f"{'OK' if good else 'WRONG, want nl=' + str(rc['nl'])}")

    if base:
        print()
        for c, (p, nl, st) in seen.items():
            d = 100.0 * (p - base) / base
            flag = ("  <- ABOVE p2dys: NOT capacity-neutral, relabel it"
                    if c == CFG_P234RICH and p > base else "")
            print(f"  {os.path.basename(c):<32} {p:>11,}  {d:+6.1f}% vs p2dys{flag}")

    print()
    for rc in todo:
        d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  RUN  {rc['name']:<22} arm {rc['arm']:<5} p={rc['prior']:.2f}  "
              f"seed {SEED}  remap {rc['remap'].start}-{rc['remap'].stop - 1}"
              f"->{rc['remap'].start + REMAP_SHIFT}-{rc['remap'].stop - 1 + REMAP_SHIFT}  "
              f"{d if d else 'STOCK LOSS'}")
    print(f"\n  {len(todo)} runs, ~{1.2 * len(todo):.1f} GPU-h, b{BATCH}, ONE SEED, no repeats")
    print(f"  control y26_remap_dys_stock {CTRL_S50:.2f}   |   "
          f"stock-graph differential b1-b0 = {STOCK_B1 - STOCK_B0:+.2f}")
    return ok, {r["name"] for r in todo}


# -------------------------------------------------------------------- remap --
def remap_pan(model, rows):
    """Copy stock head rows into the positions this graph shifts them to.

    Fixing the transfer was worth +0.81 (y26_p2_b32 55.03 -> y26_p2_remap 55.84);
    without it the whole bottom-up PAN is randomly initialised. The skip count is
    asserted because a partial copy looks healthy in the log.
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
                           f"{rows.start}-{rows.stop - 1}; the range does not match this "
                           f"graph and a silent partial copy is what this assert prevents")
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
    """beta and the GRAPH are the only things allowed to vary. Everything else off.

    y26_remap_dys already measured SCB+SBB on p2dys (79.23). Any of it left live
    would confound the comparison this round exists to make, and nothing in the
    logs would say so.
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
                raise RuntimeError(f"{rc['name']}: SCB/SNT/TSH live — y26_remap_dys already "
                                   f"measured SCB+SBB on this graph (79.23)")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0 or b.swa_enabled() \
                    or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB/SWA/SNL1/NWD live on {tag}")
        if bool(getattr(o2o.hyp, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal set — LB-TAL is CLOSED on v26")
        det = trainer.model.model[-1]
        nl, st = int(getattr(det, "nl", -1)), [int(s) for s in det.stride.tolist()]
        if nl != rc["nl"] or st != rc["strides"]:
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}, expected "
                               f"{rc['nl']} / {rc['strides']} — wrong graph")
        if 4 not in st:
            raise RuntimeError(f"{rc['name']}: no stride-4 level; that is the premise")
        print(f"  [guard] beta={rc['beta']}, alpha=0.5, nothing else live")
        print(f"  [guard] detect levels {nl}, strides {st} — stride-4 present")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    print("\n" + "=" * 88)
    print(f"  RUN {rc['name']}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 88)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  {os.path.basename(rc['cfg'])} | b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} "
          f"seed{SEED} | vs {CTRL_S50:.2f} | prior {rc['prior']:.2f} | "
          f"{d if d else 'STOCK LOSS'}\n")
    t0 = time.time()
    model = YOLO(rc["cfg"])
    n_moved = remap_pan(model, rc["remap"])
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=SEED,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    n_par = sum(p.numel() for p in model.model.parameters())
    # THE RECORDING FIX. No results JSON in this repo records a batch — params and
    # run_meta are empty in every one and no args.yaml was saved. That gap is why
    # y26_stock_b48 sat mislabelled as b32 for weeks and why 42 arch runs were
    # uninterpretable until the batch was confirmed by hand.
    out = {"name": rc["name"], "arm": rc["arm"], "cfg": rc["cfg"], "seed": SEED,
           "batch": BATCH, "imgsz": IMG_SIZE, "epochs": EPOCHS,
           "tal_beta": rc["beta"], "detect_levels": rc["nl"], "strides": rc["strides"],
           "remap_rows": [rc["remap"].start, rc["remap"].stop - 1],
           "remap_tensors": n_moved, "n_params": n_par, "loss_params": rc["params"],
           "ctrl": CTRL_S50, "prior": rc["prior"], "hours": hours, "weights": weights,
           "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "p2_grid_params.json"), "w") as f:
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
    ap.add_argument("--arm", choices=["2x2", "beta", "arch"])
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    todo = RUNS
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    todo = order(todo)
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 88)
    print(f"  P2 GRID — two graphs x the beta axis  ({len(todo)} runs, ONE SEED, no repeats)")
    print("=" * 88)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph did not build with the expected levels/strides.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_p2_grid_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "seed": SEED,
                       "results": res}, f, indent=2)

    print("\n" + "=" * 88)
    print("  P2 GRID — RESULTS (mAP50-95 only; SCORE mAP50_small BEFORE CONCLUDING)")
    print("=" * 88)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<22} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<22} nl {r['detect_levels']}  beta {r['tal_beta']:<4} "
                  f"mAP50 {r['test_map50']*100:6.2f}  mAP50-95 {r['test_map5095']*100:6.2f}  "
                  f"{r['n_params']/1e6:5.2f}M  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print(f"  CONTROL is y26_remap_dys_stock {CTRL_S50:.2f} (p2dys + STOCK loss).")
    print(f"    NOT y26_b1 {STOCK_B1:.2f} — different graph.")
    print("    NOT the P2+DySample n=10 mean 78.43 — different remap and graph family.")
    print("  1. THE 2x2 (runs 1-3 + the 78.39 control) is the attributable part.")
    print("  2. THE DIFFERENTIAL IS THE RESULT, not the level:")
    print(f"       stock graph  b1 - b0 = {STOCK_B1 - STOCK_B0:+.2f}  (1.2 SE, not distinguishable)")
    print("       p2dys graph  b1 - b0 = ?    wider => selection is live on P2 graphs")
    print("  3. EVERY NUMBER HERE IS n=1 and the arch sd on small is 0.345. Nothing")
    print("     under ~1.0 is resolvable, and the best cell is inflated by roughly")
    print("     the -0.55 winner's curse round 23 measured. SEED BEFORE TABULATING.")
    print("  4. Report parameters next to mAP for the p234 runs. Flat mAP at 30-40%")
    print("     fewer parameters is the expected outcome and IS a result.")
    print("  5. mAP50_large sd is 2.33 — write 'no detectable change'.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
