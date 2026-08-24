#!/usr/bin/env python3
r"""
ARCH x LOSS — does tal_beta do MORE on a P2 graph, and why

Four runs, ~5.0 GPU-h, 640, BATCH 48, graph = yolo26-p2dys.yaml on every run.
The GRAPH is held fixed and only the LOSS moves, which is the mirror image of
run_yolo26_arch_p234_v6i.py. Run that one first: it supplies the second seed of
this file's control.

    CONTROL   y26_remap_dys_stock   p2dys graph + STOCK loss   small 78.39
              y26_p2dys_ctrl_s1     the same, seed 1           <- from the p234 round
              -> run the p234 round first and this control is n=2 instead of n=1

    STOCK-GRAPH REFERENCE, for the differential:
              y26_identity  n=3     77.62      y26_b1   n=2   78.94   (+1.32)
                                               y26_b0   n=1   78.38   (-0.56 vs b1,
                                                                       NOT distinguishable)


=============================================================================
THE FINDING THIS ROUND IS BUILT ON
=============================================================================
The SAME loss mechanism (SCB beta_small=3 + SBB q=0.5 inverted) is worth wildly
different amounts depending on the graph underneath it:

    graph                    stock loss   + SCB/SBB   contribution
    stock 3-level  (n=3)        77.62       77.73        +0.11    <- a NULL
    p2    (remap)               77.63       78.88        +1.25
    p2dys (remap)               78.39       79.23        +0.84

The stock-graph row is n=3, so that null is solid. The two P2 rows are n=1 vs
n=1 at an arch run-to-run SE of ~0.488, so individually they are 2.6 and 1.7 SE
— suggestive, not established. What makes them worth acting on is that the
DIRECTION agrees across two independent graphs while the stock comparison is a
solid null.

WHY IT WOULD BE TRUE. From probe_topk_binding_v2.py, on the stock 3-level graph
a small GT holds ~12.90 anchor centres against topk=10. SLACK: every candidate
is already a positive, so the ranking cannot change the assignment and
tal_beta's SELECTION term is INERT. That is exactly why y26_b0 — beta=0, IoU
removed from the alignment metric entirely — cost nothing measurable, and why
beta in [0,3] is a flat plateau.

A stride-4 level changes the arithmetic. For a 12x18px GT:

    s4  13.50 centres     s8  3.38     s16  0.84     s32  0.21

so the small pool goes ~12.90 -> ~52 and topk=10 becomes BINDING at ~19%.

    ON A P2 GRAPH, SELECTION IS OPERATIVE. tal_beta should act through
    SELECTION *AND* TARGETS, not targets alone, and should therefore be
    worth MORE than the +1.32 it is worth on the stock graph.

That is a prediction that can be wrong, which is the point.


=============================================================================
THE RUNS
=============================================================================
1  y26_p2dys_b1     prior 0.40    beta=1.0 on p2dys. THE HEADLINE.
                                  Both ingredients are measured positive and on
                                  this graph the mechanism says they should be
                                  more than additive rather than substitutive.
                                  P2+DySample is +0.75 small over stock_b48;
                                  beta=1 is +1.32 over identity. If they merely
                                  add, this lands ~79.7. If selection really is
                                  live here, higher.

2  y26_p2dys_b1_s1  ----          Seed 1 of run 1. Arch sd on small is 0.345 and
                                  round 23 measured a -0.55 winner's curse on
                                  four selected maxima. A single draw of the
                                  campaign's new best number is worth nothing.

3  y26_p2dys_b0     prior 0.05    beta=0 on p2dys. THE FALSIFICATION TEST, and
                                  the most informative run in the file.
                                  On the STOCK graph, b0 scored 78.38 against
                                  b1's 78.94 — 1.2 SE, NOT DISTINGUISHABLE.
                                  Removing IoU from selection was free, because
                                  selection was doing nothing.
                                  If selection is live on p2dys, THE SAME CHANGE
                                  MUST HURT HERE. A b0/b1 gap on p2dys that is
                                  clearly larger than the 0.56 seen on the stock
                                  graph confirms the mechanism. A gap of the same
                                  size refutes it, and then runs 1-2 are just
                                  "two positives added" with no story.
                                  Prior 0.05 is P(it improves). Its VALUE is not
                                  its prior.

4  y26_p2dys_b3     prior 0.25    beta=3.0 on p2dys. Does the PLATEAU survive?
                                  On the stock graph beta in [0,3] is flat to
                                  0.21 — sixteen configs inside the noise floor —
                                  because only the target term was moving. If
                                  selection is live on p2dys the surface should
                                  acquire STRUCTURE: b1, b3 and b0 should spread
                                  out. Three points is the minimum that can show
                                  a shape rather than a level.

Everything else is stock: no SCB, no SBB, no SWA, no split, no TCN. The guard
asserts all of them OFF. y26_remap_dys already measured SCB+SBB on this graph
(79.23) and stacking more onto beta here would confound the one comparison this
round exists to make.


=============================================================================
HOW TO SCORE IT — WRITTEN BEFORE THE RUNS
=============================================================================
    the differential, not the level:

        stock graph   b1 - b0 = 78.94 - 78.38 = 0.56   (1.2 SE, not distinguishable)
        p2dys graph   b1 - b0 = ?

    IF the p2dys gap is clearly wider    -> selection is live on P2 graphs, the
                                            beta plateau is a stock-graph artifact,
                                            and that is a mechanism result worth
                                            more than the mAP number.
    IF the gap is the same               -> the mechanism story is wrong. Runs 1-2
                                            may still give the best number in the
                                            project; write them up as additive and
                                            say the explanation did not survive.

    Read run 1 against y26_remap_dys_stock (78.39), NOT against y26_b1 (78.94)
    and NOT against the P2+DySample n=10 mean (78.43, which is a DIFFERENT
    remap and a different graph family).

    Usage:
        python run_yolo26_archloss_p2dys_v6i.py                 # all four
        python run_yolo26_archloss_p2dys_v6i.py --preflight     # build, run nothing
        python run_yolo26_archloss_p2dys_v6i.py y26_p2dys_b1    # one by name
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
PROJECT_DIR = "runs_yolo26_archloss_p2dys_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 48                            # matches every arch run in this project
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_P2DYS = "ultralytics/cfg/models/26/yolo26-p2dys.yaml"
REMAP_SHIFT = 6
REMAP_4LVL = range(17, 23)            # p2dys: stock head rows 17-22 -> 23-28

# Controls
CTRL_S50 = 78.39                      # y26_remap_dys_stock, p2dys + STOCK loss, n=1
REMAP_DYS_SCBSBB = 79.23              # same graph + SCB/SBB, for context
# stock-graph reference, for the differential
STOCK_B1, STOCK_B0, STOCK_ID = 78.94, 78.38, 77.62
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
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    {"name": "y26_p2dys_b1", "order": 1, "prior": 0.40, "seed": 0,
     "params": cfg(tal_beta=1.0), "beta": 1.0, "ctrl": CTRL_S50,
     "label": "p2dys + beta 1.0 — the headline",
     "why": "P2+DySample +0.75 small, beta=1 +1.32. Merely additive lands ~79.7; "
            "if selection is live on this graph, higher."},

    {"name": "y26_p2dys_b1_s1", "order": 2, "prior": 0.00, "seed": 1,
     "params": cfg(tal_beta=1.0), "beta": 1.0, "ctrl": CTRL_S50,
     "label": "seed 1 of run 1",
     "why": "Arch sd on small is 0.345 and round 23 measured a -0.55 winner's "
            "curse on four selected maxima. One draw of a new best is worth nothing."},

    {"name": "y26_p2dys_b0", "order": 3, "prior": 0.05, "seed": 0,
     "params": cfg(tal_beta=0.0), "beta": 0.0, "ctrl": CTRL_S50,
     "label": "p2dys + beta 0 — THE FALSIFICATION TEST",
     "why": "On the stock graph b0 - b1 = -0.56, not distinguishable: removing IoU "
            "from selection was FREE because selection was inert. If selection is "
            "live here the same change MUST hurt. Prior 0.05 is P(improves); its "
            "value is not its prior."},

    {"name": "y26_p2dys_b3", "order": 4, "prior": 0.25, "seed": 0,
     "params": cfg(tal_beta=3.0), "beta": 3.0, "ctrl": CTRL_S50,
     "label": "p2dys + beta 3.0 — does the plateau survive a binding topk?",
     "why": "On the stock graph beta in [0,3] is flat to 0.21, sixteen configs "
            "inside the floor, because only the target term moved. With selection "
            "live the surface should acquire structure. Three points is the "
            "minimum that shows a shape rather than a level."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    print("=" * 84)
    print("  PREFLIGHT")
    print("=" * 84)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import DetectionModel
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False, set()
    print(f"  {'default.yaml accepts tal_beta':<44} {'tal_beta' in DEFAULT_CFG_DICT}")
    ok &= "tal_beta" in DEFAULT_CFG_DICT
    if not os.path.exists(CFG_P2DYS):
        print(f"  [ABORT] {CFG_P2DYS} not found")
        return False, set()
    try:
        m = DetectionModel(cfg=CFG_P2DYS, nc=3, verbose=False)
        det = m.model[-1]
        nl = int(getattr(det, "nl", -1))
        st = [int(s) for s in det.stride.tolist()]
        p = sum(x.numel() for x in m.parameters())
        m.eval()
        with torch.no_grad():
            m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        good = nl == 4 and st == [4, 8, 16, 32]
        ok &= good
        print(f"  {'p2dys builds':<44} params {p:,}  nl {nl}  strides {st}  "
              f"{'OK' if good else 'WRONG — expected nl 4, strides [4,8,16,32]'}")
        print(f"  {'stride-4 level present':<44} {4 in st}   "
              f"<- this is the whole premise of the round")
    except Exception as ex:
        print(f"  [ABORT] p2dys failed to build: {type(ex).__name__}: {ex}")
        return False, set()

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        tag = f"p={r['prior']:.2f}" if r["prior"] > 0 else "SEED"
        print(f"  RUN  {r['name']:<20} seed {r['seed']}  {tag:<7} vs {r['ctrl']:.2f}  {d}")
    print(f"\n  {len(todo)} runs, ~{1.25 * len(todo):.1f} GPU-h   b{BATCH}, graph FIXED at p2dys")
    print(f"\n  the differential to watch:  stock graph b1-b0 = "
          f"{STOCK_B1 - STOCK_B0:+.2f}  (1.2 SE, not distinguishable)")
    return ok, {r["name"] for r in todo}


# -------------------------------------------------------------------- remap --
def remap_pan(model, rows):
    """Copy stock head rows 17-22 into the p2 graph's 23-28.

    Without this the entire bottom-up PAN is randomly initialised — the
    "BROKEN-transfer graph" the p2dys yaml header refers to. Fixing it was worth
    +0.81 (y26_p2_b32 55.03 -> y26_p2_remap 55.84). The skip count is asserted:
    a partial copy prints a healthy-looking number and silently changes the run.
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
                           f"exactly what this assert exists to prevent")
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
    """beta is the ONLY thing that may differ from stock, and the graph must be P2.

    y26_remap_dys already measured SCB+SBB on this graph (79.23). Stacking any of
    it onto beta here would confound the single comparison the round exists for.
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
                    raise RuntimeError(f"{rc['name']}: {tag} {attr} live — beta is the ONLY "
                                       f"variable in this round")
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live")
            if a.scb_enabled() or a.snt_enabled() or a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: SCB/SNT/TSH live — y26_remap_dys already "
                                   f"measured SCB+SBB on this graph (79.23)")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0 or b.swa_enabled() \
                    or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB/SWA/SNL1/NWD live on {tag}")
        h = o2o.hyp
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal set — LB-TAL is CLOSED on v26")
        det = trainer.model.model[-1]
        nl = int(getattr(det, "nl", -1))
        st = [int(s) for s in det.stride.tolist()]
        if nl != 4 or st != [4, 8, 16, 32]:
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}; this round "
                               f"REQUIRES the stride-4 level — it is the premise")
        print(f"  [guard] beta={rc['beta']} and nothing else live")
        print(f"  [guard] detect levels {nl}, strides {st} — stride-4 present")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    print("\n" + "=" * 84)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 84)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  p2dys | b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{rc['seed']} | "
          f"vs {rc['ctrl']:.2f} (p2dys + stock loss) | {d}\n")
    t0 = time.time()
    model = YOLO(CFG_P2DYS)
    n_moved = remap_pan(model, REMAP_4LVL)
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
    # THE RECORDING FIX — see run_yolo26_arch_p234_v6i.py. No results JSON in this
    # repo records a batch, which is why y26_stock_b48 was mislabelled for weeks.
    out = {"name": rc["name"], "cfg": CFG_P2DYS, "seed": rc["seed"], "batch": BATCH,
           "imgsz": IMG_SIZE, "epochs": EPOCHS, "tal_beta": rc["beta"],
           "detect_levels": 4, "strides": [4, 8, 16, 32],
           "remap_tensors": n_moved, "params": rc["params"], "ctrl": rc["ctrl"],
           "prior": rc["prior"], "hours": hours, "weights": weights,
           "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "archloss_params.json"), "w") as f:
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
    print(f"  ARCH x LOSS — is tal_beta worth MORE on a P2 graph?  ({len(todo)} runs)")
    print("=" * 84)
    ok, runnable = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] preflight failed.")
        return
    todo = [r for r in todo if r["name"] in runnable]

    res, path = [], "runs_archloss_p2dys_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "cfg": CFG_P2DYS,
                       "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ARCH x LOSS — RESULTS (mAP50-95 only; SCORE mAP50_small BEFORE CONCLUDING)")
    print("=" * 84)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<20} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<20} beta {r['tal_beta']:<4} seed {r['seed']}  "
                  f"mAP50 {r['test_map50']*100:6.2f}  mAP50-95 {r['test_map5095']*100:6.2f}  "
                  f"{r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print()
    print("  HOW TO SCORE THIS ROUND")
    print(f"  control  y26_remap_dys_stock {CTRL_S50:.2f}  (p2dys + STOCK loss)")
    print(f"           NOT y26_b1 {STOCK_B1:.2f} and NOT the P2+DySample n=10 mean 78.43")
    print("  1. THE DIFFERENTIAL IS THE RESULT, not the level:")
    print(f"       stock graph  b1 - b0 = {STOCK_B1 - STOCK_B0:+.2f}   (1.2 SE, not distinguishable)")
    print("       p2dys graph  b1 - b0 = ?")
    print("     clearly wider -> selection is live on P2 graphs and the beta plateau")
    print("     is a stock-graph artifact. That is worth more than the mAP number.")
    print("     the same -> the mechanism story is wrong; write runs 1-2 up as")
    print("     additive and say the explanation did not survive.")
    print("  2. b1/b3/b0 spreading out is the same signal from the other direction:")
    print("     on the stock graph those three span 0.56, inside the floor.")
    print("  3. run 2 is the seed. A single draw of a new campaign maximum is worth")
    print("     nothing — round 23 regressed four selected maxima by -0.55 each.")
    print("  REMINDER: the primary metric is mAP50_small and it is NOT printed above.")


if __name__ == "__main__":
    main()
