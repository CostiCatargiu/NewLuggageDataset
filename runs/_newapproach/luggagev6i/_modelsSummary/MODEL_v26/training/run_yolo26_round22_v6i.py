#!/usr/bin/env python3
r"""
YOLO26 ROUND 22 — the alpha x beta plane at low beta, and combos moved off b3

Ten runs, ~9.0 GPU-h. NO PATCH REQUIRED — every key is already in default.yaml,
so this can start tonight. b82, stock yolo26s.pt, 640, 70 ep, matched to the
baseline in batch and graph.

PRIMARY METRIC IS mAP50_small. Do not judge this round on mAP50-95: y26_b3 reads
+3.8 SE on small and +0.0 on mAP50-95, and that metric already nearly cost this
project its largest finding.

    REFERENCE, 3 seeds of y26_identity  small 77.62  mAP50 80.40  m5095 55.30
    pooled sd (df=7)                    small  0.346  m5095 0.374

    THE DIVISOR IS AN SE, NOT THE SD. The reference is itself estimated, so a
    single run against a 3-seed mean has SE = sd*sqrt(1+1/3) = 0.400, and a run
    against ANOTHER SINGLE RUN has SE = sd*sqrt(2) = 0.489. Dividing by 0.346
    inflates every sigma by 15.5% and moved two round-21 runs out of "effect".

    vs the 3-seed reference:  2 SE = 0.80 suggestive | 3 SE = 1.20 an effect
    vs another single run  :  2 SE = 0.98 | 3 SE = 1.47


=============================================================================
WHAT ROUND 21 ESTABLISHED, AND WHY IT REOPENS EVERYTHING
=============================================================================
    beta      6      4      3     2.5      2     1.5      1
    small  77.62  78.17  79.14  78.88  78.92  78.81  79.30
    mAP50  80.40  80.95  81.31  81.43  81.56  81.47  81.80
    large  80.65  83.67  80.42  81.78  83.24  84.47  82.71

Across beta in [1,3] the spread is 0.48 = 0.98 SE run-to-run, and the sd of the
five points is 0.202, BELOW the 0.346 noise floor. That is a PLATEAU, not a
peak: beta <= 3 gives ~+1.4, beta = 4 gives +0.55, beta = 6 gives nothing. A
threshold, not an optimum.

READ EVERY DELTA AGAINST THE CURVE, NOT AGAINST THE POINT. Each ctrl below is a
SINGLE draw. y26_b3 = 79.14 sits ~+0.34 above what beta 2 / 2.5 / 4 interpolate
to at beta 3 — the same size as the bias already corrected on the reference. A
run landing at 78.9 prints as -0.24 against b3 and may in fact be level. This
caution predicted the plateau before round 21 ran; it applies again here.

The consequence for this round: ROUND 19 PUT TEN MECHANISMS ON beta = 3 AND ALL
TEN LOST — but beta = 3 turns out to be an arbitrary point on that plateau, and
beta = 1 and 2 gain on BOTH size axes where beta = 3 does not (b1 +2.15% small
AND +2.55% large; b3 +1.96% small but -0.28% large). Every combination in this
campaign was tested against the one plateau point that is worst on large.

Round 21 also refuted two things, which is why neither reappears here: the
beta_sel/beta_tgt split (target-only beta=3 lands BELOW baseline, so the terms
interact and coupling is the mechanism) and TCN (P and R both fell, so the
size-dependent target ceiling is load-bearing rather than a defect).


=============================================================================
ARM A — THE ALPHA x BETA PLANE AT LOW BETA (4 runs)
=============================================================================
Every alpha run in this project sits at beta = 6 or is tangled with SCB/SWA:
alpha075, a075_scb3_sbb50, a025_b4s2, a1_b6, a1_b6_o2o. The plane at low beta
is empty.

Selection is pure top-k over score^alpha * IoU^beta, hence monotone in
score^(alpha/beta) * IoU and invariant to the magnitudes. So a1_b2 (1.0/2.0)
selects EXACTLY the same anchors as b1 (0.5/1.0) — ratio 0.50 for both — while
producing different target magnitudes.

That is a direct replication of the a1_b6 vs b3 experiment at the other end of
the plateau. There it gave a 0.98 gap = 2.8 sd, which is the evidence that
magnitude matters independently of selection. If a1_b2 vs b1 reproduces it, the
claim holds at two ratios; if not, the a1_b6 result was a one-off and the paper
should say so.

    ratio tested so far: 0.083 (identity) .. 0.50 (b1, b4s1). a1_b1 is 1.00,
    beyond anything measured.


=============================================================================
ARM B — CLOSE THE BETA CURVE (2 runs)
=============================================================================
beta = 0.5 and beta = 0. At beta = 0 the alignment metric is score^alpha alone:
assignment IGNORES IoU ENTIRELY and becomes pure classification confidence.

If the plateau survives to 0.5 and only breaks at 0, that is a quotable claim
about what task-aligned assignment contributes on 60%-small imagery. If it
breaks at 0.5, the plateau has a floor and the curve is closed. Either way
nobody can ask "did you look lower".


=============================================================================
ARM C — ROUND 19'S BEST COMBOS, MOVED OFF beta = 3 (3 runs)
=============================================================================
                        on b3     what it did there
    sbb_q=0.5 inv       78.34     -0.81 on small, but L95 61.75 — the campaign's
                                  best large mAP50-95, above the baseline's 58.90
    o2m_final=0.3       78.79     -0.35, the least bad combo of the ten
    SWA 0.9->0.4        78.62     -0.52, though SWA is 7/7 positive on small
                                  across its own family

HONEST COUNTER-ARGUMENT, stated because it may well be right: round 19's lesson
was that these mechanisms move ALONG the small/large trade curve, and beta = 1
sits FURTHER along it than beta = 3. Stacking may overshoot worse, not better.
The reason to run them anyway is that b1 and b2 gained on BOTH axes, which means
they are not purely on that curve — so the geometry that explained the round-19
failures does not obviously apply.


=============================================================================
ARM D — ONE SEED (1 run)
=============================================================================
y26_b1 is the campaign maximum and it is n=1. It beats b3 by 0.16, which at the
run-vs-run SE of 0.489 is 0.31 SE — not a distinguishable difference by any
reading. One repeat gives a range; a second tomorrow gives an sd. Until then the
paper cannot say which config it reports.


=============================================================================
CALIBRATION AND EXECUTION ORDER
=============================================================================
Runs execute in descending PRIOR = P(clearing +0.35 on mAP50_small vs its own
control), declared per run so a queue that dies at 3am keeps what mattered.

    1  y26_a1_b2      0.32  magnitude test at ratio 0.50 — replicates a1_b6 vs b3
    2  y26_b05        0.28  the plateau's lower edge
    3  y26_a1_b1      0.26  ratio 1.00, beyond anything measured
    4  y26_b1_s1      0.00  no chance of improving; the headline is n=1
    5  y26_b1_sbb50   0.22  SBB owns the best large in the campaign; b1 has room
    6  y26_b2_swa     0.20  SWA is 7/7 on small and has never met beta=2
    7  y26_a075_b1    0.20  fills the alpha plane
    8  y26_b1_o2mf30  0.18  the least bad round-19 combo, moved off b3
    9  y26_a025_b1    0.15  the other side of the alpha plane
   10  y26_b0         0.10  IoU removed from assignment entirely

These are an ORDERING, not forecasts. Directional predictions in this campaign
are 0 for 12 and every miss was optimistic. Round 21's two useful results were
both surprises: beta=1 was expected to be past the peak, and the peak turned out
not to exist.

    Usage:
        python run_yolo26_round22_v6i.py                # all ten, prior order
        python run_yolo26_round22_v6i.py --preflight    # print the plan, run nothing
        python run_yolo26_round22_v6i.py --arm a        # a | b | c | d
        python run_yolo26_round22_v6i.py y26_a1_b2      # one by name

    FREE, DO IT FIRST:  python probe_topk_binding.py
    tal_topk is the last untouched assignment knob on v26, but it is NOT in
    default.yaml (E2ELoss hardcodes tal_topk=10 / 7), so testing it needs a
    patch. That probe answers whether it is even binding — if most small GTs
    hold fewer than 10 anchor centres, changing k cannot change their
    assignment and the patch is wasted work. It needs no GPU and no model.
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
PROJECT_DIR = "runs_yolo26_round22_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82                            # matches y26_identity and every loss run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False            # y26_p2k2_hi was lost to exist_ok=True

# 3-seed means. NEVER the seed-0 draw (small 77.30 is the low draw by 0.32).
REF_S50, REF_50, REF_5095 = 77.62, 80.40, 55.30
SD_S50 = 0.346
B1_S50, B2_S50, B3_S50 = 79.30, 78.92, 79.14

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    tal_beta_o2m=None, tal_beta_o2o=None, tal_alpha_o2m=None, tal_alpha_o2o=None,
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


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def _swa(start, end, boost=2.0, px=48):
    """Verbatim from run_yolo26_sweep_v6i.py, which produced y26_swa_a09_04.

    alpha_min/alpha_max are the ENDPOINTS (end, start), not 0.0/1.0 — a later
    helper in that file got this wrong and the two are not interchangeable.
    """
    return dict(alpha_start=start, alpha_end=end, alpha_min=end, alpha_max=start,
                area_weight_mode="sqrt", area_weight_norm="max",
                small_obj_px=px, small_obj_boost=boost)
# =============================================================================


RUNS = [
    # ------------------------------------------------------------------ ARM A
    {"name": "y26_a1_b2", "arm": "a", "prior": 0.32, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_alpha=1.0, tal_beta=2.0),
     "expect": {"alpha": 1.0, "beta": 2.0},
     "label": "alpha 1.0 / beta 2.0 — ratio 0.50, the same selection as y26_b1",
     "why": "Top-k over s^a * u^b is monotone in s^(a/b) * u, so ratio 0.50 here "
            "selects EXACTLY the anchors y26_b1 (0.5/1.0) selects, at different "
            "target magnitudes. This is the a1_b6-vs-b3 experiment repeated at the "
            "other end of the plateau: there it gave 0.98 = 2.8 sd, which is the "
            "whole basis for 'magnitude matters independently of selection'. Two "
            "ratios or one anecdote."},

    {"name": "y26_a1_b1", "arm": "a", "prior": 0.26, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_alpha=1.0, tal_beta=1.0),
     "expect": {"alpha": 1.0, "beta": 1.0},
     "label": "alpha 1.0 / beta 1.0 — align = score * IoU, ratio 1.00",
     "why": "The clean product, and the first point past the tested ratio range "
            "(0.083 to 0.50). If the plateau is really about the ratio, this "
            "should sit outside it; if it is about beta's absolute value, this "
            "should look like b1."},

    {"name": "y26_a075_b1", "arm": "a", "prior": 0.20, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_alpha=0.75, tal_beta=1.0),
     "expect": {"alpha": 0.75, "beta": 1.0},
     "label": "alpha 0.75 / beta 1.0 — ratio 0.75",
     "why": "Fills the plane between b1 (0.50) and a1_b1 (1.00). With three "
            "points at beta=1 the alpha axis is a curve rather than two dots."},

    {"name": "y26_a025_b1", "arm": "a", "prior": 0.15, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_alpha=0.25, tal_beta=1.0),
     "expect": {"alpha": 0.25, "beta": 1.0},
     "label": "alpha 0.25 / beta 1.0 — ratio 0.25",
     "why": "The low-alpha side. a025_b4s2 (ratio 0.125, with SCB) was the worst "
            "run of round 18 at 77.24, but that confounded alpha with SCB. This "
            "is the clean version."},

    # ------------------------------------------------------------------ ARM B
    {"name": "y26_b05", "arm": "b", "prior": 0.28, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_beta=0.5),
     "expect": {"beta": 0.5},
     "label": "tal_beta 0.5 — below the plateau's measured edge",
     "why": "The plateau runs 1 to 3 and nobody has been under 1. Half a step "
            "below the current best is the cheapest way to learn whether the "
            "floor is at 1 or further down."},

    {"name": "y26_b0", "arm": "b", "prior": 0.10, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_beta=0.0),
     "expect": {"beta": 0.0},
     "label": "tal_beta 0.0 — assignment ignores IoU entirely",
     "why": "The degenerate endpoint: align_metric = score^alpha, so positives are "
            "chosen on classification confidence alone and localisation plays no "
            "part in assignment. Expected to fail, and the failure is the point — "
            "it bounds how much of TAL's value is the IoU term on a dataset that "
            "is 60% small objects."},

    # ------------------------------------------------------------------ ARM C
    {"name": "y26_b1_sbb50", "arm": "c", "prior": 0.22, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_beta=1.0, sbb_q=0.5, sbb_invert=True),
     "expect": {"beta": 1.0, "sbb": 0.5},
     "label": "beta 1.0 + SBB q=0.5 inverted",
     "why": "On b3 this cost 0.81 on small but produced L95 61.75 — the highest "
            "large mAP50-95 in the campaign, above the baseline's 58.90. b1 "
            "already gains on large (+2.55%), so the question is whether the two "
            "compound or whether SBB's large-side push is redundant once beta "
            "has already delivered it."},

    {"name": "y26_b2_swa", "arm": "c", "prior": 0.20, "seed": SEED, "ctrl": B2_S50,
     "params": cfg(tal_beta=2.0, **_swa(0.9, 0.4)),
     "expect": {"beta": 2.0, "swa": True},
     "label": "beta 2.0 + SWA 0.9 -> 0.4",
     "why": "SWA is 7 for 7 above baseline on mAP50_small across its own family, "
            "the only multi-run support any mechanism here has, and it was filed "
            "as '-0.48' from a run at a DIFFERENT batch judged on mAP50-95. It has "
            "met beta=3 (-0.52) but never beta=2, which is the setting that gains "
            "on both size axes."},

    {"name": "y26_b1_o2mf30", "arm": "c", "prior": 0.18, "seed": SEED, "ctrl": B1_S50,
     "params": cfg(tal_beta=1.0, o2m_final=0.3),
     "expect": {"beta": 1.0, "blend": (0.8, 0.3, True)},
     "label": "beta 1.0 + one2many kept alive to 0.3 at the end",
     "why": "The least bad of round 19's ten combos on b3 (-0.35). The auxiliary "
            "branch is discarded at inference but shapes the shared backbone "
            "throughout; 0.1 was inherited from upstream and never tested on this "
            "dataset at any beta below 3."},

    # ------------------------------------------------------------------ ARM D
    {"name": "y26_b1_s1", "arm": "d", "prior": 0.00, "seed": 1, "ctrl": B1_S50,
     "params": cfg(tal_beta=1.0),
     "expect": {"beta": 1.0},
     "label": "y26_b1 repeated at SEED 1 — the headline is n=1",
     "why": "y26_b1 is the campaign maximum on both mAP50_small and mAP50, and it "
            "beats b3 by 0.16 = 0.5 sd, which is not a distinguishable difference. "
            "Prior is 0.00 by construction: this run cannot improve anything, it "
            "can only tell you whether the number is real. Section 6.4 of the "
            "literature review notes that in twenty years exactly ONE group in "
            "this field reported run-to-run variance."},
]


def preflight(todo):
    """No patch is needed this round, so this checks the CONFIG SURFACE only.

    Rounds 4-6 lost ten runs to a key that default.yaml accepted and the loss
    never read — but every key used here is already consumed by a shipped run
    (tal_alpha, tal_beta, sbb_q, o2m_final, the SWA block), so the consumer
    check is the epoch-2 guard rather than an import-time source scan.
    """
    print("=" * 78)
    print("  PREFLIGHT — round 22 needs no patch")
    print("=" * 78)
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False
    need = ["tal_alpha", "tal_beta", "sbb_q", "sbb_invert", "o2m_final",
            "alpha_start", "alpha_end", "area_weight_mode", "small_obj_px"]
    ok = True
    for k in need:
        good = k in DEFAULT_CFG_DICT
        print(f"  default.yaml accepts {k:<20} {good}")
        ok &= good
    if not ok:
        print("\n  [ABORT] a key is missing — this tree is not the one round 21 ran on.")
        return False
    print()
    for i, r in enumerate(todo, 1):
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  {i:>2}. {r['name']:<16} arm {r['arm']}  p={r['prior']:.2f}  seed {r['seed']}  "
              f"vs {r['ctrl']:.2f}")
        print(f"      {d}")
    print(f"\n  {len(todo)} runs, ~{0.9 * len(todo):.1f} GPU-h")
    print(f"  reference: small {REF_S50:.2f}  mAP50 {REF_50:.2f}  sd {SD_S50}")
    return True


def attach_guard(model, rc):
    """Assert at EPOCH 2 that the mechanism is live and nothing else is.

    Epoch 2, not epoch 1: the o2m blend is recomputed per epoch, so anything
    keyed on it still matches stock on the first pass. A guard that fires at
    epoch 1 passes for a config that silently reverts afterwards.
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
            a, b = br.assigner, br.bbox_loss
            if "alpha" in e and abs(float(a.alpha) - e["alpha"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected {e['alpha']}")
            if "beta" in e and abs(float(a.beta) - e["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            want_q = e.get("sbb", 0.0)
            if abs(float(getattr(b, "sbb_q", 0.0)) - want_q) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {want_q}")
            if e.get("swa"):
                if not b.swa_enabled():
                    raise RuntimeError(f"{rc['name']}: SWA requested but not live on {tag}")
            elif b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")
            # everything else provably off
            if a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB is live but was not requested")
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93 / -12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            if b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SNL1/NWD live on {tag}")
        if "blend" in e:
            got = (float(crit.o2m_copy), float(crit.final_o2m), bool(crit.o2m_decay))
            if abs(got[1] - e["blend"][1]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: blend {got}, expected {e['blend']}")
            seen.append(f"blend o2m {got[0]} -> {got[1]}")
        h = o2o.hyp   # E2ELoss has no .hyp; only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this file is the stock graph")
        a = o2o.assigner
        seen.insert(0, f"alpha={a.alpha} beta={a.beta} ratio={a.alpha/max(a.beta,1e-9):.3f} "
                       f"sbb_q={getattr(o2o.bbox_loss,'sbb_q',0.0)} swa={o2o.bbox_loss.swa_enabled()}")
        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    print("\n" + "=" * 78)
    print(f"  RUN {rc['name']}   [arm {rc['arm'].upper()}]   prior {rc['prior']:.2f}")
    print(f"  {rc['label']}")
    print("=" * 78)
    d = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} seed{rc['seed']} | vs {rc['ctrl']:.2f}")
    print(f"  {d}\n")
    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)
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
    out = {"name": rc["name"], "arm": rc["arm"], "prior": rc["prior"], "ctrl": rc["ctrl"],
           "params": rc["params"], "expect": {k: str(v) for k, v in rc["expect"].items()},
           "seed": rc["seed"], "batch": BATCH, "imgsz": IMG_SIZE, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "round22_params.json"), "w") as f:
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
    ap.add_argument("--arm", choices=["a", "b", "c", "d"])
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    todo = sorted(RUNS, key=lambda r: -r["prior"])
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 22 — alpha x beta plane, curve closure, combos off b3 "
          f"({len(todo)} runs)")
    print("=" * 84)
    if not preflight(todo) or a.preflight:
        return

    res, path = [], "runs_round22_v6i__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex)})
        with open(path, "w") as f:      # flush after EVERY run, not at the end
            json.dump({"batch": BATCH, "results": res}, f, indent=2)

    print("\n" + "=" * 84)
    print("  ROUND 22 — mAP50-95 only. Run analyze_runs_v6i.py for mAP50_small.")
    print("=" * 84)
    print(f"{'run':<20}{'arm':<5}{'seed':>5}{'mAP50':>9}{'mAP50-95':>11}{'hours':>7}")
    print("-" * 60)
    for r in sorted([x for x in res if x.get("test_map5095") == x.get("test_map5095")],
                    key=lambda x: -x["test_map5095"]):
        print(f"{r['name']:<20}{r['arm']:<5}{r['seed']:>5}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{r['hours']:>7.2f}")
    print(f"\n  reference (3 seeds): mAP50 {REF_50:.2f}  mAP50-95 {REF_5095:.2f}")
    print("\n  NOW RUN:  python analyze_runs_v6i.py")
    print("  Add y26_b1 / y26_b1_s1 to SEED_GROUPS first so the noise floor tightens.")
    print("  Section 2 ranks on mAP50_small; section 4 flags metric disagreement.")


if __name__ == "__main__":
    main()
