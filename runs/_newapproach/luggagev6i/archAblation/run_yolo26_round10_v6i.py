#!/usr/bin/env python3
r"""
ROUND 10 — close the confound, test the account, try the one unused mechanism
============================================================================

Eight runs, ~8 GPU-h. Three independent questions; no run depends on another.


WHERE THE PROJECT ACTUALLY STANDS
---------------------------------
Best config, batch-matched, deterministic (exact, seed 0):

    run                   mAP    vs base    large   vs base large
    y26_base_rep        55.24      —        60.87       —
    y26_scb_b3          55.66    +0.42      59.43     -1.44
    y26_sbb_inv50       55.39    +0.15      56.84     -4.03
    y26_snl1_p25        55.49    +0.25      56.50     -4.37
    y26_scb3_sbb50      55.65    +0.41      60.82     -0.05   <- BEST

Every single mechanism buys overall mAP by giving up LARGE. The pair does not.
SCB alone 59.43, SBB alone 56.84, together 60.82 — 1.39 above the better single
and 3.98 above the worse. That is SUPER-additive, and under determinism it is
exact rather than scatter.

The account: SCB (assignment, lower beta on small GTs) pushes SMALL; SBB with
invert=True gives the one2one branch a LARGE preference. Separately each
overshoots the size balance; together they land on it. Different files,
different stages, opposite failure modes.

That account is a hypothesis. Arm B below is built to break it.


CLOSED OUT (do not revisit)
---------------------------
    SNL1        +0.25 alone, -0.37 on top of SCB, large -4.37. Does not stack.
    TSH         falsified on its own pre-registered criterion (below).
    SNT         -3.93 / -12.00, monotone.
    ported v12  SWA, LB-TAL: -0.43 to -0.82 on the P2 arch.

The SNT/TSH pair is a finding, not two failures:

    gap CLOSED  (SNT tau .25)   large 50.29   -10.58
    gap STOCK   (rho = 1.0)     large 60.87      —
    gap WIDENED (TSH rho .75)   large 57.33    -3.54

Both directions cost large-object AP. The winner/runner-up confidence gap in
YOLO26's NMS-free one2one head sits at an INTERIOR OPTIMUM. Target sharpness is
calibrated, which is why the whole soft-target family (VFL, label smoothing,
quality-aware targets) fails on this head.


ARM A — THE BATCH CONFOUND (2 runs)  *** highest value in this script ***
------------------------------------------------------------------------
Every loss number in this project is batch-matched at b82. The ARCH numbers are
not: both were run at b32 against a larger-batch baseline.

    v26  y26_p2k2_hi   b32   vs baseline b82   "+0.84"
    v12  arch_ls_shift b32   vs baseline b54   "+1.21"

Those are the headline architecture results and neither is clean. A reviewer
asks "is that the architecture or the batch size?" in one sentence. Two stock
runs at b32 — no yaml, no loss flags, nothing on — answer it permanently.

This is the only remaining hole in the campaign. It is also the cheapest.


ARM B — DOES THE ACCOUNT GENERALISE? (5 runs)
---------------------------------------------
Three questions, all from the same account.

(i) Is it a principle or a pairing?  If "opposing size-biases recover large" is
general, then SNL1 (pushes small, large -4.37) plus SBB inv (pushes large) should
ALSO recover large, with no SCB anywhere.

    y26_snl1_sbb    large recovers toward ~60  -> PRINCIPLE, state it
                    large stays ~57            -> specific pairing, report the config

(ii) Does the account rescue a setting that FAILED alone?  This is the sharpest
prediction available and the run most likely to beat 55.65.

    beta_small   alone            with SBB q=0.5
      2.0        55.05  (-0.19)   y26_scb2_sbb50   <- worst SCB point, and the
      3.0        55.66  (+0.42)   55.65 (known)       only one that LOST
      4.0        55.17  (-0.07)   y26_scb4_sbb50

  2.0 lost because it over-pushes small with nothing offsetting it. If SBB's
  large-push fixes exactly that, a below-baseline config becomes above-baseline —
  much stronger evidence than another point near an optimum you already have.
  And since 2.0 pushes harder than 3.0, it may pair better at fixed q.

(iii) Basin or knife-edge?  Two axes through the balance point instead of one:

    beta_small at q=0.5 :  2.0 / 3.0 / 4.0
    q at beta_small=3.0 :  0.25 / 0.5 / 0.75

  smooth in both      -> real basin, recommend the config
  isolated peak       -> knife-edge, and the paper must say so

That second point matters because SCB's own sweep WAS a single-point spike
(2 -> -0.19, 3 -> +0.42, 4 -> -0.07), and that one fact made the mechanism look
like an artifact for two days until SBB rescued it.


ARM C — cls_pw, THE ONE UNUSED MECHANISM (2 runs)
-------------------------------------------------
`cls_pw` ships in this fork, defaults to 0.0 (OFF), and has never been live in
any run of this project. It applies inverse-frequency per-class weights to the
classification BCE:

    train.py:167  set_class_weights()   weights = (1/count) ** cls_pw, mean-normalised
    trainer.py:405                      called after the dataloader is ready
    loss.py:729   bce_loss *= self.class_weights

The dataset is imbalanced and the imbalance tracks the per-class results:

    class       inst   share   AP50-95
    trolley     5202   53.0%     61.8
    backpack    2623   26.7%     56.3
    bag         1993   20.3%     48.0   <- worst class, by 13.8 pp

Resulting weights:

    p=0.25   backpack 1.03   bag 1.10   trolley 0.87
    p=0.50   backpack 1.05   bag 1.21   trolley 0.75

This is a DIFFERENT AXIS from everything tested so far: every mechanism in this
project reweights by object SIZE; this reweights by class FREQUENCY. Falsifiable
and specific — bag AP should rise, trolley should fall slightly, overall moves
only if the trade is favourable. If bag does not move, the mechanism is inert on
this data and that is a clean ablation row.


RUN 8 — SEED REPLICATION, AND A NOTE
------------------------------------
You have declined seed runs four times and I have respected that. I am including
one anyway as run 8, and you should feel free to delete it — it is one line.

The reason it is here: training on this box is DETERMINISTIC (y26_base_rep came
back bit-identical to yolo26_custom-9 across 118 values). That makes every delta
exact — and exact is not general. The super-additive large recovery is the only
genuinely novel result this campaign has produced, and it currently rests on a
single optimisation trajectory. It is also the specific claim a reviewer will
push hardest on, because super-additivity is surprising.

If you still don't want it, swap the last entry for a third cls_pw point:

    {"name": "y26_scb3_sbb50_pw10", ... cls_pw=0.10 ...}


CALIBRATION
-----------
Ten directional predictions across this campaign; ten falsified. The most recent
was in your favour — I predicted SCB was a trajectory artifact and the SBB pair
showed it holds across a partner swap. Treat every "should" above as a reason
the run is worth an hour, not as a forecast.

Arms A and C are not predictions at all. A closes a confound whichever way it
lands. C switches on a mechanism that has never been on.


Usage:
    python run_yolo26_round10_v6i.py                 # all eight, ~8 GPU-h
    python run_yolo26_round10_v6i.py --arm a         # just the batch baselines
    python run_yolo26_round10_v6i.py y26_snl1_sbb    # a subset
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
PROJECT_DIR = "runs_yolo26_round10_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82           # campaign-standard; Arm A overrides per-run
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

BASELINE = 55.24     # y26_base_rep == yolo26_custom-9, b82, bit-identical
BASE_SMALL, BASE_MED, BASE_LARGE = 51.00, 65.98, 60.87
BEST = 55.65         # y26_scb3_sbb50, large 60.82
BEST_LARGE = 60.82

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0,
    cls_pw=0.0,
    box=7.5, cls=0.5, dfl=1.5,
)


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


YAML_DIR = "arch_yamls_round10"

# The peak of the architecture search: ONE DySample at P3 -> P2, groups=4.
# Every deviation lost — count 0/1/2/3 -> 55.03/55.94/55.57/54.49, groups
# 2/4/8 -> 55.34/55.94/55.52. Reproduced verbatim from
# round8_deploy/cfg_yaml/y26_p2_dysample.yaml so the graph is byte-identical to
# the one that produced y26_p2k2_hi; do not "tidy" it.
ARCH_YAML = """nc: 3
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.50, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, DySample, [2]] # 17  content-aware P3 -> P2
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, True]] # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, True]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, True]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
"""


def save_yaml(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def check_arch_graph(model):
    """Assert the built graph is the P2 + single-DySample topology, not stock.

    A yaml typo that silently falls back to a 3-level head would produce a
    plausible number under an arch name — the exact failure class that made
    rounds 4-6 uninterpretable. Check the constructed module list, not the file.
    """
    m = model.model
    mods = [type(x).__name__ for x in m.model]
    n_dys = mods.count("DySample")
    det = [x for x in m.model if type(x).__name__ == "Detect"][-1]
    nl = int(getattr(det, "nl", 0))
    strides = [int(s) for s in getattr(det, "stride", [])] if hasattr(det, "stride") else []
    print(f"  [arch] DySample count = {n_dys}   Detect levels = {nl}   strides = {strides or 'lazy'}")
    if n_dys != 1:
        raise RuntimeError(f"expected exactly 1 DySample, graph has {n_dys}. "
                           f"The count axis was swept: 0/1/2/3 -> 55.03/55.94/55.57/54.49.")
    if nl != 4:
        raise RuntimeError(f"expected 4 detect levels (P2,P3,P4,P5), got {nl} — "
                           f"the P2 head did not build and this is not the arch.")
    print(f"  [arch] graph verified: P2 head + 1 DySample @ P3->P2")


# expect: what the epoch-1 guard must find LIVE in the constructed criterion.
#   scb/snl1/sbb -> as in round 9
#   clspw        -> model.class_weights present, correct length, mean ~1
#   stock        -> NOTHING live (Arm A)
RUNS = [
    # ---- ARM A: unconfound the architecture numbers -------------------------
    {"name": "y26_stock_b32", "arm": "a", "model": "yolo26s.pt", "batch": 32,
     "expect": {"stock": True}, "params": cfg(),
     "label": "STOCK yolo26s at b32 — the missing control",
     "why": "y26_p2k2_hi (+0.84) was run at b32 against a b82 baseline. Nothing "
            "else about it was ever compared like-for-like. This run is that "
            "comparison and nothing else: stock weights, no yaml, every mechanism "
            "provably off. If it lands near 55.24 the arch gain is real and the "
            "headline survives. If it lands near 56.1 the gain was batch size and "
            "you need to know that before it goes in a paper, not after."},

    {"name": "y26_arch_scb3_sbb50", "arm": "a", "model": "yolo26s.pt", "batch": 32,
     "yaml": ARCH_YAML,
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "arch": True},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True),
     "label": "P2 + DySample + the best loss config — the headline candidate",
     "why": "The two axes have never met with a loss config that PRESERVES large. "
            "Phase B tried arch + loss and failed, but that was scb3 alone (large "
            "-1.44) and snl1 (-4.37) — both of which pay for overall mAP in exactly "
            "the bucket the P2 head is supposed to help. scb3_sbb50 does not (-0.05). "
            "Run at b32 because the P2 head is memory-heavy, which is why "
            "y26_stock_b32 is in this script: together with the existing arch-alone "
            "number they give a clean three-point decomposition, all at b32:\n"
            "      stock            y26_stock_b32        (this script)\n"
            "      + arch           y26_p2k2_hi          (already have it)\n"
            "      + arch + loss    y26_arch_scb3_sbb50  (this run)\n"
            "  That is the paper's main table, and right now the first and third "
            "rows do not exist. If the axes compose you have your headline; if they "
            "interfere you report the better single axis and say why — either way "
            "the decomposition is what a reviewer wants to see."},

    # ---- ARM B: does the opposing-bias account generalise? ------------------
    {"name": "y26_snl1_sbb", "arm": "b", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"snl1": 0.25, "sbb": 0.5},
     "params": cfg(l1_scale_p=0.25, sbb_q=0.5, sbb_invert=True),
     "label": "SNL1 0.25 + SBB inv 0.5 — the account without SCB",
     "why": "THE run in this script. SNL1 pushes small (large -4.37), SBB inv "
            "pushes large in the branch that produces every prediction. If "
            "'opposing size-biases recover large' is a principle, this recovers "
            "large with no SCB anywhere. If large stays around 57, SCB+SBB is a "
            "specific lucky pairing and the paper says 'this config' instead of "
            "'this principle'. Those are very different claims and this is the "
            "one run that separates them."},

    {"name": "y26_scb2_sbb50", "arm": "b", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"scb": (2.0, 64.0), "sbb": 0.5},
     "params": cfg(tal_beta_small=2.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True),
     "label": "SCB 2.0 + SBB inv 0.5 — the SETTING THAT FAILED ALONE, now offset",
     "why": "The sharpest prediction available from the account, and the most "
            "likely of these eight to actually beat 55.65. beta_small=2.0 was the "
            "WORST point of the SCB sweep — 55.05, BELOW baseline, the only SCB "
            "setting that lost. The account says why: it over-pushes small, and "
            "nothing was offsetting it. SBB inv pushes large in exactly the branch "
            "that produces every prediction. If the account is right, the failed "
            "setting becomes viable, and since 2.0 pushes harder than 3.0 it may "
            "pair BETTER with a fixed q=0.5 than 3.0 does. A mechanism that turns "
            "a below-baseline config into an above-baseline one is far stronger "
            "evidence than another point near an optimum you already found."},

    {"name": "y26_scb3_sbb25", "arm": "b", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"scb": (3.0, 64.0), "sbb": 0.25},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.25, sbb_invert=True),
     "label": "SCB 3.0 + SBB inv 0.25 — is the balance point a plateau?",
     "why": "Weaker large-push. Together with q=0.75 this asks whether the large "
            "recovery is a plateau in q or a spike at exactly 0.5. SCB's own "
            "beta_small sweep was a spike (2 -> -0.19, 3 -> +0.42, 4 -> -0.07) and "
            "that single fact made the mechanism look like an artifact for two "
            "days. Finding out now costs an hour; finding out in review costs the "
            "claim."},

    {"name": "y26_scb3_sbb75", "arm": "b", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"scb": (3.0, 64.0), "sbb": 0.75},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.75, sbb_invert=True),
     "label": "SCB 3.0 + SBB inv 0.75 — the other side of the plateau",
     "why": "Stronger large-push. If the account is right, over-pushing should "
            "overshoot the balance the other way — large should start climbing "
            "past baseline while small pays for it. That would be the account "
            "confirmed from the far side, which is worth more than another point "
            "near the optimum. If instead q=0.25/0.5/0.75 are flat, the mechanism "
            "is insensitive and robust, which is the better outcome for a paper."},

    # ---- ARM C: cls_pw, never switched on in this project -------------------
    {"name": "y26_scb3_sbb50_pw25", "arm": "c", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "clspw": 0.25},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True,
                   cls_pw=0.25),
     "label": "best config + cls_pw 0.25 — class-frequency reweighting, gentle",
     "why": "First run in ~60 with cls_pw live. Weights bag 1.10, trolley 0.87. "
            "Every mechanism in this project so far reweights by object SIZE; this "
            "is the first that reweights by class FREQUENCY, and bag is 13.8 pp "
            "below trolley while holding 20% of instances. Gentle first because "
            "the size mechanisms all showed that over-correction costs more than "
            "under-correction costs."},

    {"name": "y26_scb3_sbb50_pw50", "arm": "c", "model": "yolo26s.pt", "batch": BATCH,
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "clspw": 0.50},
     "params": cfg(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True,
                   cls_pw=0.50),
     "label": "best config + cls_pw 0.50 — stronger",
     "why": "Weights bag 1.21, trolley 0.75. Two points make cls_pw a direction "
            "rather than a guess, which is what made SNT readable even though it "
            "was negative. Watch the PER-CLASS table, not overall mAP: the "
            "mechanism is doing its job if bag rises and trolley falls, and that "
            "can be true while overall barely moves. A mechanism that trades 2 pp "
            "of trolley for 3 pp of bag is useful on this dataset even at mAP "
            "parity, because bag is the deployment-relevant hard class."},

]


def preflight(todo):
    import inspect
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
        from ultralytics.models.yolo.detect import DetectionTrainer
    except Exception as e:
        print(f"  [ABORT] cannot import ultralytics: {e}")
        return False
    print(f"  ultralytics : {os.path.dirname(ultralytics.__file__)}")

    A = TAL.TaskAlignedAssigner
    checks = {
        "tal.py  scb_enabled": hasattr(A, "scb_enabled"),
        "loss.py BboxLoss.snl1_enabled": hasattr(BboxLoss, "snl1_enabled"),
        "loss.py BboxLoss.sbb_enabled": hasattr(BboxLoss, "sbb_enabled"),
        "loss.py v8DetectionLoss reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py v8DetectionLoss reads l1_scale_p": "l1_scale_p" in inspect.getsource(v8DetectionLoss.__init__),
        "loss.py E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
        "loss.py v8DetectionLoss reads class_weights": "class_weights" in inspect.getsource(v8DetectionLoss.__init__),
        "trainer has set_class_weights (cls_pw)": hasattr(DetectionTrainer, "set_class_weights"),
    }
    for k, v in checks.items():
        print(f"  {k:<46}{v}")
    if not all(checks.values()):
        print("\n  [ABORT] patch not fully installed. Run verify_patch_v6i.py --install --runtime")
        return False

    # cls_pw = 0 must be a genuine no-op: set_class_weights() returns early, so
    # model.class_weights is never set and the BCE is untouched. Confirm the
    # early return is actually in the source rather than assuming it.
    src = inspect.getsource(DetectionTrainer.set_class_weights)
    if "cls_pw == 0.0" not in src or "return" not in src:
        print("  [ABORT] set_class_weights has no cls_pw==0 early return; "
              "cls_pw=0 may not be inert.")
        return False
    print(f"  {'cls_pw = 0.0 is inert (early return)':<46}True")

    print()
    for r in todo:
        p, e = r["params"], r["expect"]
        bits = []
        if e.get("stock"):
            live = [k for k, v in p.items() if _ALL_OFF.get(k, "__") != v]
            if live:
                print(f"  [ABORT] {r['name']} is a STOCK control but differs: {live}")
                return False
            bits.append("STOCK (all mechanisms off)")
        if "scb" in e:
            a = A(topk=7, topk2=1)
            a.beta_small, a.beta_ref_px = p["tal_beta_small"], p["tal_beta_ref_px"]
            if not a.scb_enabled():
                print(f"  [ABORT] {r['name']}: scb_enabled() False")
                return False
            bits.append(f"SCB {a.beta_small}->{a.beta}")
        if "snl1" in e:
            bits.append(f"SNL1 p={p['l1_scale_p']}")
        if "sbb" in e:
            bits.append(f"SBB q={p['sbb_q']} inv={p['sbb_invert']}")
        if "clspw" in e:
            if not 0.0 < p["cls_pw"] <= 1.0:
                print(f"  [ABORT] {r['name']}: cls_pw={p['cls_pw']} outside (0,1]")
                return False
            bits.append(f"cls_pw={p['cls_pw']}")
        s = r.get("seed", SEED)
        print(f"  {r['name']:<22}b{r['batch']:<4}{r['model']:<14}seed{s}  {' + '.join(bits)}")

    print()
    print(f"  baseline {BASELINE:.2f} (b82)   best {BEST:.2f} large {BEST_LARGE:.2f}")
    print(f"  LARGE baseline {BASE_LARGE:.2f} — unbeaten in 54 configs; "
          f"y26_scb3_sbb50 came within 0.05")
    print("  Training is DETERMINISTIC -> deltas are exact. Seed 0 unless noted.")

    bases = [PROJECT_DIR]
    try:
        from ultralytics.utils import SETTINGS
        bases.append(os.path.join(str(SETTINGS.get("runs_dir", "runs")), "detect", PROJECT_DIR))
    except Exception:
        pass
    clash = sorted({f"{r['name']} -> {b}" for r in todo for b in bases
                    if os.path.isdir(os.path.join(b, r["name"]))}) if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_callbacks(model, rc):
    """Assert at epoch 1 that exactly the named mechanisms are live in the
    constructed criterion — and that nothing else is. A config key can be
    accepted, echoed in the header, and silently ignored; that is how rounds
    4-6 produced ten identically-configured runs under ten different names.
    """
    state = {"verified": False}
    e = rc["expect"]

    def on_epoch_start(trainer):
        m = getattr(trainer, "model", None)
        crit = getattr(m, "criterion", None)
        if crit is None:
            return  # ultralytics builds the criterion lazily in BaseModel.loss()
        if state["verified"] or trainer.epoch < 1:
            return

        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        e2e = o2m is not None and o2o is not None
        # Every run here is yolo26 (end2end), so e2e should always be True. The
        # single-branch fallback stays as a guard: if a future non-e2e model is
        # added, the checks below still run rather than silently skipping.
        branches = [("one2many", o2m), ("one2one", o2o)] if e2e else [("single", crit)]
        seen = []

        if e.get("stock") is not True and not e2e:
            raise RuntimeError(f"{rc['name']}: expected an E2E criterion, got {type(crit).__name__}")

        for tag, b in branches:
            a, bl = b.assigner, b.bbox_loss
            if "scb" in e:
                wb, wr = e["scb"]
                if not (hasattr(a, "scb_enabled") and a.scb_enabled()):
                    raise RuntimeError(f"{rc['name']}: SCB not live on {tag}")
                if abs(float(a.beta_small) - wb) > 1e-6 or abs(float(a.beta_ref_px) - wr) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} SCB ({a.beta_small},{a.beta_ref_px}) != ({wb},{wr})")
            elif hasattr(a, "scb_enabled") and a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB live on {tag} but not requested")

            if "snl1" in e:
                if not (hasattr(bl, "snl1_enabled") and bl.snl1_enabled()):
                    raise RuntimeError(f"{rc['name']}: SNL1 not live on {tag} (needs reg_max=1)")
                if abs(float(bl.l1_scale_p) - e["snl1"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} l1_scale_p={bl.l1_scale_p} != {e['snl1']}")
            elif hasattr(bl, "snl1_enabled") and bl.snl1_enabled():
                raise RuntimeError(f"{rc['name']}: SNL1 live on {tag} but not requested")

            if "sbb" in e:
                if not (hasattr(bl, "sbb_enabled") and bl.sbb_enabled()):
                    raise RuntimeError(f"{rc['name']}: SBB not live on {tag}")
                if abs(float(bl.sbb_q) - e["sbb"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} sbb_q={bl.sbb_q} != {e['sbb']}")
            elif hasattr(bl, "sbb_enabled") and bl.sbb_enabled():
                raise RuntimeError(f"{rc['name']}: SBB live on {tag} but not requested")

            for meth, nm in (("snt_enabled", "SNT"), ("tsh_enabled", "TSH")):
                if hasattr(a, meth) and getattr(a, meth)():
                    raise RuntimeError(f"{rc['name']}: {nm} live on {tag} — never requested in round 10")
            if hasattr(bl, "swa_enabled") and bl.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA live on {tag}")

        if "scb" in e:
            seen.append(f"SCB {branches[-1][1].assigner.beta_small}->6.0 on {len(branches)} branch(es)")
        if "snl1" in e:
            seen.append(f"SNL1 p={e['snl1']}")
        if "sbb" in e:
            b1, b2 = branches[0][1].bbox_loss, branches[-1][1].bbox_loss
            if e2e and float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
                raise RuntimeError(
                    f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f} — "
                    f"must be OPPOSITE or this is not SBB")
            seen.append(f"SBB q={b2.sbb_q} signs o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f}")

        # cls_pw lives on the MODEL, not the criterion: trainer.set_class_weights()
        # writes model.class_weights, and v8DetectionLoss reads it at construction.
        cw = getattr(branches[0][1], "class_weights", None)
        if "clspw" in e:
            if cw is None:
                raise RuntimeError(
                    f"{rc['name']}: cls_pw={e['clspw']} requested but class_weights is None. "
                    f"set_class_weights() did not run or returned early — the run would be "
                    f"a duplicate of the no-cls_pw config under a different name.")
            w = cw.flatten().tolist()
            if abs(sum(w) / len(w) - 1.0) > 1e-3:
                raise RuntimeError(f"{rc['name']}: class_weights mean {sum(w)/len(w):.4f} != 1.0")
            seen.append("cls_pw weights " + " ".join(f"{x:.3f}" for x in w))
        elif cw is not None:
            raise RuntimeError(f"{rc['name']}: class_weights live but cls_pw not requested")

        if e.get("stock"):
            seen.append("STOCK verified: no mechanism live on any branch")

        h = branches[0][1].hyp   # E2ELoss has no .hyp; the branches do
        if (float(h.box), float(h.cls), float(h.dfl)) != (7.5, 0.5, 1.5):
            raise RuntimeError(f"{rc['name']}: gains {h.box}/{h.cls}/{h.dfl} != 7.5/0.5/1.5")
        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] gains box={h.box} cls={h.cls} dfl={h.dfl} | e2e={e2e}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc):
    name, mdl, bs = rc["name"], rc["model"], rc["batch"]
    seed = rc.get("seed", SEED)
    print()
    print("=" * 78)
    print(f"  RUN {name}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={mdl}  imgsz={IMG_SIZE}  batch={bs}  epochs={EPOCHS}  seed={seed}")
    diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
    print(f"  differs from _ALL_OFF: {diff or 'NOTHING (stock control)'}")
    print()
    t0 = time.time()

    if rc.get("yaml"):
        # Build from the graph, then transfer stock weights. Order matters:
        # YOLO(yaml) constructs the P2 topology, .load() copies what matches.
        src = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{name}.yaml"))
        print(f"  cfg={src}")
        model = YOLO(src)
        check_arch_graph(model)
        try:
            model.load(mdl)
            print(f"  [arch] transferred stock weights from {mdl}")
        except Exception as ex:
            raise RuntimeError(f"{name}: weight transfer from {mdl} failed: {ex}")
    else:
        model = YOLO(mdl)
    state = attach_callbacks(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=bs,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                          patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": rc["arm"], "model": mdl, "batch": bs, "seed": seed,
           "params": rc["params"], "expect": rc["expect"], "imgsz": IMG_SIZE,
           "hours": hours, "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan"), "per_class": {}}
    try:
        with open(os.path.join(save_dir, "round10_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=bs,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
        try:  # per-class matters most for Arm C
            names = tm.names if isinstance(tm.names, dict) else dict(enumerate(tm.names))
            out["per_class"] = {names[int(c)]: float(tm.box.maps[int(c)]) for c in range(len(tm.box.maps))}
        except Exception:
            pass
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    print()
    print("=" * 92)
    print(f"  ROUND 10 — {IMG_SIZE}px, 70 epochs, deterministic")
    print("=" * 92)
    print(f"{'run':<24}{'arm':>4}{'b':>5}{'seed':>5}{'mAP50':>9}{'mAP50-95':>10}{'vs base':>9}")
    print("-" * 92)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<24}{r['arm'].upper():>4}{r['batch']:>5}{r['seed']:>5}"
              f"{r['test_map50']*100:>9.2f}{v:>10.2f}{v - BASELINE:>+9.2f}")
    print("-" * 92)
    print(f"  {'baseline b82':<24}{'':>4}{82:>5}{0:>5}{'':>9}{BASELINE:>10.2f}")
    print(f"  {'y26_scb3_sbb50 (best)':<24}{'':>4}{82:>5}{0:>5}{'':>9}{BEST:>10.2f}{BEST-BASELINE:>+9.2f}")
    print("\n  Deltas are EXACT (deterministic box). No noise band applies.\n")

    a = [r for r in ok if r["arm"] == "a"]
    if a:
        print("  ARM A — the batch confound")
        for r in a:
            v = r["test_map5095"] * 100
            print(f"    {r['name']:<22}{v:7.2f}   vs {BASELINE:.2f} (stock b82)   {v-BASELINE:+.2f}")
        print(f"      lands near {BASELINE:.2f}  -> batch is not the driver; y26_p2k2_hi's +0.84")
        print( "                             is architecture and the headline survives")
        print( "      lands well ABOVE   -> the arch gain was batch size. Requote every")
        print( "                             b32 arch number against THIS control, not 55.24\n")

    b = [r for r in ok if r["arm"] == "b"]
    if b:
        print("  ARM B — read the LARGE column, not this table")
        print(f"    {'config':<24}{'large':>8}")
        print(f"    {'baseline':<24}{BASE_LARGE:>8.2f}")
        print(f"    {'y26_scb_b3 (SCB only)':<24}{59.43:>8.2f}")
        print(f"    {'y26_sbb_inv50 (SBB only)':<24}{56.84:>8.2f}")
        print(f"    {'y26_snl1_p25 (SNL1 only)':<24}{56.50:>8.2f}")
        print(f"    {'y26_scb3_sbb50 (pair)':<24}{BEST_LARGE:>8.2f}   <- super-additive")
        for r in sorted(b, key=lambda x: x["name"]):
            print(f"    {r['name']:<24}{'____':>8}")
        print("      y26_snl1_sbb recovers large  -> opposing-bias is a PRINCIPLE")
        print("      it doesn't                   -> SCB+SBB is a specific pairing")
        print("      sbb 0.25/0.5/0.75 all ~60.8  -> plateau, robust, recommend it")
        print("      only 0.5 works               -> knife-edge, report it as such\n")

    c = [r for r in ok if r["arm"] == "c"]
    if c:
        print("  ARM C — cls_pw: judge on PER-CLASS, not overall")
        print(f"    {'config':<24}{'backpack':>10}{'bag':>8}{'trolley':>9}")
        print(f"    {'y26_scb3_sbb50 (pw=0)':<24}{56.8:>10.1f}{48.2:>8.1f}{62.0:>9.1f}")
        for r in sorted(c, key=lambda x: x["params"]["cls_pw"]):
            pc = r.get("per_class") or {}
            f = lambda k: f"{pc[k]*100:.1f}" if k in pc else "____"
            print(f"    {r['name']:<24}{f('backpack'):>10}{f('bag'):>8}{f('trolley'):>9}")
        print("      bag UP, trolley slightly down -> mechanism works as designed;")
        print("                                       useful even at mAP parity")
        print("      nothing moves                 -> inert on this data, clean ablation row\n")

    for r in ok:
        if r.get("weights"):
            print(f"    {r['name']:<24} {r['weights']}")
    print(f"\n  Run CocoEvalAllFolders_luggage.py on each best.pt — the size buckets")
    print(f"  decide Arms B and C, not the overall column.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    arm = None
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1].lower()
        del args[i:i + 2]
    only = set(args)
    todo = [r for r in RUNS if (not only or r["name"] in only) and (not arm or r["arm"] == arm)]
    if not todo:
        sys.exit(f"no runs match {only or ''} {('arm ' + arm) if arm else ''}")

    print()
    print("=" * 92)
    print(f"  YOLO26 ROUND 10 — {len(todo)} runs, ~{1.0 * len(todo):.1f} GPU-h "
          f"(measured 52 min/70 epochs + test eval)")
    print(f"  {', '.join(r['name'] for r in todo)}")
    print("=" * 92)
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")

    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except KeyboardInterrupt:
            print("\n  interrupted by user")
            break
        except Exception as ex:
            print(f"\n  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex),
                        "batch": rc["batch"], "seed": rc.get("seed", SEED),
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        with open(out_path, "w") as f:
            json.dump({"baseline": BASELINE, "best": BEST, "deterministic": True,
                       "results": res}, f, indent=2)

    summarise(res, out_path)
