#!/usr/bin/env python3
r"""
OVERNIGHT — rounds 12 + 13 in one queue. Six runs, ~7.6 GPU-h.

Two arms with different batch sizes, different model sources and different
verification, so they are declared per-run rather than globally.

    ARCH  b32, built from a YAML, mechanism = Detect.head_ch (a class attribute)
    LOSS  b82, built from yolo26s.pt, mechanism = train(**params) into E2ELoss

Ordered by value, not by arm: if the queue dies at 3am you keep the runs that
mattered. Results are written to disk after EVERY run for the same reason.


=============================================================================
ARCH (runs 1-2) — is the P2 head width the reason the P2 level lost?
=============================================================================
`y26_p2_b32` 55.03 vs `y26_stock_b32` 55.76: adding P2 lost 0.73 on a dataset
that is 60% small objects. From Detect.__init__:

    c2, c3 = max((16, ch[0] // 4, reg_max * 4)), max(ch[0], min(nc, 100))

Both head widths derive from ch[0], the FINEST level, and with nc=3 the
min(nc, 100) floor never binds, so c3 IS ch[0]:

    3 levels   ch = (128, 256, 512)       -> c3 = 128
    + P2       ch = (64, 128, 256, 512)   -> c3 =  64

"Add a P2 level" therefore also halves both heads on every level. Worse, those
two runs differ a THIRD way: yolo26s.pt is a 3-level checkpoint, so every
4-level model trains its head from scratch while the 3-level one inherits.
Depth, width and transfer are all confounded in that -0.73.

The two runs complete a square where three cells share the fresh-head handicap:

                    c3=64            c3=128
    3 levels        run 2            55.76  (PRETRAINED head, not comparable)
    4 levels        55.03            run 1

    head width, 4 lvl   = p2@128  - p2@64     clean, only c3 differs
    P2 level, c3=64     = p2@64   - 3lvl@64   clean, width and transfer held
    transfer penalty    = 3lvl@128 - 3lvl@64  what the pretrained head is worth

DETERMINISTIC: neither graph contains DySample, so no F.grid_sample and no
atomic-add nondeterminism. n=1 is exact here, unlike the sd-0.19 DySample
family where the entire groups sweep established nothing.


=============================================================================
LOSS (runs 3-6) — aimed by the FP decomposition
=============================================================================
    model            total   dup     cls     loc     bg
    y26_identity      192   4.2%   35.9%   16.7%   43.2%
    y26_scb3_sbb50    172   2.3%   37.2%   16.9%   43.6%
    y26_p2k2_hi       148   0.0%   32.4%   18.2%   49.3%

Duplicates are NOT leaking (4.2% -> 0.0%), so the one loss-reachable failure in
that table is not occurring — which also settles SNT (-3.93, closed a gap that
was fine) and TSH (+0.11, widened one that did not need it). Wrong-class is the
second largest category at 32-37%, and in absolute counts that is where the P2
architecture actually won: 69 -> 48.

So the classification path is the one with measured evidence in front of it,
and `cls` has never been touched — cls=0.5 is hardcoded in the _ALL_OFF block
of all eight loss scripts across 73 runs. Runs 3-4 close it. Run 5 attacks the
same target from the assignment side. Run 6 is a one-value control on the
overlap metric.

Base for all four is y26_scb3_sbb50 (55.65). Note what its SBB arm does:
E2ELoss sets one2many=-1 / one2one=+1 and sbb_weight uses
(sqrt(area)/ref)**(sign*q), so invert=True puts one2one on SMALL and one2many
on LARGE — the opposite of the code comments, and the arm that won.


=============================================================================
CALIBRATION
=============================================================================
LOSS bar is y26_scb2_sbb50 = 55.70 (+0.46). In 73 runs exactly one config has
cleared it. Priors 20-30% each, and the record on COMBINING two positives is
1 for 4. With four single-seed loss draws the expected MAXIMUM sits ~0.25-0.35
above the mean by selection alone — treat anything under +0.2 as noise.

ARCH has no bar: both outcomes are reportable, because the question is
attribution rather than a number.

    Usage:
        python run_yolo26_overnight_r1213_v6i.py              # all six, in order
        python run_yolo26_overnight_r1213_v6i.py --arm arch   # runs 1-2
        python run_yolo26_overnight_r1213_v6i.py --arm loss   # runs 3-6
        python run_yolo26_overnight_r1213_v6i.py y26_3lvl_head64
"""

import gc
import inspect
import json
import os
import re
import sys
import time

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_overnight_r1213_v6i"
CFG_P2 = "yolo26s-p2.yaml"  # 4 levels, plain nn.Upsample -> deterministic
CFG_3LVL = "yolo26s.yaml"  # 3 levels, stock graph
# The 's' is load-bearing: guess_model_scale needs [nslmx] immediately after the digits,
# else parse_model falls back to the FIRST scales key ('n') without failing.
CFG_P2ADD = "yolo26s-p2add.yaml"  # 4 levels, P2 appended so rows 0-22 keep their names

EPOCHS = 70
IMG_SIZE = 640
BATCH_ARCH = 32  # matches y26_p2_b32 (55.03) and y26_stock_b32 (55.76)
BATCH_LOSS = 82  # matches y26_base_rep and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False
REMAP_SHIFT = 6  # yolo26-p2 pushes stock head rows 17-22 down to 23-28
REMAP_ROWS = range(17, 23)

BASELINE = 55.24  # y26_base_rep      b82, 3 levels
BEST_LOSS = 55.65  # y26_scb3_sbb50    the base config for the loss arm
BEST_RAW = 55.70  # y26_scb2_sbb50    the loss bar
STOCK_B32 = 55.76  # y26_stock_b32     3 lvl, c3=128, PRETRAINED head
P2_B32 = 55.03  # y26_p2_b32        4 lvl, c3= 64, fresh head + fresh PAN
P2_128 = 54.61  # y26_p2_headref128 4 lvl, c3=128, fresh head + fresh PAN
L3_64 = 55.25  # y26_3lvl_head64   3 lvl, c3= 64, PRETRAINED head
CLS075_SEED0 = 55.89  # y26_scb3_sbb50_cls075 — the bar the loss search has to beat

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
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
)
_BASE = dict(tal_beta_small=3.0, tal_beta_ref_px=64.0, sbb_q=0.5, sbb_invert=True)
_BASE_EXPECT = {"scb": (3.0, 64.0), "sbb": 0.5}
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(_BASE)
    d.update(over)
    return d


RUNS = [
    # ---------------------------------------------------------------- ARCH
    # COMPLETED — commented out so the clash guard does not abort on their run dirs.
    # Values live in the constants above and summarise() falls back to them.
    #
    # {"name": "y26_p2_headref128", "arm": "arch", "cfg": CFG_P2, "head_ch": 128,   -> 54.61
    #  "want_c3": 128, "want_nl": 4,
    #  "label": "yolo26-p2, head width pinned at 128 (the stock 3-level value)",
    #  "why": "Isolates head width from pyramid depth."},
    #
    # {"name": "y26_3lvl_head64", "arm": "arch", "cfg": CFG_3LVL, "head_ch": 64,     -> 55.25
    #  "want_c3": 64, "want_nl": 3,
    #  "label": "stock 3-level with the head SHRUNK to 64 (the P2 graph's value)",
    #  "why": "The cell that makes the square attributable."},

    # ------------------------------------------------------- ARCH / ADDITIVE P2
    # headref0 runs FIRST: it is the paired control for the two p2add runs, not just a
    # determinism check. 55.03 came from run_yolo26_arch2_v6i.py in a tree that predates
    # the head_ch and loss patches, so p2add - 55.03 is cross-runner and uncontrolled.
    # p2add - headref0 is same night, same code, same box.
    {"name": "y26_p2_headref0", "arm": "arch", "cfg": CFG_P2, "head_ch": 0,
     "want_c3": 64, "want_nl": 4,
     "label": "yolo26-p2 at stock head width — in-batch control for the p2add runs",
     "why": "Must reproduce y26_p2_b32 = 55.03. Config is identical to arch2_v6i (70 ep, "
            "640, b32, seed 0, close_mosaic 10, patience 100, same YOLO(cfg)->load->train "
            "path), so a miss is drift from the head_ch/loss patches rather than the graph. "
            "It also validates that this graph is bit-deterministic: no DySample means no "
            "F.grid_sample and no atomic-add nondeterminism, verified for the stock 3-level "
            "graph but never directly for yolo26-p2. If it misses by more than 0.05, read "
            "the p2add runs ONLY against this number and never against 55.03."},

    {"name": "y26_p2add_h0", "arm": "arch", "cfg": CFG_P2ADD, "head_ch": 0,
     "want_c3": 64, "want_nl": 4,
     "label": "P2 as a LEAF branch off P3 — stock P3/P4/P5 kept byte-identical",
     "why": "NOT the same graph as yolo26-p2 with better loading — it is a different, "
            "cheaper design, and it changes two things at once. (1) Topology: yolo26-p2 "
            "starts the bottom-up path at P2, so P2 feeds P3 feeds P4 feeds P5 and all four "
            "levels are perturbed; here P2 hangs off the P3 head as a leaf and P3/P4/P5 stay "
            "exactly stock, so the extra level can only ADD a scale, never disturb the three "
            "that already work. It also drops yolo26-p2's extra Conv[128]+C3k2[256] stage, so "
            "it is smaller. (2) Transfer: because rows 0-22 keep their stock INDEX, "
            "intersect_dicts matches them by name and the whole bottom-up PAN loads from "
            "yolo26s.pt. In yolo26-p2 those same six layers are shape-identical but sit at "
            "rows 23-28, so they miss and train from scratch. Read the result as 'is this "
            "design better than 55.03', not as 'transfer is worth X' — the two are confounded "
            "here, and the preflight transfer table quantifies only the second."},

    {"name": "y26_p2add_h128", "arm": "arch", "cfg": CFG_P2ADD, "head_ch": 128,
     "want_c3": 128, "want_nl": 4,
     "label": "additive P2 with the head pinned at 128 — retests the refuted width hypothesis",
     "why": "p2@128 lost 0.42 to p2@64, which refuted the head-width hypothesis. But under "
            "broken transfer a WIDER head means more randomly-initialised parameters to fit "
            "in 70 epochs, so that test was confounded by the same defect. With the PAN "
            "transferred the width comparison is fair for the first time. If 128 wins here, "
            "the -0.42 was initialisation rather than width and the axis reopens."},

    {"name": "y26_p2_remap", "arm": "arch", "cfg": CFG_P2, "head_ch": 0, "remap": True,
     "want_c3": 64, "want_nl": 4,
     "label": "THE yolo26-p2 graph, with the shifted PAN rows loaded by key remap",
     "why": "The clean version of the experiment p2add cannot give you. p2add changes the "
            "topology AND the transfer at once; this changes ONLY the transfer. Identical "
            "graph, identical parameter count, identical everything to y26_p2_b32 = 55.03 "
            "except that stock rows 17-22 are copied into p2 rows 23-28 before training, "
            "which is legitimate because those six layers are shape-identical in the two "
            "graphs and differ only in the index intersect_dicts reads. If this lands near "
            "55.8 then 'adding P2 costs 0.73' was never an architecture result, and the "
            "reading of all ten DySample replicates and the 56.08 arch mean has to change. "
            "If it stays near 55.03, the P2 penalty is real and the campaign's arch "
            "conclusion survives — which is equally worth knowing and closes the question."},

    # ---------------------------------------------------------------- LOSS
    # COMPLETED — cls075 55.89 (campaign max), cls10 55.17, a075 55.03, diou 55.14.
    # cls is now a shape, not a point: 0.5 -> 55.65, 0.75 -> 55.89, 1.0 -> 55.17.
    # a075 and diou both LOST, closing the exponent and overlap-metric axes.
    #
    # {"name": "y26_scb3_sbb50_cls075", "arm": "loss", "params": cfg(cls=0.75),      -> 55.89
    #  "expect": {**_BASE_EXPECT, "cls": 0.75}, "label": "cls gain 0.5 -> 0.75"},
    # {"name": "y26_scb3_sbb50_cls10", "arm": "loss", "params": cfg(cls=1.0),        -> 55.17
    #  "expect": {**_BASE_EXPECT, "cls": 1.0}, "label": "cls gain 0.5 -> 1.0"},
    # {"name": "y26_a075_scb3_sbb50", "arm": "loss", "params": cfg(tal_alpha=0.75),  -> 55.03
    #  "expect": {**_BASE_EXPECT, "alpha": 0.75}, "label": "tal_alpha 0.5 -> 0.75"},
    # {"name": "y26_scb3_sbb50_diou", "arm": "loss", "params": cfg(iou_type="diou"), -> 55.14
    #  "expect": {**_BASE_EXPECT, "iou_type": "diou"}, "label": "DIoU"},

    # ====================================================== LOSS SEARCH (b82)
    # New bar: y26_scb3_sbb50_cls075 = 55.89. Everything below is a NEW config that can
    # beat it. Already swept, do not repeat: SCB beta_small 2/3/4, SCB ref 32/64/128
    # (flat, -0.02/-0.04), SBB q 0.25/0.50/0.75 on scb3, SBB invert both ways, SNL1
    # (does not stack), cls_pw 0.25/0.50, tal_alpha 0.75 on the pair (-0.62), eiou, diou.
    {"name": "y26_scb2_sbb50_cls075", "arm": "loss", "params": cfg(tal_beta_small=2.0, cls=0.75),
     "expect": {"scb": (2.0, 64.0), "sbb": 0.5, "cls": 0.75},
     "label": "cls=0.75 moved onto the OTHER top config, scb2_sbb50 (55.70)",
     "why": "The campaign has two co-equal loss configs: scb3_sbb50 55.65 and scb2_sbb50 "
            "55.70, the raw maximum. cls=0.75 has only ever been applied to scb3. If the "
            "+0.24 is a generic gain it lands here too and this becomes the campaign max at "
            "~55.94. If it does not transfer, cls=0.75 is specific to beta_small=3.0, which "
            "is itself the finding. Highest prior in the batch, and it doubles as the "
            "generality test that a bare seed repeat cannot give you."},

    {"name": "y26_b8_scb3_sbb50_cls075", "arm": "loss", "params": cfg(tal_beta=8.0, cls=0.75),
     "expect": {**_BASE_EXPECT, "cls": 0.75, "beta": 8.0},
     "label": "widen the SCB gap from the TOP: beta 6 -> 8 with beta_small held at 3.0",
     "why": "SCB's mechanism is the GAP between beta_small and beta, and every run so far "
            "moved the small end (2/3/4 at beta=6) or narrowed the gap (b4s2, +0.25). The "
            "gap has never been widened. beta=8, beta_small=3 gives a gap of 5 against the "
            "winning config's 3 — more IoU trust on large, unchanged on small, exactly the "
            "direction the SCB+SBB pairing result says should work. Also the only untested "
            "direction on the axis that produced the campaign's best single mechanism."},

    {"name": "y26_scb3_sbb50_cls065", "arm": "loss", "params": cfg(cls=0.65),
     "expect": {**_BASE_EXPECT, "cls": 0.65},
     "label": "locate the cls peak on the low side: 0.5 -> 55.65, 0.65 -> ?, 0.75 -> 55.89",
     "why": "The cls curve is sharply asymmetric — +0.24 from 0.5 to 0.75, then -0.72 from "
            "0.75 to 1.0. A peak that falls off three times faster on one side is usually "
            "not centred on the sampled point. Cheapest shot at a better operating point, "
            "and a third interior point turns cls from three scattered numbers into a curve "
            "you can plot in the paper."},

    # ==================================================================== CONFIRM
    # DEFERRED — seed and attribution runs, to be run once the search above settles on a
    # final config. Replicating a number that is about to be superseded wastes the GPU.
    #
    # {"name": "y26_cls075_only", "arm": "confirm", "params": dict(_ALL_OFF, cls=0.75),
    #  "expect": {"cls": 0.75}, "label": "cls=0.75 ALONE, against baseline 55.24"},
    # {"name": "y26_cls075_seed1", "arm": "confirm", "seed": 1, "params": cfg(cls=0.75),
    #  "expect": {**_BASE_EXPECT, "cls": 0.75}, "label": "SEED 1 replication"},
]


def preflight(todo):
    """Prove every mechanism is reachable BEFORE spending a night of GPU."""
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    arch = [r for r in todo if r["arm"] == "arch"]
    loss = [r for r in todo if r["arm"] != "arch"]

    if arch:
        if not hasattr(Detect, "head_ch"):
            print("  [ABORT] Detect.head_ch missing — head.py is not patched on this machine.")
            return False
        for tag, ch, hc, want in (("yolo26-p2 (4 lvl)", (64, 128, 256, 512), 0, 64),
                                  ("yolo26-p2 (4 lvl)", (64, 128, 256, 512), 128, 128),
                                  ("yolo26    (3 lvl)", (128, 256, 512), 0, 128),
                                  ("yolo26    (3 lvl)", (128, 256, 512), 64, 64)):
            Detect.head_ch = hc
            got = Detect(nc=3, reg_max=1, end2end=True, ch=ch).cv3[0][-1].in_channels
            print(f"  {tag}  head_ch={hc:<4} -> c3={got:<4} expected {want}")
            if got != want:
                Detect.head_ch = 0
                print("  [ABORT] the head_ch knob is not doing what these runs claim.")
                return False
        Detect.head_ch = 0

        for c in sorted({r["cfg"] for r in arch}):
            m = YOLO(c).model
            sc, n = m.yaml.get("scale"), sum(p.numel() for p in m.parameters())
            print(f"  {c:<22} scale={sc or 'NONE':<5} {n / 1e6:5.2f}M params")
            if sc != "s":
                print(f"  [ABORT] {c} resolved to scale '{sc}'. guess_model_scale needs a size "
                      f"letter right after the digits; without one parse_model falls back to "
                      f"the first scales key and yolo26s.pt may transfer nothing.")
                return False

        if any(r["cfg"] == CFG_P2ADD for r in arch):
            from ultralytics.utils.torch_utils import intersect_dicts
            ref = YOLO(MODEL_WEIGHTS).model.float().state_dict()
            print(f"\n  weight transfer from {MODEL_WEIGHTS} — the premise of the p2add runs")
            pct = {}
            for tag, c in (("3 lvl  ", CFG_3LVL), ("p2     ", CFG_P2), ("p2add  ", CFG_P2ADD)):
                sd = YOLO(c).model.float().state_dict()
                tot = sum(v.numel() for v in sd.values())
                hit = sum(v.numel() for v in intersect_dicts(ref, sd).values())
                pct[c] = 100 * hit / tot
                print(f"    {tag} {c:<22} {hit / 1e6:6.2f}M / {tot / 1e6:6.2f}M = {pct[c]:5.1f}%")
            if pct[CFG_P2ADD] <= pct[CFG_P2] + 5:
                print("  [ABORT] p2add transfers no more of the checkpoint than p2 — the "
                      "premise is wrong, do not spend the GPU.")
                return False
            print(f"    -> additive recovers {pct[CFG_P2ADD] - pct[CFG_P2]:.0f} points of the "
                  f"checkpoint that the inserted graph discards")

    if loss:
        try:
            from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
            from ultralytics.utils.metrics import IOU_FLAGS
            from ultralytics.utils.tal import TaskAlignedAssigner as A
        except ImportError as ex:
            print(f"  [ABORT] cannot import the patched loss modules: {ex}")
            return False
        checks = {
            "loss.py reads tal_beta_small": "tal_beta_small" in inspect.getsource(v8DetectionLoss.__init__),
            "loss.py reads tal_alpha": "tal_alpha" in inspect.getsource(v8DetectionLoss.__init__),
            "loss.py reads iou_type": "iou_type" in inspect.getsource(v8DetectionLoss.__init__),
            "E2ELoss reads sbb_q": "sbb_q" in inspect.getsource(E2ELoss.__init__),
            "IOU_FLAGS has diou": "diou" in IOU_FLAGS,
        }
        for k, v in checks.items():
            print(f"  {k:<44}{v}")
        if not all(checks.values()):
            print("  [ABORT] the loss patch is not fully installed.")
            return False
        probe = A(topk=7, topk2=1)
        if probe.tsh_enabled() or probe.snt_enabled() or probe.sbal_enabled():
            print("  [ABORT] a loss mechanism is live at its default value — every delta "
                  "would be measured against a moving baseline.")
            return False
        print(f"  {'TSH / SNT / SBAL inert at defaults':<44}True")

    print()
    for r in todo:
        b = BATCH_ARCH if r["arm"] == "arch" else BATCH_LOSS
        sd = r.get("seed", SEED)
        if r["arm"] == "arch":
            bits = f"{r['cfg']}  c3={r['want_c3']}  levels={r['want_nl']}"
        else:
            p, e = r["params"], r["expect"]
            parts = []
            if "scb" in e:
                parts.append(f"SCB {p['tal_beta_small']}@{p['tal_beta_ref_px']}px")
            if "sbb" in e:
                parts.append(f"SBB q={p['sbb_q']} inv={p['sbb_invert']}")
            for k in ("cls", "alpha", "beta", "iou_type"):
                if k in e:
                    parts.append(f"{k}={e[k]}")
            bits = "  +  ".join(parts) if parts else "stock"
        print(f"  {r['name']:<26}{r['arm']:<6}b{b:<4}seed{sd:<3}{bits}")

    print()
    print(f"  loss bar {CLS075_SEED0} (scb3_sbb50 {BEST_LOSS}, scb2_sbb50 {BEST_RAW})   "
          f"arch refs: stock_b32 {STOCK_B32}, p2_b32 {P2_B32}")
    clash = [os.path.join(PROJECT_DIR, r["name"]) for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"]))] if not OVERWRITE_EXISTING else []
    if clash:
        print("\n  [ABORT] run directories already exist:")
        for c in clash:
            print(f"      {c}")
        return False
    return True


def attach_guard(model, rc):
    """Assert at epoch 1 that this run's mechanism is LIVE in the built model."""
    state = {"verified": False}

    def on_arch(trainer):
        det = trainer.model.model[-1]
        got = det.cv3[0][-1].in_channels
        if got != rc["want_c3"]:
            raise RuntimeError(f"{rc['name']}: built c3={got}, expected {rc['want_c3']} — "
                               f"Detect.head_ch never reached the model, exactly like rounds 4-6.")
        if det.nl != rc["want_nl"]:
            raise RuntimeError(f"{rc['name']}: built {det.nl} levels, expected {rc['want_nl']}")
        o2o = det.one2one_cv3[0][-1].in_channels if hasattr(det, "one2one_cv3") else None
        if o2o is not None and o2o != rc["want_c3"]:
            raise RuntimeError(f"{rc['name']}: one2one c3={o2o}, expected {rc['want_c3']}")
        print(f"  [guard] c3={got} on cv3 and one2one_cv3 | c2={det.cv2[0][-1].in_channels} "
              f"| levels={det.nl} | reg_max={det.reg_max}")

    def on_loss(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None:
            return False  # ultralytics builds the criterion lazily in BaseModel.loss()
        e = rc["expect"]
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss — this is not yolo26 e2e")
        a1, a2, b1, b2 = o2m.assigner, o2o.assigner, o2m.bbox_loss, o2o.bbox_loss
        seen = []

        want_b, want_r = e["scb"]
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if not a.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB not live on {tag}")
            if abs(float(a.beta_small) - want_b) > 1e-6 or abs(float(a.beta_ref_px) - want_r) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} SCB=({a.beta_small},{a.beta_ref_px}), "
                                   f"expected ({want_b},{want_r})")
        seen.append(f"SCB {a2.beta_small}->{a2.beta}@{a2.beta_ref_px}px BOTH branches")

        for tag, b in (("one2many", b1), ("one2one", b2)):
            if not b.sbb_enabled() or abs(float(b.sbb_q) - e["sbb"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: SBB not live/wrong on {tag} (q={b.sbb_q})")
        if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
            raise RuntimeError(f"{rc['name']}: SBB signs must be OPPOSITE "
                               f"(o2m={b1.sbb_sign:+.0f} o2o={b2.sbb_sign:+.0f})")
        if float(b2.sbb_sign) >= 0:
            raise RuntimeError(f"{rc['name']}: one2one sbb_sign={b2.sbb_sign:+.0f}; the arm that "
                               f"won (+0.15) is invert=True -> one2one leans SMALL (sign<0)")
        seen.append(f"SBB q={b2.sbb_q} o2m={b1.sbb_sign:+.0f}(large) o2o={b2.sbb_sign:+.0f}(small)")

        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if "cls" in e and abs(float(h.cls) - e["cls"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: hyp.cls={h.cls}, expected {e['cls']}")
        if "alpha" in e:
            for tag, a in (("one2many", a1), ("one2one", a2)):
                if abs(float(a.alpha) - e["alpha"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected {e['alpha']}")
            seen.append(f"tal_alpha={a2.alpha} BOTH branches")
        if "beta" in e:
            for tag, a in (("one2many", a1), ("one2one", a2)):
                if abs(float(a.beta) - e["beta"]) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {e['beta']}")
            seen.append(f"tal_beta={a2.beta} BOTH branches")
        if "iou_type" in e:
            from ultralytics.utils.metrics import IOU_FLAGS

            want = IOU_FLAGS[e["iou_type"]]
            for tag, obj in (("assigner", a2), ("bbox_loss", b2)):
                if getattr(obj, "iou_kwargs", None) != want:
                    raise RuntimeError(f"{rc['name']}: one2one {tag} iou_kwargs="
                                       f"{getattr(obj, 'iou_kwargs', None)}, expected {want}")
            seen.append(f"iou_type={e['iou_type']} on assigner AND loss")

        # Anything not requested must be provably off.
        for tag, a in (("one2many", a1), ("one2one", a2)):
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT live on {tag}. It cost -3.93/-12.00.")
            if a.tsh_enabled() or a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: TSH/SBAL live on {tag} but not requested")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.swa_enabled() or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SWA/SNL1/NWD live on {tag} but not requested")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        return True

    def on_epoch_start(trainer):
        if state["verified"] or trainer.epoch < 1:
            return
        if rc["arm"] == "arch":
            on_arch(trainer)
            state["verified"] = True
        elif on_loss(trainer):
            state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def remap_pan(model, weights):
    """Copy stock head rows 17-22 into the p2 graph's rows 23-28, which hold the same layers."""
    ckpt = YOLO(weights).model.float().state_dict()
    sd = model.model.state_dict()
    moved = {}
    for k, v in ckpt.items():
        m = re.match(r"model\.(\d+)\.(.+)", k)
        if not m or int(m.group(1)) not in REMAP_ROWS:
            continue
        k2 = f"model.{int(m.group(1)) + REMAP_SHIFT}.{m.group(2)}"
        if k2 in sd and sd[k2].shape == v.shape:
            moved[k2] = v
    if not moved:
        raise RuntimeError("remap matched nothing — the row shift or the graph is not what "
                           "this run assumes")
    model.model.load_state_dict(moved, strict=False)
    after = model.model.state_dict()
    for k2, v in moved.items():
        if not torch.equal(after[k2].float(), v.float()):
            raise RuntimeError(f"remap did not stick on {k2}")
    print(f"  [remap] {len(moved)} tensors, {sum(v.numel() for v in moved.values()) / 1e6:.2f}M "
          f"params copied from rows {REMAP_ROWS.start}-{REMAP_ROWS.stop - 1} "
          f"to {REMAP_ROWS.start + REMAP_SHIFT}-{REMAP_ROWS.stop - 1 + REMAP_SHIFT}")


def run_one(rc):
    name, arm = rc["name"], rc["arm"]
    batch = BATCH_ARCH if arm == "arch" else BATCH_LOSS
    seed = rc.get("seed", SEED)
    print()
    print("=" * 78)
    print(f"  RUN {name}   [{arm.upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}  seed={seed}")
    if "cfg" in rc:
        print(f"  cfg={rc['cfg']}  head_ch={rc['head_ch']}  -> c3={rc['want_c3']}, levels={rc['want_nl']}")
    else:
        diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  differs from _ALL_OFF: {diff}")
    print()
    t0 = time.time()

    Detect.head_ch = rc.get("head_ch", 0)  # must be 0 for loss runs
    try:
        if arm == "arch":
            model = YOLO(rc["cfg"])
            model.load(MODEL_WEIGHTS)  # backbone/neck transfer; heads only where shapes match
            if rc.get("remap"):
                remap_pan(model, MODEL_WEIGHTS)
            extra = {}
        else:
            model = YOLO(MODEL_WEIGHTS)
            extra = rc["params"]
        state = attach_guard(model, rc)
        results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
                              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
                              deterministic=True, exist_ok=OVERWRITE_EXISTING, **extra)
    finally:
        Detect.head_ch = 0  # never leak the class attribute into a later build

    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": arm, "batch": batch, "seed": seed, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    out.update({"cfg": rc["cfg"], "c3": rc["want_c3"], "nl": rc["want_nl"]} if arm == "arch"
               else {"params": rc["params"], "expect": rc["expect"]})
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=batch,
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


def save(res):
    """Written after EVERY run so a 3am crash does not cost the completed ones."""
    try:
        with open(f"{PROJECT_DIR}__runs.json", "w") as f:
            json.dump({"baseline": BASELINE, "loss_base": BEST_LOSS, "loss_bar": BEST_RAW,
                       "stock_b32": STOCK_B32, "p2_b32": P2_B32, "seed": SEED,
                       "results": res}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] results json not saved: {ex}")


def summarise(res):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    if not ok:
        print("\nno completed runs.")
        return
    by = {r["name"]: r["test_map5095"] * 100 for r in ok}

    arch = [r for r in ok if r["arm"] == "arch"]
    if arch:
        print("\n" + "=" * 78)
        print("  ARCH — head width vs pyramid depth")
        print("=" * 78)
        print(f"{'run':<24}{'lvls':>5}{'c3':>5}{'head':>12}{'mAP50-95':>10}{'hours':>8}")
        print("-" * 64)
        print(f"{'y26_stock_b32':<24}{3:>5}{128:>5}{'PRETRAINED':>12}{STOCK_B32:>10.2f}{'-':>8}")
        print(f"{'y26_p2_b32':<24}{4:>5}{64:>5}{'fresh':>12}{P2_B32:>10.2f}{'-':>8}")
        for r in arch:
            print(f"{r['name']:<24}{r['nl']:>5}{r['c3']:>5}{'fresh':>12}"
                  f"{r['test_map5095'] * 100:>10.2f}{r['hours']:>8.2f}")

        p2_128 = by.get("y26_p2_headref128", P2_128)  # fall back to the completed run's value
        l3_64 = by.get("y26_3lvl_head64", L3_64)
        print("\n  THE SQUARE (fresh-head cells only)")
        print(f"{'':<14}{'c3=64':>10}{'c3=128':>10}")
        print(f"{'3 levels':<14}{(f'{l3_64:.2f}' if l3_64 else '-'):>10}{f'{STOCK_B32:.2f}*':>10}")
        print(f"{'4 levels (P2)':<14}{P2_B32:>10.2f}{(f'{p2_128:.2f}' if p2_128 else '-'):>10}")
        print("    * pretrained head — NOT fresh, so not part of the clean square")

        print("\n  DECOMPOSITION")
        if p2_128 is not None:
            print(f"    head width, 4 lvl   p2@128 - p2@64      {p2_128 - P2_B32:+.2f}")
        if l3_64 is not None:
            print(f"    P2 level, c3=64     p2@64  - 3lvl@64    {P2_B32 - l3_64:+.2f}")
            print(f"    transfer penalty    3lvl@128 - 3lvl@64  {STOCK_B32 - l3_64:+.2f}"
                  f"   (width + pretrained head, confounded)")
        if p2_128 is not None and abs(p2_128 - P2_B32) <= 0.4:
            print("\n    Head width is NOT the explanation — the c3 coupling is real in the")
            print("    code but does not cost accuracy here.")
        elif p2_128 is not None and p2_128 - P2_B32 > 0.4:
            print("\n    Head width is worth real accuracy on the P2 graph; the -0.73 charged")
            print("    to 'adding P2' was substantially the head halving.")
        if l3_64 is not None and STOCK_B32 - l3_64 > 0.4:
            print("    The pretrained head is worth a lot. EVERY 4-level run in this project")
            print("    paid that penalty — state it as a limitation of the P2 comparison.")

        ctrl = by.get("y26_p2_headref0")
        if ctrl is not None:
            d = abs(ctrl - P2_B32)
            print(f"\n    CONTROL: {ctrl:.2f} vs y26_p2_b32 {P2_B32} — differs by {d:.2f}")
            if d <= 0.05:
                print("    Reproduced. The tree is calibrated against the old campaign and")
                print("    yolo26-p2 is bit-deterministic here, so n=1 is an exact measurement.")
            else:
                print("    NOT reproduced. Read the p2add rows against THIS number only — the")
                print("    stored 55.03 came from a different tree and is not comparable.")

        remap = by.get("y26_p2_remap")
        if remap is not None:
            base = ctrl if ctrl is not None else P2_B32
            print(f"\n  TRANSFER, ISOLATED — same yolo26-p2 graph, PAN rows loaded by remap")
            print(f"    y26_p2_remap {remap:.2f} vs {base:.2f} without the remap "
                  f"({remap - base:+.2f})")
            if remap - base > 0.3:
                print("    The shifted-key weight loss WAS costing accuracy. 'Adding P2 costs")
                print("    0.73' is not an architecture result — it is an artifact of the yaml")
                print("    row order, and the reading of the ten DySample replicates and the")
                print("    56.08 arch mean has to be revised.")
            else:
                print("    Transfer was NOT the explanation. The P2 penalty is a real property")
                print("    of the architecture on this dataset, the campaign's arch conclusion")
                print("    stands, and the question is now closed rather than open.")

        add0, add128 = by.get("y26_p2add_h0"), by.get("y26_p2add_h128")
        if add0 is not None or add128 is not None:
            print("\n  ADDITIVE P2 — P2 as a leaf branch, stock P3/P4/P5 kept intact")
            print(f"{'':<22}{'inserted':>11}{'additive':>11}{'delta':>11}")
            if add0 is not None:
                print(f"{'c3=64  (head_ch 0)':<22}{P2_B32:>11.2f}{add0:>11.2f}{add0 - P2_B32:>+11.2f}")
            if add128 is not None:
                print(f"{'c3=128 (head_ch 128)':<22}{P2_128:>11.2f}{add128:>11.2f}{add128 - P2_128:>+11.2f}")
            best = max(v for v in (add0, add128) if v is not None)
            if ctrl is not None and add0 is not None:
                print(f"    against the IN-BATCH control {ctrl:.2f}: h0 {add0 - ctrl:+.2f}")
            if best > STOCK_B32:
                print(f"\n    {best:.2f} BEATS the 3-level stock {STOCK_B32}. A P2 leaf branch pays")
                print("    on this dataset, and it does so without touching P3/P4/P5 at all.")
            elif best > P2_B32 + 0.3:
                print(f"\n    The leaf design is worth {best - P2_B32:+.2f} over the inserted one, but")
                print(f"    still trails stock {STOCK_B32}. Use y26_p2_remap to see how much of")
                print("    that gap was topology and how much was the weight transfer.")
            else:
                print("\n    Fair transfer does NOT rescue P2. The extra level genuinely does not")
                print("    pay here — now a clean result instead of a confounded one, and that")
                print("    is the version worth publishing.")
            if add0 is not None and add128 is not None:
                print(f"    head width, transfer-fair   128 - 64 = {add128 - add0:+.2f}"
                      f"   (inserted graph said {P2_128 - P2_B32:+.2f})")

    loss = [r for r in ok if r["arm"] != "arch"]
    if loss:
        print("\n" + "=" * 78)
        print(f"  LOSS — bar is {CLS075_SEED0} (scb3_sbb50_cls075); base {BEST_LOSS}, baseline {BASELINE}")
        print("=" * 78)
        print(f"{'run':<26}{'seed':>5}{'mAP50-95':>10}{'vs base':>10}{'vs max':>9}{'hours':>8}")
        print("-" * 68)
        for r in sorted(loss, key=lambda x: -x["test_map5095"]):
            v = r["test_map5095"] * 100
            print(f"{r['name']:<26}{r['seed']:>5}{v:>10.2f}{v - BEST_LOSS:>+10.2f}"
                  f"{v - CLS075_SEED0:>+9.2f}{r['hours']:>8.2f}")

        scb2 = by.get("y26_scb2_sbb50_cls075")
        if scb2 is not None:
            print(f"\n    cls=0.75 on scb2_sbb50: {scb2:.2f} vs {BEST_RAW} without it "
                  f"({scb2 - BEST_RAW:+.2f})")
            if scb2 - BEST_RAW > 0.15:
                print("    The gain TRANSFERS — cls=0.75 is a generic operating point, not a")
                print("    quirk of beta_small=3.0. Report it as a tuned gain on both configs.")
            else:
                print("    The gain does NOT transfer — it is specific to beta_small=3.0. That is")
                print("    an interaction between the assignment metric and the cls weight, and")
                print("    a stronger claim than another +0.2.")

        best = max(r["test_map5095"] * 100 for r in loss)
        if best <= CLS075_SEED0:
            print(f"\n    Nothing beat {CLS075_SEED0}. The loss axis is at its practical ceiling —")
            print("    spend the next night on a seed repeat of that config, not on more search.")
        else:
            print(f"\n    New maximum {best:.2f}. Seed-repeat it before it goes in the paper: the")
            print("    expected MAXIMUM of three draws sits ~0.2-0.3 above the mean by selection")
            print("    alone, and every v26 number is single-seed.")
    print("=" * 78 + "\n")


def main():
    args = [a for a in sys.argv[1:]]
    arm = None
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1]
        del args[i:i + 2]
    todo = [r for r in RUNS if (not args or r["name"] in args) and (not arm or r["arm"] == arm)]
    if not todo:
        print(f"no runs matched. available: {[r['name'] for r in RUNS]}")
        return
    if not preflight(todo):
        return
    est = sum(1.8 if r["arm"] == "arch" else 1.0 for r in todo)
    print(f"\n  {len(todo)} runs, ~{est:.1f} GPU-h\n")

    res = []
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"\n  [FAIL] {rc['name']}: {ex}\n")
            res.append({"name": rc["name"], "arm": rc["arm"], "error": str(ex), "hours": 0.0,
                        "test_map50": float("nan"), "test_map5095": float("nan")})
        save(res)  # after every run, not at the end
    summarise(res)


if __name__ == "__main__":
    main()
