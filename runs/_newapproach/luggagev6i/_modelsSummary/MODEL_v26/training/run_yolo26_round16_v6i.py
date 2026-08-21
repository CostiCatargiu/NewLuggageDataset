#!/usr/bin/env python3
r"""
YOLO26 ROUND 16 — OVERNIGHT: seeds, branch attribution, the blend, augmentation

Ten runs, ~8.7 GPU-h, all b82 on the stock 3-level graph so every row is batch
matched to y26_identity and drops straight into Table 1. Ordered by value: if
the queue dies at 3am the runs that survive are the ones that matter.

Nothing here is another mechanism variant. Twelve mechanisms across 54 b82 runs
span +0.65 to -12.00, with every positive inside +0.65 — that axis is answered.
These four arms ask what the campaign never asked: how big is the noise, WHERE
does the one working mechanism act, is the branch schedule load-bearing, and
does the completely untouched augmentation axis do anything.


=============================================================================
ARM SEED (runs 1-4) — THE NUMBER EVERY REVIEWER WILL ASK FOR
=============================================================================
The project owns exactly ONE noise estimate: 56.08 +- 0.19, produced by
accident from ten runs that were labelled as different configurations, on the
P2+DySample graph, at a different batch. It has been used to judge b82 loss
results it does not apply to.

y26_base_rep came back BIT-IDENTICAL to yolo26_custom-9 across all 118 values,
which proves determinism at a FIXED seed. It says nothing about seed variance,
and the cls curve implies sd ~0.3 — comparable to almost every effect claimed.

    y26_cls075_s1 / _s2     the best measured config (55.89, +0.65), repeated
    y26_identity_s1 / _s2   the reference, which also has no error bar

PRE-REGISTERED, so it cannot be reinterpreted afterwards:

    sd >= 0.30    the whole loss axis is inside noise. Report it as null; the
                  diagnostics become the paper. This is a RESULT, not a failure.
    sd <= 0.15    +0.65 stands with an interval.
    in between    report scb3_sbb50 (+0.41) as the mechanism and cls as scatter.


=============================================================================
ARM SCB (runs 5-6) — WHERE DOES SCB ACTUALLY ACT?
=============================================================================
SCB is the only mechanism with a real effect (+0.42, and a component of every
winning combination). Its justification is a ONE2ONE argument:

    "topk2 = 1 means the metric picks THE single anchor that produces every
     prediction, so a global beta over-trusts IoU where it is least reliable"

But tal_beta_small is set in v8DetectionLoss.__init__, and E2ELoss builds TWO
of those. SCB has always been live on BOTH branches, including one2many, where
topk=10 with topk2 unset and the single-anchor argument does not apply.

The campaign's one working mechanism has never been attributed to a branch.

    control   y26_scb_b3            SCB on BOTH        55.66   (+0.42)
    run 5     y26_scb3_o2o_only     SCB on one2one     ?
    run 6     y26_scb3_o2m_only     SCB on one2many    ?

    o2o ~ 55.66, o2m ~ 55.24   the stated mechanism is correct
    o2m ~ 55.66, o2o ~ 55.24   the published explanation is WRONG — SCB works
                              through the AUXILIARY branch, i.e. a curriculum
                              effect, not a single-anchor fix
    both ~ 55.45              additive across branches; the one2one argument is
                              at best half the story

Every outcome is publishable and one retires an explanation currently stated as
fact. SBB is OFF here on purpose — it also biases the branches, which would
make the attribution unreadable.


=============================================================================
ARM BLEND (runs 7-8) — sbb_q AND o2m ARE CONFOUNDED BY CONSTRUCTION
=============================================================================
E2ELoss decays o2m 0.8 -> 0.1 linearly over epochs-1, so the LARGE-leaning
branch carries ~80% of the loss early and the SMALL-leaning one ~90% late: SBB
implements a large->small CURRICULUM, not a static specialisation.

That explains sbb_q being a knife-edge

    0.25 -> 55.28    0.50 -> 55.65    0.75 -> 55.20

better than a narrow optimum does, because q is integrated against a moving
weight and is therefore not a single-axis knob at all. Nothing in 92 runs
varied the blend: o2m and final_o2m were hardcoded.

    control   y26_scb3_sbb50   o2m 0.8 -> 0.1 (stock)     55.65
    run 7     y26_o2m_pin50    pinned at 0.5, NO decay    ?
    run 8     y26_o2m_final30  o2m 0.8 -> 0.3             ?


=============================================================================
ARM AUG (runs 9-10) — AN AXIS WITH ZERO COVERAGE
=============================================================================
Not one of 92 runs varied copy_paste, mixup, cutmix, multi_scale, degrees,
perspective, scale or the mosaic schedule. On a dataset that is ~60% small
objects these are the standard levers, and they cost the same 0.87 h as another
loss knob whose ceiling is already known to be +0.65.

    run 9     y26_aug_ms        multi_scale=0.5
    run 10    y26_aug_scale75   scale 0.5 -> 0.75

Both on STOCK loss, so a gain is attributable to augmentation alone. Two points
make it a direction rather than a lone guess — the standard applied to nwd and
refused to the single-point SCB spike. copy_paste is excluded on purpose: it
operates on segment labels and this dataset is detection-first, so it could be
a silent no-op, which is the exact failure mode that cost rounds 4-6.

    DECISION   >= +0.5 over identity -> the augmentation axis is open and gets
               its own night before any further loss work.


=============================================================================
CALIBRATION
=============================================================================
Nine directional predictions have been made in this campaign and nine were
falsified, every one optimistic. Nothing above is a prediction. Expect the scb
and blend arms inside +-0.3 of their controls and treat anything under +0.2 as
noise. The VALUE of those two arms is attribution, not a number — and the seed
arm is worth more than any of them.


=============================================================================
REQUIRES  (scb and blend arms only)
=============================================================================
loss.py must expose the new keys, and default.yaml must accept them:

    E2ELoss.__init__   o2m_start / o2m_final / o2m_decay / scb_branch
    E2ELoss.decay()    returns o2m_copy unchanged when o2m_decay is False

The preflight asserts BOTH before spending any GPU, and skips the check when
only the seed/aug arms are selected. A config key accepted by default.yaml,
printed in the header and ignored by the consumer is exactly how rounds 4-6
lost ten runs — three different files, and only the third one matters.

    Usage:
        python run_yolo26_round16_v6i.py                  # all ten, in order
        python run_yolo26_round16_v6i.py --arm seed       # runs 1-4, no patch needed
        python run_yolo26_round16_v6i.py --arm scb        # runs 5-6
        python run_yolo26_round16_v6i.py --arm blend      # runs 7-8
        python run_yolo26_round16_v6i.py --arm aug        # runs 9-10, no patch needed
        python run_yolo26_round16_v6i.py y26_o2m_pin50    # one by name
"""

import argparse
import gc
import inspect
import json
import os
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"  # STOCK. no yaml, no P2 head, no DySample.
PROJECT_DIR = "runs_yolo26_round16_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches y26_base_rep and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

BASELINE = 55.24  # y26_base_rep      stock, b82
CTRL_SCB = 55.66  # y26_scb_b3        SCB 3.0 on BOTH branches, no SBB  -> arm A control
CTRL_SBB = 55.65  # y26_scb3_sbb50    SCB 3.0 + SBB 0.5 inv            -> arm B control
BEST_RAW = 55.89  # y26_scb3_sbb50_cls075

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

_SCB = dict(tal_beta_small=3.0, tal_beta_ref_px=64.0)          # = y26_scb_b3
_SBB = dict(**_SCB, sbb_q=0.5, sbb_invert=True)                # = y26_scb3_sbb50
# =============================================================================


def cfg(base, **over):
    d = dict(_ALL_OFF)
    d.update(base)
    d.update(over)
    return d


RUNS = [
    # ------------------------------------------------------------------ SEEDS
    # The campaign owns exactly one noise estimate and it came from an accident on a
    # different graph at a different batch. Until these exist, every claimed effect is
    # smaller than the only sd anyone can quote.
    {"name": "y26_cls075_s1", "arm": "seed", "seed": 1, "ctrl": BEST_RAW,
     "params": cfg(_SBB, cls=0.75),
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "blend": (0.8, 0.1, True), "cls": 0.75},
     "label": "scb3_sbb50 + cls 0.75, SEED 1 — the best number in the campaign, repeated",
     "why": "55.89 (+0.65) is the largest batch-matched gain measured, and it sits on a "
            "NON-monotone curve (0.50/0.65/0.75/1.00 -> 55.65/55.39/55.89/55.17) whose "
            "interior point is below both neighbours. Moving cls=0.75 onto scb2_sbb50 "
            "cost -1.01. One repeat gives a RANGE, which is already enough to say "
            "whether +0.65 survives; seed 2 tomorrow turns the range into an sd."},

    {"name": "y26_identity_s1", "arm": "seed", "seed": 1, "ctrl": BASELINE,
     "params": dict(_ALL_OFF),
     "expect": {"scb": None, "sbb": 0.0, "blend": (0.8, 0.1, True)},
     "label": "stock yolo26s, SEED 1 — the reference needs an error bar too",
     "why": "y26_base_rep reproduced yolo26_custom-9 BIT-IDENTICALLY, which proves the "
            "box is deterministic at a FIXED seed — not that the baseline is stable "
            "across seeds. Every delta in Table 1 is measured against a single draw of "
            "the reference, so the reference's own spread is half the error budget."},

    {"name": "y26_scb3_sbb50_s1", "arm": "seed", "seed": 1, "ctrl": CTRL_SBB,
     "params": cfg(_SBB),
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "blend": (0.8, 0.1, True)},
     "label": "scb3_sbb50, SEED 1 — the config that would actually be reported",
     "why": "The mechanism result (+0.41) rather than the best raw number, and the only "
            "config in 92 runs that gains overall WITHOUT giving up large (60.82 vs "
            "60.87). If the paper reports one loss config it is this one, so it needs "
            "its own repeat and not an sd borrowed from cls075."},

    {"name": "y26_cls065_s1", "arm": "seed", "seed": 1, "ctrl": 55.39,
     "params": cfg(_SBB, cls=0.65),
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "blend": (0.8, 0.1, True), "cls": 0.65},
     "label": "scb3_sbb50 + cls 0.65, SEED 1 — the DIP, repeated",
     "why": "55.39 sits BELOW both its neighbours (0.50 -> 55.65, 0.75 -> 55.89), which "
            "is what makes the cls curve unreadable. Repeating the dip tests the noise "
            "hypothesis from the other side: if 0.65 comes back near 55.7 the curve was "
            "scatter all along, and that single run invalidates cls=0.75 more cheaply "
            "than three seeds at 0.75 could confirm it."},

    # ------------------------------------------------------------------ ARM A
    {"name": "y26_scb3_o2o_only", "arm": "scb", "ctrl": CTRL_SCB,
     "params": cfg(_SCB, scb_branch="one2one"),
     "expect": {"scb": (3.0, 64.0), "scb_on": "one2one", "sbb": 0.0,
                "blend": (0.8, 0.1, True)},
     "label": "SCB 3.0 on ONE2ONE only — the branch the published argument names",
     "why": "If the single-anchor account is right this reproduces the +0.42 on its "
            "own and the mechanism section writes itself. If it lands at baseline, "
            "the explanation currently stated as fact is wrong."},

    {"name": "y26_scb3_o2m_only", "arm": "scb", "ctrl": CTRL_SCB,
     "params": cfg(_SCB, scb_branch="one2many"),
     "expect": {"scb": (3.0, 64.0), "scb_on": "one2many", "sbb": 0.0,
                "blend": (0.8, 0.1, True)},
     "label": "SCB 3.0 on ONE2MANY only — the branch the argument says is irrelevant",
     "why": "The other half of the attribution, and the one that cannot be skipped: "
            "without it a null on run 1 is uninterpretable — it could mean SCB needs "
            "both branches, or that it never acted through one2one at all. one2many "
            "has topk=10 and topk2 unset, so a win here means SCB is a curriculum "
            "effect on an auxiliary branch, which is a different paper."},

    # ------------------------------------------------------------------ ARM B
    {"name": "y26_o2m_pin50", "arm": "blend", "ctrl": CTRL_SBB,
     "params": cfg(_SBB, o2m_start=0.5, o2m_decay=False),
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "blend": (0.5, 0.1, False)},
     "label": "SBB q=0.5 with the blend PINNED at o2m=0.5 — de-confounds q",
     "why": "sbb_q has only ever been measured against a moving o2m, so the "
            "knife-edge (0.25/0.50/0.75 -> 55.28/55.65/55.20) may be a curriculum "
            "artefact rather than an optimum. Pinning the blend is the only way to "
            "ask what q does by itself. Note this also changes the total loss scale, "
            "so read it against run 4 as well as against the control."},

    {"name": "y26_o2m_final30", "arm": "blend", "ctrl": CTRL_SBB,
     "params": cfg(_SBB, o2m_final=0.3),
     "expect": {"scb": (3.0, 64.0), "sbb": 0.5, "blend": (0.8, 0.3, True)},
     "label": "o2m 0.8 -> 0.3 — keep the auxiliary branch alive to the end",
     "why": "The cheapest question in the file: is one2many still worth something "
            "late? It is discarded at inference, but it shapes the shared backbone "
            "and neck throughout. 0.1 was inherited from upstream and never tested "
            "on this dataset."},

    # ------------------------------------------------------------------ ARM C
    # Augmentation: ZERO of 92 runs varied a single augmentation parameter. On a
    # dataset that is ~60% small objects these are the standard levers, and they cost
    # the same 0.87 h as another loss knob with a measured ceiling of +0.65.
    # copy_paste is deliberately excluded: it operates on segment labels and this
    # dataset is detection-first, so it could be a silent no-op.
    {"name": "y26_aug_ms", "arm": "aug", "ctrl": BASELINE,
     "params": cfg({}, multi_scale=0.5),
     "expect": {"scb": None, "sbb": 0.0, "blend": (0.8, 0.1, True),
                "aug": {"multi_scale": 0.5}},
     "label": "stock loss + multi_scale=0.5 — resolution jitter during training",
     "why": "Stock loss on purpose, so any gain is attributable to augmentation alone. "
            "The one measured resolution result in this project is +1.41 mean on the "
            "sibling v5i dataset (640->896), larger than anything in 92 v6i runs; "
            "multi_scale is the version that costs no inference time."},

    {"name": "y26_aug_scale75", "arm": "aug", "ctrl": BASELINE,
     "params": cfg({}, scale=0.75),
     "expect": {"scb": None, "sbb": 0.0, "blend": (0.8, 0.1, True),
                "aug": {"scale": 0.75}},
     "label": "stock loss + scale=0.75 — stronger zoom jitter (stock is 0.5)",
     "why": "Directly manufactures more small instances from the same images, which is "
            "the population the diagnostics say the model is capable on but scores "
            "badly. Two aug points make this a DIRECTION rather than a lone guess, the "
            "same standard applied to nwd and refused to the single-point SCB spike."},

    {"name": "y26_aug_cm20", "arm": "aug", "ctrl": BASELINE,
     "params": cfg({}, close_mosaic=20),
     "expect": {"scb": None, "sbb": 0.0, "blend": (0.8, 0.1, True),
                "aug": {"close_mosaic": 20}},
     "label": "stock loss + close_mosaic=20 — twice as long without mosaic at the end",
     "why": "close_mosaic=10 was inherited from upstream and hardcoded in all 26 runners. "
            "It interacts with the o2m schedule: mosaic switches off at epoch 61 while "
            "o2m is already near 0.1, so the final clean-image phase is trained almost "
            "entirely by one2one. Doubling it is the cheapest way to ask whether that "
            "phase is too short on a dataset this small."},

    # ------------------------------------------------------------------ ARM D
    {"name": "y26_coslr", "arm": "sched", "ctrl": BASELINE,
     "params": cfg({}, cos_lr=True),
     "expect": {"scb": None, "sbb": 0.0, "blend": (0.8, 0.1, True),
                "aug": {"cos_lr": True}},
     "label": "stock loss + cos_lr=True — the optimiser axis has zero coverage",
     "why": "optimizer=auto, lr0=0.01, cos_lr=False and nbs=64 are identical across all "
            "92 runs, so the entire schedule axis is untested. Cosine decay is the one "
            "change here that costs nothing at inference and needs no justification in "
            "a paper. A null closes the axis; a gain means every loss delta in the "
            "campaign was measured on a suboptimal schedule."},
]


def preflight(todo):
    """Prove every mechanism is REACHABLE before spending a night of GPU.

    Checks the CONSUMER, not the config surface. Rounds 4-6 passed a preflight
    that only verified tal.py had the class and default.yaml took the keys.
    """
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils.loss import E2ELoss
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False

    src_init = inspect.getsource(E2ELoss.__init__)
    src_decay = inspect.getsource(E2ELoss.decay)
    # Only the scb/blend arms need the patch. Seeds and augmentation run on stock
    # loss.py, so a missing patch must not block them.
    if any(r["arm"] in ("scb", "blend") for r in todo):
        checks = {
            "E2ELoss reads o2m_start": "o2m_start" in src_init,
            "E2ELoss reads o2m_final": "o2m_final" in src_init,
            "E2ELoss reads o2m_decay": "o2m_decay" in src_init,
            "E2ELoss.decay honours o2m_decay": "o2m_decay" in src_decay,
            "E2ELoss reads scb_branch": "scb_branch" in src_init,
        }
        for k in ("o2m_start", "o2m_final", "o2m_decay", "scb_branch"):
            checks[f"default.yaml accepts {k}"] = k in DEFAULT_CFG_DICT
    else:
        checks = {"patch not required for the selected arms": True}
    for k, v in checks.items():
        print(f"  {k:<40} {v}")
        ok &= bool(v)
    if not ok:
        print("\n  [ABORT] patch loss.py and default.yaml first — see the REQUIRES block.")
        print("          or run only the arms that do not need it:  --arm seed | --arm aug")
        return False

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        s = r.get("seed", SEED)
        print(f"  {r['name']:<22} {r['arm']:<6} seed{s}  vs {r['ctrl']:.2f}  |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.87 * len(todo):.1f} GPU-h")
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

        # ---- SCB, and WHICH branch it is on -------------------------------
        if e.get("scb") is None:
            if a1.scb_enabled() or a2.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB is live but was not requested")
            seen.append("SCB off")
        else:
            want_b, want_r = e["scb"]
            on = e.get("scb_on", "both")
            live = {"one2many": a1.scb_enabled(), "one2one": a2.scb_enabled()}
            want = {"one2many": on in ("both", "one2many"), "one2one": on in ("both", "one2one")}
            if live != want:
                raise RuntimeError(f"{rc['name']}: SCB live={live}, expected {want} (scb_branch={on})")
            for tag, a in (("one2many", a1), ("one2one", a2)):
                if not want[tag]:
                    continue
                if abs(float(a.beta_small) - want_b) > 1e-6 or abs(float(a.beta_ref_px) - want_r) > 1e-6:
                    raise RuntimeError(f"{rc['name']}: {tag} SCB=({a.beta_small}, {a.beta_ref_px}), "
                                       f"expected ({want_b}, {want_r})")
            seen.append(f"SCB {want_b}@{want_r}px on {on.upper()}")

        # ---- the blend schedule -------------------------------------------
        w_start, w_final, w_decay = e["blend"]
        got = (float(crit.o2m_copy), float(crit.final_o2m), bool(crit.o2m_decay))
        if abs(got[0] - w_start) > 1e-6 or abs(got[1] - w_final) > 1e-6 or got[2] != w_decay:
            raise RuntimeError(f"{rc['name']}: blend is {got}, expected {(w_start, w_final, w_decay)}")
        # decay() must actually be inert when pinned — the flag existing is not enough
        if not w_decay and abs(crit.decay(crit.updates + 10) - w_start) > 1e-6:
            raise RuntimeError(f"{rc['name']}: o2m_decay=False but decay() still moves")
        seen.append(f"blend o2m {got[0]} -> {got[1]} decay={got[2]} (live o2m={crit.o2m:.3f})")

        # ---- SBB ----------------------------------------------------------
        want_q = e["sbb"]
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if abs(float(b.sbb_q) - want_q) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} sbb_q={b.sbb_q}, expected {want_q}")
        if want_q > 0.0:
            if float(b1.sbb_sign) * float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: SBB signs o2m={b1.sbb_sign:+.0f} "
                                   f"o2o={b2.sbb_sign:+.0f} — they must be OPPOSITE")
            if float(b2.sbb_sign) >= 0:
                raise RuntimeError(f"{rc['name']}: one2one sbb_sign={b2.sbb_sign:+.0f}; the arm "
                                   f"that won is invert=True -> one2one leans SMALL (sign<0)")
            seen.append(f"SBB q={want_q} o2m={b1.sbb_sign:+.0f}(large) o2o={b2.sbb_sign:+.0f}(small)")
        else:
            seen.append("SBB off")

        # ---- everything else must be provably off -------------------------
        for a in (a1, a2):
            if a.snt_enabled():
                raise RuntimeError(f"{rc['name']}: SNT is live. It cost -3.93/-12.00.")
            if a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: TSH is live but was not requested")
            if a.sbal_enabled():
                raise RuntimeError(f"{rc['name']}: SBAL is live but was not requested")
        for tag, b in (("one2many", b1), ("one2one", b2)):
            if b.swa_enabled():
                raise RuntimeError(f"{rc['name']}: SWA is live on {tag} but was not requested")
            if b.snl1_enabled():
                raise RuntimeError(f"{rc['name']}: SNL1 is live on {tag} but was not requested")
            if float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: NWD is live on {tag} but was not requested")
        h = o2o.hyp  # E2ELoss has no .hyp — only the inner v8DetectionLoss objects do
        if bool(getattr(h, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal is set; this file is the stock 3-level graph")
        if "cls" in e and abs(float(h.cls) - e["cls"]) > 1e-6:
            raise RuntimeError(f"{rc['name']}: hyp.cls={h.cls}, expected {e['cls']}")
        # augmentation and schedule are read off the TRAINER args, not the criterion
        for k, v in e.get("aug", {}).items():
            got = getattr(trainer.args, k, None)
            if isinstance(v, bool):
                if bool(got) is not v:
                    raise RuntimeError(f"{rc['name']}: trainer.args.{k}={got}, expected {v}")
            elif got is None or abs(float(got) - float(v)) > 1e-6:
                raise RuntimeError(f"{rc['name']}: trainer.args.{k}={got}, expected {v}")
            seen.append(f"{k}={got}")

        for s in seen:
            print(f"  [guard] {s}")
        print(f"  [guard] nothing else live | gains box={h.box} cls={h.cls} dfl={h.dfl}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc, cli_seed):
    seed = rc.get("seed", cli_seed)
    name = rc["name"] if ("seed" in rc or cli_seed == SEED) else f"{rc['name']}_s{cli_seed}"
    print()
    print("=" * 78)
    print(f"  RUN {name}   [arm {rc['arm'].upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  model={MODEL_WEIGHTS} (stock)  imgsz={IMG_SIZE}  batch={BATCH}  "
          f"epochs={EPOCHS}  seed={seed}  control={rc['ctrl']:.2f}")
    print(f"  differs from _ALL_OFF: {({k: v for k, v in rc['params'].items() if _ALL_OFF.get(k, '__') != v})}")
    print()
    t0 = time.time()

    model = YOLO(MODEL_WEIGHTS)
    state = attach_guard(model, rc)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(rc["params"])  # a run may override close_mosaic, cos_lr, epochs
    results = model.train(**kw)
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": rc["arm"], "ctrl": rc["ctrl"], "params": rc["params"],
           "expect": rc["expect"], "seed": seed, "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE,
           "batch": BATCH, "hours": hours, "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # name it *_params.json so the eval script's glob finds it — r5_params.json did not
    # match its hardcoded tuple, which is why y26_p2k* have no metadata at all.
    try:
        with open(os.path.join(save_dir, "round16_params.json"), "w") as f:
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
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    if not ok:
        print("\nno completed runs.")
        return
    print("\n" + "=" * 78)
    print("  ROUND 16 — RESULTS")
    print("=" * 78)
    print(f"{'run':<24}{'arm':<7}{'mAP50-95':>10}{'vs ctrl':>9}{'vs base':>9}{'hours':>7}")
    print("-" * 66)
    print(f"{'y26_base_rep':<24}{'ref':<7}{BASELINE:>10.2f}{'-':>9}{0.0:>+9.2f}{'-':>7}")
    print(f"{'y26_scb_b3 (arm A)':<24}{'ctrl':<7}{CTRL_SCB:>10.2f}{0.0:>+9.2f}{CTRL_SCB - BASELINE:>+9.2f}{'-':>7}")
    print(f"{'y26_scb3_sbb50 (arm B)':<24}{'ctrl':<7}{CTRL_SBB:>10.2f}{0.0:>+9.2f}{CTRL_SBB - BASELINE:>+9.2f}{'-':>7}")
    print("-" * 66)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<24}{r['arm']:<7}{v:>10.2f}{v - r['ctrl']:>+9.2f}"
              f"{v - BASELINE:>+9.2f}{r['hours']:>7.2f}")

    print("\n  READ IT")
    # ---- seeds: one repeat gives a RANGE. n=2 range ~ 1.13 sd, so the implied sd is
    # range/1.13 -- crude, but enough to decide whether the axis is readable at all.
    spreads = []
    for tag, ctrl, pref in (("cls075", BEST_RAW, "y26_cls075_s"),
                            ("cls065", 55.39, "y26_cls065_s"),
                            ("scb3_sbb50", CTRL_SBB, "y26_scb3_sbb50_s"),
                            ("identity", BASELINE, "y26_identity_s")):
        vals = [r["test_map5095"] * 100 for r in ok if r["name"].startswith(pref)]
        if not vals:
            continue
        allv = vals + [ctrl]
        rng = max(allv) - min(allv)
        spreads.append(rng)
        print(f"    {tag:<11} seed0 {ctrl:.2f}  ->  {' '.join(f'{v:.2f}' for v in vals)}"
              f"   range {rng:.2f}   implied sd ~{rng / 1.13:.2f}")
    if spreads:
        sd = max(spreads) / 1.13
        if sd >= 0.30:
            print("    implied sd >= 0.30: the ENTIRE loss axis (max +0.65) is inside noise,")
            print("    and runs 5-8 are uninterpretable at n=1 because their effects are +-0.3.")
            print("    Report the axis as null and lead with the ranking diagnosis. That is a")
            print("    RESULT, and it was pre-registered before these runs.")
        elif sd <= 0.15:
            print("    implied sd <= 0.15: +0.65 stands as a real effect. Seed 2 tomorrow turns")
            print("    the range into a proper interval.")
        else:
            print("    0.15 < sd < 0.30: +0.65 is marginal. Report scb3_sbb50 (+0.41) as the")
            print("    mechanism result and cls as untuned scatter. Seed 2 decides it.")
        cls65 = [r["test_map5095"] * 100 for r in ok if r["name"].startswith("y26_cls065_s")]
        if cls65 and cls65[0] > 55.60:
            print(f"    cls065 repeated at {cls65[0]:.2f} vs its 55.39 — the DIP did not")
            print("    reproduce, so the cls curve was scatter and 0.75 is not a tuned point.")
    # ---- augmentation and schedule ------------------------------------------
    probes = [r for r in ok if r["arm"] in ("aug", "sched")]
    if probes:
        best = max(probes, key=lambda x: x["test_map5095"])
        d = best["test_map5095"] * 100 - BASELINE
        print(f"    best aug/sched probe: {best['name']} {d:+.2f} vs identity")
        if d >= 0.5:
            print("    >= +0.5 on axes NOBODY has touched in 92 runs. They are open and deserve")
            print("    their own night before any further loss work.")
        else:
            print("    Flat. Four probes across augmentation and schedule close two obvious")
            print("    reviewer questions cheaply, which is worth the GPU on its own.")
    scb = {r["params"]["scb_branch"]: r["test_map5095"] * 100
           for r in ok if r["arm"] == "scb"}
    if len(scb) == 2:
        o, m = scb.get("one2one"), scb.get("one2many")
        print(f"    SCB on one2one {o:.2f}   on one2many {m:.2f}   on both {CTRL_SCB:.2f}")
        if o - BASELINE > 0.25 and m - BASELINE < 0.15:
            print("    The single-anchor account HOLDS. Report SCB as a one2one assignment")
            print("    fix and drop the 'applies to both branches' caveat.")
        elif m - BASELINE > 0.25 and o - BASELINE < 0.15:
            print("    The published explanation is WRONG. SCB acts through the AUXILIARY")
            print("    branch, so it is a curriculum effect, not a single-anchor fix. This")
            print("    is the most interesting outcome and it rewrites the mechanism section.")
        elif max(o, m) < CTRL_SCB - 0.25:
            print("    Neither branch alone reproduces +0.42 — the effect needs both, and the")
            print("    one2one-only justification is at best half the story.")
        else:
            print("    Both branches carry it. Attribution is not possible at n=1; either")
            print("    seed it or report SCB as a global assignment change and say so.")
    blend = {r["name"]: r["test_map5095"] * 100 for r in ok if r["arm"] == "blend"}
    if "y26_o2m_pin50" in blend:
        d = blend["y26_o2m_pin50"] - CTRL_SBB
        print(f"    o2m pinned at 0.5: {d:+.2f} vs the decaying schedule")
        if abs(d) < 0.2:
            print("    The curriculum is not load-bearing. sbb_q can be reported as a static")
            print("    size preference and the knife-edge stands as a knife-edge.")
        else:
            print("    The curriculum IS load-bearing, so every sbb_q number in the campaign")
            print("    is a q-x-schedule interaction. The sweep must be re-read, not re-run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    ap.add_argument("--arm", choices=["seed", "scb", "blend", "aug", "sched"], help="run only one arm")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    todo = RUNS
    if a.arm:
        todo = [r for r in todo if r["arm"] == a.arm]
    if a.names:
        todo = [r for r in todo if r["name"] in a.names]
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 84)
    print(f"  YOLO26 ROUND 16 — branch attribution + blend schedule ({len(todo)} runs)")
    print("  " + "  ".join(r["name"] for r in todo))
    print("=" * 84)
    if not preflight(todo):
        return

    res, out_path = [], f"{PROJECT_DIR}_results.json"
    for rc in todo:
        try:
            res.append(run_one(rc, a.seed))
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
