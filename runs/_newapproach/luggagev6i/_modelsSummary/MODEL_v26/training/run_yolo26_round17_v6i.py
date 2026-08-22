#!/usr/bin/env python3
r"""
YOLO26 ROUND 17 — BRANCH-SCOPED SCB + SEEDS (stock yolo26s, b82)

Nine runs, ~7.8 GPU-h. All b82 on the stock 3-level graph, so every row is batch
matched to y26_identity and drops straight into Table 1.


=============================================================================
WHAT ROUND 16 ESTABLISHED
=============================================================================
Four configs at two seeds each gave the first real noise estimate on this axis:

    config          seed0    seed1     mean
    scb3_sbb50      55.65    56.06    55.86
    cls075          55.89    55.46    55.68
    cls065          55.39    55.31    55.35
    identity        55.24    54.95    55.10

    pooled within-config sd = 0.24     ->  2 sd band = +-0.47

Stock is LAST at both seeds, Spearman rho = +0.8, and scb3_sbb50 is +0.76 over
stock (3.2 sd). The loss axis is real, but every single-seed number in the
campaign carries a +-0.47 band — wider than most effects it was used to claim.
Hence: every new config here gets TWO seeds before it is discussed.


=============================================================================
ARM A (runs 1-2) — THE CONFIG THE ATTRIBUTION POINTS AT
=============================================================================
Round 16 scoped SCB to one branch for the first time:

    SCB on one2many only    55.79   (+0.13)
    SCB on BOTH             55.66   (control)
    SCB on one2one only     54.98   (-0.68, 2.8 sd)

One2one-only is significantly WORSE, and one2many-only reproduces the whole
effect. So the published justification — "topk2=1 picks the single anchor that
produces every prediction, so beta over-trusts IoU there" — is contradicted:
SCB acts through the AUXILIARY branch.

The headline config still runs SCB on BOTH branches, so it is carrying that
-0.68 penalty. Removing it is the only mechanism change in this campaign
motivated by a measurement rather than an intuition.

There is a second reason to expect an interaction. sbb_invert=True puts one2one
on SMALL. With SCB also reweighting one2one's assignment by size, the two
mechanisms are fighting over the same branch. Scoping SCB to one2many gives
each branch exactly one mechanism.

    control   y26_scb3_sbb50   SCB both + SBB      55.86  (n=2)
    runs 1-2  y26_scb3o2m_sbb50                    ?      (n=2)

NOT a prediction. Nine directional predictions in this campaign, nine
falsified, every one optimistic. This is a hole in the table with an argument
attached, which is the only kind of run that has ever paid here.


=============================================================================
ARM B (runs 3-5) — THIRD SEEDS
=============================================================================
n=2 gives a range, not an interval. A third seed on the three configs that
matter turns "sd ~ 0.24" into something a reviewer can check, and it is the
cheapest defensible thing left on this axis.

    y26_scb3_sbb50_s2   the config that would be reported
    y26_identity_s2     the reference every delta is measured against
    y26_cls075_s2       the config that led at seed 0 and did not hold


=============================================================================
ARM C (runs 6-7) — CONFIRM THE ATTRIBUTION
=============================================================================
The reversal above rests on one run per branch. At sd 0.24 a single run has a
+-0.47 band, and this claim retires an explanation stated as fact in the
write-up, so it needs its own seed before it is published.


=============================================================================
ARM D (runs 8-9) — TWO CHEAP FOLLOW-UPS
=============================================================================
  8  scb2 scoped to one2many + SBB
     beta_small=2.0 was the WORST standalone SCB setting (55.05, the only point
     below baseline) and the BEST when paired with SBB (55.70). If the pairing
     story is right and the branch scoping is right, the strongest small-object
     push belongs on one2many with SBB opposite it. If it fails, the pairing
     story that half the loss section rests on is wrong, which is worth as much.

  9  headline loss + scale=0.75
     The only positive probe on an untouched axis (+0.42 vs the 2-seed identity
     mean, 1.8 sd). Orthogonal to the loss, never combined. multi_scale is NOT
     included: it measured -1.03 (4.3 sd) and that axis is closed.


=============================================================================
REQUIRES
=============================================================================
loss.py must expose scb_branch (the round-16 patch). The preflight asserts the
CONSUMER, not the config surface: rounds 4-6 lost ten runs to a key that
default.yaml accepted, the header printed, and loss.py ignored.

    Usage:
        python run_yolo26_round17_v6i.py                    # all nine, in order
        python run_yolo26_round17_v6i.py --arm combo        # runs 1-2
        python run_yolo26_round17_v6i.py --arm seed         # runs 3-5
        python run_yolo26_round17_v6i.py --arm attrib       # runs 6-7
        python run_yolo26_round17_v6i.py y26_scb2o2m_sbb50  # one by name
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
PROJECT_DIR = "runs_yolo26_round17_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 82  # matches y26_base_rep and every loss run in the campaign
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False  # y26_p2k2_hi was lost to exist_ok=True on a reused name

SD = 0.24  # pooled within-config sd, round 16, n=2 x 4 configs
CTRL_STOCK = 55.10  # y26_identity        2-seed mean (55.24 / 54.95)
CTRL_SBB = 55.86  # y26_scb3_sbb50      2-seed mean (55.65 / 56.06)
CTRL_CLS075 = 55.68  # y26_cls075          2-seed mean (55.89 / 55.46)
CTRL_SCB = 55.66  # y26_scb_b3          SCB on both, no SBB, n=1
CTRL_O2M = 55.79  # y26_scb3_o2m_only   n=1
CTRL_O2O = 54.98  # y26_scb3_o2o_only   n=1

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

_SCB = dict(tal_beta_small=3.0, tal_beta_ref_px=64.0)
_SBB = dict(sbb_q=0.5, sbb_invert=True)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


def _e(scb=(3.0, 64.0), on="both", sbb=0.0, **rest):
    """expect block: scb value, which branch, sbb q, plus optional cls / aug."""
    return dict(scb=scb, scb_on=on, sbb=sbb, blend=(0.8, 0.1, True), **rest)


RUNS = [
    # ---------------------------------------------------------------- ARM A
    {"name": "y26_scb3o2m_sbb50", "arm": "combo", "seed": 0, "ctrl": CTRL_SBB,
     "params": cfg(**_SCB, **_SBB, scb_branch="one2many"),
     "expect": _e(on="one2many", sbb=0.5),
     "label": "SCB 3.0 on ONE2MANY only + SBB 0.5 inv — the attribution applied",
     "why": "The headline config runs SCB on both branches and therefore carries the "
            "-0.68 that one2one-only measured. This removes it. It also stops SCB and "
            "SBB fighting over one2one, which sbb_invert=True has leaning SMALL."},

    # {"name": "y26_scb3o2m_sbb50_s1", "arm": "combo", "seed": 1, "ctrl": CTRL_SBB,
    #  "params": cfg(**_SCB, **_SBB, scb_branch="one2many"),
    #  "expect": _e(on="one2many", sbb=0.5),
    #  "label": "same, SEED 1 — paired from the start",
    #  "why": "Round 16 flipped its own conclusion twice on single-seed numbers. At "
    #         "sd 0.24 one run carries a +-0.47 band, so a new config is not discussed "
    #         "until it has two."},

    # ---------------------------------------------------------------- ARM B
    {"name": "y26_scb3_sbb50_s2", "arm": "seed", "seed": 2, "ctrl": CTRL_SBB,
     "params": cfg(**_SCB, **_SBB),
     "expect": _e(sbb=0.5),
     "label": "scb3_sbb50 SEED 2 — third point on the config that would be reported",
     "why": "n=2 is a range. Three points give an interval a reviewer can check, on "
            "the one config the paper would actually put forward (+0.76, 3.2 sd)."},

    {"name": "y26_identity_s2", "arm": "seed", "seed": 2, "ctrl": CTRL_STOCK,
     "params": dict(_ALL_OFF),
     "expect": _e(scb=None),
     "label": "stock yolo26s SEED 2 — third point on the reference",
     "why": "Every delta in Table 1 is measured against this. Its own spread is half "
            "the error budget and it moved 0.29 between seeds 0 and 1."},

    {"name": "y26_cls075_s2", "arm": "seed", "seed": 2, "ctrl": CTRL_CLS075,
     "params": cfg(**_SCB, **_SBB, cls=0.75),
     "expect": _e(sbb=0.5, cls=0.75),
     "label": "scb3_sbb50 + cls 0.75 SEED 2 — the config that led at seed 0",
     "why": "55.89 at seed 0 sent this project chasing a cls optimum that does not "
            "exist: seed 1 came back 55.46 and the curve is non-monotone "
            "(0.50/0.65/0.75/1.00 -> 55.65/55.39/55.89/55.17). A third point closes it "
            "either way."},

    # ---------------------------------------------------------------- ARM C
    # {"name": "y26_scb3_o2m_only_s1", "arm": "attrib", "seed": 1, "ctrl": CTRL_O2M,
    #  "params": cfg(**_SCB, scb_branch="one2many"),
    #  "expect": _e(on="one2many"),
    #  "label": "SCB on ONE2MANY only, SEED 1 — confirm the reversal",
    #  "why": "The claim that SCB acts through the auxiliary branch rests on one run per "
    #         "branch and it retires an explanation the write-up states as fact."},

    # {"name": "y26_scb3_o2o_only_s1", "arm": "attrib", "seed": 1, "ctrl": CTRL_O2O,
    #  "params": cfg(**_SCB, scb_branch="one2one"),
    #  "expect": _e(on="one2one"),
    #  "label": "SCB on ONE2ONE only, SEED 1 — the other half",
    #  "why": "-0.68 is 2.8 sd, the largest structural effect of round 16. If it "
    #         "reproduces, the one2one justification is dead and run 1 is the config."},

    # ---------------------------------------------------------------- ARM D
    {"name": "y26_scb2o2m_sbb50", "arm": "probe", "seed": 0, "ctrl": CTRL_SBB,
     "params": cfg(tal_beta_small=2.0, tal_beta_ref_px=64.0, **_SBB, scb_branch="one2many"),
     "expect": _e(scb=(2.0, 64.0), on="one2many", sbb=0.5),
     "label": "SCB 2.0 on ONE2MANY only + SBB — the strongest push, correctly placed",
     "why": "beta_small=2.0 was the WORST standalone SCB setting (55.05, the only point "
            "below baseline) and the BEST when paired with SBB (55.70). If both the "
            "pairing story and the branch scoping hold, the strongest small-object push "
            "belongs on one2many with SBB opposite it. A failure falsifies the pairing "
            "account that half the loss section rests on, which is worth as much."},

    {"name": "y26_scb3_sbb50_scale75", "arm": "probe", "seed": 0, "ctrl": CTRL_SBB,
     "params": cfg(**_SCB, **_SBB, scale=0.75),
     "expect": _e(sbb=0.5, aug={"scale": 0.75}),
     "label": "headline loss + scale 0.75 — the one positive augmentation probe",
     "why": "scale=0.75 measured +0.42 vs the 2-seed identity mean (1.8 sd), the only "
            "positive result on an axis none of 92 runs had touched. Orthogonal to the "
            "loss and never combined with it. multi_scale is excluded on purpose: "
            "-1.03 at 4.3 sd, that axis is closed."},
]


def preflight(todo):
    """Prove the mechanism is REACHABLE before spending a night of GPU.

    Checks the consumer (loss.py source), not the config surface. Rounds 4-6 passed a
    preflight that only verified tal.py had the class and default.yaml took the keys.
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

    if any(r["params"]["scb_branch"] != "both" for r in todo):
        checks = {
            "E2ELoss reads scb_branch": "scb_branch" in inspect.getsource(E2ELoss.__init__),
            "default.yaml accepts scb_branch": "scb_branch" in DEFAULT_CFG_DICT,
        }
    else:
        checks = {"scb_branch not required for the selected arms": True}
    for k, v in checks.items():
        print(f"  {k:<42} {v}")
        ok &= bool(v)
    if not ok:
        print("\n  [ABORT] apply the round-16 loss.py / default.yaml patch first.")
        print("          or run only:  --arm seed")
        return False

    print()
    for r in todo:
        d = {k: v for k, v in r["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  {r['name']:<24} {r['arm']:<7} seed{r['seed']}  vs {r['ctrl']:.2f}  |  {d}")
    print(f"\n  {len(todo)} runs, ~{0.87 * len(todo):.1f} GPU-h   |   sd={SD}, 2sd band=+-{2 * SD:.2f}")
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

        # ---- SCB, and WHICH branch it is on. The scoping IS the experiment here,
        # so both the presence and the ABSENCE are asserted per branch.
        if e["scb"] is None:
            if a1.scb_enabled() or a2.scb_enabled():
                raise RuntimeError(f"{rc['name']}: SCB is live but was not requested")
            seen.append("SCB off")
        else:
            want_b, want_r = e["scb"]
            on = e["scb_on"]
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

        # ---- SBB
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

        # ---- the blend schedule is stock in every run here
        w = e["blend"]
        got = (float(crit.o2m_copy), float(crit.final_o2m), bool(crit.o2m_decay))
        if abs(got[0] - w[0]) > 1e-6 or abs(got[1] - w[1]) > 1e-6 or got[2] != w[2]:
            raise RuntimeError(f"{rc['name']}: blend is {got}, expected {w}")

        # ---- everything else must be provably off
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
        if abs(float(h.cls) - e.get("cls", 0.5)) > 1e-6:
            raise RuntimeError(f"{rc['name']}: hyp.cls={h.cls}, expected {e.get('cls', 0.5)}")
        # augmentation lives on the TRAINER args, a different consumer
        for k, v in e.get("aug", {}).items():
            got_a = getattr(trainer.args, k, None)
            if got_a is None or abs(float(got_a) - float(v)) > 1e-6:
                raise RuntimeError(f"{rc['name']}: trainer.args.{k}={got_a}, expected {v}")
            seen.append(f"{k}={got_a}")
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
    kw.update(rc["params"])
    results = model.train(**kw)
    if not state["verified"]:
        raise RuntimeError(f"{name}: the mechanism guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": rc["arm"], "ctrl": rc["ctrl"], "params": rc["params"],
           "expect": {k: v for k, v in rc["expect"].items()}, "seed": seed,
           "model": MODEL_WEIGHTS, "imgsz": IMG_SIZE, "batch": BATCH, "hours": hours,
           "weights": weights, "mechanism_verified": True,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    # *_params.json so the eval script's glob binds metrics to a CONFIG, not to
    # directory order — round 16 was mis-evaluated for exactly this reason.
    try:
        with open(os.path.join(save_dir, "round17_params.json"), "w") as f:
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


# seed-0/1 values already measured, so a single new run completes an interval
PRIOR = {
    "y26_scb3_sbb50": [55.65, 56.06],
    "y26_identity": [55.24, 54.95],
    "y26_cls075": [55.89, 55.46],
    "y26_scb3_o2m_only": [55.79],
    "y26_scb3_o2o_only": [54.98],
}


def summarise(res):
    ok = [r for r in res if r["test_map5095"] == r["test_map5095"]]
    if not ok:
        print("\nno completed runs.")
        return
    print("\n" + "=" * 78)
    print("  ROUND 17 — RESULTS")
    print("=" * 78)
    print(f"{'run':<26}{'arm':<8}{'mAP50-95':>10}{'vs ctrl':>9}{'vs stock':>10}{'hours':>7}")
    print("-" * 70)
    print(f"{'y26_identity (n=2)':<26}{'ref':<8}{CTRL_STOCK:>10.2f}{'-':>9}{0.0:>+10.2f}{'-':>7}")
    print(f"{'y26_scb3_sbb50 (n=2)':<26}{'ctrl':<8}{CTRL_SBB:>10.2f}{0.0:>+9.2f}"
          f"{CTRL_SBB - CTRL_STOCK:>+10.2f}{'-':>7}")
    print("-" * 70)
    for r in sorted(ok, key=lambda x: -x["test_map5095"]):
        v = r["test_map5095"] * 100
        print(f"{r['name']:<26}{r['arm']:<8}{v:>10.2f}{v - r['ctrl']:>+9.2f}"
              f"{v - CTRL_STOCK:>+10.2f}{r['hours']:>7.2f}")

    print("\n  READ IT")
    got = {r["name"]: r["test_map5095"] * 100 for r in ok}

    # ---- arm A: the new config, paired
    new = [got[n] for n in ("y26_scb3o2m_sbb50", "y26_scb3o2m_sbb50_s1") if n in got]
    if len(new) == 2:
        m = sum(new) / 2
        d = m - CTRL_SBB
        print(f"    scb3o2m_sbb50  {new[0]:.2f} / {new[1]:.2f}  mean {m:.2f}   "
              f"vs scb3_sbb50 {CTRL_SBB:.2f}  {d:+.2f}")
        if d >= 2 * SD:
            print("    Clears 2 sd. Scoping SCB to one2many is the config to report, and the")
            print("    branch attribution stops being an aside and becomes the mechanism.")
        elif d <= -2 * SD:
            print("    Significantly WORSE. SCB needs both branches after all, and the")
            print("    one2one-only result was a single-seed artefact.")
        else:
            print("    Inside the null band. Removing SCB from one2one costs nothing and")
            print("    buys nothing — report the simpler config and say the branch makes no")
            print("    difference, which still contradicts the single-anchor justification.")

    # ---- arm B/C: third and second seeds -> intervals
    for tag, key in (("scb3_sbb50", "y26_scb3_sbb50"), ("identity", "y26_identity"),
                     ("cls075", "y26_cls075"), ("o2m_only", "y26_scb3_o2m_only"),
                     ("o2o_only", "y26_scb3_o2o_only")):
        vals = list(PRIOR[key])
        for suffix in ("_s1", "_s2"):
            if key + suffix in got:
                vals.append(got[key + suffix])
        if len(vals) < 3:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        print(f"    {tag:<12} n={len(vals)}  {' '.join(f'{v:.2f}' for v in vals)}  "
              f"mean {mean:.2f}  sd {sd:.2f}")

    # ---- arm C verdict
    if "y26_scb3_o2m_only_s1" in got and "y26_scb3_o2o_only_s1" in got:
        mo = (CTRL_O2M + got["y26_scb3_o2m_only_s1"]) / 2
        oo = (CTRL_O2O + got["y26_scb3_o2o_only_s1"]) / 2
        print(f"    SCB placement  one2many {mo:.2f}   one2one {oo:.2f}   gap {mo - oo:+.2f}")
        if mo - oo >= 2 * SD:
            print("    Confirmed at n=2 per branch: SCB acts through the AUXILIARY branch.")
            print("    The single-anchor justification in the write-up is falsified and must")
            print("    be replaced, not softened.")
        else:
            print("    Gap did not reproduce. Report SCB as a global assignment change and")
            print("    drop the branch story entirely.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="run only these runs, by name")
    ap.add_argument("--arm", choices=["combo", "seed", "attrib", "probe"])
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
    print(f"  YOLO26 ROUND 17 — branch-scoped SCB + seeds ({len(todo)} runs)")
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
