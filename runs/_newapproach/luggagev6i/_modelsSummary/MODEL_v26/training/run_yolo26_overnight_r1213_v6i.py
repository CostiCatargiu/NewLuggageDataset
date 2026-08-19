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
import sys
import time

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolo26s.pt"
PROJECT_DIR = "runs_yolo26_overnight_r1213_v6i"
CFG_P2 = "yolo26-p2.yaml"  # 4 levels, plain nn.Upsample -> deterministic
CFG_3LVL = "yolo26.yaml"  # 3 levels, stock graph

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

BASELINE = 55.24  # y26_base_rep      b82, 3 levels
BEST_LOSS = 55.65  # y26_scb3_sbb50    the base config for the loss arm
BEST_RAW = 55.70  # y26_scb2_sbb50    the loss bar
STOCK_B32 = 55.76  # y26_stock_b32     3 lvl, c3=128, PRETRAINED head
P2_B32 = 55.03  # y26_p2_b32        4 lvl, c3= 64, fresh head

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
    {"name": "y26_p2_headref128", "arm": "arch", "cfg": CFG_P2, "head_ch": 128,
     "want_c3": 128, "want_nl": 4,
     "label": "yolo26-p2, head width pinned at 128 (the stock 3-level value)",
     "why": "Isolates head width from pyramid depth. Compare against y26_p2_b32 = 55.03: "
            "both 4-level, both fresh-head, so c3 is the only variable."},

    {"name": "y26_3lvl_head64", "arm": "arch", "cfg": CFG_3LVL, "head_ch": 64,
     "want_c3": 64, "want_nl": 3,
     "label": "stock 3-level with the head SHRUNK to 64 (the P2 graph's value)",
     "why": "The cell that makes run 1 attributable. Gives p2@64 - 3lvl@64 as the pure "
            "P2-level effect with width AND transfer held, and exposes the pretrained-head "
            "penalty by subtraction. Without it run 1 cannot attribute anything."},

    # ---------------------------------------------------------------- LOSS
    {"name": "y26_scb3_sbb50_cls075", "arm": "loss", "params": cfg(cls=0.75),
     "expect": {**_BASE_EXPECT, "cls": 0.75},
     "label": "cls gain 0.5 -> 0.75 on the best config",
     "why": "The only untouched gain in 73 runs. loss[1] *= hyp.cls is the only term it "
            "scales, and wrong-class is 32-37% of what outranks the median TP. Low point "
            "first: loss[1] sums BCE over every anchor while loss[0] sums over positives "
            "only, so the balance is already tilted."},

    {"name": "y26_scb3_sbb50_cls10", "arm": "loss", "params": cfg(cls=1.0),
     "expect": {**_BASE_EXPECT, "cls": 1.0},
     "label": "cls gain 0.5 -> 1.0 on the best config",
     "why": "Second point makes cls a DIRECTION rather than a lone guess. Both flat closes "
            "the axis and confirms the classification deficit is immovable from the loss."},

    {"name": "y26_a075_scb3_sbb50", "arm": "loss", "params": cfg(tal_alpha=0.75),
     "expect": {**_BASE_EXPECT, "alpha": 0.75},
     "label": "tal_alpha 0.5 -> 0.75 on top of SCB + SBB",
     "why": "align_metric = score^alpha * IoU^beta. The two best single mechanisms "
            "(alpha075 +0.35, beta_small=3.0 +0.42) live in that one expression and have "
            "never been combined. Unlike SNL1+SCB they compose multiplicatively."},

    {"name": "y26_scb3_sbb50_diou", "arm": "loss", "params": cfg(iou_type="diou"),
     "expect": {**_BASE_EXPECT, "iou_type": "diou"},
     "label": "DIoU — delete CIoU's aspect term instead of replacing it",
     "why": "EIoU REPLACED it and cost 7.80 on large. DIoU deletes it. CIoU's aspect "
            "gradients satisfy dv/dw = -(h/w) dv/dh, always opposite in sign, so it "
            "regresses aspect RATIO and never magnitude — and this dataset is 70.6% tall "
            "boxes at class-stable ratios. giou/diou were in IOU_FLAGS all campaign and "
            "only eiou was ever run."},
]


def preflight(todo):
    """Prove every mechanism is reachable BEFORE spending a night of GPU."""
    print("=" * 78)
    print("  PREFLIGHT")
    print("=" * 78)
    arch = [r for r in todo if r["arm"] == "arch"]
    loss = [r for r in todo if r["arm"] == "loss"]

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
        if r["arm"] == "arch":
            bits = f"{r['cfg']}  c3={r['want_c3']}  levels={r['want_nl']}"
        else:
            p = r["params"]
            bits = f"SCB {p['tal_beta_small']}@{p['tal_beta_ref_px']}px + SBB q={p['sbb_q']} inv={p['sbb_invert']}"
            for k in ("cls", "alpha", "iou_type"):
                if k in r["expect"]:
                    bits += f"  +  {k}={r['expect'][k]}"
        print(f"  {r['name']:<26}{r['arm']:<6}b{b:<4}{bits}")

    print()
    print(f"  loss bar {BEST_RAW} (base {BEST_LOSS})   arch refs: stock_b32 {STOCK_B32}, p2_b32 {P2_B32}")
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


def run_one(rc):
    name, arm = rc["name"], rc["arm"]
    batch = BATCH_ARCH if arm == "arch" else BATCH_LOSS
    print()
    print("=" * 78)
    print(f"  RUN {name}   [{arm.upper()}]")
    print(f"  {rc['label']}")
    print("=" * 78)
    print(f"  batch={batch}  imgsz={IMG_SIZE}  epochs={EPOCHS}  seed={SEED}")
    if arm == "loss":
        diff = {k: v for k, v in rc["params"].items() if _ALL_OFF.get(k, "__") != v}
        print(f"  differs from _ALL_OFF: {diff}")
    else:
        print(f"  cfg={rc['cfg']}  head_ch={rc['head_ch']}  -> c3={rc['want_c3']}, levels={rc['want_nl']}")
    print()
    t0 = time.time()

    Detect.head_ch = rc.get("head_ch", 0)  # must be 0 for loss runs
    try:
        if arm == "arch":
            model = YOLO(rc["cfg"])
            model.load(MODEL_WEIGHTS)  # backbone/neck transfer; heads only where shapes match
            extra = {}
        else:
            model = YOLO(MODEL_WEIGHTS)
            extra = rc["params"]
        state = attach_guard(model, rc)
        results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
                              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
                              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
                              deterministic=True, exist_ok=OVERWRITE_EXISTING, **extra)
    finally:
        Detect.head_ch = 0  # never leak the class attribute into a later build

    if not state["verified"]:
        raise RuntimeError(f"{name}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "arm": arm, "batch": batch, "seed": SEED, "hours": hours,
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

        p2_128, l3_64 = by.get("y26_p2_headref128"), by.get("y26_3lvl_head64")
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

    loss = [r for r in ok if r["arm"] == "loss"]
    if loss:
        print("\n" + "=" * 78)
        print("  LOSS — vs base 55.65, bar 55.70")
        print("=" * 78)
        print(f"{'run':<26}{'mAP50-95':>10}{'vs base':>10}{'vs bar':>9}{'hours':>8}")
        print("-" * 63)
        for r in sorted(loss, key=lambda x: -x["test_map5095"]):
            v = r["test_map5095"] * 100
            print(f"{r['name']:<26}{v:>10.2f}{v - BEST_LOSS:>+10.2f}{v - BEST_RAW:>+9.2f}{r['hours']:>8.2f}")

        c = sorted((r for r in loss if "cls" in r.get("expect", {})),
                   key=lambda x: x["expect"]["cls"])
        if len(c) == 2:
            d = [r["test_map5095"] * 100 - BEST_LOSS for r in c]
            print(f"\n    cls 0.75 -> {d[0]:+.2f}   cls 1.00 -> {d[1]:+.2f}")
            if max(d) < 0.2:
                print("    Both flat. The last untouched gain is closed and the classification")
                print("    deficit is confirmed immovable from the loss.")
            elif d[1] > d[0] > 0:
                print("    Monotone and positive — a real direction. A third point is justified.")
            else:
                print("    Non-monotone. Single-seed spikes on a knife-edge base are how SCB")
                print("    looked for two days. Do not promote without a second seed.")
        best = max(r["test_map5095"] * 100 for r in loss)
        if best <= BEST_RAW:
            print(f"\n    Nothing beat {BEST_RAW}. With four single-seed draws that is the")
            print("    expected outcome and it leaves the campaign's conclusions intact.")
        else:
            print(f"\n    Best is {best:.2f}. The expected MAXIMUM of four draws sits ~0.25-0.35")
            print("    above the mean by selection alone — confirm on a second seed first.")
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
