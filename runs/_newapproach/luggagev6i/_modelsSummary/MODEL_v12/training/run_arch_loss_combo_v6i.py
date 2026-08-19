#!/usr/bin/env python3
"""
ARCH x LOSS COMBOS — losses that attack a bottleneck the ARCHITECTURE LEAVES OPEN.

=============================================================================
WHY THE FIRST COMBOS FAILED, AND THE RULE THAT FOLLOWS
=============================================================================
Measured on the P2 architecture (arch_ls_shift), stock-loss baseline:
    + SWA        -> -0.37   (SWA up-weights small-object BOX loss)
    + SWA+LB-TAL -> -0.84   (LB-TAL re-allocates small-object POSITIVES)
Both HURT. Diagnosis: SWA and LB-TAL COMPENSATE for a small-object SUPPLY
shortage. The P2 head SOLVES that shortage directly. Stacking them double-fixes
one bottleneck -> redundant / antagonistic. RULE: a loss+arch combo can only
help if the loss attacks a bottleneck the ARCH DOES NOT already fix.

=============================================================================
WHAT THE ARCH LEAVES OPEN (from the measured results)
=============================================================================
The arch fixes small SUPPLY (P2) and small FEATURES (snake/gctx). It does NOT
fix, and in one case MAKES WORSE:
  * HIGH-IoU LOCALISATION: the mAP50->95 ratio is stuck ~0.66 across BOTH the
    loss campaign AND the arch campaign. No axis has moved it.
  * LARGE high-IoU, specifically: arch_ls_shift HURT large mAP50-95 (60.09 vs a
    plain head; and at 896 large mAP50-95 56.82 < baseline_896 58.56) while large
    mAP50 stayed high -> large objects are FOUND but poorly localised at tight
    IoU. That is exactly a BOX-QUALITY problem.
  * CLASS IMBALANCE: bag AP 0.47 vs trolley 0.62 — a classification problem the
    geometric arch does not touch.

So the losses that can COMPLEMENT the arch (not substitute for it) are:
  - WIoU v3   -> high-IoU box quality + the stuck ratio + large's weakness
  - MPDIoU    -> alternative box metric, scale-robust, also targets high-IoU
  - class-wt  -> bag class imbalance (orthogonal to geometry entirely)
NONE of these touch small-object supply/assignment, so none re-create the
substitute failure of SWA/LB-TAL.

=============================================================================
THE RUNS — all on arch_ls_shift (the best model), single loss variable each
=============================================================================
  1. lsshift_wiou     arch_ls_shift + WIoU v3        (best bet: repairs large
                       high-IoU, targets the stuck ratio, no supply overlap)
  2. lsshift_mpdiou   arch_ls_shift + MPDIoU         (second box-quality metric,
                       cross-checks whether the ratio is movable at all)
  3. lsshift_clswt    arch_ls_shift + class-weighting (bag imbalance, fully
                       orthogonal axis)

REFERENCE: arch_ls_shift @640 stock loss = 55.98 / small 51.22 / large 60.09,
ratio(small) ~0.654. A combo WORKS if it beats 55.98 — most likely by
recovering LARGE high-IoU (mAP50-95 large > 60.09) and/or nudging the ratio,
WITHOUT dropping small below 51.22. Read the mAP50->95 RATIO, not just mAP.

HONEST ODDS: modest. The ratio has resisted 40+ runs; WIoU is the best-targeted
attempt but may only nudge it. class-wt may lift bag a little. Expect small,
possibly within-noise (0.12) effects. This is the last orthogonal lever, not a
guaranteed win.

All @640, b32 (matches arch_ls_shift exactly), stock loss otherwise, pretrained
transfer, 70 ep. REQUIRES the fork modules + WIoU/MPDIoU in the loss.

Usage:
    python run_arch_loss_combo_v6i.py
    python run_arch_loss_combo_v6i.py lsshift_wiou
"""

import sys
import time
import gc
import copy
import json
import os
import hashlib

import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_arch_loss_combo_v6i"
YAML_DIR = "arch_yamls_combo_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                    # matches arch_ls_shift exactly (clean vs its 55.98)
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# arch_ls_shift @640 stock-loss reference (the number to beat).
REF = 0.5598
REF_SMALL = 0.5122
REF_LARGE = 0.6009

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
    use_lbtal=False,
)

BACKBONE = """nc: 3
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]               # 0
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]        # 1
  - [-1, 2, C3k2,  [256, False, 0.25]]       # 2  backbone P2 (stride 4)
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]        # 3
  - [-1, 2, C3k2,  [512, False, 0.25]]       # 4  backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]              # 5
  - [-1, 4, A2C2f, [512, True, 4]]           # 6  backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]             # 7
  - [-1, 4, A2C2f, [1024, True, 1]]          # 8  backbone P5 (stride 32)
"""

HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                        # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, DySample, [2]]                        # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
"""

# arch_ls_shift tail (the best model) — identical across all three combos so
# the LOSS is the only variable.
TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

_ARCH_YAML = BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT


RUNS = [
    {"name": "lsshift_wiou", "batch": BATCH, "levels": 4, "yaml": _ARCH_YAML,
     "params": dict(_ALL_OFF, box_loss_type="wiou",
                    wiou_alpha=1.9, wiou_delta=3.0, wiou_momentum=0.02),
     "label": "arch_ls_shift + WIoU v3 — repair large high-IoU + the stuck ratio",
     "why": "The best-motivated combo. WIoU v3 focuses on ordinary-quality "
            "anchors via an EMA of the running IoU-loss mean, improving high-IoU "
            "box quality — exactly what the arch leaves open (mAP50->95 ratio "
            "stuck ~0.66) and what it made WORSE (large mAP50-95 dropped while "
            "large mAP50 held). WIoU does NOT touch supply/assignment, so it "
            "cannot re-create the SWA/LB-TAL substitute failure."},

    {"name": "lsshift_mpdiou", "batch": BATCH, "levels": 4, "yaml": _ARCH_YAML,
     "params": dict(_ALL_OFF, box_loss_type="mpdiou"),
     "label": "arch_ls_shift + MPDIoU — second box-quality metric, ratio cross-check",
     "why": "A different high-IoU box metric (min-point-distance IoU, scale-"
            "robust). If WIoU moves the ratio and MPDIoU does too, the ratio is "
            "genuinely movable by box loss; if neither does, the ~0.66 ceiling "
            "is structural. Either way it is a clean second data point on the "
            "one number no axis has shifted."},

    {"name": "lsshift_clswt", "batch": BATCH, "levels": 4, "yaml": _ARCH_YAML,
     "params": dict(_ALL_OFF, use_class_weighting=True, class_weight_mode="sqrt"),
     "label": "arch_ls_shift + class-weighting (sqrt) — bag imbalance, orthogonal axis",
     "why": "Fully orthogonal to the geometric arch: bag AP 0.47 vs trolley 0.62 "
            "is a CLASSIFICATION gap, not supply/assignment. sqrt inverse-freq "
            "weighting reweights the cls loss per class. Cannot overlap the arch's "
            "geometric fix, so it is the safest combo — though a blunt instrument."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def env_provenance():
    info = {"loss_md5": None, "modules": {}, "box_losses": {}}
    try:
        import ultralytics.utils.loss as _lm
        p = getattr(_lm, "__file__", None)
        if p and os.path.exists(p):
            src = open(p, "rb").read()
            info["loss_md5"] = hashlib.md5(src).hexdigest()[:12]
            txt = src.decode("utf-8", "ignore")
            for bl in ("wiou", "mpdiou", "focaler"):
                info["box_losses"][bl] = f"'{bl}'" in txt or f'"{bl}"' in txt
    except Exception as e:
        info["loss_error"] = str(e)
    try:
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T
        for name in ("DySample", "ZGGlobalContext2", "ZGDSConv"):
            info["modules"][name] = bool(
                hasattr(M, name) or name in getattr(T, "__dict__", {}) or
                any(hasattr(getattr(M, sub, None), name) for sub in dir(M)))
    except Exception as e:
        info["modules_error"] = str(e)
    return info


ENV = env_provenance()


def preflight(todo):
    print(f"  loss md5={ENV.get('loss_md5')}")
    print(f"  custom modules: {ENV.get('modules')}")
    print(f"  box losses in loss.py: {ENV.get('box_losses')}")
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if missing or ENV.get("modules_error"):
        print(f"\n  [ABORT] custom modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        return False
    # verify the box losses the runs need are present
    need = set()
    for r in todo:
        bt = r["params"].get("box_loss_type", "ciou")
        if bt in ("wiou", "mpdiou", "focaler"):
            need.add(bt)
    missing_bl = [b for b in need if not ENV.get("box_losses", {}).get(b)]
    if missing_bl:
        print(f"\n  [ABORT] box_loss_type(s) not in installed loss.py: {missing_bl}")
        return False
    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"])) and not OVERWRITE_EXISTING]
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        return False
    return True


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    for crit in (getattr(m, "criterion", None), getattr(trainer, "criterion", None)):
        if crit is not None:
            try:
                crit.epoch = epoch
                if hasattr(crit, "_sync_bbox_loss_state"):
                    crit._sync_bbox_loss_state()
            except Exception:
                pass


def run_one(rc):
    name, batch = rc["name"], rc["batch"]
    yaml_path = save_yaml(rc["yaml"], os.path.join(YAML_DIR, f"{name}.yaml"))
    print(f"\n{'=' * 76}\n  RUN {name}  ({rc['levels']} levels, batch {batch})\n"
          f"  {rc['label']}\n  yaml={yaml_path}  imgsz={IMG_SIZE}  epochs={EPOCHS}\n{'=' * 76}\n")
    t0 = time.time()
    model = YOLO(yaml_path)
    try:
        model.load(MODEL_WEIGHTS)
    except Exception as e:
        print(f"  [warn] backbone transfer failed: {e} — training from scratch")
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=batch,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=SEED,
              deterministic=True, exist_ok=OVERWRITE_EXISTING)
    kw.update(copy.deepcopy(rc["params"]))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "batch": batch, "levels": rc["levels"], "seed": SEED,
           "hours": hours, "weights": weights, "yaml": yaml_path,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_params.json"), "w") as f:
            json.dump({**out, "params": rc["params"], "why": rc["why"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                               batch=batch, device=DEVICE, workers=WORKERS,
                               project=PROJECT_DIR, name=f"{name}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    print(f"\n{'=' * 80}\n  ARCH x LOSS COMBOS — v6i @640, b32, on arch_ls_shift\n{'=' * 80}")
    print(f"{'run':<20}{'lvl':>4}{'mAP50':>9}{'mAP50-95':>11}{'vs ls_shift':>13}{'h':>6}")
    print("-" * 80)
    for r in sorted(res, key=lambda x: -(x["test_map5095"] if x["test_map5095"] == x["test_map5095"] else -9)):
        vs = "%+13.2f" % ((r["test_map5095"] - REF) * 100)
        print(f"{r['name']:<20}{r['levels']:>4}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{vs}{r['hours']:>6.1f}")
    print(f"\n  arch_ls_shift ref: {REF*100:.2f} (small {REF_SMALL*100:.2f}, large {REF_LARGE*100:.2f})")
    print("  vs ls_shift > 0 -> the loss COMPLEMENTED the arch (attacked an open bottleneck).")
    print("  The real test: per-size + the mAP50->95 RATIO. A WIoU/MPDIoU win should")
    print("  show up as LARGE mAP50-95 recovering toward/above 60.09 and/or the ratio")
    print("  rising above ~0.654 — NOT as a small-object gain (the arch already has that).")
    print("  Per-size: CocoEvalAllFolders_luggage.py on best.pt.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 80}\n  ARCH x LOSS COMBOS — {len(todo)} runs on arch_ls_shift (b32)")
    print(f"  losses that attack what the arch leaves OPEN (high-IoU / bag class)")
    print(f"  runs: {', '.join(r['name'] for r in todo)}  (~{1.7*len(todo):.0f} GPU-h)")
    print(f"{'=' * 80}\n")
    if not preflight(todo):
        sys.exit(1)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, "summary.json")
    res = []
    for r in todo:
        try:
            res.append(run_one(r))
        except Exception as e:
            print(f"\n  [ERROR] run '{r['name']}' failed: {e}")
            res.append({"name": r["name"], "batch": r["batch"], "levels": r["levels"],
                        "seed": SEED, "hours": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
