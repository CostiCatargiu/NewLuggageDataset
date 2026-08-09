#!/usr/bin/env python3
"""
RESOLUTION STUDY @896 — baseline + the two best arch models, TRAIN and EVAL @896.

=============================================================================
WHY THIS IS A SEPARATE STUDY, NOT A DROP-IN IMPROVEMENT
=============================================================================
"THE 896 LESSON" (stamped in every runner in this project): eval imgsz MUST
equal train imgsz. So these three TRAIN at 896 and EVAL at 896. That means the
numbers here are a NEW REGIME and are ONLY comparable to EACH OTHER — never to
the 640 results (baseline 54.77, arch_ls_shift 55.98, gctxP3 56.02).

The whole point is the INTERNAL 896 comparison:
    baseline@896   vs   arch_ls_shift@896   vs   gctxP3@896
so the architecture delta is measured at the higher resolution on its own
footing. A single arch@896 number compared to a 640 baseline would be
architecture+resolution entangled — exactly the confound this study avoids by
including baseline@896.

WHY 896 AT ALL. Data is 60% small objects; the 640->896 pairs elsewhere in the
project moved +1.25 to +1.64 pp — the largest single effect of any axis.
Resolution is the highest-ROI lever for small objects, and it has never been
combined with the P2 architecture.

DEPLOYMENT CAVEAT (report it): native frames are 640x360. Training at 896
upscales beyond native resolution, so a gain here may not fully transfer to a
640x360 deployment feed. This is a scaling analysis, not necessarily the
production config.

=============================================================================
THE THREE RUNS
=============================================================================
  1. baseline_896      stock yolov12s, stock loss — the 896 REFERENCE. Without
                       it the two arch numbers cannot be attributed to topology.
  2. arch_ls_shift_896 best balanced model (640: 55.98, all buckets up)
  3. gctxP3_896        best small model (640: 56.02, small 51.58)

BATCH: 896 + a P2 head is heavy. Batch is set per-run to the largest that fits
(baseline can go higher than the 4-level heads). Batch differs across the three,
so — like the original arch port — the arch-vs-baseline delta carries a step-
count offset. It is the best that fits; note it when reading. The INTERNAL
arch_vs_arch comparison (ls_shift vs gctxP3, same batch) stays clean.

READ per-size (CocoEvalAllFolders_luggage.py @896). The question: does the P2
architecture STILL beat a plain head once both have 896 pixels, and does small
mAP rise further than at 640?

REQUIRES the fork with DySample / ZGGlobalContext2 / ZGDSConv. preflight aborts
if the custom modules or the loss are missing.

Usage:
    python run_arch_896_v6i.py
    python run_arch_896_v6i.py baseline_896
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
PROJECT_DIR = "runs_arch_896_v6i"
YAML_DIR = "arch_yamls_896_v6i"
EPOCHS = 70
IMG_SIZE = 896                # TRAIN and EVAL both 896 — the 896 lesson
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# 640 references — for CONTEXT ONLY. Do NOT compute an 896-vs-640 delta from
# these; the resolution differs. The valid comparison is within this study.
REF_640 = {"baseline": 0.5477, "arch_ls_shift": 0.5598, "gctxP3": 0.5602}

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

# Stock 3-level head for the baseline (no P2, no custom modules).
HEAD_STOCK = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 9
  - [[-1, 6], 1, Concat, [1]]                     # 10
  - [-1, 2, A2C2f, [512, False, -1]]              # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 12
  - [[-1, 4], 1, Concat, [1]]                     # 13
  - [-1, 2, A2C2f, [256, False, -1]]              # 14  P3 head
  - [-1, 1, Conv, [256, 3, 2]]                    # 15
  - [[-1, 11], 1, Concat, [1]]                    # 16
  - [-1, 2, A2C2f, [512, False, -1]]              # 17  P4 head
  - [-1, 1, Conv, [512, 3, 2]]                    # 18
  - [[-1, 8], 1, Concat, [1]]                     # 19
  - [-1, 2, C3k2, [1024, True]]                   # 20  P5 head
  - [[14, 17, 20], 1, Detect, [nc]]              # 21  stock 3-level Detect
"""

# DySample head (used by both arch models).
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

# arch_ls_shift: gctx@P2 + snake-k9 @P3,P4,P5
TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# gctxP3: gctx@P2 + gctx+snake STACKED @P3 + snake @P4,P5
TAIL_GCTXP3 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (STACKED)
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""


RUNS = [
    {"name": "baseline_896", "batch": 28, "levels": 3,
     "yaml": BACKBONE + HEAD_STOCK,
     "label": "stock yolov12s @896 — the 896 REFERENCE (train+eval 896)",
     "why": "The reference every 896 number is measured against. Without it, "
            "arch@896 vs baseline@640 is architecture+resolution entangled. "
            "3-level head fits a larger batch than the P2 models; the batch "
            "offset applies to the arch-vs-baseline delta (note it), but the "
            "arch-vs-arch comparison stays clean."},

    {"name": "arch_ls_shift_896", "batch": 16, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT,
     "label": "arch_ls_shift @896 — best balanced model (640: 55.98, all buckets up)",
     "why": "The balanced winner at 640, now with 896 pixels. Both P2 supply AND "
            "resolution attack the small-object bottleneck; this is the first "
            "time they are combined. Does the architecture STILL beat a plain "
            "head once both have high resolution?"},

    {"name": "gctxP3_896", "batch": 16, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3,
     "label": "gctxP3 @896 — best small model (640: 56.02, small 51.58)",
     "why": "The small-object winner at 640. Same batch as arch_ls_shift_896, so "
            "the two arch models are directly comparable at 896. Tests whether "
            "the gctxP3 small-object edge holds, grows, or washes out once "
            "resolution already boosts small objects."},
]


def save_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def env_provenance():
    info = {"loss_md5": None, "modules": {}}
    try:
        import ultralytics.utils.loss as _lm
        p = getattr(_lm, "__file__", None)
        if p and os.path.exists(p):
            info["loss_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
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
    # baseline_896 needs no custom modules; the arch runs do.
    needs_modules = any(r["levels"] == 4 for r in todo)
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if needs_modules and (missing or ENV.get("modules_error")):
        print(f"\n  [ABORT] custom modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        return False
    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"])) and not OVERWRITE_EXISTING]
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        print("  Delete them, or set OVERWRITE_EXISTING=True.")
        return False
    print("\n  REMINDER: train AND eval are both 896 here (the 896 lesson). These")
    print("  numbers are comparable ONLY to each other, not to the 640 results.")
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
    print(f"\n{'=' * 76}\n  RUN {name}  ({rc['levels']} levels, batch {batch}, imgsz {IMG_SIZE})\n"
          f"  {rc['label']}\n  yaml={yaml_path}  epochs={EPOCHS}\n{'=' * 76}\n")
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
    kw.update(copy.deepcopy(_ALL_OFF))
    results = model.train(**kw)
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    weights = os.path.join(save_dir, "weights", "best.pt")
    out = {"name": name, "batch": batch, "levels": rc["levels"], "seed": SEED,
           "imgsz": IMG_SIZE, "hours": hours, "weights": weights, "yaml": yaml_path,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "arch_params.json"), "w") as f:
            json.dump({**out, "yaml_text": rc["yaml"], "why": rc["why"], "env": ENV}, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")
    try:
        # EVAL AT 896 — the 896 lesson. Same imgsz as training.
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
    base = next((r["test_map5095"] for r in res if r["name"] == "baseline_896"), None)
    print(f"\n{'=' * 84}\n  RESOLUTION STUDY @896 — v6i, stock loss (train+eval 896)\n{'=' * 84}")
    print(f"{'run':<20}{'b':>4}{'lvl':>5}{'mAP50':>9}{'mAP50-95':>11}"
          f"{'vs base896' if base else '':>11}{'(640)':>9}{'h':>6}")
    print("-" * 84)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        vb = ("%+11.2f" % ((r["test_map5095"] - base) * 100)) if base else f"{'':>11}"
        ref640 = REF_640.get(r["name"].replace("_896", "").replace("baseline", "baseline"), None)
        r640 = f"{ref640*100:8.2f}" if ref640 else f"{'—':>8}"
        print(f"{r['name']:<20}{r['batch']:>4}{r['levels']:>5}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{vb}{r640:>9}{r['hours']:>6.1f}")
    print(f"\n  vs base896 = the CLEAN architecture delta at 896 (arch runs share batch 16).")
    print("  The (640) column is CONTEXT ONLY — do NOT compute an 896-640 delta from it,")
    print("  the resolution differs. baseline_896 vs baseline_640 (54.77) shows the raw")
    print("  resolution effect; arch@896 vs base@896 shows architecture AT 896.")
    print("  NOTE baseline runs b28, arch runs b16 -> the arch-vs-base delta carries a")
    print("  step-count offset; arch-vs-arch (ls_shift vs gctxP3, both b16) is clean.")
    print("  Per-size: CocoEvalAllFolders_luggage.py @896 on best.pt.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 84}\n  RESOLUTION STUDY @896 — {len(todo)} runs (train+eval 896)")
    print(f"  runs: {', '.join(r['name'] for r in todo)}")
    print(f"  NOTE: 896 + P2 head is heavy; batch is per-run (base 28, arch 16).")
    print(f"{'=' * 84}\n")
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
                        "seed": SEED, "imgsz": IMG_SIZE, "hours": float("nan"),
                        "test_map50": float("nan"), "test_map5095": float("nan"),
                        "error": str(e)})
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    summarise(res, out_path)
