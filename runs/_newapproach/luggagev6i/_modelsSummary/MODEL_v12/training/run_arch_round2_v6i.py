#!/usr/bin/env python3
"""
ARCH ROUND 2 — 3 configs derived from CONFIRMED round-1/refine findings.

=============================================================================
WHAT THIS IS
=============================================================================
Standalone runner for three new topologies built AROUND the current best small-
object model, gctxP3 (gctx+snake STACKED at P3), which measured:

    arch_ls_shift    gctx@P2 + snake-k9 @P3,P4,P5      overall 55.98  small 51.22
    gctxP3           ls_shift + gctx ALSO @P3 (stacked) overall 56.02  small 51.58  <- best small
        ... but gctxP3 TRADED large: 60.09 -> 56.14 (-2.75%).

All three round-2 configs are derived from measured results, not guesses:
  F-a  stacking gctx+snake at a small level helps (gctxP3, +3.20% small)
  F-b  gctxP3 traded large away — goal: keep the small gain, protect large
  F-c  snake coverage is monotonic (more levels better, up to all-4)
  F-d  k9 >= k5 (kernel-down loses); k11 untested at coarse levels
  F-e  gctx belongs at P2 and P3 (small levels), not P4/P5

THE THREE:
  1. gctxp3_snake4      gctxP3 + snake ALSO at P2 — combines F-a (stack) + F-c
                        (coverage). gctxP3's one gap was no snake at P2. Most
                        likely single config to BEAT gctxP3 on small.
  2. gctx_stack_p2p3    stack gctx+snake at BOTH P2 and P3, P4/P5 snake untouched
                        — extends the win to both fine levels while leaving the
                        coarse path alone (F-b: should hold large better).
  3. gctxp3_k11coarse   gctxP3's P3 stack kept exactly; only the coarse snake
                        upsized k9->k11 @P4,P5 to recover the large capacity
                        gctxP3 sacrificed. If large recovers while small holds,
                        it is a STRICT improvement over gctxP3.

HONEST EXPECTATION: this is refinement, not breakthrough. The gains-per-config
have shrunk to ~0.3 small against a 0.12 noise floor, so a "win" here may be
within noise. gctxp3_k11coarse has the best shot at a MEANINGFUL result (a
better small/large trade-off rather than just a higher small number). Seed-
confirm any winner before believing an ordering.

All b32 (comparable to arch_ls_shift / gctxP3 / the refine runs), stock loss
(_ALL_OFF explicit), pretrained backbone transfer, 70 ep, 640, zero-gated
modules (identity at epoch 0).

READ per-size (CocoEvalAllFolders_luggage.py). Beat gctxP3 (small 51.58) —
ideally WITHOUT dropping large (56.14), and vs ls_shift's large 60.09.

REQUIRES the fork with DySample / ZGGlobalContext2 / ZGDSConv installed as the
active ultralytics. preflight() checks and aborts otherwise.

Usage:
    python run_arch_round2_v6i.py
    python run_arch_round2_v6i.py gctxp3_snake4
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
PROJECT_DIR = "runs_arch_round2_v6i"
YAML_DIR = "arch_yamls_round2_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                    # matches arch_ls_shift / gctxP3 / refine runs
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# b32 references (comparable — same batch).
REF = {"arch_ls_shift": 0.5598, "gctxP3": 0.5602}
GCTXP3_SMALL = 0.5158        # gctxP3 small mAP50-95 (the number to beat)
GCTXP3_LARGE = 0.5614        # gctxP3 large (do not drop below this)
LSSHIFT_LARGE = 0.6009       # ls_shift large (the no-trade reference)

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

# 1) gctxP3 + snake ALSO at P2 (fill gctxP3's one gap: no snake at P2)
TAIL_GCTXP3_SNAKE4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [24, 1, ZGDSConv, [256, 9]]                     # 25  P2 + snake k=9 (stacked)
  - [14, 1, ZGGlobalContext2, [256]]                # 26  P3 + gctx2
  - [26, 1, ZGDSConv, [256, 9]]                     # 27  P3 + snake k=9 (stacked, gctxP3 win)
  - [17, 1, ZGDSConv, [512, 9]]                     # 28  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 29  P5 + snake k=9
  - [[25, 27, 28, 29], 1, Detect, [nc]]             # 30
"""

# 2) stack gctx+snake at BOTH P2 and P3, coarse snake untouched (protect large)
TAIL_GCTX_STACK_P2P3 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [24, 1, ZGDSConv, [256, 9]]                     # 25  P2 + snake k=9 (stacked)
  - [14, 1, ZGGlobalContext2, [256]]                # 26  P3 + gctx2
  - [26, 1, ZGDSConv, [256, 9]]                     # 27  P3 + snake k=9 (stacked)
  - [17, 1, ZGDSConv, [512, 9]]                     # 28  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 29  P5 + snake k=9
  - [[25, 27, 28, 29], 1, Detect, [nc]]             # 30
"""

# 3) gctxP3 P3-stack kept; coarse snake k9->k11 @P4,P5 to recover large
TAIL_GCTXP3_K11COARSE = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (gctxP3 win, kept)
  - [17, 1, ZGDSConv, [512, 11]]                    # 27  P4 + snake k=11 (large recovery)
  - [20, 1, ZGDSConv, [1024, 11]]                   # 28  P5 + snake k=11 (large recovery)
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""


RUNS = [
    {"name": "gctxp3_snake4", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3_SNAKE4,
     "label": "gctxP3 winner + snake also at P2 — combine stacking + full coverage",
     "why": "The single config most likely to BEAT gctxP3 on small. Takes the "
            "gctxP3 tail (gctx+snake@P3, best small 51.58) and adds a snake at "
            "P2, combining the two confirmed positive trends: gctx+snake "
            "stacking (+3.20%) and monotonic snake coverage. Fills gctxP3's one "
            "gap (no snake at P2)."},

    {"name": "gctx_stack_p2p3", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTX_STACK_P2P3,
     "label": "gctx+snake stacked at BOTH P2 and P3 — extend the win, protect large",
     "why": "Stacks context+shape at both small-object levels while leaving P4/P5 "
            "snake untouched. Unlike gctxP3 (which traded large -2.75%) the "
            "coarse path is unchanged, so large should hold. Tests whether the "
            "stacking benefit compounds across the fine levels."},

    {"name": "gctxp3_k11coarse", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_GCTXP3_K11COARSE,
     "label": "gctxP3 (P3 stack kept) + coarse snake k9->k11 @P4P5 — recover large",
     "why": "Targets gctxP3's one weakness (large -2.75%). Keeps the P3 small "
            "stack exactly; only upsizes the coarse snake to k=11 for more "
            "receptive field on large objects. If large recovers while small "
            "holds, this is a STRICT improvement over gctxP3 — the best possible "
            "outcome of the round."},
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
    missing = [k for k, v in (ENV.get("modules") or {}).items() if not v]
    if missing or ENV.get("modules_error"):
        print(f"\n  [ABORT] custom modules not importable: "
              f"{missing or ENV.get('modules_error')}")
        return False
    clash = [r["name"] for r in todo
             if os.path.isdir(os.path.join(PROJECT_DIR, r["name"])) and not OVERWRITE_EXISTING]
    if clash:
        print(f"\n  [ABORT] run dirs already exist: {', '.join(clash)}")
        print("  Delete them, or set OVERWRITE_EXISTING=True.")
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
    kw.update(copy.deepcopy(_ALL_OFF))
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
            json.dump({**out, "yaml_text": rc["yaml"], "why": rc["why"], "env": ENV}, f, indent=2)
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
    ref = REF["gctxP3"]
    print(f"\n{'=' * 80}\n  ARCH ROUND 2 — v6i @640, stock loss, b32\n{'=' * 80}")
    print(f"{'run':<20}{'lvl':>4}{'mAP50':>9}{'mAP50-95':>11}{'vs gctxP3':>11}{'h':>6}")
    print("-" * 80)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        vs = "%+11.2f" % ((r["test_map5095"] - ref) * 100)
        print(f"{r['name']:<20}{r['levels']:>4}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{vs}{r['hours']:>6.1f}")
    print(f"\n  b32 refs: gctxP3 {REF['gctxP3']*100:.2f} (small 51.58, large 56.14) | "
          f"ls_shift {REF['arch_ls_shift']*100:.2f} (large 60.09)")
    print("  overall is near-noise between these; the target is SMALL mAP.")
    print("  READ per-size: CocoEvalAllFolders_luggage.py on best.pt.")
    print("  BEST outcome = small > 51.58 AND large recovered toward 60.09")
    print("  (a better trade-off, not just a higher small number). Seed-confirm any win.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<20} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 80}\n  ARCH ROUND 2 — {len(todo)} configs around gctxP3 (b32, stock loss)")
    print(f"  runs: {', '.join(r['name'] for r in todo)}  (~{1.6*len(todo):.0f} GPU-h)")
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
