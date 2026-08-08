#!/usr/bin/env python3
"""
ARCH REFINE — 4 configs AROUND the best topology (arch_ls_shift, 55.98) to look
for further improvement. Same backbone + DySample head + P2 tail as the port
sweep; only the P3/P4/P5 module choice varies.

=============================================================================
WHAT WE ARE REFINING, and why these four
=============================================================================
Measured v6i @640, stock loss, batch 32 (all these run at b32, so they are
mutually comparable with each other AND with the existing b32 port runs
arch_ls_shift / arch_levelspec / arch_ls_k5):

    arch_ls_shift   gctx2@P2      + snake-k9 @P3,P4,P5    overall 55.98  small 51.22
    arch_levelspec  gctx2@P2,P3   + snake-k9 @P4,P5       overall 55.90  small 51.36
    arch_ls_k5      gctx2@P2,P3   + snake-k5 @P4,P5       overall 55.68  small 50.78

The signal in those three:
  * ls_shift wins OVERALL (snake pushed down to P3).
  * levelspec wins SMALL (gctx kept at P3 instead of snake).
  => the gctx/snake boundary at P3 is the live knob, and kernel size at the
     coarse levels is unresolved (k9 vs k5 was 55.90 vs 55.68 but confounded
     with the P3 choice). These four isolate those two knobs around ls_shift.

THE FOUR (each a single, named change from ls_shift):

  1. ls_shift_k5      ls_shift with snake k9 -> k5 at P3,P4,P5.
        Pure kernel-size delta on the WINNER. ZGDSConvV6's geometry note: on
        v6i k=9 covers 2.6-5.3x the object at P4/P5 (mean box 39x55). k=5 is
        the size-matched choice. If ls_shift's snake is oversized, this gains.

  2. ls_shift_gctxP3  gctx2 @P2,P3 + snake @P4,P5, snake k=9  == arch_levelspec
        ... but we ALSO keep a snake at P3 IN ADDITION to gctx (both blocks,
        gctx then snake), testing whether P3 wants BOTH scene-context AND the
        shape prior rather than one or the other. This is the "why choose"
        variant the levelspec-vs-ls_shift split invites.

  3. ls_shift_gctxP3P4  gctx2 @P2,P3,P4 + snake @P5 only (k=9).
        Push global context DEEPER. Data is 60% small; ls_shift/levelspec both
        cap gctx at P2 or P3. If small/medium want scene context more than a
        shape prior, moving the boundary down to P4 should help small+medium
        while only P5 (7.7% of GTs) keeps the snake.

  4. ls_shift_snakeP2  snake-k5 @P2 + gctx2 also @P2 + snake-k9 @P3,P4,P5.
        The one place NO config has put a snake: the stride-4 level. P2 is where
        the smallest objects resolve; a size-matched (k5) shape prior there,
        added to the existing gctx, tests whether the finest level benefits from
        a shape prior at all. Kept k5 because objects at P2 are ~10-14 cells and
        k9 would still undershoot — k5 is the cheaper probe.

All four: DySample head, P2 tail, zero-gated modules (identity at epoch 0),
pretrained backbone transfer, stock loss (_ALL_OFF explicit), b32, 70 ep, 640.

READ per-size buckets (CocoEvalAllFolders_luggage.py). ls_shift ref: overall
55.98 / small 51.22 / large 60.09. A win must beat that on small WITHOUT
dropping large (the trap the loss combos fell into).

REQUIRES the fork with DySample / ZGGlobalContext2 / ZGDSConv / ZGDSConvV6
installed as the active ultralytics. preflight() checks and aborts otherwise.

Usage:
    python run_arch_refine_v6i.py
    python run_arch_refine_v6i.py ls_shift_k5 ls_shift_gctxP3P4
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
PROJECT_DIR = "runs_arch_refine_v6i"
YAML_DIR = "arch_yamls_refine_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 32                    # matches ls_shift / levelspec / ls_k5 exactly
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

# b32 references from the port sweep (all comparable — same batch).
REF = {"arch_ls_shift": 0.5598, "arch_levelspec": 0.5590, "arch_ls_k5": 0.5568}

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

# Best LB-TAL from the loss campaign was p4wide {8:4,16:7,32:1} on the 3-LEVEL
# head (cmb_p4wide, small 51.15). These two runs put it on the ls_shift 4-LEVEL
# topology: the P2 head CREATES the stride-4 small-object candidates, LB-TAL then
# ALLOCATES the budget across all four levels. CRITICAL (port-script warning):
# the budget is keyed by stride and iterates the head's strides {4,8,16,32};
# WITHOUT a `4:` entry stride-4 silently falls back to min_level_k=1 — the level
# with the MOST small candidates would get one slot, defeating the P2 head. Both
# budgets below include an explicit `4:`.
#   _LBTAL_A extends p4wide's small-emphasis to the new finest level (P2=P3=4),
#     keeps P4=7 (large's measured supply), P5=1. Sum 16 > tal_topk 10 -> the
#     per-GT cap re-ranks by metric (re-allocation, not inflation).
#   _LBTAL_B pushes MORE budget to P2 (now the richest small-object level).
_LBTAL_A = {4: 4, 8: 4, 16: 7, 32: 1}
_LBTAL_B = {4: 5, 8: 4, 16: 6, 32: 1}

# SWA (sqrt0703) box-weighting — the other half of cmb_p4wide.
_SWA0703 = dict(
    area_weight_mode="sqrt",
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0,
)


def _ls_shift_lbtal(level_topk, swa=False):
    """ls_shift topology (via YAML) + LB-TAL fixed budget (+ optional SWA)."""
    cfg = dict(_ALL_OFF, use_lbtal=True, lbtal_mode="fixed",
               lbtal_level_topk=level_topk, lbtal_min_level_k=1)
    if swa:
        cfg.update(_SWA0703)
    return cfg

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

# 1) ls_shift with snake k=9 -> k=5 (size-matched at P4/P5)
TAIL_LS_SHIFT_K5 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGDSConv, [256, 5]]                     # 25  P3 + snake k=5
  - [17, 1, ZGDSConv, [512, 5]]                     # 26  P4 + snake k=5
  - [20, 1, ZGDSConv, [1024, 5]]                    # 27  P5 + snake k=5
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 2) P3 gets BOTH gctx2 AND snake (context + shape prior), snake k=9
TAIL_LS_SHIFT_GCTXP3 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [25, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9 (stacked on gctx)
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[24, 26, 27, 28], 1, Detect, [nc]]             # 29
"""

# 3) global context pushed DEEPER: gctx2 @P2,P3,P4, snake only @P5
TAIL_LS_SHIFT_GCTXP3P4 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [14, 1, ZGGlobalContext2, [256]]                # 25  P3 + gctx2
  - [17, 1, ZGGlobalContext2, [512]]                # 26  P4 + gctx2
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""

# 4) snake at the finest level too: P2 gets gctx2 + snake k=5, rest = ls_shift
TAIL_LS_SHIFT_SNAKEP2 = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2
  - [24, 1, ZGDSConv, [256, 5]]                     # 25  P2 + snake k=5 (stacked on gctx)
  - [14, 1, ZGDSConv, [256, 9]]                     # 26  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 27  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 28  P5 + snake k=9
  - [[25, 26, 27, 28], 1, Detect, [nc]]             # 29
"""

# The ORIGINAL ls_shift tail (unmodified) — used by the LB-TAL combo runs so the
# ONLY change vs the best arch is the assigner, not the topology.
TAIL_LS_SHIFT = """  - [14, 1, DySample, [2]]                          # 21  P3 head -> stride 4
  - [[21, 2], 1, Concat, [1]]                       # 22  + backbone P2
  - [-1, 2, A2C2f, [256, False, -1]]                # 23  P2 head (stride 4)
  - [23, 1, ZGGlobalContext2, [256]]                # 24  P2 + gctx2 (small objects)
  - [14, 1, ZGDSConv, [256, 9]]                     # 25  P3 + snake k=9
  - [17, 1, ZGDSConv, [512, 9]]                     # 26  P4 + snake k=9
  - [20, 1, ZGDSConv, [1024, 9]]                    # 27  P5 + snake k=9
  - [[24, 25, 26, 27], 1, Detect, [nc]]             # 28
"""


RUNS = [
    {"name": "ls_shift_k5", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT_K5,
     "label": "ls_shift, snake k9->k5 @P3P4P5 — size-matched kernel on the winner",
     "why": "Pure kernel delta on the best topology. On v6i k=9 covers 2.6-5.3x "
            "the object at P4/P5; k=5 is size-matched. If ls_shift's snake is "
            "oversized this is the cheapest win."},

    {"name": "ls_shift_gctxP3", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT_GCTXP3,
     "label": "ls_shift + gctx2 ALSO at P3 (context AND snake) — resolve the P3 split",
     "why": "levelspec(gctx@P3)=best small, ls_shift(snake@P3)=best overall. "
            "P3 gets BOTH: does the finest object level want scene context AND "
            "the shape prior, not one or the other?"},

    {"name": "ls_shift_gctxP3P4", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT_GCTXP3P4,
     "label": "gctx2 @P2P3P4 + snake @P5 only — push global context deeper",
     "why": "60% small; every prior config caps gctx at P2/P3. Move the boundary "
            "down to P4 so small+medium get scene context and only P5 (7.7% of "
            "GTs) keeps the snake."},

    {"name": "ls_shift_snakeP2", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT_SNAKEP2,
     "label": "snake k5 @P2 (+gctx2) + snake k9 @P3P4P5 — shape prior at the finest level",
     "why": "No config has put a snake at stride-4. P2 is where the smallest "
            "objects resolve; a size-matched shape prior there tests if the "
            "finest level benefits from deformation at all."},

    # ---- BEST ARCH x BEST LB-TAL — the two axes combined ---------------------
    # ls_shift topology (unchanged) + the loss campaign's best assigner (p4wide),
    # extended to 4 levels with an explicit `4:` budget for the P2 head. This is
    # the arch-supply x loss-allocation combination the whole project points to.
    {"name": "ls_shift_lbtalA", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT,
     "label": "ls_shift + LB-TAL {4:4,8:4,16:7,32:1} — best arch + best assigner (P2-aware)",
     "why": "The P2 head creates stride-4 small candidates; LB-TAL allocates the "
            "budget across all four levels. Budget = p4wide {8:4,16:7,32:1} plus "
            "an explicit 4:4 for the new finest level (WITHOUT it stride-4 falls "
            "to min_level_k=1, starving the richest small level). Beats ls_shift "
            "(55.98/small 51.22) only if allocation adds to raw P2 supply.",
     "params": _ls_shift_lbtal(_LBTAL_A)},

    {"name": "ls_shift_lbtalB", "batch": BATCH, "levels": 4,
     "yaml": BACKBONE + HEAD_DYSAMPLE + TAIL_LS_SHIFT,
     "label": "ls_shift + LB-TAL {4:5,8:4,16:6,32:1} — more budget to the P2 level",
     "why": "Same combination, but pushes more of the top-k to stride-4 (now the "
            "richest small-object level) and trims P4. Tests whether the finest "
            "level should dominate the draw once the P2 head exists.",
     "params": _ls_shift_lbtal(_LBTAL_B)},
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
        for name in ("DySample", "ZGGlobalContext2", "ZGDSConv", "ZGDSConvV6"):
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
        print("  These YAMLs reference fork-only blocks; parse_model would fail.")
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
    # Use the run's own loss params if provided (the LB-TAL combos), else stock.
    kw.update(copy.deepcopy(rc.get("params", _ALL_OFF)))
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
            json.dump({**{k: v for k, v in out.items()},
                       "yaml_text": rc["yaml"], "why": rc["why"], "env": ENV}, f, indent=2)
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
    ref = REF["arch_ls_shift"]
    print(f"\n{'=' * 80}\n  ARCH REFINE — v6i @640, stock loss, b32 (comparable to port b32 runs)\n{'=' * 80}")
    print(f"{'run':<22}{'lvl':>4}{'mAP50':>9}{'mAP50-95':>11}{'vs ls_shift':>13}{'h':>6}")
    print("-" * 80)
    for r in sorted(res, key=lambda x: -(x["test_map5095"]
                                         if x["test_map5095"] == x["test_map5095"] else -9)):
        vs = "%+13.2f" % ((r["test_map5095"] - ref) * 100)
        print(f"{r['name']:<22}{r['levels']:>4}{r['test_map50']*100:>9.2f}"
              f"{r['test_map5095']*100:>11.2f}{vs}{r['hours']:>6.1f}")
    print(f"\n  b32 refs: ls_shift {REF['arch_ls_shift']*100:.2f} | "
          f"levelspec {REF['arch_levelspec']*100:.2f} | ls_k5 {REF['arch_ls_k5']*100:.2f}")
    print("  vs ls_shift > 0 -> a refinement beat the best topology. Seed noise 0.12 pp.")
    print("  READ per-size: CocoEvalAllFolders_luggage.py on best.pt (ls_shift small 51.22 / large 60.09).")
    print("  A real win beats small WITHOUT dropping large.")
    for r in res:
        if r.get("weights"):
            print(f"    {r['name']:<22} {r['weights']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = [r for r in RUNS if not only or r["name"] in only]
    print(f"\n{'=' * 80}\n  ARCH REFINE — {len(todo)} configs around arch_ls_shift (b32, stock loss)")
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
