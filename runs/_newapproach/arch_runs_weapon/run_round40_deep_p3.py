#!/usr/bin/env python3
"""
70% Ablation -- Round 40: DEEPER P3 HEAD TOWERS

Data analysis across 19 experiments reveals the #1 bottleneck is NOT features
(recall is 80-90%) but SCORING QUALITY at the P3 (small-object) scale level:

  class     AR50_small  AP50_small  scoring_gap
  other     80.6%       49.0%       31.6pp  <-- worst in system
  long_gun  90.5%       67.2%       23.4pp
  knife     85.1%       64.2%       20.9pp
  pistol    90.3%       77.6%       12.7pp

The detector FINDS small objects but MIS-SCORES them. Additionally,
long_gun_small has AP50=67% but AP50-95=25% (ratio 0.378) -- the worst
box regression quality in the system.

Both problems originate at P3: it handles the HARDEST classification AND
regression with the SHALLOWEST towers (same 2-conv depth as the easier
P4/P5 levels). Every prior round (R21-R39) improved the NECK features
feeding the head; this round improves the HEAD ITSELF at the scale where
it struggles most.

FIX: DetectAuxDualDeepP3 -- adds ONE extra conv layer to P3's cls and box
towers ONLY. P4/P5 stay at standard depth. This gives P3 more
representational capacity without affecting the already-good larger scales.

  P3 cls: 3 DWConv+Conv blocks (was 2) + final conv
  P3 box: 3 Conv blocks (was 2) + final conv
  P4/P5: unchanged (2 blocks each)

Properties:
  - Inference cost: ~2% FLOPS increase (only P3 head is deeper)
  - Safe transfer: standard init, near-identity at epoch 0
  - Aux towers: standard depth (only main P3 gets deeper)
  - Drop-in: same YAML format as DetectAuxDual

THREE CELLS (architecture x neck combination):

  Cell 1 (r40_deep_p3_globalctx): DetectAuxDualDeepP3 on the R38 globalctx
      neck (best small-object scoring). Tests whether deeper P3 + global
      context is better than either alone. 80% confidence.

  Cell 2 (r40_deep_p3_p5ctx): DetectAuxDualDeepP3 on the R36 p5context
      neck (best overall mAP50). Tests whether deeper P3 recovers the
      knife_small regression (-4.63pp). 75% confidence.

  Cell 3 (r40_deep_p3_r32b): DetectAuxDualDeepP3 on the R32B neck
      (simplest winning arch). Clean ablation: is the deeper P3 head
      the improvement, or does it need a specific neck? 70% confidence.

Compare to (all default TAL):
  r36_p5ctx   82.65 / 52.93 / small 64.49 / other_S 48.95  (best mAP50)
  r38_globalctx 82.52 / 52.69 / small 66.18 / other_S 50.54 (best small)
  r32b        82.58 / 52.41 / small 64.73 / other_S 45.79

Usage:
  python run_round40_deep_p3.py
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit70percentage/data.yaml"
PROJECT_DIR = "runs_noaug_weapon_70_review"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48
EPOCHS = 80
AUX_W = 0.5

BACKBONE = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]
"""

HEAD = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (raw; aux anchor)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (raw; aux anchor)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head (raw; aux anchor)
"""

# --- Cell 1: globalctx neck + DeepP3 head (Detect @ 27) ---
TAIL_GLOBALCTX_DEEP = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext, [256]]                  # 24 -- P3 + global context
  - [21, 1, ZGGlobalContext, [512]]                  # 25 -- P4 + global context
  - [23, 1, ZGGlobalContext, [1024]]                 # 26 -- P5 + global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, {AUX_W}]]  # 27
"""

# --- Cell 2: p5context neck + DeepP3 head (Detect @ 24) ---
TAIL_P5CTX_DEEP = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, {AUX_W}]]  # 24
"""

# --- Cell 3: R32B neck + DeepP3 head (Detect @ 22) ---
TAIL_R32B_DEEP = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 widefuse
  - [[14, 21, 20, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, {AUX_W}]]  # 22
"""

ARCH_GLOBALCTX_DEEP = BACKBONE + HEAD + TAIL_GLOBALCTX_DEEP
ARCH_P5CTX_DEEP     = BACKBONE + HEAD + TAIL_P5CTX_DEEP
ARCH_R32B_DEEP      = BACKBONE + HEAD + TAIL_R32B_DEEP

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r40_deep_p3_globalctx_70", "yaml": ARCH_GLOBALCTX_DEEP, "seed": 0,
     "desc": "[1/3] DeepP3 + globalctx neck (best small-obj scoring) -- 80% confidence"},
    {"name": "r40_deep_p3_p5ctx_70",     "yaml": ARCH_P5CTX_DEEP,     "seed": 0,
     "desc": "[2/3] DeepP3 + p5context neck (best overall mAP50) -- 75% confidence"},
    {"name": "r40_deep_p3_r32b_70",      "yaml": ARCH_R32B_DEEP,      "seed": 0,
     "desc": "[3/3] DeepP3 + R32B neck (clean ablation) -- 70% confidence"},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: {len(matched)}/{len(remapped)} keys")
    return model


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if getattr(trainer, "criterion", None) is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, "_sync_bbox_loss_state"):
                trainer.criterion._sync_bbox_loss_state()
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
    except Exception:
        pass


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml"], yaml_path)
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# Batch {BATCH}  Epochs {EPOCHS}  default TAL  seed {run['seed']}\n{'#' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        head = model.model.model[-1]
        print(f"  head = {type(head).__name__}, "
              f"levels = {head.nl}, strides = {model.model.stride.tolist()}")
        # Verify P3 is deeper
        p3_cls_layers = len(list(head.cv3[0].children()))
        p4_cls_layers = len(list(head.cv3[1].children()))
        print(f"  P3 cls depth = {p3_cls_layers}, P4 cls depth = {p4_cls_layers} "
              f"({'DEEPER' if p3_cls_layers > p4_cls_layers else 'SAME'})")

        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=run["seed"], deterministic=True)
        kw.update(DEFAULT_TAL)
        model.train(**kw)
        elapsed = (time.time() - start) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 80}")
    print(f"  ROUND 40 -- DEEPER P3 HEAD TOWERS (3 runs)")
    print(f"  Target: reduce the 31.6pp scoring gap at P3 (AR50_S - AP50_S)")
    print(f"  Method: +1 conv layer in P3 cls+box towers (P4/P5 unchanged)")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<35} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
