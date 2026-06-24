#!/usr/bin/env python3
"""
70% Ablation -- Round 37: NEW STRUCTURAL AXES (4 overnight archs, arch-only).

The neck is converged: p5context (P3 detail / P4 hybrid / P5 context + dual-path
aux) is seed-confirmed best on mAP50-95 (52.82/52.93), and round-36 refinements
(wider P5, r32b+P5) did NOT beat it. So instead of more p5 tweaks, these 4 open
DIFFERENT axes we have not explored:

  AXIS 1 - WHAT block (adaptive vs static fusion)
  AXIS 2 - WHERE to supervise (neck-raw vs backbone)
  AXIS 3 - HOW MUCH capacity at the dominant scale

All build on p5context, all default TAL / batch 48 / 80 ep, all use ALREADY-
REGISTERED modules (no code changes). Ordered by confidence.

  1. r37_selectfuse_p4 -- replace P4's static WideFuseV2 with ZGLSKASelectFuse
     (per-PIXEL adaptive receptive-field routing). This is the one block designed
     to escape the "static fuse = single global compromise" limitation that capped
     every fixed fusion. Best principled shot at a real gain. (highest conf)
  2. r37_deepaux       -- aux anchors the BACKBONE features [4,6,8] instead of the
     neck-raw [14,17,20]. Pushes dual-path supervision INTO the backbone, forcing
     the backbone itself (not just the neck) to preserve good multi-scale features.
  3. r37_selectfuse_p3 -- replace P3's ZGSmallDetail with ZGLSKASelectFuse (adaptive
     routing incl. a small-detail branch). Tests whether per-pixel routing gives the
     small-object detail WITHOUT the context-smoothing that killed wfv2_p3.
  4. r37_p4stack       -- TWO stacked WideFuseV2 @ P4 (dominant scale). Capacity test
     ("more of the best block"); lowest conf (capacity rarely helps when converged).

Compare to: r32b 82.58/52.41/64.73small | r34 .../66.07small | p5ctx 82.38/52.82.

SMOKE-TEST first:
  for n in r37_selectfuse_p4_70 r37_deepaux_70 r37_selectfuse_p3_70 r37_p4stack_70; do
    python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/${n}.yaml'); \
      print('$n', type(m.model.model[-1]).__name__, m.model.stride.tolist())"; done
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
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 48
EPOCHS = 80
AUX_W = 0.5

# Backbone+head 0-20. Backbone P3=4, P4=6, P5=8 ; neck P3=14, P4=17, P5=20.
BASE_0_20 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4  backbone P3 (stride 8)
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6  backbone P4 (stride 16)
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8  backbone P5 (stride 32)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head (raw)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up (raw)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head (raw)
"""

# 1. SelectFuse @ P4 (adaptive routing) instead of WideFuseV2
ARCH_SELECTFUSE_P4 = BASE_0_20 + f"""  - [17, 1, ZGLSKASelectFuse, [512, 11, 23, 3]]      # 21 -- P4 ADAPTIVE (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# 2. Deep aux: aux anchors BACKBONE features [4,6,8] instead of neck-raw [14,17,20]
ARCH_DEEPAUX = BASE_0_20 + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 4, 6, 8], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24  aux = backbone P3/P4/P5
"""

# 3. SelectFuse @ P3 (adaptive) instead of ZGSmallDetail
ARCH_SELECTFUSE_P3 = BASE_0_20 + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGLSKASelectFuse, [256, 7, 15, 3]]       # 22 -- P3 ADAPTIVE (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# 4. Stacked P4 hybrid: two WideFuseV2 @ P4
ARCH_P4STACK = BASE_0_20 + f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid #1
  - [21, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 22 -- P4 hybrid #2 (stacked, main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 23 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 24 -- P5 context (main)
  - [[23, 22, 24, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 25
"""

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r37_selectfuse_p4_70", "yaml": ARCH_SELECTFUSE_P4,
     "desc": "[1/4] P4 = SelectFuse (per-pixel adaptive RF) -- escape static-fuse limit"},
    {"name": "r37_deepaux_70",       "yaml": ARCH_DEEPAUX,
     "desc": "[2/4] aux anchors BACKBONE [4,6,8] -- dual-path supervision into backbone"},
    {"name": "r37_selectfuse_p3_70", "yaml": ARCH_SELECTFUSE_P3,
     "desc": "[3/4] P3 = SelectFuse (adaptive) -- detail without context-smoothing"},
    {"name": "r37_p4stack_70",       "yaml": ARCH_P4STACK,
     "desc": "[4/4] 2x WideFuseV2 @ P4 -- capacity test (lowest conf)"},
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
    print(f"\n{'#' * 80}\n# {run['name']}\n# {run['desc']}\n# Batch {BATCH}  Epochs {EPOCHS}  default TAL  seed 0\n{'#' * 80}\n")

    start = time.time()
    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run["name"], patience=100,
                  close_mosaic=10, seed=0, deterministic=True)
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
    print(f"\n{'=' * 80}\n  ROUND 37 -- NEW STRUCTURAL AXES (default TAL, arch-only)")
    print(f"  vs p5context 82.38 / 52.82 (seed-confirmed best 50-95)")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<24} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
