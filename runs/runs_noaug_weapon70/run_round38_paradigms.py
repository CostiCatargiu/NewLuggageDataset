#!/usr/bin/env python3
"""
70% Ablation -- Round 38: FOUR DIFFERENT NECK PARADIGMS (arch-only, overnight).

The local-PAN block search (rounds 21-37) is converged (p5context = best,
seed-confirmed). These 4 cells LEAVE that paradigm -- each attacks the problem
from a structurally different angle, so the result tells you WHICH DIRECTION has
headroom (a clean paper story: "we explored four neck paradigms"). All build on
p5context's validated parts (scale-matched enhancement P3-detail/P4-hybrid/P5-
context + dual-path aux); only the topology/mechanism changes.

  AXIS A -- iterative weighted fusion : r38_bifpn      (WeightedConcat, EfficientDet)
  AXIS B -- upsampling quality        : r38_dysample   (content-aware FPN, CARAFE/DySample)
  AXIS C -- global context modeling   : r38_globalctx  (GCNet/non-local per level)
  AXIS D -- gather-distribute neck     : r38_gather     (global cross-scale ctx -> P3, Gold-YOLO)

Ordered by BUILD confidence (existing tested ops first; new modules last, each in
its own try/except so a build failure can't abort the rest):
  1. r38_bifpn     -- WeightedConcat (existing, tested) replaces the neck Concats.
  2. r38_dysample  -- DySample (registered, untested in a full run) on the FPN upsamples.
  3. r38_globalctx -- ZGGlobalContext (NEW, simple: pool->MLP->gated add) per main level.
  4. r38_gather    -- ZGGatherContext (NEW, multi-input: global cross-scale ctx -> P3).

Compare to p5context 82.38 / 52.82 / small 65.16 / other 64.14 (seed-confirmed).
All default TAL / batch 48 / 80 ep / seed 0.

SMOKE-TEST first (the two NEW modules can't be tested here -- sandbox down):
  for n in r38_bifpn_70 r38_dysample_70 r38_globalctx_70 r38_gather_70; do
    python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/${n}.yaml'); \
      print('$n', type(m.model.model[-1]).__name__, m.model.stride.tolist())"; done
  -> expect DetectAuxDual [8.0, 16.0, 32.0] for all four.
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

BACKBONE = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]      # 4  backbone P3
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]          # 6  backbone P4
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]         # 8  backbone P5
"""

HEAD_NEAREST = """
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

# weighted-fusion head (BiFPN ingredient): WeightedConcat replaces Concat
HEAD_WEIGHTED = """
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, WeightedConcat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, WeightedConcat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, WeightedConcat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, WeightedConcat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
"""

# content-aware upsample head: DySample replaces nearest upsample
HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                  # 9
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, DySample, [2]]                  # 12
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
"""

# --- tails (Detect indices differ per tail) ----------------------------------
# p5context tail (Detect @ 24)
TAIL_P5CONTEXT = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# global-context per level (Detect @ 27)
TAIL_GLOBALCTX = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext, [256]]                  # 24 -- P3 + global context
  - [21, 1, ZGGlobalContext, [512]]                  # 25 -- P4 + global context
  - [23, 1, ZGGlobalContext, [1024]]                 # 26 -- P5 + global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

# gather-distribute into P3 (Detect @ 24)
TAIL_GATHER = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [[14, 17, 20], 1, ZGGatherContext, []]           # 22 -- P3 + global cross-scale ctx
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

ARCH_BIFPN     = BACKBONE + HEAD_WEIGHTED + TAIL_P5CONTEXT
ARCH_DYSAMPLE  = BACKBONE + HEAD_DYSAMPLE + TAIL_P5CONTEXT
ARCH_GLOBALCTX = BACKBONE + HEAD_NEAREST  + TAIL_GLOBALCTX
ARCH_GATHER    = BACKBONE + HEAD_NEAREST  + TAIL_GATHER

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# Ordered by build confidence (existing tested ops first, new modules last):
RUNS = [
    {"name": "r38_bifpn_70",     "yaml": ARCH_BIFPN,
     "desc": "[1/4] AXIS A: WeightedConcat fusion (BiFPN-style; existing op)"},
    {"name": "r38_dysample_70",  "yaml": ARCH_DYSAMPLE,
     "desc": "[2/4] AXIS B: DySample content-aware upsampling (registered, untested)"},
    {"name": "r38_globalctx_70", "yaml": ARCH_GLOBALCTX,
     "desc": "[3/4] AXIS C: ZGGlobalContext per level (NEW module; attacks 'other')"},
    {"name": "r38_gather_70",    "yaml": ARCH_GATHER,
     "desc": "[4/4] AXIS D: ZGGatherContext global cross-scale ctx -> P3 (NEW multi-input)"},
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
    print(f"\n{'=' * 80}\n  ROUND 38 -- FOUR NECK PARADIGMS (default TAL, arch-only)")
    print(f"  vs p5context 82.38 / 52.82 / small 65.16 / other 64.14")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<20} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
