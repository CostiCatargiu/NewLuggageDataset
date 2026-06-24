#!/usr/bin/env python3
"""
70% Ablation -- Round 36 BEST: merged, de-duplicated, ordered by confidence.

Built on the round-35 winner p5context (scale-matched dual-path: P3 detail / P4
hybrid / P5 context; best mAP50-95 of the campaign = 52.82).

NOTE: the "Full Symmetric" arch from the combo proposal is byte-identical to
p5context, so re-running it at seed 0 just reproduces it. Here it's repurposed as
a SEED CHECK (seed=1) -- which you need anyway before calling p5context the winner.

Four cells, ordered by confidence (highest-value first):

  1. r36_r32b_p5ctx   -- R32B (mAP50 king 82.58) + P5 context. Best-of-both shot:
       isolates P3-raw (r32b) vs P3-detail (p5context) UNDER P5 enhancement.
       Most likely new champion. (default TAL, seed 0)
  2. r36_p5ctx_seed1  -- p5context re-run at SEED 1. Confirms 52.82 is real, not
       single-seed luck. Guaranteed-useful control. (default TAL, seed 1)
  3. r36_p5big        -- p5context, WIDER P5 context ZGLSKAWideFuse[1024,15,31].
       Cheap P5 receptive-field sweep. (default TAL, seed 0)
  4. r36_dysample_p3  -- p5context + DySample upsampling (feeds P3 detail). Novel
       mechanism but UNTESTED module -> last, own try/except. (default TAL, seed 0)

All default TAL / batch 48 / 80 ep -> comparable to:
  r32b 82.58 / 52.41 / small 64.73 / other 65.05
  r34  82.36 / 52.28 / small 66.07
  p5ctx 82.38 / 52.82 / small 65.16 / other 64.14

SMOKE-TEST first:
  for n in r36_r32b_p5ctx_70 r36_p5ctx_seed1_70 r36_p5big_70 r36_dysample_p3_70; do
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

HEAD_DYSAMPLE = """
head:
  - [-1, 1, DySample, [2]]                  # 9  content-aware upsample
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, DySample, [2]]                  # 12 content-aware upsample (-> P3)
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
"""

# --- tails -------------------------------------------------------------------
# 1. r32b + P5 context: P3 RAW on main (= r32b), + P5 context. Detect at 23.
TAIL_R32B_P5 = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 22 -- P5 context (main)
  - [[14, 21, 22, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 23
"""

# 2/3. p5context (P3 detail + P4 hybrid + P5 context). Detect at 24.
TAIL_P5CONTEXT = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# p5context with WIDER P5 field
TAIL_P5BIG = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 15, 31]]          # 23 -- P5 context (WIDER field)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

ARCH_R32B_P5    = BACKBONE + HEAD_NEAREST  + TAIL_R32B_P5
ARCH_P5CONTEXT  = BACKBONE + HEAD_NEAREST  + TAIL_P5CONTEXT
ARCH_P5BIG      = BACKBONE + HEAD_NEAREST  + TAIL_P5BIG
ARCH_DYSAMPLE   = BACKBONE + HEAD_DYSAMPLE + TAIL_P5CONTEXT

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

# Ordered by confidence (highest-value first):
RUNS = [
    {"name": "r36_r32b_p5ctx_70",  "yaml": ARCH_R32B_P5,   "seed": 0,
     "desc": "[1/4] R32B + P5 context (best-of-both; P3-raw vs P3-detail under P5)"},
    {"name": "r36_p5ctx_seed1_70", "yaml": ARCH_P5CONTEXT, "seed": 1,
     "desc": "[2/4] p5context SEED CHECK (seed 1) -- confirm 52.82 is real"},
    {"name": "r36_p5big_70",       "yaml": ARCH_P5BIG,     "seed": 0,
     "desc": "[3/4] p5context, WIDER P5 context [1024,15,31] -- field sweep"},
    {"name": "r36_dysample_p3_70", "yaml": ARCH_DYSAMPLE,  "seed": 0,
     "desc": "[4/4] p5context + DySample upsampling (UNTESTED module) -- last"},
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
        print(f"  head = {type(model.model.model[-1]).__name__}, "
              f"levels = {model.model.model[-1].nl}, strides = {model.model.stride.tolist()}")
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
    print(f"\n{'=' * 80}\n  ROUND 36 BEST -- ordered by confidence (default TAL, arch-only)")
    print(f"  vs r32b 82.58/52.41 | r34 .../66.07 small | p5ctx 82.38/52.82")
    print(f"{'=' * 80}")
    for r in RUNS:
        print(f"  {r['desc']}")
    print(f"{'=' * 80}\n")
    results = [run_experiment(r) for r in RUNS]
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<22} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
