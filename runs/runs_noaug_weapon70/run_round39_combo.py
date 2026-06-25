#!/usr/bin/env python3
"""
70% Ablation -- Round 39: stack the two round-38 winners (globalctx + dysample).

Round-38 (4 neck paradigms, evaluated on BOTH clean splits valid+test) showed:
  - globalctx (gentle per-level global context) = WINNER: best small AND best 50-95.
  - dysample  (content-aware upsampling)        = steady 2nd: consistently +, never hurts.
  - gather (aggressive global->P3) hurt 50-95 ; bifpn (learnable fusion) worst.

globalctx and dysample help via ORTHOGONAL mechanisms -- semantic/channel global
context vs spatial detail-preserving upsampling -- so they should STACK rather than
compete. Both are gated / identity-init, so the downside is capped at "no worse
than globalctx".

  Cell 1 (r39_combo)        : p5context + per-level ZGGlobalContext + DySample upsampling.
       Hypothesis: dysample's spatial sharpening adds on top of globalctx's context
       -> edges globalctx on small / 50-95. If it doesn't add, lands at globalctx.
  Cell 2 (r39_globalctx_s1) : globalctx re-run at SEED 1 -- lock globalctx as the base
       (its win was single-seed). Needed regardless.

All default TAL / batch 48 / 80 ep. Compare (mean of clean valid+test) to:
  globalctx 82.29 / 52.57 / small 66.91   (round-38 winner)
  dysample  82.17 / 52.51 / small 66.69
  p5context ~82.5 / ~52.9 (test) -- best 50-95 before globalctx

All modules already registered (ZGGlobalContext, DySample, ZGLSKAWideFuseV2,
ZGLSKAWideFuse, ZGSmallDetail, DetectAuxDual) -- no package changes.

SMOKE-TEST first:
  for n in r39_combo_70 r39_globalctx_s1_70; do
    python -c "from ultralytics import YOLO; m=YOLO('arch_yamls/${n}.yaml'); \
      print('$n', type(m.model.model[-1]).__name__, m.model.stride.tolist())"; done
  -> expect DetectAuxDual [8.0, 16.0, 32.0] (combo Detect @ 27, globalctx @ 27).
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
  - [-1, 2, A2C2f, [256, False, -1]]        # 14
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
"""

# global-context per main level + dual-path aux (Detect @ 27)
TAIL_GLOBALCTX = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext, [256]]                  # 24 -- P3 + global context
  - [21, 1, ZGGlobalContext, [512]]                  # 25 -- P4 + global context
  - [23, 1, ZGGlobalContext, [1024]]                 # 26 -- P5 + global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

# Cell 1: globalctx + DySample (combo) -- DySample head + global-context tail
ARCH_COMBO     = BACKBONE + HEAD_DYSAMPLE + TAIL_GLOBALCTX
# Cell 2: globalctx alone (seed-check) -- nearest head + global-context tail
ARCH_GLOBALCTX = BACKBONE + HEAD_NEAREST  + TAIL_GLOBALCTX

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r39_combo_70",        "yaml": ARCH_COMBO,     "seed": 0,
     "desc": "globalctx + DySample (stack the two round-38 winners) -- pure arch, seed 0"},
]
# NOTE: seed-check is intentionally deferred -- run it only on the BEST arch once
# the pure-arch comparison picks a winner. ARCH_GLOBALCTX is kept above for that.


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
    print(f"\n{'=' * 80}\n  ROUND 39 -- globalctx + DySample combo + globalctx seed-check")
    print(f"  vs globalctx (clean mean) 82.29 / 52.57 / small 66.91")
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
