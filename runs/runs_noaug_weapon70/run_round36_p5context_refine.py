#!/usr/bin/env python3
"""
70% Ablation -- Round 36: refine the round-35 WINNER (p5context).

p5context (best4 winner) = scale-matched enhancement realized at all 3 scales:
  P3 -> DETAIL (ZGSmallDetail)         [context hurts small -- wfv2_p3 proved it]
  P4 -> HYBRID  (ZGLSKAWideFuseV2)
  P5 -> CONTEXT (ZGLSKAWideFuse)       [completing this won the round]
  + DetectAuxDual (enhanced on main, raw on aux at every scale)

Result: best mAP50-95 of the whole campaign (52.82), best P/R, recovered "other".

Two refinements (default TAL, arch-only -> directly comparable to:
  p5context 82.38 / 52.82 / small 65.16 / other 64.14
  r34       82.36 / 52.28 / small 66.07
  r32b      82.58 / 52.41 / small 64.73 / other 65.05):

  1. r36_p5big      -- wider P5 context field: ZGLSKAWideFuse[1024,15,31] (was 11,23).
       P5 is the lowest-res / large-object level; a bigger receptive field may
       help large + other-large. Cheap variant of the winner.
  2. r36_dysample_p3 -- p5context + DySample (content-aware upsampling) replacing the
       two FPN nearest upsamples. It feeds the P3 detail level, so it COMPOUNDS with
       the P3 detail block (does not compete). First DySample test in the right setting.
       NOTE: DySample untested here -> own try/except (a failure won't abort the other).

Honest expectation: the neck design is converged, so expect small/noise-level moves.
The bigger lever is p5context + best-TAL (separate script).

SMOKE-TEST first:
  for n in r36_p5big_70 r36_dysample_p3_70; do
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
# CONFIGURATION (match the r35 runs; note: best4 ran at batch 50 -> set 50 for
# exact parity with p5context; 48 is fine, the ~2 diff is within noise)
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
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 head
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

# p5context tail with a WIDER P5 context field (15/31 instead of 11/23)
TAIL_P5BIG = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 15, 31]]          # 23 -- P5 context (WIDER field)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

# exact p5context tail (used with the DySample head)
TAIL_P5CONTEXT = f"""  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid (main)
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail (main)
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context (main)
  - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 24
"""

ARCH_P5BIG       = BACKBONE + HEAD_NEAREST  + TAIL_P5BIG
ARCH_DYSAMPLE_P3 = BACKBONE + HEAD_DYSAMPLE + TAIL_P5CONTEXT

DEFAULT_TAL = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    {"name": "r36_p5big_70",       "yaml": ARCH_P5BIG,
     "desc": "[1/2] p5context, WIDER P5 context ZGLSKAWideFuse[1024,15,31]"},
    {"name": "r36_dysample_p3_70", "yaml": ARCH_DYSAMPLE_P3,
     "desc": "[2/2] p5context + DySample upsampling (feeds P3 detail; UNTESTED module)"},
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
    print(f"\n{'=' * 80}\n  ROUND 36 -- refine p5context (default TAL, arch-only)")
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
