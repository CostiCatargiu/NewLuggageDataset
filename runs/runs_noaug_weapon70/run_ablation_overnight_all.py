#!/usr/bin/env python3
"""
70% Ablation — 6 runs overnight:
  PART 1: 2 SWA runs (with custom loss — best TAL config)
  PART 2: 4 Architecture runs (DEFAULT params — no custom loss)

PART 1 — SWA exploration (with best training config):
  1. v5_tal07_swa09_04_70 — SWA 0.9/0.4
  2. v5_tal07_swa09_05_70 — SWA 0.9/0.5

PART 2 — Architecture exploration (default params, isolate arch effect):
  3. arch_p3_attn_70       — P3 head attention enabled
  4. arch_ld_70            — LD-Redistribute (P5→P3 capacity)
  5. arch_p3_attn_ld_70    — P3 attention + LD combined
  6. arch_deep_p3_bidi_70  — Deep P3 + Bidirectional P4

Usage:
  python run_ablation_overnight_all.py
"""

import time
import gc
import os
import copy
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation70/data.yaml"
PROJECT_DIR = "runs_noaug_weapon70"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
BATCH = 58

# =============================================================================
# BEST TRAINING CONFIG (for SWA runs only)
# =============================================================================
BEST_TRAIN = {
    "cls": 1.2,
    "alpha_start": 0.7,
    "alpha_end": 0.3,
    "alpha_min": 0.2,
    "alpha_max": 0.8,
    "small_obj_px": 40,
    "small_obj_boost": 2.5,
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    "iou_clip_start": 20.0,
    "iou_clip_end": 10.0,
    "dfl_clip_start": 10.0,
    "dfl_clip_end": 5.0,
    "tal_topk": 13,
    "tal_alpha": 0.7,
    "tal_beta": 4.0,
    "iou_type": "DIoU",
    "use_vfl": False,
}

# =============================================================================
# EPOCH SYNC CALLBACK (for SWA runs only)
# =============================================================================
def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


# =============================================================================
# PART 1: SWA RUNS (with custom loss)
# =============================================================================
def make_swa_exp(name, desc, **overrides):
    params = copy.deepcopy(BEST_TRAIN)
    params.update(overrides)
    return {"name": name, "description": desc, "params": params}


SWA_EXPERIMENTS = [
    make_swa_exp(
        "v5_tal07_swa09_04_70",
        "best TAL + SWA 0.9/0.4",
        alpha_start=0.9,
        alpha_end=0.4,
        alpha_min=0.3,
        alpha_max=0.9,
    ),
    make_swa_exp(
        "v5_tal07_swa09_05_70",
        "best TAL + SWA 0.9/0.5",
        alpha_start=0.9,
        alpha_end=0.5,
        alpha_min=0.4,
        alpha_max=0.9,
    ),
]


def run_swa_experiment(exp):
    name = exp["name"]
    params = exp["params"]

    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"# {exp['description']}")
    print(f"# Type: SWA (with custom loss)")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    model = YOLO("yolov12s.pt")
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    train_kwargs = {
        "data": DATA_YAML,
        "epochs": 80,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": PROJECT_DIR,
        "name": name,
        "patience": 100,
        "close_mosaic": 10,
        "seed": 0,
        "deterministic": True,
    }
    train_kwargs.update(params)

    try:
        model.train(**train_kwargs)
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {name} ({elapsed:.2f}h)")
        return {"name": name, "status": "OK", "time": elapsed}
    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {name} ({elapsed:.2f}h) -- {e}")
        return {"name": name, "status": f"FAILED: {e}", "time": elapsed}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# PART 2: ARCHITECTURE RUNS (DEFAULT params — no custom loss)
# =============================================================================

ARCH_P3_ATTN = """# YOLOv12s + P3 Head Attention
nc: 4
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

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, True, 4]]          # 14 — P3 with ATTENTION

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_LD = """# YOLOv12s + LD-Redistribute (P5->P3 capacity shift)
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2,  [512, False, 0.25]]      # P3: 2->3 reps
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # P5: 4->2 reps

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # P3 head: 2->3 reps

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_P3_ATTN_LD = """# YOLOv12s + P3 Attention + LD-Redistribute
nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2,  [512, False, 0.25]]      # P3: 2->3 reps
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 2, A2C2f, [1024, True, 1]]         # P5: 4->2 reps

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, True, 4]]          # P3: 3 reps + ATTENTION

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
"""

ARCH_DEEP_P3_BIDI = """# YOLOv12s + Deep P3 + Bidirectional P4
nc: 4
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
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 top-down

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]        # 14 — Deep P3 (3 reps)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up

  - [[11, 17], 1, Concat, [1]]              # 18 — bidi P4 merge
  - [-1, 1, Conv, [512, 1, 1]]              # 19 — project back

  - [17, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 22

  - [[14, 19, 22], 1, Detect, [nc]]
"""

ARCH_RUNS = [
    {
        "name": "arch_p3_attn_70",
        "desc": "P3 attention enabled (simplest change)",
        "yaml_content": ARCH_P3_ATTN,
        "batch": 58,
    },
    {
        "name": "arch_ld_70",
        "desc": "LD-Redistribute: P5->P3 capacity shift",
        "yaml_content": ARCH_LD,
        "batch": 58,
    },
    {
        "name": "arch_p3_attn_ld_70",
        "desc": "P3 attention + LD-Redistribute combined",
        "yaml_content": ARCH_P3_ATTN_LD,
        "batch": 56,
    },
    {
        "name": "arch_deep_p3_bidi_70",
        "desc": "Deep P3 + Bidirectional P4 (most complex)",
        "yaml_content": ARCH_DEEP_P3_BIDI,
        "batch": 54,
    },
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# Type: ARCHITECTURE (default params — no custom loss)")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")

        model.train(
            data=DATA_YAML,
            epochs=80,
            imgsz=IMG_SIZE,
            batch=run["batch"],
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run["name"],
            patience=100,
            close_mosaic=10,
            seed=0,
            deterministic=True,
        )

        elapsed = (time.time() - start_time) / 3600
        print(f"\n  DONE: {run['name']} ({elapsed:.2f}h)")
        return {"name": run["name"], "status": "OK", "time": elapsed}

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  FAILED: {run['name']} ({elapsed:.2f}h) -- {e}")
        return {"name": run["name"], "status": f"FAILED: {e}", "time": elapsed}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — 6 RUNS OVERNIGHT")
    print(f"  PART 1: 2 SWA runs (custom loss)")
    print(f"  PART 2: 4 Architecture runs (default params)")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}")
    print(f"{'=' * 70}")

    print(f"\n  --- SWA RUNS ---")
    for i, exp in enumerate(SWA_EXPERIMENTS):
        print(f"  [{i+1}] {exp['name']:<40} {exp['description']}")
    print(f"\n  --- ARCHITECTURE RUNS ---")
    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i+3}] {run['name']:<40} {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []

    # PART 1: SWA runs
    print(f"\n{'=' * 70}")
    print(f"  PART 1: SWA EXPLORATION (custom loss)")
    print(f"{'=' * 70}")
    for i, exp in enumerate(SWA_EXPERIMENTS):
        print(f"\n>>> SWA Run {i+1}/{len(SWA_EXPERIMENTS)}: {exp['name']}")
        result = run_swa_experiment(exp)
        results.append(result)

    # PART 2: Architecture runs
    print(f"\n{'=' * 70}")
    print(f"  PART 2: ARCHITECTURE EXPLORATION (default params)")
    print(f"{'=' * 70}")
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Arch Run {i+1}/{len(ARCH_RUNS)}: {run['name']}")
        result = run_arch_experiment(run)
        results.append(result)

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<40} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
