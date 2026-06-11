#!/usr/bin/env python3
"""
70% Ablation — Round 2: Zero-Init Gated (ZG) architectures.

DESIGN PRINCIPLES (from the failure analysis of all 14 previous arch runs):
  P1. Never replace pretrained blocks. All new modules are APPENDED as
      zero-gated residual branches (y = x + gamma*f(x), gamma=0 at init).
      Layers 0-20 keep their indices -> yolov12s.pt transfers fully.
      At epoch 0 every model below IS the baseline. Downside risk ~ 0.
  P2. Capacity goes where the data is: 68% large + 30% medium boxes
      -> P4/P5. P3 (2.2% small, 374 train instances) is left untouched
      in all primary runs.
  P3. PURE ARCHITECTURE ROUND: all runs use DEFAULT training params so any
      delta vs 78.45% is attributable to the architecture alone. TAL
      combination is deferred to a follow-up round with the winners only.

REQUIRES: ZGLSKA, ZGGC, ZGSE, ZGMHSA registered in your ultralytics fork
          (see gated_blocks.py — same procedure as C2fLSKA).

SANITY CHECKS on first run:
  - "Transferred x/y items from yolov12s.pt" must be near-complete
    (everything except Detect; Detect moved index so its box branch
    no longer transfers — known and acceptable, it retrains in epochs).
  - python gated_blocks.py must print identity-at-init: OK for all blocks.

Bar for EVERY run: baseline 78.45% (old best arch: 79.03%).
A result is convincing only above ~79.0% (seed noise ~ +/-0.5%).
AFTER training, also check the learned gate magnitudes per run:
  python -c "import torch; m=torch.load('runs_noaug_weapon70/<run>/weights/best.pt')['model']; \\
             [print(n, float(p.abs().mean())) for n,p in m.named_parameters() if 'gamma' in n]"
  gates ~0 -> network didn't want the capacity (real null result);
  gates open + no gain -> capacity used but redundant.

Usage:
  python run_arch_gated_ablation.py
"""

import time
import gc
import os
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

# =============================================================================
# Shared base = UNMODIFIED YOLOv12s layers 0-20 (so pretrained loads fully).
# Each variant only appends gated layers (21+) and rewires Detect.
# =============================================================================
BASE_0_20 = """nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11 — P4 top-down (pretrained, unchanged)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head (pretrained, unchanged)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up (pretrained, unchanged)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head (pretrained, unchanged)
"""

# ZG-A: gated large-kernel context at P4 + gated global context at P5.
# P3 untouched (small objects see the exact baseline path).
ARCH_ZG_P45 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 7]]               # 21 — gated LSKA on P4 (zero-init)
  - [20, 1, ZGGC,   [1024, 8]]              # 22 — gated global context on P5 (zero-init)
  - [[14, 21, 22], 1, Detect, [nc]]         # 23 — P3 raw, P4/P5 gated
"""

# ZG-B: gated everywhere — small kernel at P3 (gate decides if it helps).
ARCH_ZG_ALL = BASE_0_20 + """
  - [14, 1, ZGLSKA, [256, 5]]               # 21 — gated LSKA on P3, gentle 5x5
  - [17, 1, ZGLSKA, [512, 7]]               # 22 — gated LSKA on P4
  - [20, 1, ZGGC,   [1024, 8]]              # 23 — gated global context on P5
  - [[21, 22, 23], 1, Detect, [nc]]
"""

# ZG-C: gated self-attention at P5 only (400 tokens — cheap).
ARCH_ZG_MHSA_P5 = BASE_0_20 + """
  - [20, 1, ZGMHSA, [1024, 4]]              # 21 — gated MHSA on P5 (zero-init)
  - [[14, 17, 21], 1, Detect, [nc]]
"""

# ZG-D: cheapest control — gated SE at P4+P5. If heavier blocks don't beat
# this, the complexity isn't paying; if even this helps, gating works.
ARCH_ZG_SE_P45 = BASE_0_20 + """
  - [17, 1, ZGSE, [512, 8]]                 # 21 — gated SE on P4
  - [20, 1, ZGSE, [1024, 8]]                # 22 — gated SE on P5
  - [[14, 21, 22], 1, Detect, [nc]]
"""

# (The old SKA@P3 replacement arch was removed from this round — it will only
#  return in the follow-up TAL-combination round if pure-arch winners emerge.)

# Single-component variants (to attribute any gain of the combined P4/P5 run)
ARCH_ZG_P4_ONLY = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 7]]               # 21 — gated LSKA on P4 only
  - [[14, 21, 20], 1, Detect, [nc]]         # P3 raw, P4 gated, P5 raw
"""

ARCH_ZG_GC_P5_ONLY = BASE_0_20 + """
  - [20, 1, ZGGC, [1024, 8]]                # 21 — gated global context on P5 only
  - [[14, 17, 21], 1, Detect, [nc]]         # P3 raw, P4 raw, P5 gated
"""

# TAL parameter sets — THIS ROUND IS PURE ARCHITECTURE: all runs use DEFAULT.
# Question to answer first: can ANY arch change beat 78.45% on its own?
# TAL_BEST is kept only for the follow-up combination round (winners only).
TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)
TAL_BEST = dict(tal_topk=15, tal_alpha=0.5, tal_beta=3.0)  # = v5_topk15_beta3_70 (80.45%) — NOT used this round

# Ordered by CONFIDENCE TO WORK (run top-down; stop anytime — best bets ran first).
# Bar for every run: baseline 78.45% (and old best arch 79.03%).
ARCH_RUNS = [
    # 1: HIGHEST — gated capacity at BOTH scales where 98% of boxes live;
    #    zero-init gates mean worst case ~= baseline, pretrained loads fully.
    {"name": "arch_zg_p45_70",
     "desc": "[1/6] ZG LSKA@P4 + GC@P5 — zero-init gates, P3 untouched",
     "yaml_content": ARCH_ZG_P45, "tal": TAL_DEFAULT, "batch": 54},
    # 2: HIGH — LSKA was the only block that ever helped (+0.58% at P3);
    #    here it sits at P4 (data-rich) as a gated add-on, not a replacement.
    {"name": "arch_zg_p4_70",
     "desc": "[2/6] ZG LSKA@P4 only — proven block, right scale, gated",
     "yaml_content": ARCH_ZG_P4_ONLY, "tal": TAL_DEFAULT, "batch": 56},
    # 3: MEDIUM-HIGH — global context for the 68% large boxes; tiny param cost.
    {"name": "arch_zg_gc_p5_70",
     "desc": "[3/6] ZG GC@P5 only — global context for large objects",
     "yaml_content": ARCH_ZG_GC_P5_ONLY, "tal": TAL_DEFAULT, "batch": 56},
    # 4: MEDIUM — cheapest control; if even this helps, gating itself works,
    #    if heavier blocks don't beat it, their complexity isn't paying.
    {"name": "arch_zg_se_p45_70",
     "desc": "[4/6] ZG SE@P4+P5 — cheapest gated control",
     "yaml_content": ARCH_ZG_SE_P45, "tal": TAL_DEFAULT, "batch": 56},
    # 5: MEDIUM-LOW — touches P3 (bad history, even if gated/safe-at-init);
    #    most added params of the ZG set.
    {"name": "arch_zg_all_70",
     "desc": "[5/6] ZG P3(k5)+P4(k7)+P5(GC) — gated everywhere",
     "yaml_content": ARCH_ZG_ALL, "tal": TAL_DEFAULT, "batch": 54},
    # 6: LOWEST — most speculative; P5 backbone already has area-attention
    #    (layer 8, A2C2f), so global MHSA at P5 head may be redundant.
    {"name": "arch_zg_mhsa_p5_70",
     "desc": "[6/6] ZG MHSA@P5 — gated global self-attention",
     "yaml_content": ARCH_ZG_MHSA_P5, "tal": TAL_DEFAULT, "batch": 54},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def run_arch_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: {run['tal']}")
    print(f"# Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        model.load("yolov12s.pt")  # check 'Transferred x/y' — ZG yamls should be near-complete

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
            # --- TAL params (per run) ---
            **run["tal"],
            # --- DISABLE SWA ---
            alpha_start=0.0,
            alpha_end=0.0,
            alpha_min=0.0,
            alpha_max=0.0,
            # --- DISABLE clipping ---
            iou_clip_start=999.0,
            iou_clip_end=999.0,
            dfl_clip_start=999.0,
            dfl_clip_end=999.0,
            # --- DISABLE custom features ---
            small_obj_boost=1.0,
            small_obj_px=0,
            center_loss_weight_init=0.0,
            center_loss_weight_min=0.0,
            use_vfl=False,
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


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"  70% ABLATION — ROUND 2: ZERO-INIT GATED ARCHITECTURES (PURE ARCH)")
    print(f"  All runs DEFAULT params — bar: baseline 78.45% (old best arch 79.03%)")
    print(f"{'=' * 70}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs:  80    ImgSize: {IMG_SIZE}")
    print(f"{'=' * 70}")

    for i, run in enumerate(ARCH_RUNS):
        print(f"  [{i+1}] {run['name']:<24} batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(ARCH_RUNS):
        print(f"\n>>> Run {i+1}/{len(ARCH_RUNS)}: {run['name']}")
        results.append(run_arch_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    print(f"{'=' * 70}")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<24} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
