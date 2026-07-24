#!/usr/bin/env python3
"""
70% Ablation — Round 6: THE LSKA FAMILY (focused search of the one winner).

THE SIGNAL (from ~20 architecture runs):
  #1 zg_p4    = LSKA @ P4, zero-gated   79.05  (+0.60)
  #2 arch_ska = LSKA @ P3, replacement  79.03  (+0.58)
  #3 zg_gc_p5 = global ctx @ P5, gated  78.89  (+0.44)
  Everything else (SE, MHSA, depth, dual-path, P2-fuse, CGC head): <= ~0.
  Large-kernel spatial context is the ONLY mechanism that won twice, in two
  integration styles, at two positions. This round searches INTO it at the
  winning position (P4) instead of sampling around it.

PURE ARCH ROUND — default TAL everywhere. Bars: 78.45 baseline, 79.05 zg_p4.
No fork changes needed (ZGLSKA already takes the kernel size argument).

RUNS:
  1. r6_zgp4_k11_70 — bigger kernel (7 -> 11) at P4. Weapons are elongated;
                      effective RF grows ~23 -> ~35 cells. Cheapest probe of
                      "more context = more gain?"
  2. r6_zgp4_ms_70  — multi-scale: stacked gated LSKA k5 then k9 at P4
                      (append-only chain, no index shift, full transfer).
  3. r6_zgp4_p5_70  — LSKA at P4 AND P5 (round-2 combo used GC@P5; pure-LSKA
                      pair may stack where mixed mechanisms didn't).
  4. r6_ska_p4_70   — C2fLSKA REPLACEMENT at P4 (defined in round 1, never
                      run): replace-vs-gate ablation at the same position —
                      directly isolates the contribution of ZG integration.

Usage:
  python run_round6_lska_family.py
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 — P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 — P5 head
"""

# 1: winning config, bigger kernel
ARCH_ZGP4_K11 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 11]]              # 21 — gated LSKA @ P4, k=11
  - [[14, 21, 20], 1, Detect, [nc]]
"""

# 2: multi-scale — stacked gated LSKA (k5 then k9), append-only chain
ARCH_ZGP4_MS = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 5]]               # 21 — gated LSKA @ P4, k=5
  - [-1, 1, ZGLSKA, [512, 9]]               # 22 — gated LSKA, k=9 (stacked)
  - [[14, 22, 20], 1, Detect, [nc]]
"""

# 3: LSKA at both deep scales (pure-LSKA pair, unlike round-2's LSKA+GC mix)
ARCH_ZGP4_P5 = BASE_0_20 + """
  - [17, 1, ZGLSKA, [512, 7]]               # 21 — gated LSKA @ P4
  - [20, 1, ZGLSKA, [1024, 7]]              # 22 — gated LSKA @ P5
  - [[14, 21, 22], 1, Detect, [nc]]
"""

# 4: replacement at P4 (never run) — replace-vs-gate ablation at same position
ARCH_SKA_P4_REPLACE = """nc: 4
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
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 — P3 head (untouched)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C2fLSKA, [512, False]]          # 17 — LSKA REPLACES P4 bottom-up
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20
  - [[14, 17, 20], 1, Detect, [nc]]
"""

TAL_DEFAULT = dict(tal_topk=10, tal_alpha=0.5, tal_beta=6.0)

RUNS = [
    # ACTIVE (the two worth running first, ~4.5h total):
    {"name": "r6_zgp4_k11_70",
     "desc": "[1/2] gated LSKA @ P4, kernel 11 — more context",
     "yaml_content": ARCH_ZGP4_K11, "batch": 58},
    {"name": "r6_ska_p4_70",
     "desc": "[2/2] C2fLSKA replacement @ P4 — replace-vs-gate ablation",
     "yaml_content": ARCH_SKA_P4_REPLACE, "batch": 58},
    # ON HOLD — re-enable ONLY if r6_zgp4_k11_70 beats 79.05 (more context
    # paying off would justify mining the family further):
    # {"name": "r6_zgp4_ms_70",
    #  "desc": "stacked gated LSKA k5+k9 @ P4 — multi-scale",
    #  "yaml_content": ARCH_ZGP4_MS, "batch": 52},
    # {"name": "r6_zgp4_p5_70",
    #  "desc": "gated LSKA @ P4 + P5 — pure-LSKA pair",
    #  "yaml_content": ARCH_ZGP4_P5, "batch": 52},
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    """model.load() first (sets model.ckpt so the trainer keeps the weights),
    then remap Detect keys model.21.* -> model.N.* if the index shifted."""
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v
                for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: "
          f"{len(matched)}/{len(remapped)} Detect keys transferred on top")
    return model


def run_experiment(run):
    yaml_path = os.path.join(YAML_DIR, f"{run['name']}.yaml")
    save_yaml(run["yaml_content"], yaml_path)

    print(f"\n{'#' * 70}")
    print(f"# {run['name']}")
    print(f"# {run['desc']}")
    print(f"# TAL: DEFAULT   Batch: {run['batch']}")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    try:
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)

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
            # --- DEFAULT TAL (pure architecture effect) ---
            **TAL_DEFAULT,
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
    print(f"  70% ABLATION — ROUND 6: LSKA FAMILY @ P4 (PURE ARCH, default TAL)")
    print(f"  Bars: 78.45 baseline | 79.05 zg_p4 (the config being refined)")
    print(f"{'=' * 70}")

    for i, run in enumerate(RUNS):
        print(f"  [{i+1}] {run['name']:<16} batch={run['batch']}  {run['desc']}")

    print(f"\n{'=' * 70}\n")

    results = []
    for i, run in enumerate(RUNS):
        print(f"\n>>> Run {i+1}/{len(RUNS)}: {run['name']}")
        results.append(run_experiment(run))

    total_time = (time.time() - total_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE ({total_time:.2f}h)")
    for r in results:
        tag = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  [{tag}] {r['name']:<16} {r['time']:.2f}h")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
