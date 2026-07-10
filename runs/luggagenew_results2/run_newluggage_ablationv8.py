#!/usr/bin/env python3
"""
Round 8 Ablation: v3 Loss Refinements (Evidence-Driven)
=========================================================
Tests 4 new MULTIPLICATIVE REWEIGHTING modifications, each in isolation
on the SWA-const06 + WIoU base (the two proven winners from Rounds 1-7).

Phase 1: Isolated tests (4 runs) -- each changes ONE section
Phase 2: Combine winners pairwise (run AFTER Phase 1 results)

Base config (proven best combination from 31 prior models):
  SWA constant alpha=0.6, boost=1.75, px=48  (Round 2 winner)
  WIoU v3 alpha=1.9, delta=3.0               (Round 4 precision winner)
  TAL stock 10/0.5/6.0
  Class weighting ON (sqrt mode)
  Everything else OFF
"""

import subprocess
import sys
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================
DATASET_YAML = 'C:/DISK/luggagedataset/NewLuggageDataset/LuggageDataset_70/data.yaml'
MODEL = 'yolov12s.pt'
IMGSZ = 640
BATCH = 58
EPOCHS = 70
PATIENCE = 100
CLOSE_MOSAIC = 10
SEED = 0
DETERMINISTIC = True
PROJECT_DIR = 'runs/luggage_round8'

# ============================================================================
# BASE CONFIG: SWA-const06 + WIoU (proven winners from Rounds 1-7)
# ============================================================================
BASE = {
    # SWA (Section A) -- constant alpha=0.6 (Round 2 winner)
    'alpha_start': 0.6, 'alpha_end': 0.6,
    'alpha_min': 0.6, 'alpha_max': 0.6,
    'small_obj_px': 48, 'small_obj_boost': 1.75,
    # WIoU (Section I) -- precision winner
    'box_loss_type': 'wiou',
    'wiou_alpha': 1.9, 'wiou_delta': 3.0, 'wiou_momentum': 0.02,
    # TAL stock (Section D)
    'tal_topk': 10, 'tal_alpha': 0.5, 'tal_beta': 6.0,
    # Class weighting ON (Section F)
    'use_class_weighting': True, 'class_weight_mode': 'sqrt',
    # Everything else OFF
    'use_satal': False, 'use_nwd': False,
    'use_repulsion': False, 'center_loss_weight_init': 0.0,
    'swa_smooth': False, 'use_loss_clip': True,
    # v3 sections -- all OFF by default
    'use_ar_weight': False, 'swa_smooth_v2': False,
    'use_class_focus': False, 'use_dfl_iou_weight': False,
}

# ============================================================================
# PHASE 1: ISOLATED TESTS -- each changes EXACTLY ONE section from base
# ============================================================================
PHASE1 = {
    # Section K: AR-aware weighting
    # Root Cause #1: train AR=1.46, test AR=2.58 (+77% shift)
    # Up-weights loss for tall/thin GT boxes
    'r8_ar_weight': {
        'use_ar_weight': True,
        'ar_threshold': 2.0,
        'ar_max_weight': 1.5,
        'ar_saturation': 4.0,
    },
    # Section L: Smooth SWA v2
    # Sigmoid ramp replaces hard step at 48px
    # Better gradient properties, no discontinuity
    'r8_swa_smooth': {
        'swa_smooth_v2': True,
        'swa_sharpness': 5.0,
    },
    # Section M: WIoU class-focus
    # Bag is hardest (74% mAP50) -- give it 30% more box-loss attention
    'r8_class_focus': {
        'use_class_focus': True,
        'class_focus_backpack': 1.1,
        'class_focus_bag': 1.3,
        'class_focus_trolley': 1.0,
    },
    # Section N: IoU-conditioned DFL
    # Boost DFL for moderate-IoU boxes (0.5-0.8) that determine mAP50-95
    'r8_dfl_refine': {
        'use_dfl_iou_weight': True,
        'dfl_refine_boost': 0.5,
        'dfl_refine_center': 0.65,
        'dfl_refine_sigma': 0.15,
    },
}

# ============================================================================
# PHASE 2: COMBINE WINNERS (run ONLY after Phase 1 results are in)
# Only combine sections that individually beat the baseline
# ============================================================================
PHASE2 = {
    # AR + smooth SWA (both target shape/size distribution mismatch)
    'r8_ar_smooth': {
        'use_ar_weight': True, 'ar_threshold': 2.0,
        'ar_max_weight': 1.5, 'ar_saturation': 4.0,
        'swa_smooth_v2': True, 'swa_sharpness': 5.0,
    },
    # AR + DFL refine (shape-aware + tighter box distribution)
    'r8_ar_dfl': {
        'use_ar_weight': True, 'ar_threshold': 2.0,
        'ar_max_weight': 1.5, 'ar_saturation': 4.0,
        'use_dfl_iou_weight': True, 'dfl_refine_boost': 0.5,
        'dfl_refine_center': 0.65, 'dfl_refine_sigma': 0.15,
    },
    # AR + class focus (shape awareness + bag difficulty)
    'r8_ar_class': {
        'use_ar_weight': True, 'ar_threshold': 2.0,
        'ar_max_weight': 1.5, 'ar_saturation': 4.0,
        'use_class_focus': True, 'class_focus_backpack': 1.1,
        'class_focus_bag': 1.3, 'class_focus_trolley': 1.0,
    },
    # Full stack (ONLY if Phase 1 shows >= 2 winners)
    'r8_full': {
        'use_ar_weight': True, 'ar_threshold': 2.0,
        'ar_max_weight': 1.5, 'ar_saturation': 4.0,
        'swa_smooth_v2': True, 'swa_sharpness': 5.0,
        'use_class_focus': True, 'class_focus_backpack': 1.1,
        'class_focus_bag': 1.3, 'class_focus_trolley': 1.0,
        'use_dfl_iou_weight': True, 'dfl_refine_boost': 0.5,
        'dfl_refine_center': 0.65, 'dfl_refine_sigma': 0.15,
    },
}


def build_cmd(name, overrides):
    """Build YOLO training command."""
    cfg = {**BASE, **overrides}
    cmd = [
        sys.executable, '-m', 'ultralytics', 'detect', 'train',
        f'data={DATASET_YAML}', f'model={MODEL}',
        f'imgsz={IMGSZ}', f'batch={BATCH}', f'epochs={EPOCHS}',
        f'patience={PATIENCE}', f'close_mosaic={CLOSE_MOSAIC}',
        f'seed={SEED}', f'deterministic={DETERMINISTIC}',
        f'project={PROJECT_DIR}', f'name={name}',
    ]
    for k, v in cfg.items():
        cmd.append(f'{k}={v}')
    return cmd


def run_single(name, overrides, phase):
    """Execute a single training run."""
    cmd = build_cmd(name, overrides)
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'[{phase}] Starting: {name}')
    print(f'{sep}')
    print(f'Delta from base:')
    for k, v in overrides.items():
        print(f'  {k}: {v}')
    print(f'Command: {" ".join(str(c) for c in cmd)}\n')
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f'[WARNING] {name} exited with code {result.returncode}')
        return False
    return True


def main():
    import argparse
    p = argparse.ArgumentParser(description='Round 8 Ablation')
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--run', type=str, default=None,
                   help='Run a specific experiment by name')
    p.add_argument('--dry-run', action='store_true',
                   help='Print commands without executing')
    args = p.parse_args()

    runs = PHASE1 if args.phase == 1 else PHASE2
    phase = f'PHASE {args.phase}'

    if args.run:
        if args.run not in runs:
            print(f'Unknown: {args.run}. Available: {list(runs.keys())}')
            sys.exit(1)
        runs = {args.run: runs[args.run]}

    print(f'\n{"#" * 70}')
    print(f'  Round 8 Ablation -- {phase}')
    print(f'  Runs: {list(runs.keys())}')
    print(f'  Base: SWA-const06 + WIoU')
    print(f'  Started: {datetime.now().isoformat()}')
    print(f'{"#" * 70}\n')

    results = {}
    for name, ov in runs.items():
        if args.dry_run:
            cmd = build_cmd(name, ov)
            print(f'[DRY] {name}: {" ".join(str(c) for c in cmd)}\n')
            continue
        results[name] = 'OK' if run_single(name, ov, phase) else 'FAIL'

    if not args.dry_run:
        print(f'\n{"=" * 70}')
        print(f'Round 8 {phase} -- Summary')
        print(f'{"=" * 70}')
        for n, s in results.items():
            print(f'  {n}: {s}')
        print(f'{"=" * 70}\n')


if __name__ == '__main__':
    main()
