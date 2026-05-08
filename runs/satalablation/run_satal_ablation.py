#!/usr/bin/env python3
"""
SA-TAL v3 Ablation: 6 Targeted Experiments
===========================================

Runs 6 carefully designed SA-TAL parameter configurations
based on analysis of satal2 results.

Baseline (satal2):
    mAP50_all:       0.7405 (valid), 0.7296 (test)
    mAP50_small:     0.7123 (valid), 0.6867 (test)
    mAP50-95_small:  0.4320 (valid), 0.3986 (test)

All experiments use:
    - Standard v8DetectionLoss (NOT custom loss)
    - original6 architecture
    - satal2 base config (topk=10, alpha=0.6, beta=5.0)
    - Only SA-TAL small-object parameters change

Usage:
    python satal3_ablation.py                    # Run all 6
    python satal3_ablation.py --run R1           # Run specific
    python satal3_ablation.py --run R1 R4        # Run specific subset
    python satal3_ablation.py --status           # Show progress
    python satal3_ablation.py --results          # Show results
"""

import json
import time
import gc
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION — UPDATE THESE
# =============================================================================

DATA_YAML = "/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubset/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"              # UPDATE (YAML or .pt)
PROJECT_DIR = "runs_satal3_ablation"
EVAL_SCRIPT = None  # Path to your evaluation script if you have one

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# SATAL2 BASELINE (PROVEN BEST — DO NOT MODIFY)
# =============================================================================

SATAL2_BASE = {
    # TAL core
    "tal_topk": 10,
    "tal_alpha": 0.6,
    "tal_beta": 5.0,
    # SA-TAL
    "use_satal": True,
    "satal_alpha_small": 1.2,
    "satal_beta_small": 4.5,
    "satal_alpha_large": 1.0,
    "satal_beta_large": 6.0,
    "satal_small_area": 0.0025,
    "satal_large_area": 0.0225,
    "satal_topk_factor": 1.3,
    # Training schedule
    "alpha_start": 0.7,
    "alpha_end": 0.4,
    "alpha_min": 0.3,
    "alpha_max": 0.7,
    # Disabled features
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    "iou_clip_start": 1000.0,
    "iou_clip_end": 1000.0,
    "dfl_clip_start": 1000.0,
    "dfl_clip_end": 1000.0,
}


# =============================================================================
# 6 EXPERIMENTS
# =============================================================================

EXPERIMENTS = {
    "R1": {
        "name": "R1_beta_small_5.0",
        "description": "Recover small object localization — beta_small 4.5→5.0",
        "hypothesis": "beta_small=4.5 trades too much localization for detection. "
                      "5.0 keeps IoU relevant for tighter boxes while still "
                      "being lower than the default 6.0",
        "changes": {"satal_beta_small": 5.0},
    },
    "R2": {
        "name": "R2_alpha1.3_thresh35",
        "description": "More classification trust + wider small zone",
        "hypothesis": "Median object is 45px but threshold 0.0025=32px misses many. "
                      "Raising to 0.0035 (~38px) gives more objects small treatment. "
                      "alpha_small=1.3 trusts classification more for these objects.",
        "changes": {"satal_alpha_small": 1.3, "satal_small_area": 0.0035},
    },
    "R3": {
        "name": "R3_factor1.5_beta5.0",
        "description": "More candidates + better localization",
        "hypothesis": "1.5 factor gives small objects 15 candidates (vs 13). "
                      "Combined with beta_small=5.0 the extra candidates are "
                      "selected with better IoU quality.",
        "changes": {"satal_topk_factor": 1.5, "satal_beta_small": 5.0},
    },
    "R4": {
        "name": "R4_balanced",
        "description": "Gentle tuning on all levers simultaneously",
        "hypothesis": "Small improvements on each lever may compound. "
                      "beta=5.0 (localization), alpha=1.3 (classification), "
                      "factor=1.5 (candidates), thresh=0.0035 (coverage).",
        "changes": {
            "satal_beta_small": 5.0,
            "satal_alpha_small": 1.3,
            "satal_topk_factor": 1.5,
            "satal_small_area": 0.0035,
        },
    },
    "R5": {
        "name": "R5_aggressive_small",
        "description": "Push small object focus to find the ceiling",
        "hypothesis": "Strong cls boost (1.4) + more candidates (1.5) + "
                      "wider zone (0.0040 ~40px). Keep beta=4.5 to not "
                      "lose detection ability. Tests how far we can push.",
        "changes": {
            "satal_alpha_small": 1.4,
            "satal_topk_factor": 1.5,
            "satal_small_area": 0.0040,
        },
    },
    "R6": {
        "name": "R6_narrow_gap",
        "description": "Narrower transition zone between small and large",
        "hypothesis": "Current gap 0.0025-0.0225 leaves a big medium zone "
                      "with standard treatment. Narrowing to 0.0030-0.0180 "
                      "means more objects get adaptive alpha/beta.",
        "changes": {
            "satal_beta_small": 5.0,
            "satal_alpha_small": 1.3,
            "satal_small_area": 0.0030,
            "satal_large_area": 0.0180,
        },
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def make_config(changes: Dict) -> Dict:
    """Create full config from satal2 baseline + specific changes."""
    config = dict(SATAL2_BASE)
    config.update(changes)
    return config


def cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def on_train_epoch_start(trainer):
    """Sync epoch to loss function for dynamic alpha scheduling."""
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
# RESULTS MANAGER
# =============================================================================

class Results:
    """Track experiment results."""

    def __init__(self):
        self.project_dir = Path(PROJECT_DIR)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.project_dir / "satal3_results.json"
        self.data = self._load()

    def _load(self) -> Dict:
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {"experiments": [], "satal2_baseline": SATAL2_BASE}

    def save(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, run_id: str) -> bool:
        return any(e["run_id"] == run_id for e in self.data["experiments"])

    def get_result(self, run_id: str) -> Optional[Dict]:
        for e in self.data["experiments"]:
            if e["run_id"] == run_id:
                return e
        return None

    def add(self, run_id: str, name: str, config: Dict, metrics: Dict,
            changes: Dict, time_hours: float):
        self.data["experiments"].append({
            "run_id": run_id,
            "name": name,
            "changes_from_satal2": changes,
            "full_config": config,
            "metrics": metrics,
            "time_hours": round(time_hours, 3),
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def print_status(self):
        """Print current progress."""
        print(f"\n{'=' * 90}")
        print(f"SA-TAL v3 ABLATION STATUS")
        print(f"{'=' * 90}")
        print(f"{'Run':<6} {'Name':<30} {'Status':<10} {'mAP50_all':<12} "
              f"{'mAP50_small':<14} {'mAP50-95_sm':<14}")
        print(f"{'-' * 90}")

        # Baseline reference
        print(f"{'base':<6} {'satal2 (reference)':<30} {'---':<10} "
              f"{'0.7405':<12} {'0.7123':<14} {'0.4320':<14}")

        for run_id in ["R1", "R2", "R3", "R4", "R5", "R6"]:
            exp_def = EXPERIMENTS[run_id]
            result = self.get_result(run_id)

            if result:
                m = result["metrics"]
                map50 = m.get("mAP50_all", 0)
                map50_sm = m.get("mAP50_small", 0)
                map50_95_sm = m.get("mAP50_95_small", 0)

                # Delta vs satal2
                d_all = map50 - 0.7405
                d_sm = map50_sm - 0.7123

                arrow_all = "↑" if d_all > 0.002 else "↓" if d_all < -0.002 else "≈"
                arrow_sm = "↑" if d_sm > 0.002 else "↓" if d_sm < -0.002 else "≈"

                print(f"{run_id:<6} {exp_def['name']:<30} {'DONE':<10} "
                      f"{map50:<12.4f} {map50_sm:<14.4f} {map50_95_sm:<14.4f} "
                      f"{arrow_all}{d_all:+.3f} {arrow_sm}{d_sm:+.3f}")
            else:
                print(f"{run_id:<6} {exp_def['name']:<30} {'PENDING':<10} "
                      f"{'--':<12} {'--':<14} {'--':<14}")

        # Summary
        completed = [e for e in self.data["experiments"]]
        pending = [r for r in EXPERIMENTS if not self.is_completed(r)]
        print(f"\nCompleted: {len(completed)}/6 | Remaining: {len(pending)} "
              f"(~{len(pending) * 1.3:.1f} hours)")

    def print_results(self):
        """Print detailed results comparison."""
        completed = self.data["experiments"]
        if not completed:
            print("No results yet.")
            return

        print(f"\n{'=' * 110}")
        print(f"SA-TAL v3 DETAILED RESULTS")
        print(f"{'=' * 110}")

        # Sort by mAP50_all
        sorted_by_all = sorted(completed, key=lambda x: x["metrics"].get("mAP50_all", 0), reverse=True)

        print(f"\n--- Sorted by mAP50_all ---")
        print(f"{'Rank':<6} {'Run':<6} {'Name':<30} {'mAP50_all':<12} {'Δ_all':<10} "
              f"{'mAP50_sm':<12} {'Δ_sm':<10} {'mAP50-95_sm':<14}")
        print(f"{'-' * 110}")

        print(f"{'ref':<6} {'base':<6} {'satal2':<30} {'0.7405':<12} {'---':<10} "
              f"{'0.7123':<12} {'---':<10} {'0.4320':<14}")

        for i, exp in enumerate(sorted_by_all, 1):
            m = exp["metrics"]
            map50 = m.get("mAP50_all", 0)
            map50_sm = m.get("mAP50_small", 0)
            map50_95_sm = m.get("mAP50_95_small", 0)
            d_all = map50 - 0.7405
            d_sm = map50_sm - 0.7123

            marker = " ← BEST" if i == 1 else ""
            print(f"{i:<6} {exp['run_id']:<6} {exp['name']:<30} "
                  f"{map50:<12.4f} {d_all:<+10.4f} "
                  f"{map50_sm:<12.4f} {d_sm:<+10.4f} "
                  f"{map50_95_sm:<14.4f}{marker}")

        # Sort by mAP50_small
        sorted_by_small = sorted(completed, key=lambda x: x["metrics"].get("mAP50_small", 0), reverse=True)

        print(f"\n--- Sorted by mAP50_small ---")
        for i, exp in enumerate(sorted_by_small, 1):
            m = exp["metrics"]
            marker = " ← BEST SMALL" if i == 1 else ""
            print(f"  {i}. {exp['run_id']} {exp['name']}: "
                  f"mAP50_small={m.get('mAP50_small', 0):.4f}, "
                  f"mAP50-95_small={m.get('mAP50_95_small', 0):.4f}{marker}")

        # Print what changed in each
        print(f"\n--- Changes from satal2 baseline ---")
        for exp in completed:
            changes = exp.get("changes_from_satal2", {})
            changes_str = ", ".join(f"{k}={v}" for k, v in changes.items())
            print(f"  {exp['run_id']}: {changes_str}")


# =============================================================================
# TRAINING
# =============================================================================

def run_single_experiment(run_id: str, results: Results) -> bool:
    """Run a single experiment. Returns True if successful."""

    if run_id not in EXPERIMENTS:
        print(f"ERROR: Unknown run ID '{run_id}'. Valid: {list(EXPERIMENTS.keys())}")
        return False

    if results.is_completed(run_id):
        print(f"\n{run_id} already completed. Skipping.")
        return True

    exp_def = EXPERIMENTS[run_id]
    name = exp_def["name"]
    changes = exp_def["changes"]
    config = make_config(changes)

    # Print experiment info
    print(f"\n{'=' * 70}")
    print(f"  {run_id}: {name}")
    print(f"{'=' * 70}")
    print(f"  Description: {exp_def['description']}")
    print(f"  Hypothesis:  {exp_def['hypothesis']}")
    print(f"")
    print(f"  Changes from satal2:")
    for k, v in changes.items():
        baseline_v = SATAL2_BASE.get(k, "N/A")
        print(f"    {k}: {baseline_v} → {v}")
    print(f"")
    print(f"  Full SA-TAL config:")
    print(f"    alpha_small:  {config['satal_alpha_small']}")
    print(f"    beta_small:   {config['satal_beta_small']}")
    print(f"    topk_factor:  {config['satal_topk_factor']}")
    print(f"    small_area:   {config['satal_small_area']}")
    print(f"    large_area:   {config['satal_large_area']}")
    print(f"    alpha_large:  {config['satal_alpha_large']}")
    print(f"    beta_large:   {config['satal_beta_large']}")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    try:
        # Load model
        model = YOLO(MODEL_WEIGHTS)
        model.add_callback('on_train_epoch_start', on_train_epoch_start)
        print(f"  [OK] Model loaded: {MODEL_WEIGHTS}")
        print(f"  [OK] Epoch sync callback registered")

        # Train
        train_results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=name,
            # SA-TAL parameters
            use_satal=config["use_satal"],
            tal_topk=config["tal_topk"],
            tal_alpha=config["tal_alpha"],
            tal_beta=config["tal_beta"],
            satal_alpha_small=config["satal_alpha_small"],
            satal_beta_small=config["satal_beta_small"],
            satal_alpha_large=config["satal_alpha_large"],
            satal_beta_large=config["satal_beta_large"],
            satal_small_area=config["satal_small_area"],
            satal_large_area=config["satal_large_area"],
            satal_topk_factor=config["satal_topk_factor"],
            # Training schedule
            alpha_start=config["alpha_start"],
            alpha_end=config["alpha_end"],
            alpha_min=config["alpha_min"],
            alpha_max=config["alpha_max"],
            # Disabled
            center_loss_weight_init=config["center_loss_weight_init"],
            center_loss_weight_min=config["center_loss_weight_min"],
            center_loss_decay_epochs=config["center_loss_decay_epochs"],
            iou_clip_start=config["iou_clip_start"],
            iou_clip_end=config["iou_clip_end"],
            dfl_clip_start=config["dfl_clip_start"],
            dfl_clip_end=config["dfl_clip_end"],
            # Reproducibility
            seed=0,
            deterministic=True,
        )

        # Extract metrics
        r = train_results.results_dict
        metrics = {
            "mAP50_all": float(r.get("metrics/mAP50(B)", 0)),
            "mAP50_95_all": float(r.get("metrics/mAP50-95(B)", 0)),
            "precision": float(r.get("metrics/precision(B)", 0)),
            "recall": float(r.get("metrics/recall(B)", 0)),
        }

        # ── Run your custom evaluation for size-based metrics ──
        # Replace this section with your actual eval pipeline
        best_model_path = Path(PROJECT_DIR) / name / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"\n  Running evaluation on best model...")
            eval_model = YOLO(str(best_model_path))

            # Validation set eval
            val_results = eval_model.val(data=DATA_YAML, imgsz=IMG_SIZE, split="val", verbose=False)
            if hasattr(val_results, 'results_dict'):
                vr = val_results.results_dict
                metrics["mAP50_all"] = float(vr.get("metrics/mAP50(B)", metrics["mAP50_all"]))
                metrics["mAP50_95_all"] = float(vr.get("metrics/mAP50-95(B)", metrics["mAP50_95_all"]))

            # NOTE: For size-based metrics (mAP50_small, etc.), you need your
            # custom evaluation script. Update these with actual values:
            metrics.setdefault("mAP50_small", 0.0)
            metrics.setdefault("mAP50_95_small", 0.0)
            metrics.setdefault("mAP50_medium", 0.0)
            metrics.setdefault("mAP50_large", 0.0)

            del eval_model

        training_time = (time.time() - start_time) / 3600

        # Print results
        d_all = metrics["mAP50_all"] - 0.7405
        d_sm = metrics.get("mAP50_small", 0) - 0.7123

        print(f"\n{'=' * 60}")
        print(f"  {run_id} COMPLETED: {name}")
        print(f"{'=' * 60}")
        print(f"  mAP50_all:      {metrics['mAP50_all']:.4f}  ({d_all:+.4f} vs satal2)")
        print(f"  mAP50_small:    {metrics.get('mAP50_small', 0):.4f}  ({d_sm:+.4f} vs satal2)")
        print(f"  mAP50-95_small: {metrics.get('mAP50_95_small', 0):.4f}")
        print(f"  mAP50-95_all:   {metrics['mAP50_95_all']:.4f}")
        print(f"  precision:      {metrics['precision']:.4f}")
        print(f"  recall:         {metrics['recall']:.4f}")
        print(f"  time:           {training_time:.2f}h")
        print(f"{'=' * 60}\n")

        # Save
        results.add(run_id, name, config, metrics, changes, training_time)

        # Cleanup
        del model
        cleanup()

        return True

    except Exception as e:
        print(f"\n  ERROR in {run_id}: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SA-TAL v3 Ablation — 6 Targeted Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python satal3_ablation.py                  # Run all 6
    python satal3_ablation.py --run R1         # Run only R1
    python satal3_ablation.py --run R1 R4      # Run R1 and R4
    python satal3_ablation.py --status         # Show progress
    python satal3_ablation.py --results        # Show detailed results
        """
    )
    parser.add_argument("--run", nargs="*", default=None,
                        help="Specific runs to execute (e.g., R1 R4)")
    parser.add_argument("--status", action="store_true",
                        help="Show current progress")
    parser.add_argument("--results", action="store_true",
                        help="Show detailed results")

    args = parser.parse_args()
    results = Results()

    # Header
    print(f"\n{'#' * 70}")
    print(f"#  SA-TAL v3 ABLATION — 6 Targeted Experiments")
    print(f"#  Baseline: satal2 (mAP50=0.7405, mAP50_small=0.7123)")
    print(f"#  Project:  {PROJECT_DIR}")
    print(f"#  Loss:     Standard v8DetectionLoss")
    print(f"{'#' * 70}")

    if args.status:
        results.print_status()
        return

    if args.results:
        results.print_results()
        return

    # Determine which runs to execute
    if args.run:
        run_ids = [r.upper() for r in args.run]
        invalid = [r for r in run_ids if r not in EXPERIMENTS]
        if invalid:
            print(f"ERROR: Unknown run IDs: {invalid}")
            print(f"Valid: {list(EXPERIMENTS.keys())}")
            return
    else:
        run_ids = ["R1", "R2", "R3", "R4", "R5", "R6"]

    # Show what will run
    pending = [r for r in run_ids if not results.is_completed(r)]
    skipped = [r for r in run_ids if results.is_completed(r)]

    if skipped:
        print(f"\nSkipping completed: {', '.join(skipped)}")
    if pending:
        print(f"Will run: {', '.join(pending)}")
        print(f"Estimated time: ~{len(pending) * 1.3:.1f} hours")
    else:
        print(f"\nAll requested experiments already completed!")
        results.print_status()
        return

    print(f"\nStarting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)

    # Run experiments
    total = len(pending)
    success = 0
    failed = []

    for i, run_id in enumerate(pending, 1):
        print(f"\n{'>' * 70}")
        print(f"  [{i}/{total}] Starting {run_id}: {EXPERIMENTS[run_id]['name']}")
        print(f"{'>' * 70}")

        ok = run_single_experiment(run_id, results)
        if ok:
            success += 1
        else:
            failed.append(run_id)

        # Show progress after each run
        results.print_status()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE")
    print(f"{'=' * 70}")
    print(f"  Successful: {success}/{total}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'=' * 70}\n")

    results.print_results()


if __name__ == "__main__":
    main()
