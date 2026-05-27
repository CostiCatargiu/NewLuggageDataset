#!/usr/bin/env python3
"""
PyTorch SWA (Stochastic Weight Averaging) on top of Size-Weight Adaptive loss.

This adds actual model weight averaging from PyTorch's torch.optim.swa_utils.
SWA averages model weights over the last N epochs, finding flatter minima
that generalize better.

This is INDEPENDENT from the SWA loss (Size-Weight Adaptive) — they stack:
  - SWA loss: controls how much area vs score weighting in the loss function
  - PyTorch SWA: averages model weights for better generalization

How it works:
  1. Train normally for `swa_start_epoch` epochs (e.g., 60)
  2. From epoch 60 onwards, accumulate weight averages
  3. At the end of training, replace model weights with the averaged weights
  4. Update batch normalization statistics with the averaged model
  5. Save the SWA model as best.pt

Usage:
  python run_pytorch_swa.py

Based on: https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
"""

import time
import gc
import copy
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetHistogram17percentage2/data.yaml"  # DS2
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_new_weapon_dataset"

EPOCHS = 80
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# PyTorch SWA settings
SWA_START_EPOCH = 40  # Start averaging from epoch 40
# Averages over last 40 epochs (40-80)

# =============================================================================
# FIXED PARAMETERS (same as your best config)
# =============================================================================
FIXED_PARAMS = {
    # Phase A: SWA loss (Size-Weight Adaptive)
    "alpha_start": 0.9,
    "alpha_end": 0.4,
    "alpha_min": 0.3,
    "alpha_max": 1.0,
    "small_obj_px": 48,
    "small_obj_boost": 1.0,
    # Phase B: disabled
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
    # Phase C: disabled
    "iou_clip_start": 100.0,
    "iou_clip_end": 100.0,
    "dfl_clip_start": 100.0,
    "dfl_clip_end": 100.0,
    # Phase D: TAL
    "tal_topk": 10,
    "tal_alpha": 0.5,
    "tal_beta": 6.0,
}

# =============================================================================
# PyTorch SWA CALLBACKS
# =============================================================================

# Global SWA state
swa_state = {
    "swa_model": None,
    "n_averaged": 0,
    "start_epoch": SWA_START_EPOCH,
}


def on_train_epoch_start(trainer):
    """Sync epoch to loss function for dynamic alpha scheduling."""
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
            if epoch % 10 == 0:
                print(f"[Epoch Sync] Epoch {epoch}")
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


def on_train_epoch_end(trainer):
    """Update SWA model after each epoch (from SWA_START_EPOCH onwards)."""
    epoch = trainer.epoch

    if epoch < swa_state["start_epoch"]:
        return

    # Initialize SWA model on first SWA epoch
    if swa_state["swa_model"] is None:
        print(f"\n[PyTorch SWA] Initializing weight averaging at epoch {epoch}")
        # Get the actual model (unwrap DDP if needed)
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module

        # Create averaged model
        swa_state["swa_model"] = AveragedModel(model, device=trainer.device)
        swa_state["n_averaged"] = 0
        print(f"[PyTorch SWA] SWA model created on {trainer.device}")

    # Update averaged weights
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    swa_state["swa_model"].update_parameters(model)
    swa_state["n_averaged"] += 1

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"[PyTorch SWA] Epoch {epoch}: averaged {swa_state['n_averaged']} checkpoints")


def on_train_end(trainer):
    """Replace model weights with SWA averaged weights and update BN."""
    if swa_state["swa_model"] is None:
        print("[PyTorch SWA] No SWA model found — skipping")
        return

    print(f"\n{'=' * 60}")
    print(f"[PyTorch SWA] Training complete — applying averaged weights")
    print(f"[PyTorch SWA] Averaged over {swa_state['n_averaged']} checkpoints")
    print(f"{'=' * 60}")

    # Get the actual model
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    # Copy SWA averaged weights to the model
    swa_avg_state = swa_state["swa_model"].module.state_dict()
    model.load_state_dict(swa_avg_state)

    print(f"[PyTorch SWA] Weights replaced with averaged version")

    # Update BatchNorm statistics with averaged model
    # Need to run one pass through training data
    print(f"[PyTorch SWA] Updating BatchNorm statistics...")
    try:
        model.train()
        with torch.no_grad():
            # Reset BN running stats
            for module in model.modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    module.reset_running_stats()
                    module.momentum = None  # use cumulative moving average

            # Run through a subset of training data to update BN
            dataloader = trainer.train_loader
            n_batches = min(len(dataloader), 50)  # Use up to 50 batches
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                imgs = batch["img"].to(trainer.device).float() / 255.0
                model(imgs)
                if i % 10 == 0:
                    print(f"  BN update: {i+1}/{n_batches} batches")

        print(f"[PyTorch SWA] BatchNorm updated with {n_batches} batches")
    except Exception as e:
        print(f"[PyTorch SWA] Warning: BN update failed: {e}")
        print(f"[PyTorch SWA] Continuing with SWA weights (BN stats from last epoch)")

    # Also update the EMA model with SWA weights
    if hasattr(trainer, 'ema') and trainer.ema is not None:
        trainer.ema.ema.load_state_dict(swa_avg_state)
        print(f"[PyTorch SWA] EMA model also updated with SWA weights")

    # Save the SWA model
    save_dir = Path(trainer.save_dir)
    swa_path = save_dir / "weights" / "swa_best.pt"
    try:
        ckpt = {
            "model": copy.deepcopy(model).half(),
            "epoch": trainer.epoch,
            "swa_averaged": swa_state["n_averaged"],
        }
        torch.save(ckpt, swa_path)
        print(f"[PyTorch SWA] SWA model saved to {swa_path}")
    except Exception as e:
        print(f"[PyTorch SWA] Warning: Could not save SWA model: {e}")

    print(f"{'=' * 60}\n")


# =============================================================================
# TRAINING
# =============================================================================

def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    name = "swa_09_04_ds2_pytorch_swa"

    print(f"\n{'#' * 70}")
    print(f"# PyTorch SWA Experiment")
    print(f"# Config: swa_09_04 (Size-Weight Adaptive) + PyTorch SWA (Weight Averaging)")
    print(f"# SWA starts at epoch {SWA_START_EPOCH}, averages last {EPOCHS - SWA_START_EPOCH} epochs")
    print(f"# Dataset: DS2")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)

    # Add callbacks
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_train_end', on_train_end)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=name,
        # SWA loss schedule (Size-Weight Adaptive)
        alpha_start=FIXED_PARAMS["alpha_start"],
        alpha_end=FIXED_PARAMS["alpha_end"],
        alpha_min=FIXED_PARAMS["alpha_min"],
        alpha_max=FIXED_PARAMS["alpha_max"],
        small_obj_px=FIXED_PARAMS["small_obj_px"],
        small_obj_boost=FIXED_PARAMS["small_obj_boost"],
        # Phase B: disabled
        center_loss_weight_init=FIXED_PARAMS["center_loss_weight_init"],
        center_loss_weight_min=FIXED_PARAMS["center_loss_weight_min"],
        center_loss_decay_epochs=FIXED_PARAMS["center_loss_decay_epochs"],
        # Phase C: disabled
        iou_clip_start=FIXED_PARAMS["iou_clip_start"],
        iou_clip_end=FIXED_PARAMS["iou_clip_end"],
        dfl_clip_start=FIXED_PARAMS["dfl_clip_start"],
        dfl_clip_end=FIXED_PARAMS["dfl_clip_end"],
        # TAL parameters
        tal_topk=FIXED_PARAMS["tal_topk"],
        tal_alpha=FIXED_PARAMS["tal_alpha"],
        tal_beta=FIXED_PARAMS["tal_beta"],
        # Reproducibility
        seed=0,
        deterministic=True,
    )

    training_time = (time.time() - start_time) / 3600

    print(f"\n{'=' * 70}")
    print(f"COMPLETED: {name} ({training_time:.2f}h)")
    print(f"{'=' * 70}")

    cleanup()


if __name__ == "__main__":
    main()
