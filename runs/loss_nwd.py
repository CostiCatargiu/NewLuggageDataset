# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# SATAL-SWA-Plus-NWD: Combined Loss Function with Normalized Wasserstein Distance
# =============================================================================
#
# Base: loss_satal_swa.py (SATAL + SWA)
# Added from CustomLoss2:
#   - Class Weighting (inverse-frequency weights for class imbalance)
#   - Varifocal Loss (better for dense scenes)
# Added NWD:
#   - Normalized Wasserstein Distance for small object detection
#   - Better gradient signal for small boxes (IoU degrades rapidly)
#   - Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, crop_mask
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import OKS_SIGMA

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


# =============================================================================
# NWD (Normalized Wasserstein Distance) IMPLEMENTATION
# =============================================================================
#
# Based on: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
# Paper: https://arxiv.org/abs/2110.13389
#
# Key insight: Model bboxes as 2D Gaussian distributions, then compute
# Wasserstein-2 distance. This provides smooth gradients even for small objects
# where IoU degrades rapidly.
#
# Paper convention:
# - Each bbox modeled as 2D Gaussian with Σ = diag((w/2)², (h/2)²)
# - W2² = ||center_diff||² + ||sigma_diff||²
# - NWD = exp(-W2 / C) where C is a constant (paper uses ~12.8 for AI-TOD)
# - For YOLO at 640px input, C ≈ 12-16 works well
# =============================================================================


def bbox2gaussian(bboxes, eps=1e-7):
    """
    Convert bboxes (xyxy) to 2D Gaussian parameters.

    Each bbox is modeled as a 2D Gaussian:
    - Mean (μ) = center of bbox (cx, cy)
    - Std (σ) = (w/2, h/2) following paper convention
      Σ = diag((w/2)², (h/2)²)

    Args:
        bboxes: Bounding boxes in xyxy format (..., 4)
        eps: Small value for numerical stability

    Returns:
        cx, cy, sigma_x, sigma_y: Gaussian parameters
    """
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2
    w = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=eps)
    h = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=eps)

    # Paper convention: Σ = diag((w/2)², (h/2)²)
    # So σ_x = w/2, σ_y = h/2
    sigma_x = w / 2
    sigma_y = h / 2

    return cx, cy, sigma_x, sigma_y


def wasserstein2_squared(pred_bboxes, target_bboxes, eps=1e-7):
    """
    Compute squared Wasserstein-2 distance between bbox Gaussians.

    For 2D Gaussians with diagonal covariance (Bures metric):
    W2² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

    Where ||σ₁ - σ₂||² = (σ_x1 - σ_x2)² + (σ_y1 - σ_y2)²

    Args:
        pred_bboxes: Predicted boxes in xyxy format (N, 4)
        target_bboxes: Target boxes in xyxy format (N, 4)
        eps: Small value for numerical stability

    Returns:
        Squared Wasserstein-2 distance (N,)
    """
    # Get Gaussian parameters
    pred_cx, pred_cy, pred_sx, pred_sy = bbox2gaussian(pred_bboxes, eps)
    tgt_cx, tgt_cy, tgt_sx, tgt_sy = bbox2gaussian(target_bboxes, eps)

    # Squared distance between means (centers)
    center_dist_sq = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Squared distance between standard deviations
    sigma_dist_sq = (pred_sx - tgt_sx) ** 2 + (pred_sy - tgt_sy) ** 2

    # Total W2²
    w2_squared = center_dist_sq + sigma_dist_sq

    return w2_squared


# =============================================================================
# NWD Debug Helpers (defined BEFORE nwd_loss for proper ordering)
# =============================================================================

_NWD_DEBUG_DONE = True


def nwd_debug_print(w2, nwd, C):
    """Print NWD debug stats once per training run."""
    global _NWD_DEBUG_DONE
    if _NWD_DEBUG_DONE:
        return

    print(f"\n{'=' * 60}")
    print(f"[NWD DEBUG] First batch stats:")
    print(f"{'=' * 60}")
    print(f"  W2: mean={w2.mean().item():.3f}, median={w2.median().item():.3f}, max={w2.max().item():.3f}")
    print(f"  C={C}")
    print(f"  NWD: mean={nwd.mean().item():.4f}, min={nwd.min().item():.4f}, max={nwd.max().item():.4f}")
    print(f"  Loss: mean={(1 - nwd).mean().item():.4f}")
    print(f"")
    print(f"  → If NWD mean ≈ 1.0, C is too large (try C={C / 2})")
    print(f"  → If NWD mean ≈ 0.0, C is too small (try C={C * 2})")
    print(f"  → Target: NWD mean ≈ 0.3-0.7 for useful gradients")
    print(f"{'=' * 60}\n")

    _NWD_DEBUG_DONE = True


def reset_nwd_debug():
    """Call this to re-enable debug print for next training run."""
    global _NWD_DEBUG_DONE
    _NWD_DEBUG_DONE = False


# =============================================================================
# NWD Loss Function
# =============================================================================

def nwd_loss(pred_bboxes, target_bboxes, C=4.0, eps=1e-7):
    """
    Compute Normalized Wasserstein Distance (NWD) loss.

    NWD = exp(-sqrt(W2²) / C)
    Loss = 1 - NWD (in range [0, 1], like IoU loss)

    Args:
        pred_bboxes: Predicted boxes in xyxy format (N, 4)
        target_bboxes: Target boxes in xyxy format (N, 4)
        C: Normalization constant (scalar).
           - Paper uses ~12.8 for AI-TOD in PIXEL coordinates
           - For YOLO stride-normalized coords, use C ≈ 2-6
           - Luggage dataset with 640px input: start with C=4
        eps: Small value for numerical stability

    Returns:
        NWD loss in range [0, 1) where 0 = perfect match (N,)
    """
    # Compute squared Wasserstein distance
    w2_squared = wasserstein2_squared(pred_bboxes, target_bboxes, eps)
    w2 = torch.sqrt(w2_squared.clamp(min=eps))

    # NWD = exp(-W2 / C) ∈ (0, 1]
    # C is a scalar constant (paper-faithful approach)
    # When pred == target: W2=0, NWD=1
    # When pred is far from target: W2 large, NWD→0
    nwd = torch.exp(-w2 / C)

    # Debug: print stats once to help tune C (no recompute after first batch)
    if not _NWD_DEBUG_DONE:
        nwd_debug_print(w2, nwd, C)

    # Loss = 1 - NWD ∈ [0, 1)
    loss = 1.0 - nwd

    return loss


# =============================================================================
# BASE LOSS COMPONENTS (same as loss_satal_swa.py)
# =============================================================================


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma=1.5, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, label, gamma=None, alpha=None):
        gamma = gamma if gamma is not None else self.gamma
        alpha = alpha if alpha is not None else self.alpha

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor

        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.mean(1).sum()
        return loss


class DFLoss(nn.Module):
    """Distribution Focal Loss for bounding box regression."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """
    Bounding box loss with SWA (Size Weight Adaptive) and NWD (Normalized Wasserstein Distance).

    NWD is especially beneficial for small objects where IoU degrades rapidly.
    Can use pure NWD, pure CIoU, or a blend of both.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.reg_max = reg_max

        # Training state
        self.epoch = 0
        self.total_epochs = 70

        # Section A defaults (SWA)
        self.small_obj_px = 70
        self.small_obj_boost = 1.5
        self.alpha_start = 0.9
        self.alpha_end = 0.5
        self.alpha_min = 0.3
        self.alpha_max = 0.9

        # Section C: Adaptive clipping defaults
        self.iou_clip_start = 20.0
        self.iou_clip_end = 10.0
        self.dfl_clip_start = 10.0
        self.dfl_clip_end = 5.0

        # Section H: NWD (Normalized Wasserstein Distance) defaults
        # NWD provides better gradient signal for small objects
        self.use_nwd = False  # Enable NWD loss
        self.nwd_mode = 'blend'  # 'pure', 'blend', 'small_only'
        self.nwd_weight = 0.5  # Weight for NWD when blending (0-1)
        self.nwd_C = 4.0  # NWD normalization constant
        # Paper uses ~12.8 for PIXEL coords
        # For stride-normalized coords, use 2-6
        self.nwd_small_threshold = 32.0  # Area threshold for 'small_only' mode (stride-normalized coords²)

    def set_params(self, hyp):
        """Set parameters from hyperparameters (model.args)."""
        # Section A: Size-aware weighting
        self.small_obj_px = getattr(hyp, 'small_obj_px', self.small_obj_px)
        self.small_obj_boost = getattr(hyp, 'small_obj_boost', self.small_obj_boost)
        self.alpha_start = getattr(hyp, 'alpha_start', self.alpha_start)
        self.alpha_end = getattr(hyp, 'alpha_end', self.alpha_end)
        self.alpha_min = getattr(hyp, 'alpha_min', self.alpha_min)
        self.alpha_max = getattr(hyp, 'alpha_max', self.alpha_max)
        self.total_epochs = getattr(hyp, 'epochs', self.total_epochs)

        # Section C: Adaptive clipping
        self.iou_clip_start = getattr(hyp, 'iou_clip_start', self.iou_clip_start)
        self.iou_clip_end = getattr(hyp, 'iou_clip_end', self.iou_clip_end)
        self.dfl_clip_start = getattr(hyp, 'dfl_clip_start', self.dfl_clip_start)
        self.dfl_clip_end = getattr(hyp, 'dfl_clip_end', self.dfl_clip_end)

        # Section H: NWD parameters
        self.use_nwd = getattr(hyp, 'use_nwd', self.use_nwd)
        self.nwd_mode = getattr(hyp, 'nwd_mode', self.nwd_mode)
        self.nwd_weight = getattr(hyp, 'nwd_weight', self.nwd_weight)
        self.nwd_C = getattr(hyp, 'nwd_C', self.nwd_C)
        self.nwd_small_threshold = getattr(hyp, 'nwd_small_threshold', self.nwd_small_threshold)

    def _get_dynamic_alpha(self):
        """Calculate dynamic alpha based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        alpha = self.alpha_start * (1 - progress) + self.alpha_end * progress
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))

        if not hasattr(self, '_last_logged_epoch'):
            self._last_logged_epoch = -1

        if self.epoch != self._last_logged_epoch:
            if self.epoch % 10 == 0:
                print(f"[Alpha] Epoch {self.epoch}/{self.total_epochs}: α={alpha:.3f}")
            self._last_logged_epoch = self.epoch

        return alpha

    def _compute_target_areas(self, target_bboxes, fg_mask):
        """Compute target bounding box areas with numerical stability."""
        areas = (target_bboxes[..., 2] - target_bboxes[..., 0]) * \
                (target_bboxes[..., 3] - target_bboxes[..., 1])
        return areas.clamp(min=1e-6)

    def _compute_weights(self, target_bboxes, target_scores, fg_mask, stride=None):
        """Compute combined area and score weights for loss calculation."""
        target_areas = self._compute_target_areas(target_bboxes, fg_mask)

        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        area_weight = (1.0 / target_areas[fg_mask]).unsqueeze(-1)

        # Normalize area weights
        if area_weight.numel() > 0:
            area_weight = area_weight / (area_weight.max() + 1e-8)

        # Apply small object boost
        if stride is not None and area_weight.numel() > 0:
            min_stride = stride.min().clamp_min(1.0)
            small_threshold = (self.small_obj_px / min_stride) ** 2
            fg_areas = target_areas[fg_mask]
            small_mask = fg_areas < small_threshold

            if small_mask.any():
                area_weight = area_weight.clone()
                area_weight[small_mask] *= self.small_obj_boost

        return score_weight, area_weight

    def _get_gradient_clip_values(self):
        """Get adaptive gradient clipping values based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        max_iou = self.iou_clip_end + (self.iou_clip_start - self.iou_clip_end) * (1 - progress)
        max_dfl = self.dfl_clip_end + (self.dfl_clip_start - self.dfl_clip_end) * (1 - progress)
        return max_iou, max_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        """Compute IoU/NWD and DFL losses with SWA weighting."""

        alpha = self._get_dynamic_alpha()
        score_weight, area_weight = self._compute_weights(
            target_bboxes, target_scores, fg_mask, stride
        )

        # Combined weight (SWA)
        weight = alpha * area_weight + (1 - alpha) * score_weight

        # Get foreground boxes
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # =====================================================================
        # Compute box regression loss (IoU, NWD, or blend)
        # =====================================================================
        # Always compute CIoU first
        iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        ciou_loss = 1.0 - iou  # CIoU loss in [0, 1]

        if self.use_nwd:
            # Compute NWD loss with scalar C constant (paper-faithful)
            # Both CIoU and NWD losses are in [0, 1] range
            # Debug print fires once inside nwd_loss() on first batch
            nwd_loss_val = nwd_loss(pred_fg, target_fg, C=self.nwd_C)

            if self.nwd_mode == 'pure':
                # Pure NWD loss (no CIoU)
                per_sample_box_loss = nwd_loss_val * weight

            elif self.nwd_mode == 'blend':
                # Weighted blend of CIoU and NWD
                # Both losses are in [0, 1], so blend is also in [0, 1]
                blended_loss = (1.0 - self.nwd_weight) * ciou_loss + self.nwd_weight * nwd_loss_val
                per_sample_box_loss = blended_loss * weight

            elif self.nwd_mode == 'small_only':
                # Use NWD only for small objects, CIoU for larger ones
                # Note: target_fg is in stride-normalized coordinates
                target_areas = (target_fg[..., 2] - target_fg[..., 0]) * \
                               (target_fg[..., 3] - target_fg[..., 1])
                is_small = target_areas < self.nwd_small_threshold

                # Use NWD for small, CIoU for large
                per_sample_loss = torch.where(is_small, nwd_loss_val, ciou_loss)
                per_sample_box_loss = per_sample_loss * weight
            else:
                # Fallback to pure CIoU
                per_sample_box_loss = ciou_loss * weight
        else:
            # Standard CIoU loss (no NWD)
            per_sample_box_loss = ciou_loss * weight

        # Get adaptive clip values
        max_iou_clip, max_dfl_clip = self._get_gradient_clip_values()

        # Clip PER-SAMPLE
        per_sample_box_loss = per_sample_box_loss.clamp(max=max_iou_clip / 10.0)

        # Aggregate
        loss_iou = per_sample_box_loss.sum() / target_scores_sum

        # DFL loss per sample
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            per_sample_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight

            # Clip PER-SAMPLE
            per_sample_dfl = per_sample_dfl.clamp(max=max_dfl_clip / 10.0)

            # Aggregate
            loss_dfl = per_sample_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing rotated bounding box losses."""

    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                xywh2xyxy(target_bboxes[..., :4]),
                self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


# =============================================================================
# MAIN DETECTION LOSS CLASS
# =============================================================================


class v8DetectionLoss:
    """
    SATAL-SWA-Plus Detection Loss.

    Base: loss_satal_swa.py (SATAL + SWA)
    Added from CustomLoss2:
      - Class Weighting (inverse-frequency weights)
      - Varifocal Loss (better for dense scenes)
    """

    def __init__(self, model, tal_topk=10):
        """Initialize v8DetectionLoss with parameters from model.args."""

        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]  # Detect() module
        self._model = model

        # Model properties
        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1

        # Training state
        self.epoch = 0
        self.total_epochs = getattr(h, 'epochs', 70)

        # =====================================================================
        # Section A: Size-aware weighting (SWA)
        # =====================================================================
        self.small_obj_px = getattr(h, 'small_obj_px', 70)
        self.small_obj_boost = getattr(h, 'small_obj_boost', 1.5)
        self.alpha_start = getattr(h, 'alpha_start', 0.9)
        self.alpha_end = getattr(h, 'alpha_end', 0.5)
        self.alpha_min = getattr(h, 'alpha_min', 0.3)
        self.alpha_max = getattr(h, 'alpha_max', 0.9)

        # =====================================================================
        # Section B: Center loss
        # =====================================================================
        self.center_loss_weight_init = getattr(h, 'center_loss_weight_init', 0.0)
        self.center_loss_weight_min = getattr(h, 'center_loss_weight_min', 0.01)
        self.center_loss_decay_epochs = getattr(h, 'center_loss_decay_epochs', 35)

        # =====================================================================
        # Section C: Adaptive clipping
        # =====================================================================
        self.iou_clip_start = getattr(h, 'iou_clip_start', 20.0)
        self.iou_clip_end = getattr(h, 'iou_clip_end', 10.0)
        self.dfl_clip_start = getattr(h, 'dfl_clip_start', 10.0)
        self.dfl_clip_end = getattr(h, 'dfl_clip_end', 5.0)

        # =====================================================================
        # Section D: TAL parameters
        # =====================================================================
        self.tal_topk = getattr(h, 'tal_topk', tal_topk)
        self.tal_alpha = getattr(h, 'tal_alpha', 0.5)
        self.tal_beta = getattr(h, 'tal_beta', 6.0)

        # =====================================================================
        # Section E: SA-TAL (Scale-Adaptive Task Aligned Assigner)
        # =====================================================================
        self.use_satal = getattr(h, 'use_satal', False)
        self.satal_alpha_small = getattr(h, 'satal_alpha_small', 1.5)
        self.satal_beta_small = getattr(h, 'satal_beta_small', 3.0)
        self.satal_alpha_large = getattr(h, 'satal_alpha_large', 1.0)
        self.satal_beta_large = getattr(h, 'satal_beta_large', 6.0)
        self.satal_small_area = getattr(h, 'satal_small_area', 0.0025)
        self.satal_large_area = getattr(h, 'satal_large_area', 0.0225)
        self.satal_topk_factor = getattr(h, 'satal_topk_factor', 1.5)

        # =====================================================================
        # Section F: Class Weighting (from CustomLoss2) - ALWAYS ON
        # =====================================================================
        # Dataset: backpack=34901, bag=28628, trolley=66946
        # Inverse frequency, sqrt-dampened, mean-normalized
        class_counts = torch.tensor([34901.0, 28628.0, 66946.0], device=device)
        inv_freq = 1.0 / class_counts
        inv_freq = inv_freq / inv_freq.mean()
        self.class_weights = torch.sqrt(inv_freq)
        self.class_weights = self.class_weights / self.class_weights.mean()
        # Result: backpack≈1.08, bag≈1.19, trolley≈0.78

        # =====================================================================
        # Section G: Classification Loss Mode
        # =====================================================================
        # Using BCE + Class Weighting (no VFL - VFL hurt performance with SATAL)

        # =====================================================================
        # Section H: NWD (Normalized Wasserstein Distance) for small objects
        # =====================================================================
        # NWD provides better gradient signal for small objects where IoU degrades
        # Both NWD loss and CIoU loss are in [0, 1] range - properly normalized!
        # Modes:
        #   - 'pure': Use only NWD (no CIoU)
        #   - 'blend': Weighted combination of CIoU + NWD (recommended)
        #   - 'small_only': NWD for small objects, CIoU for larger ones
        # C: Paper uses ~12.8 for AI-TOD in PIXEL coords
        #    For stride-normalized coords (YOLO), use C ≈ 2-6
        self.use_nwd = getattr(h, 'use_nwd', True)
        self.nwd_mode = getattr(h, 'nwd_mode', 'blend')
        self.nwd_weight = getattr(h, 'nwd_weight', 0.5)  # Weight for NWD in blend mode
        self.nwd_C = getattr(h, 'nwd_C', 4.0)  # Start with 4, tune based on debug output
        self.nwd_small_threshold = getattr(h, 'nwd_small_threshold', 32.0)  # For small_only mode

        # =====================================================================
        # LOSS FUNCTIONS
        # =====================================================================

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.bbox_loss.set_params(h)

        # Task Aligned Assigner
        if self.use_satal:
            from ultralytics.utils.satal import ScaleAdaptiveTaskAlignedAssigner
            self.assigner = ScaleAdaptiveTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                alpha_small=self.satal_alpha_small,
                beta_small=self.satal_beta_small,
                alpha_large=self.satal_alpha_large,
                beta_large=self.satal_beta_large,
                small_area_thresh=self.satal_small_area,
                large_area_thresh=self.satal_large_area,
                topk_small_factor=self.satal_topk_factor
            )
        else:
            self.assigner = TaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta
            )

        # Projection for DFL
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # Print configuration
        self._print_config()

    def _print_config(self):
        """Print current configuration for verification."""
        if not hasattr(self, '_config_printed'):
            print("\n" + "=" * 60)
            print("SATAL-SWA-Plus-NWD Detection Loss Configuration")
            print("=" * 60)
            print(f"  [A] alpha_start:     {self.alpha_start}")
            print(f"  [A] alpha_end:       {self.alpha_end}")
            print(f"  [A] small_obj_px:    {self.small_obj_px}")
            print(f"  [A] small_obj_boost: {self.small_obj_boost}")
            print(f"  [B] center_loss_init: {self.center_loss_weight_init}")
            print(f"  [C] iou_clip:        {self.iou_clip_start} → {self.iou_clip_end}")
            print(f"  [D] tal_topk:        {self.tal_topk}")
            print(f"  [D] tal_alpha:       {self.tal_alpha}")
            print(f"  [D] tal_beta:        {self.tal_beta}")
            print(f"  [E] use_satal:       {self.use_satal}")
            if self.use_satal:
                print(f"      satal_alpha_small: {self.satal_alpha_small}")
                print(f"      satal_beta_small:  {self.satal_beta_small}")
                print(f"      satal_topk_factor: {self.satal_topk_factor}")
            print(f"  [F] Class Weighting: ALWAYS ON")
            print(f"      weights (bp/bg/tr): {self.class_weights.cpu().numpy().round(3)}")
            print(f"  [G] Cls Loss: BCE + Class Weighting (no VFL)")
            print(f"  [H] use_nwd:         {self.use_nwd}")
            if self.use_nwd:
                print(f"      nwd_mode:        {self.nwd_mode}")
                print(f"      nwd_weight:      {self.nwd_weight}")
                print(f"      nwd_C:           {self.nwd_C}")
                if self.nwd_mode == 'small_only':
                    print(f"      nwd_small_thresh: {self.nwd_small_threshold}")
            print(f"  epochs:              {self.total_epochs}")
            print("=" * 60 + "\n")
            self._config_printed = True

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess target counts and matches with input batch size."""
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]

        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bounding box coordinates."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_center_loss(self, pred_bboxes, target_bboxes, fg_mask, stride_tensor):
        """Compute auxiliary center loss for small objects (Section B)."""
        if self.center_loss_weight_init <= 0:
            return torch.tensor(0.0, device=self.device)

        if not fg_mask.any():
            return torch.tensor(0.0, device=self.device)

        fg_indices = torch.nonzero(fg_mask, as_tuple=True)
        if len(fg_indices[0]) == 0:
            return torch.tensor(0.0, device=self.device)

        pred_fg = pred_bboxes[fg_indices[0], fg_indices[1]]
        target_fg = target_bboxes[fg_indices[0], fg_indices[1]]

        pred_centers = (pred_fg[:, :2] + pred_fg[:, 2:]) / 2
        target_centers = (target_fg[:, :2] + target_fg[:, 2:]) / 2

        target_areas = (target_fg[:, 2] - target_fg[:, 0]) * (target_fg[:, 3] - target_fg[:, 1])

        min_stride = stride_tensor.min().clamp_min(1.0)
        small_obj_threshold = (self.small_obj_px / min_stride) ** 2
        small_obj_mask = target_areas < small_obj_threshold

        if not small_obj_mask.any():
            return torch.tensor(0.0, device=self.device)

        center_l1_loss = F.l1_loss(
            pred_centers[small_obj_mask],
            target_centers[small_obj_mask],
            reduction='mean'
        )

        progress = min(self.epoch / max(self.center_loss_decay_epochs, 1), 1.0)
        weight = self.center_loss_weight_init * (1 - progress)
        weight = max(self.center_loss_weight_min, weight)

        return center_l1_loss * weight

    def _sync_bbox_loss_state(self):
        """Synchronize epoch information with bbox_loss module."""
        self.bbox_loss.epoch = self.epoch
        self.bbox_loss.total_epochs = self.total_epochs

    # =========================================================================
    # CLASSIFICATION LOSS (BCE + Class Weighting, no VFL)
    # =========================================================================

    def _compute_cls_loss_weighted(self, pred_scores, target_scores, fg_mask,
                                   target_labels_for_fg, target_scores_sum):
        """
        Compute class-weighted BCE classification loss.

        Uses standard BCE with per-anchor class weighting based on inverse
        class frequency. No Varifocal Loss - just simple weighted BCE.

        Class weights: backpack≈1.08, bag≈1.19, trolley≈0.78
        """
        dtype = pred_scores.dtype
        bs, num_anchors, nc = pred_scores.shape

        # Standard BCE loss
        bce = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)

        # Build per-anchor weight tensor for class weighting
        weight = torch.ones(bs, num_anchors, 1, device=self.device, dtype=dtype)

        if fg_mask.any() and target_labels_for_fg.numel() > 0:
            # Apply class weights to foreground anchors
            fg_class_weights = self.class_weights.to(dtype)[target_labels_for_fg]
            weight[fg_mask] = fg_class_weights.unsqueeze(-1)

        # Weighted sum, normalized by target_scores_sum
        loss = (bce * weight).sum() / target_scores_sum

        return loss

    # =========================================================================
    # MAIN LOSS COMPUTATION
    # =========================================================================

    def __call__(self, preds, batch):
        """Calculate the sum of detection losses (box, cls, dfl)."""

        # Try to get epoch from model
        try:
            if hasattr(self._model, 'current_epoch'):
                self.epoch = self._model.current_epoch
        except:
            pass

        self._sync_bbox_loss_state()
        loss = torch.zeros(3, device=self.device)

        # Extract features
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Prepare targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Set image size for SA-TAL
        if hasattr(self.assigner, 'set_imgsz'):
            self.assigner.set_imgsz(imgsz)

        # Task Aligned Assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ── Extract class labels for foreground anchors (from CustomLoss2) ──
        if fg_mask.any():
            target_labels_for_fg = target_scores[fg_mask].argmax(dim=-1)
        else:
            target_labels_for_fg = torch.tensor([], device=self.device, dtype=torch.long)

        # ── Classification loss (class-weighted + VFL from CustomLoss2) ──
        loss[1] = self._compute_cls_loss_weighted(
            pred_scores, target_scores, fg_mask, target_labels_for_fg, target_scores_sum
        )

        # ── Bounding box losses (SWA from loss_satal_swa.py) ──
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            self._sync_bbox_loss_state()

            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )

            # Add auxiliary center loss (Section B)
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor
            )
            loss[0] = loss[0] + center_loss

        # Apply loss gains
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


# =============================================================================
# OTHER LOSS CLASSES (same as loss_satal_swa.py)
# =============================================================================


class v8ClassificationLoss:
    """Criterion class for computing classification training losses."""

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for oriented bounding box (OBB) detection."""

    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=self.tal_topk,
            num_classes=self.nc,
            alpha=self.tal_alpha,
            beta=self.tal_beta
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.focal_loss = FocalLoss(gamma=1.5, alpha=0.25)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 6, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 6, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                bboxes = targets[matches, 2:]
                bboxes[..., :4].mul_(scale_tensor)
                out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)

        return out

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes[..., :4],
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for pose estimation."""

    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for segmentation."""

    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
        except Exception:
            batch_idx = torch.zeros(batch["cls"].shape[0], 1, device=self.device)

        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes / stride_tensor, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz,
                                    overlap):
        mask_h, mask_w = masks.shape[1:]
        loss = 0

        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(
                zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, target_bboxes_normalized)
        ):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, target_bboxes_i = single_i

            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += v8SegmentationLoss.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
        return loss / fg_mask.sum()

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        pred_mask = (pred @ proto.view(proto.shape[0], -1)).view(-1, proto.shape[1], proto.shape[2])
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()


class E2EDetectLoss:
    """
    End-to-end detection loss specialized for luggage datasets.

    Uses v8DetectionLossLuggage for both one-to-many and one-to-one branches,
    applying class weighting, size-aware IoU loss, and varifocal loss.
    """

    def __init__(self, model):
        """Initialize with luggage-specialized losses for both branches."""
        self.one2many = v8DetectionLossLuggage(model, tal_topk=10)
        self.one2one = v8DetectionLossLuggage(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
