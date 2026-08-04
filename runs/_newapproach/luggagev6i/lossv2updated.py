# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# Custom YOLO detection loss (luggage-dataset adapted)
# =============================================================================
#
# A drop-in replacement for the stock Ultralytics v8 detection loss with a set
# of optional, independently-toggled mechanisms for small / thin / imbalanced
# objects. Every optional mechanism defaults to OFF (or its legacy value), so
# with no extra hyperparameters this file behaves like the stock loss.
#
# All mechanisms are configured through model hyperparameters (read via
# `getattr(hyp, ...)`), so nothing here requires code changes to tune.
#
# -----------------------------------------------------------------------------
# Tunable mechanisms
# -----------------------------------------------------------------------------
#   [A]  Size-aware weighting (SWA): blends box-loss weight between a size term
#        and the TAL score via a scheduled alpha. Small objects (area vs.
#        per-anchor stride) can be up-weighted by small_obj_boost.
#          alpha_start, alpha_end, alpha_min, alpha_max,
#          small_obj_px, small_obj_boost
#        area_weight_mode ('inv'|'sqrt'|'log') reshapes the 1/area weight;
#        area_weight_norm ('max'|'mean') sets the normalization so shape and
#        average magnitude can be varied independently.
#        Optional per-class boost:
#          small_obj_boost_backpack, small_obj_boost_bag, small_obj_boost_trolley
#
#   [B]  Center loss: per-anchor-stride threshold with size-normalized L1.
#        'crowd' mode weights by GT neighbor overlap instead of size.
#          center_loss_weight_init, center_loss_weight_min,
#          center_loss_decay_epochs, center_loss_mode ('small'|'crowd'),
#          center_crowd_iou
#
#   [C]  Adaptive loss clipping (deprecated; keep off):
#          use_loss_clip (default True for legacy reproduction — set False),
#          iou_clip_start, iou_clip_end, dfl_clip_start, dfl_clip_end
#
#   [D]  Task-Aligned Assigner parameters:
#          tal_topk, tal_alpha, tal_beta
#
#   [E]  SA-TAL (Scale-Adaptive assigner): separate alpha/beta and more
#        positives for small vs. large objects.
#          use_satal, satal_alpha_small, satal_beta_small, satal_alpha_large,
#          satal_beta_large, satal_small_area, satal_large_area,
#          satal_topk_factor
#
#   [F]  Class weighting (always on): inverse-frequency, mean-normalized.
#        Classification uses BCE, optionally modulated by QFL (cls_mode='qfl').
#          cls_mode, qfl_beta, use_class_weighting, class_weight_mode
#
#   [K]  Classification SWA: extra weight on the classification loss for
#        small-object foreground anchors (same area-vs-stride smallness test
#        as Section A).
#          use_cls_swa, cls_swa_boost
#
#   [L]  Bag asymmetric penalty: up-weights only the NEGATIVE BCE term of the
#        bag logit at anchors assigned to other classes (suppresses confident
#        wrong-class bag scores; positives untouched).
#          use_bag_penalty, bag_penalty_weight, bag_class_id
#
#   [M]  AR-aware TAL: relaxes per-GT beta for high-aspect-ratio (tall/narrow)
#        boxes; stock behavior at/below the AR threshold.
#          use_artal, artal_ar_thresh, artal_ar_scale, artal_beta_relax
#
#   [N]  Supply-normalized TAL: per-GT positive budget k_eff scaled by the
#        GEOMETRIC candidate pool (anchor centers inside the GT), so
#        supply-poor small objects are not forced onto a diluted positive set.
#          use_snatal, snatal_rho, snatal_kmin
#
#   NWD: Normalized Wasserstein Distance box term, blended with the IoU loss
#        (better gradients for tiny boxes where IoU is near-degenerate).
#          use_nwd, nwd_mode, nwd_weight, nwd_C
#
# Notes:
#   * Sections M and N are mutually exclusive (enabling both raises an error).
#   * nwd_C is stride-normalized (~2-6), not pixels.
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
# ALTERNATIVE BBOX REGRESSION LOSSES
# =============================================================================
# MPDIoU, Wise-IoU v3, and Focaler-CIoU. Each takes xyxy boxes (N,4) and returns
# a PER-SAMPLE loss (N,), slotting straight into the SWA weighting pipeline.
# Corner/center distances are normalized by the smallest-enclosing-box diagonal,
# which is bounded and scale-adaptive in stride-normalized coordinates.


def _iou_geometry(pred, target, eps=1e-7):
    """Shared IoU + enclosing-box geometry for the alternative losses."""
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tx1, ty1, tx2, ty2 = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    pw = (px2 - px1).clamp(min=0);
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0);
    th = (ty2 - ty1).clamp(min=0)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    cx1 = torch.min(px1, tx1);
    cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2);
    cy2 = torch.max(py2, ty2)
    cw = (cx2 - cx1);
    ch = (cy2 - cy1)
    c2 = cw * cw + ch * ch + eps

    pcx = (px1 + px2) / 2;
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2;
    tcy = (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    return iou, rho2, c2, (pw, ph, tw, th)


def mpdiou_loss(pred, target, eps=1e-7):
    """MPDIoU-style loss: IoU minus normalized corner-point distances (tight boxes)."""
    iou, _, c2, _ = _iou_geometry(pred, target, eps)
    d1 = (pred[..., 0] - target[..., 0]) ** 2 + (pred[..., 1] - target[..., 1]) ** 2
    d2 = (pred[..., 2] - target[..., 2]) ** 2 + (pred[..., 3] - target[..., 3]) ** 2
    mpdiou = iou - (d1 + d2) / c2
    return 1.0 - mpdiou


def focaler_ciou_loss(pred, target, d_lo=0.0, u_hi=0.95, eps=1e-7):
    """Focaler-CIoU: CIoU with an IoU-range remap that focuses on hard samples."""
    iou, rho2, c2, (pw, ph, tw, th) = _iou_geometry(pred, target, eps)
    pw = pw.clamp(min=eps);
    ph = ph.clamp(min=eps)
    tw = tw.clamp(min=eps);
    th = th.clamp(min=eps)
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        a = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + a * v)
    l_ciou = 1.0 - ciou
    iou_focaler = ((iou - d_lo) / (u_hi - d_lo + eps)).clamp(0.0, 1.0)
    return l_ciou + iou - iou_focaler


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


def _dfl_edge_entropy(pred_dist):
    """Mean Shannon entropy of the softmax bin distributions in pred_dist.

    pred_dist: (K, reg_max) logits for K edges. Lower entropy = sharper (more
    peaked) distribution = tighter edge estimate. Returned as a scalar to be
    ADDED to (and thus minimized within) the DFL loss. Ported verbatim from
    loss_dflentropy._dfl_edge_entropy (the r10_dfl_entropy mechanism).
    """
    logp = F.log_softmax(pred_dist, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1).mean()


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
        # v6i re-fit: mean object area ~2145px^2 (v5i was ~3690px^2). The old
        # default 70 (=4900px^2) sat ABOVE the mean, so "small" captured almost
        # everything. 36 (=1296px^2 ~ 0.6x mean area) restores the original v5i
        # SCOPE (v5i used 48px = 2304/3690 = 0.62x mean). Override via hyp
        # small_obj_px to sweep.
        self.small_obj_px = 36
        self.small_obj_boost = 1.5
        self.alpha_start = 0.9
        self.alpha_end = 0.5
        self.alpha_min = 0.3
        self.alpha_max = 0.9

        # Section A2: area-weight shape + per-class boost. Defaults = legacy.
        self.area_weight_mode = 'inv'   # 'inv' (legacy 1/area) | 'sqrt' | 'log'
        # Normalization of the area weight:
        #   'max'  (legacy): divide by batch max — the shape and the average
        #          box-loss magnitude move together (entangled), and the scale
        #          is set by the single most extreme box per batch.
        #   'mean': divide by batch mean — equal average loss scale across
        #          modes, so inv/sqrt/log compare SHAPES only.
        self.area_weight_norm = 'max'
        self.class_boosts = None        # tensor [backpack, bag, trolley] or None (scalar boost)

        # Section O: DFL distribution-entropy sharpening (ported from
        # loss_dflentropy / r10_dfl_entropy = 57.71, large +2.89 vs baseline).
        # Adds w * mean(entropy of the fg edge softmax distributions) to the DFL
        # loss -> penalizes flat/multi-modal edge distributions -> sharper,
        # more unimodal decoded edges. GLOBAL variant (all 4 edges), which is
        # the one that produced the best result. Default 0.0 -> inert (stock).
        self.dfl_entropy_weight = 0.0

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

        # Section I: alternative regression losses
        self.box_loss_type = 'ciou'  # 'ciou' | 'mpdiou' | 'wiou' | 'focaler'
        self.wiou_alpha = 1.9  # WIoUv3 non-monotonic focusing
        self.wiou_delta = 3.0
        self.wiou_momentum = 0.02  # EMA momentum for running IoU-loss mean
        self._wiou_mean = None
        self.focaler_d = 0.0  # Focaler-IoU lower bound
        self.focaler_u = 0.95  # Focaler-IoU upper bound

        # SWA: optional smooth (continuous) small-object boost instead of hard step
        self.swa_smooth = False
        self.swa_boost_power = 0.5

        # Optional per-sample loss clip (deprecated; kept toggleable).
        # Default stays True only for legacy reproduction — set use_loss_clip=False
        # explicitly in new hyp files.
        self.use_loss_clip = True

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

        # Section I: alternative regression losses
        self.box_loss_type = getattr(hyp, 'box_loss_type', self.box_loss_type)
        self.wiou_alpha = getattr(hyp, 'wiou_alpha', self.wiou_alpha)
        self.wiou_delta = getattr(hyp, 'wiou_delta', self.wiou_delta)
        self.wiou_momentum = getattr(hyp, 'wiou_momentum', self.wiou_momentum)
        self.focaler_d = getattr(hyp, 'focaler_d', self.focaler_d)
        self.focaler_u = getattr(hyp, 'focaler_u', self.focaler_u)
        self.swa_smooth = getattr(hyp, 'swa_smooth', self.swa_smooth)
        self.swa_boost_power = getattr(hyp, 'swa_boost_power', self.swa_boost_power)
        self.use_loss_clip = getattr(hyp, 'use_loss_clip', self.use_loss_clip)

        # Section A2: area-weight shape + normalization + per-class boost
        self.area_weight_mode = getattr(hyp, 'area_weight_mode', self.area_weight_mode)
        self.area_weight_norm = getattr(hyp, 'area_weight_norm', self.area_weight_norm)
        # Section O: DFL entropy sharpening
        self.dfl_entropy_weight = getattr(hyp, 'dfl_entropy_weight', self.dfl_entropy_weight)
        b_bp = getattr(hyp, 'small_obj_boost_backpack', -1.0)
        b_bg = getattr(hyp, 'small_obj_boost_bag', -1.0)
        b_tr = getattr(hyp, 'small_obj_boost_trolley', -1.0)
        if max(b_bp, b_bg, b_tr) > 0:
            # any unspecified class falls back to the scalar small_obj_boost
            self.class_boosts = torch.tensor([
                b_bp if b_bp > 0 else self.small_obj_boost,
                b_bg if b_bg > 0 else self.small_obj_boost,
                b_tr if b_tr > 0 else self.small_obj_boost,
            ])
        else:
            self.class_boosts = None

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

    def _compute_weights(self, target_bboxes, target_scores, fg_mask, stride=None, fg_labels=None):
        """Compute combined area and score weights for loss calculation."""
        target_areas = self._compute_target_areas(target_bboxes, fg_mask)

        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Section A2: area-weight shape. 'inv' is the exact legacy weight.
        # Dataset areas span 28x (p10..p90); raw 1/area concentrates all weight
        # on the tiniest boxes — sqrt/log spread emphasis over small+medium.
        inv_area = 1.0 / target_areas[fg_mask]
        if self.area_weight_mode == 'sqrt':
            area_weight = inv_area.sqrt().unsqueeze(-1)
        elif self.area_weight_mode == 'log':
            area_weight = torch.log1p(inv_area).unsqueeze(-1)
        else:  # 'inv' — legacy
            area_weight = inv_area.unsqueeze(-1)

        # Normalize area weights ('max' legacy | 'mean')
        if area_weight.numel() > 0:
            if self.area_weight_norm == 'mean':
                # equal average loss scale across shape modes — decouples the
                # SHAPE of the weighting from its MAGNITUDE in the A2 sweep
                area_weight = area_weight / (area_weight.mean() + 1e-8)
            else:  # 'max' — legacy (exact v2 behavior)
                area_weight = area_weight / (area_weight.max() + 1e-8)

        # Apply small-object boost with PER-ANCHOR stride (fix vs. global stride.min())
        if stride is not None and area_weight.numel() > 0:
            s_col = stride.reshape(-1)  # (total_anchors,)
            bs = fg_mask.shape[0]
            s_full = s_col.unsqueeze(0).expand(bs, -1)  # (bs, total_anchors)
            stride_fg = s_full[fg_mask].clamp_min(1.0)  # (M,)
            fg_areas = target_areas[fg_mask]  # (M,)
            small_threshold = (self.small_obj_px / stride_fg) ** 2  # per-anchor (M,)

            # Section A2: per-class boost — bag (smallest + precision
            # bottleneck) can be boosted harder than backpack/trolley.
            if self.class_boosts is not None and fg_labels is not None and fg_labels.numel() > 0:
                boost = self.class_boosts.to(area_weight.device, area_weight.dtype)[fg_labels]  # (M,)
            else:
                boost = None  # scalar legacy path

            area_weight = area_weight.clone()
            if self.swa_smooth:
                # Continuous boost: smaller boxes lifted up to the boost cap, no step
                ratio = (small_threshold / fg_areas.clamp(min=1e-9)).clamp(min=1.0)
                factor = ratio.pow(self.swa_boost_power)
                if boost is not None:
                    factor = torch.minimum(factor, boost)
                else:
                    factor = factor.clamp(max=self.small_obj_boost)
                area_weight = area_weight * factor.unsqueeze(-1)
            else:
                small_mask = fg_areas < small_threshold
                if small_mask.any():
                    if boost is not None:
                        area_weight[small_mask] *= boost[small_mask].unsqueeze(-1)
                    else:
                        area_weight[small_mask] *= self.small_obj_boost

        return score_weight, area_weight

    def _get_gradient_clip_values(self):
        """Get adaptive gradient clipping values based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        max_iou = self.iou_clip_end + (self.iou_clip_start - self.iou_clip_end) * (1 - progress)
        max_dfl = self.dfl_clip_end + (self.dfl_clip_start - self.dfl_clip_end) * (1 - progress)
        return max_iou, max_dfl

    def _regression_loss(self, pred_fg, target_fg):
        """Per-sample box regression loss selected by box_loss_type. Returns (N,1)."""
        t = self.box_loss_type
        if t == 'mpdiou':
            reg = mpdiou_loss(pred_fg, target_fg)
        elif t == 'wiou':
            reg = self._wiou_loss(pred_fg, target_fg)
        elif t == 'focaler':
            reg = focaler_ciou_loss(pred_fg, target_fg, self.focaler_d, self.focaler_u)
        else:  # 'ciou' -- exact baseline via ultralytics bbox_iou
            iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            reg = 1.0 - iou
        if reg.dim() == 1:
            reg = reg.unsqueeze(-1)
        return reg

    def _wiou_loss(self, pred_fg, target_fg, eps=1e-7):
        """Wise-IoU v3: non-monotonic dynamic focusing on ordinary-quality anchors."""
        iou, rho2, c2, _ = _iou_geometry(pred_fg, target_fg, eps)
        l_iou = 1.0 - iou
        r_wiou = torch.exp(rho2 / c2.detach())  # distance attention, [1, e)
        m = l_iou.mean().detach()
        if self._wiou_mean is None:
            self._wiou_mean = m
        else:
            self._wiou_mean = (1.0 - self.wiou_momentum) * self._wiou_mean + self.wiou_momentum * m
        beta = l_iou.detach() / (self._wiou_mean + eps)  # outlier degree
        r = beta / (self.wiou_delta * self.wiou_alpha ** (beta - self.wiou_delta))
        return r * r_wiou * l_iou

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None, fg_labels=None):
        """Compute IoU/NWD and DFL losses with SWA weighting."""

        alpha = self._get_dynamic_alpha()
        score_weight, area_weight = self._compute_weights(
            target_bboxes, target_scores, fg_mask, stride, fg_labels
        )

        # Combined weight (SWA)
        weight = alpha * area_weight + (1 - alpha) * score_weight

        # Get foreground boxes
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # =====================================================================
        # Compute box regression loss (IoU, NWD, or blend)
        # =====================================================================
        reg_loss = self._regression_loss(pred_fg, target_fg)  # (N,1), respects box_loss_type

        if self.use_nwd:
            # NWD contribution (paper-faithful scalar C); shape-safe (N,1)
            nwd_loss_val = nwd_loss(pred_fg, target_fg, C=self.nwd_C).unsqueeze(-1)
            if self.nwd_mode == 'pure':
                base_loss = nwd_loss_val
            elif self.nwd_mode == 'blend':
                base_loss = (1.0 - self.nwd_weight) * reg_loss + self.nwd_weight * nwd_loss_val
            elif self.nwd_mode == 'small_only':
                target_areas = ((target_fg[..., 2] - target_fg[..., 0]) *
                                (target_fg[..., 3] - target_fg[..., 1])).unsqueeze(-1)
                is_small = target_areas < self.nwd_small_threshold
                base_loss = torch.where(is_small, nwd_loss_val, reg_loss)
            else:
                base_loss = reg_loss
        else:
            base_loss = reg_loss

        per_sample_box_loss = base_loss * weight

        # Get adaptive clip values
        max_iou_clip, max_dfl_clip = self._get_gradient_clip_values()

        # Clip PER-SAMPLE (toggleable; deprecated — typically inert)
        if self.use_loss_clip:
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

            # Clip PER-SAMPLE (toggleable)
            if self.use_loss_clip:
                per_sample_dfl = per_sample_dfl.clamp(max=max_dfl_clip / 10.0)

            # Aggregate
            loss_dfl = per_sample_dfl.sum() / target_scores_sum

            # Section O: DFL distribution-entropy sharpening (global variant).
            # Add w * mean-entropy of the fg edge softmax distributions so flat
            # / multi-modal edge distributions are penalized -> sharper decoded
            # edges. Inert when dfl_entropy_weight == 0 (default). Uses the SAME
            # fg edge logits already gathered for the DFL loss above.
            if self.dfl_entropy_weight > 0:
                fg_edge_logits = pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max)
                loss_dfl = loss_dfl + self.dfl_entropy_weight * _dfl_edge_entropy(fg_edge_logits)
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing rotated bounding box losses."""

    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None, fg_labels=None):
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
# Section M: AR-AWARE TASK-ALIGNED ASSIGNER
# =============================================================================
# Motivation (dataset analysis): test boxes are tall/narrow (median AR 2.58,
# median width ~26px at 640 input = 1-3 anchor columns at stride 16/32). TAL
# candidates need their anchor center INSIDE the GT, so narrow boxes get
# structurally fewer candidates regardless of area — SATAL's area split misses
# this. Fix: relax beta per-GT as aspect ratio rises, so high-AR boxes are less
# punished for the low IoU their thin geometry forces. Boxes with AR <=
# ar_thresh see EXACTLY stock behavior (beta_eff == beta), so this is
# non-uniform by construction (uniform loosening hurt).
#
#   ar        = max(w/h, h/w)                       per GT
#   t         = clamp((ar - ar_thresh)/ar_scale, 0, 1)
#   beta_eff  = clamp(beta - t*beta_relax, min=1)
#   align     = score^alpha * iou^beta_eff
#
# ARTAL and SNATAL are ALTERNATIVES (M re-ranks a starved GT's few candidates;
# N culls a supply-poor GT to its best few — opposite pulls on the same thin
# small boxes). Enabling both raises at init.
# =============================================================================


class ARAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with per-GT beta relaxed for high-aspect-ratio (tall/narrow) boxes."""

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 ar_thresh=1.7, ar_scale=1.2, beta_relax=2.0, eps=1e-9):
        # v6i defaults (median AR 1.54). Old v5i defaults were 2.0 / 2.0.
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.ar_thresh = ar_thresh
        self.ar_scale = ar_scale
        self.beta_relax = beta_relax

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Same as stock TAL get_box_metrics, but with per-GT beta."""
        na = pd_bboxes.shape[-2]
        mask_gt_b = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt_b] = pd_scores[ind[0], :, ind[1]][mask_gt_b]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt_b]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt_b]
        iou_fn = getattr(self, 'iou_calculation', None)
        if iou_fn is not None:
            overlaps[mask_gt_b] = iou_fn(gt_boxes, pd_boxes)
        else:  # older ultralytics without iou_calculation helper
            overlaps[mask_gt_b] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        # Per-GT beta: relax for high-AR boxes (stock beta at/below ar_thresh)
        w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=1e-6)  # (bs, n_max_boxes)
        h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=1e-6)
        ar = torch.maximum(w / h, h / w)
        t = ((ar - self.ar_thresh) / max(self.ar_scale, 1e-6)).clamp(0.0, 1.0)
        beta_eff = (self.beta - t * self.beta_relax).clamp(min=1.0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(beta_eff.unsqueeze(-1))
        return align_metric, overlaps


# =============================================================================
# Section N: SUPPLY-NORMALIZED TAL ASSIGNER (SNA-TAL)
# =============================================================================
# Motivation (anchor-footprint diagnostic): small objects are SUPPLY-LIMITED,
# not metric-penalized. At stride 16, 58.9% of GTs have a candidate pool
# smaller than topk; small GTs are forced to accept 54.3% of their pool
# (vs 4.3% for large), training on a DILUTED positive set (lower IoU, larger
# center distance). Stock TAL uses an ABSOLUTE topk for all objects, so the
# selectivity is dictated by geometry, not design.
#
# Fix: make the per-GT budget proportional to that GT's candidate supply:
#
#   pool_g   = number of candidate anchors for GT g
#   k_eff_g  = clamp( round(rho * pool_g), k_min, topk )
#
# so a supply-poor (small) GT keeps only its best few candidates instead of
# scraping the barrel, while supply-rich (large) GTs are unaffected (they hit
# the topk cap anyway).
# rho ~ 0.15..0.4; the diagnostic suggests small-object taken/GT drops from
# 8.64 -> ~2.5-5.8 across that range.
#
# pool_g is GEOMETRIC — the count of anchor centers inside GT g (stashed from
# select_candidates_in_gts). A metric-based pool (metrics > eps) would depend on
# prediction quality: at epoch 0 predictions are noise, IoU with thin GTs is ~0
# everywhere, so the measured pool collapses and k_eff pins to k_min for exactly
# the objects this mechanism should help, easing only as training improves — the
# INVERSE of a sensible schedule. The geometric pool is stable from batch 0.
# k_min default is 2 (round(0.25*3)=1 would train a GT on a single anchor —
# fragile when ~5 adjacent objects/img contest anchors).
# =============================================================================


class SupplyNormalizedTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with a per-GT top-k budget proportional to GEOMETRIC candidate supply."""

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 rho=0.25, k_min=2, eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.rho = rho
        self.k_min = int(k_min)
        self._mask_in_gts = None  # stashed geometric candidate mask (b, n, A)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        """Stock candidate test, stashing the geometric mask for pool counting.

        Stock TAL defines this as a @staticmethod but calls it via `self`, so
        this instance-method override intercepts the call transparently and
        delegates to the parent implementation (version-tolerant).
        """
        mask_in_gts = TaskAlignedAssigner.select_candidates_in_gts(xy_centers, gt_bboxes, eps)
        self._mask_in_gts = mask_in_gts
        return mask_in_gts

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """Per-GT supply-normalized top-k selection.

        Overrides stock TAL: instead of taking a fixed `self.topk` per GT, take
        k_eff_g = clamp(round(rho * pool_g), k_min, topk), where pool_g is the
        GEOMETRIC candidate count (anchor centers inside GT g) stashed by
        select_candidates_in_gts — stable from batch 0, independent of current
        prediction quality.

        metrics : (b, n_max_boxes, num_anchors) alignment metric, already zeroed
                  outside each GT's candidate set by mask_gt upstream.
        returns : (b, n_max_boxes, num_anchors) float 0/1 selection mask.
        """
        na = metrics.shape[-1]

        # Per-GT candidate pool size. Prefer the stashed geometric mask; fall
        # back to the metric-based count only if the stash is missing or
        # shape-mismatched (defensive against upstream version changes).
        if self._mask_in_gts is not None and self._mask_in_gts.shape == metrics.shape:
            pool = self._mask_in_gts.to(metrics.dtype).sum(-1)          # (b, n) geometric
        else:
            pool = (metrics > self.eps).sum(-1).to(metrics.dtype)       # (b, n) fallback
        # Supply-normalized budget, capped at the absolute topk.
        k_eff = (self.rho * pool.float()).round().clamp_(self.k_min, self.topk).long()  # (b, n)

        # Rank all anchors per GT once, then keep only the first k_eff of them.
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)  # (b,n,topk)

        # Column position 0..topk-1; keep column j for GT g iff j < k_eff_g.
        ar = torch.arange(self.topk, device=metrics.device).view(1, 1, -1)  # (1,1,topk)
        keep = ar < k_eff.unsqueeze(-1)                                     # (b,n,topk) bool
        # Also drop ranked slots whose metric is zero (empty pool padding /
        # padded GTs — their metric is all zeros regardless of pool).
        keep = keep & (topk_metrics > self.eps)

        topk_idxs = topk_idxs.masked_fill(~keep, 0)                         # park dropped -> idx 0
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones * keep[:, :, k:k + 1].to(torch.int8))
        # Guard against any duplicate index accumulation from parked slots.
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)


class LevelBalancedTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with a PER-PYRAMID-LEVEL top-k budget instead of one global pooled draw.

    =========================================================================
    MOTIVATION (diag_anchor_footprint.py, v6i val split)
    =========================================================================
    Stock TAL ranks score^a * iou^b over ALL anchors of a GT POOLED ACROSS
    LEVELS and keeps the global top-`topk`. The footprint diagnostic showed:

      * SEL BIAS ~= 1.0 at P3/P4  -> the METRIC is already level-neutral;
        the starvation is purely GEOMETRIC candidate supply, NOT a metric bug.
        (So re-weighting the metric — SATAL/LBA-on-metric — has little to fix.)
      * Selectivity: small GTs are FORCED to accept ~60% of a tiny pool
        (pool 12.9, take 7.73) while large GTs cream the best ~1.5% of 657.
      * BUT those forced small-object extras are GOOD (mean IoU 0.806, only
        1.3% marginal) -> cutting supply (SNA-TAL) removes useful signal, which
        is why SNA-TAL/SATAL are predicted to underperform.

    The diagnostic's own conclusion (Section 1/2 notes): *"Fix by allocating
    topk PER LEVEL, not globally."* That is exactly what this assigner does.

    =========================================================================
    MECHANISM
    =========================================================================
    Split the global budget `topk` into a per-level budget and run the top-k
    selection INDEPENDENTLY within each pyramid level, then union the picks.
    A GT then receives up to `k_level[l]` positives from EACH level it has
    candidates on, guaranteeing coarse levels (P4/P5) a share the global pooled
    draw denies them, and — more importantly for v6i — letting us bias the
    budget toward the fine levels (P3/P4) where small objects actually live and
    where supply exists (57 / 14 cand/GT).

    Budget modes (level_topk_mode):
      'proportional' (default): k_level[l] = round(topk * w[l] / sum(w)), where
          w defaults to the per-level candidate SHARE measured live this batch
          (so it self-adapts — no dataset constant). Reduces to ~stock when the
          metric is already level-neutral, but removes the winner-take-all
          pooling that let large objects dominate small GTs' slots.
      'fixed': k_level[l] taken directly from `level_topk` (a per-stride dict or
          list), e.g. {8: 6, 16: 3, 32: 2} to hand P3 six, P4 three, P5 two.
      'uniform': every level gets ceil(topk / n_levels) — a clean control that
          isolates "per-level vs pooled" from any budget shaping.

    total kept per GT is capped at `topk` (largest-metric-first) so this is a
    RE-ALLOCATION of the same budget, not a positive-count inflation — keeping
    it comparable to the stock baseline.

    Requires set_strides(stride_tensor) to be called each forward pass (the loss
    already calls set_imgsz for SA-TAL; set_strides is the analogous hook). If
    strides are missing it falls back to stock global top-k and warns once.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 level_topk_mode="proportional", level_topk=None,
                 min_level_k=1, eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.level_topk_mode = level_topk_mode
        # level_topk: dict{stride:int} or list aligned to sorted unique strides.
        self.level_topk = level_topk
        self.min_level_k = int(min_level_k)
        self._strides = None       # (A,) per-anchor stride, set each fwd pass
        self._warned = False
        self._printed = False

    def set_strides(self, stride_tensor):
        """Provide per-anchor stride (pixels). stride_tensor: (A,1) or (A,)."""
        self._strides = stride_tensor.reshape(-1)

    def _print_once(self, level_ks):
        if not self._printed:
            print(f"\n{'=' * 55}\n  LB-TAL Active (per-level top-k)\n{'=' * 55}")
            print(f"  mode:   {self.level_topk_mode}")
            print(f"  budget: {level_ks}  (sum capped at topk={self.topk})")
            print(f"{'=' * 55}\n")
            self._printed = True

    def _per_level_budget(self, uniq_strides, cand_share):
        """Return dict {stride: k_level} summing to <= topk."""
        n = len(uniq_strides)
        if self.level_topk_mode == "fixed" and self.level_topk is not None:
            if isinstance(self.level_topk, dict):
                ks = {s: int(self.level_topk.get(s, self.min_level_k)) for s in uniq_strides}
            else:  # list aligned to sorted strides
                ks = {s: int(self.level_topk[i]) if i < len(self.level_topk) else self.min_level_k
                      for i, s in enumerate(uniq_strides)}
        elif self.level_topk_mode == "uniform":
            per = max(self.min_level_k, -(-self.topk // n))  # ceil
            ks = {s: per for s in uniq_strides}
        else:  # 'proportional' — share of geometric candidate supply (live)
            w = cand_share if cand_share is not None else {s: 1.0 / n for s in uniq_strides}
            tot = sum(w.values()) or 1.0
            ks = {}
            for s in uniq_strides:
                ks[s] = max(self.min_level_k, int(round(self.topk * w[s] / tot)))
        return ks

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """Per-level top-k: select within each pyramid level, union, cap at topk."""
        # Fallback to stock behaviour if strides were not provided.
        if self._strides is None or self._strides.numel() != metrics.shape[-1]:
            if not self._warned:
                print("[LB-TAL][WARN] strides not set (or size mismatch) — "
                      "falling back to stock global top-k. Call set_strides() "
                      "each forward pass.")
                self._warned = True
            return super().select_topk_candidates(metrics, largest=largest, topk_mask=topk_mask)

        b, n, A = metrics.shape
        strides = self._strides
        uniq = sorted(torch.unique(strides).tolist())

        # Live per-level candidate share (for the 'proportional' budget): count
        # anchors with a positive metric per level, averaged over GTs.
        cand_share = None
        if self.level_topk_mode == "proportional":
            cand_share = {}
            pos = (metrics > self.eps)
            for s in uniq:
                lvl = (strides == s)
                cand_share[s] = float((pos & lvl.view(1, 1, -1)).sum().item())
            ssum = sum(cand_share.values()) or 1.0
            cand_share = {s: v / ssum for s, v in cand_share.items()}

        level_ks = self._per_level_budget(uniq, cand_share)
        self._print_once(level_ks)

        # Build the union mask by selecting top-k_level within each level.
        count = torch.zeros_like(metrics, dtype=torch.int8)
        for s in uniq:
            k = int(level_ks[s])
            if k <= 0:
                continue
            lvl = (strides == s).view(1, 1, -1)                    # (1,1,A)
            m_lvl = torch.where(lvl, metrics, torch.full_like(metrics, -1.0))
            k_eff = min(k, int(lvl.sum().item()))
            if k_eff <= 0:
                continue
            tk_metrics, tk_idxs = torch.topk(m_lvl, k_eff, dim=-1, largest=largest)  # (b,n,k_eff)
            valid = tk_metrics > self.eps                          # drop empty slots
            tk_idxs = tk_idxs.masked_fill(~valid, 0)
            ones = torch.ones_like(tk_idxs[:, :, :1], dtype=torch.int8)
            for j in range(k_eff):
                count.scatter_add_(-1, tk_idxs[:, :, j:j + 1],
                                   ones * valid[:, :, j:j + 1].to(torch.int8))

        count.masked_fill_(count > 1, 0)   # a level can't double-pick an anchor

        # Cap total positives per GT at topk (keep highest-metric first) so this
        # is a re-allocation, not an inflation of the positive count.
        total = count.sum(-1)                                      # (b, n)
        if int((total > self.topk).sum().item()) > 0:
            masked = torch.where(count > 0, metrics,
                                 torch.full_like(metrics, -1.0))
            _, keep_idx = torch.topk(masked, self.topk, dim=-1, largest=True)  # (b,n,topk)
            capped = torch.zeros_like(count)
            ones = torch.ones_like(keep_idx[:, :, :1], dtype=torch.int8)
            # only keep slots that were actually selected (metric>eps) AND in count
            for j in range(self.topk):
                idx = keep_idx[:, :, j:j + 1]
                sel = torch.gather(count, -1, idx) > 0
                capped.scatter_add_(-1, idx, ones * sel.to(torch.int8))
            capped.masked_fill_(capped > 1, 0)
            count = torch.where((total > self.topk).unsqueeze(-1), capped, count)

        return count.to(metrics.dtype)


# =============================================================================
# MAIN DETECTION LOSS CLASS
# =============================================================================


class v8DetectionLoss:
    """Custom YOLO detection loss with optional luggage-adapted mechanisms.

    Combines size-aware weighting (SWA), scale/AR/supply-aware assigners,
    inverse-frequency class weighting, and an optional NWD box term. All
    optional mechanisms default OFF; see the module docstring for the full
    list of hyperparameters.
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
        # v6i re-fit: 70 -> 36 (see BboxLoss.__init__ note). 36^2=1296px^2 ~
        # 0.6x v6i mean object area, matching the v5i small-object SCOPE.
        self.small_obj_px = getattr(h, 'small_obj_px', 36)
        self.small_obj_boost = getattr(h, 'small_obj_boost', 1.5)
        self.alpha_start = getattr(h, 'alpha_start', 0.9)
        self.alpha_end = getattr(h, 'alpha_end', 0.5)
        self.alpha_min = getattr(h, 'alpha_min', 0.3)
        self.alpha_max = getattr(h, 'alpha_max', 0.9)

        # =====================================================================
        # Section B: Center loss (per-anchor stride, per-dim size-normalized L1;
        # 'crowd' mode weights by GT neighbor overlap)
        # =====================================================================
        self.center_loss_weight_init = getattr(h, 'center_loss_weight_init', 0.0)
        self.center_loss_weight_min = getattr(h, 'center_loss_weight_min', 0.01)
        self.center_loss_decay_epochs = getattr(h, 'center_loss_decay_epochs', 35)
        self.center_loss_mode = getattr(h, 'center_loss_mode', 'small')  # 'small' | 'crowd'
        self.center_crowd_iou = getattr(h, 'center_crowd_iou', 0.1)      # neighbor IoU threshold

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
        # v6i re-fit + ablation-cleanliness (see satal.py __init__ note):
        #   satal_alpha_large 1.0 -> 0.5  : match stock tal_alpha so the LARGE
        #       branch (t=1) reduces SA-TAL to EXACTLY stock TAL — removes the
        #       alpha confound that made large-object deltas uninterpretable.
        #   satal_small/large_area 0.0025/0.0225 -> 0.005/0.030 : thresholds are
        #       object_area / LETTERBOXED-canvas-area (640^2=409600). v6i mean
        #       object 2145px^2 = 0.0052, so it sat above the old small knee;
        #       0.005 re-centres "small" on the v6i small population.
        self.use_satal = getattr(h, 'use_satal', False)
        self.satal_alpha_small = getattr(h, 'satal_alpha_small', 1.5)
        self.satal_beta_small = getattr(h, 'satal_beta_small', 3.0)
        self.satal_alpha_large = getattr(h, 'satal_alpha_large', self.tal_alpha)
        self.satal_beta_large = getattr(h, 'satal_beta_large', self.tal_beta)
        self.satal_small_area = getattr(h, 'satal_small_area', 0.005)
        self.satal_large_area = getattr(h, 'satal_large_area', 0.030)
        self.satal_topk_factor = getattr(h, 'satal_topk_factor', 1.5)

        # =====================================================================
        # Section F: Class Weighting (from CustomLoss2) - ALWAYS ON
        # =====================================================================
        # v6i dataset TRAIN instance counts (from LuggageDatasetSplitv6i analysis):
        #   backpack=11491, bag=9490, trolley=21557   (order: backpack, bag, trolley)
        # The previous hardcode [34901, 28628, 66946] was the OLD v5i corpus and
        # is WRONG for v6i — the ratios differ, so the derived weights were subtly
        # off on the new data. Two ways to get the right numbers:
        #   1. class_counts hyp override — pass an explicit list, OR
        #   2. auto-derive from the training dataloader labels at init (robust:
        #      no dataset swap can silently break the weights again).
        # Falls back to the v6i literals only if neither is available.
        _V6I_CLASS_COUNTS = [11491.0, 9490.0, 21557.0]
        counts = getattr(h, 'class_weight_counts', None)
        if counts is None:
            counts = self._derive_class_counts(model, self.nc)
        if counts is None or len(counts) != self.nc:
            counts = _V6I_CLASS_COUNTS if self.nc == len(_V6I_CLASS_COUNTS) else [1.0] * self.nc
        self._class_counts_used = list(counts)
        class_counts = torch.tensor(counts, dtype=torch.float, device=device)
        inv_freq = 1.0 / class_counts.clamp(min=1.0)
        inv_freq = inv_freq / inv_freq.mean()

        # class_weight_mode selects dampening strategy
        #   'sqrt'  : gentle: [1.08, 1.19, 0.78] (bag 1.53x trolley)
        #   'linear': No dampening — aggressive: [0.92, 1.41, 0.67] (bag 2.10x trolley)
        self.class_weight_mode = getattr(h, 'class_weight_mode', 'sqrt')
        if self.class_weight_mode == 'linear':
            self.class_weights = inv_freq.clone()  # no sqrt dampening
        else:
            self.class_weights = torch.sqrt(inv_freq)
        self.class_weights = self.class_weights / self.class_weights.mean()

        # Toggle class weighting (default ON for legacy behavior; OFF = clean baseline)
        self.use_class_weighting = getattr(h, 'use_class_weighting', True)

        # Section G: classification loss mode ('bce' | 'qfl')
        self.cls_mode = getattr(h, 'cls_mode', 'bce')
        self.qfl_beta = getattr(h, 'qfl_beta', 2.0)

        # =====================================================================
        # Section J: Class-confusion repulsion
        # Penalize a predicted box that overlaps a DIFFERENT-class GT box.
        # Targets bag false positives (bag sitting on backpack/trolley).
        # =====================================================================
        self.use_repulsion = getattr(h, 'use_repulsion', False)
        self.repulsion_weight = getattr(h, 'repulsion_weight', 0.3)
        # sqrt mode: backpack≈1.08, bag≈1.19, trolley≈0.78
        # linear mode: backpack≈0.92, bag≈1.41, trolley≈0.67

        # =====================================================================
        # Section K: size-aware CLS weighting
        # Recall gap is a RANKING problem (AR50_small 0.96 vs R50_small 0.71):
        # small objects are localized but scored too low. Boost the fg cls loss
        # for small objects so their positives are pushed to higher confidence.
        # QFL/VFL alternatives failed; this only reweights the existing BCE.
        #
        # Smallness criterion uses the SAME area-vs-per-anchor-stride test as
        # Section A (area_feat < (px/stride)^2, equivalently area_px <
        # small_obj_px^2 — stride cancels). A pixel max-side test would be
        # effectively a HEIGHT test on tall/narrow boxes and would miss thin
        # small boxes (e.g. 26x75), i.e. exactly the population K targets.
        # =====================================================================
        self.use_cls_swa = getattr(h, 'use_cls_swa', False)
        self.cls_swa_boost = getattr(h, 'cls_swa_boost', 1.75)

        # =====================================================================
        # Section L: bag asymmetric penalty
        # Bag = 74% precision with CONFIDENT cross-class FPs. Upweight ONLY the
        # negative BCE term of the bag logit at fg anchors assigned to
        # backpack/trolley (their bag target is 0 there) — pushes the bag
        # score down on other-class objects without touching bag positives.
        # =====================================================================
        self.use_bag_penalty = getattr(h, 'use_bag_penalty', False)
        self.bag_penalty_weight = getattr(h, 'bag_penalty_weight', 2.0)
        self.bag_class_id = getattr(h, 'bag_class_id', 1)  # dataset order: backpack, bag, trolley

        # =====================================================================
        # Section M: AR-aware TAL assigner — see class docstring
        # =====================================================================
        # v6i re-fit: median AR fell 2.69 -> 1.54, so the old ar_thresh=2.0
        # sat ABOVE the median and the mechanism barely fired (few GTs exceeded
        # it). 1.7 places the knee just above the v6i median so genuinely
        # tall/narrow boxes trigger the beta relaxation. ar_scale 2.0->1.2
        # keeps the ramp meaningful over v6i's compressed AR range.
        self.use_artal = getattr(h, 'use_artal', False)
        self.artal_ar_thresh = getattr(h, 'artal_ar_thresh', 1.7)
        self.artal_ar_scale = getattr(h, 'artal_ar_scale', 1.2)
        self.artal_beta_relax = getattr(h, 'artal_beta_relax', 2.0)

        # =====================================================================
        # Section N: Supply-normalized TAL assigner (SNA-TAL) — see class
        # docstring. Per-GT budget k_eff = clamp(round(rho*pool), k_min, topk).
        # Targets the diagnosed supply-limited dilution of small objects.
        # snatal_kmin default is 2.
        # =====================================================================
        self.use_snatal = getattr(h, 'use_snatal', False)
        self.snatal_rho = getattr(h, 'snatal_rho', 0.25)
        self.snatal_kmin = getattr(h, 'snatal_kmin', 2)

        # =====================================================================
        # Section O: Level-Balanced TAL assigner (LB-TAL) — per-level top-k.
        # =====================================================================
        # THE mechanism the anchor-footprint diagnostic points to: it found the
        # alignment metric is already level-neutral (SEL BIAS ~1 at P3/P4) and
        # the small-object bottleneck is GEOMETRIC candidate supply under a
        # GLOBAL pooled top-k. LB-TAL splits the topk budget PER PYRAMID LEVEL
        # and selects within each level, so coarse levels get a guaranteed share
        # and the budget can be biased toward the fine levels where small objects
        # live. Unlike SNA-TAL it does NOT cut supply (the forced small-object
        # extras were measured to be good, IoU 0.806), it RE-ALLOCATES it.
        #
        # Modes:
        #   'proportional' — k_level[l] ~ live per-level candidate share
        #                    (self-adapting, no dataset constant). Default.
        #   'fixed'        — level_topk = {8:k3, 16:k4, 32:k5} or a list aligned
        #                    to sorted strides, e.g. {8:6,16:3,32:2}.
        #   'uniform'      — equal budget per level (clean per-level-vs-pooled
        #                    control).
        # Total kept per GT is still capped at tal_topk (re-allocation, not
        # inflation). Requires set_strides() each fwd pass (wired below).
        self.use_lbtal = getattr(h, 'use_lbtal', False)
        self.lbtal_mode = getattr(h, 'lbtal_mode', 'proportional')
        self.lbtal_level_topk = getattr(h, 'lbtal_level_topk', None)
        self.lbtal_min_level_k = getattr(h, 'lbtal_min_level_k', 1)

        # M / N / O are mutually exclusive assigner variants — at most one on.
        # Each is a different single override of the stock assigner; stacking
        # them is undefined (they all replace select_topk_candidates / the
        # metric). Fail loudly rather than silently letting the if/elif win.
        _assigner_flags = [
            ("use_artal", self.use_artal),
            ("use_snatal", self.use_snatal),
            ("use_lbtal", self.use_lbtal),
            ("use_satal", getattr(h, 'use_satal', False)),
        ]
        _on = [name for name, val in _assigner_flags if val]
        if len(_on) > 1:
            raise ValueError(
                f"At most one custom assigner may be enabled per run; got {_on}. "
                "SATAL / AR-TAL / SNA-TAL / LB-TAL are mutually exclusive "
                "alternatives (Sections E / M / N / O)."
            )

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
        self.use_nwd = getattr(h, 'use_nwd', False)  # honest default; drives config print
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
        elif self.use_artal:
            # Section M: AR-aware TAL — stock behavior for AR <= ar_thresh
            self.assigner = ARAwareTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                ar_thresh=self.artal_ar_thresh,
                ar_scale=self.artal_ar_scale,
                beta_relax=self.artal_beta_relax,
            )
        elif self.use_snatal:
            # Section N: supply-normalized TAL — per-GT k_eff by pool size
            self.assigner = SupplyNormalizedTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                rho=self.snatal_rho,
                k_min=self.snatal_kmin,
            )
        elif self.use_lbtal:
            # Section O: level-balanced TAL — per-pyramid-level top-k budget
            self.assigner = LevelBalancedTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                level_topk_mode=self.lbtal_mode,
                level_topk=self.lbtal_level_topk,
                min_level_k=self.lbtal_min_level_k,
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
            print(f"  [A2] area_weight_mode: {self.bbox_loss.area_weight_mode} (norm: {self.bbox_loss.area_weight_norm})")
            if self.bbox_loss.class_boosts is not None:
                print(f"  [A2] class boosts (bp/bg/tr): {self.bbox_loss.class_boosts.cpu().numpy().round(3)}")
            if self.bbox_loss.dfl_entropy_weight > 0:
                print(f"  [O] dfl_entropy_weight: {self.bbox_loss.dfl_entropy_weight} (global edge sharpening)")
            print(f"  [B] center_loss_init: {self.center_loss_weight_init}" +
                  (f" (mode={self.center_loss_mode})" if self.center_loss_weight_init > 0 else ""))
            print(f"  [C] iou_clip:        {self.iou_clip_start} → {self.iou_clip_end} (DEPRECATED — keep off)")
            print(f"  [D] tal_topk:        {self.tal_topk}")
            print(f"  [D] tal_alpha:       {self.tal_alpha}")
            print(f"  [D] tal_beta:        {self.tal_beta}")
            print(f"  [E] use_satal:       {self.use_satal}")
            print(f"  [I] box_loss_type:   {self.bbox_loss.box_loss_type}")
            print(f"  [J] repulsion:       {self.use_repulsion}" + (f" (w={self.repulsion_weight})" if self.use_repulsion else ""))
            print(f"  [K] cls_swa:         {self.use_cls_swa}" + (f" (boost={self.cls_swa_boost}, area<{self.small_obj_px}px² criterion)" if self.use_cls_swa else ""))
            print(f"  [L] bag_penalty:     {self.use_bag_penalty}" + (f" (w={self.bag_penalty_weight}, cls={self.bag_class_id})" if self.use_bag_penalty else ""))
            print(f"  [M] artal:           {self.use_artal}" + (f" (thresh={self.artal_ar_thresh}, scale={self.artal_ar_scale}, relax={self.artal_beta_relax})" if self.use_artal else ""))
            print(f"  [N] snatal:          {self.use_snatal}" + (f" (rho={self.snatal_rho}, k_min={self.snatal_kmin}, geometric pool)" if self.use_snatal else ""))
            print(f"  [O2] lbtal:          {self.use_lbtal}" + (f" (mode={self.lbtal_mode}, level_topk={self.lbtal_level_topk}, min_k={self.lbtal_min_level_k}, per-level top-k)" if self.use_lbtal else ""))
            if self.use_satal:
                print(f"      satal_alpha_small: {self.satal_alpha_small}")
                print(f"      satal_beta_small:  {self.satal_beta_small}")
                print(f"      satal_topk_factor: {self.satal_topk_factor}")
            print(f"  [F] Class Weighting: {'ON' if self.use_class_weighting else 'OFF'} (mode: {self.class_weight_mode})")
            print(f"      counts used (bp/bg/tr): {[int(c) for c in getattr(self, '_class_counts_used', [])]}")
            print(f"      weights (bp/bg/tr): {self.class_weights.cpu().numpy().round(3)}")
            print(f"  [G] Cls Loss: {self.cls_mode.upper()}" + (f" (beta={self.qfl_beta})" if self.cls_mode == 'qfl' else "") + (" (class weighting applied)" if self.use_class_weighting else ""))
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

    # =========================================================================
    # Criterion-coverage diagnostic
    # =========================================================================

    @torch.no_grad()
    def _log_criterion_coverage(self, gt_bboxes, mask_gt, anchor_points, stride_tensor):
        """Log the fraction of GTs captured by each mechanism's smallness/
        thinness criterion, and their overlaps (once every 10 epochs).

        Four mechanisms activate on different definitions of 'small/thin':
          A/K : pixel area < small_obj_px^2 (per-anchor stride cancels)
          M   : aspect ratio > artal_ar_thresh
          N   : geometric candidate pool (anchor centers inside GT) < tal_topk
        Divergent coverage between these is exactly what makes ablations hard
        to interpret — this log surfaces it for free. Never breaks training.
        """
        if self.epoch % 10 != 0 or self.epoch == getattr(self, '_cov_logged_epoch', -1):
            return
        self._cov_logged_epoch = self.epoch
        try:
            gm = mask_gt.squeeze(-1) > 0
            n_gt = int(gm.sum())
            if n_gt == 0:
                return
            boxes = gt_bboxes[gm]  # (G, 4) pixel xyxy
            w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
            h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

            # A/K criterion: area_feat < (px/stride)^2  <=>  area_px < px^2
            small = (w * h) < float(self.small_obj_px) ** 2
            # M criterion
            ar = torch.maximum(w / h, h / w)
            high_ar = ar > float(self.artal_ar_thresh)
            # N criterion: geometric pool (anchor centers inside the GT)
            apx = anchor_points * stride_tensor  # (A, 2) pixel centers
            x, y = apx[:, 0], apx[:, 1]
            inside = ((x[None, :] > boxes[:, 0:1]) & (x[None, :] < boxes[:, 2:3]) &
                      (y[None, :] > boxes[:, 1:2]) & (y[None, :] < boxes[:, 3:4]))
            pool = inside.sum(-1)
            supply_poor = pool < self.tal_topk

            f = lambda m: 100.0 * m.float().mean().item()
            print(f"[Coverage] Epoch {self.epoch} (batch sample, {n_gt} GTs): "
                  f"small(A/K)={f(small):.1f}%  high-AR(M)={f(high_ar):.1f}%  "
                  f"supply-poor(N)={f(supply_poor):.1f}%  |  "
                  f"small∩AR={f(small & high_ar):.1f}%  small∩poor={f(small & supply_poor):.1f}%  "
                  f"median pool={pool.float().median().item():.0f}")
        except Exception as e:  # diagnostics must never kill a run
            print(f"[Coverage] logging skipped ({e})")

    def _compute_center_loss(self, pred_bboxes, target_bboxes, fg_mask, stride_tensor,
                             gt_bboxes=None, mask_gt=None, target_gt_idx=None):
        """Auxiliary center loss (Section B).

        Design notes:
          - per-anchor stride threshold (a global stride.min() would make
            'small' mean <(px/8)^2 in feature units on ALL scales — a far
            broader net than SWA's)
          - per-dimension L1 normalized by GT w/h (raw feature-coord L1 is
            scale-dependent, dominated by larger 'small' boxes; and with
            2.6:1-tall boxes raw center-y error dominates x)

        Modes:
          'small' : small objects only (fixed version of the v2 intent)
          'crowd' : ALL fg, weighted by GT neighbor overlap — center accuracy
                    is what separates adjacent objects at NMS (~5 objs/img)

        Crowd weights are computed ONCE PER GT (GT-vs-GT pairwise IoU) and
        gathered per anchor via target_gt_idx, instead of recomputing identical
        neighbor counts for every anchor of the same GT. Equivalent to a
        per-anchor recompute (self-match ~1 is subtracted out).
        """
        if self.center_loss_weight_init <= 0:
            return torch.tensor(0.0, device=self.device)

        if not fg_mask.any():
            return torch.tensor(0.0, device=self.device)

        fg_b, fg_a = torch.nonzero(fg_mask, as_tuple=True)
        if fg_b.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        pred_fg = pred_bboxes[fg_b, fg_a]      # feature coords
        target_fg = target_bboxes[fg_b, fg_a]  # feature coords

        pred_centers = (pred_fg[:, :2] + pred_fg[:, 2:]) / 2
        target_centers = (target_fg[:, :2] + target_fg[:, 2:]) / 2

        tw = (target_fg[:, 2] - target_fg[:, 0]).clamp(min=1e-6)
        th = (target_fg[:, 3] - target_fg[:, 1]).clamp(min=1e-6)

        # Per-dimension, size-normalized center error (scale/AR-invariant)
        err = ((pred_centers[:, 0] - target_centers[:, 0]).abs() / tw +
               (pred_centers[:, 1] - target_centers[:, 1]).abs() / th) / 2.0  # (M,)

        # Per-anchor stride (same fix as BboxLoss._compute_weights)
        s_col = stride_tensor.reshape(-1)
        s_full = s_col.unsqueeze(0).expand(fg_mask.shape[0], -1)
        stride_fg = s_full[fg_mask].clamp_min(1.0)  # (M,)

        if (self.center_loss_mode == 'crowd' and gt_bboxes is not None
                and mask_gt is not None and target_gt_idx is not None):
            # per-GT neighbor count, gathered per anchor.
            # neighbors_g = #GTs whose IoU with GT g exceeds the threshold
            # (self excluded); crowd weight = 1 + neighbors of assigned GT.
            crowd_w = torch.ones_like(err)
            n_max = mask_gt.shape[1]
            for i in range(fg_mask.shape[0]):
                sel = fg_b == i
                if not sel.any():
                    continue
                gm = mask_gt[i].squeeze(-1) > 0
                if gm.sum() < 2:
                    continue  # 0 or 1 GT — no neighbors possible
                boxes_i = gt_bboxes[i][gm]                       # (g, 4) pixel; IoU is scale-invariant
                iou = self._pairwise_iou(boxes_i, boxes_i)       # (g, g), diag = 1
                neighbors = (iou > self.center_crowd_iou).sum(dim=1).float() - 1.0  # exclude self
                gt_crowd = torch.ones(n_max, device=err.device, dtype=err.dtype)
                gt_crowd[gm] = 1.0 + neighbors.clamp(min=0.0)
                crowd_w[sel] = gt_crowd[target_gt_idx[i][fg_mask[i]]]
            center_l1_loss = (err * crowd_w).sum() / crowd_w.sum().clamp(min=1.0)
        else:
            # 'small' mode: fixed per-anchor threshold, small objects only
            target_areas = tw * th
            small_threshold = (self.small_obj_px / stride_fg) ** 2  # per-anchor
            small_obj_mask = target_areas < small_threshold
            if not small_obj_mask.any():
                return torch.tensor(0.0, device=self.device)
            center_l1_loss = err[small_obj_mask].mean()

        progress = min(self.epoch / max(self.center_loss_decay_epochs, 1), 1.0)
        weight = self.center_loss_weight_init * (1 - progress)
        weight = max(self.center_loss_weight_min, weight)

        return center_l1_loss * weight

    @staticmethod
    def _pairwise_iou(a, b, eps=1e-7):
        """IoU between boxes a (n,4) and b (m,4), xyxy -> (n,m)."""
        area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area_a[:, None] + area_b[None, :] - inter + eps
        return inter / union

    def _compute_repulsion(self, pred_bboxes, stride_tensor, fg_mask, target_scores,
                           gt_bboxes, gt_labels, mask_gt):
        """Mean IoU of each fg predicted box with its highest-overlap
        DIFFERENT-class GT. Minimizing it pushes predictions off other-class
        objects, reducing cross-class false positives (the bag problem)."""
        if not fg_mask.any():
            return pred_bboxes.new_tensor(0.0)
        pred_px = pred_bboxes * stride_tensor            # feature -> pixel coords
        cls_anchor = target_scores.argmax(dim=-1)        # (b, A) class per anchor
        total = pred_bboxes.new_tensor(0.0)
        count = 0
        for i in range(pred_bboxes.shape[0]):
            fg = fg_mask[i]
            if fg.sum() == 0:
                continue
            gm = mask_gt[i].squeeze(-1) > 0
            if gm.sum() == 0:
                continue
            P = pred_px[i][fg]                           # (n,4)
            cP = cls_anchor[i][fg]                       # (n,)
            G = gt_bboxes[i][gm]                         # (m,4)
            cG = gt_labels[i][gm].squeeze(-1).long()     # (m,)
            iou = self._pairwise_iou(P, G)               # (n,m)
            diff = (cP[:, None] != cG[None, :]).to(iou.dtype)
            max_iou = (iou * diff).max(dim=1).values     # (n,)
            total = total + max_iou.sum()
            count += P.shape[0]
        if count == 0:
            return pred_bboxes.new_tensor(0.0)
        return total / count

    def _sync_bbox_loss_state(self):
        """Synchronize epoch information with bbox_loss module."""
        self.bbox_loss.epoch = self.epoch
        self.bbox_loss.total_epochs = self.total_epochs

    @staticmethod
    def _derive_class_counts(model, nc):
        """Best-effort per-class instance counts from the model's training data.

        Reads the YOLO data.yaml attached to the model (model.args['data'] or
        model.overrides), scans the train label .txt files once, and tallies the
        class id in column 0 of each row. Returns a list[float] of length nc, or
        None if the labels cannot be located (in which case the caller falls
        back to the v6i literals). Never raises — class weighting must not be
        able to kill a run.
        """
        import os
        import glob
        try:
            import yaml
        except Exception:
            yaml = None

        def _find_data_yaml():
            for src in (getattr(model, 'args', None), getattr(model, 'overrides', None)):
                if src is None:
                    continue
                d = src if isinstance(src, dict) else getattr(src, '__dict__', {})
                p = d.get('data') if isinstance(d, dict) else getattr(src, 'data', None)
                if p and str(p).endswith(('.yaml', '.yml')) and os.path.isfile(str(p)):
                    return str(p)
            return None

        try:
            data_yaml = _find_data_yaml()
            if data_yaml is None or yaml is None:
                return None
            with open(data_yaml, 'r') as f:
                cfg = yaml.safe_load(f)
            base = cfg.get('path', os.path.dirname(data_yaml))
            train = cfg.get('train', 'images/train')
            train_path = train if os.path.isabs(train) else os.path.join(base, train)
            # images/... -> labels/...  (standard YOLO layout)
            label_dir = train_path.replace(os.sep + 'images', os.sep + 'labels')
            label_dir = label_dir.replace('/images', '/labels')
            if os.path.isdir(train_path) and not os.path.isdir(label_dir):
                label_dir = os.path.join(os.path.dirname(train_path.rstrip('/\\')), 'labels')
            txts = glob.glob(os.path.join(label_dir, '**', '*.txt'), recursive=True)
            if not txts:
                return None
            counts = [0.0] * nc
            for t in txts:
                try:
                    with open(t, 'r') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            cid = int(float(line.split()[0]))
                            if 0 <= cid < nc:
                                counts[cid] += 1.0
                except Exception:
                    continue
            return counts if sum(counts) > 0 else None
        except Exception:
            return None

    # =========================================================================
    # CLASSIFICATION LOSS (BCE + Class Weighting, no VFL)
    # =========================================================================

    def _compute_cls_loss_weighted(self, pred_scores, target_scores, fg_mask,
                                   target_labels_for_fg, target_scores_sum,
                                   target_bboxes_px=None, stride_tensor=None):
        """
        Compute class-weighted BCE classification loss.

        Uses standard BCE with per-anchor class weighting based on inverse
        class frequency. No Varifocal Loss - just simple weighted BCE.

        Class weights: backpack≈1.08, bag≈1.19, trolley≈0.78

        Optional additions (both default OFF):
          Section K: fg anchors of SMALL objects get cls loss x cls_swa_boost.
                     'small' uses the SAME area-vs-per-anchor-stride criterion
                     as Section A (SWA): area_feat < (small_obj_px/stride)^2.
          Section L: the bag logit's NEGATIVE bce term at fg anchors assigned
                     backpack/trolley gets x bag_penalty_weight.
        """
        dtype = pred_scores.dtype
        bs, num_anchors, nc = pred_scores.shape

        # Base BCE loss
        bce = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)

        # Optional Quality Focal Loss modulation (Section G, cls_mode='qfl')
        if self.cls_mode == 'qfl':
            with torch.no_grad():
                qfl_scale = (target_scores.to(dtype) - pred_scores.sigmoid()).abs().pow(self.qfl_beta)
            bce = bce * qfl_scale

        # Build per-anchor weight tensor for class weighting
        weight = torch.ones(bs, num_anchors, 1, device=self.device, dtype=dtype)

        if self.use_class_weighting and fg_mask.any() and target_labels_for_fg.numel() > 0:
            # Apply class weights to foreground anchors (toggleable -- OFF = clean baseline)
            fg_class_weights = self.class_weights.to(dtype)[target_labels_for_fg]
            weight[fg_mask] = fg_class_weights.unsqueeze(-1)

        # ── Section K: small-object cls boost,
        #    area-vs-per-anchor-stride criterion (same as Section A) ──
        if (self.use_cls_swa and fg_mask.any() and target_bboxes_px is not None
                and stride_tensor is not None):
            s_full = stride_tensor.reshape(-1).unsqueeze(0).expand(bs, -1)  # (bs, A)
            stride_fg = s_full[fg_mask].clamp_min(1.0)                       # (M,)
            tb = target_bboxes_px[fg_mask]                                   # (M, 4) pixel coords
            area_feat = (((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]))
                         .clamp(min=1e-6) / stride_fg.pow(2))                # feature units
            small_thr = (float(self.small_obj_px) / stride_fg) ** 2          # per-anchor
            boost = torch.where(area_feat < small_thr,
                                area_feat.new_tensor(float(self.cls_swa_boost)),
                                area_feat.new_tensor(1.0)).to(dtype)
            weight[fg_mask] = weight[fg_mask] * boost.unsqueeze(-1)

        # ── Section L: bag asymmetric penalty (negative term only) ──
        if self.use_bag_penalty and fg_mask.any() and target_labels_for_fg.numel() > 0:
            # expand per-anchor weight to per-class so one column can differ
            weight = weight.expand(bs, num_anchors, nc).clone()
            fg_w = weight[fg_mask]  # (M, nc) copy
            other_cls = target_labels_for_fg != self.bag_class_id
            # bag target is 0 at these anchors -> this scales only the
            # negative (suppress-bag) bce term; bag positives untouched
            fg_w[other_cls, self.bag_class_id] = fg_w[other_cls, self.bag_class_id] * float(self.bag_penalty_weight)
            weight[fg_mask] = fg_w

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

        # criterion-coverage diagnostic (every 10 epochs)
        self._log_criterion_coverage(gt_bboxes, mask_gt, anchor_points, stride_tensor)

        # Decode predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Set image size for SA-TAL
        if hasattr(self.assigner, 'set_imgsz'):
            self.assigner.set_imgsz(imgsz)

        # Set per-anchor strides for LB-TAL (per-level top-k needs to know which
        # pyramid level each anchor belongs to). stride_tensor is (A, 1) pixels.
        if hasattr(self.assigner, 'set_strides'):
            self.assigner.set_strides(stride_tensor)

        # Task Aligned Assignment
        # capture target_gt_idx for the per-GT crowd weighting in the center loss.
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
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

        # ── Classification loss (class-weighted; Sections K + L) ──
        # NOTE: target_bboxes is still in PIXEL coords here (divided by stride
        # only inside the box-loss branch below) — Section K relies on that.
        # stride_tensor is passed for K's per-anchor smallness criterion.
        loss[1] = self._compute_cls_loss_weighted(
            pred_scores, target_scores, fg_mask, target_labels_for_fg, target_scores_sum,
            target_bboxes_px=target_bboxes, stride_tensor=stride_tensor
        )

        # ── Bounding box losses (SWA from loss_satal_swa.py) ──
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            self._sync_bbox_loss_state()

            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor,
                fg_labels=target_labels_for_fg  # Section A2 per-class boost
            )

            # Add auxiliary center loss (Section B; gt boxes + target_gt_idx
            # for the per-GT 'crowd' mode)
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor,
                gt_bboxes=gt_bboxes, mask_gt=mask_gt, target_gt_idx=target_gt_idx
            )
            loss[0] = loss[0] + center_loss

            # Class-confusion repulsion (Section J) — uses raw pixel-coord GTs
            if self.use_repulsion:
                rep_term = self._compute_repulsion(
                    pred_bboxes, stride_tensor, fg_mask, target_scores,
                    gt_bboxes, gt_labels, mask_gt
                )
                loss[0] = loss[0] + self.repulsion_weight * rep_term

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
    """Criterion class for computing training losses for segmentation.

    Complete stock-style implementation (mask interpolation + per-image
    prototype mask loss). Dormant for detection-only training.
    """

    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
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
            # bbox losses on stride-normalized copies (target_bboxes must stay
            # in pixel coords for the segmentation loss below)
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes / stride_tensor, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask
            )
            # segmentation loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample GT masks to proto size
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                proto, pred_masks, imgsz, self.overlap
            )
        else:
            # keep the graph connected so DDP doesn't complain about unused params
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box  # seg gain follows box gain (stock convention)
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        """Mask BCE for a single image, cropped to boxes and area-normalized."""
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n_fg, mask_h, mask_w)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz, overlap):
        """Prototype-mask loss over the batch (stock-style)."""
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize target boxes to 0-1, then scale to proto-mask space
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)  # (b, A)
        mxyxy = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=proto.device
        )

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i,
                    mxyxy_i[fg_mask_i], marea_i[fg_mask_i].clamp(min=1e-7)
                )
            else:
                # keep the graph connected for unused-parameter checks
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()


class DetectAuxLoss:
    """Train-only auxiliary-head deep-supervision loss.
    DetectAux adds a parallel detection head over the same feature
    maps as the main head; it is supervised during training and DROPPED at
    inference (zero deploy cost). The total loss is the main detection loss plus
    a down-weighted auxiliary detection loss, giving the shared neck features an
    extra gradient signal. Both heads share strides, so the same v8DetectionLoss
    is reused for each. Mirrors E2EDetectLoss's two-loss structure.
    """

    def __init__(self, model, aux_weight=0.25):
        self.det = v8DetectionLoss(model, tal_topk=10)
        self.aux_weight = getattr(model.model[-1], "aux_weight", aux_weight)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        if not isinstance(preds, dict):  # val/eval path: only main head present
            return self.det(preds, batch)
        loss_main = self.det(preds["main"], batch)
        loss_aux = self.det(preds["aux"], batch)
        return loss_main[0] + self.aux_weight * loss_aux[0], loss_main[1]


class DetectObjLoss(v8DetectionLoss):
    """v8 detection loss + an objectness (foreground/background) BCE term.
    Supervises DetectObj's per-anchor objectness logit against the
    TAL foreground mask (1 = assigned foreground, 0 = background), so the head
    learns to suppress background-like anchors and improve precision/ranking on
    the 'other' class. Mirrors v8DetectionLoss.__call__ with the extra term.
    """

    def __init__(self, model, obj_weight=1.0):
        super().__init__(model)
        self.obj_weight = obj_weight
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        if not isinstance(preds, dict):  # val/eval: only main head present
            return super().__call__(preds, batch)
        feats, obj_feats = preds["main"], preds["obj"]
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_obj = torch.cat(
            [oi.view(feats[0].shape[0], 1, -1) for oi in obj_feats], 2
        ).permute(0, 2, 1).contiguous()  # (b, A, 1)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # cls
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                stride_tensor
            )
        obj_target = fg_mask.unsqueeze(-1).to(dtype)  # (b, A, 1)
        loss[3] = self.bce_obj(pred_obj, obj_target).mean()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.obj_weight
        return loss.sum() * batch_size, loss[:3].detach()  # log box/cls/dfl


class E2EDetectLoss:
    """End-to-end detection loss."""

    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return (
            loss_one2many[0] + loss_one2one[0],
            loss_one2many[1] + loss_one2one[1]
        )
