# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# SATAL-SWA-Plus-NWD v3: luggage-dataset-adapted sections (Round 8)
# =============================================================================
#
# Base: loss_satal_swa_plus_v2.py (Rounds 4-7). ALL v3 additions default to
# legacy/OFF, so with no new hyp keys this file reproduces v2 EXACTLY —
# verify with the r8_anchor run (must land within ±0.35 of r2_swa_const06).
#
# v3 changes (motivated by the dataset analysis + Rounds 1-7 results):
#   [A2] area_weight_mode 'inv'|'sqrt'|'log' — reshape the 1/area weight.
#        Dataset areas span p10=0.0013..p90=0.036 (28x): raw 1/area puts all
#        emphasis on the tiniest boxes; sqrt/log spread it over small+medium.
#   [A2] per-class small-object boost (small_obj_boost_backpack/bag/trolley) —
#        bag is smallest (42.6% <48px) AND the precision bottleneck.
#   [B2] center loss FIXED: per-anchor stride threshold (was global min) and
#        per-dimension size-normalized L1 (was raw feature-coord L1 — prime
#        suspect in the r7_stack collapse). New 'crowd' mode weights by GT
#        neighbor overlap (~5 objs/img; adjacent-object NMS separation).
#   [C]  DEPRECATED — inert in Rounds 1-3; keep off (use_loss_clip=False).
#   [K]  cls SWA: small-object boost on fg classification loss. Recall gap is
#        a ranking problem (AR50_small 0.96 vs R50_small 0.71): small objects
#        are found but scored low. QFL/VFL failed; this only reweights BCE.
#   [L]  bag asymmetric penalty: upweight the NEGATIVE bce term of the bag
#        logit at fg anchors assigned backpack/trolley — punishes confident
#        wrong-class bag scores (bag P=0.74 with high-confidence FPs) without
#        touching positives (linear class weighting failed) or box geometry
#        (repulsion was inert).
#   [M]  AR-aware TAL: per-GT beta relaxed for high-aspect-ratio boxes. Test
#        boxes are tall/narrow (median AR 2.58, width ~26px @640 = 1-3 anchor
#        columns at stride 16/32) → assignment starvation that area-based
#        SATAL misses. Non-uniform by construction (uniform loosening hurt).
#
# New hyp keys (whitelist them like alpha_start / satal_*):
#   area_weight_mode, small_obj_boost_backpack, small_obj_boost_bag,
#   small_obj_boost_trolley, center_loss_mode, center_crowd_iou,
#   use_cls_swa, cls_swa_boost, use_bag_penalty, bag_penalty_weight,
#   bag_class_id, use_artal, artal_ar_thresh, artal_ar_scale, artal_beta_relax
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
# ALTERNATIVE BBOX REGRESSION LOSSES (Round 4 additions)
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

        # Section A2 (v3): area-weight shape + per-class boost. Defaults = legacy.
        self.area_weight_mode = 'inv'   # 'inv' (legacy 1/area) | 'sqrt' | 'log'
        self.class_boosts = None        # tensor [backpack, bag, trolley] or None (scalar boost)

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

        # Section I: alternative regression losses (Round 4)
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

        # Optional per-sample loss clip (Rounds 1-3 showed it inert; kept toggleable)
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

        # Section A2 (v3): area-weight shape + per-class small-object boost
        self.area_weight_mode = getattr(hyp, 'area_weight_mode', self.area_weight_mode)
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

        # Section A2 (v3): area-weight shape. 'inv' is the exact legacy weight.
        # Dataset areas span 28x (p10..p90); raw 1/area concentrates all weight
        # on the tiniest boxes — sqrt/log spread emphasis over small+medium.
        inv_area = 1.0 / target_areas[fg_mask]
        if self.area_weight_mode == 'sqrt':
            area_weight = inv_area.sqrt().unsqueeze(-1)
        elif self.area_weight_mode == 'log':
            area_weight = torch.log1p(inv_area).unsqueeze(-1)
        else:  # 'inv' — legacy
            area_weight = inv_area.unsqueeze(-1)

        # Normalize area weights
        if area_weight.numel() > 0:
            area_weight = area_weight / (area_weight.max() + 1e-8)

        # Apply small-object boost with PER-ANCHOR stride (fix vs. global stride.min())
        if stride is not None and area_weight.numel() > 0:
            s_col = stride.reshape(-1)  # (total_anchors,)
            bs = fg_mask.shape[0]
            s_full = s_col.unsqueeze(0).expand(bs, -1)  # (bs, total_anchors)
            stride_fg = s_full[fg_mask].clamp_min(1.0)  # (M,)
            fg_areas = target_areas[fg_mask]  # (M,)
            small_threshold = (self.small_obj_px / stride_fg) ** 2  # per-anchor (M,)

            # Section A2 (v3): per-class boost — bag (smallest + precision
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

        # Clip PER-SAMPLE (toggleable; Rounds 1-3 showed it inert)
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
# Section M (v3): AR-AWARE TASK-ALIGNED ASSIGNER (Round 8)
# =============================================================================
# Motivation (dataset analysis): test boxes are tall/narrow (median AR 2.58,
# median width ~26px at 640 input = 1-3 anchor columns at stride 16/32). TAL
# candidates need their anchor center INSIDE the GT, so narrow boxes get
# structurally fewer candidates regardless of area — SATAL's area split misses
# this. Fix: relax beta per-GT as aspect ratio rises, so high-AR boxes are less
# punished for the low IoU their thin geometry forces. Boxes with AR <=
# ar_thresh see EXACTLY stock behavior (beta_eff == beta), so this is
# non-uniform by construction (uniform loosening hurt in Rounds 1-2).
#
#   ar        = max(w/h, h/w)                       per GT
#   t         = clamp((ar - ar_thresh)/ar_scale, 0, 1)
#   beta_eff  = clamp(beta - t*beta_relax, min=1)
#   align     = score^alpha * iou^beta_eff
# =============================================================================


class ARAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with per-GT beta relaxed for high-aspect-ratio (tall/narrow) boxes."""

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 ar_thresh=2.0, ar_scale=2.0, beta_relax=2.0, eps=1e-9):
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
        # Section B: Center loss (v3: FIXED — per-anchor stride, per-dim
        # size-normalized L1; new 'crowd' mode weights by GT neighbor overlap)
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
        # Inverse frequency, mean-normalized, with configurable dampening
        class_counts = torch.tensor([34901.0, 28628.0, 66946.0], device=device)
        inv_freq = 1.0 / class_counts
        inv_freq = inv_freq / inv_freq.mean()

        # Round 6: class_weight_mode selects dampening strategy
        #   'sqrt'  : Rounds 1-5 default — gentle: [1.08, 1.19, 0.78] (bag 1.53x trolley)
        #   'linear': No dampening — aggressive: [0.92, 1.41, 0.67] (bag 2.10x trolley)
        self.class_weight_mode = getattr(h, 'class_weight_mode', 'sqrt')
        if self.class_weight_mode == 'linear':
            self.class_weights = inv_freq.clone()  # no sqrt dampening
        else:
            self.class_weights = torch.sqrt(inv_freq)
        self.class_weights = self.class_weights / self.class_weights.mean()

        # Toggle class weighting (default ON to reproduce Rounds 1-3; OFF = clean baseline)
        self.use_class_weighting = getattr(h, 'use_class_weighting', True)

        # Section G: classification loss mode ('bce' | 'qfl')
        self.cls_mode = getattr(h, 'cls_mode', 'bce')
        self.qfl_beta = getattr(h, 'qfl_beta', 2.0)

        # =====================================================================
        # Section J: Class-confusion repulsion (Round 7)
        # Penalize a predicted box that overlaps a DIFFERENT-class GT box.
        # Targets bag false positives (bag sitting on backpack/trolley).
        # =====================================================================
        self.use_repulsion = getattr(h, 'use_repulsion', False)
        self.repulsion_weight = getattr(h, 'repulsion_weight', 0.3)
        # sqrt mode: backpack≈1.08, bag≈1.19, trolley≈0.78
        # linear mode: backpack≈0.92, bag≈1.41, trolley≈0.67

        # =====================================================================
        # Section K (v3): size-aware CLS weighting (Round 8)
        # Recall gap is a RANKING problem (AR50_small 0.96 vs R50_small 0.71):
        # small objects are localized but scored too low. Boost the fg cls loss
        # for small objects so their positives are pushed to higher confidence.
        # QFL/VFL alternatives failed; this only reweights the existing BCE.
        # =====================================================================
        self.use_cls_swa = getattr(h, 'use_cls_swa', False)
        self.cls_swa_boost = getattr(h, 'cls_swa_boost', 1.75)

        # =====================================================================
        # Section L (v3): bag asymmetric penalty (Round 8)
        # Bag = 74% precision with CONFIDENT cross-class FPs. Upweight ONLY the
        # negative BCE term of the bag logit at fg anchors assigned to
        # backpack/trolley (their bag target is 0 there) — pushes the bag
        # score down on other-class objects without touching bag positives.
        # =====================================================================
        self.use_bag_penalty = getattr(h, 'use_bag_penalty', False)
        self.bag_penalty_weight = getattr(h, 'bag_penalty_weight', 2.0)
        self.bag_class_id = getattr(h, 'bag_class_id', 1)  # dataset order: backpack, bag, trolley

        # =====================================================================
        # Section M (v3): AR-aware TAL assigner (Round 8) — see class docstring
        # =====================================================================
        self.use_artal = getattr(h, 'use_artal', False)
        self.artal_ar_thresh = getattr(h, 'artal_ar_thresh', 2.0)
        self.artal_ar_scale = getattr(h, 'artal_ar_scale', 2.0)
        self.artal_beta_relax = getattr(h, 'artal_beta_relax', 2.0)

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
            # Section M (v3): AR-aware TAL — stock behavior for AR <= ar_thresh
            self.assigner = ARAwareTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                ar_thresh=self.artal_ar_thresh,
                ar_scale=self.artal_ar_scale,
                beta_relax=self.artal_beta_relax,
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
            print(f"  [A2] area_weight_mode: {self.bbox_loss.area_weight_mode}")
            if self.bbox_loss.class_boosts is not None:
                print(f"  [A2] class boosts (bp/bg/tr): {self.bbox_loss.class_boosts.cpu().numpy().round(3)}")
            print(f"  [B] center_loss_init: {self.center_loss_weight_init}" +
                  (f" (mode={self.center_loss_mode})" if self.center_loss_weight_init > 0 else ""))
            print(f"  [C] iou_clip:        {self.iou_clip_start} → {self.iou_clip_end} (DEPRECATED — keep off)")
            print(f"  [D] tal_topk:        {self.tal_topk}")
            print(f"  [D] tal_alpha:       {self.tal_alpha}")
            print(f"  [D] tal_beta:        {self.tal_beta}")
            print(f"  [E] use_satal:       {self.use_satal}")
            print(f"  [I] box_loss_type:   {self.bbox_loss.box_loss_type}")
            print(f"  [J] repulsion:       {self.use_repulsion}" + (f" (w={self.repulsion_weight})" if self.use_repulsion else ""))
            print(f"  [K] cls_swa:         {self.use_cls_swa}" + (f" (boost={self.cls_swa_boost} @ {self.small_obj_px}px)" if self.use_cls_swa else ""))
            print(f"  [L] bag_penalty:     {self.use_bag_penalty}" + (f" (w={self.bag_penalty_weight}, cls={self.bag_class_id})" if self.use_bag_penalty else ""))
            print(f"  [M] artal:           {self.use_artal}" + (f" (thresh={self.artal_ar_thresh}, scale={self.artal_ar_scale}, relax={self.artal_beta_relax})" if self.use_artal else ""))
            if self.use_satal:
                print(f"      satal_alpha_small: {self.satal_alpha_small}")
                print(f"      satal_beta_small:  {self.satal_beta_small}")
                print(f"      satal_topk_factor: {self.satal_topk_factor}")
            print(f"  [F] Class Weighting: {'ON' if self.use_class_weighting else 'OFF'} (mode: {self.class_weight_mode})")
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

    def _compute_center_loss(self, pred_bboxes, target_bboxes, fg_mask, stride_tensor,
                             gt_bboxes=None, mask_gt=None):
        """Auxiliary center loss (Section B, v3 FIXED).

        v2 bugs fixed:
          - per-anchor stride threshold (v2 used global stride.min()=8, so
            'small' meant <(px/8)^2 in feature units on ALL scales — a far
            broader net than SWA's, and the prime suspect in the r7 collapse)
          - per-dimension L1 normalized by GT w/h (v2 used raw feature-coord
            L1: scale-dependent, dominated by larger 'small' boxes; also test
            boxes are 2.6:1 tall so raw center-y error dominated x)

        Modes:
          'small' : small objects only (fixed version of the v2 intent)
          'crowd' : ALL fg, weighted by GT neighbor overlap — center accuracy
                    is what separates adjacent objects at NMS (~5 objs/img)
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

        if self.center_loss_mode == 'crowd' and gt_bboxes is not None and mask_gt is not None:
            # Weight every fg sample by how crowded its GT is: neighbors =
            # other GTs overlapping the assigned target box above the IoU thr.
            target_fg_px = target_fg * stride_fg.unsqueeze(-1)  # pixel coords
            crowd_w = torch.ones_like(err)
            for i in range(fg_mask.shape[0]):
                sel = fg_b == i
                if not sel.any():
                    continue
                gm = mask_gt[i].squeeze(-1) > 0
                if gm.sum() < 2:
                    continue  # 0 or 1 GT — no neighbors possible
                iou = self._pairwise_iou(target_fg_px[sel], gt_bboxes[i][gm])  # (n, m)
                # own GT matches itself with IoU~1, so subtract 1
                neighbors = (iou > self.center_crowd_iou).sum(dim=1).float() - 1.0
                crowd_w[sel] = 1.0 + neighbors.clamp(min=0.0)
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

    # =========================================================================
    # CLASSIFICATION LOSS (BCE + Class Weighting, no VFL)
    # =========================================================================

    def _compute_cls_loss_weighted(self, pred_scores, target_scores, fg_mask,
                                   target_labels_for_fg, target_scores_sum,
                                   target_bboxes_px=None):
        """
        Compute class-weighted BCE classification loss.

        Uses standard BCE with per-anchor class weighting based on inverse
        class frequency. No Varifocal Loss - just simple weighted BCE.

        Class weights: backpack≈1.08, bag≈1.19, trolley≈0.78

        v3 additions (both default OFF):
          Section K: fg anchors of SMALL objects (max side < small_obj_px,
                     pixel coords) get cls loss x cls_swa_boost.
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

        # ── Section K (v3): small-object cls boost ──
        if self.use_cls_swa and fg_mask.any() and target_bboxes_px is not None:
            tb = target_bboxes_px[fg_mask]  # (M, 4) pixel coords
            side = torch.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])
            boost = torch.where(side < float(self.small_obj_px),
                                side.new_tensor(float(self.cls_swa_boost)),
                                side.new_tensor(1.0)).to(dtype)
            weight[fg_mask] = weight[fg_mask] * boost.unsqueeze(-1)

        # ── Section L (v3): bag asymmetric penalty (negative term only) ──
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

        # ── Classification loss (class-weighted; v3: Sections K + L) ──
        # NOTE: target_bboxes is still in PIXEL coords here (divided by stride
        # only inside the box-loss branch below) — Section K relies on that.
        loss[1] = self._compute_cls_loss_weighted(
            pred_scores, target_scores, fg_mask, target_labels_for_fg, target_scores_sum,
            target_bboxes_px=target_bboxes
        )

        # ── Bounding box losses (SWA from loss_satal_swa.py) ──
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            self._sync_bbox_loss_state()

            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor,
                fg_labels=target_labels_for_fg  # v3: per-class boost (Section A2)
            )

            # Add auxiliary center loss (Section B, v3 fixed; gt boxes for 'crowd' mode)
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor,
                gt_bboxes=gt_bboxes, mask_gt=mask_gt
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
            masks = batch["masks"].to(self.device).floa


class DetectAuxLoss:
    """Train-only auxiliary-head deep-supervision loss.
    Round 20 -- DetectAux adds a parallel detection head over the same feature
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
    Round 24 -- supervises DetectObj's per-anchor objectness logit against the
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
