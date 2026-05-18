# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# WEAPON-SATAL-SWA-Plus-NWD4: NWD4 Loss for Weapon Detection
# =============================================================================
#
# Adapted from satal_swa_plus_nwd4.py for weapon dataset
#
# Dataset: GunDatasetHistogram
#   - Classes: knife, long_gun, other, pistol (4 classes)
#   - Class distribution: knife=16.6%, long_gun=30.4%, other=16.0%, pistol=37.0%
#   - Size distribution: Small=1.4%, Medium=25.4%, Large=73.1%
#
# Changes from luggage version:
#   1. Class weights updated for weapon classes (4 classes instead of 3)
#   2. Class counts from weapon dataset
#   3. Since 73% are large objects, NWD may help via gradient stabilization
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, crop_mask
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import OKS_SIGMA

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


# =============================================================================
# BASE LOSS COMPONENTS
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


# =============================================================================
# BBOX LOSS WITH NWD4
# =============================================================================


class BboxLoss(nn.Module):
    """
    Bounding box loss with SWA + NWD4 (improved Normalized Wasserstein Distance).

    NWD4 improvements over NWD3:
      - Symmetric normalization via geometric mean of pred/target dimensions
      - Gaussian kernel exp(-w²/C²) — no gradient singularity
      - Soft sigmoid gate — smooth transition at size boundary
      - Conservative ratio (default 0.1) — NWD supplements, doesn't replace CIoU
      - Auto-disables small_obj_boost to avoid redundant emphasis
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

        # NWD4 defaults
        self.use_nwd = True
        self.nwd_ratio = 0.1               # supplements CIoU
        self.nwd_constant = 3.0             # wider gradient range
        self.nwd_small_only = True          # gate NWD to small boxes
        self.nwd_gate_temperature = 0.3     # soft gate width
        self.nwd_dim_clamp = 0.5            # minimum dimension in grid units

    def normalized_wasserstein_distance_v4(self, pred, target, constant=3.0, eps=1e-7):
        """
        NWD4: Symmetric, scale-normalized Wasserstein similarity with smooth gradients.
        """
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        tgt_cx = (target[..., 0] + target[..., 2]) / 2
        tgt_cy = (target[..., 1] + target[..., 3]) / 2

        pred_w = (pred[..., 2] - pred[..., 0]).clamp(min=eps)
        pred_h = (pred[..., 3] - pred[..., 1]).clamp(min=eps)
        tgt_w = (target[..., 2] - target[..., 0]).clamp(min=eps)
        tgt_h = (target[..., 3] - target[..., 1]).clamp(min=eps)

        # Symmetric normalization via geometric mean
        norm_w = torch.sqrt(pred_w * tgt_w).clamp(min=self.nwd_dim_clamp)
        norm_h = torch.sqrt(pred_h * tgt_h).clamp(min=self.nwd_dim_clamp)

        # Center distance normalized by box dimensions
        center_dist = ((pred_cx - tgt_cx) / norm_w).pow(2) + \
                      ((pred_cy - tgt_cy) / norm_h).pow(2)

        # Size distance as relative width/height error
        size_dist = ((pred_w - tgt_w) / norm_w).pow(2) + \
                    ((pred_h - tgt_h) / norm_h).pow(2)

        w2 = center_dist + size_dist

        # Gaussian kernel: exp(-w²/C²)
        nwd = torch.exp(-w2 / (constant ** 2))
        return nwd

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

        # NWD4
        self.use_nwd = getattr(hyp, 'use_nwd', self.use_nwd)
        self.nwd_ratio = getattr(hyp, 'nwd_ratio', self.nwd_ratio)
        self.nwd_constant = getattr(hyp, 'nwd_constant', self.nwd_constant)
        self.nwd_small_only = getattr(hyp, 'nwd_small_only', self.nwd_small_only)
        self.nwd_gate_temperature = getattr(hyp, 'nwd_gate_temperature', self.nwd_gate_temperature)
        self.nwd_dim_clamp = getattr(hyp, 'nwd_dim_clamp', self.nwd_dim_clamp)

        # Auto-disable small_obj_boost when NWD is active
        if self.use_nwd:
            self.small_obj_boost = 1.0

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

        if area_weight.numel() > 0:
            area_weight = area_weight / (area_weight.max() + 1e-8)

        fg_areas = target_areas[fg_mask]
        small_threshold = None

        if stride is not None and area_weight.numel() > 0:
            min_stride = stride.min().clamp_min(1.0)
            small_threshold = (self.small_obj_px / min_stride) ** 2

            if self.small_obj_boost > 1.0:
                small_mask = fg_areas < small_threshold
                if small_mask.any():
                    area_weight = area_weight.clone()
                    area_weight[small_mask] *= self.small_obj_boost

        return score_weight, area_weight, fg_areas, small_threshold

    def _get_gradient_clip_values(self):
        """Get adaptive gradient clipping values based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        max_iou = self.iou_clip_end + (self.iou_clip_start - self.iou_clip_end) * (1 - progress)
        max_dfl = self.dfl_clip_end + (self.dfl_clip_start - self.dfl_clip_end) * (1 - progress)
        return max_iou, max_dfl

    def _compute_soft_gate(self, fg_areas, small_threshold):
        """Soft sigmoid gate for NWD application."""
        temperature = small_threshold * self.nwd_gate_temperature
        temperature = max(temperature, 1e-6)
        gate = torch.sigmoid(-(fg_areas - small_threshold) / temperature)
        return gate.unsqueeze(-1)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        """Compute IoU and DFL losses with SWA weighting + NWD4."""

        alpha = self._get_dynamic_alpha()
        score_weight, area_weight, fg_areas, small_threshold = self._compute_weights(
            target_bboxes, target_scores, fg_mask, stride
        )

        weight = alpha * area_weight + (1 - alpha) * score_weight

        # IoU loss
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        ciou_loss = 1.0 - iou

        # NWD4 blending
        if self.use_nwd:
            nwd = self.normalized_wasserstein_distance_v4(
                pred_bboxes[fg_mask], target_bboxes[fg_mask],
                constant=self.nwd_constant
            ).unsqueeze(-1)
            nwd_loss = 1.0 - nwd

            if self.nwd_small_only and small_threshold is not None:
                gate = self._compute_soft_gate(fg_areas, small_threshold)
                ratio = self.nwd_ratio * gate
            else:
                ratio = self.nwd_ratio

            box_term = (1.0 - ratio) * ciou_loss + ratio * nwd_loss

            # Debug logging
            if not hasattr(self, '_nwd_dbg_count'):
                self._nwd_dbg_count = 0
            if self._nwd_dbg_count < 30:
                if isinstance(ratio, float):
                    avg_ratio = ratio
                    n_gated = fg_areas.numel()
                else:
                    avg_ratio = ratio.mean().item()
                    n_gated = (ratio > 0.01).sum().item()
                print(f"[NWD4-Weapon] nwd mean={nwd.mean().item():.4f} "
                      f"loss mean={nwd_loss.mean().item():.4f} "
                      f"avg_ratio={avg_ratio:.4f} "
                      f"n_nwd_active={n_gated}/{fg_areas.numel()}")
                self._nwd_dbg_count += 1
        else:
            box_term = ciou_loss

        per_sample_iou_loss = box_term * weight

        # Adaptive clip
        max_iou_clip, max_dfl_clip = self._get_gradient_clip_values()
        per_sample_iou_loss = per_sample_iou_loss.clamp(max=max_iou_clip / 10.0)

        # Aggregate
        loss_iou = per_sample_iou_loss.sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            per_sample_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight

            per_sample_dfl = per_sample_dfl.clamp(max=max_dfl_clip / 10.0)
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
    WEAPON-SATAL-SWA-Plus-NWD4 Detection Loss.

    Adapted for weapon dataset (4 classes: knife, long_gun, other, pistol)
    
    Key changes from luggage version:
      - 4 classes instead of 3
      - Class weights computed from weapon dataset distribution
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
        # Section F: Class Weighting for WEAPON dataset
        # =====================================================================
        # Dataset: knife=10532, long_gun=19288, other=10152, pistol=23475
        # From 63,447 total instances (TRAIN split)
        # Order: knife, long_gun, other, pistol (alphabetical)
        class_counts = torch.tensor([10532.0, 19288.0, 10152.0, 23475.0], device=device)
        inv_freq = 1.0 / class_counts
        inv_freq = inv_freq / inv_freq.mean()
        self.class_weights = torch.sqrt(inv_freq)
        self.class_weights = self.class_weights / self.class_weights.mean()
        # Result: knife≈1.22, long_gun≈0.90, other≈1.24, pistol≈0.82

        # =====================================================================
        # Section G: Classification Loss Mode
        # =====================================================================
        # BCE + Class Weighting (no VFL)

        # =====================================================================
        # Section H: NWD4 parameters
        # =====================================================================
        self.use_nwd = getattr(h, 'use_nwd', True)
        self.nwd_ratio = getattr(h, 'nwd_ratio', 0.1)
        self.nwd_constant = getattr(h, 'nwd_constant', 3.0)
        self.nwd_small_only = getattr(h, 'nwd_small_only', True)
        self.nwd_gate_temperature = getattr(h, 'nwd_gate_temperature', 0.3)
        self.nwd_dim_clamp = getattr(h, 'nwd_dim_clamp', 0.5)

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
            print("WEAPON-SATAL-SWA-Plus-NWD4 Detection Loss Configuration")
            print("=" * 60)
            print(f"  [DATASET] WEAPON (4 classes: knife, long_gun, other, pistol)")
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
            print(f"      weights (kn/lg/ot/pi): {self.class_weights.cpu().numpy().round(3)}")
            print(f"  [G] Cls Loss: BCE + Class Weighting (no VFL)")
            print(f"  [H] NWD4 Config:")
            print(f"      use_nwd:          {self.use_nwd}")
            if self.use_nwd:
                print(f"      nwd_ratio:        {self.nwd_ratio}")
                print(f"      nwd_constant:     {self.nwd_constant}")
                print(f"      nwd_small_only:   {self.nwd_small_only}")
                print(f"      nwd_gate_temp:    {self.nwd_gate_temperature}")
                print(f"      nwd_dim_clamp:    {self.nwd_dim_clamp}")
                print(f"      small_obj_boost:  {self.bbox_loss.small_obj_boost} (auto-disabled)")
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

    def _compute_cls_loss_weighted(self, pred_scores, target_scores, fg_mask,
                                   target_labels_for_fg, target_scores_sum):
        """Compute class-weighted BCE classification loss."""
        dtype = pred_scores.dtype
        bs, num_anchors, nc = pred_scores.shape

        # Standard BCE loss
        bce = self.bce(pred_scores, target_scores.to(dtype))

        # Build per-anchor weight tensor for class weighting
        weight = torch.ones(bs, num_anchors, 1, device=self.device, dtype=dtype)

        if fg_mask.any() and target_labels_for_fg.numel() > 0:
            fg_class_weights = self.class_weights.to(dtype)[target_labels_for_fg]
            weight[fg_mask] = fg_class_weights.unsqueeze(-1)

        loss = (bce * weight).sum() / target_scores_sum

        return loss

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

        # Extract class labels for foreground anchors
        if fg_mask.any():
            target_labels_for_fg = target_scores[fg_mask].argmax(dim=-1)
        else:
            target_labels_for_fg = torch.tensor([], device=self.device, dtype=torch.long)

        # Classification loss (class-weighted BCE)
        loss[1] = self._compute_cls_loss_weighted(
            pred_scores, target_scores, fg_mask, target_labels_for_fg, target_scores_sum
        )

        # Bounding box losses (SWA + NWD4)
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
# OTHER LOSS CLASSES (unchanged from original)
# =============================================================================


class v8ClassificationLoss:
    """Criterion class for computing classification training losses."""

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class E2EDetectLoss:
    """End-to-end detection loss for weapon datasets."""

    def __init__(self, model):
        """Initialize with weapon-specialized losses for both branches."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
