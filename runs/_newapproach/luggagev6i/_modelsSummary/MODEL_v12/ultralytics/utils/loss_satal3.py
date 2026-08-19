# Ultralytics  AGPL-3.0 License - https://ultralytics.com/license
# Modified for Ablation Study - Parameters read from model.args
#
# CHANGES vs previous version (all defaults preserve old behavior unless noted):
#   BUGFIX 1: small-object mask now uses PER-ANCHOR stride, not stride.min().
#             (Before: boxes on P4/P5 up to ~140/280 px were counted "small".)
#   BUGFIX 2: hard clamp on per-sample losses replaced with a SOFT clip that
#             rescales instead of flattening — clipped samples keep gradient
#             direction instead of getting exactly zero gradient.
#   BUGFIX 3: v8SegmentationLoss `fg_mask.sum` -> `fg_mask.sum()`.
#   BUGFIX 4: alpha log actually prints every 10 epochs as the comment said.
#   NEW  [F]: size-gated NWD (Normalized Wasserstein Distance) regression term
#             for tiny boxes — IoU is near-binary for ~20 px objects, NWD stays
#             smooth. Blended with CIoU: gate=1 at 0 px, fades to 0 at nwd_gate_px.
#   NEW  [G]: aspect-ratio importance weighting — upweights tall/narrow GT boxes
#             (test/valid median AR ~2.6 vs train 1.46) to counter the shape shift.
#   NEW  [H]: optional EIoU instead of CIoU (separate width/height penalties —
#             better gradient for very narrow boxes), and optional per-class
#             BCE weights for the trolley/bag imbalance.
#
# EPOCH SYNC (required for all epoch schedules: alpha, clips, center loss):
#   Ultralytics does NOT set model.current_epoch. Attach the callback once:
#
#       from loss import attach_epoch_sync
#       model = YOLO("yolo12n.yaml")
#       attach_epoch_sync(model)
#       model.train(...)
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, crop_mask
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
try:
    from .shape_tal import ShapeAwareTaskAlignedAssigner
except ImportError:
    try:
        from shape_tal import ShapeAwareTaskAlignedAssigner
    except ImportError:
        ShapeAwareTaskAlignedAssigner = None
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import OKS_SIGMA
from .metrics import bbox_iou, probiou
from .tal import bbox2dist


def attach_epoch_sync(model):
    """Attach a trainer callback that mirrors the current epoch onto the model,
    so the loss schedules (alpha, adaptive clip, center-loss decay) actually run.
    Call once on the YOLO() object before .train()."""
    def _sync(trainer):
        for m in (trainer.model, getattr(trainer.model, "module", None)):
            if m is not None:
                m.current_epoch = trainer.epoch
    model.add_callback("on_train_epoch_start", _sync)


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
    Bounding box loss with configurable ablation parameters.
    Reads parameters from model.args (passed via model.train()).

    Ablation Parameters:
        Section A (Size-aware weighting):
            - alpha_start / alpha_end / alpha_min / alpha_max
            - small_obj_px (default 70), small_obj_boost (default 1.5)
        Section C (Adaptive soft-clipping):
            - iou_clip_start / iou_clip_end (default 20 -> 10, applied /10 per-sample)
            - dfl_clip_start / dfl_clip_end (default 10 -> 5, applied /10 per-sample)
        Section F (NWD tiny-box regression):
            - use_nwd (default False)
            - nwd_gate_px (default 48): gate=1 at 0 px, linearly fades to 0 here
            - nwd_const (default 12.8): C in exp(-W2/C), in pixels
        Section G (Aspect-ratio importance weighting):
            - ar_lambda (default 0.0 = off): weight *= 1 + ar_lambda*clamp(AR-ar_ref, 0, ar_cap)
            - ar_ref (default 1.5), ar_cap (default 2.0)
        Section H (IoU variant):
            - iou_type: 'ciou' (default) or 'eiou'
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.reg_max = reg_max
        # Training state
        self.epoch = 0
        self.total_epochs = 70
        # Section A defaults
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
        # Section F: NWD defaults (off)
        self.use_nwd = False
        self.nwd_gate_px = 48.0
        self.nwd_const = 12.8
        # Section G: AR weighting defaults (off)
        self.ar_lambda = 0.0
        self.ar_ref = 1.5
        self.ar_cap = 2.0
        # Section H: IoU variant
        self.iou_type = 'ciou'

    def set_params(self, hyp):
        """Set parameters from hyperparameters (model.args)."""
        # Section A
        self.small_obj_px = getattr(hyp, 'small_obj_px', self.small_obj_px)
        self.small_obj_boost = getattr(hyp, 'small_obj_boost', self.small_obj_boost)
        self.alpha_start = getattr(hyp, 'alpha_start', self.alpha_start)
        self.alpha_end = getattr(hyp, 'alpha_end', self.alpha_end)
        self.alpha_min = getattr(hyp, 'alpha_min', self.alpha_min)
        self.alpha_max = getattr(hyp, 'alpha_max', self.alpha_max)
        self.total_epochs = getattr(hyp, 'epochs', self.total_epochs)
        # Section C
        self.iou_clip_start = getattr(hyp, 'iou_clip_start', self.iou_clip_start)
        self.iou_clip_end = getattr(hyp, 'iou_clip_end', self.iou_clip_end)
        self.dfl_clip_start = getattr(hyp, 'dfl_clip_start', self.dfl_clip_start)
        self.dfl_clip_end = getattr(hyp, 'dfl_clip_end', self.dfl_clip_end)
        # Section F
        self.use_nwd = getattr(hyp, 'use_nwd', self.use_nwd)
        self.nwd_gate_px = getattr(hyp, 'nwd_gate_px', self.nwd_gate_px)
        self.nwd_const = getattr(hyp, 'nwd_const', self.nwd_const)
        # Section G
        self.ar_lambda = getattr(hyp, 'ar_lambda', self.ar_lambda)
        self.ar_ref = getattr(hyp, 'ar_ref', self.ar_ref)
        self.ar_cap = getattr(hyp, 'ar_cap', self.ar_cap)
        # Section H
        self.iou_type = getattr(hyp, 'iou_type', self.iou_type)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _soft_clip(x, cap):
        """Bound per-sample loss at `cap` while PRESERVING gradient direction.
        (A hard clamp gives exactly zero gradient above the cap, so the hardest
        samples stop learning; this rescales them onto the cap instead.)"""
        scale = (cap / x.detach().clamp(min=1e-8)).clamp(max=1.0)
        return x * scale

    @staticmethod
    def _fg_stride(stride, fg_mask):
        """Per-foreground-anchor stride, shape (N,). stride: (A,1), fg_mask: (b,A)."""
        return stride.view(1, -1).expand(fg_mask.shape[0], -1)[fg_mask]

    @staticmethod
    def _eiou(box1, box2, eps=1e-7):
        """EIoU (Zhang et al. 2022): IoU - center penalty - separate w/h penalties.
        Better-behaved than CIoU's coupled arctan aspect term for very narrow
        boxes, where the width error dominates. Boxes xyxy, (N,4) -> (N,1)."""
        b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
        b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
        w1, h1 = (b1x2 - b1x1).clamp(min=eps), (b1y2 - b1y1).clamp(min=eps)
        w2, h2 = (b2x2 - b2x1).clamp(min=eps), (b2y2 - b2y1).clamp(min=eps)
        inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp_(0) * \
                (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp_(0)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = ((b2x1 + b2x2 - b1x1 - b1x2).pow(2) + (b2y1 + b2y2 - b1y1 - b1y2).pow(2)) / 4
        return iou - rho2 / c2 - (w1 - w2).pow(2) / (cw.pow(2) + eps) - (h1 - h2).pow(2) / (ch.pow(2) + eps)

    @staticmethod
    def _nwd(box1, box2, C=12.8, eps=1e-7):
        """Normalized Wasserstein Distance similarity (Wang et al. 2021, tiny
        object detection). Models boxes as 2D Gaussians; exp(-W2/C) stays smooth
        where IoU is near-binary (<32 px boxes). Boxes xyxy in PIXELS, (N,4) -> (N,1)."""
        b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
        b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
        cx1, cy1 = (b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2
        cx2, cy2 = (b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2
        w1, h1 = b1x2 - b1x1, b1y2 - b1y1
        w2, h2 = b2x2 - b2x1, b2y2 - b2y1
        w2_dist = ((cx1 - cx2).pow(2) + (cy1 - cy2).pow(2) +
                   ((w1 - w2) / 2).pow(2) + ((h1 - h2) / 2).pow(2)).clamp(min=eps).sqrt()
        return torch.exp(-w2_dist / C)

    # ----------------------------------------------------------------- pieces

    def _get_dynamic_alpha(self):
        """Calculate dynamic alpha based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        alpha = self.alpha_start * (1 - progress) + self.alpha_end * progress
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        if not hasattr(self, '_last_logged_epoch'):
            self._last_logged_epoch = -1
        if self.epoch != self._last_logged_epoch:
            if self.epoch % 10 == 0:  # BUGFIX 4: was % 1 (printed every epoch)
                print(f"[Alpha] Epoch {self.epoch}/{self.total_epochs}: α={alpha:.3f}")
            self._last_logged_epoch = self.epoch
        return alpha

    def _compute_target_areas(self, target_bboxes):
        """Compute target bounding box areas with numerical stability."""
        areas = (target_bboxes[..., 2] - target_bboxes[..., 0]) * \
                (target_bboxes[..., 3] - target_bboxes[..., 1])
        return areas.clamp(min=1e-6)

    def _compute_weights(self, target_bboxes, target_scores, fg_mask, stride=None):
        """Compute combined area and score weights for loss calculation."""
        target_areas = self._compute_target_areas(target_bboxes)
        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        area_weight = (1.0 / target_areas[fg_mask]).unsqueeze(-1)
        if area_weight.numel() > 0:
            area_weight = area_weight / (area_weight.max() + 1e-8)
        # Small object boost.
        # BUGFIX 1: areas are in PER-ANCHOR grid units (boxes were divided by
        # each anchor's stride), so the pixel threshold must use the per-anchor
        # stride too. Using stride.min() flagged boxes up to small_obj_px*(s/8)
        # pixels as "small" on coarser levels (e.g. ~280 px on P5).
        if stride is not None and area_weight.numel() > 0:
            s_fg = self._fg_stride(stride, fg_mask).clamp_min(1.0)          # (N,)
            small_threshold = (self.small_obj_px / s_fg).pow(2)             # (N,)
            fg_areas = target_areas[fg_mask]
            small_mask = fg_areas < small_threshold
            if small_mask.any():
                area_weight = area_weight.clone()
                area_weight[small_mask] *= self.small_obj_boost
        return score_weight, area_weight

    def _get_clip_values(self):
        """Get adaptive soft-clip values based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        max_iou = self.iou_clip_end + (self.iou_clip_start - self.iou_clip_end) * (1 - progress)
        max_dfl = self.dfl_clip_end + (self.dfl_clip_start - self.dfl_clip_end) * (1 - progress)
        return max_iou, max_dfl

    # ---------------------------------------------------------------- forward

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        """Compute IoU (+NWD) and DFL losses with per-sample soft-clipping."""
        alpha = self._get_dynamic_alpha()
        score_weight, area_weight = self._compute_weights(
            target_bboxes, target_scores, fg_mask, stride
        )
        weight = alpha * area_weight + (1 - alpha) * score_weight

        # Section G: aspect-ratio importance weighting (AR is stride-invariant,
        # so grid units are fine). Upweights tall/narrow GT to match the
        # test/valid shape distribution (median AR ~2.6 vs train ~1.46).
        if self.ar_lambda > 0:
            tb = target_bboxes[fg_mask]
            gt_w = (tb[:, 2] - tb[:, 0]).clamp(min=1e-6)
            gt_h = (tb[:, 3] - tb[:, 1]).clamp(min=1e-6)
            ar = gt_h / gt_w
            ar_mult = 1.0 + self.ar_lambda * (ar - self.ar_ref).clamp(min=0.0, max=self.ar_cap)
            weight = weight * ar_mult.unsqueeze(-1)

        # Section H: IoU term (CIoU default, EIoU optional)
        if self.iou_type == 'eiou':
            iou = self._eiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        reg_term = 1.0 - iou

        # Section F: size-gated NWD blend for tiny boxes (needs pixel space)
        if self.use_nwd and stride is not None and fg_mask.any():
            s_fg = self._fg_stride(stride, fg_mask).unsqueeze(-1)            # (N,1)
            pb_px = pred_bboxes[fg_mask] * s_fg
            gb_px = target_bboxes[fg_mask] * s_fg
            nwd = self._nwd(pb_px, gb_px, C=self.nwd_const)                  # (N,1)
            side_px = torch.maximum(gb_px[:, 2] - gb_px[:, 0],
                                    gb_px[:, 3] - gb_px[:, 1]).unsqueeze(-1)  # (N,1)
            gate = ((self.nwd_gate_px - side_px) / max(self.nwd_gate_px, 1e-6)).clamp(0.0, 1.0)
            reg_term = gate * (1.0 - nwd) + (1.0 - gate) * reg_term

        per_sample_iou_loss = reg_term * weight

        # Section C: adaptive SOFT clip (BUGFIX 2: was a hard clamp -> zero grad)
        max_iou_clip, max_dfl_clip = self._get_clip_values()
        per_sample_iou_loss = self._soft_clip(per_sample_iou_loss, max_iou_clip / 10.0)
        loss_iou = per_sample_iou_loss.sum() / target_scores_sum

        # DFL loss per sample
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            per_sample_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            per_sample_dfl = self._soft_clip(per_sample_dfl, max_dfl_clip / 10.0)
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


class v8DetectionLoss:
    """
    YOLOv8 Detection Loss with configurable ablation parameters.
    All custom parameters are read from model.args (hyperparameters).
    Pass them via model.train(..., param_name=value, ...).

    Ablation Parameters:
        Section A: alpha_start, alpha_end, alpha_min, alpha_max, small_obj_px, small_obj_boost
        Section B: center_loss_weight_init, center_loss_weight_min, center_loss_decay_epochs
        Section C: iou_clip_start, iou_clip_end, dfl_clip_start, dfl_clip_end
        Section D: tal_topk, tal_alpha, tal_beta
        Section E: use_shape_tal, shape_gamma, shape_min
        Section F: use_nwd, nwd_gate_px, nwd_const
        Section G: ar_lambda, ar_ref, ar_cap
        Section H: iou_type ('ciou'|'eiou'), class_weights (list or 'w1,w2,w3')
    """

    def __init__(self, model, tal_topk=10):
        """Initialize v8DetectionLoss with parameters from model.args."""
        device = next(model.parameters()).device
        h = model.args  # Hyperparameters from model.train()
        m = model.model[-1]  # Detect() module
        self._model = model  # Store model reference for epoch sync
        # Model properties
        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1
        # =====================================================================
        # READ ABLATION PARAMETERS FROM model.args
        # =====================================================================
        self.epoch = 0
        self.total_epochs = getattr(h, 'epochs', 70)
        # Section A
        self.small_obj_px = getattr(h, 'small_obj_px', 70)
        self.small_obj_boost = getattr(h, 'small_obj_boost', 1.5)
        self.alpha_start = getattr(h, 'alpha_start', 0.9)
        self.alpha_end = getattr(h, 'alpha_end', 0.5)
        self.alpha_min = getattr(h, 'alpha_min', 0.3)
        self.alpha_max = getattr(h, 'alpha_max', 0.9)
        # Section B
        self.center_loss_weight_init = getattr(h, 'center_loss_weight_init', 0.0)
        self.center_loss_weight_min = getattr(h, 'center_loss_weight_min', 0.01)
        self.center_loss_decay_epochs = getattr(h, 'center_loss_decay_epochs', 35)
        # Section C (read here for _print_config)
        self.iou_clip_start = getattr(h, 'iou_clip_start', 20.0)
        self.iou_clip_end = getattr(h, 'iou_clip_end', 10.0)
        self.dfl_clip_start = getattr(h, 'dfl_clip_start', 10.0)
        self.dfl_clip_end = getattr(h, 'dfl_clip_end', 5.0)
        # Section D
        self.tal_topk = getattr(h, 'tal_topk', tal_topk)
        self.tal_alpha = getattr(h, 'tal_alpha', 0.5)
        self.tal_beta = getattr(h, 'tal_beta', 6.0)
        # Section E
        self.use_shape_tal = getattr(h, 'use_shape_tal', False)
        self.shape_gamma = getattr(h, 'shape_gamma', 1.0)
        self.shape_min = getattr(h, 'shape_min', 0.3)
        # Section F (read here for _print_config)
        self.use_nwd = getattr(h, 'use_nwd', False)
        self.nwd_gate_px = getattr(h, 'nwd_gate_px', 48.0)
        self.nwd_const = getattr(h, 'nwd_const', 12.8)
        # Section G (read here for _print_config)
        self.ar_lambda = getattr(h, 'ar_lambda', 0.0)
        self.ar_ref = getattr(h, 'ar_ref', 1.5)
        self.ar_cap = getattr(h, 'ar_cap', 2.0)
        # Section H
        self.iou_type = getattr(h, 'iou_type', 'ciou')
        cw = getattr(h, 'class_weights', None)
        if isinstance(cw, str):
            cw = [float(x) for x in cw.split(',')]
        if cw is not None:
            assert len(cw) == self.nc, f"class_weights needs {self.nc} values, got {len(cw)}"
            self.class_weights = torch.tensor(cw, dtype=torch.float, device=device).view(1, 1, -1)
        else:
            self.class_weights = None
        # =====================================================================
        # LOSS FUNCTIONS
        # =====================================================================
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.bbox_loss.set_params(h)
        # Task Aligned Assigner — standard or shape-aware
        if self.use_shape_tal and ShapeAwareTaskAlignedAssigner is not None:
            self.assigner = ShapeAwareTaskAlignedAssigner(
                topk=self.tal_topk,
                num_classes=self.nc,
                alpha=self.tal_alpha,
                beta=self.tal_beta,
                gamma=self.shape_gamma,
                shape_min=self.shape_min,
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
        self._print_config()

    def _print_config(self):
        """Print current configuration for verification."""
        if not hasattr(self, '_config_printed'):
            print("\n" + "=" * 60)
            print("v8DetectionLoss Configuration")
            print("=" * 60)
            print(f"  [A] alpha_start:     {self.alpha_start}")
            print(f"  [A] alpha_end:       {self.alpha_end}")
            print(f"  [A] small_obj_px:    {self.small_obj_px}")
            print(f"  [A] small_obj_boost: {self.small_obj_boost}")
            print(f"  [B] center_loss_init:     {self.center_loss_weight_init}")
            print(f"  [B] center_loss_min:     {self.center_loss_weight_min}")
            print(f"  [C] iou_clip:        {self.iou_clip_start} → {self.iou_clip_end}")
            print(f"  [C] dfl_clip:        {self.dfl_clip_start} → {self.dfl_clip_end}")
            print(f"  [D] tal_topk:        {self.tal_topk}")
            print(f"  [D] tal_alpha:       {self.tal_alpha}")
            print(f"  [D] tal_beta:        {self.tal_beta}")
            print(f"  [E] use_shape_tal:   {self.use_shape_tal}")
            if self.use_shape_tal:
                print(f"  [E] shape_gamma:     {self.shape_gamma}")
                print(f"  [E] shape_min:       {self.shape_min}")
            print(f"  [F] use_nwd:         {self.use_nwd}")
            if self.use_nwd:
                print(f"  [F] nwd_gate_px:     {self.nwd_gate_px}")
                print(f"  [F] nwd_const:       {self.nwd_const}")
            print(f"  [G] ar_lambda:       {self.ar_lambda}")
            if self.ar_lambda > 0:
                print(f"  [G] ar_ref:          {self.ar_ref}")
                print(f"  [G] ar_cap:          {self.ar_cap}")
            print(f"  [H] iou_type:        {self.iou_type}")
            print(f"  [H] class_weights:   {None if self.class_weights is None else self.class_weights.flatten().tolist()}")
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
        # BUGFIX 1 (same as BboxLoss): per-anchor stride threshold, not min stride
        s_fg = stride_tensor.view(-1)[fg_indices[1]].clamp_min(1.0)
        small_obj_threshold = (self.small_obj_px / s_fg) ** 2
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

    def __call__(self, preds, batch):
        """Calculate the sum of detection losses (box, cls, dfl)."""
        # Epoch from model — requires attach_epoch_sync(model) before train()!
        try:
            if hasattr(self._model, 'current_epoch'):
                self.epoch = self._model.current_epoch
        except:
            pass
        self._sync_bbox_loss_state()
        loss = torch.zeros(3, device=self.device)
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
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
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
        # Classification loss (BCE, optionally per-class weighted — Section H)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            self._sync_bbox_loss_state()
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor
            )
            loss[0] = loss[0] + center_loss
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


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

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

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
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
            )
            rw = targets[:, 4] * imgsz[0].item()
            rh = targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted.") from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


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


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing pose losses."""

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
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        batch_size = pred_scores.shape[0]
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
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
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

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
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
    """Criterion class for computing segmentation losses."""

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
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ segment dataset incorrectly formatted.") from e
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
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz, overlap):
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
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
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()  # BUGFIX 3: was fg_mask.sum (method ref)


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
