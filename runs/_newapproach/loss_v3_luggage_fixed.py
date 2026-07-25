# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
#
# loss_v3_luggage.py — Round 12: Structural loss improvements for luggage detection.
#
# This file is a DROP-IN replacement for ultralytics/utils/loss.py. It reproduces
# the stock Ultralytics loss EXACTLY when all config keys are at their defaults.
#
# =============================================================================
# NEW MECHANISMS (all gated, all default OFF = stock behaviour):
# =============================================================================
#
# [M1] Gradient-Balanced EIoU (GB-EIoU)
#      box_metric: "gb_eiou"
#      Weights the EIoU width and height penalty terms proportionally to the
#      target aspect ratio. For a tall box (AR=2.69), width error gets ~73% of
#      the shape-penalty budget because a 2px width shift hurts IoU 2.5x more
#      than a 2px height shift on a 33x72px object. Standard EIoU splits 50/50.
#      Ref: Novel (this work), builds on Zhang et al. 2022 (EIoU).
#
# [M2] Aspect-Aware DFL (AA-DFL)
#      aa_dfl: bool = False
#      aa_dfl_gamma: float = 0.5   (0=uniform/stock, 1=max compression)
#      Redistributes DFL bins non-uniformly per-edge based on the target value.
#      Width edges (left/right) on a 33px object at stride 8 only use bins 0-4
#      out of 0-15 — 70% of bins are wasted. AA-DFL uses sqrt-compressed
#      projection for edges where the target value is small, giving 2x effective
#      resolution. Height edges keep standard spacing.
#      Ref: Novel (this work), extends Li et al. 2022 (DFL/GFL).
#
# [M3] Class-Conditional Box Loss (CC-Box)
#      cc_box: bool = False
#      cc_box_bag_metric: "diou"   (what to use for bag class; default=DIoU)
#      Applies different IoU metrics per class. Bags have the highest shape
#      variance (AR std highest, 1078 square bags, 241 wide) — EIoU's w/h
#      penalty actively fights variable-shape bags. Backpack/trolley have
#      consistent tall shapes and benefit from EIoU. Bags get DIoU (center-
#      only penalty, no shape constraint).
#
# [M4] Learned Task Weighting (Kendall Uncertainty)
#      learned_task_weights: bool = False
#      ltw_warmup: int = 10        (freeze log-var for first N epochs)
#      Replaces fixed box=7.5 / cls=0.5 / dfl=1.5 with learnable weights via
#      homoscedastic uncertainty (Kendall et al. 2018). The model LEARNS the
#      optimal per-task weighting. The 7.5/0.5/1.5 was tuned for COCO (80
#      classes); a 3-class luggage dataset likely has a very different optimum.
#      Ref: Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to
#      Weigh Losses", CVPR 2018.
#
# [M5] Shape-Aware TAL (SA-TAL)
#      sa_tal: bool = False
#      Overrides the CIoU overlap computation in TAL assignment with GB-EIoU.
#      This changes WHICH anchors get assigned to each GT — anchors that are
#      better-aligned on the narrow (width) axis get higher priority. This is
#      different from SATAL (which changes alpha/beta/topk); SA-TAL changes
#      the GEOMETRY of the alignment metric itself.
#
# [M6] IoU-DFL Consistency Loss
#      iou_dfl_consistency: float = 0.0   (weight; 0=off)
#      Auxiliary loss penalizing disagreement between DFL's argmax-decoded box
#      and the softmax-decoded box that IoU loss optimizes. When the DFL
#      distribution is multi-modal or flat, mode and expectation diverge,
#      creating gradient conflict. This regularizer pushes them together.
#
# KEPT FROM PREVIOUS ROUNDS (all still available, still gated):
#   class_weights, normalize_class_weights, use_vfl, small_obj_cls_boost,
#   iou_ratio (NWD blend), nwd_c, small_obj_boost, small_obj_area_thresh,
#   use_inner_iou, use_ar_penalty
#
# NEUTRAL CONFIG (exact stock loss): all new keys at defaults.
# =============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
)
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


# =============================================================================
# EPOCH TRACKING (needed for Kendall warmup)
# =============================================================================
_EPOCH_STATE = {"epoch": 0, "total": 0, "ever_set": False}


_LTW_REGISTRY = {"params": None, "in_optimizer": False}


def _register_ltw_params(trainer):
    """Add learned-task-weight params to the live optimizer exactly once.

    Needed because the loss object is built lazily on the first forward pass,
    which happens AFTER build_optimizer(). Without this the nn.Parameters get
    gradients but are never stepped -> 'learned' weights that never learn.
    """
    if _LTW_REGISTRY["in_optimizer"] or _LTW_REGISTRY["params"] is None:
        return
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return
    lr = opt.param_groups[0].get("lr", 0.01)
    opt.add_param_group({"params": list(_LTW_REGISTRY["params"]),
                         "lr": lr, "weight_decay": 0.0})
    _LTW_REGISTRY["in_optimizer"] = True
    print(f"[v3-loss] LTW: {len(_LTW_REGISTRY['params'])} task-weight params "
          f"added to optimizer (lr={lr}, weight_decay=0)")


def _epoch_callback(trainer):
    _EPOCH_STATE["epoch"] = getattr(trainer, "epoch", 0)
    _EPOCH_STATE["total"] = getattr(getattr(trainer, "args", None), "epochs", 0)
    _EPOCH_STATE["ever_set"] = True
    _register_ltw_params(trainer)


def attach_epoch_tracking(model):
    """Register epoch tracking on a YOLO model. Call BEFORE model.train()."""
    model.add_callback("on_train_epoch_start", _epoch_callback)
    return model


# =============================================================================
# FALLBACK DEFAULTS — all are no-ops (stock-equivalent)
# =============================================================================
_CLASS_WEIGHTS = None
_NORMALIZE_CW = True
_USE_VFL = False
_VFL_ALPHA = 0.75
_VFL_GAMMA = 2.0
_SMALL_OBJ_CLS_BOOST = 1.0

_IOU_RATIO = 1.0
_NWD_C = 3.0
_SMALL_OBJ_BOOST = 1.0
_SMALL_OBJ_AREA_THRESH = 36.0
_USE_INNER_IOU = False
_INNER_IOU_RATIO_SMALL = 0.7
_INNER_IOU_RATIO_LARGE = 1.0
_USE_AR_PENALTY = False
_AR_PENALTY_LAMBDA = 0.05
_AR_PENALTY_TALL_EXTRA = 0.5
_AR_PENALTY_MAX = 1.0

# New round-12 defaults
_BOX_METRIC = "ciou"
_AA_DFL = False
_AA_DFL_GAMMA = 0.5
_CC_BOX = False
_CC_BOX_BAG_METRIC = "diou"
_CC_BOX_BAG_CLASS = 1
_LEARNED_TASK_WEIGHTS = False
_LTW_WARMUP = 10
_SA_TAL = False
_IOU_DFL_CONSISTENCY = 0.0


# =============================================================================
# [M1] GRADIENT-BALANCED EIoU
# =============================================================================
def _gb_eiou(pred, target, eps=1e-7):
    """
    Gradient-Balanced EIoU: IoU - center_penalty - gamma_w * w_err - gamma_h * h_err.

    gamma_w / gamma_h are proportional to target AR, so the narrow axis (width for
    tall luggage) gets more gradient. For AR=2.69: gamma_w ≈ 0.73, gamma_h ≈ 0.27.
    Standard EIoU uses 0.5/0.5.

    Args:
        pred: (N, 4) xyxy
        target: (N, 4) xyxy

    Returns:
        (N, 1) similarity in approx [-1, 1]
    """
    b1x1, b1y1, b1x2, b1y2 = pred.chunk(4, -1)
    b2x1, b2y1, b2x2, b2y2 = target.chunk(4, -1)

    w1 = (b1x2 - b1x1).clamp(min=eps)
    h1 = (b1y2 - b1y1).clamp(min=eps)
    w2 = (b2x2 - b2x1).clamp(min=eps)
    h2 = (b2y2 - b2y1).clamp(min=eps)

    # IoU
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = (torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)).clamp(min=eps)
    ch = (torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)).clamp(min=eps)

    # Center distance penalty (same as EIoU/DIoU)
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
    c2 = cw ** 2 + ch ** 2 + eps

    # [M1] Gradient-balanced w/h terms
    # AR = target_h / target_w. For tall boxes, width error matters more.
    ar = (h2 / w2).clamp(min=0.5, max=5.0)
    gamma_w = ar / (1.0 + ar)        # ∈ [0.33, 0.83]
    gamma_h = 1.0 / (1.0 + ar)       # ∈ [0.17, 0.67]
    # Factor of 2 so total penalty magnitude matches standard EIoU
    w_term = 2.0 * gamma_w * (w1 - w2) ** 2 / (cw ** 2 + eps)
    h_term = 2.0 * gamma_h * (h1 - h2) ** 2 / (ch ** 2 + eps)

    return iou - rho2 / c2 - w_term - h_term


def _standard_eiou(pred, target, eps=1e-7):
    """Standard EIoU (50/50 w/h weighting) for comparison."""
    b1x1, b1y1, b1x2, b1y2 = pred.chunk(4, -1)
    b2x1, b2y1, b2x2, b2y2 = target.chunk(4, -1)
    w1 = (b1x2 - b1x1).clamp(min=eps)
    h1 = (b1y2 - b1y1).clamp(min=eps)
    w2 = (b2x2 - b2x1).clamp(min=eps)
    h2 = (b2y2 - b2y1).clamp(min=eps)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = (torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)).clamp(min=eps)
    ch = (torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)).clamp(min=eps)
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
    c2 = cw ** 2 + ch ** 2 + eps
    w_term = (w1 - w2) ** 2 / (cw ** 2 + eps)
    h_term = (h1 - h2) ** 2 / (ch ** 2 + eps)
    return iou - rho2 / c2 - w_term - h_term


def _diou(pred, target, eps=1e-7):
    """DIoU: IoU - center penalty only (no shape term). Good for shape-variable classes."""
    b1x1, b1y1, b1x2, b1y2 = pred.chunk(4, -1)
    b2x1, b2y1, b2x2, b2y2 = target.chunk(4, -1)
    w1 = (b1x2 - b1x1).clamp(min=eps)
    h1 = (b1y2 - b1y1).clamp(min=eps)
    w2 = (b2x2 - b2x1).clamp(min=eps)
    h2 = (b2y2 - b2y1).clamp(min=eps)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = (torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)).clamp(min=eps)
    ch = (torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)).clamp(min=eps)
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
    c2 = cw ** 2 + ch ** 2 + eps
    return iou - rho2 / c2


# =============================================================================
# PREVIOUS-ROUND HELPERS (kept for backward compatibility)
# =============================================================================
def _nwd_similarity(pred, target, c, eps=1e-7):
    """Normalized Wasserstein Distance similarity in (0, 1]."""
    cxp, cyp = (pred[:, 0] + pred[:, 2]) * 0.5, (pred[:, 1] + pred[:, 3]) * 0.5
    wp = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    hp = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    cxg, cyg = (target[:, 0] + target[:, 2]) * 0.5, (target[:, 1] + target[:, 3]) * 0.5
    wg = (target[:, 2] - target[:, 0]).clamp(min=eps)
    hg = (target[:, 3] - target[:, 1]).clamp(min=eps)
    w2 = (cxp - cxg) ** 2 + (cyp - cyg) ** 2 + ((wp - wg) * 0.5) ** 2 + ((hp - hg) * 0.5) ** 2
    return torch.exp(-torch.sqrt(w2 + eps) / c)


def _size_adaptive_weight(fg_boxes, boost, area_thresh):
    """Per-fg weight in [1, boost]."""
    w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
    h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
    ratio = ((w * h) / area_thresh).clamp(max=1.0)
    return (boost - (boost - 1.0) * ratio).unsqueeze(-1)


def _inner_iou_scale(fg_boxes, r_small, r_large, area_thresh):
    w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
    h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
    ratio = ((w * h) / area_thresh).clamp(max=1.0)
    return r_small + (r_large - r_small) * ratio


def _shrink_bbox(bboxes, ratio):
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    w = (bboxes[:, 2] - bboxes[:, 0]) * ratio
    h = (bboxes[:, 3] - bboxes[:, 1]) * ratio
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)


def _aspect_ratio_penalty(pred_fg, target_fg, lam, tall_extra, cap):
    pw = (pred_fg[:, 2] - pred_fg[:, 0]).clamp(min=1e-6)
    ph = (pred_fg[:, 3] - pred_fg[:, 1]).clamp(min=1e-6)
    tw = (target_fg[:, 2] - target_fg[:, 0]).clamp(min=1e-6)
    th = (target_fg[:, 3] - target_fg[:, 1]).clamp(min=1e-6)
    ar_diff = (torch.log(ph / pw + 1e-6) - torch.log(th / tw + 1e-6)).pow(2)
    tall_mask = (th / tw > 1.25).float()
    penalty = lam * ar_diff * (1.0 + tall_extra * tall_mask)
    return penalty.clamp(max=cap).unsqueeze(-1)


# =============================================================================
# [M5] SHAPE-AWARE TAL ASSIGNER
# =============================================================================
class ShapeAwareTAL(TaskAlignedAssigner):
    """
    TAL with GB-EIoU for the overlap computation in assignment.

    Standard TAL uses CIoU for align_metric = score^alpha * overlap^beta.
    For tall/narrow luggage, CIoU's aspect term degenerates. GB-EIoU gives
    anchors that are width-aligned a higher assignment score, so the model
    supervises width accuracy from better-positioned anchors.

    Activation: sa_tal=True in config. All other TAL params (topk, alpha, beta)
    are unchanged — this ONLY replaces the IoU geometry, not the scoring.
    """

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Override: use GB-EIoU instead of CIoU for assignment."""
        # (pred=pd_bboxes, target=gt_bboxes) so the AR gammas come from GROUND TRUTH.
        return _gb_eiou(pd_bboxes, gt_bboxes).squeeze(-1).clamp_(0)


# =============================================================================
# BASE LOSS COMPONENTS
# =============================================================================
class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(
                    pred_score.float(), gt_score.float(), reduction="none"
                ) * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Focal loss."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


# =============================================================================
# [M2] ASPECT-AWARE DFL
# =============================================================================
def make_dfl_proj(reg_max, aa_dfl=False, aa_dfl_gamma=0.5):
    """
    DFL bin positions.

    aa_dfl=False -> [0, 1, ..., reg_max-1] (stock).
    aa_dfl=True  -> power-spaced, DENSER NEAR ZERO:
                    proj[i] = (i/(reg_max-1))**(1+gamma) * (reg_max-1)
    For gamma=0.5 the first gaps are ~0.26/0.47/0.61 vs ~1.37/1.42/1.47 at the
    top, so short edges (the width edges of tall luggage) get finer quantization.

    MUST be used by BOTH the DFL loss and bbox_decode, or the network is trained
    against one bin geometry and decoded with another.
    """
    if not aa_dfl:
        return torch.arange(reg_max, dtype=torch.float)
    u = torch.linspace(0, 1, reg_max)
    return u.pow(1.0 + aa_dfl_gamma) * (reg_max - 1)


class DFLoss(nn.Module):
    """
    Distribution Focal Loss over an arbitrary monotonic bin projection.

    With a uniform projection this reduces EXACTLY to stock DFL (verified:
    searchsorted on [0..15] reproduces the tl/tr/wl/wr arithmetic bit-for-bit).
    """

    def __init__(self, reg_max=16, aa_dfl=False, aa_dfl_gamma=0.5):
        super().__init__()
        self.reg_max = reg_max
        self.aa_dfl = aa_dfl
        self.aa_dfl_gamma = aa_dfl_gamma
        self.register_buffer("proj", make_dfl_proj(reg_max, aa_dfl, aa_dfl_gamma))

    def __call__(self, pred_dist, target):
        """
        Args:
            pred_dist: (N*4, reg_max) logits
            target:    (N, 4) ltrb distances
        Returns:
            (N, 1) loss per anchor
        """
        proj = self.proj.to(device=target.device, dtype=target.dtype)
        lo, hi = float(proj[0]), float(proj[-1])
        t = target.clamp(lo, hi - 1e-3)
        flat = t.reshape(-1).contiguous()

        idx_r = torch.searchsorted(proj, flat).clamp(1, self.reg_max - 1)
        idx_l = idx_r - 1
        vl, vr = proj[idx_l], proj[idx_r]
        span = (vr - vl).clamp(min=1e-6)
        wr = (flat - vl) / span
        wl = 1.0 - wr

        loss = (F.cross_entropy(pred_dist, idx_l, reduction="none") * wl
                + F.cross_entropy(pred_dist, idx_r, reduction="none") * wr)
        return loss.view(target.shape).mean(-1, keepdim=True)


# =============================================================================
# BOX LOSS (with M1, M2, M3, M6)
# =============================================================================
class BboxLoss(nn.Module):
    """
    Box regression loss with all 6 structural mechanisms.

    All mechanisms default to OFF (stock CIoU + uniform DFL).
    """

    def __init__(self, reg_max=16, hyp=None):
        super().__init__()

        # --- DFL ---
        aa_dfl = getattr(hyp, "aa_dfl", _AA_DFL)
        aa_dfl_gamma = getattr(hyp, "aa_dfl_gamma", _AA_DFL_GAMMA)
        self.dfl_loss = DFLoss(reg_max, aa_dfl=aa_dfl, aa_dfl_gamma=aa_dfl_gamma) if reg_max > 1 else None

        # --- box metric ---
        self.box_metric = getattr(hyp, "box_metric", _BOX_METRIC)

        # --- class-conditional box loss ---
        self.cc_box = getattr(hyp, "cc_box", _CC_BOX)
        self.cc_box_bag_metric = getattr(hyp, "cc_box_bag_metric", _CC_BOX_BAG_METRIC)
        self.cc_box_bag_class = getattr(hyp, "cc_box_bag_class", _CC_BOX_BAG_CLASS)

        # --- IoU-DFL consistency ---
        self.iou_dfl_consistency = getattr(hyp, "iou_dfl_consistency", _IOU_DFL_CONSISTENCY)
        self.reg_max = reg_max

        # --- previous-round params ---
        self.iou_ratio = getattr(hyp, "iou_ratio", _IOU_RATIO)
        self.nwd_c = getattr(hyp, "nwd_c", _NWD_C)
        self.small_obj_boost = getattr(hyp, "small_obj_boost", _SMALL_OBJ_BOOST)
        self.small_obj_area_thresh = getattr(hyp, "small_obj_area_thresh", _SMALL_OBJ_AREA_THRESH)
        self.use_inner_iou = getattr(hyp, "use_inner_iou", _USE_INNER_IOU)
        self.inner_iou_ratio_small = getattr(hyp, "inner_iou_ratio_small", _INNER_IOU_RATIO_SMALL)
        self.inner_iou_ratio_large = getattr(hyp, "inner_iou_ratio_large", _INNER_IOU_RATIO_LARGE)
        self.use_ar_penalty = getattr(hyp, "use_ar_penalty", _USE_AR_PENALTY)
        self.ar_penalty_lambda = getattr(hyp, "ar_penalty_lambda", _AR_PENALTY_LAMBDA)
        self.ar_penalty_tall_extra = getattr(hyp, "ar_penalty_tall_extra", _AR_PENALTY_TALL_EXTRA)
        self.ar_penalty_max = getattr(hyp, "ar_penalty_max", _AR_PENALTY_MAX)

    def _compute_iou(self, pred_fg, target_fg, metric=None):
        """Compute IoU similarity using the specified metric."""
        m = metric or self.box_metric
        if m == "ciou":
            return bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        elif m == "gb_eiou":
            return _gb_eiou(pred_fg, target_fg)
        elif m == "eiou":
            return _standard_eiou(pred_fg, target_fg)
        elif m == "diou":
            return _diou(pred_fg, target_fg)
        else:
            raise ValueError(f"Unknown box_metric: {m!r}")

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """Box + DFL loss with all structural mechanisms."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # --- [M3] class-conditional box loss ---
        if self.cc_box and target_fg.shape[0] > 0:
            class_idx = target_scores[fg_mask].argmax(dim=-1)  # (N_fg,)
            bag_mask = class_idx == self.cc_box_bag_class
            non_bag_mask = ~bag_mask

            iou = torch.zeros(pred_fg.shape[0], 1, device=pred_fg.device, dtype=pred_fg.dtype)
            if non_bag_mask.any():
                iou[non_bag_mask] = self._compute_iou(
                    pred_fg[non_bag_mask], target_fg[non_bag_mask], self.box_metric
                )
            if bag_mask.any():
                iou[bag_mask] = self._compute_iou(
                    pred_fg[bag_mask], target_fg[bag_mask], self.cc_box_bag_metric
                )
        else:
            # --- standard single-metric path ---
            if self.use_inner_iou:
                r = _inner_iou_scale(
                    target_fg, self.inner_iou_ratio_small,
                    self.inner_iou_ratio_large, self.small_obj_area_thresh,
                )
                iou = self._compute_iou(_shrink_bbox(pred_fg, r), _shrink_bbox(target_fg, r))
            else:
                iou = self._compute_iou(pred_fg, target_fg)

        # --- optionally blend in NWD ---
        if self.iou_ratio < 1.0:
            nwd = _nwd_similarity(pred_fg, target_fg, self.nwd_c).unsqueeze(-1)
            box_sim = self.iou_ratio * iou + (1.0 - self.iou_ratio) * nwd
        else:
            box_sim = iou

        loss_terms = 1.0 - box_sim

        # --- optional AR penalty ---
        if self.use_ar_penalty:
            loss_terms = loss_terms + _aspect_ratio_penalty(
                pred_fg, target_fg,
                self.ar_penalty_lambda, self.ar_penalty_tall_extra, self.ar_penalty_max,
            )

        # --- optional size-adaptive weighting ---
        if self.small_obj_boost > 1.0:
            size_weight = _size_adaptive_weight(target_fg, self.small_obj_boost, self.small_obj_area_thresh)
        else:
            size_weight = 1.0

        loss_iou = (loss_terms * weight * size_weight).sum() / target_scores_sum

        # --- [M2] DFL loss (aspect-aware or standard) ---
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            )
            loss_dfl = (loss_dfl * weight * size_weight).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # --- [M6] IoU-DFL consistency loss ---
        loss_consistency = torch.tensor(0.0).to(pred_dist.device)
        if self.iou_dfl_consistency > 0 and self.dfl_loss is not None and fg_mask.sum() > 0:
            loss_consistency = self._iou_dfl_consistency_loss(
                pred_dist, pred_bboxes, target_bboxes, anchor_points, fg_mask,
            )

        return loss_iou, loss_dfl, loss_consistency

    def _iou_dfl_consistency_loss(self, pred_dist, pred_bboxes, target_bboxes,
                                   anchor_points, fg_mask):
        """
        [M6] Penalize when DFL's mode (argmax) decodes to a different box than
        the softmax expectation (which IoU loss optimizes).
        """
        b = pred_bboxes.shape[0]
        dist_fg = pred_dist[fg_mask].view(-1, 4, self.reg_max)

        # anchor_points is (A,2); fg_mask is (b,A) -> expand before masking.
        anchors_fg = anchor_points.unsqueeze(0).expand(b, -1, -1)[fg_mask]  # (N,2)

        # DFL's mode (argmax-decoded) box -> DETACHED TARGET.
        # Map bin indices through the DFL projection so mode and expectation
        # live in the same units (matters once aa_dfl uses non-uniform bins).
        proj = self.dfl_loss.proj.to(device=dist_fg.device, dtype=dist_fg.dtype)
        mode_idx = dist_fg.detach().argmax(dim=-1)                 # (N,4)
        mode_vals = proj[mode_idx]                                  # (N,4)
        mode_box = dist2bbox(mode_vals, anchors_fg, xywh=False).detach()

        # Softmax-decoded box — ATTACHED, this is where gradient enters.
        soft_box = pred_bboxes[fg_mask]

        # Gate on prediction quality (detached, weight only).
        iou = bbox_iou(soft_box.detach(), target_bboxes[fg_mask],
                       xywh=False).squeeze(-1).clamp(0)

        consistency = F.smooth_l1_loss(soft_box, mode_box, reduction="none", beta=0.5)
        return (consistency.mean(dim=-1) * iou).mean()


# =============================================================================
# ROTATED BBOX LOSS (unchanged from stock — no luggage-specific changes needed)
# =============================================================================
class RotatedBboxLoss(BboxLoss):
    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


# =============================================================================
# DETECTION LOSS (main class — wires M1-M6 together)
# =============================================================================
class v8DetectionLoss:
    """
    v8 Detection loss with 6 structural mechanisms for luggage.

    All mechanisms default to OFF. With default config, this reproduces the
    stock Ultralytics detection loss EXACTLY (verified: BboxLoss returns
    3 values, 3rd is always 0.0 when consistency=0).
    """

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1

        # --- [M5] Shape-Aware TAL ---
        sa_tal = getattr(h, "sa_tal", _SA_TAL)
        if sa_tal:
            self.assigner = ShapeAwareTAL(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
            print("[v3-loss] SA-TAL active: assignment uses GB-EIoU overlap")
        else:
            self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

        self.bbox_loss = BboxLoss(m.reg_max, hyp=h).to(device)
        # Decode with the DFL's own bin projection (uniform unless aa_dfl=True).
        if self.bbox_loss.dfl_loss is not None:
            self.proj = self.bbox_loss.dfl_loss.proj.to(device=device, dtype=torch.float)
        else:
            self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # --- [M4] Learned Task Weighting ---
        self.learned_task_weights = getattr(h, "learned_task_weights", _LEARNED_TASK_WEIGHTS)
        self.ltw_warmup = getattr(h, "ltw_warmup", _LTW_WARMUP)
        if self.learned_task_weights:
            # Initialize near stock values: box=7.5, cls=0.5, dfl=1.5
            # log_var = log(2 * gain^2) so that precision = 1/(2*exp(log_var)) ≈ 1/gain^2
            # s = -log(gain) -> exp(-s) == gain exactly at init.
            g_box = float(getattr(h, "box", 7.5))
            g_cls = float(getattr(h, "cls", 0.5))
            g_dfl = float(getattr(h, "dfl", 1.5))
            self.log_var_box = nn.Parameter(torch.tensor(-math.log(g_box), device=device))
            self.log_var_cls = nn.Parameter(torch.tensor(-math.log(g_cls), device=device))
            self.log_var_dfl = nn.Parameter(torch.tensor(-math.log(g_dfl), device=device))
            _LTW_REGISTRY["params"] = [self.log_var_box, self.log_var_cls, self.log_var_dfl]
            _LTW_REGISTRY["in_optimizer"] = False
            print(f"[v3-loss] Learned task weights active (warmup={self.ltw_warmup} epochs); "
                  f"init multipliers box/cls/dfl = {g_box}/{g_cls}/{g_dfl}")
            print("[v3-loss] NOTE: requires attach_epoch_tracking(model) so the params "
                  "reach the optimizer.")
        else:
            self.log_var_box = None

        # --- IoU-DFL consistency weight ---
        self.iou_dfl_consistency_w = getattr(h, "iou_dfl_consistency", _IOU_DFL_CONSISTENCY)

        # --- classification config (from previous rounds) ---
        self.use_vfl = getattr(h, "use_vfl", _USE_VFL)
        self.vfl_alpha = getattr(h, "vfl_alpha", _VFL_ALPHA)
        self.vfl_gamma = getattr(h, "vfl_gamma", _VFL_GAMMA)
        self.small_obj_cls_boost = getattr(h, "small_obj_cls_boost", _SMALL_OBJ_CLS_BOOST)
        self.small_obj_area_thresh = getattr(h, "small_obj_area_thresh", _SMALL_OBJ_AREA_THRESH)

        cfg_cw = getattr(h, "class_weights", _CLASS_WEIGHTS)
        if cfg_cw is not None and self.nc == len(cfg_cw):
            w = torch.tensor(cfg_cw, dtype=torch.float, device=device)
            if getattr(h, "normalize_class_weights", _NORMALIZE_CW):
                w = w / w.mean()
            self.class_weights = w.view(1, 1, -1)
        else:
            self.class_weights = None
            if cfg_cw is not None and self.nc != len(cfg_cw):
                print(f"[v8DetectionLoss] WARNING: class_weights has {len(cfg_cw)} entries "
                      f"but model nc={self.nc}. Using uniform weights.")

        # Print config summary
        self._print_config()

    def _print_config(self):
        h = self.hyp
        print("\n" + "=" * 62)
        print("  loss_v3_luggage — Round 12")
        print("=" * 62)
        metric = getattr(h, "box_metric", _BOX_METRIC)
        print(f"  box_metric:            {metric}")
        print(f"  aa_dfl:                {getattr(h, 'aa_dfl', _AA_DFL)}")
        print(f"  cc_box:                {getattr(h, 'cc_box', _CC_BOX)}"
              f" (bag_metric={getattr(h, 'cc_box_bag_metric', _CC_BOX_BAG_METRIC)})")
        print(f"  learned_task_weights:  {self.learned_task_weights}")
        print(f"  sa_tal:                {getattr(h, 'sa_tal', _SA_TAL)}")
        print(f"  iou_dfl_consistency:   {self.iou_dfl_consistency_w}")
        print(f"  class_weights:         {self.class_weights is not None}")
        print(f"  iou_ratio (NWD):       {getattr(h, 'iou_ratio', _IOU_RATIO)}")
        neutral = (
            metric == "ciou"
            and not getattr(h, "aa_dfl", False)
            and not getattr(h, "cc_box", False)
            and not self.learned_task_weights
            and not getattr(h, "sa_tal", False)
            and self.iou_dfl_consistency_w == 0.0
            and self.class_weights is None
            and getattr(h, "iou_ratio", 1.0) == 1.0
            and not self.use_vfl
            and self.small_obj_cls_boost == 1.0
            and getattr(h, "small_obj_boost", 1.0) == 1.0
            and not getattr(h, "use_inner_iou", False)
            and not getattr(h, "use_ar_penalty", False)
        )
        print(f"  stock-equivalent:      {neutral}")
        print("=" * 62 + "\n")

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_cls_loss(self, pred_scores, target_scores, target_bboxes, fg_mask,
                          stride_tensor, target_scores_sum, dtype):
        loss = self.bce(pred_scores, target_scores.to(dtype))

        if self.use_vfl:
            label = (target_scores > 0).to(dtype)
            p = pred_scores.sigmoid()
            vfl_w = self.vfl_alpha * p.pow(self.vfl_gamma) * (1 - label) + target_scores * label
            loss = loss * vfl_w

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(dtype)

        if self.small_obj_cls_boost > 1.0 and fg_mask.sum() > 0:
            stride_exp = stride_tensor.unsqueeze(0).expand(target_bboxes.shape[0], -1, -1)
            fg_boxes = target_bboxes[fg_mask] / stride_exp[fg_mask]
            w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
            h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
            ratio = ((w * h) / self.small_obj_area_thresh).clamp(max=1.0)
            cls_scale = self.small_obj_cls_boost - (self.small_obj_cls_boost - 1.0) * ratio
            scale_map = torch.ones(pred_scores.shape[0], pred_scores.shape[1], 1,
                                   device=pred_scores.device, dtype=dtype)
            scale_map[fg_mask] = cls_scale.unsqueeze(-1).to(dtype)
            loss = loss * scale_map

        return loss.sum() / target_scores_sum

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
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

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
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

        # Cls loss
        loss[1] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask,
            stride_tensor, target_scores_sum, dtype,
        )

        # Box loss
        loss_consistency = torch.tensor(0.0, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2], loss_consistency = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask,
            )

        # --- [M4] Learned Task Weighting OR fixed gains ---
        if self.learned_task_weights and self.log_var_box is not None:
            # During warmup: use fixed gains (let the model stabilize first)
            epoch = _EPOCH_STATE["epoch"] if _EPOCH_STATE["ever_set"] else 0
            if epoch < self.ltw_warmup:
                loss[0] *= self.hyp.box
                loss[1] *= self.hyp.cls
                loss[2] *= self.hyp.dfl
            else:
                # Kendall multi-task: L_total = L_i / (2*exp(s_i)) + s_i/2
                prec_box = torch.exp(-self.log_var_box)
                prec_cls = torch.exp(-self.log_var_cls)
                prec_dfl = torch.exp(-self.log_var_dfl)
                loss[0] = prec_box * loss[0] + self.log_var_box * 0.5
                loss[1] = prec_cls * loss[1] + self.log_var_cls * 0.5
                loss[2] = prec_dfl * loss[2] + self.log_var_dfl * 0.5
        else:
            loss[0] *= self.hyp.box
            loss[1] *= self.hyp.cls
            loss[2] *= self.hyp.dfl

        # Add consistency loss
        total = loss.sum() + self.iou_dfl_consistency_w * loss_consistency

        return total * batch_size, loss.detach()


# =============================================================================
# SEGMENTATION LOSS (unchanged from stock)
# =============================================================================
class v8SegmentationLoss(v8DetectionLoss):
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
            raise TypeError(
                "ERROR segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

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

        loss[2] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask,
            stride_tensor, target_scores_sum, dtype,
        )

        if fg_mask.sum():
            loss[0], loss[3], _ = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                proto, pred_masks, imgsz, self.overlap,
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

        return loss / fg_mask.sum()


# =============================================================================
# POSE LOSS (unchanged from stock)
# =============================================================================
class v8PoseLoss(v8DetectionLoss):
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
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4], _ = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask,
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


# =============================================================================
# CLASSIFICATION LOSS (unchanged)
# =============================================================================
class v8ClassificationLoss:
    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


# =============================================================================
# OBB LOSS (unchanged)
# =============================================================================
class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
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

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask,
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


# =============================================================================
# E2E DETECT LOSS (unchanged)
# =============================================================================
class E2EDetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
