# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# MODIFIED: Luggage-dataset loss (backpack / bag / trolley), combined version.
#
# Every enhancement is gated by a hyp-config key and defaults to a NO-OP, so with
# default config this file reproduces the stock Ultralytics loss EXACTLY. This is
# deliberate: it gives you a clean baseline and lets you ablate one change at a time.
#
# ---- classification (loss[1]) --------------------------------------------------
#   class_weights            : list[nc] | None   (None -> uniform; stock)
#   normalize_class_weights  : bool = True        (mean-normalize so cls scale is preserved)
#   use_vfl                  : bool = False       (Varifocal modulation on top of weighted BCE)
#   vfl_alpha, vfl_gamma     : 0.75, 2.0
#   small_obj_cls_boost      : float = 1.0        (1.0 -> off; >1 boosts cls for small fg)
#
# ---- box regression (loss[0], loss[2]) ----------------------------------------
#   iou_ratio                : float = 1.0        (1.0 -> pure CIoU/stock; <1 blends in NWD)
#   nwd_c                    : float = 3.0        (NWD const in GRID-CELL units, not pixels)
#   small_obj_boost          : float = 1.0        (1.0 -> off; size-adaptive IoU+DFL weight)
#   small_obj_area_thresh    : float = 36.0       (area in grid-cell units^2)
#   use_inner_iou            : bool = False       (EXPERIMENTAL, off by default)
#   inner_iou_ratio_small    : float = 0.7
#   inner_iou_ratio_large    : float = 1.0
#   use_ar_penalty           : bool = False       (EXPERIMENTAL, off by default; see note)
#   ar_penalty_lambda        : float = 0.05
#   ar_penalty_tall_extra    : float = 0.5
#   ar_penalty_max           : float = 1.0        (clamp so it can't dominate 1-iou)
#
# NOTE on size units: inside the box loss, boxes are already divided by stride, so
# "area in grid cells" conflates true object size with FPN level. A 33x72px object
# is ~37 cells^2 on P3 but ~2 on P5, so a threshold of 36 makes most P4/P5 anchors
# count as "small". This is correlated with real size but is partly a per-level
# reweighting; keep it in mind when interpreting ablations.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


# ---------------------------------------------------------------------------
#  Fallback defaults (used only if a hyp key is missing). All are no-ops.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
#  Box-loss helpers (all operate on foreground boxes in grid-cell xyxy)
# ---------------------------------------------------------------------------
def _nwd_similarity(pred, target, c, eps=1e-7):
    """
    Normalized Wasserstein Distance similarity in (0, 1].

    Degrades smoothly with small center/size shifts instead of collapsing to 0 like
    IoU, which is why it helps on thin ~33px objects. pred/target: xyxy, same units.
    Ref: Wang et al., 2021, "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection".
    """
    cxp, cyp = (pred[:, 0] + pred[:, 2]) * 0.5, (pred[:, 1] + pred[:, 3]) * 0.5
    wp = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    hp = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    cxg, cyg = (target[:, 0] + target[:, 2]) * 0.5, (target[:, 1] + target[:, 3]) * 0.5
    wg = (target[:, 2] - target[:, 0]).clamp(min=eps)
    hg = (target[:, 3] - target[:, 1]).clamp(min=eps)
    w2 = (cxp - cxg) ** 2 + (cyp - cyg) ** 2 + ((wp - wg) * 0.5) ** 2 + ((hp - hg) * 0.5) ** 2
    return torch.exp(-torch.sqrt(w2 + eps) / c)


def _size_adaptive_weight(fg_boxes, boost, area_thresh):
    """Per-fg weight in [1, boost]: `boost` at area 0, linearly down to 1.0 at area_thresh."""
    w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
    h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
    ratio = ((w * h) / area_thresh).clamp(max=1.0)
    return (boost - (boost - 1.0) * ratio).unsqueeze(-1)  # (N_fg, 1)


def _inner_iou_scale(fg_boxes, r_small, r_large, area_thresh):
    """Per-fg inner ratio: r_small for tiny boxes -> r_large for large (no shrink)."""
    w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
    h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
    ratio = ((w * h) / area_thresh).clamp(max=1.0)
    return r_small + (r_large - r_small) * ratio


def _shrink_bbox(bboxes, ratio):
    """Shrink boxes toward their center by per-box `ratio` (1.0 = no change)."""
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    w = (bboxes[:, 2] - bboxes[:, 0]) * ratio
    h = (bboxes[:, 3] - bboxes[:, 1]) * ratio
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)


def _aspect_ratio_penalty(pred_fg, target_fg, lam, tall_extra, cap):
    """
    EXPERIMENTAL, bounded aspect-ratio penalty (opt-in, default off).

    Penalizes deviation of predicted h/w from the TARGET h/w (per object), with extra
    weight when the target is tall (>1.25). Bounded by `cap` so it can't overpower 1-iou.

    Honesty note: this overlaps with CIoU's internal aspect term, so it is a redundant
    regularizer rather than a novel signal. Kept as an ablation option; if it ever helps,
    prefer switching the base metric to DIoU + an explicit w/h term (EIoU/SIoU style)
    instead of stacking this on top of full CIoU. The original "match a fixed 2.7 prior"
    idea was dropped: pulling every box toward one ratio ignores real per-object variation.
    """
    pw = (pred_fg[:, 2] - pred_fg[:, 0]).clamp(min=1e-6)
    ph = (pred_fg[:, 3] - pred_fg[:, 1]).clamp(min=1e-6)
    tw = (target_fg[:, 2] - target_fg[:, 0]).clamp(min=1e-6)
    th = (target_fg[:, 3] - target_fg[:, 1]).clamp(min=1e-6)
    ar_diff = (torch.log(ph / pw + 1e-6) - torch.log(th / tw + 1e-6)).pow(2)
    tall_mask = (th / tw > 1.25).float()
    penalty = lam * ar_diff * (1.0 + tall_extra * tall_mask)
    return penalty.clamp(max=cap).unsqueeze(-1)  # (N_fg, 1)


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """
    Criterion class for computing training losses during training.

    Luggage additions (all toggleable, default = stock CIoU + DFL):
      - CIoU/NWD blend for small-object regression (iou_ratio, nwd_c)
      - size-adaptive IoU + DFL weighting (small_obj_boost)
      - optional Inner-IoU (use_inner_iou)
      - optional bounded aspect-ratio penalty (use_ar_penalty)
    """

    def __init__(self, reg_max=16, hyp=None):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

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

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss with optional small-object-aware terms (all no-ops under default config)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # --- base similarity: CIoU, optionally on Inner-IoU shrunk boxes ---
        if self.use_inner_iou:
            r = _inner_iou_scale(
                target_fg, self.inner_iou_ratio_small, self.inner_iou_ratio_large, self.small_obj_area_thresh
            )
            iou = bbox_iou(_shrink_bbox(pred_fg, r), _shrink_bbox(target_fg, r), xywh=False, CIoU=True)
        else:
            iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)

        # --- optionally blend in NWD (smooth for tiny boxes) ---
        if self.iou_ratio < 1.0:
            nwd = _nwd_similarity(pred_fg, target_fg, self.nwd_c).unsqueeze(-1)
            box_sim = self.iou_ratio * iou + (1.0 - self.iou_ratio) * nwd
        else:
            box_sim = iou

        loss_terms = 1.0 - box_sim

        # --- optional bounded aspect-ratio penalty (experimental, off by default) ---
        if self.use_ar_penalty:
            loss_terms = loss_terms + _aspect_ratio_penalty(
                pred_fg, target_fg, self.ar_penalty_lambda, self.ar_penalty_tall_extra, self.ar_penalty_max
            )

        # --- optional size-adaptive weighting (off by default) ---
        if self.small_obj_boost > 1.0:
            size_weight = _size_adaptive_weight(target_fg, self.small_obj_boost, self.small_obj_area_thresh)
        else:
            size_weight = 1.0

        loss_iou = (loss_terms * weight * size_weight).sum() / target_scores_sum

        # --- DFL loss (size-weighted only if boost is on) ---
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
            loss_dfl = (loss_dfl * weight * size_weight).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """
    Criterion class for computing training losses.

    Luggage additions to classification (all toggleable, default = stock BCE):
      - class-balanced weighting (class_weights, mean-normalized)
      - optional Varifocal modulation (use_vfl)
      - optional scale-aware cls boost for small objects (small_obj_cls_boost)
    Box-side additions live in BboxLoss.
    """

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max, hyp=h).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # --- classification config ---
        self.use_vfl = getattr(h, "use_vfl", _USE_VFL)
        self.vfl_alpha = getattr(h, "vfl_alpha", _VFL_ALPHA)
        self.vfl_gamma = getattr(h, "vfl_gamma", _VFL_GAMMA)
        self.small_obj_cls_boost = getattr(h, "small_obj_cls_boost", _SMALL_OBJ_CLS_BOOST)
        self.small_obj_area_thresh = getattr(h, "small_obj_area_thresh", _SMALL_OBJ_AREA_THRESH)

        cfg_cw = getattr(h, "class_weights", _CLASS_WEIGHTS)
        if cfg_cw is not None and self.nc == len(cfg_cw):
            w = torch.tensor(cfg_cw, dtype=torch.float, device=device)
            if getattr(h, "normalize_class_weights", _NORMALIZE_CW):
                w = w / w.mean()  # preserve overall cls scale relative to box/dfl
            self.class_weights = w.view(1, 1, -1)  # (1,1,nc)
        else:
            self.class_weights = None
            if cfg_cw is not None and self.nc != len(cfg_cw):
                print(f"[v8DetectionLoss] WARNING: class_weights has {len(cfg_cw)} entries "
                      f"but model nc={self.nc}. Using uniform weights.")

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
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
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_cls_loss(self, pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype):
        """
        Classification loss: BCE, optionally class-weighted, Varifocal-modulated, and
        scale-boosted. With default config this returns exactly the stock BCE result.

        NOTE: target_bboxes here is still in PIXEL coords (called before the /stride step
        in __call__), so we divide by stride to estimate grid-cell size for the boost.
        """
        loss = self.bce(pred_scores, target_scores.to(dtype))  # (B, N_anchors, nc)

        # Varifocal modulation
        if self.use_vfl:
            label = (target_scores > 0).to(dtype)
            p = pred_scores.sigmoid()
            vfl_w = self.vfl_alpha * p.pow(self.vfl_gamma) * (1 - label) + target_scores * label
            loss = loss * vfl_w

        # Class balancing
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(dtype)

        # Scale-aware boost for small foreground objects
        if self.small_obj_cls_boost > 1.0 and fg_mask.sum() > 0:
            stride_exp = stride_tensor.unsqueeze(0).expand(target_bboxes.shape[0], -1, -1)  # (B,N,1)
            fg_boxes = target_bboxes[fg_mask] / stride_exp[fg_mask]  # (N_fg,4) grid units
            w = (fg_boxes[:, 2] - fg_boxes[:, 0]).clamp(min=1e-6)
            h = (fg_boxes[:, 3] - fg_boxes[:, 1]).clamp(min=1e-6)
            ratio = ((w * h) / self.small_obj_area_thresh).clamp(max=1.0)
            cls_scale = self.small_obj_cls_boost - (self.small_obj_cls_boost - 1.0) * ratio  # boost..1
            scale_map = torch.ones(pred_scores.shape[0], pred_scores.shape[1], 1, device=pred_scores.device, dtype=dtype)
            scale_map[fg_mask] = cls_scale.unsqueeze(-1).to(dtype)
            loss = loss * scale_map

        return loss.sum() / target_scores_sum

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss (class-balanced + optional VFL + optional scale-boost)
        loss[1] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype
        )

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss (class-balanced + optional VFL + optional scale-boost)
        loss[2] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype
        )

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
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

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
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

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
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
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
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

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
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