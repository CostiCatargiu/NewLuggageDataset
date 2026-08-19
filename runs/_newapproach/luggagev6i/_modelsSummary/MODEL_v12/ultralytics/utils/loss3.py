# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
#
# MODIFIED: Luggage-dataset loss (backpack / bag / trolley), integrated version.
#
# Dataset-matched changes (from your analysis report):
#   - 94% of boxes are "tall" (h/w > 1.25), mean h/w = 2.69
#   - mean box = 33px wide x 72px tall  -> WIDTH is the hard dimension
#   - imbalance only 2.3:1 (trolley:bag)  -> mild
#
# Design: every enhancement is gated by a hyp key and defaults to a NO-OP, so with
# default config this file reproduces stock Ultralytics loss EXACTLY. Ablate one
# change at a time.
#
# BOX SIDE (the lever that matters for this dataset):
#   box_metric        : "ciou"|"eiou"|"siou"  (default "ciou" -> stock)
#                       CIoU's aspect term ~0 when the h/w RATIO matches even if
#                       absolute width is wrong -> weak on thin objects. EIoU/SIoU
#                       use explicit w & h error terms instead. START HERE.
#   iou_ratio         : float = 1.0   (1.0 -> pure metric; <1 blends in NWD)
#   nwd_c             : float          (NWD const, PIXELS)
#   nwd_width_gate_px : float|None     (None -> all fg; else NWD only where
#                                       target width < this many px)
#   small_obj_boost   : float = 1.0   (1.0 -> off; WIDTH-adaptive up-weight,
#                                       normalized so it's a reweight not a gain hike)
#   small_obj_width_thresh_px : float
#
# CLS SIDE (kept as toggleable ablation knobs; your data suggests leaving OFF):
#   class_weights / normalize_class_weights / use_vfl / small_obj_cls_boost
#
# UNITS: all size/NWD math is done in PIXELS (boxes multiplied back by stride) so
# one threshold = one physical size on every FPN level. No per-level artifact.

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
#  No-op fallback defaults
# ---------------------------------------------------------------------------
_CLASS_WEIGHTS = None
_NORMALIZE_CW = True
_USE_VFL = False
_VFL_ALPHA = 0.75
_VFL_GAMMA = 2.0
_SMALL_OBJ_CLS_BOOST = 1.0

_BOX_METRIC = "ciou"
_IOU_RATIO = 1.0
_NWD_C = 12.0                     # pixels; ~ mean object width
_NWD_WIDTH_GATE_PX = None         # None -> ungated
_SMALL_OBJ_BOOST = 1.0
_SMALL_OBJ_WIDTH_THRESH_PX = 24.0


# ---------------------------------------------------------------------------
#  Regression metrics (all operate on xyxy boxes, same units)
# ---------------------------------------------------------------------------
def _iou_metric(pred, target, metric, eps=1e-7):
    """
    Per-box similarity, shape (N,).

    metric="ciou": stock bbox_iou (identical to baseline).
    metric="eiou": IoU - center_dist_norm - w_err_norm - h_err_norm   (Zhang 2022).
    metric="siou": IoU - 0.5*(distance_cost + shape_cost)             (Gevorgyan 2022).

    EIoU/SIoU penalize width & height error directly, which is what tall/thin
    objects need. Scale-invariant, so grid-cell input is fine.
    """
    if metric == "ciou":
        return bbox_iou(pred, target, xywh=False, CIoU=True).view(-1)

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    pw = (px2 - px1).clamp(min=eps); ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps); th = (ty2 - ty1).clamp(min=eps)

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=eps)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=eps)

    pcx = (px1 + px2) * 0.5; pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5; tcy = (ty1 + ty2) * 0.5
    dx = pcx - tcx; dy = pcy - tcy

    if metric == "eiou":
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = dx ** 2 + dy ** 2
        w_term = (pw - tw) ** 2 / (cw ** 2 + eps)
        h_term = (ph - th) ** 2 / (ch ** 2 + eps)
        return iou - rho2 / c2 - w_term - h_term

    if metric == "siou":
        sigma = torch.sqrt(dx ** 2 + dy ** 2) + eps
        sin_alpha = torch.abs(dy) / sigma
        sin_beta = torch.abs(dx) / sigma
        sin_alpha = torch.where(sin_alpha < sin_beta, sin_alpha, sin_beta)
        angle_cost = torch.cos(2 * (torch.asin(sin_alpha.clamp(-1 + eps, 1 - eps)) - torch.pi / 4))
        rho_x = (dx / cw) ** 2
        rho_y = (dy / ch) ** 2
        gamma = 2 - angle_cost
        distance_cost = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))
        omega_w = torch.abs(pw - tw) / torch.max(pw, tw)
        omega_h = torch.abs(ph - th) / torch.max(ph, th)
        theta = 4.0
        shape_cost = (1 - torch.exp(-omega_w)) ** theta + (1 - torch.exp(-omega_h)) ** theta
        return iou - 0.5 * (distance_cost + shape_cost)

    raise ValueError(f"unknown box_metric: {metric!r}")


def _nwd_similarity(pred, target, c, eps=1e-7):
    """Normalized Wasserstein similarity in (0,1]. xyxy PIXEL units, c in pixels."""
    cxp, cyp = (pred[:, 0] + pred[:, 2]) * 0.5, (pred[:, 1] + pred[:, 3]) * 0.5
    wp = (pred[:, 2] - pred[:, 0]).clamp(min=eps); hp = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    cxg, cyg = (target[:, 0] + target[:, 2]) * 0.5, (target[:, 1] + target[:, 3]) * 0.5
    wg = (target[:, 2] - target[:, 0]).clamp(min=eps); hg = (target[:, 3] - target[:, 1]).clamp(min=eps)
    w2 = (cxp - cxg) ** 2 + (cyp - cyg) ** 2 + ((wp - wg) * 0.5) ** 2 + ((hp - hg) * 0.5) ** 2
    return torch.exp(-torch.sqrt(w2 + eps) / c)


def _width_adaptive_weight(fg_boxes_px, boost, width_thresh_px):
    """Per-fg weight in [1, boost]: boost at width 0 -> 1.0 at width_thresh_px.

    Keyed on WIDTH so thin trolleys (medium area, small width) still get boosted.
    """
    w = (fg_boxes_px[:, 2] - fg_boxes_px[:, 0]).clamp(min=1e-6)
    ratio = (w / width_thresh_px).clamp(max=1.0)
    return boost - (boost - 1.0) * ratio  # (N_fg,)


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

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
    """Wraps focal loss around existing loss_fcn()."""

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


class DFLoss(nn.Module):
    """Distribution Focal Loss (unchanged from stock)."""

    def __init__(self, reg_max=16) -> None:
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
    Box regression loss with a metric switch and dataset-matched small-object handling.

    Defaults reproduce stock CIoU + DFL exactly. Turn on ONE thing at a time:
      box_metric="siou"           # main lever for tall/thin objects
      iou_ratio=0.7 + width gate  # optional NWD help for the thin-small tail
      small_obj_boost=1.5         # optional width-adaptive up-weight (normalized)

    NOTE: forward() takes `stride_tensor` (final arg) so size/NWD math is in pixels.
    """

    def __init__(self, reg_max=16, hyp=None):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        self.metric = getattr(hyp, "box_metric", _BOX_METRIC)
        self.iou_ratio = getattr(hyp, "iou_ratio", _IOU_RATIO)
        self.nwd_c = getattr(hyp, "nwd_c", _NWD_C)
        self.nwd_width_gate_px = getattr(hyp, "nwd_width_gate_px", _NWD_WIDTH_GATE_PX)
        self.small_obj_boost = getattr(hyp, "small_obj_boost", _SMALL_OBJ_BOOST)
        self.small_obj_width_thresh_px = getattr(hyp, "small_obj_width_thresh_px", _SMALL_OBJ_WIDTH_THRESH_PX)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # (N_fg, 1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # Recover PIXEL-space fg boxes for size/NWD math (boxes come in grid-cell units).
        stride_fg = stride_tensor.expand(target_bboxes.shape[0], -1, -1)[fg_mask]  # (N_fg,1)
        pred_fg_px = pred_fg * stride_fg
        target_fg_px = target_fg * stride_fg

        # --- base similarity ---
        sim = _iou_metric(pred_fg, target_fg, self.metric).unsqueeze(-1)  # (N_fg,1)

        # --- optional NWD blend, pixel space, optionally width-gated ---
        if self.iou_ratio < 1.0:
            nwd = _nwd_similarity(pred_fg_px, target_fg_px, self.nwd_c).unsqueeze(-1)
            blended = self.iou_ratio * sim + (1.0 - self.iou_ratio) * nwd
            if self.nwd_width_gate_px is not None:
                tw_px = (target_fg_px[:, 2] - target_fg_px[:, 0]).unsqueeze(-1)
                gate = (tw_px < self.nwd_width_gate_px).to(sim.dtype)  # 1 where thin
                sim = gate * blended + (1.0 - gate) * sim
            else:
                sim = blended

        loss_terms = 1.0 - sim  # (N_fg, 1)

        # --- optional width-adaptive weighting (normalized -> pure reweight) ---
        if self.small_obj_boost > 1.0:
            size_weight = _width_adaptive_weight(
                target_fg_px, self.small_obj_boost, self.small_obj_width_thresh_px
            ).unsqueeze(-1)
        else:
            size_weight = torch.ones_like(weight)

        # Normalizer includes size_weight so the boost reweights rather than
        # inflating box/dfl gain (fixes the previous version's hidden coupling).
        norm = (weight * size_weight).sum().clamp(min=1e-9)

        loss_iou = (loss_terms * weight * size_weight).sum() / norm

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
            loss_dfl = (loss_dfl * weight * size_weight).sum() / norm
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Rotated box loss (probiou). Keeps the original 7-arg forward signature."""

    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Keypoint loss."""

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """
    Detection loss.

    Classification additions (all toggleable, default = stock BCE):
      - class-balanced weighting (class_weights, mean-normalized)
      - optional Varifocal modulation (use_vfl)
      - optional WIDTH-aware cls boost for thin objects (small_obj_cls_boost)
    Box-side additions live in BboxLoss.
    """

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args

        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
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
        self.small_obj_width_thresh_px = getattr(h, "small_obj_width_thresh_px", _SMALL_OBJ_WIDTH_THRESH_PX)

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
        """
        BCE, optionally class-weighted, Varifocal-modulated, and width-boosted.
        Default config returns exactly the stock BCE result.

        target_bboxes here is in PIXEL coords (called before the /stride step), so
        width in pixels is directly available for the boost.
        """
        loss = self.bce(pred_scores, target_scores.to(dtype))  # (B, N, nc)

        if self.use_vfl:
            label = (target_scores > 0).to(dtype)
            p = pred_scores.sigmoid()
            vfl_w = self.vfl_alpha * p.pow(self.vfl_gamma) * (1 - label) + target_scores * label
            loss = loss * vfl_w

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(dtype)

        if self.small_obj_cls_boost > 1.0 and fg_mask.sum() > 0:
            fg_boxes_px = target_bboxes[fg_mask]  # pixels
            w = (fg_boxes_px[:, 2] - fg_boxes_px[:, 0]).clamp(min=1e-6)
            ratio = (w / self.small_obj_width_thresh_px).clamp(max=1.0)
            cls_scale = self.small_obj_cls_boost - (self.small_obj_cls_boost - 1.0) * ratio
            scale_map = torch.ones(pred_scores.shape[0], pred_scores.shape[1], 1,
                                   device=pred_scores.device, dtype=dtype)
            scale_map[fg_mask] = cls_scale.unsqueeze(-1).to(dtype)
            loss = loss * scale_map

        return loss.sum() / target_scores_sum

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

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

        loss[1] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype
        )

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Segmentation loss."""

    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

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
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
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
            pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype
        )

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor,
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

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                                    proto, pred_masks, imgsz, overlap):
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


class v8PoseLoss(v8DetectionLoss):
    """Pose loss."""

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
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

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
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
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


class v8ClassificationLoss:
    """Classification loss."""

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """OBB loss (rotated). Uses RotatedBboxLoss with the 7-arg forward."""

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
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

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
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
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
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
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


class DetectAuxLoss:
    """Train-only auxiliary-head deep-supervision loss (dropped at inference)."""

    def __init__(self, model, aux_weight=0.25):
        self.det = v8DetectionLoss(model, tal_topk=10)
        self.aux_weight = getattr(model.model[-1], "aux_weight", aux_weight)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        if not isinstance(preds, dict):
            return self.det(preds, batch)
        loss_main = self.det(preds["main"], batch)
        loss_aux = self.det(preds["aux"], batch)
        return loss_main[0] + self.aux_weight * loss_aux[0], loss_main[1]


class DetectObjLoss(v8DetectionLoss):
    """v8 detection loss + an objectness (fg/bg) BCE term.

    Supervises DetectObj's per-anchor objectness logit against the TAL foreground
    mask. Cls loss goes through _compute_cls_loss so it picks up the same toggles
    (class_weights / VFL / width-boost) as the main detection loss.
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

        loss[1] = self._compute_cls_loss(
            pred_scores, target_scores, target_bboxes, fg_mask, stride_tensor, target_scores_sum, dtype
        )

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
        obj_target = fg_mask.unsqueeze(-1).to(dtype)  # (b, A, 1)
        loss[3] = self.bce_obj(pred_obj, obj_target).mean()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.obj_weight
        return loss.sum() * batch_size, loss[:3].detach()  # log box/cls/dfl


class E2EDetectLoss:
    """End-to-end detection loss (one-to-many + one-to-one)."""

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