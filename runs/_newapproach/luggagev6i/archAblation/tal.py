# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        stride_val (int): The stride value used for select_candidates_in_gts.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

        # --- SCB (Size-Conditioned Beta) — inert unless beta_small is set --------
        # align_metric = score^alpha * IoU^beta selects positives. In YOLO26's
        # one2one branch topk2=1, so this metric picks THE single anchor per GT in
        # the branch that produces every prediction (the head is NMS-free). IoU is
        # a high-variance ranking signal for small boxes — one pixel of shift moves
        # it a lot on a 16 px object and barely at all on a 200 px one — so a
        # single global beta over-trusts IoU exactly where it is least reliable.
        # SCB interpolates beta by GT size: beta_small for objects at or below
        # beta_ref_px, self.beta for large ones.
        #   beta_small = None  -> exactly stock (no interpolation, no extra work)
        # On YOLOv12 this lever is weak: topk=10 means the exponents mostly reorder
        # a set that is kept anyway. topk2=1 is what makes it worth testing here.
        self.beta_small = None  # float, e.g. 2.0-4.0; None disables
        self.beta_ref_px = 64.0  # GT sqrt(area) at which beta reaches self.beta

        # --- SNT (Soft Negative Targets) — one2one only, inert at tau = 0 -------
        # topk2=1 makes every non-selected anchor a hard negative, and the number
        # of well-overlapping anchors discarded that way scales with object size.
        # See snt_soft_targets() for the full argument and the 0/52 vs 26/45
        # large-object evidence it is derived from.
        self.snt_tau = 0.0  # peak soft target; 0 disables
        self.snt_gamma = 2.0  # IoU exponent; >1 concentrates on the best runner-ups
        self.snt_min_iou = 0.5  # below this overlap an anchor stays a hard negative

    def scb_enabled(self) -> bool:
        """True when size-conditioned beta can change the assignment (no-op check)."""
        return self.beta_small is not None and float(self.beta_small) != float(self.beta)

    def _size_conditioned_beta(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Per-GT localization exponent, shape (b, n_max_boxes, 1).

        t = clamp(sqrt(area) / beta_ref_px, 0, 1) ; beta = beta_small + (beta - beta_small) * t

        gt_bboxes are in PIXEL units here (the caller passes anc_points * stride and
        gt_bboxes in image scale), so beta_ref_px is directly interpretable.
        """
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp(min=0)
        side = (wh[..., 0] * wh[..., 1]).clamp(min=0).sqrt()  # sqrt(area), pixels
        t = (side / max(float(self.beta_ref_px), 1e-6)).clamp(0.0, 1.0)
        b0 = float(self.beta_small)
        return (b0 + (float(self.beta) - b0) * t).unsqueeze(-1)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # Recover outside the except block: exiting it drops e.__traceback__, releasing the failed attempt's GPU
        # intermediates back to the allocator so the copy-back below can succeed
        LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
        result = self._forward(*(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)))
        return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        # SNT: soften the target for well-overlapping anchors that were NOT selected.
        # Inert when snt_tau == 0, where snt_soft_targets() returns None.
        snt = self.snt_soft_targets(overlaps, gt_labels, fg_mask, target_scores)
        if snt is not None:
            target_scores = torch.maximum(target_scores, snt)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def snt_enabled(self) -> bool:
        """True when soft negative targets can change the loss (preflight no-op check)."""
        return self.snt_tau > 0.0

    def snt_soft_targets(self, overlaps, gt_labels, fg_mask, target_scores):
        """Soft classification targets for high-IoU anchors that lost the selection.

        WHY THIS EXISTS. Across 52 YOLO26 configurations in this project, ZERO
        improved large-object AP. On YOLOv12 with the same interventions, 26 of 45
        did. Nothing tried so far — SWA, LB-TAL, SNL1, SCB, SBB, every architecture
        variant — moved that column, and none of them predicted it either.

        The one2one branch uses topk2 = 1: exactly ONE anchor per GT is positive and
        every other anchor is a hard negative with target 0. How many well-fitting
        anchors that discards scales with object size:

            8 px bag       few anchors overlap at all    ->  few high-IoU negatives
            250 px trolley hundreds overlap well         ->  hundreds of them

        So the branch that carries ~90% of the loss and produces every prediction
        (the head is NMS-free) is told, for large objects, that hundreds of nearly
        correct boxes are background. The damage scales with size by construction.
        YOLOv12 has no topk2: one head, ten positives, and the runner-up anchors are
        POSITIVES rather than hard negatives. That asymmetry is the only structural
        difference that tracks the 26/45 vs 0/52 split.

        WHAT THIS DOES. A non-selected anchor whose best overlap with any GT exceeds
        `snt_min_iou` gets a soft target in that GT's class channel:

            target = snt_tau * IoU ** snt_gamma          (elementwise max with the
                                                          existing target, so
                                                          positives are untouched)

            snt_tau = 0     stock: every non-selected anchor stays at 0
            snt_gamma > 1   concentrates the softening on the very best runner-ups

        NOT A REWEIGHTING. SWA, SNL1 and SBB all multiply an existing term. This
        changes what the TARGET IS, and makes it depend on a quantity the
        classification loss currently ignores — the assigner computes `overlaps` and
        discards them.

        ONE2ONE ONLY. In one2many (topk=10, topk2 unset) the runner-ups are already
        positives, so there is nothing to soften; E2ELoss installs this on the
        one2one branch alone.

        FALSIFIABLE. If the account is right, LARGE-object AP rises and small barely
        moves. Small up / large flat refutes it. Note also that a mechanism of this
        kind was tried on YOLOv12 (`snt`, +0.02 — a clean null); under this account
        that is the correct result, because v12 has no topk2 and therefore no
        runner-up-as-hard-negative problem.

        Args:
            overlaps (torch.Tensor): (b, n_max_boxes, A) IoU of each anchor to each GT.
            gt_labels (torch.Tensor): (b, n_max_boxes, 1) GT class indices.
            fg_mask (torch.Tensor): (b, A) selected-anchor mask.
            target_scores (torch.Tensor): (b, A, nc) current targets, for shape/dtype.

        Returns:
            (torch.Tensor | None): (b, A, nc) soft targets, or None if inert.
        """
        if not self.snt_enabled():
            return None
        best_ov, best_gt = overlaps.max(dim=1)  # (b, A) over GTs
        cls_idx = gt_labels.long().squeeze(-1).gather(1, best_gt)  # (b, A) class of that GT
        soft = float(self.snt_tau) * best_ov.clamp(0.0, 1.0).pow(float(self.snt_gamma))
        keep = (~fg_mask.bool()) & (best_ov >= float(self.snt_min_iou))
        soft = torch.where(keep, soft, torch.zeros_like(soft))
        out = torch.zeros_like(target_scores)
        out.scatter_(2, cls_idx.unsqueeze(-1), soft.unsqueeze(-1).to(out.dtype))
        return out

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]  # b, 1
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # SCB: per-GT localization exponent. Inert when beta_small is None, in
        # which case this is bit-identical to the stock scalar-beta line.
        beta = self._size_conditioned_beta(gt_bboxes) if self.scb_enabled() else self.beta
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # Count how many of the topk lists select each anchor; scatter_add_ accumulates duplicate indices in one pass
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        target_scores = target_scores * (fg_mask[:, :, None] > 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
        return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            # (b, n_max_boxes, topk2)
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class LevelBalancedTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with a PER-PYRAMID-LEVEL top-k budget instead of one global pooled draw.

    Ported from the YOLOv12 fork (LevelBalancedTaskAlignedAssigner in
    lossv2updated.py). Stock TAL ranks score^alpha * iou^beta over ALL anchors of
    a GT pooled across levels and keeps the global top-`topk`, which lets the
    level with the most geometric candidate supply monopolise a GT's slots. This
    assigner splits the budget per level and runs the top-k selection
    INDEPENDENTLY within each level, then unions the picks and caps the total at
    `topk` so it stays a RE-ALLOCATION of the same budget, not an inflation.

    Modes:
        'uniform' : every level gets ceil(topk / n_levels). No dataset constants.
        'fixed'   : per-stride budget taken from `level_topk`, e.g. {8:4,16:7,32:1}.

    =========================================================================
    DIFFERENCES FROM THE YOLOv12 VERSION — read before comparing numbers
    =========================================================================
    1. `select_topk_candidates` lost its `largest` argument in this codebase, so
       the override signature is (metrics, topk_mask=None). Selection is always
       largest=True here, which is what every recorded v6i run used anyway.
    2. `topk2`. The parent runs a SECOND top-k of size `topk2` inside
       `select_highest_overlaps`. In the E2E one2one branch topk2=1, which would
       collapse any per-level allocation down to a single anchor per GT and
       annihilate this mechanism. The loss therefore only installs this assigner
       on the branch where `topk2 is None` (one2many, or a non-E2E model). Do not
       "fix" that by enabling it everywhere — it would silently do nothing.
    3. Stride keys are normalised (see `_norm_level_topk`). A budget dict that
       round-trips through YAML/JSON arrives with STRING keys ('8'), while
       `torch.unique(strides).tolist()` yields floats (8.0). The v12 code indexed
       with the float and would silently fall back to min_level_k on every level,
       i.e. a no-op that still prints as active. Both forms are accepted here.
    4. `set_strides(stride_tensor)` must be called each forward pass. Without it
       this falls back to stock global top-k and warns once rather than failing
       silently.

    The v12 'proportional', 'balanced_capped' and 'size_cond' modes are NOT
    ported: none of them appear in the three configs being transferred, and
    size_cond's budgets were fitted to a candidate-supply footprint measured on
    a 3-level YOLOv12 assigner that no longer describes this model.
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
        level_topk_mode: str = "uniform",
        level_topk=None,
        min_level_k: int = 1,
        quality_gate: float = 0.0,
    ):
        """Initialize the level-balanced assigner with a per-level top-k budget."""
        super().__init__(
            topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, stride=stride, eps=eps, topk2=topk2
        )
        self.level_topk_mode = str(level_topk_mode)
        self.level_topk = self._norm_level_topk(level_topk)
        self.min_level_k = int(min_level_k)
        self.quality_gate = float(quality_gate)
        self._strides = None  # (A,) per-anchor stride in pixels, set each fwd pass
        self._warned = False
        self._printed = False

    @staticmethod
    def _norm_level_topk(level_topk):
        """Normalize a per-level budget to {int_stride: int_k}, accepting str/float keys.

        A dict written in a YAML/JSON config arrives as {'8': 4, ...}; a dict built
        in Python arrives as {8: 4, ...}. Indexing with the wrong type silently
        yields the min_level_k fallback on every level, which looks identical to a
        working run in the logs. Normalizing here makes that impossible.
        """
        if level_topk is None or isinstance(level_topk, (list, tuple)):
            return level_topk
        out = {}
        for k, v in dict(level_topk).items():
            try:
                out[int(float(k))] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def set_strides(self, stride_tensor):
        """Provide the per-anchor stride in pixels. stride_tensor: (A, 1) or (A,)."""
        self._strides = stride_tensor.detach().reshape(-1)

    def _per_level_budget(self, uniq_strides):
        """Return {stride: k_level} for the configured mode."""
        n = max(len(uniq_strides), 1)
        if self.level_topk_mode == "fixed" and self.level_topk is not None:
            if isinstance(self.level_topk, dict):
                return {s: int(self.level_topk.get(int(s), self.min_level_k)) for s in uniq_strides}
            return {  # list aligned to ascending strides
                s: int(self.level_topk[i]) if i < len(self.level_topk) else self.min_level_k
                for i, s in enumerate(uniq_strides)
            }
        # 'uniform' (and the fallback for a 'fixed' mode with no budget supplied)
        per = max(self.min_level_k, -(-self.topk // n))  # ceil division
        return {s: per for s in uniq_strides}

    def _print_once(self, level_ks, n_anchors):
        if not self._printed:
            LOGGER.info(
                f"LB-TAL active (per-level top-k) | mode={self.level_topk_mode} "
                f"budget={ {int(k): v for k, v in level_ks.items()} } "
                f"topk={self.topk} topk2={self.topk2} anchors={n_anchors}"
            )
            self._printed = True

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates WITHIN each pyramid level, union, cap at topk.

        Args:
            metrics (torch.Tensor): Alignment metric, shape (b, max_num_obj, h*w).
            topk_mask (torch.Tensor, optional): Valid-GT mask, shape (b, max_num_obj, topk).

        Returns:
            (torch.Tensor): Selected-candidate counts, shape (b, max_num_obj, h*w).
        """
        # Stock behaviour if strides are missing or stale — never fail silently.
        if self._strides is None or self._strides.numel() != metrics.shape[-1]:
            if not self._warned:
                LOGGER.warning(
                    "LB-TAL: per-anchor strides not set (or size mismatch) — falling back to "
                    "stock global top-k. set_strides() must be called each forward pass."
                )
                self._warned = True
            return super().select_topk_candidates(metrics, topk_mask=topk_mask)

        strides = self._strides.to(metrics.device)
        uniq = sorted(torch.unique(strides).tolist())
        level_ks = self._per_level_budget(uniq)
        self._print_once(level_ks, metrics.shape[-1])

        # Per-GT validity. The parent receives mask_gt expanded over topk; collapse
        # it back to (b, n, 1) so it applies to a per-level pick count instead.
        gt_valid = topk_mask[..., :1].bool() if topk_mask is not None else None

        count = torch.zeros_like(metrics, dtype=torch.int8)
        for s in uniq:
            k = int(level_ks.get(s, self.min_level_k))
            if k <= 0:
                continue
            lvl = (strides == s).view(1, 1, -1)  # (1, 1, A)
            k_eff = min(k, int(lvl.sum().item()))
            if k_eff <= 0:
                continue
            m_lvl = torch.where(lvl, metrics, torch.full_like(metrics, -1.0))
            tk_metrics, tk_idxs = torch.topk(m_lvl, k_eff, dim=-1, largest=True)  # (b, n, k_eff)
            valid = tk_metrics > self.eps  # drop empty slots and padded GTs
            if gt_valid is not None:
                valid = valid & gt_valid
            if self.quality_gate > 0.0:
                # Gate against this GT's best metric ACROSS ALL LEVELS, not the
                # level-local best: torch.topk returns sorted, so a level-local
                # reference keeps every level's top-1 unconditionally and prunes
                # only within-level spread. The globally best anchor always passes,
                # so with min_level_k >= 1 every real GT retains >= 1 positive.
                gmax = metrics.amax(dim=-1, keepdim=True).clamp_min(self.eps)
                valid = valid & (tk_metrics >= self.quality_gate * gmax)
            tk_idxs = tk_idxs.masked_fill(~valid, 0)
            ones = torch.ones_like(tk_idxs[:, :, :1], dtype=torch.int8)
            for j in range(k_eff):
                count.scatter_add_(-1, tk_idxs[:, :, j : j + 1], ones * valid[:, :, j : j + 1].to(torch.int8))

        count.masked_fill_(count > 1, 0)  # a level must not double-pick an anchor

        # Cap the union at topk, highest-metric first, so this re-allocates the
        # budget rather than inflating the positive count.
        total = count.sum(-1)  # (b, n)
        if bool((total > self.topk).any()):
            masked = torch.where(count > 0, metrics, torch.full_like(metrics, -1.0))
            _, keep_idx = torch.topk(masked, self.topk, dim=-1, largest=True)  # (b, n, topk)
            capped = torch.zeros_like(count)
            ones = torch.ones_like(keep_idx[:, :, :1], dtype=torch.int8)
            for j in range(self.topk):
                idx = keep_idx[:, :, j : j + 1]
                sel = torch.gather(count, -1, idx) > 0
                capped.scatter_add_(-1, idx, ones * sel.to(torch.int8))
            capped.masked_fill_(capped > 1, 0)
            count = torch.where((total > self.topk).unsqueeze(-1), capped, count)

        return count.to(metrics.dtype)


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride[0]
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # new_* factories inherit the device at runtime, unlike torch.arange which bakes it into traced graphs
        sx = feats[0].new_full((w,), 1, dtype=dtype).cumsum(0) - (1 - grid_cell_offset)  # shift x
        sy = feats[0].new_full((h,), 1, dtype=dtype).cumsum(0) - (1 - grid_cell_offset)  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
