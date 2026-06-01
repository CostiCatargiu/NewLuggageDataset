"""
Shape-Aware Task Aligned Assigner (Shape-TAL).

Extends the standard TAL alignment metric with an aspect ratio similarity term:

  Standard TAL:   align_metric = score^alpha * iou^beta
  Shape-TAL:      align_metric = score^alpha * iou^beta * shape_sim^gamma

where shape_sim measures how well the predicted box's aspect ratio matches
the ground truth box's aspect ratio:

  shape_sim = min(ar_pred, ar_gt) / max(ar_pred, ar_gt)

This is 1.0 when aspect ratios match perfectly, and approaches 0 when they
diverge (e.g., pred is square but GT is elongated).

Why this helps for weapon detection:
  - Knives are very elongated (AR ~5:1 to 10:1)
  - Pistols are compact (AR ~1.5:1 to 2:1)
  - "Other" varies wildly
  - Standard TAL doesn't differentiate: a square anchor and an elongated
    anchor with the same IoU get the same alignment score
  - Shape-TAL rewards anchors whose predicted boxes already match the
    target shape, leading to better specialization

Parameters:
  gamma (float): Power for shape similarity term. Default 1.0.
    - gamma=0: disabled (falls back to standard TAL)
    - gamma=0.5: mild shape preference
    - gamma=1.0: moderate shape preference
    - gamma=2.0: strong shape preference
  shape_min (float): Floor for shape_sim to avoid zeroing out anchors
    that have good IoU but wrong shape. Default 0.3.

Usage in loss_custom_git.py:
  Replace TaskAlignedAssigner with ShapeAwareTaskAlignedAssigner:

    from shape_tal import ShapeAwareTaskAlignedAssigner

    self.assigner = ShapeAwareTaskAlignedAssigner(
        topk=13,
        num_classes=self.nc,
        alpha=0.5,
        beta=4.0,
        gamma=1.0,        # shape similarity power
        shape_min=0.3,     # floor for shape_sim
    )
"""

import torch
import torch.nn as nn

from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.metrics import bbox_iou


class ShapeAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """
    Task Aligned Assigner with aspect ratio similarity.

    Alignment metric: score^alpha * iou^beta * shape_sim^gamma

    shape_sim = clamp(min(ar_pred, ar_gt) / max(ar_pred, ar_gt), min=shape_min)
    """

    def __init__(self, topk=13, num_classes=80, alpha=0.5, beta=4.0,
                 gamma=1.0, shape_min=0.3, eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.gamma = gamma
        self.shape_min = shape_min

    def _compute_aspect_ratios(self, bboxes):
        """
        Compute aspect ratios from xyxy bounding boxes.

        Args:
            bboxes: (N, 4) tensor in xyxy format

        Returns:
            (N,) tensor of aspect ratios (always >= 1.0)
        """
        w = (bboxes[:, 2] - bboxes[:, 0]).clamp(min=self.eps)
        h = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=self.eps)
        ar = w / h
        # Normalize so AR is always >= 1 (wider/taller doesn't matter, just the ratio)
        return torch.max(ar, 1.0 / ar)

    def _compute_shape_similarity(self, pd_bboxes_flat, gt_bboxes_flat):
        """
        Compute shape similarity between predicted and GT boxes.

        Args:
            pd_bboxes_flat: (M, 4) predicted boxes (masked)
            gt_bboxes_flat: (M, 4) GT boxes (masked)

        Returns:
            (M,) shape similarity scores in [shape_min, 1.0]
        """
        ar_pred = self._compute_aspect_ratios(pd_bboxes_flat)
        ar_gt = self._compute_aspect_ratios(gt_bboxes_flat)

        # Similarity: min(ar1, ar2) / max(ar1, ar2)
        # This is 1.0 when identical, approaches 0 when very different
        shape_sim = torch.min(ar_pred, ar_gt) / torch.max(ar_pred, ar_gt).clamp(min=self.eps)

        # Apply floor to prevent complete suppression
        shape_sim = shape_sim.clamp(min=self.shape_min)

        return shape_sim

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric with shape awareness.

        align_metric = score^alpha * iou^beta * shape_sim^gamma
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        shape_sims = torch.ones([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)

        # Get classification scores for GT class at each anchor
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        # Expand and mask boxes
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        # IoU
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Shape similarity
        if self.gamma > 0:
            shape_sims[mask_gt] = self._compute_shape_similarity(pd_boxes, gt_boxes)

        # Combined alignment metric
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        if self.gamma > 0:
            align_metric = align_metric * shape_sims.pow(self.gamma)

        return align_metric, overlaps
