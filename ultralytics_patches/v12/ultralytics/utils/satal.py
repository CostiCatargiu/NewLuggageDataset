# ultralytics/utils/satal.py
"""
Scale-Adaptive Task-Aligned Assigner (SA-TAL)

Drop-in replacement for TaskAlignedAssigner.
Dynamically adjusts alpha, beta, and topk per ground-truth object
based on its normalized area.

Small objects get:  high alpha (trust classification), low beta (reduce IoU noise), more topk
Large objects get:  standard alpha, standard beta, base topk
Transitions are smooth (linear interpolation).
"""

import torch
import torch.nn as nn

from ultralytics.utils.tal import TaskAlignedAssigner


class ScaleAdaptiveTaskAlignedAssigner(TaskAlignedAssigner):

    def __init__(
        self,
        topk=13,
        num_classes=80,
        alpha=1.0,
        beta=6.0,
        eps=1e-9,
        # SA-TAL parameters
        alpha_small=1.5,
        beta_small=3.0,
        alpha_large=1.0,
        beta_large=6.0,
        small_area_thresh=0.002500,
        large_area_thresh=0.022500,
        topk_small_factor=1.5,
    ):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)

        self.alpha_small = alpha_small
        self.beta_small = beta_small
        self.alpha_large = alpha_large
        self.beta_large = beta_large
        self.small_area_thresh = small_area_thresh
        self.large_area_thresh = large_area_thresh
        self.topk_small_factor = topk_small_factor
        self.topk_max = int(topk * topk_small_factor)

        # Will be set externally by the loss function each forward pass
        self._imgsz = None

        self._printed = False

    def _print_once(self):
        if not self._printed:
            print(
                f"\n{'=' * 55}\n"
                f"  SA-TAL Active\n"
                f"{'=' * 55}\n"
                f"  alpha:  {self.alpha_small} (small) -> {self.alpha_large} (large)\n"
                f"  beta:   {self.beta_small} (small) -> {self.beta_large} (large)\n"
                f"  topk:   {self.topk_max} (small) -> {self.topk} (large)\n"
                f"  thresh: small<{self.small_area_thresh}, large>{self.large_area_thresh}\n"
                f"{'=' * 55}\n"
            )
            self._printed = True

    def set_imgsz(self, imgsz):
        """
        Set image size for accurate area normalization.
        Called by v8DetectionLoss each forward pass.

        Args:
            imgsz: tensor of shape (2,) — (height, width) in pixels
        """
        self._imgsz = imgsz

    def _compute_gt_scale_t(self, gt_bboxes, mask_gt):
        """
        Compute per-GT scale interpolation factor.

        t = 0.0 -> small  -> alpha_small, beta_small, topk_max
        t = 1.0 -> large  -> alpha_large, beta_large, topk

        Args:
            gt_bboxes: (bs, n_max_boxes, 4) xyxy absolute pixel coords
            mask_gt: (bs, n_max_boxes, 1) valid GT mask

        Returns:
            t: (bs, n_max_boxes) in [0, 1]
        """
        widths = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)
        heights = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)
        areas = widths * heights

        # Normalize by image area
        if self._imgsz is not None:
            img_area = self._imgsz[0] * self._imgsz[1]  # h * w
            img_area = img_area.clamp(min=1.0)
        else:
            # Fallback: estimate from max GT coordinate
            img_size = gt_bboxes.reshape(gt_bboxes.shape[0], -1).amax(dim=-1, keepdim=True).clamp(min=1.0)
            img_area = img_size ** 2

        norm_areas = areas / (img_area + 1e-8)

        t = (norm_areas - self.small_area_thresh) / (
            self.large_area_thresh - self.small_area_thresh + 1e-8
        )
        t = t.clamp(0.0, 1.0)
        t = t * mask_gt.squeeze(-1)

        return t

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """SA-TAL forward pass."""
        self._print_once()

        scale_t = self._compute_gt_scale_t(gt_bboxes, mask_gt)

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt,
            scale_t=scale_t,
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, scale_t=None):
        """Get positive mask with scale-adaptive metrics."""
        if scale_t is None:
            return super().get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)

        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes,
            mask_in_gts * mask_gt,
            scale_t=scale_t,
        )

        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
            scale_t=scale_t,
        )

        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, scale_t=None):
        """
        Alignment metric with per-GT alpha/beta.

        Standard:  metric = score^alpha * iou^beta
        SA-TAL:    metric_j = score^alpha_j * iou^beta_j
        """
        if scale_t is None:
            return super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        # Index construction — keep on CPU like original to avoid issues
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)

        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # ── Per-GT alpha/beta ──────────────────────────────
        per_gt_alpha = self.alpha_small + (self.alpha_large - self.alpha_small) * scale_t
        per_gt_beta = self.beta_small + (self.beta_large - self.beta_small) * scale_t

        alpha = per_gt_alpha.unsqueeze(-1)   # (bs, n_max_boxes, 1)
        beta = per_gt_beta.unsqueeze(-1)     # (bs, n_max_boxes, 1)

        # Safe power operation — avoid 0^fractional
        align_metric = bbox_scores.clamp(min=1e-9).pow(alpha) * overlaps.clamp(min=1e-9).pow(beta)

        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None, scale_t=None):
        """
        Top-k selection with per-GT adaptive k.

        Small objects: topk_max candidates
        Large objects: base topk candidates
        """
        if scale_t is None:
            return super().select_topk_candidates(metrics, largest=largest, topk_mask=topk_mask)

        num_anchors = metrics.shape[-1]
        effective_topk = min(self.topk_max, num_anchors)

        # Select topk_max for everyone
        topk_metrics, topk_idxs = torch.topk(metrics, effective_topk, dim=-1, largest=largest)

        # Per-GT actual k: small → topk_max, large → topk
        per_gt_k_float = self.topk_max - (self.topk_max - self.topk) * scale_t
        per_gt_k = per_gt_k_float.long().clamp(min=self.topk, max=effective_topk)

        # Rank-based mask: keep first per_gt_k candidates per GT
        rank_idx = torch.arange(effective_topk, device=metrics.device).view(1, 1, -1)
        adaptive_mask = rank_idx < per_gt_k.unsqueeze(-1)

        # Extend topk_mask to effective_topk size if needed
        if topk_mask is not None:
            if topk_mask.shape[-1] < effective_topk:
                pad = topk_mask.new_ones(
                    topk_mask.shape[0],
                    topk_mask.shape[1],
                    effective_topk - topk_mask.shape[-1],
                )
                topk_mask = torch.cat([topk_mask, pad], dim=-1)
            else:
                topk_mask = topk_mask[..., :effective_topk]
            adaptive_mask = adaptive_mask & topk_mask
        else:
            fallback_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_metrics)
            adaptive_mask = adaptive_mask & fallback_mask

        # Zero out indices for masked positions
        topk_idxs = topk_idxs.clone()
        topk_idxs[~adaptive_mask] = 0

        # Scatter to full anchor dimension
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=metrics.device)
        for k in range(effective_topk):
            valid = adaptive_mask[:, :, k:k + 1].to(torch.int8)
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones * valid)

        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)
