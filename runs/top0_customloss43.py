# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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


# ═══════════════════════════════════════════════════════════════════════════════
# BASE LOSS COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════


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
                (F.binary_cross_entropy_with_logits(
                    pred_score.float(), gt_score.float(), reduction="none"
                ) * weight)
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
        pred_prob = pred.sigmoid()
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
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
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
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + \
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# LUGGAGE-SPECIALIZED LOSS COMPONENTS (ABLATION-OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════


class SizeAwareBboxLoss(nn.Module):
    """
    Bounding box loss with size-aware weighting for luggage detection.

    Ablation-optimized with:
        - Smooth continuous weighting via sigmoid interpolation
        - Dynamic image size handling for multi-scale training
        - Stronger small-object upweighting for reduced training data

    Ablation dataset context (30% train, 40% valid, 100% test):
        Train: 7,596 imgs | 33,600 annots | Small 25.0% | Med 57.9% | Large 17.1%
        Valid: 1,183 imgs |  5,621 annots | Small 34.0% | Med 57.2% | Large  8.5%

        Per-class density when present:
            Class 0 (backpack): avg 1.87/img | median 2 | max 8
            Class 1 (bag):      avg 1.59/img | median 1 | max 9
            Class 2 (trolley):  avg 3.05/img | median 3 | max 19

        Distribution: 32% sparse (1-2 obj), 35% moderate (3-5), 29% dense (6-10), 4% very dense (11+)
    """

    def __init__(self, reg_max=16, small_threshold=0.0025,
                 large_threshold=0.0225, size_weights=None,
                 use_smooth_weighting=True):
        """
        Initialize SizeAwareBboxLoss.

        Args:
            reg_max (int): Maximum regression range for DFL.
            small_threshold (float): Normalized area below which objects are 'small'.
                                     0.0025 = 32x32 at 640x640.
            large_threshold (float): Normalized area above which objects are 'large'.
                                     0.0225 = 96x96 at 640x640.
            size_weights (dict): Multiplicative weights for small/medium/large objects.
            use_smooth_weighting (bool): Use continuous sigmoid weighting instead of hard bins.
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        self.use_smooth_weighting = use_smooth_weighting

        # Ablation-tuned weights:
        #   - small 2.5: 30% subset means ~2,100 small objects vs ~8,400 full
        #   - large 0.7: large objects converge fast even with less data
        self.size_weights = size_weights or {
            'small': 2.5,
            'medium': 1.0,
            'large': 0.7,
        }

    def _compute_size_weight_hard(self, relative_area):
        """Hard bin assignment for size weighting."""
        size_weight = torch.ones_like(relative_area)
        size_weight[relative_area < self.small_threshold] = self.size_weights['small']
        size_weight[relative_area >= self.large_threshold] = self.size_weights['large']
        mid = (relative_area >= self.small_threshold) & (relative_area < self.large_threshold)
        size_weight[mid] = self.size_weights['medium']
        return size_weight

    def _compute_size_weight_smooth(self, relative_area):
        """
        Smooth continuous weighting via sigmoid interpolation.

        Maps area → weight smoothly: small areas get high weight,
        large areas get low weight, with no hard bin boundaries.
        Avoids gradient discontinuities at bin edges.
        """
        w_small = self.size_weights['small']
        w_large = self.size_weights['large']
        midpoint = (self.small_threshold + self.large_threshold) / 2.0
        steepness = 8.0 / (self.large_threshold - self.small_threshold + 1e-9)

        t = torch.sigmoid(steepness * (relative_area - midpoint))
        size_weight = w_small + (w_large - w_small) * t
        return size_weight

    def _compute_size_weight(self, target_bboxes_fg, stride_tensor_fg, imgsz):
        """
        Compute per-object size weighting based on relative area.

        Args:
            target_bboxes_fg (torch.Tensor): (num_fg, 4) stride-normalized xyxy coords.
            stride_tensor_fg (torch.Tensor): (num_fg, 1) stride per foreground anchor.
            imgsz (torch.Tensor): (2,) tensor [height, width] in pixels.
        """
        # Convert to absolute pixel coordinates
        abs_bboxes = target_bboxes_fg * stride_tensor_fg

        widths = abs_bboxes[:, 2] - abs_bboxes[:, 0]
        heights = abs_bboxes[:, 3] - abs_bboxes[:, 1]

        # Use actual image area — handles multi-scale training correctly
        img_area = imgsz[0] * imgsz[1]
        relative_area = (widths * heights) / (img_area + 1e-9)

        if self.use_smooth_weighting:
            size_weight = self._compute_size_weight_smooth(relative_area)
        else:
            size_weight = self._compute_size_weight_hard(relative_area)

        return size_weight.unsqueeze(-1)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor, imgsz):
        """
        Size-aware IoU + DFL loss.

        Args:
            pred_dist (torch.Tensor): Predicted distance distribution.
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            anchor_points (torch.Tensor): Anchor point coordinates.
            target_bboxes (torch.Tensor): Target bounding boxes (stride-normalized).
            target_scores (torch.Tensor): Target scores from assigner.
            target_scores_sum (float): Sum of target scores for normalization.
            fg_mask (torch.Tensor): Foreground mask.
            stride_tensor (torch.Tensor): Stride tensor (num_anchors, 1).
            imgsz (torch.Tensor): (2,) tensor [height, width] in pixels.
        """
        # Standard task-alignment weight
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Get stride for each foreground anchor
        stride_fg = stride_tensor.unsqueeze(0).expand(target_bboxes.shape[0], -1, -1)
        stride_fg = stride_fg[fg_mask]  # (num_fg, 1)

        # Size-aware weight using actual image dimensions
        size_weight = self._compute_size_weight(target_bboxes[fg_mask], stride_fg, imgsz)

        # Combined weight = task_alignment_weight * size_weight
        combined_weight = weight * size_weight

        # IoU loss
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * combined_weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                ) * combined_weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class AdaptiveClassWeighter:
    """
    Computes class weights from dataset statistics.

    For ablation datasets with reduced training data, class imbalance
    effects are amplified. Uses sqrt-dampened inverse-frequency weights.

    Ablation class distribution:
        Class 0 (backpack): 27.0% | 9,082 annots  → weight ~1.07
        Class 1 (bag):      22.0% | 7,390 annots  → weight ~1.19
        Class 2 (trolley):  51.0% | 17,128 annots → weight ~0.78

    Trolley has TRIPLE advantage: 51% frequency × 3.05/img density × 74% image presence
    """

    def __init__(self, class_counts, device, damping='sqrt'):
        """
        Args:
            class_counts (list/tensor): Per-class annotation counts.
            device: Torch device.
            damping (str): 'sqrt', 'log', or 'linear' dampening.
        """
        self.device = device

        counts = torch.tensor(class_counts, dtype=torch.float, device=device)
        inv_freq = 1.0 / (counts + 1e-9)

        if damping == 'sqrt':
            weights = torch.sqrt(inv_freq)
        elif damping == 'log':
            weights = torch.log1p(1.0 / (counts / counts.sum() + 1e-9))
        else:  # linear
            weights = inv_freq

        # Normalize so mean weight = 1.0
        self.weights = weights / weights.mean()

    def get_weights(self):
        """Return current class weights tensor."""
        return self.weights


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION LOSSES
# ═══════════════════════════════════════════════════════════════════════════════


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
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

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
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

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
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
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# LUGGAGE-SPECIALIZED DETECTION LOSS (ABLATION-OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════


class v8DetectionLossLuggage:
    """
    Specialized detection loss for luggage ablation dataset.

    Improvements over vanilla v8DetectionLoss:
        1. Class-weighted VFL — handles trolley triple-advantage
           (51% freq × 3.05/img density × 74% presence)
        2. Smooth size-aware IoU weighting — upweights small objects (25%)
           with continuous sigmoid instead of hard bins
        3. Varifocal Loss with per-channel labeling — better soft-label
           learning for bimodal density (32% sparse, 29% dense images)
        4. Label smoothing — prevents overconfident memorization with 30% data
        5. IoU-aware classification targets — aligns confidence with
           localization quality in multi-class scenes (68% multi-class images)

    Ablation dataset statistics:
        Train: 7,596 imgs | 33,600 annots | avg 4.42/img | mode=1 | P90=9
        Valid: 1,183 imgs |  5,621 annots | avg 4.75/img
        Test:    798 imgs |  3,890 annots | avg 4.87/img

        Classes: backpack=27.0% (1.87/img), bag=22.0% (1.59/img), trolley=51.0% (3.05/img)
        Sizes:   small=25.0%, medium=57.9%, large=17.1%
        Density: bimodal — 32% sparse (1-2 obj), 29% dense (6-10 obj)
        Co-occurrence: 31% single-class, 37% two-class, 31% all-three-class
    """

    def __init__(self, model, tal_topk=10):
        """
        Initialize the luggage-specialized detection loss.

        Args:
            model: De-paralleled YOLO model.
            tal_topk (int): Top-k candidates for task-aligned assigner.
        """
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]  # Detect() module

        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc  # should be 3 (backpack, bag, trolley)
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1

        # ──────────────────────────────────────────────────────────
        # 1. CLASS-FREQUENCY-AWARE WEIGHTS
        # ──────────────────────────────────────────────────────────
        # Ablation train counts: backpack=9082, bag=7390, trolley=17128
        # Trolley has triple advantage: frequency + density + presence
        ablation_class_counts = [9082.0, 7390.0, 17128.0]
        self.class_weighter = AdaptiveClassWeighter(
            ablation_class_counts, device, damping='sqrt'
        )
        self.class_weights = self.class_weighter.get_weights()
        # Result: backpack≈1.07, bag≈1.19, trolley≈0.78

        # BCE with no reduction — class weights applied manually
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # ──────────────────────────────────────────────────────────
        # 2. LABEL SMOOTHING
        # ──────────────────────────────────────────────────────────
        # With 30% data (7,596 images), model is prone to memorization.
        # Light smoothing regularizes without hurting learning signal.
        self.label_smoothing = 0.01

        # ──────────────────────────────────────────────────────────
        # 3. SIZE-AWARE BBOX LOSS
        # ──────────────────────────────────────────────────────────
        self.bbox_loss = SizeAwareBboxLoss(
            reg_max=m.reg_max,
            small_threshold=0.0025,    # 32²/640² — COCO small
            large_threshold=0.0225,    # 96²/640² — COCO large
            size_weights={
                'small': 2.5,   # ↑ stronger for ablation (only ~2,100 small objects)
                'medium': 1.0,  # baseline
                'large': 0.7,   # ↓ large objects converge fast
            },
            use_smooth_weighting=True,  # sigmoid interpolation, no hard bins
        ).to(device)

        # ──────────────────────────────────────────────────────────
        # 4. TASK-ALIGNED ASSIGNER
        # ──────────────────────────────────────────────────────────
        # Keep topk=10: 29% of images have 6-10 objects, trolley
        # stacks up to 19 per image. Dense tail needs high topk.
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
        )

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # ──────────────────────────────────────────────────────────
        # 5. VARIFOCAL LOSS
        # ──────────────────────────────────────────────────────────
        # Bimodal density: 32% of images have 1-2 objects (massive neg:pos ratio)
        # → strong focal modulation (gamma=2.0) needed to suppress easy negatives
        # VFL also helps in dense trolley images with soft label assignment
        self.use_varifocal = True
        self.vfl_alpha = 0.75
        self.vfl_gamma = 2.0

        # ──────────────────────────────────────────────────────────
        # 6. IoU-AWARE CLASSIFICATION TARGET
        # ──────────────────────────────────────────────────────────
        # 68% of images are multi-class with overlapping objects.
        # Aligning confidence with localization quality reduces false
        # positives in crowded scenes and noisy assignments.
        self.iou_aware_cls = True

        print(f"[LuggageLoss-Ablation] Initialized:")
        print(f"  Class weights (backpack/bag/trolley): {self.class_weights.cpu().numpy().round(4)}")
        print(f"  Size weights      : small=2.5, medium=1.0, large=0.7")
        print(f"  Smooth weighting  : True (sigmoid interpolation)")
        print(f"  Label smoothing   : {self.label_smoothing}")
        print(f"  Varifocal loss    : True (alpha={self.vfl_alpha}, gamma={self.vfl_gamma})")
        print(f"  IoU-aware cls     : {self.iou_aware_cls}")
        print(f"  TAL topk          : {tal_topk}")

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _apply_label_smoothing(self, target_scores):
        """
        Apply label smoothing to classification targets.

        With 7,596 training images (30% of original), the model is prone
        to memorizing training targets. Light smoothing prevents
        overconfident predictions without hurting learning signal.
        """
        if self.label_smoothing > 0:
            nc = target_scores.shape[-1]
            target_scores = target_scores * (1 - self.label_smoothing) + \
                           self.label_smoothing / nc
        return target_scores

    def _compute_cls_loss_weighted(self, pred_scores, target_scores, fg_mask,
                                   target_labels_for_fg, target_scores_sum):
        """
        Compute class-weighted classification loss.

        For foreground anchors, weight the loss by the inverse-frequency weight
        of the assigned ground-truth class. Background anchors get weight=1.0.
        """
        dtype = pred_scores.dtype

        # Apply label smoothing
        target_scores_smooth = self._apply_label_smoothing(target_scores)

        if self.use_varifocal:
            loss = self._varifocal_loss_weighted(
                pred_scores, target_scores_smooth, fg_mask, target_labels_for_fg
            )
        else:
            # Standard BCE with per-class weighting
            bs, num_anchors, nc = pred_scores.shape
            bce = self.bce(pred_scores, target_scores_smooth.to(dtype))

            # Build per-anchor weight tensor
            weight = torch.ones(bs, num_anchors, 1, device=self.device, dtype=dtype)

            if fg_mask.any():
                fg_class_weights = self.class_weights[target_labels_for_fg]
                weight[fg_mask] = fg_class_weights.unsqueeze(-1)

            loss = (bce * weight).sum()

        return loss / target_scores_sum

    def _varifocal_loss_weighted(self, pred_score, gt_score, fg_mask,
                                  target_labels_for_fg):
        """
        Class-weighted Varifocal Loss.

        Fixes from original implementation:
            - Uses per-channel label indicator (gt_score > 0) instead of
              broadcasting a single boolean across all nc channels
            - Proper class weight application per foreground anchor

        VFL focuses more on high-quality positive examples and less on easy
        negatives. With bimodal density (32% images have 1-2 objects),
        gamma=2.0 aggressively suppresses the massive easy-negative population.
        """
        alpha = self.vfl_alpha
        gamma = self.vfl_gamma

        # Per-class positive indicator — NOT broadcast from single bool
        # gt_score already has soft labels only in the assigned class channel
        label = (gt_score > 0).float()

        weight = alpha * pred_score.detach().sigmoid().pow(gamma) * (1 - label) + gt_score * label

        # Apply class-specific weighting to foreground anchors
        if fg_mask.any():
            bs, na, nc = pred_score.shape
            class_weight_map = torch.ones(bs, na, 1, device=self.device, dtype=pred_score.dtype)
            fg_class_weights = self.class_weights[target_labels_for_fg]
            class_weight_map[fg_mask] = fg_class_weights.unsqueeze(-1)
            weight = weight * class_weight_map

        with autocast(enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(
                    pred_score.float(), gt_score.float(), reduction="none"
                ) * weight
            ).mean(1).sum()

        return loss

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
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # ── Targets ──
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # ── Predicted boxes ──
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

        # ── IoU-aware classification target ──
        # Modulate classification targets by localization quality.
        # In 68% multi-class scenes, this aligns confidence with actual
        # box quality and reduces false positives from noisy assignments.
        if self.iou_aware_cls and fg_mask.any():
            with torch.no_grad():
                iou_quality = bbox_iou(
                    pred_bboxes[fg_mask].detach(),
                    (target_bboxes / stride_tensor)[fg_mask],
                    xywh=False, CIoU=False  # plain IoU for quality score
                ).squeeze(-1).clamp(0, 1)

                # Multiply the positive class channels by IoU quality
                fg_scores = target_scores[fg_mask]  # (num_fg, nc)
                iou_scale = iou_quality.unsqueeze(-1)  # (num_fg, 1)
                target_scores[fg_mask] = fg_scores * iou_scale

                # Recompute normalization after modulation
                target_scores_sum = max(target_scores.sum(), 1)

        # ── Extract class labels for foreground anchors ──
        if fg_mask.any():
            target_labels_for_fg = target_scores[fg_mask].argmax(dim=-1)
        else:
            target_labels_for_fg = torch.tensor([], device=self.device, dtype=torch.long)

        # ── Classification loss (class-weighted + VFL + label smoothing) ──
        loss[1] = self._compute_cls_loss_weighted(
            pred_scores, target_scores, fg_mask, target_labels_for_fg, target_scores_sum
        )

        # ── Bbox loss (size-aware + smooth weighting) ──
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor,
                imgsz,  # actual image size for correct area computation
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
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

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
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

        # Pboxes
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

        # Cls loss
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

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
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes,
                batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor,
        xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W).
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
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
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=proto.device
        )

        for i, single_i in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)
        ):
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
                    mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# POSE LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = (
            torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose
            else torch.ones(nkpt, device=self.device) / nkpt
        )
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
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

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(
            anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape)
        )

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
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
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
                fg_mask, target_gt_idx, keypoints, batch_idx,
                stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx,
        stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

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

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
            device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(
                -1, -1, keypoints.shape[1], keypoints.shape[2]
            )
        )

        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = (
                gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3
                else torch.full_like(gt_kpt[..., 0], True)
            )
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


# ═══════════════════════════════════════════════════════════════════════════════
# OBB (ORIENTED BOUNDING BOX) LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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
        """Calculate and return the loss for the YOLO model."""
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

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
            )
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
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

        # Pboxes
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

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

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
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return torch.cat(
            (dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1
        )


# ═══════════════════════════════════════════════════════════════════════════════
# E2E (END-TO-END) DETECTION LOSS
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# E2E LUGGAGE-SPECIALIZED DETECTION LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class E2EDetectLossLuggage:
    """
    End-to-end detection loss specialized for luggage ablation dataset.

    Uses v8DetectionLossLuggage for both one-to-many and one-to-one branches,
    applying class weighting, size-aware IoU loss, varifocal loss,
    label smoothing, and IoU-aware classification targets.
    """

    def __init__(self, model):
        """Initialize with luggage-specialized losses for both branches."""
        self.one2many = v8DetectionLossLuggage(model, tal_topk=10)
        self.one2one = v8DetectionLossLuggage(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
