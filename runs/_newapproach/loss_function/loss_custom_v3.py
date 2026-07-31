# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

# =============================================================================
# CUSTOM v3 — stock Ultralytics loss + three gated mechanisms.
# =============================================================================
# Base: loss_original_stock.py, copied VERBATIM. Every addition below is behind
# a switch that defaults to OFF, and the stock code paths are untouched, so with
# no hyp keys set this file is bit-identical to stock. That is the property the
# previous rebuild lost: it re-implemented the normalisation as
# `weight.sum().clamp(min=1e-9)` instead of `max(target_scores.sum(), 1)`, which
# inflated box+dfl early in training and cost -1.68 mAP50-95 (-15.44 on large)
# at "neutral" settings. Nothing here rewrites stock arithmetic.
#
# MECHANISMS
#   use_ardfl  per-edge DFL weights  (w,h,w,h)
#   use_peu    per-edge attenuation by the DFL distribution's own variance
#   use_lba    level-balanced assignment prior on the TAL alignment metric
#
# Only one of use_ardfl / use_peu may be on at a time — both rewrite the same
# per-edge DFL reduction, so together they are unattributable.
# =============================================================================

_EPOCH = {"epoch": 0, "total": 0, "set": False}


def set_epoch(epoch, total=None):
    """Call from an on_train_epoch_start callback so PEU warmup knows the epoch."""
    _EPOCH["epoch"] = int(epoch)
    if total:
        _EPOCH["total"] = int(total)
    _EPOCH["set"] = True


def attach_epoch_callback(model, total_epochs=None):
    """Convenience: wire set_epoch into an Ultralytics model."""
    def _cb(trainer):
        set_epoch(trainer.epoch, getattr(trainer, "epochs", total_epochs))
    model.add_callback("on_train_epoch_start", _cb)


class CustomLossCfg:
    """Every knob this file reads. Defaults reproduce stock exactly."""

    def __init__(self, h):
        g = lambda k, d: getattr(h, k, d)  # noqa: E731

        # ---- AR-DFL ---------------------------------------------------------
        # Per-edge DFL weights, order (left, top, right, bottom).
        # DIRECTION: an e-px error costs e/w on a width edge and e/h on a height
        # edge, ratio h/w = 2.69 on this dataset -> the WIDTH edges are ~2.7x
        # more IoU-sensitive and should receive MORE weight. Earlier runs used
        # h=1.5/w=0.75, i.e. the opposite, and measured -0.46. Defaults below are
        # the corrected direction. Weights are mean-normalised so total DFL
        # magnitude is unchanged and any gain is not just "more DFL gain".
        self.use_ardfl = bool(g("use_ardfl", False))
        self.ardfl_w_weight = float(g("ardfl_w_weight", 1.5))
        self.ardfl_h_weight = float(g("ardfl_h_weight", 0.75))

        # ---- PEU-DFL --------------------------------------------------------
        # DFL emits a distribution per edge; its variance is a free aleatoric
        # uncertainty estimate. L_e <- L_e * exp(-beta * s_e) + lambda * s_e.
        #
        # peu_norm_by_mu: variance in BIN units grows with target magnitude, so
        # raw var attenuates LARGE objects hardest — measured as monotonic
        # large-object damage (beta=1.0 cost -3.03 mAP50-95_large). Dividing by
        # mu^2 makes the signal scale-free. Default True (the fix).
        #
        # VALID COMBINATIONS ONLY (enforced below):
        #   detach=True  -> lambda MUST be 0. Nothing can game a detached weight.
        #   detach=False -> lambda > 0. True Kendall form, self-balancing.
        # detach=True with lambda>0 collapses: the only gradient reaching the
        # variance is lambda*log(var), which drives var to the floor unopposed.
        # Measured at -4.33 and -7.31.
        self.use_peu = bool(g("use_peu", False))
        self.peu_beta = float(g("peu_beta", 0.5))
        self.peu_lambda = float(g("peu_lambda", 0.0))
        self.peu_detach = bool(g("peu_detach", True))
        self.peu_norm_by_mu = bool(g("peu_norm_by_mu", True))
        self.peu_warmup_epochs = int(g("peu_warmup_epochs", 5))
        self.peu_min_var = float(g("peu_min_var", 0.05))
        self.peu_w_clip = float(g("peu_w_clip", 3.0))

        # ---- LBA ------------------------------------------------------------
        # Soft scale-matching prior on the TAL alignment metric:
        #   octaves = log2( size / (stride * ref_cells) )
        #   prior   = exp( -octaves^2 / (2 sigma^2) );  align <- align * prior^strength
        #
        # lba_size_axis: 'max' uses max(w,h), 'geom' uses sqrt(w*h). On a 94%-tall
        # dataset the geometric mean under-reads extent by ~1.67x, so 'max' is the
        # default. lba_size_gate_px: apply the prior ONLY to GTs above this size —
        # the measured effect was large +3.99 / medium -1.42, so gating keeps the
        # gain and drops the cost. 0 = ungated (original behaviour).
        self.use_lba = bool(g("use_lba", False))
        self.lba_strength = float(g("lba_strength", 1.0))
        self.lba_ref_cells = float(g("lba_ref_cells", 4.5))
        self.lba_sigma = float(g("lba_sigma", 1.0))
        self.lba_size_axis = str(g("lba_size_axis", "max"))
        self.lba_size_gate_px = float(g("lba_size_gate_px", 0.0))
        self.lba_log = bool(g("lba_log", True))

        self._validate()

    def _validate(self):
        if self.use_peu and self.use_ardfl:
            raise ValueError(
                "use_peu and use_ardfl both set — both rewrite the per-edge DFL "
                "reduction, so the comparison would be unattributable."
            )
        if self.use_ardfl and self.ardfl_w_weight == self.ardfl_h_weight:
            raise ValueError(
                f"use_ardfl=True but w==h=={self.ardfl_w_weight} — that is stock DFL."
            )
        if self.ardfl_w_weight < 0 or self.ardfl_h_weight < 0:
            raise ValueError("ardfl weights must be >= 0")
        if self.use_peu:
            if self.peu_beta < 0 or self.peu_lambda < 0:
                raise ValueError("peu_beta / peu_lambda must be >= 0")
            if self.peu_beta == 0 and self.peu_lambda == 0:
                raise ValueError("use_peu=True but beta and lambda are both 0 — that is stock.")
            if self.peu_detach and self.peu_lambda > 0:
                raise ValueError(
                    "peu_detach=True with peu_lambda>0 is the COLLAPSING config "
                    "(measured -4.33 / -7.31). Use detach=True with lambda=0, or "
                    "detach=False with lambda>0."
                )
            if not self.peu_detach and self.peu_lambda == 0:
                raise ValueError(
                    "peu_detach=False with peu_lambda=0 lets the net inflate variance "
                    "to cut its own loss. Set peu_lambda > 0."
                )
            if self.peu_min_var <= 0:
                raise ValueError("peu_min_var must be > 0")
            if self.peu_w_clip < 1.0:
                raise ValueError("peu_w_clip must be >= 1.0")
        if self.use_lba:
            if self.lba_strength <= 0:
                raise ValueError("use_lba=True but lba_strength<=0 — that is stock TAL.")
            if self.lba_sigma <= 0 or self.lba_ref_cells <= 0:
                raise ValueError("lba_sigma / lba_ref_cells must be > 0")
            if self.lba_size_axis not in ("max", "geom"):
                raise ValueError("lba_size_axis must be 'max' or 'geom'")

    def is_stock(self):
        return not (self.use_ardfl or self.use_peu or self.use_lba)

    def banner(self):
        out = ["=" * 62, "  CUSTOM v3 loss  (stock + AR-DFL / PEU / LBA)", "=" * 62,
               f"  neutral (== stock):   {self.is_stock()}"]
        out.append(f"  AR-DFL:               {self.use_ardfl}")
        if self.use_ardfl:
            out.append(f"    w / h weight:       {self.ardfl_w_weight} / {self.ardfl_h_weight}")
        out.append(f"  PEU-DFL:              {self.use_peu}")
        if self.use_peu:
            out.append(f"    beta / lambda:      {self.peu_beta} / {self.peu_lambda}")
            out.append(f"    detach / norm_mu:   {self.peu_detach} / {self.peu_norm_by_mu}")
            out.append(f"    warmup epochs:      {self.peu_warmup_epochs}"
                       + ("" if _EPOCH["set"] else "   [!! epoch tracking NOT attached"
                                                   " — attenuating from step 0]"))
        out.append(f"  LBA:                  {self.use_lba}")
        if self.use_lba:
            out.append(f"    strength / sigma:   {self.lba_strength} / {self.lba_sigma}")
            out.append(f"    ref_cells / axis:   {self.lba_ref_cells} / {self.lba_size_axis}")
            out.append(f"    size gate (px):     {self.lba_size_gate_px or 'off'}")
        out.append("=" * 62)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# PEU helpers
# ---------------------------------------------------------------------------
_PEU_STATE = {"n": 0, "var": None, "w": None}


def _dfl_edge_moments(pred_dist, reg_max):
    """(N*4, reg_max) logits -> per-edge (mu, var) in BIN units, each (N, 4)."""
    p = pred_dist.softmax(-1)
    idx = torch.arange(reg_max, device=pred_dist.device, dtype=p.dtype)
    mu = (p * idx).sum(-1)
    var = (p * (idx.unsqueeze(0) - mu.unsqueeze(-1)) ** 2).sum(-1)
    return mu.view(-1, 4), var.view(-1, 4)


def _peu_weights(mu, var, cfg, warmed):
    """Mean-normalised per-edge attenuation weights, and the log-variance term.

    Mean-normalisation is deliberate: PEU must REDISTRIBUTE DFL weight across
    edges, not scale the total, or a gain is confounded with a larger dfl gain.
    """
    v = var.clamp(min=cfg.peu_min_var)
    if cfg.peu_norm_by_mu:
        # scale-free: raw bin-unit variance grows with target magnitude, so it
        # would attenuate large objects rather than uncertain edges.
        v = v / mu.detach().clamp(min=1.0) ** 2
    s = torch.log(v)
    if not warmed:
        return torch.ones_like(s), s
    s_w = s.detach() if cfg.peu_detach else s
    # Centre the log-variance BEFORE exponentiating. peu_w_clip is meant to bound
    # the RELATIVE spread across edges; without centring, peu_norm_by_mu divides
    # every edge by ~mu^2, pushes every weight past the clip together, and PEU
    # silently becomes a no-op (all weights -> 1.000). Verified numerically.
    s_w = s_w - s_w.mean()
    w = torch.exp(-cfg.peu_beta * s_w)
    k = cfg.peu_w_clip
    w = w.clamp(min=1.0 / k, max=k)          # bound the spread BEFORE normalising
    return w / w.mean().clamp(min=1e-9), s


def _peu_track(var, w):
    with torch.no_grad():
        st = _PEU_STATE
        for key, val in (("var", var), ("w", w)):
            m = val.mean(0)
            st[key] = m if st[key] is None else st[key] + m
        st["n"] += 1


def peu_report(reset=True):
    """Mean per-edge variance and weight since the last call."""
    st = _PEU_STATE
    if not st["n"]:
        return None
    names = ("left", "top", "right", "bottom")
    out = {"var": dict(zip(names, (st["var"] / st["n"]).tolist())),
           "weight": dict(zip(names, (st["w"] / st["n"]).tolist()))}
    if reset:
        st["n"], st["var"], st["w"] = 0, None, None
    return out


# ---------------------------------------------------------------------------
# LBA helpers
# ---------------------------------------------------------------------------
_LBA_STATE = {"n": 0, "fg": None, "strides": None}


def _lba_prior(gt_bboxes, stride_per_anchor, cfg, eps=1e-9):
    """Soft scale-matching prior, (bs, n_max_boxes, n_anchors), in (0, 1].

    gt_bboxes         : (bs, n, 4) xyxy in PIXELS
    stride_per_anchor : (n_anchors,) pixels per cell at that anchor's level
    """
    w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=eps)
    h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=eps)
    if cfg.lba_size_axis == "max":
        size = torch.maximum(w, h)
    else:
        size = (w * h).sqrt()
    size = size.unsqueeze(-1)                                    # (bs, n, 1)
    nominal = (stride_per_anchor * cfg.lba_ref_cells).view(1, 1, -1)
    octaves = torch.log2(size / nominal.clamp(min=eps))
    prior = torch.exp(-(octaves ** 2) / (2.0 * cfg.lba_sigma ** 2))
    if cfg.lba_size_gate_px > 0:
        # measured: the prior helps large objects (+3.99) and hurts medium
        # (-1.42). Gating to large GTs keeps the gain and drops the cost.
        prior = torch.where(size > cfg.lba_size_gate_px, prior, torch.ones_like(prior))
    return prior


def _lba_track(fg_mask, stride_per_anchor):
    with torch.no_grad():
        st = _LBA_STATE
        uniq = torch.unique(stride_per_anchor)
        counts = torch.stack([((stride_per_anchor.view(1, -1) == s) & fg_mask).sum()
                              for s in uniq]).float()
        st["strides"] = uniq.tolist()
        st["fg"] = counts if st["fg"] is None else st["fg"] + counts
        st["n"] += 1


def lba_report(reset=True):
    """Foreground share per FPN level since the last call."""
    st = _LBA_STATE
    if not st["n"] or st["fg"] is None:
        return None
    tot = st["fg"].sum().clamp(min=1)
    out = {int(s): {"fg": int(v.item()), "share": float((v / tot).item())}
           for s, v in zip(st["strides"], st["fg"])}
    if reset:
        st["n"], st["fg"] = 0, None
    return out


class _LBAMixin:
    """Level prior, mixed in ahead of TaskAlignedAssigner.

    Concrete classes are defined at MODULE level, not built in a factory —
    Ultralytics pickles the criterion (and its assigner) on every checkpoint
    save, and a locally-defined class cannot be pickled.
    """

    lba_cfg = None
    stride_tensor = None
    _warned = False

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, *a, **kw):
        align, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, *a, **kw)
        s, cfg = self.stride_tensor, self.lba_cfg
        if s is None or cfg is None:
            if not self._warned:
                print("[LBA] WARNING stride_tensor not set — prior NOT applied")
                self._warned = True
            return align, overlaps
        prior = _lba_prior(gt_bboxes, s.view(-1).to(align.dtype), cfg)
        return align * prior.pow(cfg.lba_strength).to(align.dtype), overlaps


class LevelBalancedTaskAlignedAssigner(_LBAMixin, TaskAlignedAssigner):
    """Stock TAL + level prior."""


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

    def per_edge(self, pred_dist, target):
        """Identical maths to __call__ but WITHOUT the mean over the 4 edges,
        so a per-edge weighting can be applied before reduction. Returns (N, 4)
        in (left, top, right, bottom) order."""
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        )


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16, cfg=None):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.cfg = cfg

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            cfg = self.cfg
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)

            if cfg is not None and cfg.use_ardfl:
                # ---- AR-DFL: fixed per-edge weights, mean-normalised --------
                pd = pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max)
                per_edge = self.dfl_loss.per_edge(pd, target_ltrb[fg_mask])       # (N,4)
                ew = torch.tensor(
                    [cfg.ardfl_w_weight, cfg.ardfl_h_weight,
                     cfg.ardfl_w_weight, cfg.ardfl_h_weight],
                    device=per_edge.device, dtype=per_edge.dtype,
                ).view(1, 4)
                ew = ew / ew.mean()                                              # total DFL unchanged
                loss_dfl = (per_edge * ew).mean(-1, keepdim=True) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum

            elif cfg is not None and cfg.use_peu:
                # ---- PEU: attenuate by the edge's own distribution variance --
                pd = pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max)
                per_edge = self.dfl_loss.per_edge(pd, target_ltrb[fg_mask])       # (N,4)
                mu, var = _dfl_edge_moments(pd, self.dfl_loss.reg_max)            # (N,4)
                # At init the bin distribution is near-uniform, so its variance
                # carries no signal and would suppress every edge equally.
                warmed = (not _EPOCH["set"]) or (_EPOCH["epoch"] >= cfg.peu_warmup_epochs)
                w_edge, s = _peu_weights(mu, var, cfg, warmed)
                _peu_track(var, w_edge)
                loss_dfl = (per_edge * w_edge).mean(-1, keepdim=True) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
                if cfg.peu_lambda > 0 and warmed:
                    reg = s.mean(-1, keepdim=True)
                    loss_dfl = loss_dfl + cfg.peu_lambda * (reg * weight).sum() / target_scores_sum

            else:
                # ---- STOCK, untouched ---------------------------------------
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
    """Criterion class for computing training losses."""

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

        self.cfg = CustomLossCfg(h)

        if self.cfg.use_lba:
            self.assigner = LevelBalancedTaskAlignedAssigner(
                topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
            )
            self.assigner.lba_cfg = self.cfg
            self.assigner.stride_tensor = None
        else:
            self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

        self.bbox_loss = BboxLoss(m.reg_max, cfg=self.cfg).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        print(self.cfg.banner())

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
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # LBA needs the per-anchor stride to know which level a candidate is on.
        if self.cfg.use_lba:
            self.assigner.stride_tensor = stride_tensor

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Track level occupancy for ANY assigner — without the baseline share
        # there is nothing to compare an LBA run against.
        if self.cfg.lba_log:
            _lba_track(fg_mask, stride_tensor)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

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

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

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


# NOTE: a second, byte-equivalent E2EDetectLoss used to sit here and shadowed the
# one defined above (audit bug B4). Removed — the baseline's definition stands.

