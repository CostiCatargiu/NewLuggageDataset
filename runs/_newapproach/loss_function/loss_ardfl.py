# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# loss_ardfl.py  —  AR-DFL loss (SATAL + SWA + NWD + Aspect-Ratio-aware DFL)
# =============================================================================
# THIS IS THE FILE TO DEPLOY. Copy it to your ultralytics install as:
#     ultralytics/utils/loss.py
# (identical content to loss_function/loss.py; renamed so it is obvious which
#  file carries the AR-DFL contribution and pairs with run_ardfl_ablation.py.)
#
# It is a superset of the stock loss: with no hyp flags set it reproduces stock
# Ultralytics loss EXACTLY. AR-DFL activates only when use_ardfl=True (+ the
# ardfl_* knobs) are passed via model.train(...), which run_ardfl_ablation.py
# does for you.
#
# WHAT IS NEW HERE (vs every prior loss experiment):
#   AR-DFL = Aspect-Ratio-aware Distribution Focal Loss.
#   ~60 prior configs reweighted WHICH samples matter or swapped the IoU flavour
#   and all plateaued (the R9/R10 changelogs proved gradient rescaling is null).
#   AR-DFL instead changes the BOX REPRESENTATION: stock DFL quantizes all four
#   box edges into the same 16-bin grid, but this dataset is 94% tall (mean AR
#   2.69), so the HEIGHT edges (top,bottom) carry the large, hard-to-localise
#   range while the WIDTH edges (left,right) are short/easy. AR-DFL raises the
#   per-edge DFL weight on the height edges (and optionally sharpens only the
#   height-edge distributions), spending regression capacity where the residual
#   error — the 25pt mAP50->mAP50-95 gap — actually lives. It is orthogonal to
#   NWD (keep NWD on) and needs NO architecture change (same reg_max, same head).
#
#   bbox2dist edge order = (left, top, right, bottom):
#       width  edges = columns [0, 2]
#       height edges = columns [1, 3]
#
#   Knobs (all default OFF -> stock behaviour):
#       use_ardfl, ardfl_h_weight, ardfl_w_weight,
#       ardfl_ar_gate, ardfl_ar_thresh, ardfl_entropy, ardfl_entropy_w
#
# =============================================================================
# INHERITED (SATAL + SWA + NWD rebuild, v2) — every silent-failure mode from v1
# removed. NEUTRAL CONFIG (reproduces stock Ultralytics loss):
#   use_satal=False, swa_alpha=0.0, swa_boost=1.0, use_nwd=False,
#   use_class_weights=False, use_loss_clip=False, use_ardfl=False
#
# EPOCH TRACKING — REQUIRED for any schedule to work:
#   from ultralytics.utils.loss import attach_epoch_tracking
#   model = YOLO(...)
#   attach_epoch_tracking(model)      # <-- do this BEFORE model.train(...)
# =============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import OKS_SIGMA

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


# =============================================================================
# EPOCH TRACKING
# =============================================================================
_EPOCH_STATE = {"epoch": 0, "total": 0, "ever_set": False, "warned": False}


def set_epoch(epoch, total_epochs=None):
    """Manually set the current epoch (use if you are not using the callback)."""
    _EPOCH_STATE["epoch"] = int(epoch)
    if total_epochs is not None:
        _EPOCH_STATE["total"] = int(total_epochs)
    _EPOCH_STATE["ever_set"] = True


def _epoch_callback(trainer):
    """Ultralytics callback: on_train_epoch_start."""
    set_epoch(getattr(trainer, "epoch", 0), getattr(getattr(trainer, "args", None), "epochs", None))


def attach_epoch_tracking(model):
    """
    Register epoch tracking on a YOLO model. Call BEFORE model.train().

    Without this, all epoch-dependent schedules (swa alpha annealing, loss
    clipping annealing) are DISABLED and a one-time warning is printed.
    """
    model.add_callback("on_train_epoch_start", _epoch_callback)
    return model


def _get_progress(total_fallback):
    """Return (progress in [0,1], schedules_active). Safe when epoch is unwired."""
    if not _EPOCH_STATE["ever_set"]:
        if not _EPOCH_STATE["warned"]:
            print(
                "\n[SATAL-SWA] WARNING: epoch tracking is not attached.\n"
                "  Schedules (swa alpha annealing, loss clip annealing) are DISABLED;\n"
                "  constant mid-range values will be used instead.\n"
                "  To enable: attach_epoch_tracking(model) before model.train(...)\n"
            )
            _EPOCH_STATE["warned"] = True
        return None, False
    total = _EPOCH_STATE["total"] or total_fallback or 1
    return min(max(_EPOCH_STATE["epoch"] / max(total, 1), 0.0), 1.0), True


# =============================================================================
# CONFIG OBJECT — single source of truth
# =============================================================================


class SataLSwaConfig:
    """Parses all loss hyperparameters exactly once. No defaults live anywhere else."""

    def __init__(self, hyp=None, nc=3, total_epochs=70):
        g = lambda k, d: getattr(hyp, k, d)  # noqa: E731

        self.nc = nc
        self.total_epochs = g("epochs", total_epochs)

        # ---- SATAL (scale-adaptive assigner) --------------------------------
        self.use_satal = bool(g("use_satal", False))
        self.tal_topk = g("tal_topk", 10)
        self.tal_alpha = g("tal_alpha", 0.5)
        self.tal_beta = g("tal_beta", 6.0)
        self.satal_alpha_small = g("satal_alpha_small", 1.5)
        self.satal_beta_small = g("satal_beta_small", 3.0)
        self.satal_alpha_large = g("satal_alpha_large", 1.0)
        self.satal_beta_large = g("satal_beta_large", 6.0)
        self.satal_small_area = g("satal_small_area", 0.0025)
        self.satal_large_area = g("satal_large_area", 0.0225)
        self.satal_topk_factor = g("satal_topk_factor", 1.5)

        # ---- SWA (size weight adaptive) -------------------------------------
        self.swa_mode = g("swa_mode", "scale")
        self.swa_alpha = float(g("swa_alpha", 0.0))          # 0.0 -> SWA off (stock)
        self.swa_alpha_end = g("swa_alpha_end", None)        # None -> no annealing
        self.swa_size_axis = g("swa_size_axis", "width")     # "width" | "area"
        self.swa_boost = float(g("swa_boost", 1.0))          # 1.0 -> off
        self.swa_width_thresh_px = float(g("swa_width_thresh_px", 24.0))
        self.swa_area_thresh_px2 = float(g("swa_area_thresh_px2", 32.0 ** 2))

        # ---- box regression metric -------------------------------------------
        # "ciou" (stock) | "eiou" | "siou" | "mpdiou" | "wiou"
        self.box_loss_type = g("box_loss_type", "ciou")
        self.wiou_alpha = float(g("wiou_alpha", 1.9))
        self.wiou_delta = float(g("wiou_delta", 3.0))
        self.wiou_momentum = float(g("wiou_momentum", 0.02))

        # ---- NWD ------------------------------------------------------------
        self.use_nwd = bool(g("use_nwd", False))
        self.nwd_mode = g("nwd_mode", "blend")               # "blend"|"pure"|"small_only"
        self.nwd_weight = float(g("nwd_weight", 0.5))
        self.nwd_c_px = float(g("nwd_c_px", 12.0))           # PIXELS
        self.nwd_small_width_px = float(g("nwd_small_width_px", 24.0))
        self.nwd_debug = bool(g("nwd_debug", False))

        # ---- classification --------------------------------------------------
        self.use_class_weights = bool(g("use_class_weights", False))
        self.class_counts = g("class_counts", None)          # list[nc] | None
        self.use_vfl = bool(g("use_vfl", False))
        self.vfl_alpha = float(g("vfl_alpha", 0.75))
        self.vfl_gamma = float(g("vfl_gamma", 2.0))

        # ---- loss clipping ---------------------------------------------------
        self.use_loss_clip = bool(g("use_loss_clip", False))
        self.iou_clip = float(g("iou_clip", 2.0))
        self.dfl_clip = float(g("dfl_clip", 5.0))

        # ---- AR-DFL (Aspect-Ratio-aware DFL) --------------------------------
        # THE genuinely untried axis. Reweights per-edge DFL toward the HEIGHT
        # edges (top,bottom), where tall-object localization error concentrates.
        # bbox2dist order = (left, top, right, bottom):
        #   width  edges = columns [0, 2]
        #   height edges = columns [1, 3]
        #   use_ardfl        : master switch (default off -> stock behaviour)
        #   ardfl_h_weight   : multiplier on height-edge (top,bottom) DFL  (>1)
        #   ardfl_w_weight   : multiplier on width-edge  (left,right) DFL  (<=1)
        #   ardfl_ar_gate    : apply only to boxes with GT h/w > ardfl_ar_thresh
        #   ardfl_ar_thresh  : h/w gate threshold (dataset mean ~2.69)
        #   ardfl_entropy    : sharpen ONLY the height-edge distributions
        #   ardfl_entropy_w  : weight of that entropy term
        self.use_ardfl = bool(g("use_ardfl", False))
        self.ardfl_h_weight = float(g("ardfl_h_weight", 1.5))
        self.ardfl_w_weight = float(g("ardfl_w_weight", 1.0))
        self.ardfl_ar_gate = bool(g("ardfl_ar_gate", False))
        self.ardfl_ar_thresh = float(g("ardfl_ar_thresh", 1.5))
        self.ardfl_entropy = bool(g("ardfl_entropy", False))
        self.ardfl_entropy_w = float(g("ardfl_entropy_w", 0.05))

        self._validate()

    def _validate(self):
        if self.box_loss_type not in ("ciou", "eiou", "siou", "mpdiou", "wiou"):
            raise ValueError(f"box_loss_type must be ciou|eiou|siou|mpdiou|wiou, got {self.box_loss_type!r}")
        if self.swa_mode not in ("scale", "blend"):
            raise ValueError(f"swa_mode must be 'scale' or 'blend', got {self.swa_mode!r}")
        if self.nwd_mode not in ("blend", "pure", "small_only"):
            raise ValueError(f"nwd_mode must be blend|pure|small_only, got {self.nwd_mode!r}")
        if self.swa_size_axis not in ("width", "area"):
            raise ValueError(f"swa_size_axis must be 'width' or 'area', got {self.swa_size_axis!r}")
        if not 0.0 <= self.swa_alpha <= 1.0:
            raise ValueError(f"swa_alpha must be in [0,1], got {self.swa_alpha}")
        if self.swa_boost < 1.0:
            raise ValueError(f"swa_boost must be >= 1.0, got {self.swa_boost}")
        if not 0.0 <= self.nwd_weight <= 1.0:
            raise ValueError(f"nwd_weight must be in [0,1], got {self.nwd_weight}")
        if self.class_counts is not None and len(self.class_counts) != self.nc:
            raise ValueError(f"class_counts has {len(self.class_counts)} entries but nc={self.nc}")
        if self.ardfl_h_weight < 0 or self.ardfl_w_weight < 0:
            raise ValueError("ardfl_h_weight / ardfl_w_weight must be >= 0")
        if self.ardfl_entropy_w < 0:
            raise ValueError("ardfl_entropy_w must be >= 0")

    def is_neutral(self):
        """True if this config reproduces stock Ultralytics loss."""
        return (self.box_loss_type == "ciou"
                and not self.use_satal and self.swa_alpha == 0.0 and self.swa_boost == 1.0
                and not self.use_nwd and not self.use_class_weights and not self.use_vfl
                and not self.use_loss_clip and not self.use_ardfl)

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


# =============================================================================
# NWD
# =============================================================================
# Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
# https://arxiv.org/abs/2110.13389
# Box -> 2D Gaussian, mu = center, Sigma = diag((w/2)^2, (h/2)^2)
# W2^2 = ||mu1-mu2||^2 + ||sigma1-sigma2||^2 ;  NWD = exp(-sqrt(W2^2)/C)
# C is in PIXELS here.

_NWD_DEBUG_FIRED = {"done": False}


def _nwd_similarity(pred_px, target_px, c, eps=1e-7, debug=False):
    """NWD similarity in (0, 1]. Boxes xyxy in PIXELS. Returns (N,)."""
    cxp = (pred_px[:, 0] + pred_px[:, 2]) * 0.5
    cyp = (pred_px[:, 1] + pred_px[:, 3]) * 0.5
    sxp = (pred_px[:, 2] - pred_px[:, 0]).clamp(min=eps) * 0.5
    syp = (pred_px[:, 3] - pred_px[:, 1]).clamp(min=eps) * 0.5

    cxt = (target_px[:, 0] + target_px[:, 2]) * 0.5
    cyt = (target_px[:, 1] + target_px[:, 3]) * 0.5
    sxt = (target_px[:, 2] - target_px[:, 0]).clamp(min=eps) * 0.5
    syt = (target_px[:, 3] - target_px[:, 1]).clamp(min=eps) * 0.5

    w2_sq = (cxp - cxt) ** 2 + (cyp - cyt) ** 2 + (sxp - sxt) ** 2 + (syp - syt) ** 2
    w2 = torch.sqrt(w2_sq.clamp(min=eps))
    nwd = torch.exp(-w2 / c)

    if debug and not _NWD_DEBUG_FIRED["done"]:
        _NWD_DEBUG_FIRED["done"] = True
        print(
            f"\n[NWD] first batch — W2(px): mean={w2.mean():.2f} med={w2.median():.2f} "
            f"max={w2.max():.2f} | C={c}\n"
            f"      NWD: mean={nwd.mean():.4f} min={nwd.min():.4f} max={nwd.max():.4f}\n"
            f"      target NWD mean ~0.3-0.7. If ~1.0 lower C; if ~0.0 raise C.\n"
        )
    return nwd


# =============================================================================
# BOX REGRESSION METRICS
# =============================================================================


def _corner_geometry(pred, target, eps=1e-7):
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=eps)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=eps)
    dx = (px1 + px2) * 0.5 - (tx1 + tx2) * 0.5
    dy = (py1 + py2) * 0.5 - (ty1 + ty2) * 0.5
    return dict(px1=px1, py1=py1, px2=px2, py2=py2, tx1=tx1, ty1=ty1, tx2=tx2, ty2=ty2,
                pw=pw, ph=ph, tw=tw, th=th, iou=iou, cw=cw, ch=ch, dx=dx, dy=dy)


def _box_loss_terms(pred, target, kind, eps=1e-7):
    """Return per-box regression loss (N,), i.e. 1 - similarity. WIoU handled separately."""
    if kind == "ciou":
        return (1.0 - bbox_iou(pred, target, xywh=False, CIoU=True).view(-1))

    g = _corner_geometry(pred, target, eps)
    iou = g["iou"]

    if kind == "eiou":
        c2 = g["cw"] ** 2 + g["ch"] ** 2 + eps
        rho2 = g["dx"] ** 2 + g["dy"] ** 2
        w_term = (g["pw"] - g["tw"]) ** 2 / (g["cw"] ** 2 + eps)
        h_term = (g["ph"] - g["th"]) ** 2 / (g["ch"] ** 2 + eps)
        return 1.0 - (iou - rho2 / c2 - w_term - h_term)

    if kind == "siou":
        sigma = torch.sqrt(g["dx"] ** 2 + g["dy"] ** 2) + eps
        sin_a = torch.abs(g["dy"]) / sigma
        sin_b = torch.abs(g["dx"]) / sigma
        sin_a = torch.where(sin_a < sin_b, sin_a, sin_b)
        angle = torch.cos(2 * (torch.asin(sin_a.clamp(-1 + eps, 1 - eps)) - math.pi / 4))
        gamma = 2 - angle
        dist = (1 - torch.exp(-gamma * (g["dx"] / g["cw"]) ** 2)) + \
               (1 - torch.exp(-gamma * (g["dy"] / g["ch"]) ** 2))
        ow = torch.abs(g["pw"] - g["tw"]) / torch.max(g["pw"], g["tw"])
        oh = torch.abs(g["ph"] - g["th"]) / torch.max(g["ph"], g["th"])
        shape = (1 - torch.exp(-ow)) ** 4 + (1 - torch.exp(-oh)) ** 4
        return 1.0 - (iou - 0.5 * (dist + shape))

    if kind == "mpdiou":
        d1 = (g["px1"] - g["tx1"]) ** 2 + (g["py1"] - g["ty1"]) ** 2
        d2 = (g["px2"] - g["tx2"]) ** 2 + (g["py2"] - g["ty2"]) ** 2
        d = g["cw"] ** 2 + g["ch"] ** 2 + eps
        return 1.0 - (iou - d1 / d - d2 / d)

    raise ValueError(f"unknown box_loss_type: {kind!r}")


def _wiou_v3(pred, target, running_mean, alpha, delta, eps=1e-7):
    """WIoU v3 (Tong et al. 2023). Returns (loss (N,), new_running_mean)."""
    g = _corner_geometry(pred, target, eps)
    l_iou = 1.0 - g["iou"]

    cw_d = g["cw"].detach()
    ch_d = g["ch"].detach()
    r_wiou = torch.exp((g["dx"] ** 2 + g["dy"] ** 2) / (cw_d ** 2 + ch_d ** 2 + eps))

    mean = l_iou.mean().item() if running_mean is None else running_mean
    beta = (l_iou.detach() / (mean + eps)).clamp(min=eps)
    r = beta / (delta * torch.pow(torch.tensor(alpha, device=pred.device), beta - delta) + eps)
    return (r * r_wiou * l_iou), mean


# =============================================================================
# SIZE WEIGHTING (bounded — replaces v1's 1/area)
# =============================================================================


def _size_weight(target_px, cfg):
    """Bounded per-object weight in [1, swa_boost]. axis='width' default (94% tall)."""
    if cfg.swa_boost <= 1.0:
        return torch.ones(target_px.shape[0], device=target_px.device, dtype=target_px.dtype)

    if cfg.swa_size_axis == "width":
        s = (target_px[:, 2] - target_px[:, 0]).clamp(min=1e-6)
        ratio = (s / cfg.swa_width_thresh_px).clamp(max=1.0)
    else:
        w = (target_px[:, 2] - target_px[:, 0]).clamp(min=1e-6)
        h = (target_px[:, 3] - target_px[:, 1]).clamp(min=1e-6)
        ratio = ((w * h) / cfg.swa_area_thresh_px2).clamp(max=1.0)

    return cfg.swa_boost - (cfg.swa_boost - 1.0) * ratio


# =============================================================================
# BASE COMPONENTS
# =============================================================================


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367"""

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
    """Focal loss."""

    def __init__(self, gamma=1.5, alpha=0.25, reduction="sum"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, pred, label, gamma=None, alpha=None):
        gamma = self.gamma if gamma is None else gamma
        alpha = self.alpha if alpha is None else alpha
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        p = pred.sigmoid()
        p_t = label * p + (1 - label) * (1 - p)
        loss = loss * (1.0 - p_t) ** gamma
        if alpha > 0:
            loss = loss * (label * alpha + (1 - label) * (1 - alpha))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.mean(1).sum()
        return loss


class DFLoss(nn.Module):
    """Distribution Focal Loss (with per-edge access for AR-DFL)."""

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

    def per_edge(self, pred_dist, target):
        """Per-edge DFL loss, shape (N, 4) in (left, top, right, bottom) order.

        Identical math to __call__ but WITHOUT the mean over the 4 edges, so an
        AR-aware weighting can be applied per edge before reduction.
        pred_dist: (N*4, reg_max)   target: (N, 4)
        """
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        )  # (N, 4)


def _dfl_edge_entropy(pred_dist):
    """Mean entropy of the softmax bin distributions in pred_dist (N*E, reg_max).

    Lower entropy = sharper (more peaked) distribution = tighter edge estimate.
    Returned as a scalar to be ADDED (minimized) — used only on height edges.
    """
    logp = F.log_softmax(pred_dist, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1).mean()


# =============================================================================
# BBOX LOSS
# =============================================================================


class BboxLoss(nn.Module):
    """
    Box loss with bounded SWA weighting, optional pixel-space NWD, and AR-DFL.

    Takes the SHARED config object — it has no defaults of its own, so the
    printed config and the computed loss cannot disagree.
    forward() requires `stride_tensor` so all size/NWD math is done in pixels.
    """

    def __init__(self, reg_max=16, cfg=None):
        super().__init__()
        if cfg is None:
            raise ValueError("BboxLoss requires the shared SataLSwaConfig (no independent defaults).")
        self.cfg = cfg
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self._wiou_mean = None  # momentum-smoothed mean of L_IoU (WIoU v3)

    def _current_alpha(self):
        """SWA alpha. Anneals only if epoch tracking is attached AND an end is set."""
        cfg = self.cfg
        if cfg.swa_alpha_end is None:
            return cfg.swa_alpha
        progress, active = _get_progress(cfg.total_epochs)
        if not active:
            return 0.5 * (cfg.swa_alpha + cfg.swa_alpha_end)  # FAIL-SAFE midpoint
        return cfg.swa_alpha * (1 - progress) + cfg.swa_alpha_end * progress

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor):
        cfg = self.cfg
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        stride_fg = stride_tensor.expand(target_bboxes.shape[0], -1, -1)[fg_mask]  # (N,1)
        pred_px = pred_fg * stride_fg
        target_px = target_fg * stride_fg

        # ---- weights --------------------------------------------------------
        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)      # (N,1)
        size_weight = _size_weight(target_px, cfg).unsqueeze(-1)          # (N,1) in [1,boost]

        if cfg.swa_mode == "scale":
            weight = score_weight * size_weight
        else:  # "blend"
            a = self._current_alpha()
            weight = a * size_weight + (1.0 - a) * score_weight

        # ---- box regression loss ---------------------------------------------
        if cfg.box_loss_type == "wiou":
            base_loss, batch_mean = _wiou_v3(
                pred_fg, target_fg, self._wiou_mean, cfg.wiou_alpha, cfg.wiou_delta
            )
            cur = (1.0 - bbox_iou(pred_fg, target_fg, xywh=False, CIoU=False).view(-1)).mean().item()
            m = cfg.wiou_momentum
            self._wiou_mean = cur if self._wiou_mean is None else (1 - m) * self._wiou_mean + m * cur
            ciou_loss = base_loss.unsqueeze(-1)
        else:
            ciou_loss = _box_loss_terms(pred_fg, target_fg, cfg.box_loss_type).unsqueeze(-1)

        if cfg.use_nwd:
            nwd = _nwd_similarity(pred_px, target_px, cfg.nwd_c_px, debug=cfg.nwd_debug).unsqueeze(-1)
            nwd_loss = 1.0 - nwd
            if cfg.nwd_mode == "pure":
                box_loss = nwd_loss
            elif cfg.nwd_mode == "blend":
                box_loss = (1.0 - cfg.nwd_weight) * ciou_loss + cfg.nwd_weight * nwd_loss
            else:  # "small_only" — gate on WIDTH in pixels
                tw = (target_px[:, 2] - target_px[:, 0]).unsqueeze(-1)
                is_thin = tw < cfg.nwd_small_width_px
                box_loss = torch.where(is_thin, nwd_loss, ciou_loss)
        else:
            box_loss = ciou_loss

        if cfg.use_loss_clip:
            box_loss = box_loss.clamp(max=cfg.iou_clip)

        # ---- normalization ---------------------------------------------------
        norm = weight.sum().clamp(min=1e-9)
        loss_iou = (box_loss * weight).sum() / norm

        # ---- DFL (with optional AR-DFL) --------------------------------------
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            pred_dist_fg = pred_dist[fg_mask].view(-1, self.reg_max)   # (N*4, reg_max)
            target_fg_ltrb = target_ltrb[fg_mask]                      # (N, 4) = (l,t,r,b)

            if cfg.use_ardfl:
                # AR-DFL: reweight per-edge DFL toward the HEIGHT edges (t,b).
                #   0=left  1=top  2=right  3=bottom  ->  width=[0,2] height=[1,3]
                per_edge = self.dfl_loss.per_edge(pred_dist_fg, target_fg_ltrb)  # (N,4)

                edge_w = torch.tensor(
                    [cfg.ardfl_w_weight, cfg.ardfl_h_weight,
                     cfg.ardfl_w_weight, cfg.ardfl_h_weight],
                    device=per_edge.device, dtype=per_edge.dtype,
                ).view(1, 4)  # (1,4)

                if cfg.ardfl_ar_gate:
                    # Only tall boxes (h/w > thresh) get the asymmetric reweight;
                    # near-square boxes keep symmetric DFL (edge_w -> 1).
                    tw = (target_px[:, 2] - target_px[:, 0]).clamp(min=1e-6)
                    th = (target_px[:, 3] - target_px[:, 1]).clamp(min=1e-6)
                    is_tall = (th / tw) > cfg.ardfl_ar_thresh          # (N,)
                    edge_w = torch.where(
                        is_tall.view(-1, 1),
                        edge_w.expand(per_edge.shape[0], -1),
                        torch.ones_like(per_edge),
                    )

                dfl_per_box = (per_edge * edge_w).mean(-1, keepdim=True)  # (N,1)

                if cfg.use_loss_clip:
                    dfl_per_box = dfl_per_box.clamp(max=cfg.dfl_clip)
                loss_dfl = (dfl_per_box * weight).sum() / norm

                # Optional: sharpen ONLY the height-edge distributions (t,b).
                if cfg.ardfl_entropy and cfg.ardfl_entropy_w > 0:
                    pe = pred_dist_fg.view(-1, 4, self.reg_max)          # (N,4,reg_max)
                    height_logits = pe[:, [1, 3], :].reshape(-1, self.reg_max)
                    loss_dfl = loss_dfl + cfg.ardfl_entropy_w * _dfl_edge_entropy(height_logits)
            else:
                dfl = self.dfl_loss(pred_dist_fg, target_fg_ltrb)
                if cfg.use_loss_clip:
                    dfl = dfl.clamp(max=cfg.dfl_clip)
                loss_dfl = (dfl * weight).sum() / norm
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Rotated box loss (probiou). Keeps the 7-arg signature used by OBB."""

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max - 1)
            dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Keypoint loss."""

    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        f = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (f.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


# =============================================================================
# DETECTION LOSS
# =============================================================================


class v8DetectionLoss:
    """SATAL + SWA + NWD + AR-DFL detection loss. All settings from one config object."""

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]

        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1

        self.cfg = SataLSwaConfig(h, nc=self.nc, total_epochs=getattr(h, "epochs", 70))
        if getattr(h, "tal_topk", None) is None:
            self.cfg.tal_topk = tal_topk

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(m.reg_max, cfg=self.cfg).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.class_weights = None
        if self.cfg.use_class_weights:
            counts = self.cfg.class_counts
            if counts is None:
                raise ValueError(
                    "use_class_weights=True requires class_counts=[n0,n1,...] in hyp "
                    "(in data.yaml names[] order)."
                )
            c = torch.tensor([float(x) for x in counts], device=device)
            inv = 1.0 / c
            inv = inv / inv.mean()
            w = torch.sqrt(inv)
            self.class_weights = (w / w.mean()).view(1, 1, -1)

        if self.cfg.use_satal:
            try:
                from ultralytics.utils.satal import ScaleAdaptiveTaskAlignedAssigner
            except ImportError as e:
                raise ImportError(
                    "use_satal=True but ultralytics.utils.satal is missing. "
                    "Install/copy the SATAL assigner or set use_satal=False."
                ) from e
            self.assigner = ScaleAdaptiveTaskAlignedAssigner(
                topk=self.cfg.tal_topk, num_classes=self.nc,
                alpha=self.cfg.tal_alpha, beta=self.cfg.tal_beta,
                alpha_small=self.cfg.satal_alpha_small, beta_small=self.cfg.satal_beta_small,
                alpha_large=self.cfg.satal_alpha_large, beta_large=self.cfg.satal_beta_large,
                small_area_thresh=self.cfg.satal_small_area,
                large_area_thresh=self.cfg.satal_large_area,
                topk_small_factor=self.cfg.satal_topk_factor,
            )
        else:
            self.assigner = TaskAlignedAssigner(
                topk=self.cfg.tal_topk, num_classes=self.nc,
                alpha=self.cfg.tal_alpha, beta=self.cfg.tal_beta,
            )

        self._print_config()

    def verify_config(self, verbose=True):
        """Report the LIVE state of the objects that actually compute the loss."""
        live = {
            "assigner_class": type(self.assigner).__name__,
            "satal_active": type(self.assigner).__name__.startswith("ScaleAdaptive"),
            "bbox_loss_cfg_is_shared": self.bbox_loss.cfg is self.cfg,
            "box_loss_type_live": self.bbox_loss.cfg.box_loss_type,
            "use_nwd_live": self.bbox_loss.cfg.use_nwd,
            "use_ardfl_live": self.bbox_loss.cfg.use_ardfl,
            "swa_mode_live": self.bbox_loss.cfg.swa_mode,
            "swa_alpha_live": self.bbox_loss.cfg.swa_alpha,
            "swa_boost_live": self.bbox_loss.cfg.swa_boost,
            "class_weights_live": None if self.class_weights is None
            else self.class_weights.flatten().tolist(),
            "epoch_tracking_attached": _EPOCH_STATE["ever_set"],
            "is_neutral_stock_equivalent": self.cfg.is_neutral(),
        }
        assert live["bbox_loss_cfg_is_shared"], "BboxLoss is not sharing the config object!"
        assert live["use_nwd_live"] == self.cfg.use_nwd, "use_nwd mismatch!"
        assert live["use_ardfl_live"] == self.cfg.use_ardfl, "use_ardfl mismatch!"
        assert live["box_loss_type_live"] == self.cfg.box_loss_type, "box_loss_type mismatch!"
        assert live["satal_active"] == self.cfg.use_satal, "SATAL flag does not match assigner!"
        if verbose:
            print("\n[verify_config] live loss state")
            for k, v in live.items():
                print(f"    {k:32s} {v}")
            print()
        return live

    def _print_config(self):
        c = self.cfg
        print("\n" + "=" * 62)
        print("  SATAL-SWA-NWD-ARDFL Loss")
        print("=" * 62)
        print(f"  neutral (== stock loss):  {c.is_neutral()}")
        print(f"  assigner:                 {type(self.assigner).__name__}")
        print(f"    topk/alpha/beta:        {c.tal_topk} / {c.tal_alpha} / {c.tal_beta}")
        if c.use_satal:
            print(f"    satal small a/b:        {c.satal_alpha_small} / {c.satal_beta_small}")
            print(f"    satal topk_factor:      {c.satal_topk_factor}")
        print(f"  box_loss_type:            {c.box_loss_type}")
        if c.box_loss_type == "wiou":
            print(f"    wiou a/d/momentum:      {c.wiou_alpha} / {c.wiou_delta} / {c.wiou_momentum}")
        print(f"  SWA mode:                 {c.swa_mode}")
        print(f"    alpha (blend only):     {c.swa_alpha} -> {c.swa_alpha_end}")
        print(f"    size axis / boost:      {c.swa_size_axis} / {c.swa_boost}")
        print(f"    width thresh (px):      {c.swa_width_thresh_px}")
        print(f"  NWD:                      {c.use_nwd}")
        if c.use_nwd:
            print(f"    mode / weight / C(px):  {c.nwd_mode} / {c.nwd_weight} / {c.nwd_c_px}")
        print(f"  class weights:            {c.use_class_weights}")
        if self.class_weights is not None:
            print(f"    values:                 {self.class_weights.flatten().cpu().numpy().round(4)}")
        print(f"  VFL / loss clip:          {c.use_vfl} / {c.use_loss_clip}")
        print(f"  AR-DFL:                   {c.use_ardfl}")
        if c.use_ardfl:
            print(f"    h/w edge weight:        {c.ardfl_h_weight} / {c.ardfl_w_weight}")
            print(f"    AR gate:                {c.ardfl_ar_gate}" +
                  (f" (h/w > {c.ardfl_ar_thresh})" if c.ardfl_ar_gate else ""))
            print(f"    height entropy:         {c.ardfl_entropy}" +
                  (f" (w={c.ardfl_entropy_w})" if c.ardfl_entropy else ""))
        print(f"  epoch tracking attached:  {_EPOCH_STATE['ever_set']}")
        print("=" * 62 + "\n")

    def preprocess(self, targets, batch_size, scale_tensor):
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
        if self.use_dfl:
            b, a, ch = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, ch // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_cls_loss(self, pred_scores, target_scores, target_scores_sum, dtype):
        """BCE, optionally VFL-modulated and class-weighted. Neutral -> stock BCE."""
        loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.cfg.use_vfl:
            label = (target_scores > 0).to(dtype)
            p = pred_scores.sigmoid()
            loss = loss * (self.cfg.vfl_alpha * p.pow(self.cfg.vfl_gamma) * (1 - label) + target_scores * label)
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(dtype)
        return loss.sum() / target_scores_sum

    def __call__(self, preds, batch):
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

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        if hasattr(self.assigner, "set_imgsz"):
            self.assigner.set_imgsz(imgsz)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


# =============================================================================
# OTHER TASK LOSSES  (delegate to v8DetectionLoss machinery; unchanged from
# the base rebuild — AR-DFL flows through BboxLoss automatically for seg/pose/obb
# because they all call self.bbox_loss).
# =============================================================================


class v8ClassificationLoss:
    """Classification loss."""

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Segmentation loss."""

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

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        if hasattr(self.assigner, "set_imgsz"):
            self.assigner.set_imgsz(imgsz)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, stride_tensor,
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
        tb_norm = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(tb_norm)[..., 2:].prod(2)
        mxyxy = tb_norm * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, si in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_i, gt_idx_i, pm_i, proto_i, mxyxy_i, marea_i, masks_i = si
            if fg_i.any():
                mask_idx = gt_idx_i[fg_i]
                if overlap:
                    gt_mask = (masks_i == (mask_idx + 1).view(-1, 1, 1)).float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += self.single_mask_loss(gt_mask, pm_i[fg_i], proto_i, mxyxy_i[fg_i], marea_i[fg_i])
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
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        if hasattr(self.assigner, "set_imgsz"):
            self.assigner.set_imgsz(imgsz)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor,
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
        batched = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                              device=keypoints.device)
        for i in range(batch_size):
            ki = keypoints[batch_idx == i]
            batched[i, : ki.shape[0]] = ki

        idx_exp = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected = batched.gather(1, idx_exp.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))
        selected /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            gt_kpt = selected[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss


class v8OBBLoss(v8DetectionLoss):
    """OBB (rotated) loss."""

    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=self.cfg.tal_topk, num_classes=self.nc,
            alpha=self.cfg.tal_alpha, beta=self.cfg.tal_beta,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max, cfg=self.cfg).to(self.device)

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

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
        targets = targets[(rw >= 2) & (rh >= 2)]
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

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
        loss[1] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

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
            b, a, ch = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, ch // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
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
    (class_weights / VFL) as the main detection loss.
    """

    def __init__(self, model, obj_weight=1.0):
        super().__init__(model)
        self.obj_weight = getattr(model.model[-1], "obj_weight", obj_weight)
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

        # NOTE: _compute_cls_loss takes 4 args (pred_scores, target_scores,
        # target_scores_sum, dtype). The original loss.py called it here with a
        # stale 7-arg signature (leftover from an older SWA cls path) which would
        # crash if DetectObj were ever used. Fixed to the real signature.
        loss[1] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

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
