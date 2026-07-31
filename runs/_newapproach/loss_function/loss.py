# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# =============================================================================
# SATAL + SWA + NWD  —  rebuilt (v2)
# =============================================================================
# Rewrite of loss_satal_swa_plus_nwd.py. Same ideas, but every silent-failure
# mode from v1 is removed. Behaviour is now observable and ablatable.
#
# WHAT WAS BROKEN IN v1 (and is fixed here)
# -----------------------------------------------------------------------------
# 1. use_nwd split-brain default: v8DetectionLoss defaulted True, BboxLoss
#    defaulted False. With no hyp key set, the config PRINTED "use_nwd: True"
#    while NWD never executed.  -> FIX: one config object, read once, passed
#    explicitly. BboxLoss has no independent defaults.
# 2. self.epoch never updated (relied on model.current_epoch, which Ultralytics
#    does not set). alpha froze at alpha_start=0.9 for all 70 epochs — the most
#    aggressive setting — and every clip schedule froze too.  -> FIX: explicit
#    epoch callback + fail-SAFE fallback (constant mid alpha, loud warning)
#    instead of silently sitting at worst case.
# 3. area_weight = 1/area, batch-max normalized: 400:1 spread on this dataset
#    (1-cell box weighted 1.0, typical 37-cell object 0.027). Most objects
#    contributed ~2% of the gradient of the smallest box in the batch, and the
#    scale jittered per batch.  -> FIX: bounded size weight in [1, boost],
#    keyed on WIDTH (dataset is 94% tall, mean 33x72px — width is the hard axis).
# 4. Normalizer mismatch: numerator weighted by SWA weights, denominator still
#    target_scores_sum.  -> FIX: divide by the sum of the weights actually used.
#    (With swa_alpha=0 this is mathematically identical to target_scores_sum,
#    so the stock path is reproduced exactly — see NOTE below.)
# 5. Class weighting hardcoded ALWAYS ON from counts that did not match the
#    dataset (34901/28628/66946 vs actual 11491/9490/21557).  -> FIX: hyp-driven,
#    ablatable, counts configurable.
# 6. nwd_C in grid-cell units -> different physical size per FPN level.
#    -> FIX: all size/NWD math in PIXELS.
# 7. E2EDetectLoss referenced undefined v8DetectionLossLuggage (NameError).
# 8. Segmentation mask_h/mask_w unpack failed when overlap=True.
#
# NOTE on the normalizer: target_scores is zero on background anchors, so
# target_scores.sum() == score_weight.sum() exactly. Therefore with
# swa_alpha=0 / size weighting off, dividing by the weight sum reproduces the
# stock normalization bit-for-bit. This gives a clean neutral baseline.
#
# NEUTRAL CONFIG (reproduces stock Ultralytics loss):
#   use_satal=False, swa_alpha=0.0, swa_boost=1.0, use_nwd=False,
#   use_class_weights=False, use_loss_clip=False
#
# EPOCH TRACKING — REQUIRED for any schedule to work:
#   from ultralytics.utils.loss import attach_epoch_tracking
#   model = YOLO(...)
#   attach_epoch_tracking(model)      # <-- do this BEFORE model.train(...)
# If you skip it, schedules are disabled and a warning is printed once. They do
# NOT silently freeze at the aggressive end any more.
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
# v1's bug: self.epoch was read from model.current_epoch, which Ultralytics never
# sets, so it stayed 0 forever and every schedule froze at its start value.
# Here epoch lives in one module-level slot, written by an explicit callback.
# _EPOCH_EVER_SET lets the loss detect "nobody wired this up" and degrade SAFELY.

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
    clipping annealing) are DISABLED and a one-time warning is printed. This is
    deliberate: v1 silently froze at the most aggressive setting instead.
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
# v1 kept parallel copies of the same settings in v8DetectionLoss and BboxLoss,
# each with its own default. That is exactly how use_nwd ended up printed as True
# while computing as False. Here the config is parsed ONCE and handed to BboxLoss.


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
        # swa_mode:
        #   "scale" (default) -> weight = score_weight * size_weight
        #        Multiplicative. Preserves TAL's quality signal and only modulates
        #        it by size. This is the recommended form.
        #   "blend"           -> weight = a*size_weight + (1-a)*score_weight
        #        v1's additive form, kept for fidelity/ablation. Note it REPLACES
        #        the quality signal rather than modulating it.
        self.swa_mode = g("swa_mode", "scale")
        self.swa_alpha = float(g("swa_alpha", 0.0))          # 0.0 -> SWA off (stock)
        self.swa_alpha_end = g("swa_alpha_end", None)        # None -> no annealing
        # size weighting: bounded, in [1, boost]. NOT 1/area.
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
        # ONE flag. No second default anywhere.
        self.use_nwd = bool(g("use_nwd", False))
        self.nwd_mode = g("nwd_mode", "blend")               # "blend"|"pure"|"small_only"
        self.nwd_weight = float(g("nwd_weight", 0.5))
        self.nwd_c_px = float(g("nwd_c_px", 12.0))           # PIXELS (v1 used grid cells)
        self.nwd_small_width_px = float(g("nwd_small_width_px", 24.0))
        self.nwd_debug = bool(g("nwd_debug", False))

        # ---- classification --------------------------------------------------
        self.use_class_weights = bool(g("use_class_weights", False))
        self.class_counts = g("class_counts", None)          # list[nc] | None
        self.use_vfl = bool(g("use_vfl", False))
        self.vfl_alpha = float(g("vfl_alpha", 0.75))
        self.vfl_gamma = float(g("vfl_gamma", 2.0))

        # ---- loss clipping ---------------------------------------------------
        # v1 clipped at max_clip/10 which was mostly inert, except it could
        # truncate DFL gradient on exactly the small boxes it meant to help.
        # Off by default now; values are the real clamp, no hidden /10.
        self.use_loss_clip = bool(g("use_loss_clip", False))
        self.iou_clip = float(g("iou_clip", 2.0))
        self.dfl_clip = float(g("dfl_clip", 5.0))

        # ---- AR-DFL (Aspect-Ratio-aware DFL) --------------------------------
        # THE genuinely untried axis. Every prior loss (60 runs) reweighted
        # *which samples* matter or swapped the IoU flavour; NONE changed the
        # box REPRESENTATION. The 25pt mAP50->mAP50-95 gap is a box-tightness
        # problem, and DFL quantizes all 4 edges identically. For 94%-tall
        # objects (AR 2.69) the HEIGHT edges (top,bottom) carry the large,
        # hard-to-localize range while WIDTH edges (left,right) are short/easy.
        #
        # bbox2dist orders edges as (left, top, right, bottom):
        #   width  edges = columns [0, 2]
        #   height edges = columns [1, 3]
        #
        # AR-DFL raises the DFL loss weight on the HEIGHT edges so the network
        # spends its regression capacity where the residual error actually lives.
        # It is orthogonal to NWD (keep NWD on) and needs NO architecture change
        # (same reg_max, same head) — it only reweights the existing per-edge DFL.
        #
        #   use_ardfl        : master switch (default off -> stock behaviour)
        #   ardfl_h_weight   : multiplier on height-edge (top,bottom) DFL  (>1)
        #   ardfl_w_weight   : multiplier on width-edge  (left,right) DFL  (<=1)
        #   ardfl_ar_gate    : if True, apply the reweight ONLY to boxes whose
        #                      GT aspect ratio (h/w) exceeds ardfl_ar_thresh,
        #                      so near-square boxes keep symmetric DFL.
        #   ardfl_ar_thresh  : h/w threshold for the gate (dataset mean ~2.69)
        #   ardfl_entropy    : optional entropy penalty on the HEIGHT-edge
        #                      distributions (sharpen -> tighter height). The
        #                      r10_dfl_entropy run gave the best bag AP; this
        #                      applies that idea only where it helps (height).
        #   ardfl_entropy_w  : weight of that entropy term.
        self.use_ardfl = bool(g("use_ardfl", False))
        self.ardfl_h_weight = float(g("ardfl_h_weight", 1.5))
        self.ardfl_w_weight = float(g("ardfl_w_weight", 1.0))
        self.ardfl_ar_gate = bool(g("ardfl_ar_gate", False))
        self.ardfl_ar_thresh = float(g("ardfl_ar_thresh", 1.5))
        self.ardfl_entropy = bool(g("ardfl_entropy", False))
        self.ardfl_entropy_w = float(g("ardfl_entropy_w", 0.05))

        # ---- A-DFL (Anisotropic DFL) — THE representation change -------------
        # Everything above (and all ~60 prior configs) reweights the loss. This
        # changes what the box CAN express.
        #
        # Stock DFL encodes every edge distance d (in stride units) over the
        # same reg_max bins, so 1 bin = 1 stride = 8/16/32 px on ALL four edges.
        # A-DFL introduces a per-edge range scale s_e:
        #
        #     encode:  t_e = d_e / s_e     (then the usual clamp to reg_max-1)
        #     decode:  d_e = E[bin] * s_e
        #
        # Bin spacing on edge e becomes s_e * stride px. With s_w < 1 the width
        # edges get FINER bins (higher resolution) over a SHORTER range; with
        # s_h >= 1 the height edges trade resolution for reach.
        #
        # Why width: an edge error of e px costs e/w IoU on a width edge and
        # e/h on a height edge — ratio exactly h/w (2.69 on this dataset). Width
        # edges are the most IoU-sensitive AND, being short, occupy only ~1.6-2.6
        # of the 16 bins. The rest of the width budget is spent on a range the
        # data never reaches. Compressing it is free resolution.
        #
        # reg_max is UNCHANGED, so pretrained weights load and the head keeps
        # reg_max*4 channels. The identical scale MUST be applied in the
        # inference decode (see adfl_patch_dfl.py) or train/test will disagree.
        #
        #   use_adfl        master switch (False -> stock DFL exactly)
        #   adfl_w_scale    range scale for width  edges (left, right)   <= 1
        #   adfl_h_scale    range scale for height edges (top, bottom)
        #   adfl_log_clamp  log the per-edge saturation rate once per epoch
        self.use_adfl = bool(g("use_adfl", False))
        self.adfl_w_scale = float(g("adfl_w_scale", 1.0))
        self.adfl_h_scale = float(g("adfl_h_scale", 1.0))
        self.adfl_log_clamp = bool(g("adfl_log_clamp", True))

        # ---- PEU-DFL (Per-Edge Uncertainty-attenuated DFL) -------------------
        # MOTIVATED BY MEASUREMENT, not by assumption. diag_per_edge_dfl.py on
        # this dataset reports per-edge residuals of:
        #     top 4.96px (6.67% of H)   bottom 2.19px (3.40%)
        #     left 1.34px (5.43% of W)  right 1.44px (5.61%)
        # The TOP edge is 2.26x worse than the bottom and 3.6x worse than the
        # width edges. That asymmetry is not explained by aspect ratio, scale or
        # quantisation (all residuals are SUB-BIN, 0.12-0.43 bins, 0% saturation
        # -> the model is not bin-limited). It is semantic: the bottom edge is
        # ground contact (sharp, unambiguous), the top edge is handles,
        # telescoping poles, straps and occlusion by the carrier — ambiguous for
        # the annotator as much as for the model.
        #
        # Forcing the network to fit an intrinsically ambiguous target is label
        # noise fitting. The standard remedy is learned attenuation (Kendall &
        # Gal 2017; KL-Loss, He et al. 2019; Gaussian YOLO), which normally
        # needs an extra head predicting variance.
        #
        # KEY POINT: DFL ALREADY EMITS A DISTRIBUTION PER EDGE. Its variance is
        # a free aleatoric uncertainty estimate that every implementation throws
        # away by taking only the expectation. PEU-DFL uses it:
        #
        #     mu_e   = sum_i p_i * i                 (this is already the decode)
        #     var_e  = sum_i p_i * (i - mu_e)^2      (free, currently discarded)
        #     s_e    = log var_e
        #     L_e   <- L_e * exp(-beta * s_e) + lambda * s_e
        #
        # The attenuation term down-weights edges the model is unsure about; the
        # lambda*s_e term stops it declaring everything uncertain. At the optimum
        # var_e tracks L_e, so intrinsically hard edges (top) are attenuated and
        # easy ones (bottom) are not — self-calibrating, zero new parameters,
        # zero architecture change.
        #
        # The weights are MEAN-NORMALISED to 1, so PEU redistributes DFL weight
        # across edges without changing the total DFL magnitude. Without that,
        # any gain is confounded with simply turning the DFL gain up.
        #
        #   use_peu           master switch
        #   peu_beta          attenuation strength (1.0 = full Kendall form)
        #   peu_lambda        weight of the log-variance penalty
        #   peu_detach        compute the WEIGHT from a detached variance
        #                     (prevents the net gaming variance to cut its loss;
        #                      the lambda term still receives gradient)
        #   peu_warmup_epochs no attenuation before this epoch — at init the
        #                     distribution is near-uniform, so variance is
        #                     meaningless and would suppress all learning
        #   peu_min_var       floor on variance before the log
        #   peu_min_var       floor on variance before the log. NOT a numerical
        #                     epsilon: DFL targets are interpolated between two
        #                     ADJACENT bins, so even a perfectly fitted edge has
        #                     var ~0.19-0.25. A floor of 1e-3 would let a
        #                     confident edge take weight 1/1e-3, and mean
        #                     normalisation preserves the mean but not the
        #                     spread — measured blow-up to 86x at beta=1.
        #   peu_w_clip        hard bound on the weight before normalisation,
        #                     so the spread stays bounded regardless of beta
        # ---- LBA (Level-Balanced Assignment) — THE new mechanism -------------
        # MEASURED PATHOLOGY (diag_per_edge_dfl.py, v12s_default2, 67,685 fg
        # anchors over 4,600 val images):
        #
        #   stride   % of anchor grid   % of foreground   ratio
        #      8          76.2%              59.3%         0.78
        #     16          19.0%              39.4%         2.07
        #     32           4.8%               1.3%         0.28
        #
        # P5 receives 887 foreground anchors in total — an entire pyramid level
        # gets essentially no box/DFL gradient, while P4 is over-subscribed 7x
        # relative to it. Meanwhile LARGE objects are 29.5% of all foreground,
        # carry the worst top-edge residual (8.17px vs 3.43px for small) and the
        # worst AP (44.41 stock, recoverable to 59.85 — so it is not a capacity
        # limit, it is a supervision-allocation problem).
        #
        # WHY THIS HAPPENS: TAL selects topk candidates per GT by
        # score^alpha * iou^beta, which is completely LEVEL-AGNOSTIC. Nothing in
        # the metric knows which pyramid level a candidate came from, so levels
        # with more anchors and easier early IoU win the topk, and coarse levels
        # starve. OTA / SimOTA / TAL all balance across GTs; none balance across
        # LEVELS.
        #
        # THE MECHANISM: multiply the alignment metric by a soft scale-matching
        # prior. Each level has a nominal object size of stride * lba_ref_cells
        # (measured: assigned objects sit at ~7-8 cells tall on every level).
        # For a GT of geometric size s:
        #
        #     octaves = log2( s / (stride * lba_ref_cells) )
        #     prior   = exp( -octaves^2 / (2 * lba_sigma^2) )
        #     align  <- align * prior^lba_strength
        #
        # Soft, not a hard size range (FCOS/ATSS use hard ranges and cannot
        # express partial preference); no new parameters; no architecture change.
        # lba_strength=0 reproduces stock TAL exactly.
        #
        #   use_lba        master switch
        #   lba_strength   prior exponent; 0 = off, 1.0 = full prior
        #   lba_ref_cells  nominal object size per level, in stride units
        #   lba_sigma      prior width in octaves (1.0 = +/- one octave is 0.61x)
        self.use_lba = bool(g("use_lba", False))
        self.lba_strength = float(g("lba_strength", 1.0))
        self.lba_ref_cells = float(g("lba_ref_cells", 8.0))
        self.lba_sigma = float(g("lba_sigma", 1.0))
        self.lba_log = bool(g("lba_log", True))

        # ---- EDGEW: fixed per-edge DFL weights (the PEU control) -------------
        # AR-DFL can only express [w, h, w, h] — it cannot separate top from
        # bottom, which is precisely the asymmetry the diagnostic found
        # (top 6.67% vs bottom 3.40% relative residual). A control for PEU needs
        # four INDEPENDENT weights, so it lives here rather than in AR-DFL.
        # Weights are mean-normalised, matching PEU, so the two arms differ only
        # in fixed-vs-learned, not in total DFL magnitude.
        self.use_edgew = bool(g("use_edgew", False))
        self.edgew_l = float(g("edgew_l", 1.0))
        self.edgew_t = float(g("edgew_t", 1.0))
        self.edgew_r = float(g("edgew_r", 1.0))
        self.edgew_b = float(g("edgew_b", 1.0))

        # !! MEASURED FAILURE MODE (peu_b05, first attempt) !!
        # detach=True + lambda>0 COLLAPSES. Detaching removes gradient from the
        # attenuation term, so the ONLY gradient reaching the variance is
        # lambda*log(var), which pushes var -> 0 with nothing opposing it. In
        # Kendall's form the L/var term blows up as var shrinks and balances it;
        # detaching destroys that balance. Observed: top-edge var fell 0.99 ->
        # 0.25 (the floor) within one epoch of warmup ending, all weights
        # converged to 1.000 +/- 0.017, and PEU became a no-op with a negative
        # constant offset on dfl_loss.
        #
        # The two VALID configurations:
        #   detach=True   -> lambda MUST be 0. The net cannot game a detached
        #                    weight, so no penalty is needed. Weights follow the
        #                    natural variance the DFL CE already produces.
        #   detach=False  -> lambda > 0. True Kendall attenuation, self-balancing.
        self.use_peu = bool(g("use_peu", False))
        self.peu_beta = float(g("peu_beta", 0.5))
        self.peu_lambda = float(g("peu_lambda", 0.0))
        self.peu_detach = bool(g("peu_detach", True))
        self.peu_warmup_epochs = int(g("peu_warmup_epochs", 5))
        # natural per-edge variance measured in-training is ~0.2-1.0, so a floor
        # of 0.25 clipped most of the useful signal. peu_w_clip bounds the weight
        # anyway, so the floor only needs to prevent log(0).
        self.peu_min_var = float(g("peu_min_var", 0.05))
        self.peu_w_clip = float(g("peu_w_clip", 3.0))
        self.peu_log = bool(g("peu_log", True))

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
        if not 0.05 <= self.adfl_w_scale <= 4.0:
            raise ValueError(f"adfl_w_scale must be in [0.05, 4.0], got {self.adfl_w_scale}")
        if not 0.05 <= self.adfl_h_scale <= 4.0:
            raise ValueError(f"adfl_h_scale must be in [0.05, 4.0], got {self.adfl_h_scale}")
        if self.use_adfl and self.adfl_w_scale == 1.0 and self.adfl_h_scale == 1.0:
            raise ValueError(
                "use_adfl=True but both scales are 1.0 — that is stock DFL. "
                "Set adfl_w_scale < 1.0 (finer width bins) or leave use_adfl=False."
            )
        if self.peu_beta < 0 or self.peu_lambda < 0:
            raise ValueError("peu_beta / peu_lambda must be >= 0")
        if self.use_peu and self.peu_beta == 0.0 and self.peu_lambda == 0.0:
            raise ValueError("use_peu=True but beta and lambda are both 0 — that is stock DFL.")
        if self.use_peu and self.peu_detach and self.peu_lambda > 0:
            raise ValueError(
                "peu_detach=True with peu_lambda>0 is a degenerate configuration and was "
                "MEASURED to collapse: detaching removes the counterbalancing gradient, so "
                "lambda*log(var) drives every variance to peu_min_var, all weights converge "
                "to 1.0 and PEU becomes a no-op with a negative offset on dfl_loss.\n"
                "  Use peu_detach=True  with peu_lambda=0.0  (weights follow natural variance), "
                "or peu_detach=False with peu_lambda>0 (true Kendall, self-balancing)."
            )
        if self.use_peu and not self.peu_detach and self.peu_lambda == 0.0:
            raise ValueError(
                "peu_detach=False with peu_lambda=0 lets the network minimise its loss by "
                "inflating variance without penalty. Set peu_lambda > 0."
            )
        if self.peu_min_var <= 0:
            raise ValueError("peu_min_var must be > 0")
        if self.peu_w_clip < 1.0:
            raise ValueError(f"peu_w_clip must be >= 1.0, got {self.peu_w_clip}")
        if self.use_lba:
            if self.lba_strength <= 0:
                raise ValueError("use_lba=True but lba_strength<=0 — that is stock TAL.")
            if self.lba_sigma <= 0:
                raise ValueError("lba_sigma must be > 0")
            if self.lba_ref_cells <= 0:
                raise ValueError("lba_ref_cells must be > 0")
        if self.use_edgew:
            ew = (self.edgew_l, self.edgew_t, self.edgew_r, self.edgew_b)
            if min(ew) < 0:
                raise ValueError("edgew_* must be >= 0")
            if sum(ew) == 0:
                raise ValueError("all edgew_* are 0")
            if sum(1 for m in (self.use_peu, self.use_ardfl) if m):
                raise ValueError(
                    "use_edgew is the FIXED-weight control for PEU. Enabling it together "
                    "with use_peu or use_ardfl makes the comparison unattributable."
                )

    def edgew_vec(self):
        """Mean-normalised fixed per-edge weights (l, t, r, b), or None."""
        if not self.use_edgew:
            return None
        ew = [self.edgew_l, self.edgew_t, self.edgew_r, self.edgew_b]
        m = sum(ew) / 4.0
        return [v / m for v in ew]
        if self.use_peu and self.use_ardfl:
            raise ValueError(
                "use_peu and use_ardfl both set. PEU derives per-edge weights from the "
                "predicted uncertainty; AR-DFL imposes fixed ones. Running both makes the "
                "result unattributable — pick one."
            )

    def adfl_scales(self):
        """Per-edge range scale in bbox2dist order (left, top, right, bottom)."""
        if not self.use_adfl:
            return None
        return (self.adfl_w_scale, self.adfl_h_scale, self.adfl_w_scale, self.adfl_h_scale)

    def is_neutral(self):
        """True if this config reproduces stock Ultralytics loss."""
        return (self.box_loss_type == "ciou"
                and not self.use_satal and self.swa_alpha == 0.0 and self.swa_boost == 1.0
                and not self.use_nwd and not self.use_class_weights and not self.use_vfl
                and not self.use_loss_clip and not self.use_ardfl and not self.use_adfl
                and not self.use_peu and not self.use_edgew and not self.use_lba)

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


# =============================================================================
# NWD
# =============================================================================
# Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
# https://arxiv.org/abs/2110.13389
# Box -> 2D Gaussian, mu = center, Sigma = diag((w/2)^2, (h/2)^2)
# W2^2 = ||mu1-mu2||^2 + ||sigma1-sigma2||^2 ;  NWD = exp(-sqrt(W2^2)/C)
#
# C is in PIXELS here (paper uses ~12.8 px for AI-TOD). v1 applied C to
# stride-normalized coords, which silently meant a different physical scale on
# every FPN level.

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
# Dataset is 94% tall (mean h/w 2.69, 33x72px), so metrics with EXPLICIT width
# and height error terms are better matched than CIoU, whose aspect term goes to
# ~0 whenever the h/w RATIO matches even if absolute width is wrong.
#
#   ciou   : stock (baseline)
#   eiou   : IoU - center - w_err - h_err            (Zhang 2022)
#   siou   : IoU - 0.5*(distance + shape)            (Gevorgyan 2022)
#   mpdiou : IoU - d1^2/d^2 - d2^2/d^2               (Ma & Xu 2023) corner-based
#   wiou   : WIoU v3, outlier-aware dynamic focusing (Tong 2023)


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
        # Corner-distance form; normalizer is the enclosing-box diagonal.
        d1 = (g["px1"] - g["tx1"]) ** 2 + (g["py1"] - g["ty1"]) ** 2
        d2 = (g["px2"] - g["tx2"]) ** 2 + (g["py2"] - g["ty2"]) ** 2
        d = g["cw"] ** 2 + g["ch"] ** 2 + eps
        return 1.0 - (iou - d1 / d - d2 / d)

    raise ValueError(f"unknown box_loss_type: {kind!r}")


def _wiou_v3(pred, target, running_mean, alpha, delta, eps=1e-7):
    """
    WIoU v3 (Tong et al. 2023). Returns (loss (N,), new_running_mean).

    L_WIoUv1 = R * L_IoU, R = exp(center_dist^2 / enclosing_diag^2) with the
    enclosing box DETACHED (this is what stops the penalty producing gradient
    that fights the IoU term).
    v3 scales by an outlier-degree gradient gain r = beta / (delta * alpha^(beta-delta)),
    beta = L_IoU / mean(L_IoU), which de-emphasises both easy and very-low-quality
    anchors. `running_mean` is the momentum-smoothed mean of L_IoU.
    """
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
    """
    Bounded per-object weight in [1, swa_boost].

    v1 used 1/area normalized by the batch max -> up to 400:1 spread on this
    dataset, batch-dependent scale. This is linear and bounded: `boost` at
    size 0, decaying to 1.0 at the threshold.

    axis="width" is the default because the dataset is 94% tall (mean 33x72px):
    a thin trolley has medium AREA but small WIDTH, so an area-keyed weight
    under-serves exactly the hard objects.
    """
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
    """Distribution Focal Loss."""

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


# =============================================================================
# ANISOTROPIC DFL (A-DFL)
# =============================================================================
# Per-edge range scaling of the DFL bin grid. See SataLSwaConfig for the
# rationale. Edge order is bbox2dist's: (left, top, right, bottom).
#
#   encode:  t_e = d_e / s_e      decode:  d_e = E[bin] * s_e
#
# With s_e < 1 the same reg_max bins cover a shorter range, i.e. finer spacing.
# The scale is a fixed constant vector, so encode/decode are exact inverses and
# the neutral setting (all ones) is bit-identical to stock.

_ADFL_CLAMP_STATE = {"n": 0, "sat": None, "epoch_logged": -1}


def _adfl_vec(scales, device, dtype):
    """(1,4) tensor of per-edge scales, or None."""
    if scales is None:
        return None
    return torch.tensor(scales, device=device, dtype=dtype).view(1, 4)


def adfl_encode(target_ltrb, scales, reg_max, track=False):
    """Distances (…,4) in stride units -> DFL bin targets under per-edge scaling.

    Returns the scaled target. Saturation (targets pushed past the last bin) is
    the one real risk of compressing a range, so it is tracked and reported
    rather than left silent.
    """
    if scales is None:
        return target_ltrb
    v = _adfl_vec(scales, target_ltrb.device, target_ltrb.dtype)
    t = target_ltrb / v
    if track and t.numel():
        with torch.no_grad():
            sat = (t >= (reg_max - 1 - 0.02)).float().mean(0) if t.dim() == 2 else \
                  (t >= (reg_max - 1 - 0.02)).float().reshape(-1, 4).mean(0)
            s = _ADFL_CLAMP_STATE
            s["sat"] = sat if s["sat"] is None else s["sat"] + sat
            s["n"] += 1
    return t


def adfl_decode(dist_bins, scales):
    """DFL expectation (…,4) in bin units -> distances in stride units."""
    if scales is None:
        return dist_bins
    return dist_bins * _adfl_vec(scales, dist_bins.device, dist_bins.dtype)


def adfl_clamp_report(reset=True):
    """Mean per-edge saturation rate since the last call, or None."""
    s = _ADFL_CLAMP_STATE
    if not s["n"] or s["sat"] is None:
        return None
    rates = (s["sat"] / s["n"]).tolist()
    if reset:
        s["n"], s["sat"] = 0, None
    return dict(zip(("left", "top", "right", "bottom"), rates))


# =============================================================================
# LEVEL-BALANCED ASSIGNMENT (LBA)
# =============================================================================
# TAL ranks candidates by score^alpha * iou^beta with no notion of which FPN
# level a candidate came from. Measured consequence on this dataset: P5 receives
# 1.3% of foreground against a 4.8% grid share (ratio 0.28) while P4 takes 39.4%
# against 19.0% (ratio 2.07). One pyramid level is effectively untrained, and
# large objects — which should own it — are the worst-performing bucket.
#
# LBA multiplies the alignment metric by a soft scale-matching prior so each
# level preferentially receives the objects whose size matches its resolution.

_LBA_STATE = {"n": 0, "fg_per_level": None, "strides": None}


def _lba_prior(gt_bboxes, stride_per_anchor, ref_cells, sigma, eps=1e-9):
    """Soft scale-matching prior, shape (bs, n_max_boxes, n_anchors), in (0, 1].

    gt_bboxes         : (bs, n, 4) xyxy in PIXELS
    stride_per_anchor : (n_anchors,) pixels per cell at that anchor's level
    """
    w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=eps)
    h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=eps)
    size = (w * h).sqrt().unsqueeze(-1)                       # (bs, n, 1)
    nominal = (stride_per_anchor * ref_cells).view(1, 1, -1)  # (1, 1, a)
    octaves = torch.log2(size / nominal.clamp(min=eps))
    return torch.exp(-(octaves ** 2) / (2.0 * sigma ** 2))


def _lba_track(fg_mask, stride_per_anchor):
    """Accumulate foreground counts per pyramid level — the hypothesis test."""
    with torch.no_grad():
        st = _LBA_STATE
        uniq = torch.unique(stride_per_anchor)
        counts = torch.stack([((stride_per_anchor.view(1, -1) == s) & fg_mask).sum()
                              for s in uniq]).float()
        st["strides"] = uniq.tolist()
        st["fg_per_level"] = counts if st["fg_per_level"] is None else st["fg_per_level"] + counts
        st["n"] += 1


def lba_report(reset=True):
    """Foreground share per FPN level since the last call."""
    st = _LBA_STATE
    if not st["n"] or st["fg_per_level"] is None:
        return None
    c = st["fg_per_level"]
    tot = c.sum().clamp(min=1)
    out = {int(s): {"fg": int(v.item()), "share": float((v / tot).item())}
           for s, v in zip(st["strides"], c)}
    if reset:
        st["n"], st["fg_per_level"] = 0, None
    return out


class LevelBalancedTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with a soft scale-matching prior over FPN levels.

    `stride_tensor` must be set on the instance before __call__ (v8DetectionLoss
    does this); without it the assigner degrades to stock TAL and says so once.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 strength=1.0, ref_cells=8.0, sigma=1.0, eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.strength, self.ref_cells, self.sigma = strength, ref_cells, sigma
        self.stride_tensor = None
        self._warned = False

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        align, overlaps = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        s = self.stride_tensor
        if s is None:
            if not self._warned:
                print("[LBA] WARNING stride_tensor not set — running stock TAL")
                self._warned = True
            return align, overlaps
        prior = _lba_prior(gt_bboxes, s.view(-1).to(align.dtype), self.ref_cells, self.sigma)
        return align * prior.pow(self.strength).to(align.dtype), overlaps


# =============================================================================
# PEU-DFL — per-edge uncertainty from the DFL distribution itself
# =============================================================================
# DFL predicts a categorical distribution over reg_max bins for each of the four
# edges, then keeps only its MEAN (the integral / decode). Its VARIANCE is
# computed for free by the same softmax and is thrown away. That variance is an
# aleatoric uncertainty estimate: a wide distribution means the network cannot
# commit to an edge location. On this dataset that is exactly the top edge
# (handles, straps, occlusion) — measured at 2.26x the bottom-edge residual.

_PEU_STATE = {"n": 0, "var": None, "w": None, "loss": None}


def _dfl_edge_moments(pred_dist, reg_max):
    """(N*4, reg_max) logits -> per-edge (mu, var) in BIN units, each (N, 4).

    mu is exactly the decode used by bbox_decode; var is its second central
    moment under the same distribution.
    """
    p = pred_dist.softmax(-1)                                   # (N*4, reg_max)
    idx = torch.arange(reg_max, device=pred_dist.device, dtype=p.dtype)
    mu = (p * idx).sum(-1)                                      # (N*4,)
    var = (p * (idx.unsqueeze(0) - mu.unsqueeze(-1)) ** 2).sum(-1)
    return mu.view(-1, 4), var.view(-1, 4)


def _peu_weights(var, cfg, warmed_up):
    """Mean-normalised per-edge attenuation weights, and the log-variance term.

    w_e = exp(-beta * log var_e), renormalised so w.mean() == 1. The
    normalisation is deliberate: PEU must REDISTRIBUTE DFL weight across edges,
    not scale the total. Otherwise any gain is confounded with a larger dfl gain.
    """
    s = torch.log(var.clamp(min=cfg.peu_min_var))               # (N,4) log-variance
    if not warmed_up:
        return torch.ones_like(s), s
    s_w = s.detach() if cfg.peu_detach else s
    w = torch.exp(-cfg.peu_beta * s_w)
    # Bound the spread BEFORE normalising. Mean-normalisation fixes the mean but
    # not the tail: an almost-one-hot edge would otherwise take a weight of
    # 1/peu_min_var and dominate the batch (measured 86x at beta=1 unclipped).
    k = cfg.peu_w_clip
    w = w.clamp(min=1.0 / k, max=k)
    w = w / w.mean().clamp(min=1e-9)
    return w, s


def _peu_track(var, w, per_edge_loss):
    st = _PEU_STATE
    with torch.no_grad():
        for k, v in (("var", var), ("w", w), ("loss", per_edge_loss)):
            m = v.detach().mean(0)
            st[k] = m if st[k] is None else st[k] + m
        st["n"] += 1


def peu_report(reset=True):
    """Mean per-edge variance / attenuation weight / DFL loss since last call.

    This is the experiment's own hypothesis test: if the top edge really carries
    the highest aleatoric uncertainty, its variance and DFL loss should be
    highest and its attenuation weight lowest.
    """
    st = _PEU_STATE
    if not st["n"] or st["var"] is None:
        return None
    names = ("left", "top", "right", "bottom")
    out = {k: dict(zip(names, (st[k] / st["n"]).tolist())) for k in ("var", "w", "loss")}
    if reset:
        st["n"], st["var"], st["w"], st["loss"] = 0, None, None, None
    return out


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
    Box loss with bounded SWA weighting and optional pixel-space NWD.

    Takes the SHARED config object — it has no defaults of its own, so the
    printed config and the computed loss cannot disagree (v1's core bug).

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

    # -- schedules -----------------------------------------------------------
    def _current_alpha(self):
        """SWA alpha. Anneals only if epoch tracking is attached AND an end is set."""
        cfg = self.cfg
        if cfg.swa_alpha_end is None:
            return cfg.swa_alpha
        progress, active = _get_progress(cfg.total_epochs)
        if not active:
            # FAIL-SAFE: midpoint, not the aggressive start value (v1's behaviour).
            return 0.5 * (cfg.swa_alpha + cfg.swa_alpha_end)
        return cfg.swa_alpha * (1 - progress) + cfg.swa_alpha_end * progress

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor):
        cfg = self.cfg
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # pixel-space copies for all size-dependent math
        stride_fg = stride_tensor.expand(target_bboxes.shape[0], -1, -1)[fg_mask]  # (N,1)
        pred_px = pred_fg * stride_fg
        target_px = target_fg * stride_fg

        # ---- weights --------------------------------------------------------
        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)      # (N,1)
        size_weight = _size_weight(target_px, cfg).unsqueeze(-1)          # (N,1) in [1,boost]

        if cfg.swa_mode == "scale":
            # multiplicative: modulate the quality signal, don't replace it
            weight = score_weight * size_weight
        else:  # "blend" — v1's additive form, kept for ablation
            a = self._current_alpha()
            weight = a * size_weight + (1.0 - a) * score_weight

        # ---- box regression loss ---------------------------------------------
        if cfg.box_loss_type == "wiou":
            base_loss, batch_mean = _wiou_v3(
                pred_fg, target_fg, self._wiou_mean, cfg.wiou_alpha, cfg.wiou_delta
            )
            # momentum update of the outlier-degree normalizer
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
        # Divide by the weights ACTUALLY used (v1 divided by target_scores_sum
        # regardless). With swa off this equals target_scores_sum exactly, because
        # target_scores is zero on background anchors.
        norm = weight.sum().clamp(min=1e-9)
        loss_iou = (box_loss * weight).sum() / norm

        # ---- DFL --------------------------------------------------------------
        if self.dfl_loss:
            # A-DFL: bbox2dist's clamp is applied in the SCALED space, so the
            # usable range per edge is reg_max-1 bins of width s_e stride units.
            # Encode first, then let DFLoss do its own final clamp.
            adfl = cfg.adfl_scales()
            if adfl is None:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            else:
                raw_ltrb = bbox2dist(anchor_points, target_bboxes, 1e9)   # unclamped
                target_ltrb = adfl_encode(raw_ltrb, adfl, self.reg_max,
                                          track=cfg.adfl_log_clamp)
            pred_dist_fg = pred_dist[fg_mask].view(-1, self.reg_max)   # (N*4, reg_max)
            target_fg_ltrb = target_ltrb[fg_mask]                      # (N, 4) = (l,t,r,b)

            if cfg.use_peu:
                # ---- PEU-DFL ------------------------------------------------
                # Attenuate each edge by the uncertainty the network itself
                # expresses in that edge's bin distribution.
                per_edge = self.dfl_loss.per_edge(pred_dist_fg, target_fg_ltrb)   # (N,4)
                _, var = _dfl_edge_moments(pred_dist_fg, self.reg_max)            # (N,4)

                # warmup: at init the bin distribution is near-uniform, so its
                # variance is meaningless and would suppress every edge equally.
                # If epoch tracking is unwired we cannot know the epoch, so we
                # attenuate from step 0 and say so loudly in _print_config.
                warmed = (not _EPOCH_STATE["ever_set"]) or \
                         (_EPOCH_STATE["epoch"] >= cfg.peu_warmup_epochs)
                w_edge, s = _peu_weights(var, cfg, warmed)

                if cfg.peu_log:
                    _peu_track(var, w_edge, per_edge)

                dfl_per_box = (per_edge * w_edge).mean(-1, keepdim=True)          # (N,1)
                if cfg.use_loss_clip:
                    dfl_per_box = dfl_per_box.clamp(max=cfg.dfl_clip)
                loss_dfl = (dfl_per_box * weight).sum() / norm

                # log-variance penalty — stops the net declaring every edge
                # uncertain. Normalised by the SAME weight/norm as the main term
                # so its relative scale is stable through training.
                if cfg.peu_lambda > 0 and warmed:
                    reg = s.mean(-1, keepdim=True)                                # (N,1)
                    loss_dfl = loss_dfl + cfg.peu_lambda * (reg * weight).sum() / norm

            elif cfg.use_edgew:
                # ---- EDGEW: fixed per-edge weights (the PEU control) --------
                # Four independent, mean-normalised weights. Differs from PEU
                # ONLY in fixed-vs-learned, so the comparison isolates whether
                # adaptivity matters or a static reweight is enough.
                per_edge = self.dfl_loss.per_edge(pred_dist_fg, target_fg_ltrb)   # (N,4)
                w_edge = torch.tensor(cfg.edgew_vec(), device=per_edge.device,
                                      dtype=per_edge.dtype).view(1, 4)
                if cfg.peu_log:
                    _peu_track(torch.zeros_like(per_edge),
                               w_edge.expand_as(per_edge), per_edge)
                dfl_per_box = (per_edge * w_edge).mean(-1, keepdim=True)
                if cfg.use_loss_clip:
                    dfl_per_box = dfl_per_box.clamp(max=cfg.dfl_clip)
                loss_dfl = (dfl_per_box * weight).sum() / norm

            elif cfg.use_ardfl:
                # AR-DFL: reweight per-edge DFL toward the HEIGHT edges (t,b),
                # where tall-object localization error concentrates. Columns:
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
                # r10_dfl_entropy (global sharpening) gave the best bag AP;
                # here it is targeted at the axis that actually needs tightening.
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
    """SATAL + SWA + NWD detection loss. All settings come from one config object."""

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

        # ---- ONE config, parsed once ---------------------------------------
        self.cfg = SataLSwaConfig(h, nc=self.nc, total_epochs=getattr(h, "epochs", 70))
        if getattr(h, "tal_topk", None) is None:
            self.cfg.tal_topk = tal_topk

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(m.reg_max, cfg=self.cfg).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # ---- class weights: hyp-driven and ablatable ------------------------
        self.class_weights = None
        if self.cfg.use_class_weights:
            counts = self.cfg.class_counts
            if counts is None:
                raise ValueError(
                    "use_class_weights=True requires class_counts=[n0,n1,...] in hyp "
                    "(in data.yaml names[] order). v1 hardcoded counts that did not "
                    "match the dataset; that is no longer allowed."
                )
            c = torch.tensor([float(x) for x in counts], device=device)
            inv = 1.0 / c
            inv = inv / inv.mean()
            w = torch.sqrt(inv)
            self.class_weights = (w / w.mean()).view(1, 1, -1)

        # ---- assigner --------------------------------------------------------
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
        elif self.cfg.use_lba:
            self.assigner = LevelBalancedTaskAlignedAssigner(
                topk=self.cfg.tal_topk, num_classes=self.nc,
                alpha=self.cfg.tal_alpha, beta=self.cfg.tal_beta,
                strength=self.cfg.lba_strength, ref_cells=self.cfg.lba_ref_cells,
                sigma=self.cfg.lba_sigma,
            )
        else:
            self.assigner = TaskAlignedAssigner(
                topk=self.cfg.tal_topk, num_classes=self.nc,
                alpha=self.cfg.tal_alpha, beta=self.cfg.tal_beta,
            )

        self._print_config()

    # ---------------------------------------------------------------------
    def verify_config(self, verbose=True):
        """
        Report the LIVE state of the objects that actually compute the loss.

        v1's failure mode was a config printout that disagreed with reality
        (use_nwd printed True while BboxLoss had it False). Everything here is
        read back off the live objects, not off the hyp dict.
        """
        live = {
            "assigner_class": type(self.assigner).__name__,
            "satal_active": type(self.assigner).__name__.startswith("ScaleAdaptive"),
            "bbox_loss_cfg_is_shared": self.bbox_loss.cfg is self.cfg,
            "box_loss_type_live": self.bbox_loss.cfg.box_loss_type,
            "use_nwd_live": self.bbox_loss.cfg.use_nwd,
            "swa_mode_live": self.bbox_loss.cfg.swa_mode,
            "swa_alpha_live": self.bbox_loss.cfg.swa_alpha,
            "swa_boost_live": self.bbox_loss.cfg.swa_boost,
            "class_weights_live": None if self.class_weights is None
            else self.class_weights.flatten().tolist(),
            "epoch_tracking_attached": _EPOCH_STATE["ever_set"],
            "is_neutral_stock_equivalent": self.cfg.is_neutral(),
        }
        assert live["bbox_loss_cfg_is_shared"], "BboxLoss is not sharing the config object!"
        assert live["use_nwd_live"] == self.cfg.use_nwd, "use_nwd mismatch (the v1 bug)!"
        assert live["box_loss_type_live"] == self.cfg.box_loss_type, "box_loss_type mismatch!"
        assert live["satal_active"] == self.cfg.use_satal, "SATAL flag does not match assigner!"
        if verbose:
            print("\n[verify_config] live loss state")
            for k, v in live.items():
                print(f"    {k:32s} {v}")
            if self.cfg.swa_alpha_end is not None and not _EPOCH_STATE["ever_set"]:
                print("    !! alpha annealing requested but epoch tracking NOT attached")
                print("       -> call attach_epoch_tracking(model) before train()")
            print()
        return live

    def _print_config(self):
        c = self.cfg
        print("\n" + "=" * 62)
        print("  SATAL-SWA-NWD Loss v2")
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
        print(f"  LBA (level-balanced):     {c.use_lba}")
        if c.use_lba:
            print(f"    strength / sigma:       {c.lba_strength} / {c.lba_sigma} octaves")
            print(f"    nominal obj per level:  {c.lba_ref_cells} cells "
                  f"(= {[int(c.lba_ref_cells*s) for s in (8,16,32)]} px at stride 8/16/32)")
            print("    measured pathology:     P5 fg share 1.3% vs 4.8% grid (ratio 0.28)")
        print(f"  EDGEW (fixed per-edge):   {c.use_edgew}"
              + (f"  l/t/r/b = {[round(v,3) for v in c.edgew_vec()]}" if c.use_edgew else ""))
        print(f"  PEU-DFL (uncertainty):    {c.use_peu}")
        if c.use_peu:
            print(f"    beta / lambda:          {c.peu_beta} / {c.peu_lambda}")
            print(f"    min var / weight clip:  {c.peu_min_var} / [{1/c.peu_w_clip:.3f}, {c.peu_w_clip}]")
            print(f"    variance detached:      {c.peu_detach}")
            print(f"    warmup epochs:          {c.peu_warmup_epochs}"
                  + ("" if _EPOCH_STATE["ever_set"] else "   !! epoch tracking NOT attached "
                                                          "-> warmup disabled, attenuation from step 0"))
            print(f"    weights mean-normalised: True (redistributes, does not rescale DFL)")
        print(f"  A-DFL (anisotropic):      {c.use_adfl}")
        if c.use_adfl:
            rm = self.reg_max
            print(f"    w/h range scale:        {c.adfl_w_scale} / {c.adfl_h_scale}")
            print(f"    width  bins:            {rm} over {c.adfl_w_scale*(rm-1):.2f} stride units "
                  f"({c.adfl_w_scale:.3f} stride/bin)")
            print(f"    height bins:            {rm} over {c.adfl_h_scale*(rm-1):.2f} stride units "
                  f"({c.adfl_h_scale:.3f} stride/bin)")
            print(f"    width resolution gain:  {1.0/max(c.adfl_w_scale,1e-9):.2f}x vs stock")
            print("    !! DFL.forward must apply the SAME scale at inference")
            print("       -> python adfl_patch_dfl.py --check")
        print(f"  epoch tracking attached:  {_EPOCH_STATE['ever_set']}")
        print("=" * 62 + "\n")

    # ---------------------------------------------------------------------
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
            # A-DFL: bins -> stride units. Must mirror DFL.forward at inference
            # (adfl_patch_dfl.py) or training and detection disagree.
            pred_dist = adfl_decode(pred_dist, self.cfg.adfl_scales())
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_cls_loss(self, pred_scores, target_scores, target_scores_sum, dtype):
        """BCE, optionally VFL-modulated and class-weighted. Neutral config -> stock BCE."""
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

        # LBA needs the per-anchor stride to know which pyramid level each
        # candidate belongs to. TaskAlignedAssigner's signature does not carry
        # it, so hand it over explicitly here.
        if self.cfg.use_lba:
            self.assigner.stride_tensor = stride_tensor

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        if self.cfg.use_lba and self.cfg.lba_log:
            _lba_track(fg_mask, stride_tensor)

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
# OTHER TASK LOSSES
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
        batch_size, _, mask_h, mask_w = proto.shape  # from proto, not masks (v1 bug)
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


class E2EDetectLoss:
    """End-to-end detection loss. (v1 referenced an undefined class here.)"""

    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        l_many = self.one2many(preds["one2many"], batch)
        l_one = self.one2one(preds["one2one"], batch)
        return l_many[0] + l_one[0], l_many[1] + l_one[1]



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