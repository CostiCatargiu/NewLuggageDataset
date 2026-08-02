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
#
# -----------------------------------------------------------------------------
# FIX/IMPROVEMENT PATCH (review round):
#   F1  DetectObjLoss train path was dead on arrival: it called
#       self._compute_cls_loss (does not exist in this file — carried over from
#       a sibling lineage) and passed stride_tensor to BboxLoss.forward, which
#       takes no stride arg here. Eval path worked, so it survived val smoke
#       tests and died on the first TRAIN step. Fixed to stock cls BCE and the
#       correct 7-arg bbox call.
#   F2  PEU warmup is now FAIL-CLOSED: warmed requires _EPOCH["set"] AND
#       epoch >= warmup. Previously a missing epoch callback meant attenuating
#       from step 0 on init-time near-uniform distributions — the exact case
#       warmup exists to prevent. The "epoch tracking not attached" warning
#       moved OUT of the banner (which prints at __init__, before any callback
#       can have fired, so it ALWAYS false-alarmed) and into the forward pass:
#       it fires once, only if the callback is genuinely absent after ~500
#       criterion calls.
#   F3  PEU detach=False + lambda>0 is now TRUE per-edge Kendall:
#       L_e * exp(-beta*s_e) + lambda*s_e, UNCENTERED. The previous code
#       centred s for this config too, which (verified numerically) makes the
#       weights exactly invariant to a uniform variance shift — lambda then
#       drives global variance to the min_var floor with zero counter-pressure,
#       and the "self-balancing" comment was no longer true. Uncentred Kendall
#       restores the balance (beta*L*w == lambda at equilibrium). A degeneration
#       monitor warns once if mean variance camps at the floor anyway.
#   F4  peu_center: 'global' (legacy) | 'edge'. 'global' centres/normalises the
#       redistribution weights over the whole (N,4) batch, which ALSO
#       down-weights wholly-uncertain OBJECTS (implicit self-paced sampling —
#       a second mechanism hiding inside the first). 'edge' centres per row and
#       tests the pure across-edge redistribution hypothesis stated in the
#       original comment. Default 'global' preserves prior-run comparability.
#   F5  PEU/LBA diagnostic state moved from module globals onto the criterion
#       instances — with E2EDetectLoss the one2many and one2one criteria used
#       to blend into the same dicts, corrupting peu_report()/lba_report().
#       Module-level report functions remain as shims (last-constructed
#       criterion); for E2E call criterion.one2many.lba_report() etc.
#   F6  Banner prints once per process (E2E constructs two criteria from the
#       same hyp and printed two identical banners).
#   F7  AR-DFL ardfl_mode: 'fixed' (legacy) | 'per_box'. 'fixed' bakes the
#       dataset-median AR into every box, square ones included; 'per_box' sets
#       each row's width/height edge ratio to clamp((h/w)^power, 1/clip, clip),
#       row-mean-normalised, so the weighting follows each GT's own geometry.
#       Run 'fixed' first (cleaner hypothesis test); 'per_box' is the follow-up
#       if it wins.
#
# New hyp keys added by this patch:
#   peu_center ('global'|'edge'), ardfl_mode ('fixed'|'per_box'),
#   ardfl_ratio_power (default 1.0), ardfl_ratio_clip (default 3.0)
# -----------------------------------------------------------------------------

_EPOCH = {"epoch": 0, "total": 0, "set": False}
_PRINTED_BANNERS = set()  # F6 (content-keyed: distinct configs print, duplicates suppressed)


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
        # F7: 'fixed' (legacy, the two weights above) | 'per_box' (each row's
        # width/height ratio = clamp((h/w)^power, 1/clip, clip), row-normalised
        # — follows each GT's own geometry instead of the dataset median).
        self.ardfl_mode = str(g("ardfl_mode", "fixed"))
        self.ardfl_ratio_power = float(g("ardfl_ratio_power", 1.0))
        self.ardfl_ratio_clip = float(g("ardfl_ratio_clip", 3.0))

        # ---- PEU-DFL --------------------------------------------------------
        # DFL emits a distribution per edge; its variance is a free aleatoric
        # uncertainty estimate. L_e <- L_e * exp(-beta * s_e) + lambda * s_e.
        #
        # peu_norm_by_mu: variance in BIN units grows with target magnitude, so
        # raw var attenuates LARGE objects hardest — measured as monotonic
        # large-object damage (beta=1.0 cost -3.03 mAP50-95_large). Dividing by
        # mu^2 makes the signal scale-free. Default True (the fix).
        #
        # VALID COMBINATIONS ONLY (enforced below) — the two combos are now two
        # DIFFERENT mechanisms (F3):
        #   detach=True,  lambda=0 : REDISTRIBUTION mode. Detached, centred,
        #       clipped, mean-normalised weights — moves DFL weight between
        #       edges, total unchanged. Nothing can game a detached weight.
        #   detach=False, lambda>0 : KENDALL mode. TRUE per-edge Kendall,
        #       L_e*exp(-beta*s_e) + lambda*s_e, UNCENTRED and UNnormalised —
        #       centring would make the weights invariant to a uniform variance
        #       shift (verified numerically), leaving lambda to drive global
        #       variance to the floor with zero counter-pressure. Uncentred, the
        #       balance beta*L*w == lambda holds. NOTE: total DFL magnitude
        #       self-adjusts in this mode by construction — a gain here is a
        #       joint (weighting + effective-gain) effect, unlike the other mode.
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
        # F4: centring scope for REDISTRIBUTION mode only.
        #   'global' (legacy): centre/normalise over the whole (N,4) batch —
        #       also down-weights wholly-uncertain OBJECTS (implicit self-paced
        #       sampling mixed into the edge redistribution).
        #   'edge': centre/normalise per row — pure across-edge redistribution,
        #       per-sample DFL total preserved; isolates the stated hypothesis.
        self.peu_center = str(g("peu_center", "global"))

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
        if self.ardfl_mode not in ("fixed", "per_box"):
            raise ValueError("ardfl_mode must be 'fixed' or 'per_box'")
        if self.use_ardfl and self.ardfl_mode == "fixed" and self.ardfl_w_weight == self.ardfl_h_weight:
            raise ValueError(
                f"use_ardfl=True (fixed) but w==h=={self.ardfl_w_weight} — that is stock DFL."
            )
        if self.ardfl_w_weight < 0 or self.ardfl_h_weight < 0:
            raise ValueError("ardfl weights must be >= 0")
        if self.use_ardfl and self.ardfl_mode == "per_box":
            if self.ardfl_ratio_clip < 1.0:
                raise ValueError("ardfl_ratio_clip must be >= 1.0")
            if self.ardfl_ratio_power <= 0:
                raise ValueError(
                    "ardfl_mode='per_box' with ardfl_ratio_power<=0 — power 0 is stock "
                    "DFL for every box; use a positive power."
                )
        if self.use_peu:
            if self.peu_beta < 0 or self.peu_lambda < 0:
                raise ValueError("peu_beta / peu_lambda must be >= 0")
            if self.peu_beta == 0 and self.peu_lambda == 0:
                raise ValueError("use_peu=True but beta and lambda are both 0 — that is stock.")
            if self.peu_detach and self.peu_lambda > 0:
                raise ValueError(
                    "peu_detach=True with peu_lambda>0 is the COLLAPSING config "
                    "(measured -4.33 / -7.31). Use detach=True with lambda=0 "
                    "(redistribution mode) or detach=False with lambda>0 (Kendall mode)."
                )
            if not self.peu_detach and self.peu_lambda == 0:
                raise ValueError(
                    "peu_detach=False with peu_lambda=0 lets the net inflate variance "
                    "to cut its own loss (uncentred Kendall weights, nothing opposing). "
                    "Set peu_lambda > 0."
                )
            if self.peu_min_var <= 0:
                raise ValueError("peu_min_var must be > 0")
            if self.peu_w_clip < 1.0:
                raise ValueError("peu_w_clip must be >= 1.0")
            if self.peu_center not in ("global", "edge"):
                raise ValueError("peu_center must be 'global' or 'edge'")
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
            out.append(f"    mode:               {self.ardfl_mode}")
            if self.ardfl_mode == "fixed":
                out.append(f"    w / h weight:       {self.ardfl_w_weight} / {self.ardfl_h_weight}")
            else:
                out.append(f"    ratio power / clip: {self.ardfl_ratio_power} / {self.ardfl_ratio_clip}")
        out.append(f"  PEU-DFL:              {self.use_peu}")
        if self.use_peu:
            mode = "redistribution" if self.peu_detach else "kendall"
            out.append(f"    mode:               {mode}"
                       + (f" (center={self.peu_center})" if self.peu_detach else ""))
            out.append(f"    beta / lambda:      {self.peu_beta} / {self.peu_lambda}")
            out.append(f"    detach / norm_mu:   {self.peu_detach} / {self.peu_norm_by_mu}")
            # F2: warmup is FAIL-CLOSED — PEU stays neutral until set_epoch()
            # has been called AND epoch >= warmup. The not-attached warning now
            # fires from the forward pass (where it can actually be evaluated),
            # not here: this banner prints at __init__, before any
            # on_train_epoch_start callback can possibly have run.
            out.append(f"    warmup epochs:      {self.peu_warmup_epochs}  (fail-closed)")
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
# F5: diagnostic state lives on the criterion instances now (module globals
# blended one2many/one2one under E2EDetectLoss). _LAST_PEU_HOST is a
# convenience pointer so the module-level peu_report() shim keeps working for
# the common single-criterion case.
_LAST_PEU_HOST = None


def _new_peu_state():
    return {"n": 0, "var": None, "w": None}


def _dfl_edge_moments(pred_dist, reg_max):
    """(N*4, reg_max) logits -> per-edge (mu, var) in BIN units, each (N, 4)."""
    p = pred_dist.softmax(-1)
    idx = torch.arange(reg_max, device=pred_dist.device, dtype=p.dtype)
    mu = (p * idx).sum(-1)
    var = (p * (idx.unsqueeze(0) - mu.unsqueeze(-1)) ** 2).sum(-1)
    return mu.view(-1, 4), var.view(-1, 4)


def _peu_weights(mu, var, cfg, warmed):
    """Per-edge weights and the log-variance term, mode-dependent (F3).

    REDISTRIBUTION mode (detach=True): detached, centred, clipped,
    mean-normalised weights — moves DFL weight ACROSS edges, total unchanged,
    so a gain is not confounded with a larger effective dfl gain.
    peu_center (F4) selects the centring/normalisation scope:
      'global' — over the whole (N,4) batch (legacy): also down-weights wholly
                 uncertain objects (implicit self-paced sampling).
      'edge'   — per row: pure across-edge redistribution, per-sample total
                 preserved.

    KENDALL mode (detach=False, lambda>0): TRUE per-edge Kendall weights
    exp(-beta*s_e), UNCENTRED and UNnormalised. Centring here would make the
    weights exactly invariant to a uniform variance shift (verified
    numerically), so lambda*s would drive global variance to the min_var floor
    with zero counter-pressure. Uncentred, beta*L*w == lambda at equilibrium.
    The clip is a symmetric stability rail only. Total DFL magnitude
    self-adjusts by construction in this mode.
    """
    v = var.clamp(min=cfg.peu_min_var)
    if cfg.peu_norm_by_mu:
        # scale-free: raw bin-unit variance grows with target magnitude, so it
        # would attenuate large objects rather than uncertain edges.
        v = v / mu.detach().clamp(min=1.0) ** 2
    s = torch.log(v)
    if not warmed:
        return torch.ones_like(s), s

    if cfg.peu_detach:
        # ---- REDISTRIBUTION -------------------------------------------------
        s_w = s.detach()
        # Centre the log-variance BEFORE exponentiating. peu_w_clip is meant to
        # bound the RELATIVE spread across edges; without centring,
        # peu_norm_by_mu divides every edge by ~mu^2, pushes every weight past
        # the clip together, and PEU silently degenerates. Verified numerically.
        if cfg.peu_center == "edge":
            s_w = s_w - s_w.mean(dim=-1, keepdim=True)          # per row (F4)
        else:
            s_w = s_w - s_w.mean()                              # global (legacy)
        w = torch.exp(-cfg.peu_beta * s_w)
        k = cfg.peu_w_clip
        w = w.clamp(min=1.0 / k, max=k)      # bound the spread BEFORE normalising
        if cfg.peu_center == "edge":
            w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            w = w / w.mean().clamp(min=1e-9)
        return w, s

    # ---- KENDALL (F3) -------------------------------------------------------
    # Uncentred, gradient flows through s; symmetric clamp as a stability rail.
    w = torch.exp(-cfg.peu_beta * s)
    k = cfg.peu_w_clip
    w = w.clamp(min=1.0 / k, max=k)
    return w, s


def _peu_track(state, var, w):
    with torch.no_grad():
        for key, val in (("var", var), ("w", w)):
            m = val.mean(0)
            state[key] = m if state[key] is None else state[key] + m
        state["n"] += 1


def _peu_report_from(state, reset=True):
    if not state["n"]:
        return None
    names = ("left", "top", "right", "bottom")
    out = {"var": dict(zip(names, (state["var"] / state["n"]).tolist())),
           "weight": dict(zip(names, (state["w"] / state["n"]).tolist()))}
    if reset:
        state["n"], state["var"], state["w"] = 0, None, None
    return out


def peu_report(reset=True):
    """Mean per-edge variance and weight since the last call.

    Shim (F5): reads the LAST-CONSTRUCTED criterion's state. With
    E2EDetectLoss, call criterion.one2many.bbox_loss.peu_report() /
    criterion.one2one.bbox_loss.peu_report() instead — the shim would only
    show the last one built.
    """
    if _LAST_PEU_HOST is None:
        return None
    return _peu_report_from(_LAST_PEU_HOST._peu_state, reset)


# ---------------------------------------------------------------------------
# LBA helpers
# ---------------------------------------------------------------------------
# F5: state per criterion instance; _LAST_LBA_HOST backs the module-level shim.
_LAST_LBA_HOST = None


def _new_lba_state():
    return {"n": 0, "fg": None, "strides": None}


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


def _lba_track(state, fg_mask, stride_per_anchor, gt_bboxes=None, mask_gt=None, cfg=None):
    with torch.no_grad():
        uniq = torch.unique(stride_per_anchor)
        counts = torch.stack([((stride_per_anchor.view(1, -1) == s) & fg_mask).sum()
                              for s in uniq]).float()
        state["strides"] = uniq.tolist()
        state["fg"] = counts if state["fg"] is None else state["fg"] + counts
        state["n"] += 1
        # Gate pass-rate: fraction of REAL GTs above lba_size_gate_px, measured
        # on the SAME size axis as the prior, in AUGMENTED-image pixels — i.e.
        # exactly the population the gated prior acts on. Purpose: quantify the
        # mosaic interaction (mosaic shrinks objects, so the gate is effectively
        # stricter until close_mosaic; expect this rate to jump at that epoch).
        if cfg is not None and cfg.lba_size_gate_px > 0 and gt_bboxes is not None and mask_gt is not None:
            gm = mask_gt.squeeze(-1) > 0
            if gm.any():
                boxes = gt_bboxes[gm]
                w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-9)
                h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-9)
                size = torch.maximum(w, h) if cfg.lba_size_axis == "max" else (w * h).sqrt()
                state["gate_pass"] = state.get("gate_pass", 0.0) + float((size > cfg.lba_size_gate_px).sum())
                state["gate_total"] = state.get("gate_total", 0.0) + float(gm.sum())


def _lba_report_from(state, reset=True):
    if not state["n"] or state["fg"] is None:
        return None
    tot = state["fg"].sum().clamp(min=1)
    out = {int(s): {"fg": int(v.item()), "share": float((v / tot).item())}
           for s, v in zip(state["strides"], state["fg"])}
    if reset:
        state["n"], state["fg"] = 0, None
    return out


def _lba_gate_report_from(state, reset=True):
    """Gate pass-rate since the last call, or None if gate tracking inactive.

    Deliberately a SEPARATE report (not a key inside lba_report's per-stride
    dict): older runner scripts iterate that dict's int keys with sorted(),
    and a mixed-type key would crash them.
    """
    tot = state.get("gate_total", 0.0)
    if not tot:
        return None
    out = {"pass_rate": state.get("gate_pass", 0.0) / tot, "gts": int(tot)}
    if reset:
        state["gate_pass"], state["gate_total"] = 0.0, 0.0
    return out


def lba_gate_report(reset=True):
    """Module-level shim, same pattern as lba_report (last-constructed host)."""
    if _LAST_LBA_HOST is None:
        return None
    return _lba_gate_report_from(_LAST_LBA_HOST._lba_state, reset)


def lba_report(reset=True):
    """Foreground share per FPN level since the last call.

    Shim (F5): reads the LAST-CONSTRUCTED criterion's state. With
    E2EDetectLoss, call criterion.one2many.lba_report() /
    criterion.one2one.lba_report() instead.
    """
    if _LAST_LBA_HOST is None:
        return None
    return _lba_report_from(_LAST_LBA_HOST._lba_state, reset)


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
        # F5: per-instance PEU diagnostics (module globals blended E2E heads)
        self._peu_state = _new_peu_state()
        # F2: forward-pass epoch-callback check (the banner cannot evaluate it)
        self._peu_calls = 0
        self._peu_epoch_warned = False
        # F3: Kendall-mode degeneration monitor (variance camping at the floor)
        self._peu_floor_hits = 0
        self._peu_floor_warned = False
        if cfg is not None and cfg.use_peu:
            global _LAST_PEU_HOST
            _LAST_PEU_HOST = self

    def peu_report(self, reset=True):
        """Mean per-edge variance and weight since the last call (this instance)."""
        return _peu_report_from(self._peu_state, reset)

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
                # ---- AR-DFL: per-edge weights, mean-normalised --------------
                pd = pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max)
                per_edge = self.dfl_loss.per_edge(pd, target_ltrb[fg_mask])       # (N,4)
                if cfg.ardfl_mode == "per_box":
                    # F7: each row's width/height ratio follows its OWN box
                    # geometry: r = clamp((h/w)^power, 1/clip, clip). Ratio is
                    # scale-invariant, so feature-coord targets are fine.
                    tb = target_bboxes[fg_mask]                                   # (N,4) xyxy
                    bw = (tb[:, 2] - tb[:, 0]).clamp(min=1e-6)
                    bh = (tb[:, 3] - tb[:, 1]).clamp(min=1e-6)
                    r = (bh / bw).pow(cfg.ardfl_ratio_power).clamp(
                        min=1.0 / cfg.ardfl_ratio_clip, max=cfg.ardfl_ratio_clip)  # (N,)
                    # (left, top, right, bottom) = (width, height, width, height)
                    ew = torch.stack([r, torch.ones_like(r), r, torch.ones_like(r)], dim=-1)
                    ew = ew / ew.mean(dim=-1, keepdim=True)                       # per-row total unchanged
                else:
                    # 'fixed' (legacy): dataset-level weights
                    ew = torch.tensor(
                        [cfg.ardfl_w_weight, cfg.ardfl_h_weight,
                         cfg.ardfl_w_weight, cfg.ardfl_h_weight],
                        device=per_edge.device, dtype=per_edge.dtype,
                    ).view(1, 4)
                    ew = ew / ew.mean()                                           # total DFL unchanged
                loss_dfl = (per_edge * ew).mean(-1, keepdim=True) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum

            elif cfg is not None and cfg.use_peu:
                # ---- PEU: attenuate by the edge's own distribution variance --
                pd = pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max)
                per_edge = self.dfl_loss.per_edge(pd, target_ltrb[fg_mask])       # (N,4)
                mu, var = _dfl_edge_moments(pd, self.dfl_loss.reg_max)            # (N,4)
                # At init the bin distribution is near-uniform, so its variance
                # carries no signal and would suppress every edge equally.
                # F2: FAIL-CLOSED — until set_epoch() has actually been called
                # AND epoch >= warmup, PEU stays neutral. A missing callback now
                # means a recoverable neutral run, not attenuation from step 0.
                warmed = _EPOCH["set"] and (_EPOCH["epoch"] >= cfg.peu_warmup_epochs)
                self._peu_calls += 1
                if not _EPOCH["set"] and not self._peu_epoch_warned and self._peu_calls >= 500:
                    print("[PEU] WARNING: epoch callback not attached after "
                          f"{self._peu_calls} criterion calls — PEU is staying "
                          "NEUTRAL (fail-closed). Wire it with "
                          "attach_epoch_callback(model) or set_epoch(...).")
                    self._peu_epoch_warned = True
                w_edge, s = _peu_weights(mu, var, cfg, warmed)
                _peu_track(self._peu_state, var, w_edge)
                # F3: Kendall-mode degeneration monitor — if the RAW variance
                # camps at the min_var floor, lambda has won and the config has
                # degenerated; surface it instead of silently training on.
                if warmed and not cfg.peu_detach:
                    with torch.no_grad():
                        at_floor = (var < 1.1 * cfg.peu_min_var).float().mean() > 0.9
                    self._peu_floor_hits = self._peu_floor_hits + 1 if at_floor else 0
                    if self._peu_floor_hits >= 200 and not self._peu_floor_warned:
                        print("[PEU] WARNING: >90% of edge variances have sat at "
                              "the min_var floor for 200 consecutive calls — the "
                              "Kendall config has degenerated (lambda dominates). "
                              "Lower peu_lambda or raise peu_beta.")
                        self._peu_floor_warned = True
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

        # F5: per-instance level-occupancy diagnostics
        self._lba_state = _new_lba_state()
        if self.cfg.lba_log:
            global _LAST_LBA_HOST
            _LAST_LBA_HOST = self

        # F6: E2EDetectLoss builds two criteria from the same hyp — print each
        # DISTINCT banner once. Keyed on content, not a boolean: sequential
        # ablation runs in one process (different configs, e.g.
        # run_custom_v3_ablation.py) still get their banner; only true
        # duplicates (E2E's second identical criterion) are suppressed.
        b = self.cfg.banner()
        if b not in _PRINTED_BANNERS:
            print(b)
            _PRINTED_BANNERS.add(b)

    def lba_report(self, reset=True):
        """Foreground share per FPN level since the last call (this instance)."""
        return _lba_report_from(self._lba_state, reset)

    def lba_gate_report(self, reset=True):
        """Gate pass-rate since the last call (this instance), or None."""
        return _lba_gate_report_from(self._lba_state, reset)

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
            _lba_track(self._lba_state, fg_mask, stride_tensor,
                       gt_bboxes=gt_bboxes, mask_gt=mask_gt, cfg=self.cfg)

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
    mask (1 = assigned foreground, 0 = background).

    F1: the previous version called self._compute_cls_loss (does not exist in
    this file — carried from a sibling lineage) and passed stride_tensor to
    BboxLoss.forward (which takes no stride arg here), so the TRAIN path died
    on the first step while the eval path worked. Now uses stock cls BCE and
    the correct bbox call; the DFL branch still picks up AR-DFL/PEU via cfg.

    Note: the objectness .mean() is dominated by background (~98%+ of anchors),
    which pressures the logit toward 0 everywhere. If the head trains toward
    all-negative, consider pos_weight on bce_obj or separate fg/bg averaging.
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

        # F1: stock cls BCE (this file's BboxLoss/cls stack has no
        # _compute_cls_loss; the previous call crashed on the first train step)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
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