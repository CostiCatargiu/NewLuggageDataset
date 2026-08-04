#!/usr/bin/env python3
"""
Anchor-footprint diagnostic — decomposes P5 starvation into SUPPLY vs METRIC.
v6i EDITION — see "CHANGES FROM THE v5i SCRIPT" below.

WHAT IT ANSWERS
---------------
diag_per_edge_dfl.py measured the SYMPTOM: P5 receives a tiny share of
foreground while holding 4.9% of the anchor grid. This script measures the
CAUSE, and separates the two candidates, which need opposite fixes.

Stock TAL builds its positive set in two stages:
  1. mask_in_gts  — keep only anchors whose CENTRE falls inside the GT box.
                    This is pure geometry: a box of w x h px offers
                    (w/stride) x (h/stride) candidate cells at that level.
  2. topk         — rank the surviving candidates by score^alpha * iou^beta and
                    keep the best `topk` **pooled across ALL levels**.

Because stage 2 pools, a level's share of the selected positives is bounded by
its share of the candidate pool from stage 1. For the v6i mean box
(39 x 55 px at 640 — labels are ALREADY 640-wide, so no upscale):
    level      cells offered          share of pool
    P3 (s8)    4.9 x  6.9 = 33.7        ~76%
    P4 (s16)   2.4 x  3.4 =  8.4        ~19%
    P5 (s32)   1.2 x  1.7 =  2.1        ~4.9%
(measured cand/GT: 29.76 / 7.45 / 1.91 — the geometry predicts it closely.)

  For reference, the v5i box was 41 x 90 at 640 (33 x 72 at 512, upscaled
  x1.25) and offered 5.1 x 11.3 = 55 cells at P3. Pools have roughly HALVED,
  because the v6i re-export removed an aspect-ratio distortion AND the 1.25x
  upscale. That is why supply pressure moved down into P3.

So P5 can never win more than ~5% of a topk=10 draw on geometry alone — and
with ~2 candidate cells it cannot supply topk=10 at all. The question this
script answers is whether the observed share equals that geometric ceiling or
falls below it:
    selected_share ~= candidate_share -> SUPPLY-limited.
                                          The pool is the constraint. Fix by
                                          allocating topk PER LEVEL rather than
                                          globally (a level-aware topk).
    selected_share <  candidate_share -> METRIC-biased.
                                          The alignment metric additionally
                                          penalises coarse anchors. Fix by
                                          reweighting the metric (LBA prior).
    selected_share >  candidate_share -> no starvation to fix at that level.
The ratio  selected_share / candidate_share  is reported as SELECTION BIAS.
1.0 = the metric is level-neutral and all starvation is geometric.

It also reports, per level, the fraction of GTs whose candidate pool at that
level is smaller than topk — those GTs *cannot* be assigned topk anchors there
no matter what the metric says.

=============================================================================
CHANGES FROM THE v5i SCRIPT — all three matter for the numbers
=============================================================================
1. SIZE_BINS 60/120 -> 48/96.
   The v5i script scaled the dataset report's 48/96 px thresholds by 1.25
   because v5i labels were 512 px and training was at 640. v6i labels are
   ALREADY 640-wide, so the factor is 1.00. Running with 60/120 on v6i put
   ~13% of GTs (the 48-60 px band) into "small" that the dataset report calls
   medium: it produced small = 75.1% of GTs where the report says 62.2%.
   Sections 4, 5 and 6 were all computed on that inflated bucket.

2. SNA simulation k_min 1 -> 2.
   The real runs use snatal_kmin=2 (see run_assigner_isolated.py). With min=1
   the simulation understated k_eff for exactly the thin-pool GTs the
   mechanism targets. Now driven by SNA_KMIN so the two cannot drift.

3. The loss-config banner is annotated.
   model.init_criterion() instantiates the loss with lossv2updated.py's
   HARDCODED defaults (alpha 0.9->0.5, px 70, boost 1.5, clip 20->10, class
   weighting ON) — NOT the checkpoint's training config. It does not affect
   this diagnostic, because only the assigner is used and assignment does not
   read any SWA/clip/cls key. But the banner ends up in footprint_report.txt,
   so it is now explicitly flagged.

4. BATCHES raised so the whole split is covered (60 x 16 = 960 of 1827 val).

USAGE
-----
Edit the CONFIGURATION block below, then:
    python diag_anchor_footprint_v6i.py

OUTPUT
------
    <OUT_DIR>/footprint_stats.json    machine-readable
    <OUT_DIR>/footprint_report.txt    the tables

NOTE: run on the STOCK assigner (use_lba=False, use_satal=False, use_snatal=
False) — this measures the baseline pathology. Re-run with a mechanism on to
verify it moved.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# =============================================================================
# CONFIGURATION  — edit these, no CLI
# =============================================================================
WEIGHTS = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_newl_luggagev6i/yolov12s_default/weights/best.pt"
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"

SPLIT = "val"          # "train" | "val" | "test"
IMG_SIZE = 640         # must match how the checkpoint was trained
BATCH = 16
BATCHES = 115          # 115 x 16 = 1840 >= 1827 val images -> full split
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
OUT_DIR = "diag_fp_out_v6i"

# Object-size buckets by MAX SIDE in pixels at IMG_SIZE.
# v6i: labels are ALREADY 640-wide, so the dataset report's 48/96 px
# thresholds apply DIRECTLY — no x1.25 upscale as there was on v5i.
# Cross-check after running: 'small' should land near the dataset report's
# 62.2% of valid-split GTs, not the 75.1% the 60/120 bins produced.
SIZE_BINS = (48.0, 96.0)           # small < 48 <= medium <= 96 < large

SNA_RHO = (0.15, 0.25, 0.40)       # supply-normalised budgets to simulate
SNA_KMIN = 2                       # MUST match snatal_kmin in the run scripts
MARGINAL_IOU = 0.30                # a selected anchor below this is "marginal"
# =============================================================================

CLASS_NAMES_FALLBACK = {0: "class0", 1: "class1", 2: "class2"}


def candidates_in_gts(anc_points, gt_bboxes, eps=1e-9):
    """Replicates TaskAlignedAssigner.select_candidates_in_gts.

    anc_points : (a, 2)   anchor centres, PIXELS
    gt_bboxes  : (b, n, 4) xyxy, PIXELS
    returns    : (b, n, a) bool — anchor centre inside that GT
    """
    b, n, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)          # (b*n, 1, 2) each
    deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
    # NOTE: out-of-place gt(), not ultralytics' in-place gt_(). The in-place form
    # keeps the float dtype (returns 0.0/1.0), which is fine where ultralytics
    # combines masks with `*` but breaks the bitwise `&` used below.
    return deltas.view(b, n, -1, 4).amin(3).gt(eps)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    device = torch.device(
        f"cuda:{DEVICE}" if torch.cuda.is_available() and DEVICE != "cpu" else "cpu"
    )
    print(f"device: {device}")

    if not os.path.exists(WEIGHTS):
        sys.exit(f"WEIGHTS not found: {WEIGHTS}\nEdit the CONFIGURATION block at the top.")

    yolo = YOLO(WEIGHTS)
    model = yolo.model.to(device).eval()
    det = model.model[-1]
    reg_max, nc = det.reg_max, det.nc
    stride = det.stride.to(device)
    print(f"reg_max={reg_max}  nc={nc}  strides={stride.tolist()}")

    crit = model.init_criterion()
    assigner = crit.assigner
    topk = int(getattr(assigner, "topk", 10))

    # ---- FIX 3: the banner printed by init_criterion() is NOT the checkpoint's
    # config. Flag it loudly so nobody misreads footprint_report.txt later.
    print("")
    print("!" * 78)
    print("[NOTE] The loss-configuration banner above shows lossv2updated.py's")
    print("       HARDCODED DEFAULTS (alpha 0.9->0.5, px 70, boost 1.5,")
    print("       clip 20->10, class weighting ON) — NOT the config this")
    print("       checkpoint was trained with. It does not affect this")
    print("       diagnostic: only the ASSIGNER is used, and assignment reads")
    print("       none of the SWA / clip / cls-weight keys. The line that")
    print("       matters is the assigner line printed next.")
    print("!" * 78)
    print(f"assigner={type(assigner).__name__}  topk={topk}  "
          f"alpha={getattr(assigner,'alpha','?')}  beta={getattr(assigner,'beta','?')}")
    if type(assigner).__name__ != "TaskAlignedAssigner":
        print(f"  [WARN] assigner is NOT stock TaskAlignedAssigner. This script is")
        print(f"         meant to measure the BASELINE pathology — re-run with all")
        print(f"         of use_lba / use_satal / use_snatal / use_artal set False,")
        print(f"         unless you are deliberately verifying a mechanism moved it.")
    print(f"size bins (max side @{IMG_SIZE}px): small < {SIZE_BINS[0]:.0f} "
          f"<= medium <= {SIZE_BINS[1]:.0f} < large")
    print(f"SNA simulation: k_eff = clamp(rho*pool, {SNA_KMIN}, {topk})")
    print("")

    vcfg = get_cfg(DEFAULT_CFG)
    vcfg.imgsz, vcfg.batch = IMG_SIZE, BATCH
    vcfg.rect = vcfg.augment = vcfg.cache = vcfg.single_cls = False
    vcfg.fraction, vcfg.task, vcfg.mode = 1.0, "detect", "val"

    data = check_det_dataset(DATA_YAML)
    if SPLIT not in data:
        sys.exit(f"split '{SPLIT}' not in data.yaml "
                 f"(have: {[k for k in data if k in ('train','val','test')]})")
    names = data.get("names", CLASS_NAMES_FALLBACK)

    ds = build_yolo_dataset(cfg=vcfg, img_path=data[SPLIT], batch=BATCH,
                            data=data, mode="val", rect=False, stride=int(stride.max()))
    loader = build_dataloader(ds, BATCH, workers=WORKERS, shuffle=False)

    # ---- accumulators --------------------------------------------------------
    cand = defaultdict(float)        # stride -> total candidate cells over all GTs
    sel = defaultdict(float)         # stride -> total selected positives
    n_gt_short = defaultdict(float)  # stride -> #GTs whose pool < topk
    n_gt_zero = defaultdict(float)   # stride -> #GTs with an EMPTY pool
    cand_c = defaultdict(lambda: defaultdict(float))
    sel_c = defaultdict(lambda: defaultdict(float))
    fp_samples = defaultdict(list)   # stride -> per-GT candidate counts
    n_gt_total = 0
    n_fg_total = 0

    # ---- selectivity / positive-set quality, bucketed by GT max side --------
    # SEL: per GT, how much of its OWN candidate pool it is forced to accept.
    #      topk is absolute, so supply-poor (small) objects take nearly all of
    #      their candidates while supply-rich ones keep only the best few.
    # QUAL: are the extra positives small objects are forced to take actually
    #       bad? Measured two ways — IoU of the predicted box (what TAL ranks
    #       on) and normalised centre distance (model-free geometry).
    BUCKETS = ("small", "medium", "large")
    sel_pool = {b: [] for b in BUCKETS}    # per-GT candidate pool (all levels)
    sel_taken = {b: [] for b in BUCKETS}   # per-GT positives assigned
    q_iou = {b: [] for b in BUCKETS}       # per-fg-anchor IoU
    q_dist = {b: [] for b in BUCKETS}      # per-fg-anchor |anc-gtc| / sqrt(wh)

    print(f"sampling up to {BATCHES} batches from '{SPLIT}' ...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= BATCHES:
                break
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            feats = model(imgs)
            feats = feats[1] if isinstance(feats, (list, tuple)) and len(feats) == 2 else feats

            pred_distri, pred_scores = torch.cat(
                [xi.view(feats[0].shape[0], det.no, -1) for xi in feats], 2
            ).split((reg_max * 4, nc), 1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()
            b = pred_scores.shape[0]

            imgsz = torch.tensor(feats[0].shape[2:], device=device,
                                 dtype=pred_scores.dtype) * stride[0]
            anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
            anc_px = anchor_points * stride_tensor              # (a, 2) pixels
            s_per_anchor = stride_tensor.view(-1)               # (a,)

            targets = crit.preprocess(
                torch.cat((batch["batch_idx"].view(-1, 1),
                           batch["cls"].view(-1, 1),
                           batch["bboxes"]), 1).to(device),
                b, scale_tensor=imgsz[[1, 0, 1, 0]],
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)     # pixels
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)    # (b, n, 1) bool

            pred_bboxes = crit.bbox_decode(anchor_points, pred_distri)

            out = assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anc_px, gt_labels, gt_bboxes,
                mask_gt.to(gt_bboxes.dtype),   # ultralytics expects the float mask
            )
            target_bboxes = out[1]      # (b, a, 4) xyxy in PIXELS (gt_bboxes were px)
            fg_mask = out[3].bool()     # (b, a)
            target_gt_idx = out[4]      # (b, a) long

            if fg_mask.sum() == 0:
                continue
            n_fg_total += int(fg_mask.sum())

            # ---- STAGE 1: candidate pool, per GT per level -------------------
            in_gts = candidates_in_gts(anc_px, gt_bboxes)                # (b,n,a)
            in_gts = in_gts & mask_gt.bool()                             # drop pad GTs
            valid_gt = mask_gt.squeeze(-1).bool()                        # (b, n)
            n_gt_total += int(valid_gt.sum())

            tot_c = torch.zeros_like(valid_gt, dtype=torch.long)   # pool, all levels
            tot_s = torch.zeros_like(valid_gt, dtype=torch.long)   # positives, all levels

            for s in stride.tolist():
                lvl = (s_per_anchor == s)                                # (a,)
                c_per_gt = (in_gts & lvl.view(1, 1, -1)).sum(-1)         # (b, n)
                c_per_gt = torch.where(valid_gt, c_per_gt, torch.zeros_like(c_per_gt))

                sel_mask = fg_mask & lvl.view(1, -1)                     # (b, a)
                s_per_gt = torch.zeros_like(c_per_gt)
                if sel_mask.any():
                    bidx, aidx = sel_mask.nonzero(as_tuple=True)
                    gidx = target_gt_idx[bidx, aidx]
                    flat = bidx * c_per_gt.shape[1] + gidx
                    counts = torch.bincount(flat, minlength=c_per_gt.numel())
                    s_per_gt = counts.view_as(c_per_gt)

                tot_c += c_per_gt
                tot_s += s_per_gt

                cand[s] += float(c_per_gt.sum())
                sel[s] += float(s_per_gt.sum())
                cvals = c_per_gt[valid_gt].float()
                n_gt_short[s] += float((cvals < topk).sum())
                n_gt_zero[s] += float((cvals == 0).sum())
                fp_samples[s].append(cvals.cpu().numpy())

                cls_ids = gt_labels.squeeze(-1).long()                   # (b, n)
                for ci in range(nc):
                    m = valid_gt & (cls_ids == ci)
                    if m.any():
                        cand_c[ci][s] += float(c_per_gt[m].sum())
                        sel_c[ci][s] += float(s_per_gt[m].sum())

            # ---- selectivity per GT, bucketed by max side -------------------
            gw = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=1e-6)
            gh = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=1e-6)
            gmax = torch.maximum(gw, gh)                                 # (b, n) px
            bmask = {
                "small": valid_gt & (gmax < SIZE_BINS[0]),
                "medium": valid_gt & (gmax >= SIZE_BINS[0]) & (gmax <= SIZE_BINS[1]),
                "large": valid_gt & (gmax > SIZE_BINS[1]),
            }
            for bname, m in bmask.items():
                if m.any():
                    sel_pool[bname].append(tot_c[m].float().cpu().numpy())
                    sel_taken[bname].append(tot_s[m].float().cpu().numpy())

            # ---- quality of the positives actually assigned -----------------
            if fg_mask.any():
                bidx, aidx = fg_mask.nonzero(as_tuple=True)
                gidx = target_gt_idx[bidx, aidx]
                tb = target_bboxes[bidx, aidx]                            # (N,4) px
                pb = (pred_bboxes.detach() * stride_tensor)[bidx, aidx]    # (N,4) px
                iou = bbox_iou(pb, tb, xywh=False, CIoU=False).view(-1)
                tcx = (tb[:, 0] + tb[:, 2]) * 0.5
                tcy = (tb[:, 1] + tb[:, 3]) * 0.5
                ac = anc_px[aidx]
                geo = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).clamp(min=1.0).sqrt()
                dist = ((ac[:, 0] - tcx) ** 2 + (ac[:, 1] - tcy) ** 2).sqrt() / geo
                gm = gmax[bidx, gidx]
                qsel = {
                    "small": gm < SIZE_BINS[0],
                    "medium": (gm >= SIZE_BINS[0]) & (gm <= SIZE_BINS[1]),
                    "large": gm > SIZE_BINS[1],
                }
                for bname, m in qsel.items():
                    if m.any():
                        q_iou[bname].append(iou[m].float().cpu().numpy())
                        q_dist[bname].append(dist[m].float().cpu().numpy())

            if (bi + 1) % 10 == 0:
                print(f"  batch {bi+1}/{BATCHES}  fg={n_fg_total}  gts={n_gt_total}")

    if n_gt_total == 0:
        sys.exit("no ground-truth boxes sampled — check SPLIT / DATA_YAML")

    # ---- report --------------------------------------------------------------
    strides = sorted(cand.keys())
    tot_cand = sum(cand.values()) or 1.0
    tot_sel = sum(sel.values()) or 1.0

    lines = []
    add = lines.append
    add("=" * 78)
    add("ANCHOR-FOOTPRINT DIAGNOSTIC — supply vs metric decomposition")
    add("=" * 78)
    add(f"weights   : {WEIGHTS}")
    add(f"data      : {DATA_YAML}")
    add(f"split     : {SPLIT}   imgsz: {IMG_SIZE}   batches: {BATCHES}")
    add(f"assigner  : {type(assigner).__name__}   topk: {topk}")
    add(f"size bins : small < {SIZE_BINS[0]:.0f} <= medium <= {SIZE_BINS[1]:.0f} < large "
        f"(max side, px @{IMG_SIZE}) — v6i labels are already 640-wide, NO x1.25")
    add(f"SNA sim   : k_eff = clamp(rho*pool, {SNA_KMIN}, {topk})   "
        f"(k_min matches snatal_kmin in the run scripts)")
    add(f"GTs       : {int(n_gt_total)}     foreground anchors: {n_fg_total}")
    add("")
    add("NOTE: the loss-config banner printed by init_criterion() at startup is")
    add("      lossv2updated.py's FILE DEFAULTS, not this checkpoint's training")
    add("      config. It does not affect anything below — only the assigner is")
    add("      used, and assignment reads no SWA / clip / cls-weight key.")
    add("")

    add("1) CANDIDATE SUPPLY vs SELECTED POSITIVES, per pyramid level")
    add("-" * 78)
    add(f"{'stride':>7}{'cand/GT':>10}{'cand share':>12}{'sel/GT':>9}"
        f"{'sel share':>11}{'SEL BIAS':>10}{'verdict':>16}")
    add("-" * 78)
    per_level = {}
    for s in strides:
        cshare = cand[s] / tot_cand
        sshare = sel[s] / tot_sel
        bias = (sshare / cshare) if cshare > 1e-12 else float("nan")
        if not np.isfinite(bias):
            verdict = "no candidates"
        elif bias >= 0.9:
            verdict = "supply-limited"
        elif bias >= 0.5:
            verdict = "mild bias"
        else:
            verdict = "METRIC-BIASED"
        add(f"{s:>7.0f}{cand[s]/n_gt_total:>10.2f}{cshare*100:>11.2f}%"
            f"{sel[s]/n_gt_total:>9.2f}{sshare*100:>10.2f}%{bias:>10.2f}{verdict:>16}")
        per_level[int(s)] = {
            "cand_per_gt": cand[s] / n_gt_total,
            "cand_share": cshare,
            "sel_per_gt": sel[s] / n_gt_total,
            "sel_share": sshare,
            "selection_bias": bias,
            "verdict": verdict,
        }
    add("")
    add("  SEL BIAS = sel_share / cand_share.")
    add("  ~1.0 -> the metric is level-neutral; starvation is purely GEOMETRIC.")
    add("         Fix by allocating topk per level, not globally.")
    add("  <1.0 -> the metric additionally penalises this level.")
    add("         Fix by reweighting the alignment metric (LBA prior).")
    add("")

    add("2) CAN THIS LEVEL EVEN SUPPLY topk ANCHORS?")
    add("-" * 78)
    add(f"{'stride':>7}{'pool<topk':>12}{'pool==0':>10}{'p10':>7}{'p50':>7}{'p90':>7}")
    add("-" * 78)
    for s in strides:
        v = np.concatenate(fp_samples[s]) if fp_samples[s] else np.zeros(1)
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        add(f"{s:>7.0f}{n_gt_short[s]/n_gt_total*100:>11.1f}%"
            f"{n_gt_zero[s]/n_gt_total*100:>9.1f}%{p10:>7.1f}{p50:>7.1f}{p90:>7.1f}")
        per_level[int(s)].update({
            "frac_gt_pool_lt_topk": n_gt_short[s] / n_gt_total,
            "frac_gt_pool_zero": n_gt_zero[s] / n_gt_total,
            "pool_p10": float(p10), "pool_p50": float(p50), "pool_p90": float(p90),
        })
    add("")
    add("  'pool<topk' = GTs that CANNOT receive topk positives at this level,")
    add("  regardless of the metric. If that is high at s32, a per-level topk is")
    add("  the only thing that can change the allocation.")
    add("")

    add("3) PER CLASS — selection bias by level")
    add("-" * 78)
    add(f"{'class':<14}" + "".join(f"{'s'+str(int(s)):>12}" for s in strides))
    add("-" * 78)
    per_class = {}
    for ci in sorted(cand_c.keys()):
        cname = names[ci] if isinstance(names, dict) and ci in names else str(ci)
        tc = sum(cand_c[ci].values()) or 1.0
        ts = sum(sel_c[ci].values()) or 1.0
        row, entry = f"{str(cname):<14}", {}
        for s in strides:
            cs = cand_c[ci][s] / tc
            ss = sel_c[ci][s] / ts
            bias = (ss / cs) if cs > 1e-12 else float("nan")
            row += f"{bias:>12.2f}"
            entry[int(s)] = {"cand_share": cs, "sel_share": ss, "selection_bias": bias}
        add(row)
        per_class[str(cname)] = entry
    add("")

    add("4) SELECTIVITY — how much of its OWN pool does a GT have to accept?")
    add("-" * 78)
    add(f"{'size':<9}{'n GT':>8}{'% of GTs':>10}{'pool/GT':>10}{'taken/GT':>10}"
        f"{'SELECTIVITY':>13}{'p10 pool':>10}{'p90 pool':>10}")
    add("-" * 78)
    per_size = {}
    for b in ("small", "medium", "large"):
        if not sel_pool[b]:
            continue
        pool = np.concatenate(sel_pool[b])
        taken = np.concatenate(sel_taken[b])
        selectivity = taken.sum() / max(pool.sum(), 1e-9)
        add(f"{b:<9}{len(pool):>8}{len(pool)/n_gt_total*100:>9.1f}%"
            f"{pool.mean():>10.1f}{taken.mean():>10.2f}"
            f"{selectivity * 100:>12.1f}%{np.percentile(pool, 10):>10.0f}"
            f"{np.percentile(pool, 90):>10.0f}")
        per_size[b] = {"n_gt": int(len(pool)),
                       "frac_of_gts": float(len(pool) / n_gt_total),
                       "pool_mean": float(pool.mean()),
                       "taken_mean": float(taken.mean()), "selectivity": float(selectivity),
                       "pool_p10": float(np.percentile(pool, 10)),
                       "pool_p50": float(np.percentile(pool, 50)),
                       "pool_p90": float(np.percentile(pool, 90))}
    add("")
    add("  SANITY CHECK: with the 48/96 px bins, 'small' should be ~62% of")
    add("  valid-split GTs, matching the dataset report. If it reads ~75%, the")
    add("  bins have reverted to the v5i 60/120 values.")
    add("")
    add("  topk is ABSOLUTE, so selectivity is set by geometry, not by design.")
    add("  A supply-poor GT is forced to accept nearly all of its candidates;")
    add("  a supply-rich one keeps only its best few. If small << large here,")
    add("  small objects are training on a diluted positive set.")
    add("")

    add(f"5) POSITIVE-SET QUALITY — are the forced extras actually bad?")
    add("-" * 78)
    add(f"{'size':<9}{'mean IoU':>10}{'p10 IoU':>9}{'worst':>8}"
        f"{f'%<{MARGINAL_IOU}':>9}{'ctr dist':>10}{'p90 dist':>10}")
    add("-" * 78)
    for b in ("small", "medium", "large"):
        if not q_iou[b]:
            continue
        iou = np.concatenate(q_iou[b])
        dist = np.concatenate(q_dist[b])
        marg = float((iou < MARGINAL_IOU).mean())
        add(f"{b:<9}{iou.mean():>10.3f}{np.percentile(iou, 10):>9.3f}{iou.min():>8.3f}"
            f"{marg * 100:>8.1f}%{dist.mean():>10.3f}{np.percentile(dist, 90):>10.3f}")
        per_size.setdefault(b, {}).update({
            "iou_mean": float(iou.mean()), "iou_p10": float(np.percentile(iou, 10)),
            "iou_min": float(iou.min()), "frac_marginal": marg,
            "ctr_dist_mean": float(dist.mean()),
            "ctr_dist_p90": float(np.percentile(dist, 90))})
    add("")
    add("  'ctr dist' = |anchor centre - GT centre| / sqrt(w*h), model-free.")
    add("  Higher = positives sit further out toward the box edge. If small")
    add("  objects show higher marginal-% AND higher centre distance, the")
    add("  dilution is real and supply-normalised topk should REDUCE rho.")
    add("")

    add(f"6) WHAT WOULD SUPPLY-NORMALISED topk DO?  "
        f"k_eff = clamp(rho*pool, {SNA_KMIN}, {topk})")
    add("-" * 78)
    add(f"{'size':<9}{'now':>7}" + "".join(f"{'rho=' + str(r):>11}" for r in SNA_RHO))
    add("-" * 78)
    sna = {}
    for b in ("small", "medium", "large"):
        if not sel_pool[b]:
            continue
        pool = np.concatenate(sel_pool[b])
        taken = np.concatenate(sel_taken[b])
        row = f"{b:<9}{taken.mean():>7.2f}"
        sna[b] = {}
        for r in SNA_RHO:
            k = np.clip(np.round(r * pool), SNA_KMIN, topk)
            row += f"{k.mean():>11.2f}"
            sna[b][str(r)] = float(k.mean())
        add(row)
    add("")
    add("=" * 78)

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, "footprint_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "footprint_stats.json"), "w") as f:
        json.dump({
            "weights": WEIGHTS, "data_yaml": DATA_YAML, "split": SPLIT,
            "imgsz": IMG_SIZE, "batches": BATCHES,
            "assigner": type(assigner).__name__, "topk": topk,
            "n_gt": int(n_gt_total), "n_fg": int(n_fg_total),
            "size_bins_px": list(SIZE_BINS), "sna_kmin": SNA_KMIN,
            "marginal_iou": MARGINAL_IOU,
            "per_level": per_level, "per_class": per_class,
            "per_size": per_size, "sna_simulation": sna,
        }, f, indent=2)

    print(f"\nsaved -> {OUT_DIR}/footprint_report.txt")
    print(f"saved -> {OUT_DIR}/footprint_stats.json")


if __name__ == "__main__":
    main()