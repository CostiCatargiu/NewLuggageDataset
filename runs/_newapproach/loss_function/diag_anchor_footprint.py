

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# =============================================================================
# CONFIGURATION  — edit these, no CLI
# =============================================================================
WEIGHTS = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_custom_v3/v3_anchor2/weights/best.pt"
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"

SPLIT = "val"          # "train" | "val" | "test"
IMG_SIZE = 640         # must match how the checkpoint was trained
BATCH = 16
BATCHES = 60           # how many batches to sample; 60 x 16 ~ 960 images
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
OUT_DIR = "diag_fp_out"

# Object-size buckets by MAX SIDE in pixels at IMG_SIZE. Defaults are the
# dataset report's 48/96 px at 512, scaled to 640 (x1.25).
SIZE_BINS = (60.0, 120.0)          # small < 60 <= medium <= 120 < large
SNA_RHO = (0.15, 0.25, 0.40)       # supply-normalised budgets to simulate
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
    print(f"assigner={type(assigner).__name__}  topk={topk}  "
          f"alpha={getattr(assigner,'alpha','?')}  beta={getattr(assigner,'beta','?')}")

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
    add(f"split     : {SPLIT}   imgsz: {IMG_SIZE}   batches: {BATCHES}")
    add(f"assigner  : {type(assigner).__name__}   topk: {topk}")
    add(f"GTs       : {int(n_gt_total)}     foreground anchors: {n_fg_total}")
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
    add(f"{'size':<9}{'n GT':>8}{'pool/GT':>10}{'taken/GT':>10}{'SELECTIVITY':>13}"
        f"{'p10 pool':>10}{'p90 pool':>10}")
    add("-" * 78)
    per_size = {}
    for b in ("small", "medium", "large"):
        if not sel_pool[b]:
            continue
        pool = np.concatenate(sel_pool[b])
        taken = np.concatenate(sel_taken[b])
        selectivity = taken.sum() / max(pool.sum(), 1e-9)
        add(f"{b:<9}{len(pool):>8}{pool.mean():>10.1f}{taken.mean():>10.2f}"
            f"{selectivity * 100:>12.1f}%{np.percentile(pool, 10):>10.0f}"
            f"{np.percentile(pool, 90):>10.0f}")
        per_size[b] = {"n_gt": int(len(pool)), "pool_mean": float(pool.mean()),
                       "taken_mean": float(taken.mean()), "selectivity": float(selectivity),
                       "pool_p10": float(np.percentile(pool, 10)),
                       "pool_p50": float(np.percentile(pool, 50)),
                       "pool_p90": float(np.percentile(pool, 90))}
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

    add("6) WHAT WOULD SUPPLY-NORMALISED topk DO?  k_eff = clamp(rho*pool, 1, topk)")
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
            k = np.clip(np.round(r * pool), 1, topk)
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
            "weights": WEIGHTS, "split": SPLIT, "imgsz": IMG_SIZE,
            "batches": BATCHES, "assigner": type(assigner).__name__, "topk": topk,
            "n_gt": int(n_gt_total), "n_fg": int(n_fg_total),
            "size_bins_px": list(SIZE_BINS), "marginal_iou": MARGINAL_IOU,
            "per_level": per_level, "per_class": per_class,
            "per_size": per_size, "sna_simulation": sna,
        }, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/footprint_report.txt")
    print(f"saved -> {OUT_DIR}/footprint_stats.json")


if __name__ == "__main__":
    main()