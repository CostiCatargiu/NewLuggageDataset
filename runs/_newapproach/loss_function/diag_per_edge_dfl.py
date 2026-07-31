#!/usr/bin/env python3
"""
Per-edge DFL diagnostic — the measurement that motivates Anisotropic DFL.

WHAT IT ANSWERS
---------------
Stock DFL encodes all four box edges (left, top, right, bottom) into the SAME
reg_max bins, in units of that anchor's stride. So every edge gets identical
ABSOLUTE precision (1 bin = 1 stride = 8/16/32 px).

But IoU is scale-relative. For a box w x h, moving one edge by e px costs:
    width  edge (left/right):  e / w   IoU
    height edge (top/bottom):  e / h   IoU
Ratio = h / w  =  2.69 on this dataset.

So the width edges are 2.69x more IoU-sensitive while receiving the same
absolute bin resolution. This script measures, on a REAL trained model with the
REAL assigner, three things per edge:

  1. BIN OCCUPANCY   how many of the reg_max bins each edge actually uses
  2. SATURATION      fraction of targets clipped at reg_max-1 (range too small)
  3. RESIDUAL        |decoded - target| in bins and in pixels (the actual error)

If width-edge occupancy is low (few bins used) and width residual in *relative*
terms (px / box width) exceeds height's, Anisotropic DFL is justified with
evidence rather than argument. That table/figure is the paper's motivation.

USAGE
-----
    python diag_per_edge_dfl.py --weights runs/.../best.pt --data data.yaml
    python diag_per_edge_dfl.py --weights best.pt --data data.yaml --split val \
                                --batches 60 --imgsz 640 --out diag_out

OUTPUT
------
    <out>/per_edge_stats.json     machine-readable
    <out>/per_edge_report.txt     the table
    <out>/per_edge_hist.png       target-distribution + residual figure

NOTE: this requires the STOCK loss path (no AR-DFL / no anisotropic scaling).
Run it on your current best checkpoint BEFORE changing anything.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

EDGE_NAMES = ["left", "top", "right", "bottom"]
WIDTH_EDGES = [0, 2]
HEIGHT_EDGES = [1, 3]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a trained .pt")
    p.add_argument("--data", required=True, help="data.yaml")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batches", type=int, default=60, help="how many batches to sample")
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="diag_per_edge_out")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.utils.tal import make_anchors, dist2bbox, bbox2dist
    from ultralytics.utils.ops import xywh2xyxy

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"device: {device}")

    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    det = model.model[-1]                       # Detect head
    reg_max = det.reg_max
    nc = det.nc
    stride = det.stride.to(device)
    print(f"reg_max={reg_max}  nc={nc}  strides={stride.tolist()}")

    # -- the loss object gives us the SAME assigner the model trained with -----
    crit = model.init_criterion()
    assigner = crit.assigner
    proj = torch.arange(reg_max, dtype=torch.float, device=device)

    # -- dataloader ------------------------------------------------------------
    # build_yolo_dataset wants an IterableSimpleNamespace (cfg.imgsz, cfg.augment,
    # ...), not a plain dict — get_cfg gives us one with every key populated.
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    vcfg = get_cfg(DEFAULT_CFG)
    vcfg.imgsz = args.imgsz
    vcfg.batch = args.batch
    vcfg.rect = False
    vcfg.augment = False
    vcfg.cache = False
    vcfg.single_cls = False
    vcfg.fraction = 1.0
    vcfg.task = "detect"
    vcfg.mode = "val"

    data = check_det_dataset(args.data)
    if args.split not in data:
        sys.exit(f"split '{args.split}' not in data.yaml (have: {[k for k in data if k in ('train','val','test')]})")

    ds = build_yolo_dataset(
        cfg=vcfg,
        img_path=data[args.split],
        batch=args.batch,
        data=data,
        mode="val",
        rect=False,
        stride=int(stride.max()),
    )
    loader = build_dataloader(ds, args.batch, workers=4, shuffle=False)

    # -- accumulators ----------------------------------------------------------
    tgt_bins = [[] for _ in range(4)]     # target value in bins, per edge
    resid_bins = [[] for _ in range(4)]   # |decoded - target| in bins
    resid_px = [[] for _ in range(4)]     # same, in pixels
    box_w_px, box_h_px, strides_px = [], [], []
    n_fg_total = 0

    print(f"sampling up to {args.batches} batches from '{args.split}' ...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.batches:
                break
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            feats = model(imgs)
            feats = feats[1] if isinstance(feats, (list, tuple)) and len(feats) == 2 else feats

            pred_distri, pred_scores = torch.cat(
                [xi.view(feats[0].shape[0], det.no, -1) for xi in feats], 2
            ).split((reg_max * 4, nc), 1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()   # (b, a, 4*reg_max)

            b = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=pred_scores.dtype) * stride[0]
            anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

            # ground truth -> (b, n, 5) xyxy scaled to feature units
            targets = crit.preprocess(
                torch.cat((batch["batch_idx"].view(-1, 1),
                           batch["cls"].view(-1, 1),
                           batch["bboxes"]), 1).to(device),
                b, scale_tensor=imgsz[[1, 0, 1, 0]],
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            pred_bboxes = crit.bbox_decode(anchor_points, pred_distri)   # (b, a, 4) xyxy

            _, target_bboxes, target_scores, fg_mask, _ = assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels, gt_bboxes, mask_gt,
            )
            if fg_mask.sum() == 0:
                continue

            target_bboxes = target_bboxes / stride_tensor
            # ---- per-edge DFL targets (this is exactly what DFLoss sees) -----
            target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)   # (b, a, 4)
            t = target_ltrb[fg_mask]                                            # (n, 4)

            # ---- decoded value = softmax expectation over bins ---------------
            pd = pred_distri[fg_mask].view(-1, 4, reg_max)                       # (n, 4, reg_max)
            dec = pd.softmax(-1).matmul(proj)                                    # (n, 4)

            s = stride_tensor.view(1, -1, 1).expand(b, -1, 1)[fg_mask].view(-1)  # (n,) px per bin
            tb = target_bboxes[fg_mask]
            w_px = (tb[:, 2] - tb[:, 0]) * s
            h_px = (tb[:, 3] - tb[:, 1]) * s

            r = (dec - t).abs()
            for e in range(4):
                tgt_bins[e].append(t[:, e].cpu().numpy())
                resid_bins[e].append(r[:, e].cpu().numpy())
                resid_px[e].append((r[:, e] * s).cpu().numpy())
            box_w_px.append(w_px.cpu().numpy())
            box_h_px.append(h_px.cpu().numpy())
            strides_px.append(s.cpu().numpy())
            n_fg_total += int(fg_mask.sum())

            if (bi + 1) % 10 == 0:
                print(f"  batch {bi+1}: {n_fg_total} fg anchors so far")

    if n_fg_total == 0:
        sys.exit("no foreground anchors collected — check --split / --weights")

    cat = lambda L: np.concatenate(L)                                   # noqa: E731
    tgt = [cat(x) for x in tgt_bins]
    rb = [cat(x) for x in resid_bins]
    rp = [cat(x) for x in resid_px]
    W, H, S = cat(box_w_px), cat(box_h_px), cat(strides_px)
    sat_edge = reg_max - 1 - 0.02        # bbox2dist clamps at reg_max-1-0.01

    # ---------------------------------------------------------------- report --
    lines = []
    add = lines.append
    add("=" * 86)
    add(f"PER-EDGE DFL DIAGNOSTIC   reg_max={reg_max}  fg anchors={n_fg_total}  "
        f"split={args.split}  imgsz={args.imgsz}")
    add("=" * 86)
    add(f"box stats (px): mean W={W.mean():.1f}  mean H={H.mean():.1f}  "
        f"median W={np.median(W):.1f}  median H={np.median(H):.1f}  mean h/w={np.mean(H/np.maximum(W,1e-6)):.2f}")
    add("")
    add(f"{'edge':<8s} {'mean tgt':>9s} {'p95 tgt':>8s} {'bins used':>10s} {'saturated':>10s} "
        f"{'resid(bin)':>11s} {'resid(px)':>10s} {'resid/dim':>10s}")
    add("-" * 86)
    per_edge = {}
    for e in range(4):
        used = np.percentile(tgt[e], 99)
        sat = float((tgt[e] >= sat_edge).mean())
        dim = W if e in WIDTH_EDGES else H
        rel = float(np.mean(rp[e] / np.maximum(dim, 1e-6)))
        per_edge[EDGE_NAMES[e]] = dict(
            mean_target_bins=float(tgt[e].mean()), p95_target_bins=float(np.percentile(tgt[e], 95)),
            p99_bins_used=float(used), saturation_rate=sat,
            resid_bins=float(rb[e].mean()), resid_px=float(rp[e].mean()), resid_rel=rel,
        )
        add(f"{EDGE_NAMES[e]:<8s} {tgt[e].mean():>9.2f} {np.percentile(tgt[e],95):>8.2f} "
            f"{used:>9.1f}/{reg_max-1} {sat*100:>9.2f}% {rb[e].mean():>11.3f} "
            f"{rp[e].mean():>10.2f} {rel*100:>9.2f}%")

    wr = np.mean([per_edge[EDGE_NAMES[e]]["resid_rel"] for e in WIDTH_EDGES])
    hr = np.mean([per_edge[EDGE_NAMES[e]]["resid_rel"] for e in HEIGHT_EDGES])
    wu = np.mean([per_edge[EDGE_NAMES[e]]["p99_bins_used"] for e in WIDTH_EDGES])
    hu = np.mean([per_edge[EDGE_NAMES[e]]["p99_bins_used"] for e in HEIGHT_EDGES])
    add("")
    add("SUMMARY")
    add(f"  width  edges: p99 bin usage {wu:.1f}/{reg_max-1}   relative residual {wr*100:.2f}%")
    add(f"  height edges: p99 bin usage {hu:.1f}/{reg_max-1}   relative residual {hr*100:.2f}%")
    add(f"  relative-residual ratio (width/height) = {wr/max(hr,1e-9):.2f}")
    add("")
    add("READ THIS AS:")
    add("  * width p99 usage << height p99 usage  -> width edges waste most of the")
    add("    16-bin budget on a range they never reach. Compressing the width range")
    add(f"    to ~{wu/(reg_max-1):.2f}x gives ~{(reg_max-1)/max(wu,1e-6):.1f}x finer width bins for free.")
    add("  * relative-residual ratio > 1 -> width error dominates in IoU terms,")
    add("    which is what mAP50-95 actually measures.")
    add("")
    add("SUGGESTED ANISOTROPIC-DFL SCALES (adfl_w_scale / adfl_h_scale):")
    safe_w = max(0.25, min(1.0, float(np.ceil(np.percentile(np.concatenate([tgt[e] for e in WIDTH_EDGES]), 99.9)) / (reg_max - 1))))
    safe_h = max(0.25, min(2.0, float(np.ceil(np.percentile(np.concatenate([tgt[e] for e in HEIGHT_EDGES]), 99.9)) / (reg_max - 1))))
    add(f"  adfl_w_scale = {safe_w:.2f}   (covers 99.9% of observed width targets)")
    add(f"  adfl_h_scale = {safe_h:.2f}   (covers 99.9% of observed height targets)")
    add("  -> start conservative; the ablation runner logs live clamp rates.")
    add("=" * 86)

    # ------------------------------------------------------- STRATIFIED -----
    # The aggregate numbers above average over every object size, which hides
    # the question that actually matters. The measured performance gap on this
    # dataset is in LARGE objects (mAP50-95 44.4 stock vs 59.9 with the tuned
    # assigner; large remains the worst bucket even in the good config). A large
    # object needs half_side/stride bins: at 200 px on stride 8 that is 12.5 of
    # 15, i.e. near the ceiling. If large objects saturate the DFL range while
    # small ones use ~4 bins, that is a representation failure on the axis where
    # the gap lives — and the fix is MORE range for height, not finer width.
    maxside = np.maximum(W, H)
    BUCKETS = [("small", maxside < 48), ("medium", (maxside >= 48) & (maxside <= 96)),
               ("large", maxside > 96)]

    lines.append("")
    lines.append("=" * 86)
    lines.append("STRATIFIED BY OBJECT SIZE   (max bbox side: small <48px, 48-96 medium, >96 large)")
    lines.append("=" * 86)
    lines.append(f"{'bucket':<8s}{'n':>8s}{'edge':>8s}{'mean tgt':>10s}{'p99 tgt':>9s}"
                 f"{'SATURATED':>11s}{'resid(bin)':>11s}{'resid(px)':>10s}")
    strat = {}
    for bname, bmask in BUCKETS:
        n = int(bmask.sum())
        strat[bname] = {"n": n}
        if n == 0:
            continue
        lines.append("-" * 86)
        for e in range(4):
            t, r, p = tgt[e][bmask], rb[e][bmask], rp[e][bmask]
            sat = float((t >= sat_edge).mean())
            strat[bname][EDGE_NAMES[e]] = dict(
                mean_target_bins=float(t.mean()), p99_target_bins=float(np.percentile(t, 99)),
                saturation_rate=sat, resid_bins=float(r.mean()), resid_px=float(p.mean()))
            flag = "  <<<" if sat > 0.02 else ""
            lines.append(f"{bname if e == 0 else '':<8s}{n if e == 0 else '':>8}{EDGE_NAMES[e]:>8s}"
                         f"{t.mean():>10.2f}{np.percentile(t, 99):>9.2f}"
                         f"{sat*100:>10.2f}%{r.mean():>11.3f}{p.mean():>10.2f}{flag}")

    # ------------------------------------------------------ BY FPN STRIDE ----
    lines.append("")
    lines.append("=" * 86)
    lines.append("STRATIFIED BY FPN STRIDE   (bin width in px == stride; saturation risk grows with"
                 " object size at LOW stride)")
    lines.append("=" * 86)
    lines.append(f"{'stride':>7s}{'n':>9s}{'mean W':>8s}{'mean H':>8s}"
                 f"{'sat L/R':>10s}{'sat T/B':>10s}{'p99 T/B tgt':>13s}")
    lines.append("-" * 86)
    by_stride = {}
    for s_val in sorted(set(np.round(S).astype(int).tolist())):
        m = np.round(S).astype(int) == s_val
        n = int(m.sum())
        if n == 0:
            continue
        satw = float(((tgt[0][m] >= sat_edge) | (tgt[2][m] >= sat_edge)).mean())
        sath = float(((tgt[1][m] >= sat_edge) | (tgt[3][m] >= sat_edge)).mean())
        p99h = float(np.percentile(np.concatenate([tgt[1][m], tgt[3][m]]), 99))
        by_stride[int(s_val)] = dict(n=n, sat_width=satw, sat_height=sath, p99_height_bins=p99h)
        lines.append(f"{s_val:>7d}{n:>9d}{W[m].mean():>8.1f}{H[m].mean():>8.1f}"
                     f"{satw*100:>9.2f}%{sath*100:>9.2f}%{p99h:>13.2f}")

    # ------------------------------------------------------------ VERDICT ---
    lines.append("")
    lines.append("=" * 86)
    lines.append("VERDICT — is there a DFL representation problem on LARGE objects?")
    lines.append("=" * 86)
    lg = strat.get("large", {})
    if lg.get("n", 0) == 0:
        lines.append("  no large objects sampled — increase --batches")
    else:
        h_sat = max(lg["top"]["saturation_rate"], lg["bottom"]["saturation_rate"])
        h_p99 = max(lg["top"]["p99_target_bins"], lg["bottom"]["p99_target_bins"])
        lines.append(f"  large objects: n={lg['n']}, height-edge saturation {h_sat*100:.2f}%, "
                     f"p99 height target {h_p99:.1f}/{reg_max-1} bins")
        if h_sat > 0.02 or h_p99 > 0.85 * (reg_max - 1):
            lines.append("  >>> YES. Height targets are pressing against the top of the bin range on")
            lines.append("      large objects. A-DFL with adfl_h_scale > 1 (MORE range for height) is")
            lines.append("      justified, and it targets the size bucket where the gap actually is.")
            lines.append("      Suggested: adfl_h_scale = "
                         f"{max(1.0, round(h_p99/(reg_max-1)/0.8, 2))}, adfl_w_scale = 1.0")
        else:
            lines.append("  >>> NO. Large-object height targets sit well inside the range and do not")
            lines.append("      saturate. DFL has no representation problem at any size. A-DFL is")
            lines.append("      finished alongside PEU — put the remaining effort into the assigner,")
            lines.append("      which is the only mechanism with a measured effect (+1.68).")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out, "per_edge_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "per_edge_stats.json"), "w") as f:
        json.dump(dict(reg_max=reg_max, n_fg=n_fg_total, split=args.split,
                       per_edge=per_edge, suggested_w_scale=safe_w,
                       suggested_h_scale=safe_h,
                       width_rel_resid=float(wr), height_rel_resid=float(hr),
                       by_size=strat, by_stride=by_stride), f, indent=2)

    # ------------------------------------------------------------------ plot --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        COL = {0: "tab:blue", 2: "tab:cyan", 1: "tab:red", 3: "tab:orange"}

        # (a) how much of the bin budget each edge actually uses
        for e in range(4):
            ax[0].hist(tgt[e], bins=np.arange(0, reg_max + 1, 0.5), histtype="step",
                       ls="-" if e in WIDTH_EDGES else "--", color=COL[e],
                       label=EDGE_NAMES[e], density=True, lw=1.6)
        ax[0].axvline(reg_max - 1, color="k", lw=0.9)
        ax[0].set_xlabel(f"DFL target (bins, reg_max={reg_max})")
        ax[0].set_ylabel("density")
        ax[0].set_title("(a) Per-edge DFL target distribution\nsolid = width, dashed = height")
        ax[0].legend()

        # (b) residual in BINS — is the model bin-limited at all?
        vals_b = [per_edge[n]["resid_bins"] for n in EDGE_NAMES]
        ax[1].bar(EDGE_NAMES, vals_b, color=[COL[i] for i in range(4)])
        ax[1].axhline(1.0, color="k", ls=":", lw=1.2)
        ax[1].text(0.02, 1.03, "1 bin", transform=ax[1].get_yaxis_transform(), fontsize=8)
        ax[1].set_ylabel("mean |decoded − target|  (bins)")
        ax[1].set_title("(b) Residual vs bin width\nall edges sub-bin ⇒ not quantisation-limited")

        # (c) IoU-relevant error: residual normalised by the dimension it controls
        vals_c = [per_edge[n]["resid_rel"] * 100 for n in EDGE_NAMES]
        ax[2].bar(EDGE_NAMES, vals_c, color=[COL[i] for i in range(4)])
        ax[2].set_ylabel("mean residual / box dimension  (%)")
        ax[2].set_title("(c) Relative localisation error per edge")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "per_edge_hist.png"), dpi=160)
        print(f"\nfigure -> {os.path.join(args.out, 'per_edge_hist.png')}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    print(f"report -> {os.path.join(args.out, 'per_edge_report.txt')}")


if __name__ == "__main__":
    main()
