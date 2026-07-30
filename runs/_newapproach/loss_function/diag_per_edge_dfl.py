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
    data = check_det_dataset(args.data)
    ds = build_yolo_dataset(
        cfg=yolo.overrides if isinstance(yolo.overrides, dict) else {},
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

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out, "per_edge_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "per_edge_stats.json"), "w") as f:
        json.dump(dict(reg_max=reg_max, n_fg=n_fg_total, split=args.split,
                       per_edge=per_edge, suggested_w_scale=safe_w,
                       suggested_h_scale=safe_h,
                       width_rel_resid=float(wr), height_rel_resid=float(hr)), f, indent=2)

    # ------------------------------------------------------------------ plot --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for e in range(4):
            style = "-" if e in WIDTH_EDGES else "--"
            ax[0].hist(tgt[e], bins=np.arange(0, reg_max + 1, 0.5), histtype="step",
                       linestyle=style.replace("-", "solid") if False else None,
                       ls="-" if e in WIDTH_EDGES else "--", label=EDGE_NAMES[e], density=True)
        ax[0].set_xlabel(f"DFL target (bins, reg_max={reg_max})")
        ax[0].set_ylabel("density")
        ax[0].set_title("Per-edge DFL target distribution\n(solid = width edges, dashed = height)")
        ax[0].legend()
        ax[0].axvline(reg_max - 1, color="k", lw=0.8)

        labels = EDGE_NAMES
        vals = [per_edge[n]["resid_rel"] * 100 for n in labels]
        cols = ["tab:blue" if i in WIDTH_EDGES else "tab:orange" for i in range(4)]
        ax[1].bar(labels, vals, color=cols)
        ax[1].set_ylabel("mean residual / box dimension  (%)")
        ax[1].set_title("Relative localisation error per edge\n(blue = width, orange = height)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "per_edge_hist.png"), dpi=160)
        print(f"\nfigure -> {os.path.join(args.out, 'per_edge_hist.png')}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    print(f"report -> {os.path.join(args.out, 'per_edge_report.txt')}")


if __name__ == "__main__":
    main()
