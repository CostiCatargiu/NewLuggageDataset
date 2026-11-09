#!/usr/bin/env python3
"""Combine the four model outputs of each test video into one 2x2 comparison video.

For every video stem in VIDEOS it reads the annotated outputs written by
unattended_luggage.py, e.g.

    <VIDEO_DIR>/video9_yolov12-official_output.mp4
    <VIDEO_DIR>/video9_yolov12-custom_output.mp4
    <VIDEO_DIR>/video9_yolo26-official_output.mp4
    <VIDEO_DIR>/video9_yolo26-custom_output.mp4

and writes <OUT_DIR>/video9_grid_2x2.mp4 with the layout

    +--------------------+--------------------+
    |  YOLOv12 official  |  YOLOv12 custom    |
    +--------------------+--------------------+
    |  YOLO26 official   |  YOLO26 custom     |
    +--------------------+--------------------+

The four videos are played frame by frame in lock-step (they come from the same input,
so frame n is the same moment in all four). A shorter or missing input is shown as its
last frame, or as a grey tile marked "missing".

Usage:
    python3 make_grid_videos.py                      # all VIDEOS below
    python3 make_grid_videos.py --videos video9 video10
    python3 make_grid_videos.py --scale 0.5          # half-size grid (smaller file)
"""
import argparse
import os

import cv2
import numpy as np

# ------------------------------- configuration -------------------------------
VIDEO_DIR = r"videos"  # where unattended_luggage.py wrote the *_output.mp4 files
OUT_DIR = VIDEO_DIR  # grid videos are written next to the inputs
VIDEOS = ["AVSS07_medium", "AVSS07_easy", "video1", "video9", "video10"]
# (profile name used in the file names, label drawn on the tile) in grid order:
# top-left, top-right, bottom-left, bottom-right
PROFILES = [("yolov12-official", "YOLOv12 official"),
            ("yolov12-custom", "YOLOv12 custom"),
            ("yolo26-official", "YOLO26 official"),
            ("yolo26-custom", "YOLO26 custom")]
OUTPUT_SUFFIX = "_output.mp4"  # as written by the inference script
GRID_SUFFIX = "_grid_2x2.mp4"
SCALE = 1.0          # 1.0 = every tile at the original size of the output videos
GAP = 4              # px of dark separator between tiles
LABEL = True         # draw the model name in a bar on top of each tile
FONT = cv2.FONT_HERSHEY_SIMPLEX
# -----------------------------------------------------------------------------


def open_inputs(stem):
    caps, labels = [], []
    for prof, lab in PROFILES:
        path = os.path.join(VIDEO_DIR, f"{stem}_{prof}{OUTPUT_SUFFIX}")
        cap = cv2.VideoCapture(path) if os.path.exists(path) else None
        if cap is not None and not cap.isOpened():
            cap = None
        print(f"    {'ok     ' if cap else 'MISSING'} {path}")
        caps.append(cap)
        labels.append(lab)
    return caps, labels


def tile_size(caps):
    """Largest width/height among the inputs, so every tile has the same size."""
    w = h = 0
    for c in caps:
        if c is not None:
            w = max(w, int(c.get(cv2.CAP_PROP_FRAME_WIDTH)))
            h = max(h, int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return (w or 1080), (h or 620)


def fit(frame, tw, th):
    """Place a frame on a tw x th tile without distorting it (letterbox if needed)."""
    tile = np.full((th, tw, 3), 26, np.uint8)
    h, w = frame.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = img
    return tile


def draw_label(tile, text):
    scale = max(0.5, tile.shape[1] / 1080 * 0.8)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = int(8 * scale)
    cv2.rectangle(tile, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(tile, text, (pad, th + pad), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def missing_tile(tw, th, text):
    tile = np.full((th, tw, 3), 60, np.uint8)
    cv2.putText(tile, f"{text}: missing", (20, th // 2), FONT, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return tile


def make_grid(stem):
    print(f"\n  {stem}")
    caps, labels = open_inputs(stem)
    if all(c is None for c in caps):
        print("    no inputs found, skipped")
        return None
    fps = next(c.get(cv2.CAP_PROP_FPS) for c in caps if c is not None) or 25.0
    n = max(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps if c is not None)
    tw, th = tile_size(caps)
    tw, th = int(round(tw * SCALE)) // 2 * 2, int(round(th * SCALE)) // 2 * 2
    W, H = 2 * tw + GAP, 2 * th + GAP
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{stem}{GRID_SUFFIX}")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    last = [None] * 4
    done = [c is None for c in caps]
    frame_i = 0
    while not all(done):
        canvas = np.full((H, W, 3), 10, np.uint8)
        for k, cap in enumerate(caps):
            if not done[k]:
                ok, fr = cap.read()
                if ok:
                    last[k] = fr
                else:
                    done[k] = True
            if last[k] is None:
                tile = missing_tile(tw, th, labels[k])
            else:
                tile = fit(last[k], tw, th)
                if LABEL:
                    draw_label(tile, labels[k])
            r, c = divmod(k, 2)
            y0, x0 = r * (th + GAP), c * (tw + GAP)
            canvas[y0:y0 + th, x0:x0 + tw] = tile
        if all(done):
            break
        writer.write(canvas)
        frame_i += 1
        if frame_i % 300 == 0:
            print(f"    frame {frame_i}/{n}")
    writer.release()
    for c in caps:
        if c is not None:
            c.release()
    print(f"    -> {out_path}  ({W}x{H}, {frame_i} frames @ {fps:.2f} fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2x2 comparison videos of the four detector profiles")
    ap.add_argument("--videos", nargs="+", help="video stems (default: VIDEOS in the script)")
    ap.add_argument("--dir", help="folder with the *_output.mp4 files (default: VIDEO_DIR)")
    ap.add_argument("--out-dir", help="where to write the grids (default: next to the inputs)")
    ap.add_argument("--scale", type=float, help="tile size factor, e.g. 0.5 for a smaller file")
    ap.add_argument("--no-label", action="store_true", help="do not draw the model name on the tiles")
    a = ap.parse_args()
    if a.dir:
        VIDEO_DIR = a.dir
        OUT_DIR = a.out_dir or a.dir
    if a.out_dir:
        OUT_DIR = a.out_dir
    if a.scale:
        SCALE = a.scale
    if a.no_label:
        LABEL = False
    made = [make_grid(s) for s in (a.videos or VIDEOS)]
    print(f"\n  {sum(m is not None for m in made)} grid video(s) written")
