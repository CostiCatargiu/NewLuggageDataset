#!/usr/bin/env python3
r"""
infer_video_detections.py — visualise luggage + person detections on a video.

Two detectors on every frame:
    LUGGAGE  your fine-tuned YOLO26 model  (backpack / bag / trolley)
    PERSON   stock yolo26l.pt, COCO class 0 only

Detection only. No tracking, no ownership, no abandonment logic — this is the
"what does the model actually see" tool. For the full pipeline use
unattended_luggage.py in this folder.

NO ARGUMENTS. Edit the CONFIG block and run:
    python infer_video_detections.py

Writes an annotated .mp4 next to the source and prints a per-class summary.

WHY A SEPARATE PERSON MODEL. Your luggage dataset has no `person` class, so the
person boxes have to come from a COCO model. Be aware of what that costs:
Vrsalovic et al. (Sensors 2025) measured COCO-pretrained detectors on bird's-eye
CCTV at mAP@0.5 = 1.1-4.2%, recall as low as 6.5%. On a steep overhead camera
expect the person boxes to be much worse than the luggage boxes. That asymmetry
is the point of looking at them side by side.
"""

import os
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ======================= CONFIG — edit these ================================
SOURCE = "clip.mp4"                    # input video (or 0 for a webcam)
OUT_PATH = ""                          # "" -> auto: <source>_detections.mp4
SHOW_WINDOW = False                    # True to preview live (needs a display)

# --- models ---------------------------------------------------------------
# Pick ONE luggage model. Ranked by what you want to LOOK at:
#   y26_remap_scb3          mAP50 81.66  <- best mAP50 in the fixed tree; best for eyeballing
#   y26_remap_dys           mAP50 81.65  large mAP50 83.19
#   y26_p2_remap            mAP50 80.78  best mAP50-95 (55.84), stock loss, simplest to defend
#   y26_scb3_sbb50_cls075   mAP50 81.39  what unattended_luggage.py currently deploys
RUNS_DIR = "runs_yolo26_overnight_r1213_v6i"
LUGGAGE_WEIGHTS = os.path.join(RUNS_DIR, "y26_remap_scb3", "weights", "best.pt")
PERSON_WEIGHTS = "yolo26l.pt"          # auto-downloads; falls back to yolo26m/yolov8l

# --- thresholds -----------------------------------------------------------
LUGGAGE_CONF = 0.25                    # lower than the 0.35 default on purpose:
                                       # the diagnostics show ~20% of small GTs have a
                                       # correct box sitting BELOW 0.25 (scoring failure).
                                       # Drop to 0.10 to see the recoverable ones.
PERSON_CONF = 0.35
IOU_NMS = 0.70
IMG_SIZE = 640                         # match training. 960/1280 will change behaviour.
MAX_DET = 300                          # raise to 1000 for dense scenes
PERSON_CLASS = 0                       # COCO 'person'
DRAW_PERSON = True
DRAW_LUGGAGE = True

# --- output ---------------------------------------------------------------
FRAME_STRIDE = 1                       # 2 = process every other frame (faster preview)
MAX_FRAMES = 0                         # 0 = whole video
SHOW_CONF = True
SHOW_PANEL = True                      # running counts + fps overlay
BOX_THICK_LUGGAGE = 2
BOX_THICK_PERSON = 1
DEVICE = 0                             # 0 = first GPU, "cpu" to force CPU

# BGR, one per luggage class index. Person is drawn in grey so luggage pops.
CLASS_COLORS = {
    "backpack": (60, 200, 255),        # amber
    "bag": (80, 255, 120),             # green
    "trolley": (255, 140, 60),         # blue
}
PERSON_COLOR = (150, 150, 150)
DEFAULT_COLOR = (200, 200, 200)
# ============================================================================


def load_model(path, label):
    try:
        m = YOLO(path)
        print(f"  [{label:<7}] {path}")
        return m
    except Exception as ex:
        raise SystemExit(
            f"  [ABORT] could not load {label} model '{path}': {ex}\n"
            f"          for the person model try yolo26m.pt / yolo26s.pt / yolov8l.pt"
        )


def draw_box(img, xyxy, color, label, thick, small_ok=True):
    """Box + label. Label flips inside the box when the box is tiny, so a 20 px
    trolley is not buried under its own text — which is most of this dataset."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if not label:
        return
    scale = 0.42 if small_ok else 0.5
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = y1 - 4
    if ty - th < 0:                      # no room above -> draw below the top edge
        ty = y1 + th + 4
    cv2.rectangle(img, (x1, ty - th - base + 1), (x1 + tw + 4, ty + base - 1), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), 1, cv2.LINE_AA)


def panel(img, lines):
    """Semi-transparent stats box, pinned TOP-RIGHT.

    Not top-left: on a 640x360 CCTV frame a panel there sits exactly where a
    small detection can land, and it would hide the boxes this script exists
    to show. Right edge is emptier in practice, and it is drawn last so the
    text stays readable over any box underneath.
    """
    pad, lh = 8, 18
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines)
    w, h = tw + 2 * pad, lh * len(lines) + pad
    x0 = max(img.shape[1] - w, 0)
    ov = img.copy()
    cv2.rectangle(ov, (x0, 0), (x0 + w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + pad, pad + lh * (i + 1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    print("=" * 78)
    print("  LUGGAGE + PERSON DETECTION — video inference")
    print("=" * 78)
    if not (SOURCE == 0 or os.path.exists(SOURCE)):
        raise SystemExit(f"  [ABORT] source not found: {SOURCE}")
    if not os.path.exists(LUGGAGE_WEIGHTS):
        raise SystemExit(f"  [ABORT] luggage weights not found: {LUGGAGE_WEIGHTS}\n"
                         f"          check RUNS_DIR / the run name in the CONFIG block")

    lug = load_model(LUGGAGE_WEIGHTS, "luggage")
    per = load_model(PERSON_WEIGHTS, "person") if DRAW_PERSON else None
    lug_names = lug.names if isinstance(lug.names, dict) else dict(enumerate(lug.names))
    print(f"  luggage classes: {list(lug_names.values())}")
    print(f"  conf: luggage {LUGGAGE_CONF}  person {PERSON_CONF}   imgsz {IMG_SIZE}   max_det {MAX_DET}")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"  [ABORT] cannot open {SOURCE}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  video: {W}x{H} @ {fps_in:.1f} fps, {total or '?'} frames")

    out_path = OUT_PATH or (os.path.splitext(str(SOURCE))[0] + "_detections.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in / max(FRAME_STRIDE, 1), (W, H))
    if not writer.isOpened():
        raise SystemExit(f"  [ABORT] cannot open writer for {out_path}")
    print(f"  writing: {out_path}\n")

    counts = defaultdict(int)
    confs = defaultdict(list)
    areas = defaultdict(list)
    n_proc, n_read, t0 = 0, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        if MAX_FRAMES and n_proc >= MAX_FRAMES:
            break
        if (n_read - 1) % max(FRAME_STRIDE, 1):
            continue
        n_proc += 1
        per_frame = defaultdict(int)

        if DRAW_PERSON and per is not None:
            r = per.predict(frame, conf=PERSON_CONF, iou=IOU_NMS, imgsz=IMG_SIZE,
                            max_det=MAX_DET, classes=[PERSON_CLASS], device=DEVICE,
                            verbose=False)[0]
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                c = float(b.conf)
                draw_box(frame, xyxy, PERSON_COLOR,
                         f"person {c:.2f}" if SHOW_CONF else "person", BOX_THICK_PERSON)
                counts["person"] += 1
                confs["person"].append(c)
                per_frame["person"] += 1

        if DRAW_LUGGAGE:
            r = lug.predict(frame, conf=LUGGAGE_CONF, iou=IOU_NMS, imgsz=IMG_SIZE,
                            max_det=MAX_DET, device=DEVICE, verbose=False)[0]
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                c = float(b.conf)
                name = lug_names.get(int(b.cls), str(int(b.cls)))
                col = CLASS_COLORS.get(name, DEFAULT_COLOR)
                draw_box(frame, xyxy, col, f"{name} {c:.2f}" if SHOW_CONF else name,
                         BOX_THICK_LUGGAGE)
                counts[name] += 1
                confs[name].append(c)
                areas[name].append(max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
                per_frame[name] += 1

        if SHOW_PANEL:
            el = time.time() - t0
            lines = [f"frame {n_read}" + (f"/{total}" if total else ""),
                     f"{n_proc / el:.1f} fps proc" if el > 0 else "…"]
            lines += [f"{k}: {per_frame.get(k, 0)}" for k in
                      list(lug_names.values()) + (["person"] if DRAW_PERSON else [])]
            panel(frame, lines)

        writer.write(frame)
        if SHOW_WINDOW:
            cv2.imshow("detections", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("  [stopped by user]")
                break
        if n_proc % 100 == 0:
            el = time.time() - t0
            print(f"  {n_proc} frames  {n_proc / el:5.1f} fps  " +
                  "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    cap.release()
    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    el = time.time() - t0
    print("\n" + "=" * 78)
    print(f"  DONE — {n_proc} frames in {el:.1f}s ({n_proc / max(el, 1e-9):.1f} fps)")
    print("=" * 78)
    print(f"{'class':<12}{'dets':>8}{'per frame':>11}{'mean conf':>11}"
          f"{'p10 conf':>10}{'mean maxside px':>17}")
    print("-" * 78)
    for k in sorted(counts, key=lambda x: -counts[x]):
        cs = np.array(confs[k])
        side = f"{np.mean(areas[k]):.1f}" if areas.get(k) else "-"
        print(f"{k:<12}{counts[k]:>8}{counts[k] / max(n_proc, 1):>11.2f}"
              f"{cs.mean():>11.3f}{np.percentile(cs, 10):>10.3f}{side:>17}")
    print(f"\n  saved: {out_path}")
    print("\n  READING THE NUMBERS")
    print("  * mean conf well below ~0.6 on luggage = the scoring failure the")
    print("    diagnostics found (detections exist, they score low). Re-run with")
    print("    LUGGAGE_CONF = 0.10 and see how many extra correct boxes appear.")
    print("  * person dets far sparser than expected = COCO domain shift on a")
    print("    bird's-eye camera; that is a known 1-4% mAP regime, not a bug.")


if __name__ == "__main__":
    main()
