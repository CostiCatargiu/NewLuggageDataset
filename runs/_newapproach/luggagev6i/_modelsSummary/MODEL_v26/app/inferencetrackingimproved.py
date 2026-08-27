#!/usr/bin/env python3
"""Unattended-luggage detection from a fixed camera.

Two detectors run on every frame -- a luggage model and a person model (optionally the
same COCO model in both roles, see SINGLE_MODEL) -- each feeding its own Hungarian
tracker, so the two ID spaces never mix. Every luggage track is paired with an owner and
an alarm fires when that owner stays away longer than UNATTENDED_SECONDS.

Three things here are where naive versions go wrong:

1. TIME IS VIDEO TIME, NOT WALL-CLOCK TIME. Timers advance by 1/video_fps per frame, not
   by how long inference took. Otherwise "10 seconds unattended" means 10 seconds of GPU
   time and every result depends on the machine that produced it.

2. DISTANCE IS NORMALISED BY PERSON HEIGHT. Raw pixel distance is meaningless under
   perspective -- 120 px near the camera is metres, 120 px at the back of the hall is
   centimetres. A standing adult is ~1.7 m, so the person's own bbox height is a local
   pixels-per-metre estimate. All distances below are in PERSON-HEIGHTS ("h"), measured
   between box BOTTOMS (feet, bag base) so they approximate ground-plane distance.

3. OWNERSHIP IS ELECTED OVER A WINDOW, AND OCCLUSION IS NOT DEPARTURE. In a crowd the
   nearest person in any single frame is frequently not the owner, so the owner is the
   person minimising MEAN distance over the first OWNERSHIP_SEC of the bag's life. After
   that only THAT person counts as supervision -- a stranger walking past does not reset
   the clock. The timer advances only while the bag itself is visible, and a lost owner
   track counts as away only after OWNER_GRACE_SEC.

Keys: q quit, space pause, s screenshot.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ================================ I/O ========================================
VIDEO_IN = r"D:\ultralytics\_modelsSummary\videos\video1.avi"
VIDEO_OUT = r"D:\ultralytics\_modelsSummary\videos\unattended_output.mp4"
EVENTS_JSON = r"unattended_events.json"
SAVE_OUTPUT = True
SHOW = True
FRAME_LIMIT = 0  # stop after N frames (0 = whole video)
WINDOW_NAME = "Unattended Luggage   [q] quit   [space] pause   [s] screenshot"

# =============================== MODELS ======================================
MODEL_PERSON = r"yolov12x.pt"
MODEL_LUGGAGE = r"runs_yolo26_overnight_r1213_v6i/y26_scb3_sbb50_cls075/weights/best.pt"
# One COCO model for BOTH roles (e.g. "yolov26x.pt"); overrides the two paths above.
SINGLE_MODEL = None

PERSON_CLASS_IDS = [0]  # COCO 'person'
# None = keep every class the luggage model emits (correct for a custom luggage model).
LUGGAGE_CLASS_IDS = None
COCO_LUGGAGE_CLASS_IDS = [24, 26, 28]  # backpack, handbag, suitcase -- used with SINGLE_MODEL

CONF_PERSON = 0.25
CONF_LUGGAGE = 0.40
IOU = 0.45
IMGSZ = 960
DEVICE = "0"  # "0" GPU, or "cpu"

# ======================== UNATTENDED PARAMETERS ==============================
# distances in PERSON-HEIGHTS (note 2)
D_OWN = 1.5  # must be this close to be a candidate owner
D_AWAY = 2.5  # farther than this counts as "away"
OWNERSHIP_SEC = 2.0  # window used to elect the owner (note 3)
UNATTENDED_SECONDS = 10.0  # seconds away before the alarm fires
OWNER_GRACE_SEC = 2.0  # owner track may vanish this long before counting as away
MIN_PERSON_H = 20.0  # px; below this the height estimate is too noisy to divide by

# =========================
# STABLE TRACKING PARAMS
# =========================
PERSON_TTL_SECONDS = 6.0
LUGGAGE_TTL_SECONDS = 10.0

PERSON_MAX_RELINK_AGE = 4.0
LUGGAGE_MAX_RELINK_AGE = 6.0

PERSON_MATCH_IOU_THR = 0.10
PERSON_MATCH_DIST_THR = 220

LUGGAGE_MATCH_IOU_THR = 0.08
LUGGAGE_MATCH_DIST_THR = 260

# Match scoring weights: minimize cost = (-W_IOU*iou) + (W_DIST*dist) + penalty
W_IOU = 4.0
W_DIST = 1.0 / 180.0

# Smoothing
PERSON_SMOOTH_ALPHA = 0.65
LUGGAGE_SMOOTH_ALPHA = 0.70
PERSON_VEL_ALPHA = 0.55
LUGGAGE_VEL_ALPHA = 0.60

# Class stabilization (backpack<->bag jitter)
CLS_DECAY = 0.95
CLASS_MISMATCH_PENALTY = 0.25

DRAW_RECENT_SECONDS = 0.6
DRAW_PREDICTED_WHEN_MISSING = False
# Counts should reflect "visible now", not "kept alive"
PERSON_ACTIVE_MAX_AGE = 0.50  # seconds
LUGGAGE_ACTIVE_MAX_AGE = 0.70  # seconds

# =========================
# DUPLICATE-SUPPRESSION (IMPORTANT)
# =========================
# 1) class-agnostic NMS on luggage detections (kills bag/backpack duplicates)
LUGGAGE_DET_NMS_IOU = 0.60

# 2) suppress spawning a new LID if the det overlaps/near an existing track
NEW_TRACK_SUPPRESS_IOU = 0.35
NEW_TRACK_SUPPRESS_DIST = 120
NEW_TRACK_SUPPRESS_MAX_AGE = 1.5  # seconds; only compare to very recent tracks

# 3) merge tracks if two luggage tracks overlap strongly (cleanup if duplicates slipped)
MERGE_TRACK_IOU = 0.70
MERGE_TRACK_MAX_AGE = 0.8

# =========================
# TRAJECTORY HISTORY
# =========================
TRAJECTORY_MAX_POINTS = 45
TRAJECTORY_MAX_AGE_SEC = 4.0  # seconds of video time drawn behind each track

# =========================
# ALARM SYSTEM
# =========================
ALARM_COOLDOWN_SECONDS = 5.0  # minimum video time between re-triggers of the same bag
ALARM_FLASH_DURATION = 2.0
ALARM_SOUND_ENABLED = False  # terminal beep on every trigger

# =========================
# ZONE DETECTION (optional)
# =========================
ENABLE_ZONES = False
RESTRICTED_ZONES = []  # e.g. [[(100, 100), (300, 100), (300, 300), (100, 300)]]

# =========================
# ANNOTATION
# =========================
SHOW_TRAJECTORIES = True
# The info panel is rendered in its own column NEXT TO the video, never on top of it.
SIDEBAR_WIDTH = 360  # px; 0 disables the panel entirely
SIDEBAR_SIDE = "right"  # "right" or "left"
SIDEBAR_BG = (26, 26, 26)
SIDEBAR_MAX_TRACK_ROWS = 8  # per-bag rows listed under "tracked luggage"
FONT = cv2.FONT_HERSHEY_SIMPLEX

PENDING, OWNED, UNATTENDED, ALARM = "PENDING", "OWNED", "UNATTENDED", "ALARM"

COLOR_PERSON = (255, 180, 0)
COLOR_OWNER = (0, 255, 255)
COLOR_LINK = (255, 255, 0)
COLOR_PENDING = (200, 200, 200)
COLOR_OWNED = (80, 200, 80)
COLOR_UNATTENDED = (0, 165, 255)
COLOR_ALARM = (0, 0, 255)
ZONE_ALERT_COLOR = (0, 140, 255)
STATE_COLOR = {PENDING: COLOR_PENDING, OWNED: COLOR_OWNED,
               UNATTENDED: COLOR_UNATTENDED, ALARM: COLOR_ALARM}


# -----------------------------
# Geometry helpers
# -----------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def center_xyxy(bb):
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def bottom_center(bb):
    """Feet of a person / resting point of a bag -- the ground-plane contact point."""
    return ((bb[0] + bb[2]) / 2.0, bb[3])


def _text_color(bg):
    b, g, r = bg
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


def draw_label(img, text, x, y, color, scale=0.6, thickness=2, bg=True, alpha=0.75):
    """Text on a filled translucent plate; (x, y) is the text baseline. Returns its box."""
    x, y = int(x), int(y)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    pad = 4
    x = int(np.clip(x, pad, max(pad, img.shape[1] - tw - pad - 1)))
    y = int(np.clip(y, th + pad + 1, img.shape[0] - base - pad - 1))
    x1, y1, x2, y2 = x - pad, y - th - pad, x + tw + pad, y + base + pad

    if bg:
        roi = img[max(0, y1):y2, max(0, x1):x2]
        if roi.size:
            plate = np.full(roi.shape, color, dtype=np.uint8)
            cv2.addWeighted(plate, alpha, roi, 1.0 - alpha, 0, roi)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    cv2.putText(img, text, (x, y), FONT, scale, _text_color(color) if bg else color,
                thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def draw_label_block(img, lines, x, y, color, scale=0.5, thickness=1, above=True):
    """Stack of labels anchored above (default) or below the point."""
    step = int(cv2.getTextSize("Ag", FONT, scale, thickness)[0][1] + 11)
    for i, line in enumerate(lines):
        yy = y - (len(lines) - 1 - i) * step if above else y + (i + 1) * step
        draw_label(img, line, x, yy, color, scale, thickness)


def predict_bbox(st, dt_pred):
    """Constant-velocity prediction in center space."""
    x1, y1, x2, y2 = st["bbox"]
    cx, cy = center_xyxy(st["bbox"])
    w = x2 - x1
    h = y2 - y1
    cx_p = cx + st.get("vx", 0.0) * dt_pred
    cy_p = cy + st.get("vy", 0.0) * dt_pred
    return np.array([cx_p - w / 2, cy_p - h / 2, cx_p + w / 2, cy_p + h / 2], dtype=float)


def bbox_in_zone(bbox, zones):
    """True if the bbox ground-contact point falls inside any restricted zone."""
    pt = tuple(float(v) for v in bottom_center(bbox))
    return any(cv2.pointPolygonTest(np.asarray(z, np.int32), pt, False) >= 0 for z in zones)


# -----------------------------
# Detection post-processing
# -----------------------------
def nms_dets_xyxy(dets, iou_thr=0.60, class_agnostic=True):
    """
    dets: list of {"bbox": np.array([x1,y1,x2,y2]), "conf": float, "cls": int}
    Greedy NMS. With class_agnostic=True, suppress duplicates across classes.
    """
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if (not class_agnostic) and (int(d["cls"]) != int(k["cls"])):
                continue
            if iou_xyxy(d["bbox"], k["bbox"]) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


# -----------------------------
# Hungarian matching
# -----------------------------
def hungarian_match(
        dets, tracks, now_t,
        require_same_class: bool,
        max_relink_age: float,
        match_iou_thr: float,
        match_dist_thr: float,
        class_mismatch_penalty: float = 0.0,
        use_stable_class_for_penalty: bool = False
):
    track_ids = list(tracks.keys())
    if len(dets) == 0 or len(track_ids) == 0:
        return [], set(range(len(dets))), set(track_ids)

    BIG = 1e9
    cost = np.full((len(dets), len(track_ids)), BIG, dtype=float)

    for di, d in enumerate(dets):
        dbb = d["bbox"]
        dcls = int(d.get("cls", 0))
        dcx, dcy = center_xyxy(dbb)

        for tj, sid in enumerate(track_ids):
            st = tracks[sid]

            # Hard class gating
            if require_same_class and int(st.get("cls", 0)) != dcls:
                continue

            age = now_t - st["last_seen_t"]
            if age > max_relink_age:
                continue

            pbb = predict_bbox(st, age)
            iou = iou_xyxy(dbb, pbb)
            pcx, pcy = center_xyxy(pbb)
            dist = ((dcx - pcx) ** 2 + (dcy - pcy) ** 2) ** 0.5

            # geometry gating
            if not ((iou >= match_iou_thr) or (dist <= match_dist_thr)):
                continue

            penalty = 0.0
            if class_mismatch_penalty > 0.0:
                tcls = int(st.get("cls_stable", st.get("cls", 0))) if use_stable_class_for_penalty else int(
                    st.get("cls", 0))
                if dcls != tcls:
                    penalty = class_mismatch_penalty

            cost[di, tj] = (-W_IOU * iou) + (W_DIST * dist) + penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    used_d = set()
    used_t = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= BIG:
            continue
        di = int(r)
        sid = track_ids[int(c)]
        matches.append((di, sid))
        used_d.add(di)
        used_t.add(sid)

    unmatched_det = set(range(len(dets))) - used_d
    unmatched_tracks = set(track_ids) - used_t
    return matches, unmatched_det, unmatched_tracks


def update_tracks_with_matches(
        dets, tracks, matches, unmatched_track_ids,
        dt_frame: float, now_t: float,
        smooth_alpha: float, vel_alpha: float,
        update_class: bool = False,
        num_classes: int = 0
):
    # matched updates
    for di, sid in matches:
        d = dets[di]
        st = tracks[sid]

        old_cx, old_cy = center_xyxy(st["bbox"])
        new_cx, new_cy = center_xyxy(d["bbox"])

        dt_age = max(1e-6, now_t - st["last_seen_t"])
        vx_new = (new_cx - old_cx) / dt_age
        vy_new = (new_cy - old_cy) / dt_age

        st["vx"] = vel_alpha * vx_new + (1 - vel_alpha) * st.get("vx", 0.0)
        st["vy"] = vel_alpha * vy_new + (1 - vel_alpha) * st.get("vy", 0.0)

        st["bbox"] = smooth_alpha * np.array(d["bbox"], dtype=float) + (1 - smooth_alpha) * np.array(st["bbox"],
                                                                                                     dtype=float)
        st["conf"] = float(d.get("conf", st.get("conf", 0.0)))
        if "cls" in d:
            st["cls"] = int(d["cls"])

        # NEW: Update trajectory
        if "trajectory" in st:
            cx, cy = center_xyxy(st["bbox"])
            st["trajectory"].append((int(cx), int(cy), now_t))
            # Keep only recent points
            while len(st["trajectory"]) > TRAJECTORY_MAX_POINTS:
                st["trajectory"].popleft()

        # class stabilization
        if update_class:
            if num_classes <= 0:
                raise ValueError("num_classes must be > 0 when update_class=True")
            if "cls_scores" not in st:
                st["cls_scores"] = np.zeros(num_classes, dtype=float)

            st["cls_scores"] *= CLS_DECAY
            k = int(d.get("cls", 0))
            if 0 <= k < num_classes:
                st["cls_scores"][k] += float(d.get("conf", 1.0))

            st["cls_stable"] = int(np.argmax(st["cls_scores"]))
            st["cls_last"] = int(d.get("cls", 0))

        st["last_seen_t"] = now_t
        st["missed_s"] = 0.0

    # unmatched tracks: keep alive, accumulate missed time
    for sid in unmatched_track_ids:
        tracks[sid]["missed_s"] = tracks[sid].get("missed_s", 0.0) + dt_frame


def prune_tracks_by_ttl(tracks, ttl_seconds: float):
    for sid in list(tracks.keys()):
        if tracks[sid].get("missed_s", 0.0) > ttl_seconds:
            del tracks[sid]


# -----------------------------
# Duplicate suppression
# -----------------------------
def should_spawn_new_track(det, tracks, now_t):
    """Prevent creating a new ID for a det that matches an existing recent track."""
    dbb = det["bbox"]
    dcx, dcy = center_xyxy(dbb)

    for sid, st in tracks.items():
        age = now_t - st["last_seen_t"]
        if age > NEW_TRACK_SUPPRESS_MAX_AGE:
            continue

        pbb = predict_bbox(st, age)
        iou = iou_xyxy(dbb, pbb)
        pcx, pcy = center_xyxy(pbb)
        dist = ((dcx - pcx) ** 2 + (dcy - pcy) ** 2) ** 0.5

        if iou >= NEW_TRACK_SUPPRESS_IOU or dist <= NEW_TRACK_SUPPRESS_DIST:
            return False
    return True


def merge_overlapping_tracks(tracks, now_t, merge_iou=0.70, max_age=0.8):
    """Merge duplicated luggage tracks that overlap strongly."""
    ids = list(tracks.keys())
    to_delete = set()

    for i in range(len(ids)):
        a_id = ids[i]
        if a_id in to_delete:
            continue
        a = tracks[a_id]
        if now_t - a["last_seen_t"] > max_age:
            continue

        for j in range(i + 1, len(ids)):
            b_id = ids[j]
            if b_id in to_delete:
                continue
            b = tracks[b_id]
            if now_t - b["last_seen_t"] > max_age:
                continue

            if iou_xyxy(a["bbox"], b["bbox"]) >= merge_iou:
                # keep the one seen more recently (or keep smaller id if tie)
                if a["last_seen_t"] > b["last_seen_t"]:
                    keep_id, drop_id = a_id, b_id
                elif b["last_seen_t"] > a["last_seen_t"]:
                    keep_id, drop_id = b_id, a_id
                else:
                    keep_id, drop_id = (a_id, b_id) if a_id < b_id else (b_id, a_id)

                K = tracks[keep_id]
                D = tracks[drop_id]

                # Merge state
                K["unattended_s"] = max(K.get("unattended_s", 0.0), D.get("unattended_s", 0.0))
                K["first_t"] = min(K.get("first_t", 0.0), D.get("first_t", 0.0))
                if K.get("owner_pid") is None and D.get("owner_pid") is not None:
                    K["owner_pid"] = D["owner_pid"]
                    K["state"] = D["state"]
                    K["away_since"] = D["away_since"]

                if "cls_scores" in K and "cls_scores" in D and K["cls_scores"].shape == D["cls_scores"].shape:
                    K["cls_scores"] = K["cls_scores"] + D["cls_scores"]
                    K["cls_stable"] = int(np.argmax(K["cls_scores"]))

                # NEW: Merge trajectories
                if "trajectory" in K and "trajectory" in D:
                    # Combine and sort by timestamp
                    combined = list(K["trajectory"]) + list(D["trajectory"])
                    combined.sort(key=lambda x: x[2])
                    K["trajectory"] = deque(combined[-TRAJECTORY_MAX_POINTS:], maxlen=TRAJECTORY_MAX_POINTS)

                to_delete.add(drop_id)

    for sid in to_delete:
        del tracks[sid]


# -----------------------------
# Ownership (notes 2 and 3)
# -----------------------------
def get_person_by_id(persons, pid):
    for p in persons:
        if p["tid"] == pid:
            return p
    return None


def person_heights_away(bag_bbox, persons):
    """Bag-to-person distances in person-heights, measured on the ground plane."""
    bx, by = bottom_center(bag_bbox)
    out = {}
    for p in persons:
        h = p["bbox"][3] - p["bbox"][1]
        if h < MIN_PERSON_H:
            continue
        px, py = bottom_center(p["bbox"])
        out[p["tid"]] = math.hypot(bx - px, by - py) / h
    return out


def update_ownership(st, persons, now_t, dt):
    """Elect an owner over a window, then follow only that person. Returns distances."""
    dists = person_heights_away(st["bbox"], persons)

    if st["owner_pid"] is None and st["state"] == PENDING:
        for pid, d in dists.items():
            if d <= D_OWN:
                st["votes"].setdefault(pid, []).append(d)
        if now_t - st["first_t"] >= OWNERSHIP_SEC:
            if st["votes"]:
                st["owner_pid"] = min(st["votes"],
                                      key=lambda k: sum(st["votes"][k]) / len(st["votes"][k]))
                st["state"] = OWNED
            else:  # nobody was ever near it -- already unattended when it entered view
                st["state"] = UNATTENDED
                st["away_since"] = now_t
        return dists

    owner_d = dists.get(st["owner_pid"])
    if owner_d is not None:
        st["owner_missing_s"] = 0.0
        near = owner_d <= D_AWAY
    else:  # owner not visible -- occlusion or departure (note 3)
        st["owner_missing_s"] += dt
        near = st["owner_missing_s"] < OWNER_GRACE_SEC

    if near:
        st["away_since"] = None
        st["unattended_s"] = 0.0
        st["state"] = OWNED
    else:
        if st["away_since"] is None:
            st["away_since"] = now_t
        st["unattended_s"] = now_t - st["away_since"]
        st["state"] = ALARM if st["unattended_s"] >= UNATTENDED_SECONDS else UNATTENDED

    return dists


# -----------------------------
# Alarm management
# -----------------------------
class AlarmManager:
    def __init__(self):
        self.active = {}  # lid -> {"start_time", "last_trigger", "info"}
        self.history = []  # every alarm ever raised, in order

    def trigger(self, lid, now_t, info):
        """Raise or refresh an alarm. True only the first time this bag alarms."""
        alarm = self.active.get(lid)
        if alarm is None:
            self.active[lid] = {"start_time": now_t, "last_trigger": now_t, "info": info}
            self.history.append({"lid": lid, "time": now_t, "info": info})
            self._beep()
            return True
        if now_t - alarm["last_trigger"] >= ALARM_COOLDOWN_SECONDS:
            alarm["last_trigger"] = now_t
            self._beep()
        return False

    def clear(self, lid):
        self.active.pop(lid, None)

    def is_flashing(self, lid, now_t):
        alarm = self.active.get(lid)
        if alarm is None:
            return False
        elapsed = now_t - alarm["last_trigger"]
        return elapsed <= ALARM_FLASH_DURATION and int(elapsed * 4) % 2 == 0

    @staticmethod
    def _beep():
        if ALARM_SOUND_ENABLED:
            print("\a", end="", flush=True)


# -----------------------------
# Drawing
# -----------------------------
def draw_trajectory(img, trajectory, color, now_t, max_age=TRAJECTORY_MAX_AGE_SEC):
    """Recent path, thinning with age so the current position reads as the head."""
    pts = [(x, y, now_t - ts) for x, y, ts in trajectory if 0.0 <= now_t - ts <= max_age]
    if len(pts) < 2:
        return
    for (x1, y1, _), (x2, y2, age) in zip(pts, pts[1:]):
        w = max(1, int(1 + 2 * (1.0 - age / max_age)))
        cv2.line(img, (x1, y1), (x2, y2), color, w, cv2.LINE_AA)
    cv2.circle(img, pts[-1][:2], 4, color, -1, cv2.LINE_AA)


def draw_zones(img, zones, color=ZONE_ALERT_COLOR, thickness=2):
    """Draw restricted zones on the image."""
    for zone in zones:
        pts = np.asarray(zone, np.int32).reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def draw_sidebar(canvas, x0, width, rows, alarm_count, flash):
    """Info column drawn beside the footage; `rows` are (text, colour, scale) or None gaps."""
    h = canvas.shape[0]
    x, right = x0 + 14, x0 + width - 14
    cv2.line(canvas, (x0, 0), (x0, h), (70, 70, 70), 1)

    y = 32
    cv2.putText(canvas, "UNATTENDED LUGGAGE", (x, y), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y += 21
    cv2.putText(canvas, "MONITOR", (x, y), FONT, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    y += 14
    cv2.line(canvas, (x, y), (right, y), (90, 90, 90), 1)
    y += 26

    for row in rows:
        if row is None:  # section separator
            cv2.line(canvas, (x, y - 15), (right, y - 15), (60, 60, 60), 1)
            y += 8
            continue
        text, color, scale = row
        if y > h - 90:
            break
        cv2.putText(canvas, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += 24

    if alarm_count:
        box_h = 58
        top = h - box_h - 14
        fill = (255, 255, 255) if flash else COLOR_ALARM
        ink = (0, 0, 0) if flash else (255, 255, 255)
        cv2.rectangle(canvas, (x0 + 10, top), (x0 + width - 10, top + box_h), fill, -1)
        cv2.putText(canvas, "ALARM", (x, top + 26), FONT, 0.85, ink, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{alarm_count} unattended bag(s)", (x, top + 47), FONT, 0.5, ink, 1,
                    cv2.LINE_AA)


def legend_rows():
    return [("person", COLOR_PERSON, 0.45), ("owner of a bag", COLOR_OWNER, 0.45),
            ("finding owner", COLOR_PENDING, 0.45), ("owned", COLOR_OWNED, 0.45),
            ("unattended", COLOR_UNATTENDED, 0.45), ("ALARM", COLOR_ALARM, 0.45)]


def draw_alarm_border(img, flash):
    """Edge-only alert so nothing is hidden behind a banner."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255) if flash else COLOR_ALARM, 6)


# -----------------------------
# Main
# -----------------------------
def run():
    assert os.path.exists(VIDEO_IN), f"Input video not found: {VIDEO_IN}"

    person_weights = SINGLE_MODEL or MODEL_PERSON
    luggage_weights = SINGLE_MODEL or MODEL_LUGGAGE
    model_person = YOLO(person_weights)
    # one model in both roles: load once, infer once per frame, split the result by class
    shared_model = luggage_weights == person_weights
    model_luggage = model_person if shared_model else YOLO(luggage_weights)

    luggage_class_ids = LUGGAGE_CLASS_IDS
    if shared_model and luggage_class_ids is None:
        luggage_class_ids = COCO_LUGGAGE_CLASS_IDS
    luggage_names = model_luggage.names  # read from the weights, never hardcoded
    num_luggage_classes = max(luggage_names) + 1 if luggage_names else 1

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_IN}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1e-3:
        video_fps = 25.0
    dt = 1.0 / video_fps  # every timer below runs on video time, not wall clock (note 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  input   : {VIDEO_IN}  {w}x{h} @ {video_fps:.2f} fps, {total_frames} frames")
    print(f"  person  : {person_weights}  classes {PERSON_CLASS_IDS}")
    print(f"  luggage : {luggage_weights}  classes {luggage_class_ids or 'all'}"
          f"{'  [shared]' if shared_model else ''}")
    print(f"  owner within {D_OWN}h, away beyond {D_AWAY}h, alarm after {UNATTENDED_SECONDS:.0f}s\n")

    # the panel gets its own column, so the rendered canvas is wider than the video
    side_w = max(0, SIDEBAR_WIDTH)
    canvas_w = w + side_w
    video_x0 = side_w if SIDEBAR_SIDE == "left" else 0
    side_x0 = 0 if SIDEBAR_SIDE == "left" else w

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if VIDEO_OUT.lower().endswith(".mp4") else "XVID"))
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, video_fps, (canvas_w, h))
        print(f"[SAVE] {VIDEO_OUT}  ({canvas_w}x{h})")

    if SHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1800, canvas_w), min(1000, h))

    # Tracks (IDs never reused)
    next_person_id = 1
    person_tracks = {}  # PID -> {bbox, conf, last_seen_t, missed_s, vx, vy, trajectory}
    next_luggage_id = 1
    luggage_tracks = {}  # LID -> the same + {state, owner_pid, votes, away_since, unattended_s}

    alarm_manager = AlarmManager()
    events = []

    frame_count = 0
    now_t = 0.0
    fps_smooth = None
    infer_ms = 0.0
    wall_last = time.perf_counter()
    paused = False

    while True:
        ok, frame = cap.read()
        if not ok or (FRAME_LIMIT and frame_count >= FRAME_LIMIT):
            break

        now_t = frame_count / video_fps  # video timestamp in seconds
        frame_count += 1

        wall_now = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, wall_now - wall_last)
        wall_last = wall_now
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

        # -----------------------------
        # Inference
        # -----------------------------
        t0 = time.perf_counter()

        if shared_model:
            wanted = list(PERSON_CLASS_IDS) + list(luggage_class_ids or [])
            res_p = res_l = model_person.predict(
                source=frame,
                conf=min(CONF_PERSON, CONF_LUGGAGE),
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                classes=wanted or None,
                verbose=False
            )[0]
        else:
            res_p = model_person.predict(
                source=frame,
                conf=CONF_PERSON,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                classes=PERSON_CLASS_IDS,
                verbose=False
            )[0]

            res_l = model_luggage.predict(
                source=frame,
                conf=CONF_LUGGAGE,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                classes=luggage_class_ids,
                verbose=False
            )[0]

        infer_ms = (time.perf_counter() - t0) * 1000.0
        annotated = frame.copy()

        # Draw zones if enabled
        if ENABLE_ZONES and RESTRICTED_ZONES:
            draw_zones(annotated, RESTRICTED_ZONES)

        # -----------------------------
        # PERSON DETS -> stable person tracks
        # -----------------------------
        person_dets = []
        if res_p.boxes is not None and len(res_p.boxes) > 0:
            p_xyxy = res_p.boxes.xyxy.detach().cpu().numpy().astype(float)
            p_conf = res_p.boxes.conf.detach().cpu().numpy().astype(float)
            p_cls = res_p.boxes.cls.detach().cpu().numpy().astype(int)
            for bb, cf, cid in zip(p_xyxy, p_conf, p_cls):
                if int(cid) in PERSON_CLASS_IDS and cf >= CONF_PERSON:
                    person_dets.append({"bbox": bb, "conf": float(cf), "cls": 0})

        p_matches, p_unmatched_det, p_unmatched_tracks = hungarian_match(
            person_dets, person_tracks, now_t,
            require_same_class=False,
            max_relink_age=PERSON_MAX_RELINK_AGE,
            match_iou_thr=PERSON_MATCH_IOU_THR,
            match_dist_thr=PERSON_MATCH_DIST_THR,
            class_mismatch_penalty=0.0
        )

        update_tracks_with_matches(
            person_dets, person_tracks, p_matches, p_unmatched_tracks,
            dt_frame=dt, now_t=now_t,
            smooth_alpha=PERSON_SMOOTH_ALPHA, vel_alpha=PERSON_VEL_ALPHA,
            update_class=False
        )

        # create new person tracks
        for di in p_unmatched_det:
            d = person_dets[di]
            pid = next_person_id
            next_person_id += 1
            cx, cy = center_xyxy(d["bbox"])
            person_tracks[pid] = {
                "bbox": np.array(d["bbox"], dtype=float),
                "conf": float(d["conf"]),
                "cls": 0,
                "last_seen_t": now_t,
                "missed_s": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "trajectory": deque([(int(cx), int(cy), now_t)], maxlen=TRAJECTORY_MAX_POINTS)
            }

        prune_tracks_by_ttl(person_tracks, PERSON_TTL_SECONDS)

        # visible persons list (recent only)
        persons = []
        for pid, st in person_tracks.items():
            age = now_t - st["last_seen_t"]
            if age <= DRAW_RECENT_SECONDS:
                bb = st["bbox"]
                cx, cy = center_xyxy(bb)
                persons.append({"tid": pid, "bbox": bb, "cx": cx, "cy": cy, "conf": st.get("conf", 0.0)})
            elif DRAW_PREDICTED_WHEN_MISSING and age <= PERSON_MAX_RELINK_AGE:
                bb = predict_bbox(st, age)
                cx, cy = center_xyxy(bb)
                persons.append({"tid": pid, "bbox": bb, "cx": cx, "cy": cy, "conf": st.get("conf", 0.0), "pred": True})

        # -----------------------------
        # LUGGAGE DETS -> stable luggage tracks
        # -----------------------------
        luggage_dets = []
        if res_l.boxes is not None and len(res_l.boxes) > 0:
            l_cls = res_l.boxes.cls.detach().cpu().numpy().astype(int)
            l_conf = res_l.boxes.conf.detach().cpu().numpy().astype(float)
            l_xyxy = res_l.boxes.xyxy.detach().cpu().numpy().astype(float)
            for cid, cf, bb in zip(l_cls, l_conf, l_xyxy):
                if cf < CONF_LUGGAGE:
                    continue
                if luggage_class_ids is not None and int(cid) not in luggage_class_ids:
                    continue
                luggage_dets.append({"cls": int(cid), "conf": float(cf), "bbox": bb})

        # (1) class-agnostic NMS to kill duplicate boxes across classes
        luggage_dets = nms_dets_xyxy(luggage_dets, iou_thr=LUGGAGE_DET_NMS_IOU, class_agnostic=True)

        l_matches, l_unmatched_det, l_unmatched_tracks = hungarian_match(
            luggage_dets, luggage_tracks, now_t,
            require_same_class=False,  # allow cross-class relink
            max_relink_age=LUGGAGE_MAX_RELINK_AGE,
            match_iou_thr=LUGGAGE_MATCH_IOU_THR,
            match_dist_thr=LUGGAGE_MATCH_DIST_THR,
            class_mismatch_penalty=CLASS_MISMATCH_PENALTY,
            use_stable_class_for_penalty=True
        )

        update_tracks_with_matches(
            luggage_dets, luggage_tracks, l_matches, l_unmatched_tracks,
            dt_frame=dt, now_t=now_t,
            smooth_alpha=LUGGAGE_SMOOTH_ALPHA, vel_alpha=LUGGAGE_VEL_ALPHA,
            update_class=True, num_classes=num_luggage_classes
        )

        # (2) create new luggage tracks but suppress duplicates
        for di in l_unmatched_det:
            d = luggage_dets[di]

            # prevent duplicate LIDs for same physical bag
            if not should_spawn_new_track(d, luggage_tracks, now_t):
                continue

            lid = next_luggage_id
            next_luggage_id += 1

            scores = np.zeros(num_luggage_classes, dtype=float)
            k = int(d["cls"])
            if 0 <= k < num_luggage_classes:
                scores[k] = float(d["conf"])

            cx, cy = center_xyxy(d["bbox"])
            luggage_tracks[lid] = {
                "bbox": np.array(d["bbox"], dtype=float),
                "cls": int(d["cls"]),  # last observed class
                "conf": float(d["conf"]),
                "last_seen_t": now_t,
                "missed_s": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "cls_scores": scores,
                "cls_stable": int(np.argmax(scores)),
                "cls_last": int(d["cls"]),
                # ownership state (note 3)
                "first_t": now_t,
                "state": PENDING,
                "votes": {},
                "owner_pid": None,
                "owner_missing_s": 0.0,
                "away_since": None,
                "unattended_s": 0.0,
                "trajectory": deque([(int(cx), int(cy), now_t)], maxlen=TRAJECTORY_MAX_POINTS)
            }

        # (3) merge any duplicates that slipped through
        merge_overlapping_tracks(luggage_tracks, now_t, merge_iou=MERGE_TRACK_IOU, max_age=MERGE_TRACK_MAX_AGE)

        prune_tracks_by_ttl(luggage_tracks, LUGGAGE_TTL_SECONDS)

        # -----------------------------
        # Unattended logic + drawing
        # -----------------------------
        owner_of = {}  # PID -> [LID, ...], so a person can be annotated as an owner
        for lid, st in luggage_tracks.items():
            if st["owner_pid"] is not None and now_t - st["last_seen_t"] <= DRAW_RECENT_SECONDS:
                owner_of.setdefault(st["owner_pid"], []).append(lid)

        # --- persons first, so the bag/owner links land on top of the boxes ---
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            pred_flag = p.get("pred", False)
            owns = owner_of.get(p["tid"], [])
            color = COLOR_OWNER if owns else COLOR_PERSON
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          1 if pred_flag else 2)

            if SHOW_TRAJECTORIES and p["tid"] in person_tracks:
                draw_trajectory(annotated, person_tracks[p["tid"]]["trajectory"], color, now_t)

            label = f"P{p['tid']} person {p['conf']:.2f}"
            if owns:
                label += "  owns " + ",".join(f"L{i}" for i in owns)
            if pred_flag:
                label += "  PRED"
            draw_label(annotated, label, x1, y1 - 6, color, scale=0.5, thickness=1)

        # --- luggage: ownership state machine + annotation ---
        visible_luggage = 0
        owned_now = 0
        unattended_now = 0
        track_rows = []

        for lid, st in list(luggage_tracks.items()):
            age = now_t - st["last_seen_t"]
            is_recent = age <= DRAW_RECENT_SECONDS
            if is_recent:
                bb, pred_flag = st["bbox"], False
                visible_luggage += 1
            else:
                if not DRAW_PREDICTED_WHEN_MISSING or age > LUGGAGE_MAX_RELINK_AGE:
                    continue
                bb, pred_flag = predict_bbox(st, age), True

            # the clock only advances while the BAG itself is visible (note 3)
            dists = update_ownership(st, persons, now_t, dt) if is_recent else {}
            state = st["state"]
            owner = get_person_by_id(persons, st["owner_pid"])
            owner_d = dists.get(st["owner_pid"])

            zone_violation = bool(ENABLE_ZONES and RESTRICTED_ZONES
                                  and bbox_in_zone(bb, RESTRICTED_ZONES))

            if state == ALARM:
                if alarm_manager.trigger(lid, now_t, {"lid": lid, "owner": st["owner_pid"]}):
                    events.append({
                        "luggage_track": lid,
                        "owner_track": st["owner_pid"],
                        "class": str(luggage_names.get(int(st["cls_stable"]), st["cls_stable"])),
                        "first_seen_sec": round(st["first_t"], 2),
                        "left_at_sec": round(st["away_since"], 2),
                        "alarm_at_sec": round(now_t, 2),
                        "alarm_frame": frame_count,
                    })
                    print(f"  [ALARM] t={now_t:7.1f}s  L{lid}  owner P{st['owner_pid']}  "
                          f"unattended since {st['away_since']:.1f}s")
                unattended_now += 1
            else:
                alarm_manager.clear(lid)
                if state == UNATTENDED:
                    unattended_now += 1
                elif state == OWNED:
                    owned_now += 1

            color = STATE_COLOR[state]
            if state == ALARM and alarm_manager.is_flashing(lid, now_t):
                color = (255, 255, 255)
            elif zone_violation and state != ALARM:
                color = ZONE_ALERT_COLOR

            x1, y1, x2, y2 = bb
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          3 if state == ALARM else (1 if pred_flag else 2))

            if SHOW_TRAJECTORIES:
                draw_trajectory(annotated, st["trajectory"], color, now_t)

            cid = int(st["cls_stable"])
            cname = str(luggage_names.get(cid, cid))
            head = f"L{lid} {cname} {float(st['conf']):.2f}"
            if int(st.get("cls_last", cid)) != cid:  # this frame disagrees with the class vote
                head += f" (~{luggage_names.get(int(st['cls_last']), '?')})"
            if pred_flag:
                head += "  PRED"
            if zone_violation:
                head += "  [ZONE]"

            if state == PENDING:
                tail = f"finding owner {now_t - st['first_t']:.1f}/{OWNERSHIP_SEC:.0f}s"
                brief = "finding owner"
            elif state == OWNED:
                tail = f"owner P{st['owner_pid']}" + (
                    f"  {owner_d:.1f}h" if owner_d is not None else "  (occluded)")
                brief = f"owner P{st['owner_pid']}" + (f" {owner_d:.1f}h" if owner_d is not None else " occl")
            elif state == UNATTENDED:
                tail = (f"away {st['unattended_s']:.0f}s  "
                        f"alarm in {UNATTENDED_SECONDS - st['unattended_s']:.0f}s")
                brief = f"away {st['unattended_s']:.0f}s"
            else:
                tail = f"UNATTENDED {st['unattended_s']:.0f}s"
                brief = f"ALARM {st['unattended_s']:.0f}s"
            draw_label_block(annotated, [head, tail], x1, y1 - 6, color, scale=0.5, thickness=1)
            track_rows.append((f"L{lid} {cname[:9]:<10s}{brief}", color, 0.45))

            # the bag/owner pairing this whole script exists for
            if owner is not None:
                bx, by = bottom_center(bb)
                ox, oy = bottom_center(owner["bbox"])
                cv2.line(annotated, (int(bx), int(by)), (int(ox), int(oy)), COLOR_LINK, 2, cv2.LINE_AA)
                if owner_d is not None:
                    draw_label(annotated, f"P{owner['tid']}  {owner_d:.1f}h",
                               (bx + ox) / 2, (by + oy) / 2, COLOR_LINK, scale=0.45, thickness=1)

        # -----------------------------
        # Sidebar (beside the video, never on top of it)
        # -----------------------------
        alarms_active = len(alarm_manager.active)
        flash = any(alarm_manager.is_flashing(i, now_t) for i in alarm_manager.active)
        if alarms_active:
            draw_alarm_border(annotated, flash)

        if side_w:
            canvas = np.full((h, canvas_w, 3), SIDEBAR_BG, np.uint8)
            canvas[:, video_x0:video_x0 + w] = annotated
            rows = [
                (f"frame  {frame_count}/{total_frames or '?'}", (215, 215, 215), 0.5),
                (f"video  t = {now_t:.1f}s", (215, 215, 215), 0.5),
                (f"speed  {fps_smooth:.1f} fps   {infer_ms:.0f} ms", (120, 255, 120), 0.5),
                None,
                (f"persons     {len(persons)}", COLOR_PERSON, 0.52),
                (f"luggage     {visible_luggage}", COLOR_OWNED, 0.52),
                (f"owned       {owned_now}", COLOR_OWNED, 0.52),
                (f"unattended  {unattended_now}", COLOR_UNATTENDED, 0.52),
                (f"alarms      {alarms_active} now / {len(alarm_manager.history)} total",
                 COLOR_ALARM, 0.52),
                None,
                ("TRACKED LUGGAGE", (255, 255, 255), 0.48),
                *(track_rows[:SIDEBAR_MAX_TRACK_ROWS] or [("- none -", (130, 130, 130), 0.45)]),
                None,
                (f"own <= {D_OWN}h    away > {D_AWAY}h", (170, 170, 170), 0.45),
                (f"alarm after {UNATTENDED_SECONDS:.0f}s away", (170, 170, 170), 0.45),
                None,
                *legend_rows(),
            ]
            draw_sidebar(canvas, side_x0, side_w, rows, alarms_active, flash)
        else:
            canvas = annotated

        if SAVE_OUTPUT and writer is not None:
            writer.write(canvas)

        if SHOW:
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                path = f"screenshot_{frame_count:06d}.jpg"
                cv2.imwrite(path, canvas)
                print(f"  screenshot -> {path}")

        if frame_count % 200 == 0:
            print(f"    frame {frame_count}/{total_frames or '?'}  t={now_t:.0f}s  "
                  f"tracked bags={len(luggage_tracks)}  alarms={len(alarm_manager.history)}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    with open(EVENTS_JSON, "w") as f:
        json.dump({
            "source": VIDEO_IN,
            "fps": video_fps,
            "frames": frame_count,
            "params": {"d_own": D_OWN, "d_away": D_AWAY, "ownership_sec": OWNERSHIP_SEC,
                       "owner_grace_sec": OWNER_GRACE_SEC,
                       "unattended_seconds": UNATTENDED_SECONDS},
            "events": events,
        }, f, indent=2)

    print(f"\n  {frame_count} frames, {len(events)} alarm(s) -> {VIDEO_OUT}, {EVENTS_JSON}")
    for e in events:
        print(f"    L{e['luggage_track']} ({e['class']}) owner P{e['owner_track']}  "
              f"left {e['left_at_sec']}s, alarm {e['alarm_at_sec']}s")


if __name__ == "__main__":
    run()
