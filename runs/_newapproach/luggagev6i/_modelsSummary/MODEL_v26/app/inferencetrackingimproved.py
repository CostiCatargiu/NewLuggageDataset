#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# Use your local YOLOv12 Ultralytics fork
sys.path.insert(0, "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12")
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_IN = r"/home/constantin/Downloads/ABODA-master/video1.avi"
VIDEO_OUT = r"/home/constantin/Downloads/ABODA-master/unattended_output.avi"
SAVE_OUTPUT = True

# Model A (COCO) for persons
MODEL_PERSON = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_custom/yolov12m_custom_train/weights/yolov12x.pt"
PERSON_CLASS_IDS = [0]  # COCO: person

# Model B (your dataset) for luggage
MODEL_LUGGAGE = r"/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/runs_custom/yolov12m_custom_train/weights/best.pt"
LUGGAGE_NAMES = ["backpack", "bag", "trolley"]  # class 0/1/2

CONF_PERSON = 0.20
CONF_LUGGAGE = 0.40
IOU = 0.45
IMGZ = 960
DEVICE = "0"  # "0" GPU, or "cpu"

WINDOW_NAME = "Unattended Luggage (press q to quit)"

# --- unattended parameters ---
OWNER_RADIUS_PX = 120
UNATTENDED_SECONDS = 10.0

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
# OWNER STABILIZATION (anti-flip)
# =========================
OWNER_SWITCH_MARGIN_PX = 35  # new owner must be at least this much closer
OWNER_CONFIRM_TIME = 0.6  # seconds before switching owner

# =========================
# NEW: TRAJECTORY HISTORY
# =========================
TRAJECTORY_MAX_POINTS = 30  # max points to keep in trajectory
TRAJECTORY_DRAW_POINTS = 15  # how many recent points to draw

# =========================
# NEW: ALARM SYSTEM
# =========================
ALARM_COOLDOWN_SECONDS = 5.0  # minimum time between alarms for same luggage
ALARM_FLASH_DURATION = 2.0  # how long to flash the alarm
ALARM_SOUND_ENABLED = False  # set to True if you want beep sounds

# =========================
# NEW: ZONE DETECTION
# =========================
ENABLE_ZONES = False  # set to True to enable zone-based alerts
# Define zones as list of polygons: [[(x1,y1), (x2,y2), ...], ...]
RESTRICTED_ZONES = []  # e.g., [[(100,100), (300,100), (300,300), (100,300)]]
ZONE_ALERT_COLOR = (0, 140, 255)  # orange for zone violations

# =========================
# NEW: STATISTICS TRACKING
# =========================
ENABLE_STATISTICS = True
STATS_WINDOW_SECONDS = 60.0  # rolling window for statistics


# -----------------------------
# Geometry helpers
# -----------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]);
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def center_xyxy(bb):
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def draw_label(img, text, x, y, color, scale=0.6, thickness=2):
    x, y = int(x), int(y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def predict_bbox(st, dt_pred):
    """Constant-velocity prediction in center space."""
    x1, y1, x2, y2 = st["bbox"]
    cx, cy = center_xyxy(st["bbox"])
    w = x2 - x1
    h = y2 - y1
    cx_p = cx + st.get("vx", 0.0) * dt_pred
    cy_p = cy + st.get("vy", 0.0) * dt_pred
    return np.array([cx_p - w / 2, cy_p - h / 2, cx_p + w / 2, cy_p + h / 2], dtype=float)


def point_in_polygon(point, polygon):
    """Check if point (x, y) is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def bbox_in_zone(bbox, zones):
    """Check if bbox center is in any restricted zone."""
    cx, cy = center_xyxy(bbox)
    for zone in zones:
        if point_in_polygon((cx, cy), zone):
            return True
    return False


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
                if K.get("owner_person_id") is None and D.get("owner_person_id") is not None:
                    K["owner_person_id"] = D.get("owner_person_id")

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
# Owner stabilization
# -----------------------------
def get_person_by_id(persons, pid):
    for p in persons:
        if p["tid"] == pid:
            return p
    return None


def update_owner_with_hysteresis(st, persons, cx, cy, dt):
    """
    Stable owner assignment:
    - pick nearest within OWNER_RADIUS_PX
    - only switch if new is closer by margin and confirmed for OWNER_CONFIRM_TIME
    """
    # Find nearest candidate
    cand = None
    cand_dist = None
    for p in persons:
        dist = ((p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2) ** 0.5
        if cand_dist is None or dist < cand_dist:
            cand_dist = dist
            cand = p

    if cand is None or cand_dist is None or cand_dist > OWNER_RADIUS_PX:
        # Not supervised; keep current owner id (optional) but do not build candidate
        st["owner_candidate_id"] = None
        st["owner_candidate_s"] = 0.0
        return False, None  # supervised, owner

    # supervised
    cur_owner = st.get("owner_person_id", None)

    if cur_owner is None:
        st["owner_person_id"] = cand["tid"]
        st["owner_candidate_id"] = None
        st["owner_candidate_s"] = 0.0
        return True, cand

    if cur_owner == cand["tid"]:
        st["owner_candidate_id"] = None
        st["owner_candidate_s"] = 0.0
        return True, cand

    # compute current owner's distance if visible
    cur_p = get_person_by_id(persons, cur_owner)
    cur_dist = None
    if cur_p is not None:
        cur_dist = ((cur_p["cx"] - cx) ** 2 + (cur_p["cy"] - cy) ** 2) ** 0.5

    # Only consider switching if candidate is meaningfully closer OR current owner not visible
    if cur_dist is None or (cur_dist - cand_dist) >= OWNER_SWITCH_MARGIN_PX:
        if st.get("owner_candidate_id") == cand["tid"]:
            st["owner_candidate_s"] += dt
        else:
            st["owner_candidate_id"] = cand["tid"]
            st["owner_candidate_s"] = dt

        if st["owner_candidate_s"] >= OWNER_CONFIRM_TIME:
            st["owner_person_id"] = cand["tid"]
            st["owner_candidate_id"] = None
            st["owner_candidate_s"] = 0.0
            return True, cand
    else:
        st["owner_candidate_id"] = None
        st["owner_candidate_s"] = 0.0

    # still supervised; owner remains current
    return True, get_person_by_id(persons, st.get("owner_person_id", None))


# -----------------------------
# NEW: Alarm Management
# -----------------------------
class AlarmManager:
    def __init__(self):
        self.active_alarms = {}  # lid -> {"start_time": t, "last_trigger": t}
        self.alarm_history = deque(maxlen=100)  # Keep last 100 alarms

    def trigger_alarm(self, lid, now_t, luggage_info):
        """Trigger or update an alarm for a luggage item."""
        if lid not in self.active_alarms:
            # New alarm
            self.active_alarms[lid] = {
                "start_time": now_t,
                "last_trigger": now_t,
                "luggage_info": luggage_info
            }
            self.alarm_history.append({
                "lid": lid,
                "time": now_t,
                "info": luggage_info
            })
            if ALARM_SOUND_ENABLED:
                # You can add sound here: print('\a') or use a library
                print('\a')  # Terminal beep
            return True  # New alarm
        else:
            # Update existing alarm
            alarm = self.active_alarms[lid]
            if now_t - alarm["last_trigger"] >= ALARM_COOLDOWN_SECONDS:
                alarm["last_trigger"] = now_t
                if ALARM_SOUND_ENABLED:
                    print('\a')
                return True
            return False

    def clear_alarm(self, lid):
        """Clear an alarm when luggage is attended again."""
        if lid in self.active_alarms:
            del self.active_alarms[lid]

    def is_flashing(self, lid, now_t):
        """Check if alarm should be flashing."""
        if lid not in self.active_alarms:
            return False
        alarm = self.active_alarms[lid]
        elapsed = now_t - alarm["last_trigger"]
        if elapsed > ALARM_FLASH_DURATION:
            return False
        # Flash at 2 Hz
        return (int(elapsed * 4) % 2) == 0

    def get_active_count(self):
        return len(self.active_alarms)


# -----------------------------
# NEW: Statistics Tracker
# -----------------------------
class StatisticsTracker:
    def __init__(self, window_seconds=60.0):
        self.window_seconds = window_seconds
        self.events = deque()  # (timestamp, event_type, data)

    def add_event(self, now_t, event_type, data=None):
        """Add an event to the tracker."""
        self.events.append((now_t, event_type, data))
        self._cleanup(now_t)

    def _cleanup(self, now_t):
        """Remove events outside the time window."""
        while self.events and (now_t - self.events[0][0]) > self.window_seconds:
            self.events.popleft()

    def get_stats(self, now_t):
        """Get statistics for the current window."""
        self._cleanup(now_t)

        stats = {
            "total_detections": 0,
            "person_count": 0,
            "luggage_count": 0,
            "unattended_count": 0,
            "alarms_triggered": 0,
            "avg_unattended_time": 0.0
        }

        unattended_times = []

        for timestamp, event_type, data in self.events:
            if event_type == "detection":
                stats["total_detections"] += 1
            elif event_type == "person":
                stats["person_count"] = max(stats["person_count"], data)
            elif event_type == "luggage":
                stats["luggage_count"] = max(stats["luggage_count"], data)
            elif event_type == "unattended":
                stats["unattended_count"] += 1
                if data:
                    unattended_times.append(data)
            elif event_type == "alarm":
                stats["alarms_triggered"] += 1

        if unattended_times:
            stats["avg_unattended_time"] = sum(unattended_times) / len(unattended_times)

        return stats


# -----------------------------
# NEW: Drawing Functions
# -----------------------------
def draw_trajectory(img, trajectory, color, now_t, max_age=5.0):
    """Draw trajectory path with fading effect."""
    if len(trajectory) < 2:
        return

    points = []
    for x, y, t in trajectory:
        age = now_t - t
        if age <= max_age:
            points.append((x, y, age))

    if len(points) < 2:
        return

    for i in range(len(points) - 1):
        x1, y1, age1 = points[i]
        x2, y2, age2 = points[i + 1]

        # Fade based on age
        alpha = max(0.0, 1.0 - (age2 / max_age))
        thickness = max(1, int(3 * alpha))

        # Interpolate color with background (fade effect)
        faded_color = tuple(int(c * alpha) for c in color)

        cv2.line(img, (x1, y1), (x2, y2), faded_color, thickness)

    # Draw current position as a circle
    if points:
        x, y, _ = points[-1]
        cv2.circle(img, (x, y), 4, color, -1)


def draw_zones(img, zones, color=(0, 140, 255), thickness=2):
    """Draw restricted zones on the image."""
    for zone in zones:
        pts = np.array(zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, thickness)
        # Semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


def draw_statistics_panel(img, stats, alarm_count, fps, infer_ms):
    """Draw statistics panel on the image."""
    panel_height = 200
    panel_width = 350
    margin = 10

    # Create semi-transparent panel
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + panel_width, margin + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Draw border
    cv2.rectangle(img, (margin, margin),
                  (margin + panel_width, margin + panel_height),
                  (255, 255, 255), 2)

    # Draw statistics
    y_offset = margin + 30
    line_height = 25

    draw_label(img, "=== STATISTICS ===", margin + 10, y_offset,
               (255, 255, 255), 0.7, 2)
    y_offset += line_height

    draw_label(img, f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms",
               margin + 10, y_offset, (0, 255, 0), 0.6, 2)
    y_offset += line_height

    draw_label(img, f"Persons: {stats['person_count']}",
               margin + 10, y_offset, (255, 180, 0), 0.6, 2)
    y_offset += line_height

    draw_label(img, f"Luggage: {stats['luggage_count']}",
               margin + 10, y_offset, (0, 200, 0), 0.6, 2)
    y_offset += line_height

    draw_label(img, f"Unattended: {stats['unattended_count']}",
               margin + 10, y_offset, (0, 165, 255), 0.6, 2)
    y_offset += line_height

    draw_label(img, f"Active Alarms: {alarm_count}",
               margin + 10, y_offset, (0, 0, 255), 0.6, 2)
    y_offset += line_height

    if stats['avg_unattended_time'] > 0:
        draw_label(img, f"Avg Unattended: {stats['avg_unattended_time']:.1f}s",
                   margin + 10, y_offset, (255, 255, 0), 0.6, 2)


BOTTOM_MARGIN = 20
LINE_HEIGHT = 28


# -----------------------------
# Main
# -----------------------------
def run():
    assert os.path.exists(VIDEO_IN), f"Input video not found: {VIDEO_IN}"
    assert os.path.exists(MODEL_PERSON), f"Person model not found: {MODEL_PERSON}"
    assert os.path.exists(MODEL_LUGGAGE), f"Luggage model not found: {MODEL_LUGGAGE}"

    model_person = YOLO(MODEL_PERSON)
    model_luggage = YOLO(MODEL_LUGGAGE)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_IN}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {w}x{h} @ {video_fps:.2f} FPS")

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, video_fps if video_fps > 0 else 25.0, (w, h))
        print(f"[SAVE] {VIDEO_OUT}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Tracks (IDs never reused)
    next_person_id = 1
    person_tracks = {}  # PID -> {bbox, conf, last_seen_t, missed_s, vx, vy, trajectory}

    next_luggage_id = 1
    luggage_tracks = {}  # LID -> {bbox, cls, conf, last_seen_t, missed_s, vx, vy, unattended_s, owner_person_id, cls_scores, cls_stable, owner_candidate_id, owner_candidate_s, trajectory}

    # NEW: Initialize alarm manager and statistics tracker
    alarm_manager = AlarmManager()
    stats_tracker = StatisticsTracker(STATS_WINDOW_SECONDS) if ENABLE_STATISTICS else None

    last_loop_t = time.perf_counter()
    fps_smooth = None

    num_luggage_classes = len(LUGGAGE_NAMES)

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        now_t = time.perf_counter()
        dt = now_t - last_loop_t
        last_loop_t = now_t
        if dt <= 0:
            dt = 1e-6

        inst_fps = 1.0 / dt
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

        # -----------------------------
        # Inference
        # -----------------------------
        t0 = time.perf_counter()

        res_p = model_person.predict(
            source=frame,
            conf=CONF_PERSON,
            iou=IOU,
            imgsz=IMGZ,
            device=DEVICE,
            classes=PERSON_CLASS_IDS,
            verbose=False
        )[0]

        res_l = model_luggage.predict(
            source=frame,
            conf=CONF_LUGGAGE,
            iou=IOU,
            imgsz=IMGZ,
            device=DEVICE,
            verbose=False
        )[0]

        infer_ms = (time.perf_counter() - t0) * 1000.0
        annotated = frame.copy()

        # NEW: Draw zones if enabled
        if ENABLE_ZONES and RESTRICTED_ZONES:
            draw_zones(annotated, RESTRICTED_ZONES)

        # -----------------------------
        # PERSON DETS -> stable person tracks
        # -----------------------------
        person_dets = []
        if res_p.boxes is not None and len(res_p.boxes) > 0:
            p_xyxy = res_p.boxes.xyxy.detach().cpu().numpy().astype(float)
            p_conf = res_p.boxes.conf.detach().cpu().numpy().astype(float)
            for bb, cf in zip(p_xyxy, p_conf):
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
                "unattended_s": 0.0,
                "owner_person_id": None,
                "cls_scores": scores,
                "cls_stable": int(np.argmax(scores)),
                "cls_last": int(d["cls"]),
                # owner hysteresis state
                "owner_candidate_id": None,
                "owner_candidate_s": 0.0,
                # NEW: trajectory
                "trajectory": deque([(int(cx), int(cy), now_t)], maxlen=TRAJECTORY_MAX_POINTS)
            }

        # (3) merge any duplicates that slipped through
        merge_overlapping_tracks(luggage_tracks, now_t, merge_iou=MERGE_TRACK_IOU, max_age=MERGE_TRACK_MAX_AGE)

        prune_tracks_by_ttl(luggage_tracks, LUGGAGE_TTL_SECONDS)

        # -----------------------------
        # Unattended logic + drawing
        # -----------------------------
        any_alarm = False
        unattended_count = 0

        for lid, st in list(luggage_tracks.items()):
            age = now_t - st["last_seen_t"]

            is_recent = age <= DRAW_RECENT_SECONDS
            if is_recent:
                bb = st["bbox"]
                pred_flag = False
            else:
                if not DRAW_PREDICTED_WHEN_MISSING or age > LUGGAGE_MAX_RELINK_AGE:
                    continue
                bb = predict_bbox(st, age)
                pred_flag = True

            cid_stable = int(st.get("cls_stable", st.get("cls", 0)))
            cf = float(st.get("conf", 0.0))
            cx, cy = center_xyxy(bb)

            # Stable owner association
            supervised, owner = update_owner_with_hysteresis(st, persons, cx, cy, dt)

            # Only grow unattended timer if luggage track is "recently seen"
            if is_recent:
                if supervised:
                    st["unattended_s"] = 0.0
                    alarm_manager.clear_alarm(lid)
                else:
                    st["unattended_s"] += dt
                    unattended_count += 1

            alarm = st["unattended_s"] >= UNATTENDED_SECONDS

            # NEW: Check zone violation
            zone_violation = False
            if ENABLE_ZONES and RESTRICTED_ZONES:
                zone_violation = bbox_in_zone(bb, RESTRICTED_ZONES)

            # NEW: Trigger alarm
            if alarm:
                luggage_info = {
                    "lid": lid,
                    "class": LUGGAGE_NAMES[cid_stable] if 0 <= cid_stable < len(LUGGAGE_NAMES) else str(cid_stable),
                    "unattended_time": st["unattended_s"],
                    "position": (cx, cy)
                }
                alarm_manager.trigger_alarm(lid, now_t, luggage_info)
                any_alarm = True

                if stats_tracker:
                    stats_tracker.add_event(now_t, "alarm", st["unattended_s"])

            # Determine color based on state
            if alarm and alarm_manager.is_flashing(lid, now_t):
                color = (255, 255, 255)  # white flash
            elif alarm:
                color = (0, 0, 255)  # red
            elif zone_violation:
                color = ZONE_ALERT_COLOR  # orange for zone
            elif supervised:
                color = (0, 200, 0)  # green
            else:
                color = (0, 165, 255)  # orange

            thickness = 3 if alarm else (2 if not pred_flag else 1)

            x1, y1, x2, y2 = bb
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # NEW: Draw trajectory
            if "trajectory" in st and len(st["trajectory"]) > 1:
                draw_trajectory(annotated, st["trajectory"], color, now_t)

            cname = LUGGAGE_NAMES[cid_stable] if 0 <= cid_stable < len(LUGGAGE_NAMES) else str(cid_stable)
            last_cls = int(st.get("cls_last", cid_stable))
            last_name = LUGGAGE_NAMES[last_cls] if 0 <= last_cls < len(LUGGAGE_NAMES) else str(last_cls)
            tag = "PRED" if pred_flag else ""
            zone_tag = " [ZONE!]" if zone_violation else ""

            label = f"LID:{lid} {cname} ({last_name}) {cf:.2f} un:{st['unattended_s']:.1f}s {tag}{zone_tag}"
            draw_label(annotated, label, x1, y1 - 8, color, scale=0.55, thickness=2)

            # draw owner line + label if supervised
            if supervised and owner is not None:
                cv2.line(annotated, (int(cx), int(cy)), (int(owner["cx"]), int(owner["cy"])), (255, 255, 0), 2)
                draw_label(annotated, f"owner PID:{owner['tid']}", x1, y2 + 18, (255, 255, 0), scale=0.55, thickness=2)

        # draw persons
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            pred_flag = p.get("pred", False)
            thickness = 2 if not pred_flag else 1
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), thickness)

            # NEW: Draw person trajectory
            pid = p["tid"]
            if pid in person_tracks and "trajectory" in person_tracks[pid]:
                draw_trajectory(annotated, person_tracks[pid]["trajectory"], (255, 180, 0), now_t)

            tag = "PRED" if pred_flag else ""
            draw_label(annotated, f"person PID:{p['tid']} {p['conf']:.2f} {tag}", x1, y1 - 8, (255, 180, 0), scale=0.55,
                       thickness=2)

        # NEW: Update statistics
        if stats_tracker:
            stats_tracker.add_event(now_t, "detection")
            stats_tracker.add_event(now_t, "person", len(persons))
            stats_tracker.add_event(now_t, "luggage", len([l for l in luggage_tracks.values() if
                                                           now_t - l["last_seen_t"] <= DRAW_RECENT_SECONDS]))
            if unattended_count > 0:
                stats_tracker.add_event(now_t, "unattended", unattended_count)

        # NEW: Draw statistics panel
        if ENABLE_STATISTICS and stats_tracker:
            stats = stats_tracker.get_stats(now_t)
            draw_statistics_panel(annotated, stats, alarm_manager.get_active_count(), fps_smooth, infer_ms)

        # Draw alarm banner
        if any_alarm:
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 255), -1)
            draw_label(annotated, f"⚠️ ALARM: {alarm_manager.get_active_count()} UNATTENDED LUGGAGE!",
                       15, 45, (255, 255, 255), scale=1.1, thickness=3)

        cv2.imshow(WINDOW_NAME, annotated)
        if SAVE_OUTPUT and writer is not None:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):  # NEW: Save screenshot
            screenshot_path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, annotated)
            print(f"Screenshot saved: {screenshot_path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # NEW: Print final statistics
    if ENABLE_STATISTICS and stats_tracker:
        print("\n" + "=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)
        final_stats = stats_tracker.get_stats(now_t)
        print(f"Total Alarms Triggered: {len(alarm_manager.alarm_history)}")
        print(f"Average Unattended Time: {final_stats['avg_unattended_time']:.2f}s")
        print(f"Peak Person Count: {final_stats['person_count']}")
        print(f"Peak Luggage Count: {final_stats['luggage_count']}")
        print("=" * 50)

    print("✅ Finished.")


if __name__ == "__main__":
    run()
