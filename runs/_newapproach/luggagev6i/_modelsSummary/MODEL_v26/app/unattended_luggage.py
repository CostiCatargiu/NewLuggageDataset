"""Unattended-luggage detection from fixed-camera video.

Two detectors on every frame -- the custom luggage model and an off-the-shelf person
model -- each with its own BoT-SORT instance so the ID spaces stay separate. Every
luggage track is assigned an owner, and an alarm fires when that owner stays away for
longer than T_ALARM.

Three things here are not obvious and are where naive versions go wrong:

1. DISTANCE IS NORMALISED BY PERSON HEIGHT. Raw pixel distance is meaningless under
   perspective -- 100 px near the camera is metres, 100 px at the back of the hall is
   centimetres. A standing adult is ~1.7 m, so the person's own bbox height is a local
   pixels-per-metre estimate. Distances below are in PERSON-HEIGHTS and transfer across
   cameras without calibration. Both points are taken at the box BOTTOM (feet, bag base)
   so the measurement approximates ground-plane distance rather than image distance.

2. OWNERSHIP IS ASSIGNED OVER A WINDOW, NOT INSTANTANEOUSLY. In a crowd the nearest
   person in any single frame is frequently not the owner. The owner is the person that
   minimises MEAN distance over the first OWNERSHIP_SEC of the bag's life.

3. OCCLUSION IS NOT DEPARTURE. If the bag vanishes because its owner is standing in
   front of it, that is not abandonment -- the timer only advances while the BAG track
   is alive. Conversely a lost OWNER track is treated as away only after a grace period,
   so a brief occlusion of the owner does not start the clock.

Everything is configured in the CONFIG block below -- there are no command-line
arguments. Edit the constants and run:

    python unattended_luggage.py

The person and luggage detectors can be the same model (set SINGLE_MODEL) if that model
detects both COCO 'person' (0) and the luggage classes ('backpack' 24, 'handbag' 26,
'suitcase' 28). Supply yolov12x.pt or yolov26x.pt for either role.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

# ============================== CONFIG =======================================
SOURCE = "clip.mp4"  # input video file, RTSP url or camera index
OUT = "unattended_out.mp4"  # annotated video written here
EVENTS_JSON = "unattended_events.json"  # alarm log written here
SHOW = True  # live preview window (ESC quits); needs GUI opencv-python
LIMIT = 0  # stop after N frames (0 = whole video)

LUGGAGE_WEIGHTS = "runs_yolo26_overnight_r1213_v6i/y26_scb3_sbb50_cls075/weights/best.pt"
PERSON_WEIGHTS = "yolov12x.pt"
# One COCO model for BOTH roles, e.g. "yolov26x.pt". Overrides the two paths above and
# implies COCO_LUGGAGE. Set to None to run the two separate detectors.
SINGLE_MODEL = None
# Filter the luggage detector to COCO backpack/handbag/suitcase. Leave False for a
# custom single-class luggage model, which reports whatever class it emits.
COCO_LUGGAGE = False

TRACKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_static.yaml")

PERSON_CLASS = 0  # COCO 'person'
# COCO luggage classes -- used only when the luggage detector is a stock COCO model
# (custom single-class luggage models ignore this and report whatever class they emit).
COCO_LUGGAGE_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
PERSON_CONF = 0.35
LUGGAGE_CONF = 0.35

# distances in PERSON-HEIGHTS (see note 1)
D_OWN = 1.5  # must be at least this close to be considered a candidate owner
D_ALARM = 2.5  # farther than this counts as "away"

OWNERSHIP_SEC = 2.0  # window used to pick the owner
T_ALARM = 30.0  # seconds away before the alarm fires (PETS2006 / i-LIDS convention)
OWNER_GRACE_SEC = 2.0  # owner track may vanish this long before counting as away
BAG_DROP_SEC = 10.0  # bag unseen this long -> forget the track entirely
MIN_PERSON_H = 20.0  # px; below this the height estimate is too noisy to divide by
# =============================================================================

PENDING, OWNED, UNATTENDED, ALARM = "PENDING", "OWNED", "UNATTENDED", "ALARM"


@dataclass
class Det:
    tid: int
    box: tuple[float, float, float, float]
    conf: float
    cls: int
    name: str = ""

    @property
    def base(self) -> tuple[float, float]:
        """Bottom-centre: feet for a person, resting point for a bag."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2.0, y2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class BagState:
    tid: int
    first_frame: int
    last_seen: int
    owner: int | None = None
    votes: dict[int, list[float]] = field(default_factory=dict)
    state: str = PENDING
    away_since: float | None = None
    owner_gone: int = 0
    alarm_frame: int | None = None
    alarm_time: float | None = None


def norm_dist(bag: Det, person: Det) -> float:
    """Ground-plane distance in person-heights (note 1)."""
    if person.height < MIN_PERSON_H:
        return float("inf")
    bx, by = bag.base
    px, py = person.base
    return math.hypot(bx - px, by - py) / person.height


def detections(res, keep: set[int] | None = None) -> dict[int, Det]:
    """Tracked boxes only -- detections without an ID cannot be reasoned about over time.

    keep: if given, only detections whose class is in this set are returned. Used to
    split a single COCO tracker's output into people and luggage without running the
    tracker twice (which would corrupt its shared state).
    """
    out: dict[int, Det] = {}
    b = getattr(res, "boxes", None)
    if b is None or b.id is None:
        return out
    names = getattr(res, "names", {}) or {}
    for box, tid, conf, cls in zip(b.xyxy.tolist(), b.id.tolist(), b.conf.tolist(), b.cls.tolist()):
        c = int(cls)
        if keep is not None and c not in keep:
            continue
        out[int(tid)] = Det(int(tid), tuple(box), float(conf), c, str(names.get(c, c)))
    return out


def update(bags: dict[int, Det], people: dict[int, Det], states: dict[int, BagState],
           frame_i: int, t: float, fps: float, events: list) -> None:
    own_frames = max(1, int(OWNERSHIP_SEC * fps))

    for tid, bag in bags.items():
        st = states.get(tid)
        if st is None:
            st = states[tid] = BagState(tid=tid, first_frame=frame_i, last_seen=frame_i)
        st.last_seen = frame_i

        dists = {pid: norm_dist(bag, p) for pid, p in people.items()}

        if st.state == PENDING:
            for pid, d in dists.items():
                if d <= D_OWN:
                    st.votes.setdefault(pid, []).append(d)
            if frame_i - st.first_frame >= own_frames:
                if st.votes:
                    st.owner = min(st.votes, key=lambda p: sum(st.votes[p]) / len(st.votes[p]))
                    st.state = OWNED
                else:
                    # nobody was ever near it: already unattended when it entered view
                    st.state = UNATTENDED
                    st.away_since = t
            continue

        if st.owner is not None and st.owner in dists:
            st.owner_gone = 0
            near = dists[st.owner] <= D_ALARM
        else:
            # owner not visible -- occlusion or departure (note 3)
            st.owner_gone += 1
            near = st.owner_gone < int(OWNER_GRACE_SEC * fps)

        if near:
            st.away_since = None
            if st.state != ALARM:
                st.state = OWNED
        else:
            if st.away_since is None:
                st.away_since = t
                st.state = UNATTENDED
            elif st.state != ALARM and t - st.away_since >= T_ALARM:
                st.state = ALARM
                st.alarm_frame, st.alarm_time = frame_i, t
                events.append({"bag_track": tid, "owner_track": st.owner,
                               "first_seen_frame": st.first_frame,
                               "left_at_sec": round(st.away_since, 2),
                               "alarm_at_sec": round(t, 2), "alarm_frame": frame_i})
                print(f"  [ALARM] t={t:7.1f}s  bag#{tid}  owner#{st.owner}  "
                      f"unattended since {st.away_since:.1f}s")

    for tid in [k for k, s in states.items() if frame_i - s.last_seen > BAG_DROP_SEC * fps]:
        del states[tid]


COLOR = {PENDING: (200, 200, 200), OWNED: (80, 200, 80),
         UNATTENDED: (0, 165, 255), ALARM: (0, 0, 255)}


def draw(frame, bags, people, states, t):
    for p in people.values():
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 1)
        cv2.putText(frame, f"{p.name} P{p.tid} {p.conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 0), 1)

    alarms = 0
    for tid, bag in bags.items():
        st = states.get(tid)
        state = st.state if st else PENDING
        c = COLOR[state]
        x1, y1, x2, y2 = (int(v) for v in bag.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        tag = f"{bag.name} B{tid} {bag.conf:.2f}"
        if st and st.owner is not None:
            tag += f" own:P{st.owner}"
        if st and st.away_since is not None and state != ALARM:
            tag += f" {T_ALARM - (t - st.away_since):.0f}s"
        if state == ALARM:
            tag, alarms = f"{bag.name} B{tid} UNATTENDED", alarms + 1
        cv2.putText(frame, tag, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        if st and st.owner in people:  # link the bag to its owner
            bx, by = (int(v) for v in bag.base)
            px, py = (int(v) for v in people[st.owner].base)
            cv2.line(frame, (bx, by), (px, py), c, 1)

    cv2.putText(frame, f"t={t:6.1f}s  bags={len(bags)}  people={len(people)}  alarms={alarms}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if alarms:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 4)
    return frame


def main():
    coco_luggage = COCO_LUGGAGE or SINGLE_MODEL is not None
    luggage_weights = SINGLE_MODEL or LUGGAGE_WEIGHTS
    person_weights = SINGLE_MODEL or PERSON_WEIGHTS
    luggage_classes = COCO_LUGGAGE_CLASSES if coco_luggage else None

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  {SOURCE}  {w}x{h} @ {fps:.2f} fps, {total} frames")
    print(f"  luggage: {luggage_weights}"
          f"{'  (COCO classes 24/26/28)' if coco_luggage else ''}")
    print(f"  person : {person_weights}")
    print(f"  owner within {D_OWN} person-heights, away beyond {D_ALARM}, alarm after {T_ALARM}s\n")

    luggage_model = YOLO(luggage_weights)
    # reuse the same object when both roles share a model -- avoids loading twice
    person_model = luggage_model if person_weights == luggage_weights else YOLO(person_weights)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    states: dict[int, BagState] = {}
    events: list = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (LIMIT and i >= LIMIT):
            break
        t = i / fps

        if person_model is luggage_model:
            # single shared model: ONE track() call, then split by class. Calling
            # track(persist=True) twice on one instance would corrupt the tracker's
            # single internal state (note: person and bag IDs then share one space).
            wanted = [PERSON_CLASS] + (luggage_classes or COCO_LUGGAGE_CLASSES)
            conf = min(PERSON_CONF, LUGGAGE_CONF)
            r = person_model.track(frame, persist=True, tracker=TRACKER, classes=wanted,
                                   conf=conf, verbose=False)[0]
            people = detections(r, keep={PERSON_CLASS})
            bags = detections(r, keep=set(luggage_classes or COCO_LUGGAGE_CLASSES))
        else:
            pr = person_model.track(frame, persist=True, tracker=TRACKER, classes=[PERSON_CLASS],
                                    conf=PERSON_CONF, verbose=False)[0]
            lr = luggage_model.track(frame, persist=True, tracker=TRACKER, classes=luggage_classes,
                                     conf=LUGGAGE_CONF, verbose=False)[0]
            people, bags = detections(pr), detections(lr)

        update(bags, people, states, i, t, fps, events)
        writer.write(draw(frame, bags, people, states, t))
        if SHOW:
            cv2.imshow("unattended", frame)
            if cv2.waitKey(1) == 27:
                break
        i += 1
        if i % 200 == 0:
            print(f"    frame {i}/{total or '?'}  t={t:.0f}s  tracked bags={len(states)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(EVENTS_JSON, "w") as f:
        json.dump({"source": SOURCE, "fps": fps, "frames": i,
                   "params": {"d_own": D_OWN, "d_alarm": D_ALARM, "t_alarm": T_ALARM},
                   "events": events}, f, indent=2)

    print(f"\n  {i} frames, {len(events)} alarm(s) -> {OUT}, {EVENTS_JSON}")
    for e in events:
        print(f"    bag#{e['bag_track']} owner#{e['owner_track']} "
              f"left {e['left_at_sec']}s, alarm {e['alarm_at_sec']}s")


if __name__ == "__main__":
    main()
