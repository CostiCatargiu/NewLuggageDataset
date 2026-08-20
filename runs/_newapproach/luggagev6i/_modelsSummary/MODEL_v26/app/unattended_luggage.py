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

Usage:
    python unattended_luggage.py --source clip.mp4
    python unattended_luggage.py --source clip.mp4 --out annotated.mp4 --limit 3000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

# =============================================================================
LUGGAGE_WEIGHTS = "runs_yolo26_overnight_r1213_v6i/y26_scb3_sbb50_cls075/weights/best.pt"
PERSON_WEIGHTS = "yolov12x.pt"
TRACKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_static.yaml")

PERSON_CLASS = 0  # COCO 'person'
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


def detections(res) -> dict[int, Det]:
    """Tracked boxes only -- detections without an ID cannot be reasoned about over time."""
    out: dict[int, Det] = {}
    b = getattr(res, "boxes", None)
    if b is None or b.id is None:
        return out
    for box, tid, conf, cls in zip(b.xyxy.tolist(), b.id.tolist(), b.conf.tolist(), b.cls.tolist()):
        out[int(tid)] = Det(int(tid), tuple(box), float(conf), int(cls))
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
        cv2.putText(frame, f"P{p.tid}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 0), 1)

    alarms = 0
    for tid, bag in bags.items():
        st = states.get(tid)
        state = st.state if st else PENDING
        c = COLOR[state]
        x1, y1, x2, y2 = (int(v) for v in bag.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        tag = f"B{tid}"
        if st and st.owner is not None:
            tag += f" own:P{st.owner}"
        if st and st.away_since is not None and state != ALARM:
            tag += f" {T_ALARM - (t - st.away_since):.0f}s"
        if state == ALARM:
            tag, alarms = f"B{tid} UNATTENDED", alarms + 1
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="unattended_out.mp4")
    ap.add_argument("--luggage", default=LUGGAGE_WEIGHTS)
    ap.add_argument("--person", default=PERSON_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  {a.source}  {w}x{h} @ {fps:.2f} fps, {total} frames")
    print(f"  luggage: {a.luggage}")
    print(f"  person : {a.person}")
    print(f"  owner within {D_OWN} person-heights, away beyond {D_ALARM}, alarm after {T_ALARM}s\n")

    luggage_model, person_model = YOLO(a.luggage), YOLO(a.person)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    states: dict[int, BagState] = {}
    events: list = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.limit and i >= a.limit):
            break
        t = i / fps

        pr = person_model.track(frame, persist=True, tracker=TRACKER, classes=[PERSON_CLASS],
                                conf=PERSON_CONF, verbose=False)[0]
        lr = luggage_model.track(frame, persist=True, tracker=TRACKER,
                                 conf=LUGGAGE_CONF, verbose=False)[0]
        people, bags = detections(pr), detections(lr)

        update(bags, people, states, i, t, fps, events)
        writer.write(draw(frame, bags, people, states, t))
        if a.show:
            cv2.imshow("unattended", frame)
            if cv2.waitKey(1) == 27:
                break
        i += 1
        if i % 200 == 0:
            print(f"    frame {i}/{total or '?'}  t={t:.0f}s  tracked bags={len(states)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open("unattended_events.json", "w") as f:
        json.dump({"source": a.source, "fps": fps, "frames": i,
                   "params": {"d_own": D_OWN, "d_alarm": D_ALARM, "t_alarm": T_ALARM},
                   "events": events}, f, indent=2)

    print(f"\n  {i} frames, {len(events)} alarm(s) -> {a.out}, unattended_events.json")
    for e in events:
        print(f"    bag#{e['bag_track']} owner#{e['owner_track']} "
              f"left {e['left_at_sec']}s, alarm {e['alarm_at_sec']}s")


if __name__ == "__main__":
    main()
