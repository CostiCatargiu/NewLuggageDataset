#!/usr/bin/env python3
"""Regression tests for the tracking-identity machinery.

Run it from this folder:   python test_id_stability.py

No weights, no GPU, no dataset: the detector is stubbed and the clips are drawn with
OpenCV, so the whole suite finishes in a few seconds. Every check here exists because
something once went wrong -- the last three in GROUP D are the crowd re-bind failure
found on AVSSS07_MEDIUmo, where a colour match handed bags to whoever walked past.

Exit code is 0 only if every check passes.
"""

import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# By default the module beside this file. Pass a path to test a different copy:
#     python test_id_stability.py /somewhere/else/inferencetrackingimprovedUpdatedorig.py
TARGET = (sys.argv[1] if len(sys.argv) > 1
          else os.path.join(HERE, "inferencetrackingimprovedUpdatedorig.py"))
if not os.path.exists(TARGET):
    sys.exit("no module to test at %s -- pass its path as an argument" % TARGET)
print("testing: %s  (%d bytes, modified %s)"
      % (TARGET, os.path.getsize(TARGET),
         __import__("time").strftime("%Y-%m-%d %H:%M:%S",
                                     __import__("time").localtime(os.path.getmtime(TARGET)))))

# stub the detector package: importing the real one drags in torch for nothing
_stub = types.ModuleType("ultralytics")
_stub.YOLO = lambda *a, **k: None
sys.modules.setdefault("ultralytics", _stub)

_spec = importlib.util.spec_from_file_location("_uut", TARGET)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

# Fail with one clear line instead of a KeyError deep inside a scenario when the file under
# test is an older copy -- with several checkouts of this script around, that is the single
# most likely reason for a red run.
_REQUIRED = ["output_video_path", "appearance_of", "appearance_sim", "update_appearance",
             "recover_with_low_conf", "revive_from_memory", "away_evidence",
             "OWNER_REBIND_NEWBORN_SLACK", "PERSON_MEMORY_SEC", "LOW_CONF_RECOVERY"]
_missing = [a for a in _REQUIRED if not hasattr(M, a)]
if _missing:
    sys.exit("this copy of the module predates the tests -- it has no %s.\n"
             "Point the tests at the current file, or copy the current file over this one."
             % ", ".join(_missing))

# The suite tests the LOGIC, so it pins every threshold its assertions depend on. Without
# this, tuning the pipeline for a dataset turns the regression tests red for no reason.
PINNED = dict(
    CONF_PERSON=0.25, CONF_LUGGAGE=0.40, CONF_PERSON_LOW=0.10, CONF_LUGGAGE_LOW=0.15,
    LOW_CONF_RECOVERY=True, APPEARANCE_ENABLED=True,
    PERSON_TTL_SECONDS=6.0, PERSON_MEMORY_SEC=20.0,
    OWNER_REBIND_SEC=3.0, OWNER_REBIND_IOU=0.40, OWNER_REBIND_BY_APPEARANCE=True,
    OWNER_REBIND_MIN_GAP=0.5, OWNER_REBIND_APP_SEC=5.0, OWNER_REBIND_MIN_SIM=0.80,
    OWNER_REBIND_SIM_MARGIN=0.05, OWNER_REBIND_MAX_H=2.0, OWNER_REBIND_NEWBORN_SLACK=0.5,
    APP_MIN_SIM_REVIVE=0.35, D_OWN=1.5, OWNERSHIP_SEC=2.0, OWNER_GRACE_SEC=2.0,
)
_differs = {k: getattr(M, k) for k, v in PINNED.items() if getattr(M, k) != v}
for k, v in PINNED.items():
    setattr(M, k, v)
if _differs:
    print("note: these are pinned to their defaults for the tests; your file has %s"
          % ", ".join("%s=%s" % kv for kv in sorted(_differs.items())))

import cv2  # noqa: E402  (after M, so a missing cv2 blames the module first)

FAILED = []


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


# =============================================================================
# GROUP A -- output location and appearance signature
# =============================================================================
print("\nA. basics")
p = M.output_video_path("/data/vids/AVSSS07_MEDIUmo.mpg")
ok(p == "/data/vids/AVSSS07_MEDIUmo" + M.OUT_VIDEO_SUFFIX + M.OUT_VIDEO_EXT,
   "the annotated video lands beside the input -> %s" % p)

FRAME = np.full((400, 600, 3), 30, np.uint8)
RED_BOX = [100, 100, 160, 260]
BLUE_BOX = [400, 100, 460, 260]
cv2.rectangle(FRAME, (100, 100), (160, 260), (0, 0, 220), -1)
cv2.rectangle(FRAME, (400, 100), (460, 260), (220, 0, 0), -1)
red = M.appearance_of(FRAME, RED_BOX, "person")
blue = M.appearance_of(FRAME, BLUE_BOX, "person")
ok(M.appearance_sim(red, red) > 0.99 and M.appearance_sim(red, blue) < 0.20,
   "colour signature separates two people (self %.2f, other %.2f)"
   % (M.appearance_sim(red, red), M.appearance_sim(red, blue)))
ok(M.appearance_sim(red, None) is None,
   "an unusable signature degrades to None instead of a wrong number")


# =============================================================================
# GROUP B -- one role driven frame by frame, the way run() drives it
# =============================================================================
DT = 1 / 25.0


def det(bb, conf, role="person"):
    return {"bbox": np.array(bb, float), "conf": conf, "cls": 0,
            "app": M.appearance_of(FRAME, bb, role)}


def step(hi, lo, tracks, retired, now_t):
    """One frame of association + spawning, mirroring the person block in run()."""
    m, un_d, un_t = M.hungarian_match(
        hi, tracks, now_t, require_same_class=False,
        max_relink_age=M.PERSON_MAX_RELINK_AGE,
        match_iou_thr=M.PERSON_MATCH_IOU_THR,
        match_dist_thr=M.PERSON_MATCH_DIST_THR,
        class_mismatch_penalty=0.0, app_weight=M.W_APP)
    M.update_tracks_with_matches(hi, tracks, m, set(), dt_frame=DT, now_t=now_t,
                                 smooth_alpha=M.PERSON_SMOOTH_ALPHA,
                                 vel_alpha=M.PERSON_VEL_ALPHA,
                                 update_class=False, update_app=True)
    low, un_t = M.recover_with_low_conf(lo, tracks, un_t, now_t, role="person")
    M.update_tracks_with_matches(lo, tracks, low, un_t, dt_frame=DT, now_t=now_t,
                                 smooth_alpha=M.PERSON_SMOOTH_ALPHA * M.LOW_CONF_SMOOTH_SCALE,
                                 vel_alpha=M.PERSON_VEL_ALPHA * M.LOW_CONF_SMOOTH_SCALE,
                                 update_class=False, update_app=False)
    outcome = []
    for di in un_d:
        d = hi[di]
        if not M.should_spawn_new_track(d, tracks, now_t, M.PERSON_NEW_TRACK_SUPPRESS_IOU,
                                        M.PERSON_NEW_TRACK_SUPPRESS_DIST,
                                        M.PERSON_NEW_TRACK_SUPPRESS_MAX_AGE):
            continue
        old = M.revive_from_memory(d, retired, now_t, M.PERSON_MEMORY_SEC,
                                   M.PERSON_REVIVE_IOU, M.PERSON_REVIVE_DIST)
        if old is not None:
            st = retired.pop(old)
            st.update(bbox=np.array(d["bbox"], float), conf=d["conf"],
                      last_seen_t=now_t, missed_s=0.0, vx=0.0, vy=0.0)
            M.update_appearance(st, d.get("app"))
            tracks[old] = st
            outcome.append(("revived", old))
            continue
        nid = max(list(tracks) + list(retired) + [0]) + 1
        tracks[nid] = {"bbox": np.array(d["bbox"], float), "conf": d["conf"], "cls": 0,
                       "first_t": now_t, "last_seen_t": now_t, "missed_s": 0.0,
                       "vx": 0.0, "vy": 0.0, "app": d.get("app")}
        outcome.append(("new", nid))
    for tid, st in M.prune_tracks_by_ttl(tracks, M.PERSON_TTL_SECONDS).items():
        retired[tid] = st
    return outcome


print("\nB. an identity survives a gap in the detections")
tracks, retired, t = {}, {}, 0.0
for _ in range(10):
    step([det(RED_BOX, 0.9)], [], tracks, retired, t); t += DT
first = list(tracks)[0]

for _ in range(25):                       # 1 s where the detector only manages 0.15
    step([], [det(RED_BOX, 0.15)], tracks, retired, t); t += DT
ok(list(tracks) == [first] and tracks[first]["missed_s"] == 0.0,
   "weak boxes keep the id alive through a 1 s partial occlusion (ids %s)" % list(tracks))

for _ in range(int(M.PERSON_TTL_SECONDS / DT) + 5):   # nothing at all: the track dies
    step([], [], tracks, retired, t); t += DT
ok(not tracks and first in retired, "the track is retired, not deleted, after the TTL")

out = step([det([115, 100, 175, 260], 0.9)], [], tracks, retired, t); t += DT
ok(out == [("revived", first)], "the same person re-appearing keeps their id -> %s" % out)

tracks2, retired2, t2 = {}, {}, 0.0
for _ in range(10):
    step([det(RED_BOX, 0.9)], [], tracks2, retired2, t2); t2 += DT
for _ in range(int(M.PERSON_TTL_SECONDS / DT) + 5):
    step([], [], tracks2, retired2, t2); t2 += DT
impostor = {"bbox": np.array(RED_BOX, float), "conf": 0.9, "cls": 0,
            "app": M.appearance_of(FRAME, BLUE_BOX, "person")}
out = step([impostor], [], tracks2, retired2, t2)
ok(out and out[0][0] == "new",
   "somebody else standing in that exact spot does NOT inherit the id -> %s" % out)


# =============================================================================
# GROUP D -- owner re-binding (the AVSSS07_MEDIUmo failure)
# =============================================================================
print("\nD. re-binding an owner whose track was re-created")
bag = {"lid": 1, "owner_pid": 7, "owner_bbox": np.array(RED_BOX, float),
       "owner_seen_t": 10.0, "owner_app": red, "state": M.OWNED}
newborn = [{"tid": 9, "bbox": np.array([112, 100, 172, 260], float), "first_t": 12.0,
            "app": red},
           {"tid": 11, "bbox": np.array(BLUE_BOX, float), "first_t": 12.0, "app": blue}]

pid, how = M.rebind_owner(bag, newborn, 14.0)
ok(pid == 9 and how == "appearance", "the owner is re-bound by colour -> P%s via %s" % (pid, how))
pid, _ = M.rebind_owner(bag, newborn, 14.0, taken_pids={9})
ok(pid is None, "a person who already owns another bag is never stolen")
pid, _ = M.rebind_owner(bag, [dict(p, first_t=2.0) for p in newborn], 14.0)
ok(pid is None, "a bystander id that predates the owner's disappearance is refused")

apart = [{"tid": 15, "bbox": np.array([200, 100, 260, 260], float), "first_t": 10.2,
          "app": red}]
pid, _ = M.rebind_owner(bag, apart, 10.4)
ok(pid is None, "colour re-bind does not fire on a momentary drop (0.4 s)")
pid, how = M.rebind_owner(bag, apart, 11.0)
ok(pid == 15 and how == "appearance", "...but does once the owner is really gone (1.0 s)")
pid, _ = M.rebind_owner(bag, [dict(apart[0], first_t=2.0)], 11.0)
ok(pid is None, "a long-lived look-alike is still refused at that gap")
twins = [dict(apart[0]), dict(apart[0], tid=13, bbox=np.array([180, 100, 240, 260], float))]
pid, _ = M.rebind_owner(bag, twins, 11.0)
ok(pid is None, "two equally similar candidates decide nothing")


# =============================================================================
# GROUP E -- whole pipeline, synthetic clip, stubbed detector
# =============================================================================
class _T:
    def __init__(self, a): self.a = a
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self, xy, cf, cl):
        self.xyxy = _T(np.array(xy, float).reshape(-1, 4))
        self.conf = _T(np.array(cf, float))
        self.cls = _T(np.array(cl, float))

    def __len__(self): return len(self.conf.a)


class _Res:
    def __init__(self, b): self.boxes = b


def make_fake_yolo(script):
    class FakeYOLO:
        names = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

        def __init__(self, *a, **k): self.n = -1

        def predict(self, **kw):
            self.n += 1
            bb, cf, cl = script(self.n)
            return [_Res(_Boxes(bb, cf, cl) if bb else _Boxes(np.zeros((0, 4)), [], []))]
    return FakeYOLO


def render(path, n_frames, boxes_at, size=(640, 480), fps=25):
    w, h = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n_frames):
        img = np.full((h, w, 3), 60, np.uint8)
        for (x1, y1, x2, y2), col in boxes_at(f):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
        vw.write(img)
    vw.release()


def run_clip(tmp, name, n_frames, boxes_at, script, **overrides):
    vid = os.path.join(tmp, name + ".mp4")
    render(vid, n_frames, boxes_at)
    runs = os.path.join(tmp, name + "_runs")
    shutil.rmtree(runs, ignore_errors=True)
    saved = {k: getattr(M, k) for k in
             ("YOLO", "VIDEO_IN", "OUT_DIR", "SINGLE_MODEL", "SHOW", "SAVE_OUTPUT", "DEVICE",
              "LOG_DETECTIONS", "LOG_PAIRS_EVERY_N", "OWNERSHIP_SEC", "UNATTENDED_SECONDS",
              "D_AWAY")}
    for k, v in PINNED.items():
        setattr(M, k, v)
    M.YOLO = make_fake_yolo(script)
    M.VIDEO_IN, M.OUT_DIR, M.SINGLE_MODEL = vid, runs, "fake.pt"
    M.SHOW, M.SAVE_OUTPUT, M.DEVICE = False, True, "cpu"
    M.LOG_DETECTIONS, M.LOG_PAIRS_EVERY_N = False, 25
    for k, v in overrides.items():
        setattr(M, k, v)
    try:
        M.run()
    finally:
        for k, v in saved.items():
            setattr(M, k, v)
    rd = os.path.join(runs, sorted(os.listdir(runs))[-1])
    ev = json.load(open(os.path.join(rd, "events.json")))
    end = [json.loads(l) for l in open(os.path.join(rd, "trace.jsonl"))
           if '"end"' in l][-1]
    return ev, end, os.path.splitext(vid)[0] + M.OUT_VIDEO_SUFFIX + M.OUT_VIDEO_EXT


BAG = (300, 300, 340, 345)

print("\nE. end to end: the owner is watched walking away")
with tempfile.TemporaryDirectory() as tmp:
    def person_box(f):
        # beside the bag, then a brisk walk to the far side of the frame
        x = 260 if f < 150 else min(280 + (f - 150) * 2.0, 600)
        return (x - 22, 250, x + 23, 360)

    def bconf(f):
        if 420 <= f < 450: return None      # the bag vanishes briefly
        if 350 <= f < 400: return 0.20      # ...and is detected weakly for 2 s
        return 0.85

    def make_script(pconf):
        def script(f):
            bb, cf, cl = [], [], []
            c = pconf(f)
            if c is not None: bb.append(person_box(f)); cf.append(c); cl.append(0)
            c = bconf(f)
            if c is not None: bb.append(BAG); cf.append(c); cl.append(28)
            return bb, cf, cl
        return script

    def watched(f):
        if 450 <= f < 625: return None      # 7 s dropout: the track dies, must revive
        if 100 <= f < 140: return 0.15      # 1.6 s of weak boxes
        return 0.90

    draw = lambda f: [(BAG, (200, 120, 0)), (person_box(f), (0, 0, 200))]
    ev, end, outvid = run_clip(tmp, "watched", 700, draw, make_script(watched),
                               OWNERSHIP_SEC=1.0, D_AWAY=1.5, UNATTENDED_SECONDS=3.0)

    ok(os.path.exists(outvid) and os.path.getsize(outvid) > 10000,
       "the annotated video was written beside the input")
    ok(end["person_ids_used"] == 1 and end["luggage_ids_used"] == 1,
       "one person id and one luggage id survive the whole clip (%d / %d)"
       % (end["person_ids_used"], end["luggage_ids_used"]))
    ok(len(ev["events"]) == 1 and ev["events"][0]["owner_track"] == 1,
       "exactly one alarm, naming the original owner")
    c = end["event_counts"]
    ok(c.get("low_conf_recovered[person]", 0) > 20
       and c.get("low_conf_recovered[luggage]", 0) > 20,
       "weak boxes rescued both roles (%s / %s)"
       % (c.get("low_conf_recovered[person]"), c.get("low_conf_recovered[luggage]")))
    ok(c.get("track_revived[person]", 0) == 1, "the person came back under their own id")
    a = ev["events"][0]
    ok(a["bag_drift_w"] < 1.0 and a["owner_seen_frac"] > 0.9 and a["owner_d_max_h"] > 1.5,
       "the alarm is graded as OBSERVED: drift %.2fw, owner seen %.0f%%, reached %s h"
       % (a["bag_drift_w"], a["owner_seen_frac"] * 100, a["owner_d_max_h"]))

    # the same clip, but the detector loses the owner exactly while the clock runs. The
    # alarm is identical; only the evidence separates a departure from a tracking failure.
    def lost(f):
        if f >= 200: return None            # he vanishes from the detector, not the scene
        if 100 <= f < 140: return 0.15
        return 0.90

    ev2, _, _ = run_clip(tmp, "lost", 700, draw, make_script(lost),
                         OWNERSHIP_SEC=1.0, D_AWAY=1.5, UNATTENDED_SECONDS=3.0)
    ok(len(ev2["events"]) == 1 and ev2["events"][0]["owner_seen_frac"] == 0.0,
       "an alarm raised on a LOST owner is marked as such (seen %.0f%%)"
       % (ev2["events"][0]["owner_seen_frac"] * 100 if ev2["events"] else -1))

print("\nF. end to end: the owner never leaves, the detector just loses him")
with tempfile.TemporaryDirectory() as tmp:
    PERSON = (248, 250, 293, 360)
    GAP = range(100, 275)                       # 7 s: long enough to kill the track

    def script(f):
        bb, cf, cl = [BAG], [0.85], [28]
        if f not in GAP:
            bb.append(PERSON); cf.append(0.90); cl.append(0)
        return bb, cf, cl

    ev, end, _ = run_clip(
        tmp, "stays", 500,
        lambda f: [(BAG, (200, 120, 0)), (PERSON, (0, 0, 200))], script,
        OWNERSHIP_SEC=1.0, UNATTENDED_SECONDS=10.0)
    ok(len(ev["events"]) == 0 and end["person_ids_used"] == 1,
       "no alarm and no id churn when the owner was there all along "
       "(%d alarm(s), %d id(s))" % (len(ev["events"]), end["person_ids_used"]))

print()
if FAILED:
    print("%d CHECK(S) FAILED:" % len(FAILED))
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all checks passed")
