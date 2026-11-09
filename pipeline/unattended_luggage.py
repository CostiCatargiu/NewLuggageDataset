#!/usr/bin/env python3
"""Unattended-luggage detection from a fixed camera.

Repository usage (run from the repository root; see pipeline/README.md):
    python pipeline/unattended_luggage.py --model yolov12-custom videos/video1.avi     # Environment A
    python pipeline/unattended_luggage.py --model yolo26-custom  videos/video1.avi     # Environment B
    python pipeline/unattended_luggage.py --all --models yolov12-official,yolov12-custom  # all TEST_VIDEOS

Two detectors run on every frame -- a luggage model and a person model (optionally the
same COCO model in both roles, see MODEL_PROFILES) -- each feeding its own Hungarian
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

4. AN IDENTITY MUST SURVIVE A GAP IN THE DETECTIONS. A detector misses an object for a few
   frames whenever someone walks in front of it, and a tracker that answers by killing the
   track and giving the object a new number destroys the very thing this script measures:
   the bag/owner link. Three mechanisms keep the number: weak detections below the
   reporting threshold get their own association pass (they cannot start a track, but they
   can keep one alive); every track carries a colour signature that re-attaches its old id
   when it re-appears; and people, like bags, are remembered after their track dies so a
   returning person comes back as themselves rather than as a stranger.

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

# Bumped whenever the tracking or ownership logic changes. Every run prints the file it
# came from and records it in the log, because with several checkouts of this script around
# the most expensive mistake is analysing results produced by a copy you did not edit.
SCRIPT_VERSION = "2026-09-28b.hud-big"  # tracking/ownership logic = 2026-09-25j (HUD restyled only)


def script_identity():
    """Path, size and mtime of the file actually executing, for the banner and the log."""
    try:
        path = os.path.abspath(__file__)
        return {"script": path, "script_bytes": os.path.getsize(path),
                "script_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(os.path.getmtime(path))),
                "version": SCRIPT_VERSION}
    except (NameError, OSError):
        return {"script": "<unknown>", "script_bytes": -1,
                "script_mtime": "?", "version": SCRIPT_VERSION}


# ================================ I/O ========================================
VIDEO_IN = r"videos/AVSS07_easy.mpg"  # default input; relative paths are from the folder you run in
# Parameters were tuned on the AVSS 2007 medium clip (the development sequence).
# A video given on the command line overrides this:  python3 <script>.py /path/to/video
# The annotated video is written NEXT TO THE INPUT, named after the input AND the model
# profile (e.g. video9_yolo26-custom_output.mp4), so runs of different models never
# overwrite each other.
OUT_VIDEO_SUFFIX = "_output"
OUT_VIDEO_EXT = ".mp4"  # ".mp4" (mp4v) or ".avi" (XVID); the codec follows the extension
# Every artefact of a run lands in OUT_DIR/<video>__<model profile>__<stamp><tag>/
OUT_DIR = r"runs_pipeline"
RUN_TAG = ""  # optional suffix for the folder name, e.g. "_down2.0"
EVENTS_JSON = "events.json"
# Newline-delimited JSON trace of everything that happened, for offline analysis:
#   import pandas as pd; df = pd.read_json("trace.jsonl", lines=True)
LOG_JSONL = "trace.jsonl"
LOG_DETECTIONS = True  # one record per raw detection (large, but complete)
LOG_PAIRS_EVERY_N = 1  # bag/owner snapshot cadence in frames (0 = never)
SAVE_OUTPUT = True
SHOW = True
FRAME_LIMIT = 0  # stop after N frames (0 = whole video)
WINDOW_NAME = "Unattended Luggage   [q] quit   [space] pause   [s] screenshot"

# =============================== MODELS ======================================
# Four detector set-ups are compared on the same videos. Choose one with MODEL_PROFILE, or
# on the command line:
#   python3 <script>.py --model yolo26-custom /path/to/video     one model, one video
#   python3 <script>.py --all                                   every model x TEST_VIDEOS
# A custom luggage model has no 'person' class, so persons always come from the official
# COCO model of the same family; official vs custom differ ONLY in the luggage detector.
# Relative paths are resolved from the folder you run the script in (as before).
MODEL_PROFILES = {
# Luggage detectors = the released checkpoints in weights/ (download them from the Release).
# Persons = the official COCO model of the same family (a bare name is fetched by Ultralytics
# when possible; otherwise put the file in the folder you run from).
# YOLOv12 profiles need Environment A, YOLO26 profiles Environment B (see the main README).
    "yolov12-official": {"label": "YOLOv12 official",
                         "person": r"yolov12x.pt",
                         "luggage": r"weights/yolov12s_default.pt"},
    "yolov12-custom": {"label": "YOLOv12 custom",
                       "person": r"yolov12x.pt",
                       "luggage": r"weights/yolov12s_lsshift_sqrt.pt"},
    "yolo26-official": {"label": "YOLO26 official",
                        "person": r"yolo26x.pt",
                        "luggage": r"weights/yolo26s_default.pt"},
    "yolo26-custom": {"label": "YOLO26 custom",
                      "person": r"yolo26x.pt",
                      "luggage": r"weights/yolo26s_p234rich_beta1.pt"},
}
MODEL_PROFILE = "yolov12-custom"  # used when --model is not given
# True: pressing Run with no arguments (e.g. in PyCharm) runs EVERY profile on EVERY video
# in TEST_VIDEOS and prints one summary table at the end (same as --all).
RUN_ALL_ON_START = False  # the two families need different environments
MODEL_LABEL = MODEL_PROFILES[MODEL_PROFILE]["label"]  # set again in run()

# The test set: --all runs every profile on each of these, one after another.
# The five clips of the paper (AVSS 2007 medium is the development sequence).
TEST_VIDEOS = [
    r"videos/AVSS07_medium.mpg",
    r"videos/AVSS07_easy.mpg",
    r"videos/video1.avi",
    r"videos/video9.avi",
    r"videos/video10.avi",
]

PERSON_CLASS_IDS = [0]  # COCO 'person'
# None = keep every class the luggage model emits (correct for a custom luggage model).
LUGGAGE_CLASS_IDS = None
COCO_LUGGAGE_CLASS_IDS = [24, 26, 28]  # backpack, handbag, suitcase -- used with an official model

# Report every detector in the custom dataset's label space, so a stock COCO model and the
# custom model produce identical annotations and identical event logs.
CUSTOM_LUGGAGE_NAMES = {0: "backpack", 1: "bag", 2: "trolley"}
COCO_TO_CUSTOM = {24: 0, 26: 1, 28: 2}  # backpack->backpack, handbag->bag, suitcase->trolley
REMAP_COCO_LUGGAGE = True  # applied only when the luggage detector is a COCO model

CONF_PERSON = 0.25
CONF_LUGGAGE = 0.20
IOU = 0.45
IMGSZ = 640
DEVICE = "0"  # "0" GPU, or "cpu"

# ======================== UNATTENDED PARAMETERS ==============================
# distances in PERSON-HEIGHTS (note 2)
D_OWN = 1.5  # must be this close to be a candidate owner
D_AWAY = 2.5  # farther than this counts as "away"
OWNERSHIP_SEC = 2.0  # window used to elect the owner (note 3)
# The election takes the smallest MEAN distance, which on its own lets one lucky frame beat
# a person who stood by the bag for a second. Measured on AVSSS07_MEDIUmo: a bag was given
# to a passer-by seen beside it for ONE frame over the person carrying it, seen for 25.
OWNERSHIP_MIN_VOTES = 5  # frames a candidate must be near the bag to be electable
OWNERSHIP_MIN_VOTE_FRAC = 0.5  # ...and at least this share of the best-supported candidate
# A person the detector never saw clearly should not be made responsible for a bag.
OWNER_REQUIRE_STRONG = True
OWNER_STRONG_MAX_AGE = 3.0  # seconds since the candidate's last above-threshold detection
UNATTENDED_SECONDS = 10.0  # seconds away before the alarm fires
OWNER_GRACE_SEC = 4.0  # owner track may vanish this long before counting as away
# (was 2.0) -- only delays the verdict for an owner who was NEAR when they vanished;
# an owner already walking away is counted as away before they disappear.
MIN_PERSON_H = 20.0  # px; below this the height estimate is too noisy to divide by

# =========================
# STABLE TRACKING PARAMS
# =========================
PERSON_TTL_SECONDS = 6.0
# An abandoned bag must outlive long crowd occlusions: losing the track means losing the
# owner with it, and the fresh track would elect whoever happens to stand there next.
LUGGAGE_TTL_SECONDS = 45.0

# The relink age must never be SHORTER than the TTL. A track that is still alive but no
# longer matchable is a guaranteed id change the moment the object is detected again.
PERSON_MAX_RELINK_AGE = PERSON_TTL_SECONDS
LUGGAGE_MAX_RELINK_AGE = LUGGAGE_TTL_SECONDS
# The matching gate widens with the time a track has been missing -- after two seconds of
# occlusion the object is genuinely somewhere else, and a fixed radius rejects it.
GATE_GROWTH = 0.6  # extra gate radius per second missing
GATE_MAX_SCALE = 1.5  # ...never more than this multiple of the base radius

# EVERY matching radius below is relative to the object, not to the image. A flat 260 px is
# one step for a person standing near the camera and half the platform for a handbag at the
# far end, so a fixed radius lets a small bag's track jump to a completely different object
# and carry its owner and its timer along. Measured on AVSSS07_MEDIUmo: bag tracks with
# ~40 px boxes wandering 324-514 px across the scene, and 11 ownership resets in one clip.
MATCH_DIST_DIAG = 3.0  # matching gate, in multiples of the track's own box diagonal
MATCH_DIST_MIN_PX = 40  # ...with a floor, so a tiny box can still move at all
LOW_MATCH_DIAG = 2.0  # the weak-box recovery pass is stricter
SUPPRESS_DIAG = 1.5  # "this detection is already a track" is a near-overlap question
REVIVE_DIAG = 2.5  # and so is "this is the same object coming back"
PREDICT_MAX_SEC = 2.0  # never extrapolate a track's motion further than this
VEL_DECAY_WHEN_MISSED = 0.90  # unmatched tracks stop drifting instead of flying off

# Identity memory: a bag that vanished completely keeps its ownership if it comes back
# in the same place, instead of being reborn with a new ID and a new owner.
LUGGAGE_MEMORY_SEC = 90.0
LUGGAGE_REVIVE_IOU = 0.30
LUGGAGE_REVIVE_DIST = 90  # px between ground-contact points

# The same memory for PEOPLE. Losing a person id is worse than losing a bag id: every bag
# that named them as owner is instantly orphaned and elects whoever stands there next.
PERSON_MEMORY_SEC = 20.0
PERSON_REVIVE_IOU = 0.20
PERSON_REVIVE_DIST = 160  # px between ground-contact points
REVIVE_DIST_GROWTH = 0.5  # the revive radius grows this much per second out of sight
# Overlapping the place an object was last seen only proves identity for a moment. After
# this long, "standing where they stood" is what a DIFFERENT person does, so past it the
# colour signature has to agree no matter how good the overlap looks.
REVIVE_TRUST_IOU_SEC = 1.0

# Duplicate-id suppression for people (IoU only: two people may legitimately stand close)
PERSON_NEW_TRACK_SUPPRESS_IOU = 0.55
PERSON_NEW_TRACK_SUPPRESS_DIST = 0  # 0 = disabled
PERSON_NEW_TRACK_SUPPRESS_MAX_AGE = 1.0

# =========================
# MATCHING CASCADE
# =========================
# A detection is offered first to the tracks confirmed most recently, and only then to
# staler ones. In a single Hungarian round a track that has not been seen for seconds --
# coasting on extrapolated velocity -- can outbid a track matched on every frame, and the
# two exchange objects. Measured on AVSSS07_MEDIUmo: a bag track unconfirmed for 7.0 s
# took the abandoned bag away from a track that had held it for 398 consecutive frames,
# and the same bag then alarmed twice under two identities.
MATCH_CASCADE = True
CASCADE_TIERS = (0.25, 1.0, 3.0, 1e9)  # upper bounds, seconds since last confirmation

# Matches worth explaining in the log: an old track winning, or a box teleporting.
LOG_ASSOC = True  # one record per role per frame: who matched what, and how far it moved
LOG_ALL_MATCHES = False  # True = one record per assignment (large, but fully replayable)
MATCH_SUSPECT_AGE = 1.0  # seconds
MATCH_SUSPECT_JUMP = 100  # px between the track's last box and the detection it took

# A bag track that jumps this far after going quiet is almost certainly a DIFFERENT object.
# Carrying the old owner across would let a stranger's departure alarm on a bag they never
# touched, so the identity survives and the ownership does not.
OWNERSHIP_RESET_MIN_AGE = 1.0  # seconds unconfirmed before a jump counts as suspicious
OWNERSHIP_RESET_JUMP = 2.0  # multiples of the track's own box diagonal

# Merging two tracks is irreversible, so one frame of overlap must not trigger it.
MERGE_MIN_FRAMES = 5  # consecutive frames above MERGE_TRACK_IOU

# =========================
# APPEARANCE RE-IDENTIFICATION
# =========================
# Geometry alone cannot carry an identity across an occlusion: after a second out of sight
# the predicted box is wrong and the nearest detection is often somebody else. So every
# track also carries a colour signature -- an HS histogram of its own pixels, updated
# slowly -- which survives the gap and is what actually re-attaches the old id.
APPEARANCE_ENABLED = True
APP_BINS = (8, 8)  # hue x saturation bins
APP_EMA = 0.80  # weight kept from the stored template on every update
APP_SHRINK = 0.12  # shrink the box before sampling, to drop background pixels
APP_PERSON_BAND = (0.10, 0.60)  # sample the torso only; head and legs change too much
W_APP = 2.5  # weight of (1 - similarity) in the matching cost
APP_MIN_SIM_REVIVE = 0.35  # a re-appearing detection must look at least this similar
APP_MIN_SIM_RELINK = 0.25  # ...to take over a track that has been missing a while

# =========================
# LOW-CONFIDENCE RECOVERY (second association pass)
# =========================
# Most "lost" frames are not missing detections -- the object is still detected, just under
# the confidence threshold because it is half occluded. Those weak boxes are useless for
# STARTING a track but perfect for KEEPING one, so they get their own association pass.
LOW_CONF_RECOVERY = True
CONF_PERSON_LOW = 0.10
CONF_LUGGAGE_LOW = 0.15
LOW_MATCH_IOU_THR = 0.20  # stricter than the high-confidence pass
LOW_MATCH_DIST_THR = 120
LOW_RECOVERY_MAX_AGE = 2.0  # seconds; a weak box may not resurrect an ancient track
LOW_CONF_SMOOTH_SCALE = 0.6  # a weak detection moves the box less than a strong one

# The person tracker may re-create the owner under a new ID after an occlusion.
OWNER_REBIND_SEC = 3.0  # how long the owner's last box stays valid for re-binding
OWNER_REBIND_IOU = 0.40
# If the owner is re-created too far away for IoU to catch, their colour signature is the
# only thing left. Deliberately conservative: a wrong re-bind silently cancels an alarm.
OWNER_REBIND_BY_APPEARANCE = True
OWNER_REBIND_APP_SEC = 5.0  # appearance re-bind is allowed this long after the owner vanished
OWNER_REBIND_MIN_GAP = 0.5  # ...but not before: one dropped frame is not id churn
OWNER_REBIND_MIN_SIM = 0.80  # and only for a strong colour match...
OWNER_REBIND_SIM_MARGIN = 0.05  # ...that clearly beats the runner-up...
OWNER_REBIND_MAX_H = 2.0  # ...within this many person-heights of where the owner last was
# The decisive constraint. A re-bind exists for ONE situation: the tracker destroyed the
# owner's track and re-created them under a fresh id. Only a track born after the owner
# vanished can be that. An id that already existed alongside the owner is somebody else,
# however similar the colours -- and in a crowd under one light, everybody matches.
OWNER_REBIND_NEWBORN_SLACK = 0.5  # seconds of tolerance on "born after the owner vanished"

# =========================
# BAG MOTION, TIGHT LUGGAGE GATING, CARRY-BASED OWNERSHIP  (v2026-09-16)
# =========================
# A bag is either being CARRIED (it moves, and somebody moves with it) or STANDING (it has
# not moved for BAG_STATIC_SEC). Only a standing bag can be abandoned, and the person who
# carried it there is its owner -- in a crowd, standing close to a bag proves nothing.
BAG_MOTION_WINDOW = 0.6  # seconds of trajectory used to estimate velocities
BAG_MOVING_SPEED = 0.8  # bag diagonals per second above which a bag counts as moving
BAG_STATIC_TOL = 0.5  # a standing bag may jitter this many diagonals around its anchor...
BAG_STATIC_SEC = 1.5  # ...and must stay inside that radius this long to count as standing
REQUIRE_STATIC_FOR_ALARM = True  # a moving bag is being carried: it is never unattended

# Luggage association. The old radius (3 diagonals, growing to 4.5) let a bag track take the
# neighbouring bag whenever its own was missed for ONE frame -- 15 such jumps on L1 alone in
# the 2026-09-16 AVSSS07_MEDIUmo run, each one dragging an owner and a timer along.
LUGGAGE_TIGHT_GATING = True
LUGGAGE_FRESH_AGE = 0.5  # seconds; a track missing less than this...
LUGGAGE_GATE_FRESH_DIAG = 1.0  # ...may move at most one diagonal from its prediction
LUGGAGE_GATE_STALE_DIAG = 1.5  # older tracks: 1.5 diagonals, still widened by GATE_GROWTH
LUGGAGE_GATE_IOU = 0.20  # ...unless the boxes overlap at least this much
LUGGAGE_STATIC_MATCH_IOU = 0.15  # a STANDING bag is only re-taken by a box overlapping it
LUGGAGE_STATIC_DIST_DIAG = 0.5  # ...or one landing within half a diagonal of it
# A box three times the size of the track is a different object (or a bag+person blob),
# however close its centre is. Overlapping boxes are exempt: a bag being picked up grows.
LUGGAGE_MAX_AREA_RATIO = 2.5
LUGGAGE_AREA_EXEMPT_IOU = 0.30
# A match that moves a quiet track further than OWNERSHIP_RESET_JUMP diagonals is refused:
# the detection is handled as a new (or revived) object and the track keeps coasting. The
# old behaviour kept the id on the wrong object and only re-elected its owner.
REFUSE_JUMPED_MATCHES = True
# A detection sitting NEXT to a track that already took its own detection this frame is a
# second bag, not a duplicate: only real overlap may suppress it then.
SUPPRESS_MATCHED_BY_IOU_ONLY = True
SUPPRESS_CONTAIN_FRAC = 0.6  # ...or being mostly inside that track's box (a partial box)
SUPPRESS_MATCHED_DIAG = 1.0  # ...or closer than this many diagonals (detector double boxes)

# A carried bag is looked for where its carrier went, not where its own jittery boxes point.
CARRIER_PREDICTION = True
CARRIER_MAX_AGE = 3.0  # seconds the carrier link stays usable after the last carry evidence

# Ownership evidence, accumulated per (bag, person) pair.
OWNER_SCORE_TAU = 30.0  # seconds; all evidence decays with this time constant
W_NEAR = 1.0  # per second standing within D_OWN of the bag (closer counts more)
W_CARRY = 5.0  # per second moving WITH the bag
D_CARRY = 1.0  # person-heights; a carrier is at least this close to the bag
CARRY_MIN_COS = 0.7  # direction agreement between the bag's and the person's velocity
CARRY_SPEED_RATIO = 2.0  # ...and speeds within this factor of each other
# After the election the owner changes only when somebody else is seen CARRYING the bag,
# for a while, with clearly more evidence. Standing next to it never steals it.
OWNER_SWITCH_CARRY_MIN = 0.5  # seconds of carry evidence the challenger needs
OWNER_SWITCH_SEC = 0.7  # ...while leading continuously for this long
OWNER_SWITCH_RATIO = 1.5  # ...with score > ratio * owner score + margin
OWNER_SWITCH_MARGIN = 1.0
OWNERLESS_CLAIM_CARRY = 0.3  # carry seconds needed to claim a bag that had no owner
# People who walked WITH the bag (a family, two people pushing a trolley) attend it too.
GROUP_ATTENDANCE = True
COMPANION_CARRY_MIN = 0.5  # carry seconds needed to count as a companion
# Distance hysteresis: "away" beyond D_AWAY, "back" only inside D_RETURN_FRAC * D_AWAY.
D_RETURN_FRAC = 0.8

# =========================
# v2026-09-17 -- fixes from the AVSSS07_MEDIUmo screenshots
# =========================
# (1) OCCLUSION HOLD. A person walking in front of a standing bag makes the detector report
# a box that shrinks from one side and slides with the walker; following those boxes carried
# the abandoned bag's track away with a passer-by (tr2 -> P9 at 5.7 s) and the bag was
# re-born without its owner. While a NON-owner overlaps a standing bag, its box is frozen:
# detections may keep it alive only if they overlap the frozen box well.
OCCLUSION_HOLD = True
OCCLUDER_OVERLAP = 0.15  # share of the bag box covered by a person box to count as occluded
OCCLUSION_HOLD_TAIL = 0.4  # seconds the hold outlasts the last overlapping frame
LUGGAGE_HELD_MATCH_IOU = 0.30

# (2) STABLE PERSON HEIGHT. A half-hidden person gives a box a third of their height, and
# every distance divided by it triples (P5 at 0.8 h instead of 0.2 h during the election).
# Each person keeps a reference height from boxes that look complete; a much shorter box is
# treated as truncated at the bottom and the feet are placed at top + reference height.
PERSON_STABLE_HEIGHT = True
PERSON_FULL_ASPECT = 2.0  # h / w at or above which a person box counts as complete
PERSON_HEIGHT_EMA = 0.3  # weight of a new complete box in the reference height
PERSON_PARTIAL_FRAC = 0.8  # a box shorter than this share of the reference is truncated

# (3) PROVISIONAL OWNER. An owner elected by standing nearby with a small margin is a guess;
# until the bag becomes unattended it goes to whoever clearly leads on that evidence.
PROVISIONAL_OWNER = True
PROVISIONAL_MARGIN = 0.25  # elections closer than this stay provisional
PROVISIONAL_LEAD = 0.10  # the leader needs 10 % more evidence than the owner...
PROVISIONAL_SEC = 2.0  # ...continuously for this long
WEAK_OWNER_RULES = ("most-evidence-among-supported", "nearest-among-supported",
                    "most-supported", "provisional-reelected")

# (4) TAKEOVER. When the owner is away or out of sight and somebody else carries the bag for
# a second, the bag is theirs -- however much evidence the absent owner collected earlier.
OWNER_TAKEOVER_CARRY = 1.0  # carry seconds needed for a takeover

# (5) PERSON REVIVE. Colour decides identity after a long gap, and under one light many
# people look alike (0.60-0.69 similarity revived the wrong man as P5 at 37.6 s).
PERSON_REVIVE_MIN_SIM = 0.70

# =========================
# (16) PERSON RE-IDENTIFICATION WITH OSNet (v2026-09-24d)
# =========================
# Every "is this the same person?" decision -- matching after a gap, taking back a lost id,
# checking that an id still belongs to a bag's owner -- used a colour histogram. On
# AVSSS07_MEDIUmo it cannot tell people apart: over ~1,900 pairs from the video, different
# people scored a median 0.87 and the same person 0.95; at a threshold keeping 90 % of true
# matches, 64 % of strangers passed too. At 12.4 s it rated a passer-by MORE like P10 than
# P10 himself (0.89 vs 0.83), gave him P10, and the real man came back as P13.
#
# OSNet (person_reid.py, next to this script) is a network trained for exactly this. People
# get its 512-d embedding instead of the histogram; bags keep the histogram. Similarity is
# then cosine similarity, which lives on a different scale, so people get their own
# thresholds below. reid_check.py measures them on your footage and writes
# REID_THRESHOLDS_FILE, which is loaded automatically; these defaults are only a start.
# If torch/the weights are unavailable the script says so and falls back to the histogram.
PERSON_REID = True
REID_ARCH = "osnet_x1_0"
REID_WEIGHTS = "osnet_x1_0_msmt17.pt"  # relative paths are next to this script
REID_THRESHOLDS_FILE = "reid_thresholds.json"

# =========================
# (17) A PIECE OF A PERSON MUST NOT CARRY AN IDENTITY (v2026-09-24e)
# =========================
# How the trolley owner went P5 -> P9 -> P13 on AVSSS07_MEDIUmo (23:02 run): at 12.9 s a man
# (P9) walks in front of him. The detector now reports the owner as fragments -- a head-and-
# shoulders box, an upper-body box. P5 takes one fragment; at 13.2 s the walker's track P9,
# left without a box, grabs the OTHER fragment of the owner (a box a quarter of its height,
# 60 px away), and at 13.7 s, when the owner is whole again, P9 is sitting on him. P5 drifts
# off onto a stranger (14.2 s), the walker is re-born as P13, and at 15.8 s P13 jumps onto
# the owner too. Every step went through a box that was only part of a person.
# Three rules close it:
#   shape   -- a track only takes a box of about its own height (or one overlapping it well);
#              a head box is not the whole walker, and a walker is not a head
#   contain -- a box lying inside a person seen just now is a piece of THAT person: no other
#              track may take it, and it may not start a new id
#   edges   -- a box cut off by the image border, or with a shape no whole person has,
#              may keep an id alive but never take over a lost id (reclaim / revive)
PERSON_SHAPE_GATE = True
PERSON_MAX_H_RATIO = 1.8  # det height vs track height, either way
PERSON_SHAPE_EXEMPT_IOU = 0.50  # ...unless the boxes overlap this much
PERSON_FRAGMENT_CONTAIN = 0.70  # share of a box inside a known person that makes it theirs
PERSON_FRAGMENT_MAX_AGE = 1.0  # seconds; "known" = seen this recently
PERSON_BORDER_PX = 3  # a box this close to the left/right/bottom edge is cut off
PERSON_WHOLE_ASPECT = (1.3, 5.0)  # h / w of a whole person; outside it is a fragment

# =========================
# (18) WHEN THE OWNER'S ID GOES ELSEWHERE, THE BAG FOLLOWS THE PERSON (v2026-09-25h)
# =========================
# 00:28 run: the owner kept P5 until he walked along the back of the platform at 19.5 s. A
# woman passed in front of him and for six frames the detector drew one box around both;
# P5 went with that box, and when the owner came out whole at 19.7 s he got a new id, P15.
# The trolley stayed with P5 (who by then was on somebody else), was later handed to a wrong
# man (P11) by appearance, and only reached P15 at 22.8 s.
# The moment is easy to recognise, even though the tracker cannot avoid it: every bag keeps
# the owner's last WHOLE-body box (normal human shape, full height, head where it was), and
# if within OWNER_SPLIT_SEC a NEW id appears on that exact spot while the owner's own id is
# somewhere else, the new id is the owner. Checked on the logged detections of the 22:26,
# 23:02 and 00:28 runs: it fires twice, both at 19.79 s, both correct (the trolley and the
# other bag P5 owned), and nowhere else.
OWNER_SPLIT = True
OWNER_SPLIT_SEC = 1.0  # the owner's last whole box stays usable this long
OWNER_SPLIT_IOU = 0.40  # the new id must overlap it at least this much...
OWNER_SPLIT_LEAD = 0.10  # ...and this much more than the owner's own id does now
WHOLE_TOP_TOL = 0.15  # head may move this x height (+ WHOLE_TOP_SPEED per second) between whole boxes
WHOLE_TOP_SPEED = 0.5
WHOLE_H_RANGE = (0.85, 1.20)  # a whole box is this share of the person's full height
REID_SIM = {
    "relink": 0.30,  # a stale track is only re-taken by a box at least this similar
    "revive": 0.60,  # taking back a lost id (coasting or retired)
    "verify": 0.40,  # the id holding a bag's owner role must still look like the owner
    "resolve": 0.70,  # handing the owner role to another id by appearance
    "rebind": 0.70,  # re-binding an owner who was re-created under a new id
}

# =========================
# (10) PEOPLE JUMP TOO -- the same failure the luggage gate had
# =========================
# The person gate allowed 3 box diagonals, capped at PERSON_MATCH_DIST_THR = 220 px. On
# AVSSS07_MEDIUmo frame 427 that let P5 move 134 px in ONE frame onto a person who had just
# appeared, while the man it had been following -- still detected at conf 0.88 -- was left
# to be re-born under another id. The runner-up for that detection was P8, so the two ids
# swapped. A person walking at 30 fps moves a few px per frame; a jump of one box diagonal
# is already generous, and anything past it is a different human.
PERSON_TIGHT_GATING = True
# Measured on the AVSSS07_MEDIUmo detections: implausible one-frame moves (>60 px by a
# track matched on the previous frame too) fall 35 -> 14 -> 7 -> 4 as the gate tightens
# through 0.75 / 0.50 / 0.35 diagonals, while person ids rise 14 -> 15 -> 17 -> 21 because
# a refused match makes a new id instead of a wrong one. 0.50 is where that trade sits
# before the id count starts to run away.
PERSON_GATE_FRESH_DIAG = 0.50  # age <= PERSON_FRESH_AGE: half a box diagonal
PERSON_GATE_STALE_DIAG = 1.50  # older tracks, still widened by GATE_GROWTH
PERSON_FRESH_AGE = 0.5
PERSON_GATE_IOU = 0.35  # overlapping this much is the same person whatever the distance
PERSON_JUMP_DIAG = 2.0  # a quiet track landing this far away is refused outright

# =========================
# (11) CLAIM WHAT IS CLEARLY YOURS, BEFORE ANYONE ELSE BIDS
# =========================
# The remaining id swaps needed no big jump at all. On AVSSS07_MEDIUmo the man standing at
# the back was detected as TWO boxes for a moment -- his full body, and an upper-body
# fragment once somebody crossed in front of his legs. P5 followed the fragment (30 px), a
# track arriving from 175 px away took the full box (23 px), and from then on P5 was free
# to drift onto somebody else: P5 -> P11 -> P15 -> P1. Every single step was inside the
# gate, so no gate can catch it.
#
# The cure is to settle the obvious pairs first: when a track and a detection overlap
# strongly AND are each other's best overlap, nothing else may bid for either. Only what is
# left over goes to the cost-based assignment, where a newcomer can win on a small margin.
CLAIM_LOCK = True
PERSON_LOCK_IOU = 0.60
LUGGAGE_LOCK_IOU = 0.60

# =========================
# (12) A TRACK THAT LOSES ITS OBJECT MUST NOT GO SHOPPING
# =========================
# Two people at the back of AVSSS07_MEDIUmo overlap; the detector merges them into one box;
# the track with the better overlap wins it, which is fair. The harm is what the loser does
# NEXT: with nothing to match, it drifts a few tens of px per frame onto a THIRD person and
# carries its id (and any bag that named it owner) there -- P5 -> P11 -> P15 -> P1.
#
# So a track whose own detection was just taken by another track is "eclipsed": it keeps
# coasting where it is, and for the next ECLIPSE_SEC it may only re-attach to a box that
# really overlaps it, or one that clearly looks like it. If the crowd clears it gets its
# person back; if not it dies quietly, which costs one id -- not a chain of them.
# OFF by default: measured on AVSSS07_MEDIUmo it did not reduce drift at all (6 tracks
# moving >120 px in a second, before and after) and cost 3 extra person ids (17 -> 20),
# because a frozen track dies and is re-born instead of recovering. The chain it was meant
# to stop comes from the DETECTOR merging two overlapping people into one box, and no
# association rule can undo a merged box -- that needs appearance re-identification.
# Left here, with its test, for scenes where people overlap less than on a platform.
ECLIPSE_HOLD = False
ECLIPSE_BY_IOU = 0.40  # a matched detection overlapping an unmatched track this much...
ECLIPSE_SEC = 2.0  # ...eclipses it for this long
ECLIPSE_MATCH_IOU = 0.50  # while eclipsed: re-attach only on this overlap...
ECLIPSE_MATCH_SIM = 0.80  # ...or this colour similarity

# =========================
# (13) THE OWNER IS A PERSON, NOT A TRACK NUMBER
# =========================
# Every id fix so far fought the tracker. This stops depending on it. A bag remembers its
# owner as a DESCRIPTION -- colour signature, where they were, how tall they are -- and
# every frame it asks two questions instead of one:
#
#     does the id still look like my owner?      (verification)
#     if not, who here does?                     (resolution)
#
# That covers the failure the tracker cannot: on AVSSS07_MEDIUmo two people at the back
# merge into one detection, P5 is left holding a fragment and drifts onto a THIRD person,
# and the bag keeps pointing at P5 -- silently, because the id still exists. With a
# description the bag notices that "P5" stopped looking like its owner and moves its
# ownership to whoever now matches, whatever number the tracker gave them.
OWNER_BY_DESCRIPTION = True
OWNER_VERIFY_MIN_SIM = 0.55  # the id holding the role must still look like the owner...
OWNER_VERIFY_JUMP_H = 2.0  # ...and must not teleport: person-heights since last verified
OWNER_VERIFY_SEC = 0.3  # a mismatch must persist this long before the role is re-opened
OWNER_RESOLVE_MIN_SIM = 0.75  # to TAKE the role by description: a strong colour match...
OWNER_RESOLVE_SIM_MARGIN = 0.05  # ...clearly ahead of the runner-up
OWNER_RESOLVE_MAX_H = 1.5  # within this many heights of where the owner was last seen...
OWNER_RESOLVE_GROWTH_H = 1.5  # ...plus this much per second since then
OWNER_RESOLVE_MAX_GAP = 5.0  # after this long unverified, the trail is cold
# Without a usable colour signature (dark or tiny boxes) position alone can decide -- but
# ONLY when the id that holds the role turned out to be somebody else, never when the owner
# simply vanished. An owner who walked away leaves a gap that a bystander standing where
# they used to be would fill, and that silences the very alarm this script exists for.
OWNER_RESOLVE_DIST_MARGIN = 1.5  # the nearest candidate must be this much nearer than the next
OWNER_RESOLVE_POS_MAX_H = 0.75  # ...and this close to where the owner stood, no growth
# (19) v2026-09-25i. When the owner's id merely DISAPPEARS, the only person who can be "the
# owner under a number this bag has never heard of" is a number created around the time he
# vanished. 20:32 run: the trolley was correctly handed to P15 at 19.7 s; P15 walked behind
# the sign at the right end for half a second and the trolley went to P8 -- a man who had
# stood at that end since 2 s -- on appearance alone (0.741). The same bystander hand-over
# happened 8-9 times per run (new owners born 5-19 s before the owner vanished). A bystander
# can still take a bag the ordinary way: by being seen carrying it.
OWNER_RESOLVE_NEWBORN_SLACK = 1.0  # seconds before the owner was last seen

# (20) v2026-09-25j. ONE FRAME OF "THE OWNER IS BACK" MUST NOT UNDO A DEPARTURE.
# 20:43 run: the trolley's owner (P15) was watched walking away to 3.5 person-heights and the
# clock started at 19.9 s. At 22.8 s, after he vanished at the far end, his id jumped onto a
# woman walking past the bag (1.5x his height, 1.9 heights from the trolley). One frame of
# that reset the clock, her 1.1 s nearby plus the 4 s occlusion grace kept it reset, and the
# alarm came at 39.2 s -- flagged for review, with "owner seen 0 %", because the departure
# that really happened had been thrown away. A real return is a person who comes back and
# STAYS by the bag. So while the clock runs, a return must hold for OWNER_RETURN_CONFIRM_SEC
# (at the stricter return distance, without the occlusion grace) before it counts; until then
# the clock is only paused. Replayed on that run's log: the trolley alarms at 30.5 s with the
# departure observed, and no other bag's clock reaches 10 s.
OWNER_RETURN_CONFIRM_SEC = 1.5

# =========================
# v2026-09-22 -- capped identity memory, graded alarms
# =========================
# (6) The revive radius grew by half the base radius for every second out of sight, with no
# ceiling: after 50 s it covered the whole frame and six bags came back 70-425 px from where
# they vanished (AVSSS07_MEDIUmo, 23:26 run), carrying their old id and owner onto whatever
# stood there. A bag nobody carried away does not move while nobody can see it.
LUGGAGE_REVIVE_MAX_SCALE = 2.0  # multiples of the base revive radius, whatever the gap
PERSON_REVIVE_MAX_SCALE = 4.0  # people do move while out of sight -- but not that far

# (7) ALARM GRADING. Every alarm already carries its evidence; this folds it into one number
# so alarms can be ranked and the doubtful ones flagged, instead of being judged by eye.
# Nothing here changes WHEN an alarm fires -- only how it is reported.
W_CONF_STILL = 0.30  # the bag did not drift from where its owner left it
W_CONF_SEEN = 0.25  # ...and was actually visible while the clock ran
W_CONF_DEPART = 0.20  # the owner was watched leaving, rather than merely lost
W_CONF_OWNER = 0.25  # ...and the owner was not a coin toss between neighbours
ALARM_REVIEW_CONFIDENCE = 0.60  # below this an alarm is flagged for a human look
# (8) A bag already in place when the clip starts, that nobody was ever seen carrying, is
# furniture as far as the evidence goes -- there is no departure to observe.
SCENE_OBJECT_SEC = 1.0
SCENE_OBJECT_MIN_CARRY = 0.2  # seconds of carry evidence that make it a real bag
SCENE_OBJECT_PENALTY = 0.15
# ...but only if it never went anywhere: a bag carried into view in the first second and put
# down 200 px away (ABODA video1) is not furniture, it is exactly the event being looked for.
SCENE_OBJECT_MAX_DISP_DIAG = 1.5

# (9) A BAG IN SOMEBODY'S HANDS. Co-motion alone missed a backpack carried across a room --
# at that scale a walking man moves it 0.5 box diagonals a second, under BAG_MOVING_SPEED,
# so its carrier was credited 0.1 s over the whole clip. A box sitting INSIDE a person's box
# is being held or worn, whether it moves or not, and that is the plainest evidence there is.
# OFF by default: on AVSSS07_MEDIUmo a bystander's box swallows the bag at the back of the
# platform in the first second, before the bag counts as standing, and he collects evidence
# for a bag he never touched (the alarm moved to the wrong man, 5 s early). The absolute
# speed floor below is what actually recovered the carried backpack, so this stays a switch
# for scenes where bags are held still in shot for a long time.
HELD_BY_PERSON = False
HELD_INSIDE_FRAC = 0.70  # share of the bag box inside the person box
W_HELD = 3.0  # per second, against W_CARRY = 5.0 for walking together
# A small box crossing the room is carried even when that is under BAG_MOVING_SPEED: a
# backpack 54 px across moved 28 px/s (0.5 diagonals/s) through ABODA video1, and its
# carrier was credited 0.1 s of evidence over the whole clip.
BAG_MOVING_PX = 12.0

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
ALARM_LATCH = True  # once raised, an alarm stays raised for the life of the track

# =========================
# ZONE DETECTION (optional)
# =========================
ENABLE_ZONES = False
RESTRICTED_ZONES = []  # e.g. [[(100, 100), (300, 100), (300, 300), (100, 300)]]

# =========================
# ANNOTATION
# =========================
SHOW_TRAJECTORIES = False  # motion traces clutter a busy scene; the log keeps the history
TRAJECTORY_OWNERS_ONLY = True  # if traces are on, only trace people who own a bag
# On-screen tags are kept to <tag><id> <conf>; the sidebar legend spells them out.
SHORT_NAMES = {"backpack": "bck", "bag": "bg", "trolley": "tr",
               "handbag": "bg", "suitcase": "tr"}
# The info panel is rendered in its own column NEXT TO the video, never on top of it.
SIDEBAR_WIDTH = 540  # px; 0 disables the panel entirely
SIDEBAR_SIDE = "right"  # "right" or "left"
SIDEBAR_BG = (26, 26, 26)
SIDEBAR_MAX_TRACK_ROWS = 6  # per-bag rows listed under "luggage / owner"
SIDEBAR_MIN_HEIGHT = 860  # canvas is padded to this so the panel is never clipped
FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT = cv2.FONT_HERSHEY_DUPLEX  # sidebar text (a little rounder / easier to read)
# Box labels (P3 0.91 / bg2 0.77 12s). Big enough to stay readable when four outputs
# are shrunk into a 2x2 comparison grid.
LABEL_SCALE = 0.8
LABEL_THICK = 2
LABEL_ALPHA = 0.85  # plate opacity (the ALARM plate is always fully opaque)

PENDING, OWNED, UNATTENDED, ALARM = "PENDING", "OWNED", "UNATTENDED", "ALARM"

COLOR_PERSON = (255, 180, 0)
COLOR_OWNER = (0, 255, 255)
COLOR_PENDING = (200, 200, 200)
COLOR_OWNED = (80, 200, 80)
COLOR_UNATTENDED = (0, 165, 255)
COLOR_ALARM = (0, 0, 255)
ZONE_ALERT_COLOR = (0, 140, 255)
STATE_COLOR = {PENDING: COLOR_PENDING, OWNED: COLOR_OWNED,
               UNATTENDED: COLOR_UNATTENDED, ALARM: COLOR_ALARM}


# -----------------------------
# Analysis log
# -----------------------------
def _json_safe(o):
    if isinstance(o, np.ndarray):
        return [round(float(v), 1) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


class EventLog:
    """One JSON record per line: every detection, track, association and state change."""

    def __init__(self, path=None):
        self.f = open(path, "w", encoding="utf-8") if path else None
        self.t, self.frame, self.n = 0.0, 0, 0
        self.counts = {}

    def mark(self, t, frame):
        self.t, self.frame = t, frame

    def event(self, kind, **fields):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if "role" in fields:  # so the end-of-run summary separates people from bags
            key = "%s[%s]" % (kind, fields["role"])
            self.counts[key] = self.counts.get(key, 0) + 1
        if self.f is None:
            return
        rec = {"t": round(float(self.t), 3), "frame": int(self.frame), "event": kind}
        rec.update(fields)
        self.f.write(json.dumps(rec, default=_json_safe) + "\n")
        self.n += 1

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


LOG = EventLog()  # replaced in run()


def _slug(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return "".join(c if c.isalnum() or c in "-." else "_" for c in stem)[:40]


def output_video_path(video_in, profile=None):
    """<input dir>/<input stem>_<profile><OUT_VIDEO_SUFFIX><OUT_VIDEO_EXT> -- beside the source."""
    folder = os.path.dirname(os.path.abspath(str(video_in)))
    stem = os.path.splitext(os.path.basename(str(video_in)))[0]
    tag = f"_{profile or MODEL_PROFILE}"
    return os.path.join(folder, f"{stem}{tag}{OUT_VIDEO_SUFFIX}{OUT_VIDEO_EXT}")


def make_run_dir(video, profile=None):
    """OUT_DIR/<video>__<model profile>__<timestamp><tag>, created on the spot."""
    name = (f"{_slug(video)}__{profile or MODEL_PROFILE}"
            f"__{time.strftime('%Y%m%d-%H%M%S')}{RUN_TAG}")
    path = os.path.join(OUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


# Custom models may spell their classes differently ("Backpack", "suitcase"); the pipeline
# and the on-screen tags use backpack / bag / trolley.
CANON_LUGGAGE_NAMES = {"handbag": "bag", "suitcase": "trolley"}


def canon_luggage_name(name):
    n = str(name).strip().lower()
    return CANON_LUGGAGE_NAMES.get(n, n)


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
    if tuple(bg) == tuple(COLOR_ALARM):
        return (0, 0, 0)  # alarm: black on red
    b, g, r = bg
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


def draw_label(img, text, x, y, color, scale=LABEL_SCALE, thickness=LABEL_THICK, bg=True,
               alpha=LABEL_ALPHA):
    """Text on a filled translucent plate; (x, y) is the text baseline. Returns its box."""
    x, y = int(x), int(y)
    if tuple(color) == tuple(COLOR_ALARM):
        alpha = 1.0
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    pad = max(4, int(round(6 * scale)))
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


def short_name(name):
    """Compact tag for a class name, e.g. backpack -> bck."""
    name = str(name).lower()
    return SHORT_NAMES.get(name, name[:3])


def scale_gate(bbox, thr_px, diag_mult, min_px):
    """A search radius that means the same thing near the camera and at the far end."""
    diag = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    return min(float(thr_px), max(float(min_px), diag_mult * diag))


def predict_bbox(st, dt_pred):
    """Constant-velocity prediction in center space."""
    x1, y1, x2, y2 = st["bbox"]
    cx, cy = center_xyxy(st["bbox"])
    w = x2 - x1
    h = y2 - y1
    dt_pred = min(dt_pred, PREDICT_MAX_SEC)  # a long gap must not fling the box away
    cx_p = cx + st.get("vx", 0.0) * dt_pred
    cy_p = cy + st.get("vy", 0.0) * dt_pred
    return np.array([cx_p - w / 2, cy_p - h / 2, cx_p + w / 2, cy_p + h / 2], dtype=float)


def bbox_in_zone(bbox, zones):
    """True if the bbox ground-contact point falls inside any restricted zone."""
    pt = tuple(float(v) for v in bottom_center(bbox))
    return any(cv2.pointPolygonTest(np.asarray(z, np.int32), pt, False) >= 0 for z in zones)


def box_diag(bb):
    return math.hypot(bb[2] - bb[0], bb[3] - bb[1])


def overlap_frac(a, b):
    """Share of box a that lies inside box b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return iw * ih / area


def carrier_prediction(st, now_t):
    """The box a CARRIED bag should have now, from its carrier's position (or None)."""
    if not CARRIER_PREDICTION or st.get("carrier_pred_t") != now_t:
        return None
    return st.get("carrier_pred")


def track_predictions(st, age, now_t):
    """Candidate boxes for a track: its own motion model, and its carrier if it has one."""
    out = [predict_bbox(st, age)]
    cp = carrier_prediction(st, now_t)
    if cp is not None:
        out.append(np.asarray(cp, dtype=float))
    return out


def closest_prediction(dbb, st, age, now_t):
    """(iou, centre distance, box) against the best-fitting prediction of a track."""
    dcx, dcy = center_xyxy(dbb)
    best = None
    for pbb in track_predictions(st, age, now_t):
        iou = iou_xyxy(dbb, pbb)
        pcx, pcy = center_xyxy(pbb)
        dist = math.hypot(dcx - pcx, dcy - pcy)
        key = dist - 100.0 * iou
        if best is None or key < best[0]:
            best = (key, iou, dist, pbb)
    return best[1], best[2], best[3]


def luggage_gate_ok(st, iou, dist, pbb, age, now_t, fresh_diag=None, stale_diag=None,
                    dbb=None):
    """Tight association gate for luggage (see LUGGAGE_TIGHT_GATING)."""
    diag = max(MATCH_DIST_MIN_PX / 2.0, box_diag(pbb))
    if is_held(st, now_t):  # somebody is in front of it: only a box ON it is it
        return iou >= LUGGAGE_HELD_MATCH_IOU
    if dbb is not None and iou < LUGGAGE_AREA_EXEMPT_IOU:
        a_d = max(1.0, (dbb[2] - dbb[0]) * (dbb[3] - dbb[1]))
        a_t = max(1.0, (pbb[2] - pbb[0]) * (pbb[3] - pbb[1]))
        if max(a_d, a_t) / min(a_d, a_t) > LUGGAGE_MAX_AREA_RATIO:
            return False
    if is_static(st, now_t) and carrier_prediction(st, now_t) is None:
        # a standing bag does not wander: only a box on top of it can be it
        return iou >= LUGGAGE_STATIC_MATCH_IOU or dist <= LUGGAGE_STATIC_DIST_DIAG * diag
    if iou >= LUGGAGE_GATE_IOU:
        return True
    if age <= LUGGAGE_FRESH_AGE:
        gate = max(MATCH_DIST_MIN_PX / 2.0,
                   (LUGGAGE_GATE_FRESH_DIAG if fresh_diag is None else fresh_diag) * diag)
    else:
        gate = (max(float(MATCH_DIST_MIN_PX),
                    (LUGGAGE_GATE_STALE_DIAG if stale_diag is None else stale_diag) * diag)
                * min(1.0 + GATE_GROWTH * max(0.0, age), GATE_MAX_SCALE))
    return dist <= gate


def is_eclipsed(st, now_t):
    """Did another track just take the detection this one was sitting on? (note 12)"""
    return ECLIPSE_HOLD and now_t - st.get("eclipsed_t", -1e9) <= ECLIPSE_SEC


def mark_eclipsed(dets, tracks, matches, unmatched_track_ids, now_t, role=""):
    """Flag the tracks whose object was taken by somebody else this frame."""
    if not ECLIPSE_HOLD:
        return
    taken = [dets[di]["bbox"] for di, _ in matches]
    for sid in unmatched_track_ids:
        st = tracks.get(sid)
        if st is None or now_t - st["last_seen_t"] > 0.5:
            continue  # it was already coasting: this is not a theft, just an absence
        if any(iou_xyxy(st["bbox"], b) >= ECLIPSE_BY_IOU for b in taken):
            if not is_eclipsed(st, now_t):
                LOG.event("eclipsed", role=role, tid=sid,
                          hold_s=ECLIPSE_SEC)
            st["eclipsed_t"] = now_t


def person_shape_ok(dbb, pbb, iou):
    """(17) Is this box about the height of the track it is offered to?"""
    if not PERSON_SHAPE_GATE or dbb is None or iou >= PERSON_SHAPE_EXEMPT_IOU:
        return True
    hd, ht = max(1.0, dbb[3] - dbb[1]), max(1.0, pbb[3] - pbb[1])
    return max(hd, ht) / min(hd, ht) <= PERSON_MAX_H_RATIO


def owner_of_fragment(dbb, tracks, now_t, exclude=None):
    """(17) The recently seen person a box lies inside, or None."""
    if not PERSON_SHAPE_GATE:
        return None
    best, best_f = None, PERSON_FRAGMENT_CONTAIN
    for sid, st in tracks.items():
        if sid == exclude or now_t - st["last_seen_t"] > PERSON_FRAGMENT_MAX_AGE:
            continue
        f = overlap_frac(dbb, st["bbox"])
        if f >= best_f:
            best, best_f = sid, f
    return best


FRAME_WH = (0, 0)  # set in run(); used to recognise boxes cut off by the image border


def is_fragment_box(bb):
    """(17) Cut off by the image border, or a shape no whole person has."""
    if not PERSON_SHAPE_GATE:
        return False
    x1, y1, x2, y2 = (float(v) for v in bb)
    fw, fh = FRAME_WH
    if fw and (x1 <= PERSON_BORDER_PX or x2 >= fw - PERSON_BORDER_PX
               or y2 >= fh - PERSON_BORDER_PX):
        return True
    a = (y2 - y1) / max(1.0, x2 - x1)
    return not (PERSON_WHOLE_ASPECT[0] <= a <= PERSON_WHOLE_ASPECT[1])


def person_gate_ok(st, iou, dist, pbb, age, dbb=None):
    """Tight association gate for people (see PERSON_TIGHT_GATING)."""
    if not person_shape_ok(dbb, pbb, iou):
        return False
    if iou >= PERSON_GATE_IOU:
        return True
    diag = max(MATCH_DIST_MIN_PX, box_diag(pbb))
    if age <= PERSON_FRESH_AGE:
        gate = PERSON_GATE_FRESH_DIAG * diag
    else:
        gate = (PERSON_GATE_STALE_DIAG * diag
                * min(1.0 + GATE_GROWTH * max(0.0, age), GATE_MAX_SCALE))
    return dist <= min(gate, float(PERSON_MATCH_DIST_THR))


def is_static(st, now_t):
    """True once a bag has stayed put for BAG_STATIC_SEC."""
    s = st.get("static_since")
    return s is not None and now_t - s >= BAG_STATIC_SEC


def velocity(traj, now_t, window=BAG_MOTION_WINDOW):
    """(vx, vy) in px/s over the recent trajectory points, or None with too little history."""
    if not traj:
        return None
    pts = [p for p in traj if now_t - p[2] <= window + 1e-6]
    if len(pts) < 2 or pts[-1][2] - pts[0][2] < 0.5 * window:
        return None
    span = pts[-1][2] - pts[0][2]
    return ((pts[-1][0] - pts[0][0]) / span, (pts[-1][1] - pts[0][1]) / span)


def update_bag_motion(st, now_t):
    """Standing / moving status of a bag, from the positions it was actually matched at."""
    if is_held(st, now_t):
        st["moving"] = False
        return
    bx, by = bottom_center(st["bbox"])
    diag = max(8.0, box_diag(st["bbox"]))
    anchor = st.get("static_anchor")
    if anchor is None or math.hypot(bx - anchor[0], by - anchor[1]) > BAG_STATIC_TOL * diag:
        if anchor is not None and is_static(st, now_t):
            LOG.event("bag_moved", lid=st.get("lid"),
                      stood_s=round(now_t - st["static_since"], 2))
        st["static_anchor"] = (bx, by)
        st["static_since"] = now_t
    v = velocity(st.get("trajectory"), now_t)
    st["bag_vel"] = v
    speed = 0.0 if v is None else math.hypot(*v)
    st["moving"] = v is not None and (speed / diag > BAG_MOVING_SPEED or speed > BAG_MOVING_PX)
    home = st.get("home_xy")
    if home is None:
        st["home_xy"] = home = (bx, by)
    st["disp_max"] = max(st.get("disp_max", 0.0), math.hypot(bx - home[0], by - home[1]))
    st["disp_diag"] = st["disp_max"] / diag


def is_held(st, now_t):
    return OCCLUSION_HOLD and st.get("held_t") == now_t


def update_occlusion_hold(luggage_tracks, person_tracks, now_t):
    """Freeze standing bags that a person other than their owner is walking in front of."""
    if not OCCLUSION_HOLD:
        return
    live = [(pid, pt["bbox"]) for pid, pt in person_tracks.items()
            if now_t - pt["last_seen_t"] <= 0.1]
    for lid, st in luggage_tracks.items():
        if not is_static(st, now_t):
            continue
        over = {pid for pid, pb in live if overlap_frac(st["bbox"], pb) >= OCCLUDER_OVERLAP}
        if over and st.get("owner_pid") not in over:
            if st.get("held_until", -1e9) < now_t:
                LOG.event("bag_held", lid=lid, by=sorted(over), owner=st.get("owner_pid"))
            st["held_until"] = now_t + OCCLUSION_HOLD_TAIL
        if now_t <= st.get("held_until", -1e9):
            st["held_t"] = now_t


def person_reference(st, frame_h=None):
    """Update a person track's reference height from complete-looking boxes."""
    if not PERSON_STABLE_HEIGHT:
        return
    x1, y1, x2, y2 = st["bbox"]
    h, w = y2 - y1, max(1.0, x2 - x1)
    cut = frame_h is not None and y2 >= frame_h - 2  # leaving through the bottom edge
    if h / w >= PERSON_FULL_ASPECT and not cut:
        ref = st.get("h_ref")
        st["h_ref"] = h if ref is None else (1 - PERSON_HEIGHT_EMA) * ref + PERSON_HEIGHT_EMA * h


def update_person_whole(st, now_t):
    """(18) Remember the last raw box that showed this person whole, and their full height."""
    b = st.get("det_bbox")
    if b is None or st.get("det_t") != now_t or is_fragment_box(b):
        return
    h = float(b[3] - b[1])
    w = st.get("whole_bb")
    if w is None:
        st["whole_bb"], st["whole_t"], st["whole_pk"] = np.array(b, float), now_t, h
        return
    dt = now_t - st["whole_t"]
    pk = st["whole_pk"]
    pk_now = pk * 0.9 ** dt
    if abs(b[1] - w[1]) > (WHOLE_TOP_TOL + WHOLE_TOP_SPEED * dt) * pk:
        return  # the head jumped: a merged box or somebody else
    if WHOLE_H_RANGE[0] * pk_now <= h <= WHOLE_H_RANGE[1] * pk:
        st["whole_bb"], st["whole_t"], st["whole_pk"] = np.array(b, float), now_t, max(h, pk_now)


def person_ground(p):
    """(foot x, foot y, height) of a person, repairing boxes truncated at the bottom."""
    x1, y1, x2, y2 = p["bbox"]
    h = y2 - y1
    ref = p.get("h_ref") if PERSON_STABLE_HEIGHT else None
    if ref is not None and h < PERSON_PARTIAL_FRAC * ref:
        return (x1 + x2) / 2.0, y1 + ref, ref
    return (x1 + x2) / 2.0, y2, h


def match_jumped(st, det, now_t, role="luggage"):
    """A quiet track matched to a box far from where it was: a different object."""
    gap = now_t - st["last_seen_t"]
    if gap <= OWNERSHIP_RESET_MIN_AGE:
        return False, gap, 0.0, 0.0
    if role == "person":
        ocx, ocy = center_xyxy(st["bbox"])
        ncx, ncy = center_xyxy(det["bbox"])
        jump = math.hypot(ncx - ocx, ncy - ocy)
        diag = box_diag(st["bbox"])
        return jump > PERSON_JUMP_DIAG * max(8.0, diag), gap, jump, diag
    ocx, ocy = center_xyxy(st["bbox"])
    ncx, ncy = center_xyxy(det["bbox"])
    jump = math.hypot(ncx - ocx, ncy - ocy)
    diag = box_diag(st["bbox"])
    if carrier_prediction(st, now_t) is not None:  # a carried bag moves with its carrier
        return False, gap, jump, diag
    return jump > OWNERSHIP_RESET_JUMP * max(8.0, diag), gap, jump, diag


# -----------------------------
# Appearance signature (re-identification)
# -----------------------------
def _sample_patch(frame, bbox, band=None, shrink=APP_SHRINK):
    """The pixels that belong to the object: the box, shrunk, optionally a vertical band."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return None
    x1 += bw * shrink
    x2 -= bw * shrink
    if band is None:
        y1 += bh * shrink
        y2 -= bh * shrink
    else:
        y1, y2 = y1 + bh * band[0], y1 + bh * band[1]
    xi1, yi1 = int(max(0, min(fw - 1, x1))), int(max(0, min(fh - 1, y1)))
    xi2, yi2 = int(max(0, min(fw, x2))), int(max(0, min(fh, y2)))
    if xi2 - xi1 < 3 or yi2 - yi1 < 3:
        return None
    return frame[yi1:yi2, xi1:xi2]


def appearance_of(frame, bbox, role="luggage"):
    """L1-normalised hue/saturation histogram of the object. None when it is unusable."""
    if not APPEARANCE_ENABLED:
        return None
    patch = _sample_patch(frame, bbox, APP_PERSON_BAND if role == "person" else None)
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # dark and washed-out pixels have meaningless hue, so they do not get a vote
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0, 1], mask, list(APP_BINS), [0, 180, 0, 256])
    total = float(hist.sum())
    if total < 25.0:  # almost everything was masked out -- no usable colour
        return None
    return (hist.flatten() / total).astype(np.float32)


REID = None  # PersonReID instance when OSNet is active (set in run())


def appearance_sim(a, b):
    """1.0 = identical. Cosine similarity for OSNet embeddings (clipped at 0), the
    Bhattacharyya coefficient for colour histograms; None if either is missing or they are
    of different kinds (an embedding cannot be compared with a histogram)."""
    if a is None or b is None or a.shape != b.shape:
        return None
    if REID is not None and a.shape[0] == REID.dim:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return None
        return max(0.0, float(np.dot(a, b)) / (na * nb))
    return float(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)).sum())


def sim_thr(key, colour_value, role="person"):
    """The threshold for a person-identity decision on the scale currently in use."""
    return REID_SIM[key] if (REID is not None and role == "person") else colour_value


def update_appearance(st, app):
    """Slow EMA: the template must follow lighting, not the frame the object was occluded."""
    if app is None:
        return
    old = st.get("app")
    st["app"] = app if old is None else (APP_EMA * old + (1.0 - APP_EMA) * app)


def _r3(x):
    return None if x is None else round(float(x), 3)


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
        use_stable_class_for_penalty: bool = False,
        app_weight: float = 0.0,
        role: str = ""
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

            iou, dist, pbb = closest_prediction(dbb, st, age, now_t)

            if role == "luggage" and LUGGAGE_TIGHT_GATING:
                if not luggage_gate_ok(st, iou, dist, pbb, age, now_t, dbb=dbb):
                    continue
            elif role == "person" and PERSON_TIGHT_GATING:
                if not person_gate_ok(st, iou, dist, pbb, age, dbb=dbb):
                    continue
            else:
                # geometry gating -- the radius widens with the time the track has been
                # missing, because an occluded object keeps moving while it is invisible
                gate = (scale_gate(pbb, match_dist_thr, MATCH_DIST_DIAG, MATCH_DIST_MIN_PX)
                        * min(1.0 + GATE_GROWTH * max(0.0, age), GATE_MAX_SCALE))
                if not ((iou >= match_iou_thr) or (dist <= gate)):
                    continue

            sim = appearance_sim(d.get("app"), st.get("app")) if app_weight > 0.0 else None
            # an eclipsed track may only take back something that is plainly its own
            if is_eclipsed(st, now_t) and iou < ECLIPSE_MATCH_IOU and not (
                    sim is not None and sim >= ECLIPSE_MATCH_SIM):
                continue
            # a track that has been out of sight may only be re-taken by a detection that
            # also LOOKS like it; without this the widened gate steals neighbouring ids
            if (sim is not None and age > 0.5 and iou < match_iou_thr
                    and sim < sim_thr("relink", APP_MIN_SIM_RELINK, role)):
                continue

            penalty = 0.0
            if class_mismatch_penalty > 0.0:
                tcls = int(st.get("cls_stable", st.get("cls", 0))) if use_stable_class_for_penalty else int(
                    st.get("cls", 0))
                if dcls != tcls:
                    penalty = class_mismatch_penalty

            cost[di, tj] = (-W_IOU * iou) + (W_DIST * dist) + penalty
            if sim is not None:
                cost[di, tj] += app_weight * (1.0 - sim)

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

        # Record the matches that deserve suspicion -- a stale track winning, or a box
        # jumping -- with the runner-up, so a swap can be read off the log instead of
        # reconstructed from screenshots.
        if role:
            st = tracks[sid]
            age = now_t - st["last_seen_t"]
            dcx, dcy = center_xyxy(dets[di]["bbox"])
            tcx, tcy = center_xyxy(st["bbox"])
            jump = math.hypot(dcx - tcx, dcy - tcy)
            if LOG_ALL_MATCHES or age > MATCH_SUSPECT_AGE or jump > MATCH_SUSPECT_JUMP:
                alt = sorted((float(cost[r, j]), track_ids[j])
                             for j in range(len(track_ids))
                             if j != int(c) and cost[r, j] < BIG)
                LOG.event("match_suspect", role=role, tid=sid, age_s=round(age, 2),
                          jump_px=round(jump, 1), cost=round(float(cost[r, c]), 3),
                          runner_up=None if not alt else
                          {"tid": alt[0][1], "cost": round(alt[0][0], 3)})

    unmatched_det = set(range(len(dets))) - used_d
    unmatched_tracks = set(track_ids) - used_t
    return matches, unmatched_det, unmatched_tracks


def log_assoc(role, prev, dets_hi, m_hi, dets_lo, m_lo, unmatched_det, tracks, now_t):
    """The association itself -- the one thing a detection log and a track log cannot show.

    Detections are anonymous in the log and track boxes are smoothed, so without this the
    only way to tell which detection a track took is to invert the smoothing by hand. One
    compact record per role per frame makes any frame replayable: every detection, every
    assignment, which pass made it, where the track came from and how far it jumped.
    """
    if not LOG_ASSOC:
        return

    def compact(ds):
        return [[round(float((x["bbox"][0] + x["bbox"][2]) / 2), 1),
                 round(float((x["bbox"][1] + x["bbox"][3]) / 2), 1),
                 round(float(x["conf"]), 2), int(x.get("cls", 0))] for x in ds]

    def rows(ms, ds, tag):
        out = []
        for di, sid in ms:
            cx, cy = center_xyxy(ds[di]["bbox"])
            src = prev.get(sid)
            out.append({"d": di, "t": sid, "p": tag,
                        "to": [round(cx, 1), round(cy, 1)],
                        "from": None if src is None else [round(src[0], 1), round(src[1], 1)],
                        "jump": None if src is None else
                        round(math.hypot(cx - src[0], cy - src[1]), 1)})
        return out

    LOG.event("assoc", role=role, dets=compact(dets_hi), dets_weak=compact(dets_lo),
              m=rows(m_hi, dets_hi, "hi") + rows(m_lo, dets_lo, "low"),
              spawned_from=sorted(int(i) for i in unmatched_det),
              coasting=sorted(sid for sid, st in tracks.items()
                              if st["last_seen_t"] < now_t))


def claim_locks(dets, tracks, now_t, role=""):
    """Pairs that are obvious: strong overlap, and mutually each other's best (note 11).

    Returns [(det index, track id)]. Only tracks confirmed on the last frame or two can
    claim -- a coasting track's box is a guess, and a guess must not lock anything.
    """
    lock_iou = PERSON_LOCK_IOU if role == "person" else LUGGAGE_LOCK_IOU
    if not CLAIM_LOCK or not dets or not tracks or lock_iou <= 0:
        return []
    live = {sid: st for sid, st in tracks.items() if now_t - st["last_seen_t"] <= 0.2}
    if not live:
        return []

    best_for_det, best_for_trk = {}, {}
    for di, d in enumerate(dets):
        for sid, st in live.items():
            ov = iou_xyxy(d["bbox"], st["bbox"])
            if ov < lock_iou:
                continue
            if ov > best_for_det.get(di, (0.0, None))[0]:
                best_for_det[di] = (ov, sid)
            if ov > best_for_trk.get(sid, (0.0, None))[0]:
                best_for_trk[sid] = (ov, di)
    return [(di, sid) for di, (ov, sid) in best_for_det.items()
            if best_for_trk.get(sid, (0.0, None))[1] == di]


def cascade_match(dets, tracks, now_t, role="", **kw):
    """Claim locks first, then Hungarian matching in tiers of freshness.

    Each tier sees only the detections the fresher tiers did not take, so a track that has
    been coasting can still recover its object -- but never at the expense of a track that
    is being matched right now.
    """
    no_lock = kw.pop("_no_lock", False)
    if not MATCH_CASCADE or not dets or not tracks:
        return hungarian_match(dets, tracks, now_t, role=role, **kw)

    locked = [] if no_lock else claim_locks(dets, tracks, now_t, role)
    if locked:
        for di, sid in locked:
            LOG.event("claim_lock", role=role, tid=sid,
                      iou=round(iou_xyxy(dets[di]["bbox"], tracks[sid]["bbox"]), 3)) \
                if LOG_ALL_MATCHES else None
        locked_dets = {di for di, _ in locked}
        locked_trks = {sid for _, sid in locked}
        rest_dets = [i for i in range(len(dets)) if i not in locked_dets]
        rest = {sid: st for sid, st in tracks.items() if sid not in locked_trks}
        if rest_dets and rest:
            m, un_d, un_t = cascade_match([dets[i] for i in rest_dets], rest, now_t,
                                          role=role, **dict(kw, _no_lock=True))
            return (locked + [(rest_dets[di], sid) for di, sid in m],
                    {rest_dets[i] for i in un_d}, un_t)
        return locked, set(rest_dets), set(rest)

    remaining = list(range(len(dets)))
    matches, unmatched_tracks = [], set()
    lo = 0.0
    for hi in CASCADE_TIERS:
        tier = {sid: st for sid, st in tracks.items()
                if lo <= now_t - st["last_seen_t"] < hi}
        lo = hi
        if not tier:
            continue
        if not remaining:
            unmatched_tracks |= set(tier)
            continue
        sub = [dets[i] for i in remaining]
        m, _, un_t = hungarian_match(sub, tier, now_t, role=role, **kw)
        taken = set()
        for di, sid in m:
            matches.append((remaining[di], sid))
            taken.add(di)
        unmatched_tracks |= un_t
        remaining = [g for k, g in enumerate(remaining) if k not in taken]
    return matches, set(remaining), unmatched_tracks


def update_tracks_with_matches(
        dets, tracks, matches, unmatched_track_ids,
        dt_frame: float, now_t: float,
        smooth_alpha: float, vel_alpha: float,
        update_class: bool = False,
        num_classes: int = 0,
        update_app: bool = True,
        strong: bool = True
):
    # matched updates
    for di, sid in matches:
        d = dets[di]
        st = tracks[sid]

        old_cx, old_cy = center_xyxy(st["bbox"])
        new_cx, new_cy = center_xyxy(d["bbox"])
        if is_held(st, now_t):  # alive, but the partial box must not move it
            st["conf"] = float(d.get("conf", st.get("conf", 0.0)))
            st["vx"] = st["vy"] = 0.0
            if "trajectory" in st:
                st["trajectory"].append((int(old_cx), int(old_cy), now_t))
            if strong:
                st["last_strong_t"] = now_t
            st["last_seen_t"] = now_t
            st["missed_s"] = 0.0
            continue

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

        if update_app:
            update_appearance(st, d.get("app"))
        if strong:  # last time the detector saw this object clearly, not by inference
            st["last_strong_t"] = now_t

        st["last_seen_t"] = now_t
        st["missed_s"] = 0.0
        st["det_bbox"], st["det_t"] = np.array(d["bbox"], dtype=float), now_t

    # unmatched tracks: keep alive, accumulate missed time
    for sid in unmatched_track_ids:
        st = tracks[sid]
        st["missed_s"] = st.get("missed_s", 0.0) + dt_frame
        st["vx"] = st.get("vx", 0.0) * VEL_DECAY_WHEN_MISSED
        st["vy"] = st.get("vy", 0.0) * VEL_DECAY_WHEN_MISSED


def recover_with_low_conf(dets_lo, tracks, unmatched_ids, now_t, role="person"):
    """Second association pass over the detections that were too weak to start a track.

    A track that found no strong detection this frame is usually not gone -- it is half
    occluded and its score fell under the threshold. Re-attaching it to its own weak box
    is what keeps the id alive across the gap instead of killing it and re-birthing it
    with a new number (and, for a person, orphaning every bag they owned).
    """
    unmatched_ids = set(unmatched_ids)
    if not LOW_CONF_RECOVERY or not dets_lo or not unmatched_ids:
        return [], unmatched_ids

    scored = []
    for di, d in enumerate(dets_lo):
        dcx, dcy = center_xyxy(d["bbox"])
        for sid in unmatched_ids:
            st = tracks[sid]
            age = now_t - st["last_seen_t"]
            if age > LOW_RECOVERY_MAX_AGE:
                continue
            iou, dist, pbb = closest_prediction(d["bbox"], st, age, now_t)
            if role == "luggage" and LUGGAGE_TIGHT_GATING:
                if not luggage_gate_ok(st, iou, dist, pbb, age, now_t,
                                       fresh_diag=LUGGAGE_GATE_FRESH_DIAG,
                                       stale_diag=LUGGAGE_GATE_FRESH_DIAG,
                                       dbb=d["bbox"]):
                    continue
            elif role == "person" and PERSON_TIGHT_GATING:
                if not person_gate_ok(st, iou, dist, pbb, min(age, PERSON_FRESH_AGE),
                                      dbb=d["bbox"]):
                    continue  # a weak box may keep a person, never move them far
                holder = owner_of_fragment(d["bbox"], tracks, now_t, exclude=sid)
                if holder is not None and overlap_frac(d["bbox"], st["bbox"]) < overlap_frac(
                        d["bbox"], tracks[holder]["bbox"]):
                    continue  # (17) a piece of somebody else
            elif iou < LOW_MATCH_IOU_THR and dist > scale_gate(
                    pbb, LOW_MATCH_DIST_THR, LOW_MATCH_DIAG, MATCH_DIST_MIN_PX):
                continue
            sim = appearance_sim(d.get("app"), st.get("app"))
            if sim is not None and sim < sim_thr("relink", APP_MIN_SIM_RELINK, role):
                continue
            scored.append((iou - dist / 1000.0 + (0.0 if sim is None else 0.5 * sim),
                           di, sid, iou, sim))

    scored.sort(key=lambda s: s[0], reverse=True)  # greedy: the sets here are tiny
    matches, used_d, used_t = [], set(), set()
    for _, di, sid, iou, sim in scored:
        if di in used_d or sid in used_t:
            continue
        matches.append((di, sid))
        used_d.add(di)
        used_t.add(sid)
        LOG.event("low_conf_recovered", role=role, tid=sid,
                  conf=round(float(dets_lo[di]["conf"]), 3), iou=round(iou, 3), sim=_r3(sim))
    return matches, unmatched_ids - used_t


def prune_tracks_by_ttl(tracks, ttl_seconds: float):
    """Drop expired tracks and return them, so the caller can remember their identity."""
    dead = {sid: st for sid, st in tracks.items() if st.get("missed_s", 0.0) > ttl_seconds}
    for sid in dead:
        del tracks[sid]
    return dead


# -----------------------------
# Duplicate suppression
# -----------------------------
def should_spawn_new_track(det, tracks, now_t,
                           suppress_iou=NEW_TRACK_SUPPRESS_IOU,
                           suppress_dist=NEW_TRACK_SUPPRESS_DIST,
                           max_age=NEW_TRACK_SUPPRESS_MAX_AGE, matched=None):
    """Prevent creating a new ID for a det that matches an existing recent track.

    `matched` = ids that already took a detection this frame. Such a track is not missing
    anything, so a box merely NEAR it is a second object; only overlap suppresses then.
    """
    dbb = det["bbox"]
    dcx, dcy = center_xyxy(dbb)

    for sid, st in tracks.items():
        age = now_t - st["last_seen_t"]
        if age > max_age:
            continue

        pbb = predict_bbox(st, age)
        iou = iou_xyxy(dbb, pbb)
        pcx, pcy = center_xyxy(pbb)
        dist = ((dcx - pcx) ** 2 + (dcy - pcy) ** 2) ** 0.5

        if suppress_iou > 0 and iou >= suppress_iou:
            return False
        if matched is not None and SUPPRESS_MATCHED_BY_IOU_ONLY and sid in matched:
            if (overlap_frac(dbb, pbb) >= SUPPRESS_CONTAIN_FRAC
                    or dist <= SUPPRESS_MATCHED_DIAG * max(8.0, box_diag(pbb))):
                return False
            continue
        if suppress_dist > 0 and dist <= scale_gate(
                pbb, suppress_dist, SUPPRESS_DIAG, MATCH_DIST_MIN_PX):
            return False
    return True


def revive_from_memory(det, retired, now_t, memory_sec, revive_iou, revive_dist,
                       min_sim=None, max_scale=None):
    """Give a re-appearing object its old ID back instead of a fresh identity.

    Position alone is not enough once the gap is more than a moment long, so the search
    radius grows with the time out of sight and the colour signature has to agree. For a
    bag that means keeping its owner and its timer; for a person it means every bag that
    named them is still bound to the right human.
    """
    if min_sim is None:
        min_sim = APP_MIN_SIM_REVIVE
    dbx, dby = bottom_center(det["bbox"])
    best_id, best_score = None, None
    for tid, st in retired.items():
        gap = now_t - st["last_seen_t"]
        if gap > memory_sec:
            continue
        iou = iou_xyxy(det["bbox"], st["bbox"])
        sx, sy = bottom_center(st["bbox"])
        dist = math.hypot(dbx - sx, dby - sy)
        max_dist = (scale_gate(st["bbox"], revive_dist, REVIVE_DIAG, MATCH_DIST_MIN_PX)
                    * min(1.0 + REVIVE_DIST_GROWTH * gap,
                          1e9 if max_scale is None else float(max_scale)))
        if iou < revive_iou and dist > max_dist:
            continue
        sim = appearance_sim(det.get("app"), st.get("app"))
        # a fresh, strongly overlapping box speaks for itself; anything older or looser
        # needs the colours to agree, or the id goes to whoever took the empty spot
        if sim is not None and sim < min_sim and (iou < revive_iou
                                                  or gap > REVIVE_TRUST_IOU_SEC):
            continue
        score = iou + (0.0 if sim is None else sim) - dist / 1000.0
        if best_score is None or score > best_score:
            best_id, best_score = tid, score
    return best_id


# (14) LIMITS ON RECLAIMING A COASTING TRACK (v2026-09-24b)
# Measured on the 2026-09-24 22:19 AVSSS07_MEDIUmo run: of 47 reclaims, 12 were implausible.
# Luggage came back at colour similarity 0.45-0.62 (the generic revive floor is 0.35), and
# bags nobody carried "moved" 4-5 box diagonals while hidden. One person reclaim jumped
# 6.3 person-heights in 3 s, another 2.2 heights in 0.4 s -- faster than anyone walks.
RECLAIM_LUGGAGE_MIN_SIM = 0.65  # a coasting bag needs at least this colour match
RECLAIM_LUGGAGE_MAX_JUMP_DIAG = 3.0  # an uncarried bag cannot move further than this
RECLAIM_PERSON_BASE_H = 1.0  # a person may re-appear this many heights away...
RECLAIM_PERSON_SPEED_H = 1.5  # ...plus this many heights per second hidden (a brisk walk)


def reclaim_plausible(role, st, det, now_t):
    """Could the coasting track `st` physically have got to where `det` is?"""
    gap = max(0.0, now_t - st["last_seen_t"])
    ocx, ocy = center_xyxy(st["bbox"])
    ncx, ncy = center_xyxy(det["bbox"])
    jump = math.hypot(ncx - ocx, ncy - ocy)
    if role == "luggage":
        carried = st["last_seen_t"] - st.get("carrier_t", -1e9) <= CARRIER_MAX_AGE
        return carried or jump <= RECLAIM_LUGGAGE_MAX_JUMP_DIAG * max(8.0, box_diag(st["bbox"]))
    h = st.get("h_ref") or (st["bbox"][3] - st["bbox"][1])
    return jump <= (RECLAIM_PERSON_BASE_H + RECLAIM_PERSON_SPEED_H * gap) * max(MIN_PERSON_H, h)


def reclaim_identity(det, retired, tracks, unmatched_ids, claimed, now_t, memory_sec,
                     revive_iou, revive_dist, min_sim, max_scale, role="person"):
    """Is this 'new' detection really an object we already know? (v2026-09-24)

    Before this, revive_from_memory only searched RETIRED tracks -- ones missing longer
    than the TTL. A person hidden for 1-6 s whose re-appearance the tight gates refused
    was in neither place: too old for spawn suppression (1 s), too young to be retired.
    So a new id was minted while the old one coasted to its death, and every bag that
    named the old id lost its owner. Two pools are searched now, in order:

      1. tracks still ALIVE but unmatched this frame (coasting through an occlusion).
         The gates already refused them, so geometry alone may not bring them back:
         only a detection whose colour signature agrees (sim >= min_sim) is considered;
      2. retired tracks (the old behaviour, unchanged).

    Returns (id, "reclaimed" | "revived") or (None, None). `claimed` stops two detections
    in the same frame from taking the same identity.
    """
    coasting = {}
    for k in unmatched_ids:
        st = tracks.get(k)
        if st is None or k in claimed:
            continue
        sim = appearance_sim(det.get("app"), st.get("app"))
        floor = max(min_sim, RECLAIM_LUGGAGE_MIN_SIM) if role == "luggage" else min_sim
        if sim is None or sim < floor:  # no colour, no reclaim
            continue
        if not reclaim_plausible(role, st, det, now_t):
            LOG.event("reclaim_refused", role=role, tid=k, sim=_r3(sim),
                      gap_s=round(now_t - st["last_seen_t"], 2))
            continue
        coasting[k] = st
    tid = revive_from_memory(det, coasting, now_t, memory_sec, revive_iou, revive_dist,
                             min_sim=min_sim, max_scale=max_scale) if coasting else None
    if tid is not None:
        claimed.add(tid)
        return tid, "reclaimed"

    pool = {k: v for k, v in retired.items() if k not in claimed}
    tid = revive_from_memory(det, pool, now_t, memory_sec, revive_iou, revive_dist,
                             min_sim=min_sim, max_scale=max_scale)
    if tid is not None:
        claimed.add(tid)
        return tid, "revived"
    return None, None


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

            overlap = iou_xyxy(a["bbox"], b["bbox"])
            streak_a = a.setdefault("_merge_streak", {})
            if overlap < merge_iou:
                streak_a.pop(b_id, None)  # the overlap has to be sustained, not momentary
                continue
            streak_a[b_id] = streak_a.get(b_id, 0) + 1
            if streak_a[b_id] >= MERGE_MIN_FRAMES:
                # keep the OLDER identity -- ownership belongs to the first observation
                if a.get("first_t", 0.0) <= b.get("first_t", 0.0):
                    keep_id, drop_id = a_id, b_id
                else:
                    keep_id, drop_id = b_id, a_id

                K = tracks[keep_id]
                D = tracks[drop_id]
                LOG.event("track_merged", role="luggage", keep=keep_id, drop=drop_id,
                          iou=round(iou_xyxy(a["bbox"], b["bbox"]), 3),
                          keep_owner=K.get("owner_pid"), drop_owner=D.get("owner_pid"))

                # Merge state
                K["unattended_s"] = max(K.get("unattended_s", 0.0), D.get("unattended_s", 0.0))
                if D["last_seen_t"] > K["last_seen_t"]:  # the duplicate carries fresher geometry
                    K["bbox"] = D["bbox"]
                    K["conf"] = D["conf"]
                    K["last_seen_t"] = D["last_seen_t"]
                    K["missed_s"] = 0.0
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
                streak_a.pop(b_id, None)

    for sid in to_delete:
        del tracks[sid]
    for st in tracks.values():  # never keep a streak against a track that no longer exists
        if "_merge_streak" in st:
            for gone in [k for k in st["_merge_streak"] if k in to_delete]:
                del st["_merge_streak"][gone]


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
        px, py, h = person_ground(p)
        if h < MIN_PERSON_H:
            continue
        out[p["tid"]] = math.hypot(bx - px, by - py) / h
    return out


# --- evidence recorded while a bag is unattended -----------------------------------
# Two alarms can look identical in the log and be completely different in kind: one where
# the owner was watched walking away, one where the tracker merely lost them; one where the
# bag sat still, one where it was carried across the scene. Everything below is pure
# measurement -- no threshold here changes when an alarm fires -- so alarms can be graded
# after the fact and any cut-off can be chosen from the data instead of guessed.
def _start_away_evidence(st):
    """Anchor the bag where it stood at the moment its owner left."""
    st["away_anchor"] = bottom_center(st["bbox"])
    st["away_w"] = max(1.0, float(st["bbox"][2] - st["bbox"][0]))
    st["away_drift"] = 0.0
    st["away_frames"] = 0
    st["away_seen"] = 0
    st["away_d_max"] = None
    st["away_observed_s"] = 0.0


def _accrue_away_evidence(st, owner_d, dt, seen=True):
    if st.get("away_anchor") is None:
        _start_away_evidence(st)
    st["away_frames"] = st.get("away_frames", 0) + 1
    if seen:
        ax, ay = st["away_anchor"]
        bx, by = bottom_center(st["bbox"])
        st["away_drift"] = max(st.get("away_drift", 0.0), math.hypot(bx - ax, by - ay))
        # the countdown only advances on frames where the BAG is visible; this is how much
        # of the away time was actually observed rather than inferred
        st["away_observed_s"] = st.get("away_observed_s", 0.0) + dt
    if owner_d is not None:  # the owner was actually SEEN, not merely assumed gone
        st["away_seen"] = st.get("away_seen", 0) + 1
        st["away_d_max"] = (owner_d if st.get("away_d_max") is None
                            else max(st["away_d_max"], owner_d))


def alarm_confidence(ev):
    """Fold an alarm's evidence into one number in [0, 1]. Reporting only, never gating.

    still    the bag stayed where its owner left it (drift in bag-widths)
    seen     the countdown was observed rather than inferred while the bag was hidden
    depart   the owner was watched walking away, not merely lost by the tracker
    owner    the owner carried the bag, or won the election clearly
    """
    still = 1.0 - min(1.0, float(ev.get("bag_drift_w", 0.0)))
    seen = float(ev.get("bag_seen_frac", 0.0))
    d_max = ev.get("owner_d_max_h")
    if d_max is not None and d_max >= D_AWAY:
        depart = 1.0  # watched walking out of range
    else:
        # partial credit: how far they had got when the tracker last measured them
        last = ev.get("owner_last_d_h")
        walking = 0.0 if last is None else min(1.0, float(last) / max(1e-6, D_AWAY))
        depart = max(float(ev.get("owner_seen_frac", 0.0)), walking)
    if (ev.get("owner_carry_s") or 0.0) >= OWNER_SWITCH_CARRY_MIN:
        owner = 1.0
    else:
        margin = ev.get("owner_margin")
        owner = 0.0 if margin is None else min(1.0, float(margin) / max(1e-6, PROVISIONAL_MARGIN))
    conf = (W_CONF_STILL * still + W_CONF_SEEN * seen
            + W_CONF_DEPART * depart + W_CONF_OWNER * owner)
    if ev.get("scene_object"):
        conf -= SCENE_OBJECT_PENALTY
    return round(max(0.0, min(1.0, conf)), 3)


def away_evidence(st):
    """Did the bag stay put, and was the owner ever observed while the clock ran?"""
    n = max(1, st.get("away_frames", 0))
    drift = st.get("away_drift", 0.0)
    # wall-video time since the owner left (the countdown itself only counts seen frames)
    elapsed = st.get("away_elapsed", st.get("unattended_s", 0.0))
    seen = st.get("away_observed_s", 0.0)
    ev = {"bag_seen_frac": round(min(1.0, seen / elapsed), 3) if elapsed > 1e-6 else 0.0,
            "bag_drift_px": round(drift, 1),
            "bag_drift_w": round(drift / max(1.0, st.get("away_w", 1.0)), 2),
            "owner_seen_frac": round(st.get("away_seen", 0) / n, 3),
            "away_frames": int(st.get("away_frames", 0)),
            "away_elapsed_s": round(elapsed, 2),
            "owner_rule": st.get("owner_rule"),
            "owner_margin": st.get("owner_margin"),
            "owner_carry_s": round(st.get("carry", {}).get(st.get("owner_pid"), 0.0), 2),
            "owner_d_max_h": _r3(st.get("away_d_max")),
            "owner_last_d_h": _r3(st.get("owner_last_d")),
            "moved_diag": round(st.get("disp_diag", 0.0), 2)}
    # furniture: already in place when the clip started and never in anybody's hands
    ev["carry_max_s"] = round(st.get("carry_max", 0.0), 2)
    ev["scene_object"] = bool(st.get("first_t", 0.0) <= SCENE_OBJECT_SEC
                              and ev["carry_max_s"] < SCENE_OBJECT_MIN_CARRY
                              and ev["moved_diag"] <= SCENE_OBJECT_MAX_DISP_DIAG)
    ev["confidence"] = alarm_confidence(ev)
    # being furniture only matters when nobody was watched leaving it: an owner seen walking
    # away from a bag that was in place at frame 0 is still an observed abandonment
    watched = ((ev["owner_d_max_h"] is not None and ev["owner_d_max_h"] >= D_AWAY)
               or ev["owner_seen_frac"] >= 0.5)
    ev["review"] = bool(ev["confidence"] < ALARM_REVIEW_CONFIDENCE
                        or (ev["scene_object"] and not watched))
    return ev


def owner_desc_update(st, person, now_t):
    """Remember what the owner looks like and where they were (note 13)."""
    x, y, h = person_ground(person)
    st["owner_desc"] = {"xy": (x, y), "h": max(MIN_PERSON_H, h), "t": now_t}
    if person.get("app") is not None:
        old = st.get("owner_app")
        st["owner_app"] = (person["app"] if old is None
                           else APP_EMA * old + (1.0 - APP_EMA) * person["app"])


def owner_matches(st, person, now_t):
    """(is this still the owner?, similarity, distance moved in person-heights)."""
    desc = st.get("owner_desc")
    sim = appearance_sim(person.get("app"), st.get("owner_app"))
    if desc is None:
        return True, sim, 0.0
    x, y, h = person_ground(person)
    moved = math.hypot(x - desc["xy"][0], y - desc["xy"][1]) / max(MIN_PERSON_H, desc["h"])
    gap = max(1e-3, now_t - desc["t"])
    # a person covers about one height per second at a walk; allow well beyond that
    plausible = moved <= OWNER_VERIFY_JUMP_H * max(1.0, gap / 0.2)
    looks_right = sim is None or sim >= sim_thr("verify", OWNER_VERIFY_MIN_SIM)
    return bool(plausible and looks_right), sim, moved


def resolve_owner_by_description(st, persons, now_t, taken_pids=frozenset(),
                                 allow_position=False, newborn_only=False):
    """Who here IS the owner? -> (pid, how, score) or (None, None, None). (note 13)

    Deliberately conservative in both directions: a wrong answer either hands the bag to a
    stranger (silencing a real alarm) or keeps the wrong person (raising a false one). A
    colour match must be strong and clearly ahead of the runner-up; without colour, the
    nearest candidate must be clearly nearest.
    """
    desc = st.get("owner_desc")
    if not OWNER_BY_DESCRIPTION or desc is None:
        return None, None, None
    gap = now_t - desc["t"]
    if gap > OWNER_RESOLVE_MAX_GAP:
        return None, None, None
    radius = (OWNER_RESOLVE_MAX_H + OWNER_RESOLVE_GROWTH_H * max(0.0, gap)) * desc["h"]

    by_sim, by_dist = [], []
    for p in persons:
        if p["tid"] in taken_pids:
            continue
        if newborn_only and p.get("first_t", -1e9) < desc["t"] - OWNER_RESOLVE_NEWBORN_SLACK:
            continue  # (19) existed while the owner was still here: a bystander
        if OWNER_REQUIRE_STRONG and not p.get("strong", True):
            continue
        x, y, _ = person_ground(p)
        dist = math.hypot(x - desc["xy"][0], y - desc["xy"][1])
        if dist > radius:
            continue
        sim = appearance_sim(p.get("app"), st.get("owner_app"))
        if sim is not None:
            by_sim.append((sim, p["tid"]))
        by_dist.append((dist, p["tid"]))

    by_sim.sort(reverse=True)
    if by_sim and by_sim[0][0] >= sim_thr("resolve", OWNER_RESOLVE_MIN_SIM) and (
            len(by_sim) == 1 or by_sim[0][0] - by_sim[1][0] >= OWNER_RESOLVE_SIM_MARGIN):
        return by_sim[0][1], "appearance", round(by_sim[0][0], 3)

    if not allow_position:
        return None, None, None
    by_dist = [(d, p) for d, p in by_dist if d <= OWNER_RESOLVE_POS_MAX_H * desc["h"]]
    by_dist.sort()
    if by_dist and (len(by_dist) == 1
                    or by_dist[1][0] >= OWNER_RESOLVE_DIST_MARGIN * max(1.0, by_dist[0][0])):
        return by_dist[0][1], "position", round(by_dist[0][0] / desc["h"], 3)
    return None, None, None


def take_over_owner(st, new_pid, now_t, how, score, reason):
    """Move the owner role to another id, carrying its evidence with it."""
    old = st["owner_pid"]
    LOG.event("owner_resolved", lid=st.get("lid"), old_owner=old, new_owner=new_pid,
              via=how, score=score, reason=reason)
    for key in ("score", "carry"):  # the evidence belongs to the human, not the number
        d = st.setdefault(key, {})
        if old in d:
            d[new_pid] = max(d.get(new_pid, 0.0), d.pop(old))
    st["owner_pid"] = new_pid
    st["owner_missing_s"] = 0.0
    st["owner_mismatch_s"] = 0.0
    st["owner_seen_t"] = now_t


def rebind_owner(st, persons, now_t, taken_pids=frozenset()):
    """A person track re-created at the owner's last position IS the owner (ID churn).

    Returns (pid, how) or (None, None). Deliberately hard to satisfy: a wrong re-bind
    silently restarts the unattended clock, which turns a real abandonment into silence.
    """
    last = st.get("owner_bbox")
    seen_gap = now_t - st.get("owner_seen_t", -1e9)
    if last is None:
        return None, None

    born_after = st.get("owner_seen_t", -1e9) - OWNER_REBIND_NEWBORN_SLACK

    def candidate(p):
        return (p["tid"] not in taken_pids
                and p.get("first_t", -1e9) >= born_after)  # a re-creation, not a bystander

    if seen_gap <= OWNER_REBIND_SEC:
        best_pid, best_iou = None, OWNER_REBIND_IOU
        for p in persons:
            if not candidate(p):
                continue
            overlap = iou_xyxy(p["bbox"], last)
            if overlap >= best_iou:
                best_pid, best_iou = p["tid"], overlap
        if best_pid is not None:
            return best_pid, "iou"

    # Geometry failed: the owner may have been re-created several boxes away after the
    # crowd cleared. Their colour signature is all that survived, so match on that -- but
    # only near where they were last seen, and only on a strong match: a wrong re-bind
    # silently cancels an alarm, which is the one failure this system must not have.
    if (not OWNER_REBIND_BY_APPEARANCE
            or not (OWNER_REBIND_MIN_GAP <= seen_gap <= OWNER_REBIND_APP_SEC)):
        return None, None
    ref = st.get("owner_app")
    if ref is None:
        return None, None
    lx, ly = bottom_center(last)
    ranked = []
    for p in persons:
        if not candidate(p):
            continue
        sim = appearance_sim(p.get("app"), ref)
        if sim is None:
            continue
        h = max(MIN_PERSON_H, p["bbox"][3] - p["bbox"][1])
        px, py = bottom_center(p["bbox"])
        if math.hypot(px - lx, py - ly) > OWNER_REBIND_MAX_H * h:
            continue  # too far from the owner's last position to be the same walk
        ranked.append((sim, p["tid"]))
    if not ranked:
        return None, None
    ranked.sort(reverse=True)
    best_sim, best_pid = ranked[0]
    if best_sim < sim_thr("rebind", OWNER_REBIND_MIN_SIM):
        return None, None
    # two people who look equally like the owner mean the colours decided nothing
    if len(ranked) > 1 and best_sim - ranked[1][0] < OWNER_REBIND_SIM_MARGIN:
        return None, None
    return best_pid, "appearance"


def clear_ownership(st, now_t):
    """Forget everything a bag track knew about who owns it (a fresh election follows)."""
    st["owner_pid"] = None
    st["owner_bbox"] = None
    st["owner_app"] = None
    st["owner_missing_s"] = 0.0
    st["owner_seen_t"] = -1e9
    st["votes"] = {}
    st["score"] = {}
    st["carry"] = {}
    st["challenger"] = None
    st["carrier_pid"] = None
    st["carrier_pred"] = None
    st["state"] = PENDING
    st["first_t"] = now_t  # a fresh election window on the object it now holds
    st["away_since"] = None
    st["unattended_s"] = 0.0
    st["away_anchor"] = None
    st["last_near"] = True


def reset_ownership_if_jumped(st, det, now_t, lid=None):
    """A track that went quiet and reappears far away is holding a DIFFERENT object.

    Used only when REFUSE_JUMPED_MATCHES is off (then the match is kept and the ownership is
    re-elected instead). Returns True when the ownership was torn up.
    """
    jumped, gap, jump, diag = match_jumped(st, det, now_t)
    if not jumped:
        return False
    LOG.event("ownership_reset", role="luggage", lid=lid, gap_s=round(gap, 2),
              jump_px=round(jump, 1), diag_px=round(diag, 1),
              owner=st.get("owner_pid"), state=st.get("state"))
    clear_ownership(st, now_t)
    return True


# --- ownership evidence -------------------------------------------------------------
def accrue_ownership_evidence(st, persons, dists, now_t, dt):
    """Who stands by the bag, and -- far more telling -- who MOVES with it.

    In a crowd half a dozen people are within a person-height of any bag, so nearness is
    weak evidence (W_NEAR). Walking with the bag while it moves is strong (W_CARRY): a
    bystander does not follow the bag. Returns the carrier's pid this frame, or None.
    """
    pmap = {p["tid"]: p for p in persons}
    score = st.setdefault("score", {})
    carry = st.setdefault("carry", {})
    k = math.exp(-dt / OWNER_SCORE_TAU) if OWNER_SCORE_TAU > 0 else 1.0
    for d in (score, carry):
        for p in list(d):
            d[p] *= k
            if d[p] < 1e-4:
                del d[p]

    seen_now = st.get("last_seen_t", -1e9) >= now_t - 1e-9
    bag_v = st.get("bag_vel")
    moving = bool(st.get("moving")) and seen_now and bag_v is not None
    nb = math.hypot(*bag_v) if moving else 0.0
    best = None
    for pid, d in dists.items():
        p = pmap[pid]
        if OWNER_REQUIRE_STRONG and not p.get("strong", True):
            continue
        if d <= D_OWN:
            score[pid] = score.get(pid, 0.0) + W_NEAR * (1.0 - 0.5 * d / D_OWN) * dt
        pv = p.get("vel")
        if not moving or d > D_CARRY or pv is None:
            continue
        npv = math.hypot(*pv)
        if nb < 1e-6 or npv < 1e-6:
            continue
        cos = (bag_v[0] * pv[0] + bag_v[1] * pv[1]) / (nb * npv)
        if cos < CARRY_MIN_COS or max(nb, npv) / min(nb, npv) > CARRY_SPEED_RATIO:
            continue
        carry[pid] = carry.get(pid, 0.0) + dt
        st["carry_max"] = max(st.get("carry_max", 0.0), carry[pid])
        score[pid] = score.get(pid, 0.0) + W_CARRY * cos * dt
        if best is None or d < best[1]:
            best = (pid, d)

    if best is None and HELD_BY_PERSON and seen_now and not (
            is_static(st, now_t) or is_held(st, now_t)):
        # nobody was seen walking WITH it, but somebody may simply be holding it: a bag box
        # inside a person box is in their hands (or on their back), moving or not
        for pid, d in dists.items():
            p = pmap[pid]
            if OWNER_REQUIRE_STRONG and not p.get("strong", True):
                continue
            inside = overlap_frac(st["bbox"], p["bbox"])
            if inside < HELD_INSIDE_FRAC:
                continue
            carry[pid] = carry.get(pid, 0.0) + dt
            st["carry_max"] = max(st.get("carry_max", 0.0), carry[pid])
            score[pid] = score.get(pid, 0.0) + W_HELD * dt
            if best is None or d < best[1]:
                best = (pid, d)

    if best is None:
        return None
    pid = best[0]
    if st.get("carrier_pid") != pid:
        LOG.event("carrier", lid=st.get("lid"), pid=pid, d_h=round(best[1], 3),
                  owner=st.get("owner_pid"))
    px, py = bottom_center(pmap[pid]["bbox"])
    cx, cy = center_xyxy(st["bbox"])
    st["carrier_pid"] = pid
    st["carrier_t"] = now_t
    st["carrier_off"] = (cx - px, cy - py)
    return pid


def maybe_switch_owner(st, now_t, owner_far=False):
    """Hand the bag to somebody else only when they are seen CARRYING it (hysteresis).

    owner_far: the owner is beyond D_AWAY or out of sight -- then a second of carrying is
    enough, whatever evidence the absent owner collected before (fix 4).
    """
    lid = st.get("lid")
    owner = st["owner_pid"]
    carry, score = st.get("carry", {}), st.get("score", {})
    if not carry:
        st["challenger"] = None
        return
    ch = max(carry, key=lambda p: score.get(p, 0.0))
    cur = st.get("carrier_pid")
    if (owner is not None and owner_far and cur is not None and cur != owner
            and cur in carry and now_t - st.get("carrier_t", -1e9) <= 0.5):
        ch = cur  # the person carrying it NOW is the candidate, not the absent owner
    if owner is None:
        if carry[ch] >= OWNERLESS_CLAIM_CARRY:
            st["owner_pid"] = ch
            st["owner_missing_s"] = 0.0
            st["challenger"] = None
            st["owner_rule"], st["owner_margin"] = "claimed-by-carrying", 1.0
            LOG.event("owner_elected", lid=lid, owner=ch, rule="claimed-by-carrying",
                      carry_s=round(carry[ch], 2), score=round(score.get(ch, 0.0), 3))
        return
    takeover = (owner_far and ch != owner and carry[ch] >= OWNER_TAKEOVER_CARRY
                and st.get("carrier_pid") == ch and now_t - st.get("carrier_t", -1e9) <= 0.5)
    if not takeover and (ch == owner or carry[ch] < OWNER_SWITCH_CARRY_MIN
                         or score.get(ch, 0.0) <= OWNER_SWITCH_RATIO * score.get(owner, 0.0)
                         + OWNER_SWITCH_MARGIN):
        st["challenger"] = None
        return
    if st.get("challenger") != ch:
        st["challenger"] = ch
        st["challenger_since"] = now_t
        return
    if now_t - st["challenger_since"] < OWNER_SWITCH_SEC:
        return
    LOG.event("owner_switch", lid=lid, old_owner=owner, new_owner=ch,
              old_score=round(score.get(owner, 0.0), 3), new_score=round(score[ch], 3),
              carry_s=round(carry[ch], 2), state=st["state"],
              rule="takeover" if takeover else "carried-more")
    st["owner_pid"] = ch
    st["owner_rule"] = "switched-by-carrying"
    st["owner_margin"] = round(1.0 - score.get(owner, 0.0) / score[ch], 3)
    st["owner_missing_s"] = 0.0
    st["owner_bbox"] = None
    st["owner_app"] = None
    st["owner_seen_t"] = -1e9
    st["challenger"] = None


def maybe_reelect_provisional(st, pmap, now_t):
    """A coin-toss owner gives way to whoever clearly leads on proximity evidence (fix 3)."""
    owner, score = st["owner_pid"], st.get("score", {})
    visible = [p for p in score if p in pmap and p != owner]
    if not visible:
        st["prov_ch"] = None
        return
    lead = max(visible, key=score.get)
    if score[lead] <= (1.0 + PROVISIONAL_LEAD) * score.get(owner, 0.0):
        st["prov_ch"] = None
        return
    if st.get("prov_ch") != lead:
        st["prov_ch"], st["prov_since"] = lead, now_t
        return
    if now_t - st["prov_since"] < PROVISIONAL_SEC:
        return
    LOG.event("owner_switch", lid=st.get("lid"), old_owner=owner, new_owner=lead,
              old_score=round(score.get(owner, 0.0), 3), new_score=round(score[lead], 3),
              carry_s=round(st.get("carry", {}).get(lead, 0.0), 2), state=st["state"],
              rule="provisional")
    st["owner_pid"] = lead
    st["owner_rule"] = "provisional-reelected"
    st["owner_margin"] = round(1.0 - score.get(owner, 0.0) / score[lead], 3)
    st["owner_missing_s"] = 0.0
    st["owner_bbox"] = None
    st["owner_app"] = None
    st["owner_seen_t"] = -1e9
    st["prov_ch"] = None


def update_ownership(st, persons, now_t, dt, taken_pids=frozenset()):
    """Elect an owner, keep them unless someone else carries the bag, run the clock.

    Returns the bag-to-person distances in person-heights.
    """
    lid = st.get("lid")
    prev_state = st["state"]
    pmap = {p["tid"]: p for p in persons}
    dists = person_heights_away(st["bbox"], persons)
    seen_now = st.get("last_seen_t", -1e9) >= now_t - 1e-9
    st.setdefault("score", {})
    st.setdefault("carry", {})
    accrue_ownership_evidence(st, persons, dists, now_t, dt)

    if st["owner_pid"] is None and st["state"] == PENDING:
        for pid, dist_h in dists.items():
            if dist_h > D_OWN:
                continue
            # a person the detector never actually saw clearly cannot be held responsible
            if OWNER_REQUIRE_STRONG and not pmap[pid].get("strong", True):
                continue
            st["votes"].setdefault(pid, []).append(dist_h)
        if now_t - st["first_t"] >= OWNERSHIP_SEC:
            means = {p: sum(v) / len(v) for p, v in st["votes"].items()}
            if means:
                # Only candidates who were actually there for a while are electable.
                # Without the floor, one frame at 0.5 h beats twenty-five at 0.6 h.
                best_n = max(len(v) for v in st["votes"].values())
                floor = max(OWNERSHIP_MIN_VOTES, OWNERSHIP_MIN_VOTE_FRAC * best_n)
                eligible = {p: m for p, m in means.items() if len(st["votes"][p]) >= floor}
                carriers = {p: c for p, c in st["carry"].items() if p in pmap or p in means}
                if carriers:  # whoever brought the bag owns it, near or not right now
                    owner = max(carriers, key=lambda p: st["score"].get(p, 0.0))
                    rule = "carried-it"
                elif eligible and any(st["score"].get(p, 0.0) > 0 for p in eligible):
                    owner = max(eligible, key=lambda p: st["score"].get(p, 0.0))
                    rule = "most-evidence-among-supported"
                elif eligible:
                    owner, rule = min(eligible, key=eligible.get), "nearest-among-supported"
                else:  # nobody cleared the bar; the most-seen candidate is the best guess
                    owner = max(st["votes"], key=lambda p: len(st["votes"][p]))
                    rule = "most-supported"
                st["owner_pid"] = owner
                st["state"] = OWNED
                # how clearly the evidence picked this person: 1.0 = nobody else had any,
                # near 0 = a coin toss in a crowd (worth knowing when the alarm is graded)
                top = st["score"].get(owner, 0.0)
                runner = max([v for p, v in st["score"].items() if p != owner] or [0.0])
                st["owner_rule"] = rule
                st["owner_margin"] = round((top - runner) / top, 3) if top > 1e-9 else 0.0
                LOG.event("owner_elected", lid=lid, owner=owner, rule=rule,
                          margin=st["owner_margin"],
                          mean_h=_r3(means.get(owner)),
                          votes_n=len(st["votes"].get(owner, [])),
                          min_votes=round(floor, 1),
                          carry_s=round(st["carry"].get(owner, 0.0), 2),
                          votes={str(p): [round(m, 3), len(st["votes"][p]),
                                          round(st["score"].get(p, 0.0), 2),
                                          round(st["carry"].get(p, 0.0), 2)]
                                 for p, m in means.items()},
                          window_s=OWNERSHIP_SEC)
            else:  # nobody was ever near it -- ownerless; the clock starts once it stands
                st["state"] = UNATTENDED
                st["away_since"] = None
                st["unattended_s"] = 0.0
                LOG.event("owner_none", lid=lid, candidates=len(dists))
        return dists

    if ALARM_LATCH and st["state"] == ALARM:  # never hand a raised alarm back to anyone
        if seen_now:
            st["unattended_s"] += dt
        st["away_elapsed"] = now_t - st["away_since"]
        return dists

    op = st["owner_pid"]
    owner_far = op is not None and (op not in pmap or dists.get(op, 0.0) > D_AWAY)
    maybe_switch_owner(st, now_t, owner_far)
    if (PROVISIONAL_OWNER and st["owner_pid"] is not None and st["state"] == OWNED
            and st.get("owner_rule") in WEAK_OWNER_RULES
            and (st.get("owner_margin") or 0.0) < PROVISIONAL_MARGIN):
        maybe_reelect_provisional(st, pmap, now_t)
    owner_pid = st["owner_pid"]
    last_near = st.get("last_near", True)
    d_limit = D_AWAY if last_near else D_RETURN_FRAC * D_AWAY  # hysteresis

    # (18) the owner's id went elsewhere and a new id stands where the owner stood
    if OWNER_SPLIT and owner_pid is not None:
        op = pmap.get(owner_pid)
        if op is not None and op.get("whole_t") == now_t:
            st["owner_whole"] = (np.array(op["whole_bb"], float), now_t)
        ow = st.get("owner_whole")
        if ow is not None and 0.0 < now_t - ow[1] <= OWNER_SPLIT_SEC:
            wb, wt = ow
            base = iou_xyxy(op["bbox"], wb) if op is not None else 0.0
            best = None
            for p in persons:
                if p["tid"] == owner_pid or p["tid"] in taken_pids:
                    continue
                if p.get("first_t", -1e9) < wt - 0.1:
                    continue  # already existed while the owner was whole: a bystander
                v = iou_xyxy(p["bbox"], wb)
                if v >= OWNER_SPLIT_IOU and v > base + OWNER_SPLIT_LEAD and (
                        best is None or v > best[1]):
                    best = (p["tid"], v)
            if best is not None:
                take_over_owner(st, best[0], now_t, "split", round(best[1], 3),
                                "owner's id went elsewhere; a new id stands where he stood")
                owner_pid = best[0]
                st["owner_mismatch_s"] = 0.0
                owner_desc_update(st, pmap[owner_pid], now_t)
                st["owner_whole"] = None

    # (note 13) the id is only a handle: check that whoever holds it is still the owner,
    # and if not, ask who here is -- before any distance is measured from them
    if OWNER_BY_DESCRIPTION and owner_pid is not None and owner_pid in pmap:
        good, sim, moved = owner_matches(st, pmap[owner_pid], now_t)
        if good:
            st["owner_mismatch_s"] = 0.0
            owner_desc_update(st, pmap[owner_pid], now_t)
        else:
            st["owner_mismatch_s"] = st.get("owner_mismatch_s", 0.0) + dt
            if st["owner_mismatch_s"] >= OWNER_VERIFY_SEC:
                new_pid, how, score = resolve_owner_by_description(
                    st, persons, now_t, taken_pids | {owner_pid}, allow_position=True)
                LOG.event("owner_mismatch", lid=lid, owner=owner_pid, sim=_r3(sim),
                          moved_h=round(moved, 2), replacement=new_pid, via=how)
                if new_pid is not None:
                    take_over_owner(st, new_pid, now_t, how, score, "id no longer matches")
                    owner_pid = new_pid
                    owner_desc_update(st, pmap[new_pid], now_t)
                else:  # nobody here is the owner: treat the role as vacant, not as present
                    st["owner_mismatch_s"] = 0.0
                    pmap = {k: v for k, v in pmap.items() if k != owner_pid}
                    dists = {k: v for k, v in dists.items() if k != owner_pid}

    owner_d = dists.get(owner_pid) if owner_pid is not None else None
    # "no distance" and "not there" are different: a person too small for MIN_PERSON_H is
    # on screen but unmeasurable, and re-binding them to a lookalike is how a real
    # abandonment gets its clock reset by a passer-by
    owner_visible = owner_pid is not None and owner_pid in pmap
    if owner_d is None and owner_pid is not None and not owner_visible:
        new_pid, via = rebind_owner(st, persons, now_t, taken_pids)
        if new_pid is None and OWNER_BY_DESCRIPTION:
            # the track died and nothing was re-created in its place: the owner may simply
            # be walking around under a number this bag has never heard of
            cand, how, score = resolve_owner_by_description(st, persons, now_t, taken_pids,
                                                            newborn_only=True)
            if cand is not None:
                take_over_owner(st, cand, now_t, how, score, "id gone")
                owner_pid = cand
                owner_d = dists.get(cand)
                owner_visible = cand in pmap
                owner_desc_update(st, pmap[cand], now_t)
        if new_pid is not None:
            LOG.event("owner_rebind", lid=lid, old_owner=owner_pid, new_owner=new_pid,
                      via=via, gap_s=round(now_t - st["owner_seen_t"], 2),
                      iou=round(iou_xyxy(pmap[new_pid]["bbox"], st["owner_bbox"]), 3),
                      sim=_r3(appearance_sim(pmap[new_pid].get("app"), st.get("owner_app"))))
            for key in ("score", "carry"):  # the evidence belongs to the human, not the id
                if owner_pid in st[key]:
                    st[key][new_pid] = max(st[key].get(new_pid, 0.0), st[key].pop(owner_pid))
            st["owner_pid"] = owner_pid = new_pid
            owner_d = dists.get(new_pid)
            owner_visible = new_pid in pmap

    was_missing = st["owner_missing_s"] > 0.0
    if owner_pid is None:
        owner_near = False
    elif owner_d is not None:
        if was_missing:
            LOG.event("owner_back", lid=lid, owner=owner_pid,
                      hidden_s=round(st["owner_missing_s"], 2), d_h=round(owner_d, 3))
        st["owner_missing_s"] = 0.0
        owner = pmap.get(owner_pid)
        if owner is not None:
            st["owner_bbox"] = np.array(owner["bbox"], dtype=float)
            st["owner_seen_t"] = now_t
            # the colour signature is NOT taken from here: whoever holds the id right now
            # may not be the owner, and copying their colours in would make the id verify
            # against itself. owner_desc_update() does it, on verified frames only.
        st["owner_last_d"] = owner_d  # how far they had got when last measured
        # NOTE: the description is refreshed ONLY by the verification block above, on
        # frames where the id still checked out. Refreshing it here as well let a hijacked
        # id rewrite the owner's description within a few frames -- after which it always
        # "matched", and the bag could never notice it was following the wrong person.
        owner_near = owner_d <= d_limit
    elif owner_visible:
        # on screen but too small for a trustworthy person-height: hold the last verdict
        st["owner_missing_s"] = 0.0
        owner = pmap.get(owner_pid)
        if owner is not None:
            st["owner_bbox"] = np.array(owner["bbox"], dtype=float)
            st["owner_seen_t"] = now_t
        owner_near = last_near
    else:  # owner not visible -- occlusion or departure (note 3)
        if not was_missing:
            LOG.event("owner_lost", lid=lid, owner=owner_pid,
                      grace_s=OWNER_GRACE_SEC, visible_people=len(dists))
        st["owner_missing_s"] += dt
        # the grace covers an occlusion of someone who was NEAR; an owner already walking
        # away does not come back by disappearing
        owner_near = last_near and st["owner_missing_s"] < OWNER_GRACE_SEC

    companion_near = None
    if GROUP_ATTENDANCE and not owner_near:
        for pid, c in st["carry"].items():
            if pid == owner_pid or c < COMPANION_CARRY_MIN:
                continue
            d = dists.get(pid)
            if d is not None and d <= d_limit:
                companion_near = pid
                break

    near = owner_near or companion_near is not None
    standing = is_static(st, now_t) or not REQUIRE_STATIC_FOR_ALARM

    # (20) while the clock runs, somebody counting as "back" has to stay back for a while
    confirming = False
    if (OWNER_RETURN_CONFIRM_SEC > 0 and near and standing
            and st.get("away_since") is not None):
        rs = st.get("return_since")
        if rs is None:
            st["return_since"] = rs = now_t
            LOG.event("owner_return_pending", lid=lid, owner=owner_pid, companion=companion_near,
                      d_h=None if owner_d is None else round(owner_d, 3),
                      unattended_s=round(st["unattended_s"], 2))
        if now_t - rs < OWNER_RETURN_CONFIRM_SEC:
            confirming = True
        else:
            LOG.event("owner_return_confirmed", lid=lid, owner=owner_pid,
                      held_s=round(now_t - rs, 2))
            st["return_since"] = None
    else:
        if st.get("return_since") is not None and not near:
            LOG.event("owner_return_rejected", lid=lid, owner=owner_pid,
                      held_s=round(now_t - st["return_since"], 2),
                      unattended_s=round(st["unattended_s"], 2))
        st["return_since"] = None

    if not standing:
        near = True  # carried, or still settling: a moving bag is never abandoned
        confirming = False
        st["return_since"] = None

    if confirming:
        # clock paused, not reset; keep the strict return distance and no occlusion grace
        st["last_near"] = False
        st["away_elapsed"] = now_t - st["away_since"]
    elif near:
        st["last_near"] = near
        st["away_since"] = None
        st["unattended_s"] = 0.0
        st["away_anchor"] = None
        st["state"] = OWNED if owner_pid is not None else UNATTENDED
    else:
        st["last_near"] = near
        if st["away_since"] is None:
            st["away_since"] = now_t
            st["unattended_s"] = 0.0
            _start_away_evidence(st)
        if seen_now:  # the countdown only advances while the bag itself is observed
            st["unattended_s"] += dt
        st["away_elapsed"] = now_t - st["away_since"]
        _accrue_away_evidence(st, owner_d, dt, seen=seen_now)
        st["state"] = ALARM if st["unattended_s"] >= UNATTENDED_SECONDS else UNATTENDED

    if st["state"] != prev_state:
        LOG.event("state", lid=lid, frm=prev_state, to=st["state"], owner=owner_pid,
                  d_h=None if owner_d is None else round(owner_d, 3),
                  unattended_s=round(st["unattended_s"], 2),
                  owner_hidden_s=round(st["owner_missing_s"], 2),
                  standing=standing, companion=companion_near)

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


# -- sidebar styling -------------------------------------------------------------------
# Every text in the panel is >= 0.6 so it stays readable when four outputs are shrunk
# into the 2x2 comparison grid.
UI_TEXT = (240, 240, 240)
UI_MUTED = (170, 170, 170)
UI_RULE = (75, 75, 75)
UI_CARD = (46, 46, 46)
STATE_CHIP = {PENDING: "FINDING", OWNED: "OWNED", UNATTENDED: "AWAY", ALARM: "ALARM"}


def _ui_text(img, text, org, scale, color, thick=1, font=None):
    cv2.putText(img, text, (int(org[0]), int(org[1])), font or UI_FONT, scale, color, thick,
                cv2.LINE_AA)


def _ui_size(text, scale, thick=1, font=None):
    return cv2.getTextSize(text, font or UI_FONT, scale, thick)[0]


def _ui_text_right(img, text, right, y, scale, color, thick=1):
    _ui_text(img, text, (right - _ui_size(text, scale, thick)[0], y), scale, color, thick)


def _ui_fit(text, max_w, scale, thick=1, min_scale=0.5):
    """Largest scale <= `scale` at which `text` fits in `max_w` px."""
    while scale > min_scale and _ui_size(text, scale, thick)[0] > max_w:
        scale -= 0.02
    return scale


def _ui_section(img, title, x, right, y):
    """Grey caps title with a rule after it; returns the y below it."""
    _ui_text(img, title, (x, y), 0.62, UI_MUTED, 2)
    tw = _ui_size(title, 0.62, 2)[0]
    cv2.line(img, (x + tw + 12, y - 8), (right, y - 8), UI_RULE, 2, cv2.LINE_AA)
    return y + 18


def _ui_chip(img, text, x, y_top, h, color, min_w=0, scale=0.62):
    """Filled tag, text colour picked for contrast (black on red for ALARM)."""
    tw, th = _ui_size(text, scale, 2)
    w = max(min_w, tw + 16)
    cv2.rectangle(img, (x, y_top), (x + w, y_top + h), color, -1, cv2.LINE_AA)
    _ui_text(img, text, (x + (w - tw) // 2, y_top + (h + th) // 2), scale, _text_color(color), 2)
    return w


def draw_sidebar(canvas, x0, width, stats, bags, names, alarm_count, flash):
    """Info column beside the footage: header, frame/time/speed, scene counters, a table
    of bags with their owner and state, and thresholds + colour key pinned to the bottom."""
    h = canvas.shape[0]
    x, right = x0 + 16, x0 + width - 16
    inner = right - x
    cv2.line(canvas, (x0, 0), (x0, h), UI_RULE, 2)

    # ---- header ------------------------------------------------------------------
    cv2.rectangle(canvas, (x0 + 2, 0), (x0 + width, 78), (38, 38, 38), -1)
    _ui_text(canvas, "UNATTENDED LUGGAGE", (x, 34), _ui_fit("UNATTENDED LUGGAGE", inner, 0.9, 2),
             (255, 255, 255), 2)
    label = str(MODEL_LABEL)
    sc = _ui_fit(label, inner - 16, 0.7, 2)
    tw, th = _ui_size(label, sc, 2)
    cv2.rectangle(canvas, (x, 44), (x + tw + 16, 72), (100, 66, 20), -1)  # model pill
    _ui_text(canvas, label, (x + 8, 58 + th // 2), sc, (255, 228, 175), 2)

    # ---- frame / time / speed ----------------------------------------------------
    y = 114
    total = stats.get("frames") or 0
    _ui_text(canvas, "frame", (x, y), 0.66, UI_MUTED, 2)
    _ui_text_right(canvas, f"{stats['frame']} / {total or '?'}", right, y, 0.72, UI_TEXT, 2)
    y += 11
    cv2.rectangle(canvas, (x, y), (right, y + 6), UI_CARD, -1)
    if total:  # progress through the clip
        frac = min(1.0, stats["frame"] / float(total))
        cv2.rectangle(canvas, (x, y), (x + int(inner * frac), y + 6), (205, 165, 60), -1)
    y += 38
    _ui_text(canvas, f"{stats['t']:.1f} s", (x, y), 0.72, UI_TEXT, 2)
    _ui_text_right(canvas, f"{stats['fps']:.1f} fps | {stats['ms']:.0f} ms", right, y, 0.66,
                   (130, 240, 130), 2)
    y += 20

    # ---- scene counters (cards) --------------------------------------------------
    cards = [("PERSONS", stats["persons"], COLOR_PERSON),
             ("LUGGAGE", stats["luggage"], (215, 215, 215)),
             ("OWNED", stats["owned"], COLOR_OWNED),
             ("AWAY", stats["unattended"], COLOR_UNATTENDED),
             ("ALARMS", f"{stats['alarms']}/{stats['alarms_total']}", COLOR_ALARM)]
    gap, per_row, card_h = 10, 3, 64
    cw = (inner - gap * (per_row - 1)) // per_row
    for i, (cap, val, col) in enumerate(cards):
        r, c = divmod(i, per_row)
        cx, cy = x + c * (cw + gap), y + r * (card_h + gap)
        hot = (cap == "ALARMS" and stats["alarms"]) or (cap == "AWAY" and stats["unattended"])
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + card_h), col if hot else UI_CARD, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + 5), col, -1)  # colour strip
        ink = _text_color(col) if hot else UI_TEXT
        _ui_text(canvas, cap, (cx + 10, cy + 25), 0.6, ink if hot else UI_MUTED, 2)
        _ui_text(canvas, str(val), (cx + 10, cy + card_h - 9), 1.0, ink, 2)
    y += 2 * card_h + gap + 40

    # ---- thresholds + colour key (pinned to the bottom) ----------------------------
    alarm_h = 80 if alarm_count else 0
    key = [("owner?", COLOR_PENDING), ("owned", COLOR_OWNED), ("away", COLOR_UNATTENDED),
           ("ALARM", COLOR_ALARM), ("person", COLOR_PERSON), ("owner", COLOR_OWNER)]
    tags = ["P=person"] + sorted({f"{short_name(n)}={n}" for n in names.values()})
    tag_lines, line = [], ""
    for t in tags:  # wrap the tag glossary to the panel width
        cand = (line + "   " + t) if line else t
        if line and _ui_size(cand, 0.62, 2)[0] > inner:
            tag_lines.append(line)
            cand = t
        line = cand
    tag_lines.append(line)
    key_rows = (len(key) + 2) // 3
    foot_h = 34 + 36 + key_rows * 42 + len(tag_lines) * 28 + 6
    foot_top = h - alarm_h - foot_h

    # ---- bag / owner table -------------------------------------------------------------
    y = _ui_section(canvas, "LUGGAGE  -  OWNER", x, right, y) + 2
    row_h = 42
    max_rows = max(1, (foot_top - 12 - y) // row_h)
    if not bags:
        _ui_text(canvas, "no luggage in view", (x, y + 26), 0.66, (130, 130, 130))
    shown = bags if len(bags) <= max_rows else bags[:max_rows - 1]
    for b in shown:
        col = STATE_COLOR[b["state"]]
        _ui_chip(canvas, STATE_CHIP[b["state"]], x, y, 30, col, min_w=112)
        _ui_text(canvas, b["tag"], (x + 124, y + 24), 0.75, UI_TEXT, 2)
        _ui_text(canvas, "->", (x + 200, y + 23), 0.62, UI_MUTED)
        _ui_text(canvas, b["owner"], (x + 234, y + 24), 0.75,
                 COLOR_OWNER if b["owner"] != "-" else UI_MUTED, 2)
        dcol = col if b["state"] in (UNATTENDED, ALARM) else UI_TEXT
        _ui_text_right(canvas, b["detail"], right, y + 24,
                       _ui_fit(b["detail"], right - x - 300, 0.72, 2), dcol, 2)
        if b.get("progress") is not None:  # time away, filling up towards the alarm
            cv2.rectangle(canvas, (x + 124, y + 32), (right, y + 36), UI_CARD, -1)
            cv2.rectangle(canvas, (x + 124, y + 32),
                          (x + 124 + int((right - x - 124) * b["progress"]), y + 36), col, -1)
        y += row_h
    if len(shown) < len(bags):
        _ui_text(canvas, f"+ {len(bags) - len(shown)} more", (x, y + 20), 0.62, UI_MUTED)

    # ---- footer ----------------------------------------------------------------------
    y = foot_top
    cv2.line(canvas, (x, y), (right, y), UI_RULE, 2)
    y += 34
    kw = inner // 3
    for i, txt in enumerate((f"own <= {D_OWN}h", f"away > {D_AWAY}h",
                             f"alarm {UNATTENDED_SECONDS:.0f}s")):
        _ui_text(canvas, txt, (x + i * kw, y), _ui_fit(txt, kw - 8, 0.66, 2), (215, 215, 215), 2)
    y += 14
    for i, (txt, col) in enumerate(key):
        r, c = divmod(i, 3)
        kx, ky = x + c * kw, y + 6 + r * 42
        cv2.rectangle(canvas, (kx, ky), (kx + 44, ky + 32), col, -1)  # colour swatch
        cv2.rectangle(canvas, (kx, ky), (kx + 44, ky + 32), (235, 235, 235), 1)
        _ui_text(canvas, txt, (kx + 52, ky + 24), _ui_fit(txt, kw - 58, 0.66, 2),
                 (225, 225, 225), 2)
    y += 6 + key_rows * 42 + 20
    for t in tag_lines:
        _ui_text(canvas, t, (x, y), 0.62, (190, 190, 190), 2)
        y += 28

    # ---- alarm banner: black text on red, a white frame blinks ------------------------
    if alarm_count:
        box_h = 66
        top = h - box_h - 10
        cv2.rectangle(canvas, (x0 + 10, top), (x0 + width - 10, top + box_h), COLOR_ALARM, -1)
        if flash:
            cv2.rectangle(canvas, (x0 + 10, top), (x0 + width - 10, top + box_h),
                          (255, 255, 255), 5)
        _ui_text(canvas, "ALARM", (x + 6, top + 47), 1.35, (0, 0, 0), 3)
        msg = f"{alarm_count} unattended bag{'s' if alarm_count != 1 else ''}"
        _ui_text_right(canvas, msg, right - 8, top + 43, 0.75, (0, 0, 0), 2)


def draw_alarm_border(img, flash):
    """Edge-only alert so nothing is hidden behind a banner."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255) if flash else COLOR_ALARM, 6)


# -----------------------------
# Main
# -----------------------------
def run():
    assert os.path.exists(VIDEO_IN), f"Input video not found: {VIDEO_IN}"

    global MODEL_LABEL
    if MODEL_PROFILE not in MODEL_PROFILES:
        raise KeyError(f"Unknown MODEL_PROFILE {MODEL_PROFILE!r}; choose from {list(MODEL_PROFILES)}")
    profile = MODEL_PROFILES[MODEL_PROFILE]
    MODEL_LABEL = profile.get("label", MODEL_PROFILE)
    person_weights, luggage_weights = profile["person"], profile["luggage"]
    for role, wpath in (("person", person_weights), ("luggage", luggage_weights)):
        # a bare name ("yolov12x.pt") may be fetched by Ultralytics; a path must exist
        if (os.sep in wpath or wpath.startswith("SET_PATH")) and not os.path.exists(wpath):
            raise FileNotFoundError(f"{MODEL_PROFILE}: {role} weights not found: {wpath} "
                                    f"(edit MODEL_PROFILES at the top of the script)")
    model_person = YOLO(person_weights)
    # one model in both roles: load once, infer once per frame, split the result by class
    shared_model = luggage_weights == person_weights
    model_luggage = model_person if shared_model else YOLO(luggage_weights)

    global REID
    REID, reid_note = None, "off (colour histogram)"
    if PERSON_REID:
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            import sys as _sys
            if here not in _sys.path:
                _sys.path.insert(0, here)
            from person_reid import PersonReID
            wpath = REID_WEIGHTS if os.path.isabs(REID_WEIGHTS) else os.path.join(here, REID_WEIGHTS)
            REID = PersonReID(wpath, arch=REID_ARCH, device=DEVICE)
            tpath = os.path.join(here, REID_THRESHOLDS_FILE)
            if os.path.exists(tpath):
                with open(tpath) as f:
                    REID_SIM.update({k: float(v) for k, v in json.load(f)["thresholds"].items()
                                     if k in REID_SIM})
                reid_note = f"{REID.name}, thresholds from {REID_THRESHOLDS_FILE}"
            else:
                reid_note = (f"{REID.name}, DEFAULT thresholds -- run reid_check.py "
                             f"to measure them on this footage")
        except Exception as e:  # never let the re-ID model stop a run
            REID = None
            reid_note = f"UNAVAILABLE, using colour histogram ({type(e).__name__}: {e})"

    luggage_class_ids = LUGGAGE_CLASS_IDS
    if shared_model and luggage_class_ids is None:
        luggage_class_ids = COCO_LUGGAGE_CLASS_IDS
    luggage_names = model_luggage.names  # read from the weights, never hardcoded
    # a COCO detector emits 24/26/28 -- fold them onto the custom 3-label space
    label_map = COCO_TO_CUSTOM if REMAP_COCO_LUGGAGE and luggage_names.get(0) == "person" else None
    if label_map is not None:
        if luggage_class_ids is None:
            luggage_class_ids = sorted(label_map)
        luggage_names = dict(CUSTOM_LUGGAGE_NAMES)
    else:  # custom model: keep its classes, spell them the way the pipeline does
        luggage_names = {int(k): canon_luggage_name(v) for k, v in luggage_names.items()}
    num_luggage_classes = max(luggage_names) + 1 if luggage_names else 1

    run_dir = make_run_dir(VIDEO_IN, MODEL_PROFILE)
    out_video = output_video_path(VIDEO_IN, MODEL_PROFILE)  # beside the input, named after it

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
    global FRAME_WH
    FRAME_WH = (w, h)

    print(f"  model   : {MODEL_PROFILE}  ({MODEL_LABEL})")
    print(f"  input   : {VIDEO_IN}  {w}x{h} @ {video_fps:.2f} fps, {total_frames} frames")
    print(f"  person  : {person_weights}  classes {PERSON_CLASS_IDS}")
    print(f"  luggage : {luggage_weights}  classes {luggage_class_ids or 'all'}"
          f"{'  [shared]' if shared_model else ''}")
    if label_map is not None:
        print(f"  labels  : {', '.join(f'{k}->{luggage_names[v]}' for k, v in sorted(label_map.items()))}")
    ident = script_identity()
    print(f"  script  : {ident['script']}")
    print(f"            v{ident['version']}  {ident['script_bytes']} bytes  "
          f"modified {ident['script_mtime']}")
    print(f"  run dir : {run_dir}")
    print(f"  video   : {out_video}")
    print(f"  re-ID   : {reid_note}")
    if REID is not None:
        print("            " + "  ".join(f"{k} {v:.2f}" for k, v in REID_SIM.items()))
    print(f"  owner within {D_OWN}h, away beyond {D_AWAY}h, alarm after {UNATTENDED_SECONDS:.0f}s\n")

    # the panel gets its own column, so the rendered canvas is wider than the video
    side_w = max(0, SIDEBAR_WIDTH)
    canvas_w = w + side_w
    canvas_h = max(h, SIDEBAR_MIN_HEIGHT) if side_w else h
    video_x0 = side_w if SIDEBAR_SIDE == "left" else 0
    video_y0 = (canvas_h - h) // 2
    side_x0 = 0 if SIDEBAR_SIDE == "left" else w

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if out_video.lower().endswith(".avi") else "mp4v"))
        writer = cv2.VideoWriter(out_video, fourcc, video_fps, (canvas_w, canvas_h))
        print(f"[SAVE] {out_video}  ({canvas_w}x{canvas_h})")

    if SHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1800, canvas_w), min(1000, canvas_h))

    # Tracks (IDs never reused)
    next_person_id = 1
    person_tracks = {}  # PID -> {bbox, conf, last_seen_t, missed_s, vx, vy, trajectory}
    next_luggage_id = 1
    luggage_tracks = {}  # LID -> the same + {state, owner_pid, votes, away_since, unattended_s}
    retired_bags = {}  # LID -> state of bags that vanished, kept for LUGGAGE_MEMORY_SEC
    retired_persons = {}  # PID -> same, so a returning person keeps their id and their bags

    alarm_manager = AlarmManager()
    events = []

    global LOG
    LOG = EventLog(os.path.join(run_dir, LOG_JSONL))
    LOG.event("run", source=VIDEO_IN, size=[w, h], fps=round(video_fps, 3), frames=total_frames,
              model_profile=MODEL_PROFILE, model_label=MODEL_LABEL,
              person_model=person_weights, luggage_model=luggage_weights,
              shared_model=shared_model, luggage_classes=luggage_class_ids,
              run_dir=run_dir, out_video=out_video, **script_identity(),
              label_map=None if label_map is None else {str(k): v for k, v in label_map.items()},
              names={str(k): v for k, v in luggage_names.items()},
              params={"d_own": D_OWN, "d_away": D_AWAY, "ownership_sec": OWNERSHIP_SEC,
                      "unattended_seconds": UNATTENDED_SECONDS,
                      "owner_grace_sec": OWNER_GRACE_SEC, "alarm_latch": ALARM_LATCH,
                      "luggage_ttl": LUGGAGE_TTL_SECONDS, "luggage_memory": LUGGAGE_MEMORY_SEC,
                      "owner_rebind_sec": OWNER_REBIND_SEC,
                      "conf_person": CONF_PERSON, "conf_luggage": CONF_LUGGAGE, "imgsz": IMGSZ,
                      "tight_gating": LUGGAGE_TIGHT_GATING,
                      "person_tight_gating": PERSON_TIGHT_GATING,
                      "person_gate_fresh_diag": PERSON_GATE_FRESH_DIAG,
                      "refuse_jumps": REFUSE_JUMPED_MATCHES,
                      "require_static": REQUIRE_STATIC_FOR_ALARM,
                      "bag_static_sec": BAG_STATIC_SEC, "w_carry": W_CARRY,
                      "d_carry": D_CARRY, "group_attendance": GROUP_ATTENDANCE,
                      "d_return": round(D_RETURN_FRAC * D_AWAY, 3),
                      "carrier_prediction": CARRIER_PREDICTION},
              reid=reid_note, reid_sim=dict(REID_SIM) if REID is not None else None)

    # Detections below the reporting threshold are kept for the recovery pass -- they may
    # not start a track, but they can keep one alive through a partial occlusion.
    person_conf_floor = min(CONF_PERSON, CONF_PERSON_LOW) if LOW_CONF_RECOVERY else CONF_PERSON
    luggage_conf_floor = min(CONF_LUGGAGE, CONF_LUGGAGE_LOW) if LOW_CONF_RECOVERY else CONF_LUGGAGE

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
        LOG.mark(now_t, frame_count)

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
                conf=min(person_conf_floor, luggage_conf_floor),
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                classes=wanted or None,
                verbose=False
            )[0]
        else:
            res_p = model_person.predict(
                source=frame,
                conf=person_conf_floor,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                classes=PERSON_CLASS_IDS,
                verbose=False
            )[0]

            res_l = model_luggage.predict(
                source=frame,
                conf=luggage_conf_floor,
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
        person_dets, person_dets_low = [], []
        if res_p.boxes is not None and len(res_p.boxes) > 0:
            p_xyxy = res_p.boxes.xyxy.detach().cpu().numpy().astype(float)
            p_conf = res_p.boxes.conf.detach().cpu().numpy().astype(float)
            p_cls = res_p.boxes.cls.detach().cpu().numpy().astype(int)
            for bb, cf, cid in zip(p_xyxy, p_conf, p_cls):
                if int(cid) not in PERSON_CLASS_IDS or cf < person_conf_floor:
                    continue
                det = {"bbox": bb, "conf": float(cf), "cls": 0,
                       "app": None if REID is not None else appearance_of(frame, bb, "person")}
                if cf >= CONF_PERSON:
                    person_dets.append(det)
                    if LOG_DETECTIONS:
                        LOG.event("det", role="person", conf=round(float(cf), 3), bbox=bb)
                elif LOW_CONF_RECOVERY:
                    person_dets_low.append(det)  # too weak to start a track, good enough to keep one
                    if LOG_DETECTIONS:  # the weak pool has to be in the record too, or a
                        LOG.event("det", role="person", conf=round(float(cf), 3),
                                  bbox=bb, weak=True)  # frame cannot be replayed later

        if REID is not None and (person_dets or person_dets_low):
            embs = REID.embed(frame, [d["bbox"] for d in person_dets + person_dets_low])
            for d, e in zip(person_dets + person_dets_low, embs):
                d["app"] = e

        p_prev = {sid: center_xyxy(st["bbox"]) for sid, st in person_tracks.items()}
        p_matches, p_unmatched_det, p_unmatched_tracks = cascade_match(
            person_dets, person_tracks, now_t, role="person",
            require_same_class=False,
            max_relink_age=PERSON_MAX_RELINK_AGE,
            match_iou_thr=PERSON_MATCH_IOU_THR,
            match_dist_thr=PERSON_MATCH_DIST_THR,
            class_mismatch_penalty=0.0,
            app_weight=W_APP
        )

        if REFUSE_JUMPED_MATCHES and PERSON_TIGHT_GATING:
            # the same refusal the luggage tracks get: a quiet track that reappears far
            # away is a different human, and taking that detection strands the one it
            # was actually following (AVSSS07_MEDIUmo frame 427, P5 <-> P8)
            for di, sid in list(p_matches):
                jumped, gap, jump, diag = match_jumped(person_tracks[sid], person_dets[di],
                                                       now_t, role="person")
                if not jumped:
                    continue
                LOG.event("match_refused", role="person", tid=sid, gap_s=round(gap, 2),
                          jump_px=round(jump, 1), diag_px=round(diag, 1))
                p_matches.remove((di, sid))
                p_unmatched_det.add(di)
                p_unmatched_tracks.add(sid)

        update_tracks_with_matches(
            person_dets, person_tracks, p_matches, set(),
            dt_frame=dt, now_t=now_t,
            smooth_alpha=PERSON_SMOOTH_ALPHA, vel_alpha=PERSON_VEL_ALPHA,
            update_class=False, update_app=True
        )

        mark_eclipsed(person_dets, person_tracks, p_matches, p_unmatched_tracks, now_t,
                      role="person")

        # second pass over the weak detections, then age whatever is still unmatched
        p_low, p_unmatched_tracks = recover_with_low_conf(
            person_dets_low, person_tracks, p_unmatched_tracks, now_t, role="person")
        update_tracks_with_matches(
            person_dets_low, person_tracks, p_low, p_unmatched_tracks,
            dt_frame=dt, now_t=now_t,
            smooth_alpha=PERSON_SMOOTH_ALPHA * LOW_CONF_SMOOTH_SCALE,
            vel_alpha=PERSON_VEL_ALPHA * LOW_CONF_SMOOTH_SCALE,
            update_class=False, update_app=False, strong=False
        )
        log_assoc("person", p_prev, person_dets, p_matches, person_dets_low, p_low,
                  p_unmatched_det, person_tracks, now_t)

        # create new person tracks -- but only when this really is a new person
        claimed_p = set()
        for di in p_unmatched_det:
            d = person_dets[di]

            # a detection sitting on top of a recent track IS that track; spawning here is
            # the textbook way to end up with two ids for one human
            holder = owner_of_fragment(d["bbox"], person_tracks, now_t)
            if holder is not None:
                LOG.event("spawn_suppressed", role="person", reason="fragment", of=holder,
                          conf=round(d["conf"], 3), bbox=d["bbox"])
                continue

            if not should_spawn_new_track(d, person_tracks, now_t,
                                          PERSON_NEW_TRACK_SUPPRESS_IOU,
                                          PERSON_NEW_TRACK_SUPPRESS_DIST,
                                          PERSON_NEW_TRACK_SUPPRESS_MAX_AGE):
                LOG.event("spawn_suppressed", role="person", conf=round(d["conf"], 3),
                          bbox=d["bbox"])
                continue

            # someone who walked back into view keeps the id they had -- whether their old
            # track is still coasting or already retired -- so every bag that elected them
            # as owner stays bound to the right person (note 3)
            old_pid, how = (None, None) if is_fragment_box(d["bbox"]) else reclaim_identity(
                d, retired_persons, person_tracks, p_unmatched_tracks, claimed_p, now_t,
                PERSON_MEMORY_SEC, PERSON_REVIVE_IOU, PERSON_REVIVE_DIST,
                min_sim=sim_thr("revive", max(APP_MIN_SIM_REVIVE, PERSON_REVIVE_MIN_SIM)),
                max_scale=PERSON_REVIVE_MAX_SCALE, role="person")
            if old_pid is not None:
                st = (retired_persons.pop(old_pid) if how == "revived"
                      else person_tracks[old_pid])
                LOG.event("track_" + how, role="person", tid=old_pid,
                          gap_s=round(now_t - st["last_seen_t"], 2),
                          iou=round(iou_xyxy(d["bbox"], st["bbox"]), 3),
                          sim=_r3(appearance_sim(d.get("app"), st.get("app"))),
                          conf=round(d["conf"], 3), bbox=d["bbox"])
                st["bbox"] = np.array(d["bbox"], dtype=float)
                st["conf"] = float(d["conf"])
                st["last_seen_t"] = now_t
                st["last_strong_t"] = now_t  # a strong detection brought them back
                st["missed_s"] = 0.0
                st["vx"] = st["vy"] = 0.0
                update_appearance(st, d.get("app"))
                person_tracks[old_pid] = st
                continue

            pid = next_person_id
            next_person_id += 1
            cx, cy = center_xyxy(d["bbox"])
            LOG.event("track_new", role="person", tid=pid, conf=round(d["conf"], 3),
                      bbox=d["bbox"])
            person_tracks[pid] = {
                "bbox": np.array(d["bbox"], dtype=float),
                "conf": float(d["conf"]),
                "cls": 0,
                "first_t": now_t,
                "last_seen_t": now_t,
                "last_strong_t": now_t,  # it was spawned from a strong detection
                "missed_s": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "app": d.get("app"),
                "trajectory": deque([(int(cx), int(cy), now_t)], maxlen=TRAJECTORY_MAX_POINTS)
            }

        for pid, st in prune_tracks_by_ttl(person_tracks, PERSON_TTL_SECONDS).items():
            LOG.event("track_lost", role="person", tid=pid,
                      alive_s=round(st["last_seen_t"] - st.get("first_t", st["last_seen_t"]), 2),
                      last_seen_t=round(st["last_seen_t"], 2), bbox=st["bbox"])
            retired_persons[pid] = st  # remembered, not deleted -- they may come back
        for pid in [k for k, s in retired_persons.items()
                    if now_t - s["last_seen_t"] > PERSON_MEMORY_SEC]:
            LOG.event("track_forgotten", role="person", tid=pid)
            del retired_persons[pid]

        for st in person_tracks.values():
            if st["last_seen_t"] >= now_t:
                person_reference(st, h)
                update_person_whole(st, now_t)

        # visible persons list (recent only)
        persons = []
        for pid, st in person_tracks.items():
            age = now_t - st["last_seen_t"]
            if age <= DRAW_RECENT_SECONDS:
                bb = st["bbox"]
                cx, cy = center_xyxy(bb)
                persons.append({"tid": pid, "bbox": bb, "cx": cx, "cy": cy,
                                "conf": st.get("conf", 0.0), "app": st.get("app"),
                                "first_t": st.get("first_t", 0.0),
                                "vel": (velocity(st.get("trajectory"), now_t)
                                        if st["last_seen_t"] >= now_t else None),
                                "h_ref": st.get("h_ref"),
                                "whole_bb": st.get("whole_bb"), "whole_t": st.get("whole_t"),
                                "strong": now_t - st.get("last_strong_t", -1e9)
                                <= OWNER_STRONG_MAX_AGE})
            elif DRAW_PREDICTED_WHEN_MISSING and age <= PERSON_MAX_RELINK_AGE:
                bb = predict_bbox(st, age)
                cx, cy = center_xyxy(bb)
                persons.append({"tid": pid, "bbox": bb, "cx": cx, "cy": cy,
                                "conf": st.get("conf", 0.0), "app": st.get("app"),
                                "first_t": st.get("first_t", 0.0), "pred": True,
                                "strong": now_t - st.get("last_strong_t", -1e9)
                                <= OWNER_STRONG_MAX_AGE})

        # -----------------------------
        # LUGGAGE DETS -> stable luggage tracks
        # -----------------------------
        luggage_dets, luggage_dets_low = [], []
        if res_l.boxes is not None and len(res_l.boxes) > 0:
            l_cls = res_l.boxes.cls.detach().cpu().numpy().astype(int)
            l_conf = res_l.boxes.conf.detach().cpu().numpy().astype(float)
            l_xyxy = res_l.boxes.xyxy.detach().cpu().numpy().astype(float)
            for cid, cf, bb in zip(l_cls, l_conf, l_xyxy):
                if cf < luggage_conf_floor:
                    continue
                cid = int(cid)
                if luggage_class_ids is not None and cid not in luggage_class_ids:
                    continue
                if label_map is not None:
                    if cid not in label_map:
                        continue
                    cid = label_map[cid]
                det = {"cls": cid, "conf": float(cf), "bbox": bb,
                       "app": appearance_of(frame, bb, "luggage")}
                if cf >= CONF_LUGGAGE:
                    luggage_dets.append(det)
                    if LOG_DETECTIONS:
                        LOG.event("det", role="luggage", cls=cid,
                                  name=str(luggage_names.get(cid, cid)),
                                  conf=round(float(cf), 3), bbox=bb)
                elif LOW_CONF_RECOVERY:
                    luggage_dets_low.append(det)
                    if LOG_DETECTIONS:
                        LOG.event("det", role="luggage", cls=cid,
                                  name=str(luggage_names.get(cid, cid)),
                                  conf=round(float(cf), 3), bbox=bb, weak=True)

        # (1) class-agnostic NMS to kill duplicate boxes across classes
        n_raw = len(luggage_dets)
        luggage_dets = nms_dets_xyxy(luggage_dets, iou_thr=LUGGAGE_DET_NMS_IOU, class_agnostic=True)
        if n_raw != len(luggage_dets):
            LOG.event("nms", role="luggage", before=n_raw, after=len(luggage_dets))
        luggage_dets_low = nms_dets_xyxy(luggage_dets_low, iou_thr=LUGGAGE_DET_NMS_IOU,
                                         class_agnostic=True)

        # a standing bag somebody walks in front of keeps its box (fix 1)
        update_occlusion_hold(luggage_tracks, person_tracks, now_t)

        # a carried bag is looked for next to its carrier, wherever they walked to
        for st in luggage_tracks.values():
            st["carrier_pred"] = None
            cp = st.get("carrier_pid")
            pt = person_tracks.get(cp) if (CARRIER_PREDICTION and cp is not None) else None
            if (pt is None or not st.get("moving") or st.get("carrier_off") is None
                    or pt["last_seen_t"] < now_t
                    or now_t - st.get("carrier_t", -1e9) > CARRIER_MAX_AGE):
                continue
            px, py = bottom_center(pt["bbox"])
            ox, oy = st["carrier_off"]
            bw, bh = st["bbox"][2] - st["bbox"][0], st["bbox"][3] - st["bbox"][1]
            st["carrier_pred"] = np.array([px + ox - bw / 2, py + oy - bh / 2,
                                           px + ox + bw / 2, py + oy + bh / 2])
            st["carrier_pred_t"] = now_t

        l_prev = {sid: center_xyxy(st["bbox"]) for sid, st in luggage_tracks.items()}
        l_matches, l_unmatched_det, l_unmatched_tracks = cascade_match(
            luggage_dets, luggage_tracks, now_t, role="luggage",
            require_same_class=False,  # allow cross-class relink
            max_relink_age=LUGGAGE_MAX_RELINK_AGE,
            match_iou_thr=LUGGAGE_MATCH_IOU_THR,
            match_dist_thr=LUGGAGE_MATCH_DIST_THR,
            class_mismatch_penalty=CLASS_MISMATCH_PENALTY,
            use_stable_class_for_penalty=True,
            app_weight=W_APP
        )

        # A track that went quiet and then reappears far away has almost certainly landed
        # on a different object. Keeping its owner would let a stranger's departure alarm
        # on a bag they never touched, so the ownership is torn up and re-elected.
        if REFUSE_JUMPED_MATCHES:
            # ...and with REFUSE_JUMPED_MATCHES the match is not made at all: the box is a
            # new (or revived) object, and the track keeps its own object, owner and alarm
            for di, sid in list(l_matches):
                jumped, gap, jump, diag = match_jumped(luggage_tracks[sid], luggage_dets[di],
                                                       now_t)
                if not jumped:
                    continue
                LOG.event("match_refused", role="luggage", lid=sid, gap_s=round(gap, 2),
                          jump_px=round(jump, 1), diag_px=round(diag, 1),
                          owner=luggage_tracks[sid].get("owner_pid"),
                          state=luggage_tracks[sid].get("state"))
                l_matches.remove((di, sid))
                l_unmatched_det.add(di)
                l_unmatched_tracks.add(sid)
        else:
            for di, sid in l_matches:
                if reset_ownership_if_jumped(luggage_tracks[sid], luggage_dets[di], now_t, sid):
                    alarm_manager.clear(sid)

        update_tracks_with_matches(
            luggage_dets, luggage_tracks, l_matches, set(),
            dt_frame=dt, now_t=now_t,
            smooth_alpha=LUGGAGE_SMOOTH_ALPHA, vel_alpha=LUGGAGE_VEL_ALPHA,
            update_class=True, num_classes=num_luggage_classes, update_app=True
        )

        mark_eclipsed(luggage_dets, luggage_tracks, l_matches, l_unmatched_tracks, now_t,
                      role="luggage")

        # a bag half hidden behind a passer-by is still detected, just weakly -- keeping
        # its track alive here is what stops the timer from restarting under a new id
        l_low, l_unmatched_tracks = recover_with_low_conf(
            luggage_dets_low, luggage_tracks, l_unmatched_tracks, now_t, role="luggage")
        update_tracks_with_matches(
            luggage_dets_low, luggage_tracks, l_low, l_unmatched_tracks,
            dt_frame=dt, now_t=now_t,
            smooth_alpha=LUGGAGE_SMOOTH_ALPHA * LOW_CONF_SMOOTH_SCALE,
            vel_alpha=LUGGAGE_VEL_ALPHA * LOW_CONF_SMOOTH_SCALE,
            update_class=False, update_app=False, strong=False
        )
        log_assoc("luggage", l_prev, luggage_dets, l_matches, luggage_dets_low, l_low,
                  l_unmatched_det, luggage_tracks, now_t)

        # (2) create new luggage tracks but suppress duplicates
        l_matched_now = {sid for _, sid in l_matches} | {sid for _, sid in l_low}
        claimed_l = set()
        for di in sorted(l_unmatched_det):
            d = luggage_dets[di]

            # prevent duplicate LIDs for same physical bag
            if not should_spawn_new_track(d, luggage_tracks, now_t, matched=l_matched_now):
                LOG.event("spawn_suppressed", role="luggage", cls=d["cls"],
                          conf=round(d["conf"], 3), bbox=d["bbox"])
                continue

            # a bag returning to the same spot keeps its ID, its owner and its timer
            # ...whether its old track is still coasting or already retired
            old_lid, how = reclaim_identity(
                d, retired_bags, luggage_tracks, l_unmatched_tracks, claimed_l, now_t,
                LUGGAGE_MEMORY_SEC, LUGGAGE_REVIVE_IOU, LUGGAGE_REVIVE_DIST,
                min_sim=APP_MIN_SIM_REVIVE, max_scale=LUGGAGE_REVIVE_MAX_SCALE,
                role="luggage")
            if old_lid is not None:
                st = (retired_bags.pop(old_lid) if how == "revived"
                      else luggage_tracks[old_lid])
                gap = now_t - st["last_seen_t"]  # time out of sight is not observed time
                LOG.event("track_" + how, role="luggage", lid=old_lid, gap_s=round(gap, 2),
                          owner=st["owner_pid"], state=st["state"],
                          iou=round(iou_xyxy(d["bbox"], st["bbox"]), 3),
                          sim=_r3(appearance_sim(d.get("app"), st.get("app"))),
                          conf=round(d["conf"], 3), bbox=d["bbox"])
                if how == "revived":  # only a retired track had its clocks stopped
                    st["first_t"] += gap
                    if st["away_since"] is not None:
                        st["away_since"] += gap
                st["bbox"] = np.array(d["bbox"], dtype=float)
                st["conf"] = float(d["conf"])
                st["last_seen_t"] = now_t
                st["missed_s"] = 0.0
                st["vx"] = st["vy"] = 0.0
                update_appearance(st, d.get("app"))
                luggage_tracks[old_lid] = st
                continue

            lid = next_luggage_id
            next_luggage_id += 1
            LOG.event("track_new", role="luggage", lid=lid, cls=int(d["cls"]),
                      name=str(luggage_names.get(int(d["cls"]), d["cls"])),
                      conf=round(d["conf"], 3), bbox=d["bbox"])

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
                "app": d.get("app"),
                # ownership state (note 3)
                "lid": lid,
                "first_t": now_t,
                "state": PENDING,
                "votes": {},
                "owner_pid": None,
                "owner_missing_s": 0.0,
                "owner_bbox": None,
                "owner_app": None,
                "last_near": True,
                "owner_seen_t": -1e9,
                "away_since": None,
                "unattended_s": 0.0,
                "away_anchor": None,
                "score": {},
                "carry": {},
                "challenger": None,
                "carrier_pid": None,
                "static_anchor": None,
                "static_since": None,
                "moving": False,
                "bag_vel": None,
                "trajectory": deque([(int(cx), int(cy), now_t)], maxlen=TRAJECTORY_MAX_POINTS)
            }

        # (3) merge any duplicates that slipped through
        merge_overlapping_tracks(luggage_tracks, now_t, merge_iou=MERGE_TRACK_IOU, max_age=MERGE_TRACK_MAX_AGE)

        # standing / moving, from the frames each bag was actually matched on
        for st in luggage_tracks.values():
            if st["last_seen_t"] >= now_t:
                update_bag_motion(st, now_t)

        retired = prune_tracks_by_ttl(luggage_tracks, LUGGAGE_TTL_SECONDS)
        for lid, st in retired.items():
            LOG.event("track_lost", role="luggage", lid=lid, owner=st["owner_pid"],
                      state=st["state"], alive_s=round(st["last_seen_t"] - st["first_t"], 2),
                      unattended_s=round(st["unattended_s"], 2), bbox=st["bbox"])
        retired_bags.update(retired)
        for lid in [k for k, s in retired_bags.items()
                    if now_t - s["last_seen_t"] > LUGGAGE_MEMORY_SEC]:
            LOG.event("track_forgotten", role="luggage", lid=lid, owner=retired_bags[lid]["owner_pid"])
            del retired_bags[lid]

        # -----------------------------
        # Unattended logic + drawing
        # -----------------------------
        # a person already bound to one bag must not be re-bound as another bag's owner
        owned_pids = {s["owner_pid"] for s in luggage_tracks.values()
                      if s.get("owner_pid") is not None}

        owner_of = {}  # PID -> [LID, ...], so a person can be annotated as an owner
        for lid, st in luggage_tracks.items():
            if st["owner_pid"] is not None and now_t - st["last_seen_t"] <= DRAW_RECENT_SECONDS:
                owner_of.setdefault(st["owner_pid"], []).append(lid)

        # --- persons first, so the bag/owner links land on top of the boxes ---
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            owns = owner_of.get(p["tid"], [])
            color = COLOR_OWNER if owns else COLOR_PERSON
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          1 if p.get("pred") else 2)

            # only owners get a trace, otherwise the frame fills up with paths
            if SHOW_TRAJECTORIES and p["tid"] in person_tracks and (owns or not TRAJECTORY_OWNERS_ONLY):
                draw_trajectory(annotated, person_tracks[p["tid"]]["trajectory"], color, now_t)

            draw_label(annotated, f"P{p['tid']} {p['conf']:.2f}", x1, y1 - 6, color)

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
            dists = update_ownership(st, persons, now_t, dt,
                                     owned_pids - {st.get("owner_pid")}) if is_recent else {}
            state = st["state"]
            owner = get_person_by_id(persons, st["owner_pid"])
            owner_d = dists.get(st["owner_pid"])

            zone_violation = bool(ENABLE_ZONES and RESTRICTED_ZONES
                                  and bbox_in_zone(bb, RESTRICTED_ZONES))

            if state == ALARM:
                if alarm_manager.trigger(lid, now_t, {"lid": lid, "owner": st["owner_pid"]}):
                    evid = away_evidence(st)
                    events.append({
                        "luggage_track": lid,
                        "owner_track": st["owner_pid"],
                        "class": str(luggage_names.get(int(st["cls_stable"]), st["cls_stable"])),
                        "first_seen_sec": round(st["first_t"], 2),
                        "left_at_sec": round(st["away_since"], 2),
                        "alarm_at_sec": round(now_t, 2),
                        "alarm_frame": frame_count,
                        **evid,
                    })
                    print(f"  [ALARM] t={now_t:7.1f}s  L{lid}  owner P{st['owner_pid']}  "
                          f"unattended since {st['away_since']:.1f}s   "
                          f"bag drift {evid['bag_drift_w']:.1f}w   "
                          f"owner seen {evid['owner_seen_frac'] * 100:.0f}% of the window   "
                          f"confidence {evid['confidence']:.2f}"
                          f"{'  [REVIEW]' if evid['review'] else ''}"
                          f"{'  [scene object]' if evid['scene_object'] else ''}")
                    LOG.event("alarm", lid=lid, owner=st["owner_pid"],
                              name=str(luggage_names.get(int(st["cls_stable"]), st["cls_stable"])),
                              left_at_sec=round(st["away_since"], 2),
                              first_seen_sec=round(st["first_t"], 2),
                              unattended_s=round(st["unattended_s"], 2), bbox=bb, **evid)
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
            tag = f"{short_name(cname)}{lid}"

            label = f"{tag} {float(st['conf']):.2f}"
            if state in (UNATTENDED, ALARM):
                label += f"  {st['unattended_s']:.0f}s"
            # alarm plate stays red with black text even while the box flashes
            draw_label(annotated, label, x1, y1 - 6, COLOR_ALARM if state == ALARM else color)

            own = f"P{st['owner_pid']}" if st["owner_pid"] is not None else "-"
            progress = None
            if state == PENDING:
                detail = f"{now_t - st['first_t']:.0f}/{OWNERSHIP_SEC:.0f}s"
            elif state == OWNED:
                detail = f"{owner_d:.1f}h" if owner_d is not None else "hidden"
                if st.get("moving") and st.get("carrier_pid") is not None:
                    detail = f"carried P{st['carrier_pid']}"
            else:
                detail = f"{st['unattended_s']:.0f}s"
                progress = min(1.0, st["unattended_s"] / max(1e-6, UNATTENDED_SECONDS))
            if pred_flag:
                detail += " ?"
            if zone_violation:
                detail += " Z"
            track_rows.append({"tag": tag, "owner": own, "state": state, "detail": detail,
                               "progress": progress})

            if LOG_PAIRS_EVERY_N and frame_count % LOG_PAIRS_EVERY_N == 0:
                near3 = sorted(dists.items(), key=lambda kv: kv[1])[:3]
                LOG.event("pair", lid=lid, tag=tag, cls=cid, name=cname,
                          conf=round(float(st["conf"]), 3), bbox=bb, state=state,
                          owner=st["owner_pid"],
                          d_h=None if owner_d is None else round(owner_d, 3),
                          owner_visible=owner is not None,
                          owner_hidden_s=round(st["owner_missing_s"], 2),
                          age_s=round(now_t - st["first_t"], 2),
                          unattended_s=round(st["unattended_s"], 2),
                          predicted=pred_flag, zone=zone_violation,
                          standing=is_static(st, now_t), moving=bool(st.get("moving")),
                          held=is_held(st, now_t),
                          carrier=st.get("carrier_pid") if st.get("moving") else None,
                          evidence={str(p): [round(v, 2), round(st.get("carry", {}).get(p, 0.0), 2)]
                                    for p, v in sorted(st.get("score", {}).items(),
                                                       key=lambda kv: -kv[1])[:3]},
                          people_near={str(p): round(d, 2) for p, d in near3})

            # the bag/owner pairing this whole script exists for
            if owner is not None:
                bx, by = bottom_center(bb)
                ox, oy = bottom_center(owner["bbox"])
                cv2.line(annotated, (int(bx), int(by)), (int(ox), int(oy)), color, 2, cv2.LINE_AA)

        # -----------------------------
        # Sidebar (beside the video, never on top of it)
        # -----------------------------
        alarms_active = len(alarm_manager.active)
        flash = any(alarm_manager.is_flashing(i, now_t) for i in alarm_manager.active)
        LOG.event("frame", persons=len(persons), luggage=visible_luggage, owned=owned_now,
                  unattended=unattended_now, alarms=alarms_active,
                  person_tracks=len(person_tracks), luggage_tracks=len(luggage_tracks),
                  retired=len(retired_bags), person_dets=len(person_dets),
                  luggage_dets=len(luggage_dets),
                  person_dets_low=len(person_dets_low), luggage_dets_low=len(luggage_dets_low),
                  retired_persons=len(retired_persons), infer_ms=round(infer_ms, 1),
                  loop_fps=round(fps_smooth or 0.0, 2))
        if alarms_active:
            draw_alarm_border(annotated, flash)

        if side_w:
            canvas = np.full((canvas_h, canvas_w, 3), SIDEBAR_BG, np.uint8)
            canvas[video_y0:video_y0 + h, video_x0:video_x0 + w] = annotated
            stats = {"frame": frame_count, "frames": total_frames, "t": now_t,
                     "fps": fps_smooth or 0.0, "ms": infer_ms,
                     "persons": len(persons), "luggage": visible_luggage, "owned": owned_now,
                     "unattended": unattended_now, "alarms": alarms_active,
                     "alarms_total": len(alarm_manager.history)}
            # alarms first, then unattended, pending, owned
            order = {ALARM: 0, UNATTENDED: 1, PENDING: 2, OWNED: 3}
            track_rows.sort(key=lambda r: order.get(r["state"], 9))
            draw_sidebar(canvas, side_x0, side_w, stats, track_rows, luggage_names,
                         alarms_active, flash)
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
                path = os.path.join(run_dir, f"screenshot_{frame_count:06d}.jpg")
                cv2.imwrite(path, canvas)
                print(f"  screenshot -> {path}")

        if frame_count % 200 == 0:
            print(f"    frame {frame_count}/{total_frames or '?'}  t={now_t:.0f}s  "
                  f"tracked bags={len(luggage_tracks)}  alarms={len(alarm_manager.history)}")

    cap.release()
    if writer is not None:
        writer.release()
    if SHOW:  # a headless build has no window subsystem and raises here
        cv2.destroyAllWindows()

    LOG.mark(now_t, frame_count)
    alarmed_lids = {e["lid"] for e in alarm_manager.history}
    for lid, st in sorted({**retired_bags, **luggage_tracks}.items()):
        LOG.event("bag_summary", lid=lid, owner=st["owner_pid"], state=st["state"],
                  name=str(luggage_names.get(int(st["cls_stable"]), st["cls_stable"])),
                  first_t=round(st["first_t"], 2), last_seen_t=round(st["last_seen_t"], 2),
                  unattended_s=round(st["unattended_s"], 2),
                  away_since=None if st["away_since"] is None else round(st["away_since"], 2),
                  alarmed=lid in alarmed_lids, **away_evidence(st))
    LOG.event("end", frames=frame_count, alarms=len(events),
              person_ids_used=next_person_id - 1, luggage_ids_used=next_luggage_id - 1,
              event_counts=LOG.counts)
    LOG.close()

    with open(os.path.join(run_dir, EVENTS_JSON), "w") as f:
        json.dump({
            "source": VIDEO_IN,
            "model_profile": MODEL_PROFILE,
            "model_label": MODEL_LABEL,
            "person_model": person_weights,
            "luggage_model": luggage_weights,
            "script_version": SCRIPT_VERSION,
            "fps": video_fps,
            "frames": frame_count,
            "params": {"d_own": D_OWN, "d_away": D_AWAY, "ownership_sec": OWNERSHIP_SEC,
                       "owner_grace_sec": OWNER_GRACE_SEC,
                       "unattended_seconds": UNATTENDED_SECONDS},
            "events": events,
        }, f, indent=2)

    c = LOG.counts
    print(f"\n  {frame_count} frames, {len(events)} alarm(s) -> {run_dir}")
    print(f"  produced by v{SCRIPT_VERSION}  ({script_identity()['script']})")
    print(f"  video   : {out_video}")
    print("  id stability: "
          f"person ids={next_person_id - 1} "
          f"(reclaimed {c.get('track_reclaimed[person]', 0)}, "
          f"revived {c.get('track_revived[person]', 0)}, "
          f"dup spawns blocked {c.get('spawn_suppressed[person]', 0)}, "
          f"weak-box saves {c.get('low_conf_recovered[person]', 0)})  |  "
          f"luggage ids={next_luggage_id - 1} "
          f"(reclaimed {c.get('track_reclaimed[luggage]', 0)}, "
          f"revived {c.get('track_revived[luggage]', 0)}, "
          f"dup spawns blocked {c.get('spawn_suppressed[luggage]', 0)}, "
          f"weak-box saves {c.get('low_conf_recovered[luggage]', 0)})  |  "
          f"owner re-binds {c.get('owner_rebind', 0)}")
    print(f"  trace log: {LOG_JSONL}  ({LOG.n} records)  " +
          "  ".join(f"{k}={v}" for k, v in sorted(LOG.counts.items())))
    if events:
        print("  alarm evidence (nothing below changes when an alarm fires -- it grades them):")
        print("    bag        owner   left    alarm   drift  bag seen  owner seen   max dist"
              "   conf  owner chosen by (margin, carried)")
    for e in events:
        print(f"    L{e['luggage_track']:<3} {e['class']:<9} P{str(e['owner_track']):<5} "
              f"{e['left_at_sec']:6.1f}s {e['alarm_at_sec']:6.1f}s "
              f"{e['bag_drift_w']:5.1f}w {e['bag_seen_frac'] * 100:8.0f}% "
              f"{e['owner_seen_frac'] * 100:10.0f}%  "
              f"{'   n/a' if e['owner_d_max_h'] is None else '%6.1f h' % e['owner_d_max_h']}"
              f"  {e.get('confidence', 0):.2f}{'!' if e.get('review') else ' '} "
              f"  {e.get('owner_rule')} ({e.get('owner_margin')}, {e.get('owner_carry_s')}s)"
              f"{'  SCENE OBJECT' if e.get('scene_object') else ''}")
    if events:
        print("    conf = 0.30 still + 0.25 bag seen + 0.20 owner watched leaving + 0.25 owner"
              " certainty;  '!' = worth a look")


def _read_run_summary(run_dir):
    """alarms / ids of a finished run, from events.json and the last trace record."""
    out = {"run_dir": run_dir, "alarms": None, "first_alarm_s": None, "owner": None,
           "confidence": None, "person_ids": None, "luggage_ids": None}
    if not run_dir:
        return out
    try:
        with open(os.path.join(run_dir, EVENTS_JSON)) as f:
            ev = json.load(f).get("events", [])
        out["alarms"] = len(ev)
        if ev:
            first = min(ev, key=lambda e: e["alarm_at_sec"])
            out.update(first_alarm_s=first["alarm_at_sec"], owner=first.get("owner_track"),
                       confidence=first.get("confidence"))
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(os.path.join(run_dir, LOG_JSONL), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            for line in f.read().decode("utf-8", "ignore").splitlines()[::-1]:
                if '"end"' in line:
                    rec = json.loads(line)
                    out.update(person_ids=rec.get("person_ids_used"),
                               luggage_ids=rec.get("luggage_ids_used"))
                    break
    except (OSError, ValueError):
        pass
    return out


def run_batch(profiles, videos, show=False):
    """Every profile on every video, each in a fresh process, then one summary table + CSV."""
    import csv
    import glob
    import subprocess
    import sys as _sys
    script = os.path.abspath(__file__)
    # check every weight file and video BEFORE starting hours of runs
    missing = [f"{p}: {role} weights  {MODEL_PROFILES[p][role]}"
               for p in profiles for role in ("person", "luggage")
               if (os.sep in MODEL_PROFILES[p][role] or MODEL_PROFILES[p][role].startswith("SET_PATH"))
               and not os.path.exists(MODEL_PROFILES[p][role])]
    missing += [f"video  {v}" for v in videos if not os.path.exists(v)]
    if missing:
        print("\n  Not found -- fix the paths at the top of the script, nothing was run:")
        for m in missing:
            print(f"    {m}")
        raise SystemExit(2)
    print(f"\n  batch: {len(profiles)} model(s) x {len(videos)} video(s) = "
          f"{len(profiles) * len(videos)} runs", flush=True)
    started = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for video in videos:
        for prof in profiles:
            print(f"\n{'=' * 78}\n  {os.path.basename(video)}   x   {prof}\n{'=' * 78}", flush=True)
            cmd = [_sys.executable, script, "--model", prof, video] + ([] if show else ["--no-show"])
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            dirs = sorted(glob.glob(os.path.join(OUT_DIR, f"{_slug(video)}__{prof}__*")),
                          key=os.path.getmtime)
            run_dir = dirs[-1] if rc == 0 and dirs else None
            rows.append({"video": os.path.basename(video), "model": prof, "exit": rc,
                         "minutes": round((time.time() - t0) / 60, 1), **_read_run_summary(run_dir)})

    cols = ["video", "model", "exit", "alarms", "first_alarm_s", "owner", "confidence",
            "person_ids", "luggage_ids", "minutes", "run_dir"]
    csv_path = os.path.join(OUT_DIR, f"batch_{started}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'=' * 78}\n  BATCH SUMMARY  ({len(rows)} runs, script v{SCRIPT_VERSION})\n{'=' * 78}")
    print(f"  {'video':<20}{'model':<18}{'alarms':>7}{'first':>9}{'owner':>7}{'conf':>7}"
          f"{'p-ids':>7}{'l-ids':>7}")
    for r in rows:
        if r["exit"] != 0:
            print(f"  {r['video']:<20}{r['model']:<18}   FAILED (exit {r['exit']})")
            continue
        fa = "-" if r["first_alarm_s"] is None else f"{r['first_alarm_s']:.1f}s"
        cf = "-" if r["confidence"] is None else f"{r['confidence']:.2f}"
        print(f"  {r['video']:<20}{r['model']:<18}{str(r['alarms']):>7}{fa:>9}"
              f"{('P' + str(r['owner'])) if r['owner'] is not None else '-':>7}{cf:>7}"
              f"{str(r['person_ids']):>7}{str(r['luggage_ids']):>7}")
    print(f"\n  table saved to {csv_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unattended-luggage monitor")
    ap.add_argument("video", nargs="?", help="input video (default: VIDEO_IN in the script)")
    ap.add_argument("--model", choices=list(MODEL_PROFILES),
                    help=f"detector profile (default: {MODEL_PROFILE})")
    ap.add_argument("--all", action="store_true",
                    help="run every profile on every video in TEST_VIDEOS")
    ap.add_argument("--models", help="with --all: comma-separated subset of profiles")
    ap.add_argument("--videos", nargs="+", help="with --all: use these videos instead of TEST_VIDEOS")
    ap.add_argument("--show", action="store_true", help="with --all: show the window for every run")
    ap.add_argument("--no-show", action="store_true", help="do not open the preview window")
    args = ap.parse_args()

    import sys as _sys
    if args.all or (RUN_ALL_ON_START and len(_sys.argv) == 1):
        profs = args.models.split(",") if args.models else list(MODEL_PROFILES)
        unknown = [p for p in profs if p not in MODEL_PROFILES]
        if unknown:
            ap.error(f"unknown profile(s) {unknown}; choose from {list(MODEL_PROFILES)}")
        run_batch(profs, args.videos or TEST_VIDEOS, show=args.show)
    else:
        if args.video:
            VIDEO_IN = args.video
        if args.model:
            MODEL_PROFILE = args.model
        if args.no_show:
            SHOW = False
        run()