#!/usr/bin/env python3
"""Read a run's trace.jsonl and show what the tracker actually did.

    python inspect_run.py <run_dir>                  summary + alarm evidence + anomalies
    python inspect_run.py <run_dir> --frames 936 958 frame-by-frame replay
    python inspect_run.py <run_dir> --track 2        the whole life of luggage track 2
    python inspect_run.py <run_dir> --track 4 --role person
    python inspect_run.py <run_dir> --swaps          every anomaly, in order

The frame replay needs the "assoc" records written from v2026-08-28.replayable onward; on
older traces it falls back to detections and track states and says so.
"""

import argparse
import collections
import json
import os
import signal
import sys

try:  # so piping into head/less does not raise on exit
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass


def load(run_dir):
    path = run_dir if run_dir.endswith(".jsonl") else os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        sys.exit("no trace at %s" % path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def by(records, kind):
    return [r for r in records if r["event"] == kind]


def head(records):
    r = by(records, "run")
    return r[0] if r else {}


def summary(R):
    h, e = head(R), (by(R, "end") or [{}])[-1]
    print("source   : %s" % h.get("source", "?"))
    print("script   : %s" % h.get("script", "(not recorded -- pre-banner run)"))
    print("version  : %s" % h.get("version", "(none)"))
    print("frames   : %s at %s fps   models p=%s l=%s"
          % (h.get("frames"), h.get("fps"), h.get("person_model"), h.get("luggage_model")))
    p = h.get("params", {})
    print("params   : conf p=%s l=%s  d_own=%s d_away=%s  unattended=%ss  grace=%ss"
          % (p.get("conf_person"), p.get("conf_luggage"), p.get("d_own"), p.get("d_away"),
             p.get("unattended_seconds"), p.get("owner_grace_sec")))
    c = e.get("event_counts", {})
    print("\nid stability")
    print("  person ids %-4s  revived %-4s  weak-box saves %-5s  spawns blocked %s"
          % (e.get("person_ids_used"), c.get("track_revived[person]", 0),
             c.get("low_conf_recovered[person]", 0), c.get("spawn_suppressed[person]", 0)))
    print("  luggage ids %-3s  revived %-4s  weak-box saves %-5s  spawns blocked %s"
          % (e.get("luggage_ids_used"), c.get("track_revived[luggage]", 0),
             c.get("low_conf_recovered[luggage]", 0), c.get("spawn_suppressed[luggage]", 0)))
    print("  merges %-4s  owner re-binds %-4s  ownership resets %-4s  suspicious matches %s"
          % (c.get("track_merged", 0), c.get("owner_rebind", 0),
             c.get("ownership_reset", 0), c.get("match_suspect", 0)))
    alarms(R)


def alarms(R):
    al = by(R, "alarm")
    print("\nalarms (%d)" % len(al))
    if not al:
        return
    graded = "bag_drift_w" in al[0]
    if graded:
        print("  bag  class      owner   left    alarm   drift  bag seen  owner seen  max dist")
        for a in al:
            print("  L%-3d %-10s P%-5s %6.1fs %6.1fs %5.1fw %8.0f%% %10.0f%%  %s"
                  % (a["lid"], a["name"], a["owner"], a["left_at_sec"], a["t"],
                     a["bag_drift_w"], a.get("bag_seen_frac", 0) * 100,
                     a["owner_seen_frac"] * 100,
                     "     n/a" if a["owner_d_max_h"] is None else "%6.1f h" % a["owner_d_max_h"]))
        print("\n  drift is in bag-widths from where the bag stood when its owner left;")
        print("  'bag seen' is how much of the countdown the bag was actually visible;")
        print("  'owner seen' near 0 means the owner left the scene OR the tracker lost them.")
    else:
        for a in al:
            print("  L%-3d %-10s owner P%-4s left %.1fs alarm %.1fs  (no evidence fields "
                  "-- older run)" % (a["lid"], a["name"], a["owner"], a["left_at_sec"], a["t"]))


def swaps(R):
    kinds = ("match_suspect", "ownership_reset", "track_merged", "track_revived",
             "owner_rebind", "owner_elected", "track_lost", "nms")
    rows = [r for r in R if r["event"] in kinds and r["event"] != "nms"]
    print("anomalies and identity events (%d)\n" % len(rows))
    for r in sorted(rows, key=lambda x: x["frame"]):
        k, t, f = r["event"], r["t"], r["frame"]
        if k == "match_suspect":
            ru = r.get("runner_up")
            print("  f%-6d %6.2fs  MATCH      %s %s took a box %.0f px away after %.2fs quiet"
                  "  cost %.2f%s" % (f, t, r.get("role", "?"), r["tid"], r["jump_px"],
                                     r["age_s"], r["cost"],
                                     "" if not ru else "   runner-up %s at %.2f"
                                     % (ru["tid"], ru["cost"])))
        elif k == "ownership_reset":
            print("  f%-6d %6.2fs  RESET      L%s jumped %.0f px after %.1fs quiet -- owner "
                  "P%s dropped, was %s" % (f, t, r["lid"], r["jump_px"], r["gap_s"],
                                           r.get("owner"), r.get("state")))
        elif k == "track_merged":
            print("  f%-6d %6.2fs  MERGE      keep L%s, drop L%s at iou %.2f (owners %s / %s)"
                  % (f, t, r["keep"], r["drop"], r["iou"], r.get("keep_owner"),
                     r.get("drop_owner")))
        elif k == "track_revived":
            print("  f%-6d %6.2fs  REVIVED    %s %s after %.1fs   iou %.2f sim %s"
                  % (f, t, r.get("role"), r.get("tid", r.get("lid")), r["gap_s"], r["iou"],
                     r.get("sim")))
        elif k == "owner_rebind":
            print("  f%-6d %6.2fs  RE-BIND    L%s  P%s -> P%s via %s (gap %.2fs, sim %s)"
                  % (f, t, r["lid"], r["old_owner"], r["new_owner"], r.get("via", "?"),
                     r["gap_s"], r.get("sim")))
        elif k == "owner_elected":
            n = r.get("votes_n")
            weak = "" if n is None else ("" if n >= 5 else "   <-- ON %d FRAME(S)" % n)
            print("  f%-6d %6.2fs  ELECTED    L%s -> P%s  mean %.3f on %s frame(s)  [%s]%s"
                  % (f, t, r["lid"], r["owner"], r["mean_h"], n, r.get("rule", "nearest"), weak))
        elif k == "track_lost":
            print("  f%-6d %6.2fs  LOST       %s %s after %.1fs alive"
                  % (f, t, r.get("role"), r.get("tid", r.get("lid")), r.get("alive_s", 0)))


def frames(R, a, b, role):
    assoc = {r["frame"]: r for r in R if r["event"] == "assoc" and r.get("role") == role}
    pairs = collections.defaultdict(list)
    for r in by(R, "pair"):
        pairs[r["frame"]].append(r)
    if not assoc:
        print("(this trace has no 'assoc' records -- showing track states only; re-run with"
              " v2026-08-28.replayable or later for the full association)\n")
    for f in range(a, b + 1):
        rec = assoc.get(f)
        line = "f%-6d" % f
        if rec:
            ds = " ".join("[%d](%.0f,%.0f)%.2f" % (i, d[0], d[1], d[2])
                          for i, d in enumerate(rec["dets"]))
            wk = " ".join("(%.0f,%.0f)%.2f*" % (d[0], d[1], d[2]) for d in rec["dets_weak"])
            print("%s  dets %s %s" % (line, ds or "-none-", wk))
            for m in rec["m"]:
                jump = "" if m["jump"] is None else "  moved %.0f px" % m["jump"]
                big = "   <-- JUMP" if (m["jump"] or 0) > 100 else ""
                print("           det[%s] -> %s  %s  %s -> %s%s%s"
                      % (m["d"], m["t"], m["p"], m["from"], m["to"], jump, big))
            if rec["spawned_from"]:
                print("           unmatched dets (new tracks or suppressed): %s"
                      % rec["spawned_from"])
            if rec["coasting"]:
                print("           coasting (no detection this frame): %s" % rec["coasting"])
        elif f in pairs:
            print("%s  %s" % (line, "  ".join(
                "L%s(%.0f,%.0f)%.2f %s" % (p["lid"], (p["bbox"][0] + p["bbox"][2]) / 2,
                                           (p["bbox"][1] + p["bbox"][3]) / 2, p["conf"],
                                           p["state"][:4]) for p in pairs[f])))


def track(R, tid, role):
    key = "lid" if role == "luggage" else "tid"
    print("life of %s track %s\n" % (role, tid))
    run_ = []

    def flush(acc):
        if not acc:
            return
        print("  f%-6d %6.2fs  %-18s held by weak boxes for %d frame(s), to f%d (conf %.2f-%.2f)"
              % (acc[0]["frame"], acc[0]["t"], "low_conf_recovered", len(acc),
                 acc[-1]["frame"], min(x["conf"] for x in acc), max(x["conf"] for x in acc)))
        acc.clear()

    for r in R:
        if r["event"] in ("assoc", "pair", "frame", "det", "run", "end", "nms"):
            continue
        # ownership events carry "lid" and no role; per-track events carry "tid" and a role
        mine = (r.get(key) == tid or (r.get("role") == role and r.get("tid") == tid))
        if not (mine and r.get("role", role) == role):
            continue
        # a weak-box rescue every frame is noise; report the stretch, not each frame
        if r["event"] == "low_conf_recovered":
            run_.append(r)
            continue
        flush(run_)
        extra = {k: v for k, v in r.items()
                 if k not in ("t", "frame", "event", "role", key, "bbox")}
        print("  f%-6d %6.2fs  %-18s %s" % (r["frame"], r["t"], r["event"], extra))
    flush(run_)
    if role == "luggage":
        pr = [p for p in by(R, "pair") if p["lid"] == tid]
        if pr:
            print("\n  seen in %d frames, t=%.1f..%.1f" % (len(pr), pr[0]["t"], pr[-1]["t"]))
            own, last = [], object()
            for p in pr:
                if p["owner"] != last:
                    own.append((p["t"], p["owner"]))
                    last = p["owner"]
            print("  owner history: %s"
                  % " -> ".join("P%s@%.0fs" % (o, t) for t, o in own))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("FROM", "TO"))
    ap.add_argument("--track", type=int)
    ap.add_argument("--role", default="luggage", choices=("luggage", "person"))
    ap.add_argument("--swaps", action="store_true")
    ap.add_argument("--alarms", action="store_true")
    args = ap.parse_args()
    R = load(args.run_dir)
    if args.frames:
        frames(R, args.frames[0], args.frames[1], args.role)
    elif args.track is not None:
        track(R, args.track, args.role)
    elif args.swaps:
        swaps(R)
    elif args.alarms:
        alarms(R)
    else:
        summary(R)


if __name__ == "__main__":
    main()
