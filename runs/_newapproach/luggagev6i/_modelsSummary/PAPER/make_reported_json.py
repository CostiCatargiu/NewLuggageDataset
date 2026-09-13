#!/usr/bin/env python3
"""
make_reported_json.py — emit the two result files the paper actually reports.

Produces PAPER/reported/reported_v12.json and reported_v26.json, each holding
the four cells of the 2x2 (baseline / loss / arch / arch+loss) with every seed
kept separately, the aggregate over seeds, and the delta against that model's
own baseline.

All metric values are PERCENT (source fractions x 100). Run it from PAPER/.
"""
import json, glob, os, statistics as st, datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reported")
BOX  = "/home/constantin/Doctorat/YoloLib/runs/detect"

def load(model_dir):
    """name -> (metrics, per_class, project_dir)"""
    M = {}
    for f in glob.glob(os.path.join(ROOT, model_dir, "results", "*.json")):
        proj = os.path.basename(f).split("__")[0]
        for r in json.load(open(f))["results"]:
            M[r["name"]] = (r["metrics"], r.get("per_class", {}), proj)
    return M

def pct(d):  return {k: round(100.0 * v, 4) for k, v in d.items()}
def f1(m):   return round(200.0 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]), 4)

SPEC = {
 "v26": {
   "model": "YOLO26s", "weights": "yolo26s.pt", "dir": "MODEL_v26",
   "batch_uniform": 48,
   "noise_floor": {"metric": "mAP50_small", "sd": 0.428, "df": 12,
                   "groups": 13, "selection_bias": -0.55,
                   "se_run_vs_run": 0.605, "se_run_vs_3seed_mean": 0.494},
   "cells": [
     ("baseline", "stock YOLO26s, all mechanisms off",
      ["y26_base_rep", "y26_identity_s1", "y26_identity_s2"], 48),
     ("loss", "tal_beta 6.0 -> 1.0 (alignment exponent)",
      ["y26_b1", "y26_b1_s1"], 48),
     ("arch", "p2dys-p234rich: +P2 stride-4, -P5 detect, depth 2->4 at P2/P3",
      ["y26_p2dys_p234rich"], 48),
     ("arch_plus_loss", "p2dys-p234rich + tal_beta=1.0  [RECOMMENDED]",
      ["y26_p234rich_b1", "y26_p234rich_b1_s1", "y26_p234rich_b1_s2"], 48),
   ]},
 "v12": {
   "model": "YOLOv12s", "weights": "yolov12s.pt", "dir": "MODEL_v12",
   "batch_uniform": None,
   "noise_floor": {"metric": "mAP50_small", "sd": 0.428, "df": 12,
                   "note": "borrowed from YOLO26; v12 has one seed pair only",
                   "se_run_vs_run": 0.605},
   "batch_control": {"run": "v12_stock_b32", "batch": 32,
                     "purpose": "matched-batch control for the b32 arch cells"},
   "cells": [
     ("baseline", "stock YOLOv12s, all mechanisms off", ["yolov12s_default"], 54),
     ("loss", "curriculum area weighting, sqrt mode, alpha 0.7->0.3, boost 2.0 @48px",
      ["yolov12s_sqrt0703"], 54),
     ("arch", "ls_shift: +P2 stride-4 appended, DySample, gctx@P2, snake k=9 @P3-P5  [RECOMMENDED]",
      ["arch_ls_shift"], 32),
     ("arch_plus_loss", "ls_shift + curriculum weighting (NOT recommended: 3/9 metrics improve)",
      ["ls_shift_sqrt"], 32),
   ]},
}

for tag, spec in SPEC.items():
    M = load(spec["dir"])
    doc = {
      "model": spec["model"], "pretrained": spec["weights"],
      "dataset": "LuggageDataset.v6i", "split": "test_full_dataset",
      "test_images": 1219, "test_instances": 6172,
      "imgsz": 640, "epochs": 70, "close_mosaic": 10,
      "units": "percent (source fractions x 100)",
      "generated": datetime.date.today().isoformat(),
      "run_root_on_training_box": BOX,
      "noise_floor": spec["noise_floor"],
      "cells": {},
    }
    if "batch_control" in spec: doc["batch_control"] = spec["batch_control"]

    base_agg = None
    for cell, desc, names, batch in spec["cells"]:
        runs, keys = [], None
        for n in names:
            if n not in M: continue
            met, per_cls, proj = M[n]
            keys = keys or sorted(met)
            runs.append({
              "run": n, "batch": batch, "project_dir": proj,
              "path": f"{BOX}/{proj}/{n}",
              "metrics": pct(met) | {"F1": f1(met)},
              "per_class": {c: pct(v) for c, v in per_cls.items()},
            })
        agg = {}
        for k in (keys or []) :
            v = [r["metrics"][k] for r in runs]
            agg[k] = {"mean": round(st.mean(v), 4),
                      "sd": round(st.stdev(v), 4) if len(v) > 1 else None,
                      "n": len(v)}
        v = [r["metrics"]["F1"] for r in runs]
        agg["F1"] = {"mean": round(st.mean(v), 4),
                     "sd": round(st.stdev(v), 4) if len(v) > 1 else None, "n": len(v)}
        entry = {"description": desc, "batch": batch, "n_seeds": len(runs),
                 "runs": runs, "aggregate": agg}
        if cell == "baseline": base_agg = agg
        else:
            entry["delta_vs_baseline"] = {k: round(agg[k]["mean"] - base_agg[k]["mean"], 4) for k in agg}
            entry["pct_vs_baseline"]   = {k: round(100*(agg[k]["mean"]-base_agg[k]["mean"])/base_agg[k]["mean"], 4)
                                          for k in agg if base_agg[k]["mean"]}
        doc["cells"][cell] = entry

    # v12: batch-corrected view for the b32 cells
    if tag == "v12" and "v12_stock_b32" in M:
        cm, _, cproj = M["v12_stock_b32"]
        bm = doc["cells"]["baseline"]["aggregate"]
        corr = {k: round(100*cm[k] - bm[k]["mean"], 4) for k in cm}
        corr["F1"] = round(f1(cm) - bm["F1"]["mean"], 4)
        doc["batch_correction"] = {
          "definition": "v12_stock_b32 minus yolov12s_default; subtract from any batch-32 cell",
          "control_path": f"{BOX}/{cproj}/v12_stock_b32",
          "values": corr}
        for cell in ("arch", "arch_plus_loss"):
            e = doc["cells"][cell]
            e["aggregate_batch_corrected"] = {
              k: round(e["aggregate"][k]["mean"] - corr.get(k, 0.0), 4) for k in e["aggregate"]}
            e["pct_vs_baseline_batch_corrected"] = {
              k: round(100*(e["aggregate_batch_corrected"][k] - bm[k]["mean"])/bm[k]["mean"], 4)
              for k in e["aggregate"] if bm[k]["mean"]}

    p = os.path.join(OUT, f"reported_{tag}.json")
    json.dump(doc, open(p, "w"), indent=2)
    print(f"  {p}  ({os.path.getsize(p)/1024:.1f} KB)")
    for c, e in doc["cells"].items():
        sm = e["aggregate"]["mAP50_small"]
        print(f"     {c:16s} n={e['n_seeds']}  sm50 {sm['mean']:6.2f}" +
              (f" +/- {sm['sd']:.2f}" if sm["sd"] else "        ") +
              (f"   {e['pct_vs_baseline']['mAP50_small']:+.2f}%" if c != "baseline" else ""))
