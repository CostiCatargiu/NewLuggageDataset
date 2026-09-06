#!/usr/bin/env python3
"""
Batch COCO evaluation over multiple phase folders of YOLO training runs.
v6i EDITION — see "CHANGES" below.

Runs evaluation for every model against the configured split(s) and writes a
JSON per phase folder per split:

    <phase_folder>/<phase_folder>__<output_suffix>.json

Optionally a single combined JSON aggregating everything (COMBINED_OUTPUT_JSON).

=============================================================================
CHANGES FROM THE PREVIOUS VERSION
=============================================================================
1. PARAMS ARE FOUND AGAIN.  load_run_params() looked ONLY for
   "ablation_params.json". The newer runners write different filenames:
       run_model_scale.py            -> scale_params.json
       run_custom_v3_ablation*.py    -> v3_params.json
       run_posboost.py               -> v3_params.json
       run_combo_sqrt_nwd_entropy.py -> combo_params.json
       run_newluggage_*.py           -> ablation_params.json
   That is why runs_newl_luggagev6i__test_full_dataset.json came out with
   "params": {} for BOTH yolov12s_default and yolov12s_sqrt0703 — the eval
   could not find the file, warned, and carried on with an empty dict.
   Now every known filename is searched, in order.

2. THE FULL RUN META IS CARRIED THROUGH, not just the hyp dict. Each entry
   now gets a "run_meta" block with model / batch / seed / epochs / imgsz /
   close_mosaic / hours / loss_file{path,md5}. The loss-file md5 is the field
   that proves two runs used the same code — the check that would have caught
   r0a2_dflboost, shape-TAL, NWD C=64 and the other silent no-ops.
   => the eval JSON becomes self-contained for the report.

3. split= IS NO LONGER HARDCODED.  model.val() had split="test" pinned while
   the script ALSO patched the YAML's val: field. With one split configured
   that was merely redundant; the moment a "valid" split is re-enabled it
   becomes a silent bug — every pass would evaluate TEST regardless of the
   patched YAML, and the output would be written under the valid_* filename.
   split is now taken from the split config.

4. imgsz=img_size instead of a hardcoded literal. The old copies in
   luggagev5i/ and _newapproach/new_try/ still carry imgsz=896 — the "896
   lesson" — while the function was being passed img_size and ignoring it.

=============================================================================
SIZE-BUCKET THRESHOLDS — read before quoting small/medium/large anywhere
=============================================================================
This script buckets by COCO **AREA**: small < SMALL_THRESH^2,
large > LARGE_THRESH^2. With 48/128 that is area < 2304 and area > 16384 px^2.

The dataset report and diag_anchor_footprint_v6i.py bucket by **MAX SIDE**
with 48/96. These are DIFFERENT definitions and they will not agree.

They are left as-is here deliberately: every historical eval JSON (v5i and
v6i) used 48/128 by area, so changing them now would break comparability with
~90 existing results. But say which definition you mean in the paper, because
"small" means two different things in the two artefacts.

For context on v6i: mean object is 39 x 55 = 2145 px^2, just BELOW the 2304
small threshold, so slightly over half of all instances land in "small" by
area. "large" (>16384 px^2, e.g. 128x128) is rare — which is why the large
bucket is the noisiest column in every results table.
"""

import json
import re
import shutil
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

# =============================================================================
# EDIT THESE
# =============================================================================
RUNS_ROOT = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12"
PHASE_FOLDER_PATTERN = r"runs_v12_p234rich_transfer_v6i"

DATA_YAMLDS1 = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
DATA_YAMLfull = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"

EVAL_SPLITS = {
    "test_full_dataset": {
        "coco_ann": "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/annotations_coco_test.json",
        "yaml_val_path": "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/test/images",
        "data_yaml": DATA_YAMLfull,
        "output_suffix": "test_full_dataset",
        "ultra_split": "test",       # <- CHANGE 3: which split model.val() reads
    },
    # "valid_full_dataset": {
    #     "coco_ann": ".../annotations_coco_valid.json",
    #     "yaml_val_path": ".../valid/images",
    #     "data_yaml": DATA_YAMLfull,
    #     "output_suffix": "valid_full_dataset",
    #     "ultra_split": "val",
    # },
}

IMG_SIZE = 640
SKIP_DONE = True

# COCO AREA thresholds. See the header note — these are NOT the max-side 48/96
# used by the dataset report and the footprint diagnostic.
SMALL_THRESH = 48
LARGE_THRESH = 128

COMBINED_OUTPUT_JSON = RUNS_ROOT + "/all_results.json"

# CHANGE 1: every params filename any runner in this project has ever written,
# searched in this order. Add new ones here rather than renaming outputs.
PARAMS_FILENAMES = (
    "scale_params.json",       # run_model_scale.py
    "v3_params.json",          # run_custom_v3_ablation*.py, run_posboost.py
    "ablation_params.json",    # run_newluggage_*.py
    "combo_params.json",       # run_combo_sqrt_nwd_entropy.py
)


# =============================================================================
# YAML PATCHING (val: field swap)
# =============================================================================
@contextmanager
def patch_yaml_val(data_yaml: str, new_val_path: str):
    """Temporarily rewrite the `val:` field of the data.yaml, then restore."""
    yaml_path = Path(data_yaml)
    backup_path = yaml_path.with_suffix(yaml_path.suffix + ".bak")
    with open(yaml_path) as f:
        original_text = f.read()
    shutil.copy2(yaml_path, backup_path)
    try:
        data = yaml.safe_load(original_text) or {}
        original_val = data.get("val")
        data["val"] = new_val_path
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"  [YAML] val: {original_val!r} -> {new_val_path!r}")
        yield
    finally:
        with open(yaml_path, "w") as f:
            f.write(original_text)
        print(f"  [YAML] restored original val: field")


# =============================================================================
# TRAINING-PARAMS READER  (CHANGE 1 + 2)
# =============================================================================
def load_run_meta(run_dir: Path) -> tuple[dict, dict]:
    """Read the ground-truth record written by whichever training script ran.

    Returns (params, meta):
      params -> the hyp dict actually passed to model.train()
      meta   -> everything else: model, batch, seed, epochs, imgsz,
                close_mosaic, hours, loss_file{path, md5, ...}

    Searches every filename in PARAMS_FILENAMES. Returns ({}, {}) with a loud
    warning if none is found, so evaluation still proceeds — but a run with no
    params in the output JSON is a run whose config you cannot reconstruct, so
    treat that warning as an error in practice.
    """
    for fname in PARAMS_FILENAMES:
        pfile = run_dir / fname
        if not pfile.exists():
            continue
        try:
            with open(pfile) as f:
                data = json.load(f) or {}
        except Exception as e:
            print(f"  [WARN] could not read {pfile}: {e}")
            continue

        params = data.get("params", {})
        if not isinstance(params, dict):
            print(f"  [WARN] 'params' in {pfile} is not a dict — ignoring.")
            params = {}

        meta = {k: v for k, v in data.items() if k != "params"}
        meta["_params_file"] = fname
        print(f"  [OK] params from {fname}: {len(params)} key(s)"
              + (f", loss md5 {meta['loss_file'].get('md5')}"
                 if isinstance(meta.get("loss_file"), dict) else ""))
        return params, meta

    print(f"  [WARN] none of {PARAMS_FILENAMES} found in {run_dir.name} "
          f"— params AND meta will be empty for this run. Its configuration "
          f"cannot be reconstructed from the eval JSON.")
    return {}, {}


# =============================================================================
# PRECISION/RECALL @ BEST-F1 HELPERS
# =============================================================================
def _pr_best_f1_from_curve(pr_curve_1d: np.ndarray,
                           recall_pts: np.ndarray) -> tuple[float, float]:
    """Walk a 1-D precision array over the 101 COCO recall points and return
    (precision, recall) at the best-F1 point. (-1, -1) if no valid points."""
    valid = pr_curve_1d > -1
    if not valid.any():
        return -1.0, -1.0
    p_v = pr_curve_1d[valid]
    r_v = recall_pts[valid]
    f1 = 2 * p_v * r_v / (p_v + r_v + 1e-10)
    idx = int(f1.argmax())
    return float(p_v[idx]), float(r_v[idx])


def _best_f1_precision_recall_mean(precision_tensor: np.ndarray,
                                   iou_idx: int,
                                   area_idx: int) -> tuple[float, float]:
    """(precision, recall) at best-F1, averaged across classes.
    precision_tensor shape: [T, R, K, A, M] (COCO standard)."""
    pr_slice = precision_tensor[iou_idx, :, :, area_idx, -1]  # [R, K]
    recall_pts = np.linspace(0, 1, pr_slice.shape[0])
    ps, rs = [], []
    for k in range(pr_slice.shape[1]):
        p, r = _pr_best_f1_from_curve(pr_slice[:, k], recall_pts)
        if p >= 0:
            ps.append(p)
            rs.append(r)
    if not ps:
        return -1.0, -1.0
    return float(np.mean(ps)), float(np.mean(rs))


# =============================================================================
# COCO EVALUATION (size-specific + per-class)
# =============================================================================
def run_coco_evaluation(pred_json: Path, ann_file: str,
                        small_thresh: int = 32,
                        large_thresh: int = 96) -> tuple[dict, dict]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(cat_ids)}
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    print(f"\n[INFO] Categories found: {cat_id_to_name}")

    filename_to_id = {}
    for img_id, img_info in coco_gt.imgs.items():
        fname = img_info["file_name"]
        stem = Path(fname).stem
        clean = stem.split(".rf")[0]
        filename_to_id[fname] = img_id
        filename_to_id[stem] = img_id
        filename_to_id[clean] = img_id
        filename_to_id[clean + ".jpg"] = img_id

    with open(pred_json) as f:
        preds = json.load(f)

    converted, skipped = [], 0
    for p in preds:
        img_id = p["image_id"]
        if isinstance(img_id, str):
            stem = Path(img_id).stem
            clean = stem.split(".rf")[0]
            key = Path(img_id).name
            resolved = (filename_to_id.get(key) or filename_to_id.get(stem)
                        or filename_to_id.get(clean)
                        or filename_to_id.get(clean + ".jpg"))
            if resolved is None:
                skipped += 1
                continue
            p["image_id"] = resolved
        p["category_id"] = int(p["category_id"])
        converted.append(p)

    if skipped:
        print(f"[WARN] {skipped} prediction(s) could not be matched to a COCO image.")
    if not converted:
        raise RuntimeError("No predictions matched any COCO image.")

    fixed_json = pred_json.parent / "predictions_fixed.json"
    with open(fixed_json, "w") as f:
        json.dump(converted, f)
    coco_dt = coco_gt.loadRes(str(fixed_json))

    area_ranges = [
        [0 ** 2, 1e5 ** 2],
        [0 ** 2, small_thresh ** 2],
        [small_thresh ** 2, large_thresh ** 2],
        [large_thresh ** 2, 1e5 ** 2],
    ]
    area_labels = ["all", "small", "medium", "large"]

    metrics = {}
    per_class_metrics = {}

    print("\n" + "─" * 50)
    print("COCO Evaluation @ IoU=0.50 (size + class breakdown)")
    print("─" * 50)
    coco_eval_50 = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_50.params.iouThrs = np.array([0.5])
    coco_eval_50.params.areaRng = area_ranges
    coco_eval_50.params.areaRngLbl = area_labels
    coco_eval_50.params.catIds = cat_ids
    coco_eval_50.evaluate()
    coco_eval_50.accumulate()
    coco_eval_50.summarize()

    precision_50 = coco_eval_50.eval['precision']
    recall_50 = coco_eval_50.eval['recall']
    recall_pts_50 = np.linspace(0, 1, precision_50.shape[1])

    for area_idx, area_name in enumerate(area_labels):
        if area_name == "all":
            continue
        ap_values = precision_50[0, :, :, area_idx, -1]
        ap_values = ap_values[ap_values > -1]
        metrics[f"mAP50_{area_name}"] = float(np.mean(ap_values)) if len(ap_values) else -1.0

        ar_values = recall_50[0, :, area_idx, -1]
        ar_values = ar_values[ar_values > -1]
        metrics[f"AR50_{area_name}"] = float(np.mean(ar_values)) if len(ar_values) else -1.0

        p_bf1, r_bf1 = _best_f1_precision_recall_mean(precision_50, 0, area_idx)
        metrics[f"P50_{area_name}"] = p_bf1
        metrics[f"R50_{area_name}"] = r_bf1

    for cat_id in cat_ids:
        cat_name = cat_id_to_name[cat_id]
        cat_idx = cat_id_to_idx[cat_id]
        per_class_metrics.setdefault(cat_name, {})
        for area_idx, area_name in enumerate(area_labels):
            ap_values = precision_50[0, :, cat_idx, area_idx, -1]
            ap_values = ap_values[ap_values > -1]
            per_class_metrics[cat_name][f"AP50_{area_name}"] = (
                float(np.mean(ap_values)) if len(ap_values) else -1.0)

            ar_value = recall_50[0, cat_idx, area_idx, -1]
            per_class_metrics[cat_name][f"AR50_{area_name}"] = (
                float(ar_value) if ar_value > -1 else -1.0)

            pr_curve = precision_50[0, :, cat_idx, area_idx, -1]
            p_bf1, r_bf1 = _pr_best_f1_from_curve(pr_curve, recall_pts_50)
            per_class_metrics[cat_name][f"P50_{area_name}"] = p_bf1
            per_class_metrics[cat_name][f"R50_{area_name}"] = r_bf1

    print("\n" + "─" * 50)
    print("COCO Evaluation @ IoU=0.50:0.95 (size + class breakdown)")
    print("─" * 50)
    coco_eval_5095 = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_5095.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval_5095.params.areaRng = area_ranges
    coco_eval_5095.params.areaRngLbl = area_labels
    coco_eval_5095.params.catIds = cat_ids
    coco_eval_5095.evaluate()
    coco_eval_5095.accumulate()
    coco_eval_5095.summarize()

    precision_5095 = coco_eval_5095.eval['precision']
    recall_5095 = coco_eval_5095.eval['recall']
    recall_pts_5095 = np.linspace(0, 1, precision_5095.shape[1])

    for area_idx, area_name in enumerate(area_labels):
        if area_name == "all":
            continue
        ap_values = precision_5095[:, :, :, area_idx, -1]
        ap_values = ap_values[ap_values > -1]
        metrics[f"mAP50_95_{area_name}"] = float(np.mean(ap_values)) if len(ap_values) else -1.0

        ar_values = recall_5095[:, :, area_idx, -1]
        ar_values = ar_values[ar_values > -1]
        metrics[f"AR50_95_{area_name}"] = float(np.mean(ar_values)) if len(ar_values) else -1.0

        ps, rs = [], []
        for t in range(precision_5095.shape[0]):
            p_t, r_t = _best_f1_precision_recall_mean(precision_5095, t, area_idx)
            if p_t >= 0:
                ps.append(p_t)
                rs.append(r_t)
        metrics[f"P50_95_{area_name}"] = float(np.mean(ps)) if ps else -1.0
        metrics[f"R50_95_{area_name}"] = float(np.mean(rs)) if rs else -1.0

    for cat_id in cat_ids:
        cat_name = cat_id_to_name[cat_id]
        cat_idx = cat_id_to_idx[cat_id]
        for area_idx, area_name in enumerate(area_labels):
            ap_values = precision_5095[:, :, cat_idx, area_idx, -1]
            ap_values = ap_values[ap_values > -1]
            per_class_metrics[cat_name][f"AP50_95_{area_name}"] = (
                float(np.mean(ap_values)) if len(ap_values) else -1.0)

            ar_values = recall_5095[:, cat_idx, area_idx, -1]
            ar_values = ar_values[ar_values > -1]
            per_class_metrics[cat_name][f"AR50_95_{area_name}"] = (
                float(np.mean(ar_values)) if len(ar_values) else -1.0)

            cps, crs = [], []
            for t in range(precision_5095.shape[0]):
                pr_curve = precision_5095[t, :, cat_idx, area_idx, -1]
                p_bf1, r_bf1 = _pr_best_f1_from_curve(pr_curve, recall_pts_5095)
                if p_bf1 >= 0:
                    cps.append(p_bf1)
                    crs.append(r_bf1)
            per_class_metrics[cat_name][f"P50_95_{area_name}"] = (
                float(np.mean(cps)) if cps else -1.0)
            per_class_metrics[cat_name][f"R50_95_{area_name}"] = (
                float(np.mean(crs)) if crs else -1.0)

    print("\n" + "─" * 100)
    print("Per-Class × Size Breakdown")
    print("─" * 100)
    print(f"{'Class':<12} │ {'AP50':<6} │ {'AP50_S':<6} │ {'AP50_M':<6} │ {'AP50_L':<6} │ "
          f"{'AP5095':<6} │ {'AP5095_S':<8} │ {'AP5095_M':<8} │ {'AP5095_L':<8}")
    print("─" * 100)
    for cat_name, cm in per_class_metrics.items():
        print(f"{cat_name:<12} │ "
              f"{cm.get('AP50_all', -1):>6.3f} │ {cm.get('AP50_small', -1):>6.3f} │ "
              f"{cm.get('AP50_medium', -1):>6.3f} │ {cm.get('AP50_large', -1):>6.3f} │ "
              f"{cm.get('AP50_95_all', -1):>6.3f} │ {cm.get('AP50_95_small', -1):>8.3f} │ "
              f"{cm.get('AP50_95_medium', -1):>8.3f} │ {cm.get('AP50_95_large', -1):>8.3f}")
    print("─" * 100)

    print("\n" + "─" * 80)
    print("Per-Size Precision/Recall @ best-F1")
    print("─" * 80)
    print(f"{'Size':<8} │ {'P50':>7} │ {'R50':>7} │ {'P50:95':>8} │ {'R50:95':>8} │ "
          f"{'AR50':>7} │ {'AR50:95':>8}")
    print("─" * 80)
    for area_name in ["small", "medium", "large"]:
        print(f"{area_name:<8} │ {metrics.get(f'P50_{area_name}', -1):>7.3f} │ "
              f"{metrics.get(f'R50_{area_name}', -1):>7.3f} │ "
              f"{metrics.get(f'P50_95_{area_name}', -1):>8.3f} │ "
              f"{metrics.get(f'R50_95_{area_name}', -1):>8.3f} │ "
              f"{metrics.get(f'AR50_{area_name}', -1):>7.3f} │ "
              f"{metrics.get(f'AR50_95_{area_name}', -1):>8.3f}")
    print("─" * 80)

    return metrics, per_class_metrics


def get_ultralytics_metrics(val_results) -> dict:
    try:
        box = val_results.box
        return {
            "mAP50_all": float(box.map50),
            "mAP50_95_all": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
    except Exception as e:
        print(f"[WARN] Could not extract Ultralytics metrics: {e}")
        return {"mAP50_all": None, "mAP50_95_all": None,
                "precision": None, "recall": None}


def get_ultralytics_per_class_metrics(val_results, class_names: list) -> dict:
    per_class = {}
    try:
        box = val_results.box
        for idx, name in enumerate(class_names):
            per_class[name] = {
                "AP50_all_ultralytics": float(box.ap50[idx]) if idx < len(box.ap50) else None,
                "AP50_95_all_ultralytics": float(box.ap[idx]) if idx < len(box.ap) else None,
                "precision_ultralytics": float(box.p[idx]) if idx < len(box.p) else None,
                "recall_ultralytics": float(box.r[idx]) if idx < len(box.r) else None,
            }
    except Exception as e:
        print(f"[WARN] Could not extract per-class Ultralytics metrics: {e}")
    return per_class


# =============================================================================
# PER-RUN EVALUATION
# =============================================================================
def eval_run(run_dir: Path, data_yaml: str, coco_ann: str,
             img_size: int, skip_done: bool, done_names: set,
             split_name: str, ultra_split: str) -> dict | None:
    name = run_dir.name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"best.pt not found: {weights}")
    if skip_done and name in done_names:
        print(f"  [SKIP] {name} already in {split_name} results.")
        return None

    print(f"\n{'=' * 70}")
    print(f"  RUN    : {name}")
    print(f"  SPLIT  : {split_name}  (ultralytics split='{ultra_split}')")
    print(f"{'=' * 70}")

    # CHANGE 1 + 2: params AND the full run meta
    params, run_meta = load_run_meta(run_dir)

    from ultralytics import YOLO
    model = YOLO(str(weights))
    val_results = model.val(
        data=data_yaml,
        imgsz=img_size,          # CHANGE 4: was a hardcoded literal
        save_json=True,
        verbose=True,
        split=ultra_split,       # CHANGE 3: was hardcoded "test"
    )

    save_dir = Path(val_results.save_dir)
    pred_json = save_dir / "predictions.json"
    if not pred_json.exists():
        raise FileNotFoundError(f"predictions.json not found in {save_dir}")

    class_names = list(model.names.values()) if hasattr(model, 'names') else []
    ul_metrics = get_ultralytics_metrics(val_results)
    ul_per_class = get_ultralytics_per_class_metrics(val_results, class_names)
    coco_metrics, coco_per_class = run_coco_evaluation(
        pred_json, coco_ann, SMALL_THRESH, LARGE_THRESH)

    per_class_final = {}
    for cls_name in sorted(set(coco_per_class) | set(ul_per_class)):
        entry = {}
        if cls_name in ul_per_class:
            u = ul_per_class[cls_name]
            entry["AP50_all"] = u.get("AP50_all_ultralytics")
            entry["AP50_95_all"] = u.get("AP50_95_all_ultralytics")
            entry["precision"] = u.get("precision_ultralytics")
            entry["recall"] = u.get("recall_ultralytics")
        entry.setdefault("precision", None)
        entry.setdefault("recall", None)
        if cls_name in coco_per_class:
            c = coco_per_class[cls_name]
            for k in ("AP50_small", "AP50_medium", "AP50_large",
                      "AP50_95_small", "AP50_95_medium", "AP50_95_large",
                      "AR50_all", "AR50_small", "AR50_medium", "AR50_large",
                      "AR50_95_all", "AR50_95_small", "AR50_95_medium", "AR50_95_large",
                      "P50_small", "P50_medium", "P50_large",
                      "R50_small", "R50_medium", "R50_large",
                      "P50_95_small", "P50_95_medium", "P50_95_large",
                      "R50_95_small", "R50_95_medium", "R50_95_large"):
                entry[k] = c.get(k)
        per_class_final[cls_name] = entry

    metrics = {**ul_metrics, **coco_metrics}
    ordered_metrics = {k: metrics.get(k) for k in (
        "mAP50_all", "mAP50_95_all", "precision", "recall",
        "mAP50_small", "mAP50_medium", "mAP50_large",
        "mAP50_95_small", "mAP50_95_medium", "mAP50_95_large",
        "AR50_small", "AR50_medium", "AR50_large",
        "AR50_95_small", "AR50_95_medium", "AR50_95_large",
        "P50_small", "P50_medium", "P50_large",
        "P50_95_small", "P50_95_medium", "P50_95_large",
        "R50_small", "R50_medium", "R50_large",
        "R50_95_small", "R50_95_medium", "R50_95_large",
    )}

    print("\n" + "─" * 70)
    print(f"{'Class':<12} │ {'Precision':>9} │ {'Recall':>7} │ {'AP50':>6} │ {'AP50_95':>7}")
    print("─" * 70)
    for cls_name, m in per_class_final.items():
        def _f(v, w, p=3):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}"
        print(f"{cls_name:<12} │ {_f(m.get('precision'), 9)} │ {_f(m.get('recall'), 7)} │ "
              f"{_f(m.get('AP50_all'), 6)} │ {_f(m.get('AP50_95_all'), 7)}")
    print("─" * 70)

    return {
        "name": name,
        "split": split_name,
        "params": params,
        "run_meta": run_meta,          # CHANGE 2
        "metrics": ordered_metrics,
        "per_class": per_class_final,
    }


# =============================================================================
# PROCESS ONE PHASE FOLDER ON ONE SPLIT
# =============================================================================
def process_phase_folder_on_split(phase_folder: Path, split_name: str,
                                  coco_ann: str, data_yaml: str, img_size: int,
                                  skip_done: bool, output_suffix: str = None,
                                  ultra_split: str = "test"):
    phase_name = phase_folder.name
    suffix = output_suffix or split_name
    output_json = phase_folder / f"{phase_name}__{suffix}.json"

    print(f"\n{'#' * 80}")
    print(f"# PHASE FOLDER : {phase_name}")
    print(f"# SPLIT        : {split_name}  (ultralytics split='{ultra_split}')")
    print(f"# COCO ANN     : {coco_ann}")
    print(f"# OUTPUT       : {output_json}")
    print(f"{'#' * 80}")

    existing: list = []
    if output_json.exists():
        try:
            with open(output_json) as f:
                data = json.load(f)
                existing = data.get("results", data) if isinstance(data, dict) else data
            print(f"[INFO] Loaded {len(existing)} existing results from {output_json}")
        except Exception:
            print("[WARN] Could not load existing results — starting fresh.")

    done_names = {r["name"] for r in existing if "name" in r}
    run_dirs = sorted([d for d in phase_folder.iterdir()
                       if d.is_dir() and (d / "weights" / "best.pt").exists()])
    print(f"[INFO] {len(run_dirs)} run(s) with best.pt in {phase_name}")
    if not run_dirs:
        return 0, 0, list(existing), []

    print("\n[INFO] Runs to process:")
    for rd in run_dirs:
        print(f"  {'[DONE]' if rd.name in done_names and skip_done else '[TODO]'} {rd.name}")
    print()

    results, failed = list(existing), []
    for run_dir in run_dirs:
        try:
            entry = eval_run(run_dir, data_yaml, coco_ann, img_size,
                             skip_done, done_names, split_name, ultra_split)
            if entry is None:
                continue
            results.append(entry)
            done_names.add(entry["name"])
            with open(output_json, "w") as f:
                json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)
            print(f"  [OK]  Saved -> {output_json}")
        except Exception as e:
            print(f"\n  [FAILED] {run_dir.name}\n  {e}")
            traceback.print_exc()
            failed.append({"name": run_dir.name, "error": str(e)})
            with open(output_json, "w") as f:
                json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)

    with open(output_json, "w") as f:
        json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)

    num_new = len(results) - len(existing)
    print(f"\n[INFO] {phase_name}/{split_name}: {num_new} new, {len(failed)} failed, "
          f"total {len(results)}")

    # CHANGE 2: loud summary of which runs have no reconstructable config
    no_params = [r["name"] for r in results if not r.get("params")]
    if no_params:
        print(f"[WARN] {len(no_params)} run(s) have EMPTY params — their config "
              f"cannot be reconstructed from this JSON: {no_params}")

    return num_new, len(failed), results, failed


# =============================================================================
# COMBINED OUTPUT WRITER
# =============================================================================
def write_combined_json(combined_path: Path, runs_root: Path, by_phase_split: dict) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    flat_results, flat_failed = [], []
    for phase_name, splits in by_phase_split.items():
        for split_name, data in splits.items():
            for r in data.get("results", []):
                rc = dict(r); rc["phase_folder"] = phase_name; rc["split"] = split_name
                flat_results.append(rc)
            for fl in data.get("failed", []):
                fc = dict(fl); fc["phase_folder"] = phase_name; fc["split"] = split_name
                flat_failed.append(fc)

    payload = {
        "metadata": {
            "runs_root": str(runs_root),
            "phase_folder_pattern": PHASE_FOLDER_PATTERN,
            "data_yamls": {"ds1": DATA_YAMLDS1, "full": DATA_YAMLfull},
            "splits": {k: {"coco_ann": v["coco_ann"],
                           "yaml_val_path": v["yaml_val_path"],
                           "data_yaml": v["data_yaml"],
                           "ultra_split": v.get("ultra_split", "test")}
                       for k, v in EVAL_SPLITS.items()},
            "img_size": IMG_SIZE,
            "size_bucket_definition": "COCO AREA: small < SMALL_THRESH^2, "
                                      "large > LARGE_THRESH^2 — NOT the max-side "
                                      "48/96 used by the dataset report and the "
                                      "footprint diagnostic",
            "small_thresh": SMALL_THRESH,
            "large_thresh": LARGE_THRESH,
            "params_filenames_searched": list(PARAMS_FILENAMES),
            "total_results": len(flat_results),
            "total_failed": len(flat_failed),
            "num_phase_folders": len(by_phase_split),
        },
        "by_phase_split": by_phase_split,
        "all_results": flat_results,
        "all_failed": flat_failed,
    }
    with open(combined_path, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    runs_root = Path(RUNS_ROOT)
    phase_folders = sorted([d for d in runs_root.iterdir()
                            if d.is_dir() and re.match(PHASE_FOLDER_PATTERN, d.name)])
    if not phase_folders:
        print(f"[ERROR] No phase folders matching '{PHASE_FOLDER_PATTERN}' in {runs_root}")
        return

    for yaml_path in {DATA_YAMLDS1, DATA_YAMLfull}:
        if not Path(yaml_path).exists():
            print(f"[ERROR] DATA_YAML not found: {yaml_path}")
            return

    print("=" * 80)
    print("BATCH COCO EVALUATION — v6i")
    print("=" * 80)
    print(f"Root folder      : {runs_root}")
    print(f"Phase pattern    : {PHASE_FOLDER_PATTERN}")
    for split_name, info in EVAL_SPLITS.items():
        print(f"Split '{split_name}':  ann={info['coco_ann']}")
        print(f"                     yaml={info['data_yaml']}  "
              f"ultra_split={info.get('ultra_split', 'test')}")
    print(f"Image size       : {IMG_SIZE}")
    print(f"Skip done        : {SKIP_DONE}")
    print(f"Size buckets     : COCO AREA — small < {SMALL_THRESH}^2 = {SMALL_THRESH**2}, "
          f"large > {LARGE_THRESH}^2 = {LARGE_THRESH**2} px^2")
    print(f"                   (NOT the max-side 48/96 of the dataset report)")
    print(f"Params files     : {', '.join(PARAMS_FILENAMES)}")
    print(f"Combined output  : {COMBINED_OUTPUT_JSON or 'DISABLED'}")
    print("=" * 80)

    print(f"\n[INFO] Found {len(phase_folders)} phase folder(s):")
    for pf in phase_folders:
        n = sum(1 for d in pf.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists())
        print(f"  - {pf.name:<30} ({n} runs)")

    by_phase_split = {pf.name: {} for pf in phase_folders}
    total_evaluated = total_failed = 0

    for split_name, info in EVAL_SPLITS.items():
        coco_ann = info["coco_ann"]
        data_yaml = info["data_yaml"]
        ultra_split = info.get("ultra_split", "test")
        print(f"\n\n{'=' * 80}")
        print(f"=== SPLIT: {split_name.upper()} (yaml: {data_yaml}, split='{ultra_split}') ===")
        print(f"{'=' * 80}")

        if not Path(coco_ann).exists():
            print(f"[SKIP] Annotations not found: {coco_ann}")
            continue

        with patch_yaml_val(data_yaml, info["yaml_val_path"]):
            for phase_folder in phase_folders:
                try:
                    num_new, num_failed, results, failed_list = process_phase_folder_on_split(
                        phase_folder=phase_folder, split_name=split_name,
                        coco_ann=coco_ann, data_yaml=data_yaml, img_size=IMG_SIZE,
                        skip_done=SKIP_DONE, output_suffix=info["output_suffix"],
                        ultra_split=ultra_split)
                    by_phase_split[phase_folder.name][split_name] = {
                        "results": results, "failed": failed_list}
                    total_evaluated += num_new
                    total_failed += num_failed
                except Exception as e:
                    print(f"\n[FAILED] {phase_folder.name} on split {split_name}\n  {e}")
                    traceback.print_exc()
                    by_phase_split[phase_folder.name][split_name] = {
                        "results": [], "failed": [{"name": phase_folder.name, "error": str(e)}]}
                    total_failed += 1

    if COMBINED_OUTPUT_JSON:
        try:
            write_combined_json(Path(COMBINED_OUTPUT_JSON), runs_root, by_phase_split)
            print(f"\n[INFO] Combined results saved to {COMBINED_OUTPUT_JSON}")
        except Exception as e:
            print(f"\n[WARN] Failed to write combined JSON: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 80}\nALL DONE\n{'=' * 80}")
    print(f"Total new runs evaluated: {total_evaluated}")
    print(f"Total failed: {total_failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()