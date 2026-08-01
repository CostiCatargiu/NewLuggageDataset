#!/usr/bin/env python3
"""
Batch COCO evaluation over multiple phase folders of YOLO training runs.

Runs evaluation TWICE for every model — once against the validation split
and once against the test split — and writes a separate JSON for each.



Per phase folder, two JSON files are written:
    <phase_folder>/<phase_folder>__valid.json
    <phase_folder>/<phase_folder>__test.json

Optionally a single combined JSON aggregating all results across all
phase folders AND both splits is also written (see COMBINED_OUTPUT_JSON).

How the val/test routing works
------------------------------
The data.yaml is expected to look like:

    train: ../train/images
    val:   ../test/images
    test:  ../test/images

For each evaluation pass, this script temporarily rewrites the `val:`
field of the YAML to point at the right images directory:

    "valid" pass -> val: ../valid/images
    "test"  pass -> val: ../test/images

(Ultralytics' `model.val(data=...)` uses the `val:` field by default.)
The original YAML text is restored at the end, even on error.
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
# ✏️  EDIT THESE
# =============================================================================

RUNS_ROOT = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/"
PHASE_FOLDER_PATTERN = r"runs_lossv2"

DATA_YAMLDS1 = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
DATA_YAMLfull = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"

# Two annotation files per dataset — one per split.
# Each split has its own data_yaml so the correct dataset is loaded.
# "output_suffix" controls the output JSON filename:
#   <phase_folder>__<output_suffix>.json
EVAL_SPLITS = {
    # "valid_ds1": {
    #     "coco_ann": "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation/annotations_coco_valid_abl1.json",
    #     "yaml_val_path": "../valid/images",
    #     "data_yaml": DATA_YAMLDS1,
    #     "output_suffix": "valid_ds1",
    # },
    # "test_ds1": {
    #     "coco_ann": "/home/constantin/Doctorat/GunDatasetNoAugSplitAblation/annotations_coco_test_abl1.json",
    #     "yaml_val_path": "../test/images",
    #     "data_yaml": DATA_YAMLDS1,
    #     "output_suffix": "test_ds1",
    # },
    # "valid_full_dataset": {
    #     "coco_ann": "/home/constantin/Doctorat/GunDatasetNoAugSplit/annotations_coco_valid_full.json",
    #     "yaml_val_path": "../valid/images",
    #     "data_yaml": DATA_YAMLfull,
    #     "output_suffix": "valid_full_dataset",
    # },
    "test_full_dataset": {
        "coco_ann": "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/annotations_coco_test.json",
        "yaml_val_path": "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/test/images",
        "data_yaml": DATA_YAMLfull,
        "output_suffix": "test_full_dataset",
    },
}

IMG_SIZE = 640

# Set True to skip a run whose entry already exists in the JSON for that split
SKIP_DONE = True

# Custom area thresholds for COCO small/medium/large (in pixels)
SMALL_THRESH = 48
LARGE_THRESH = 128

# Optional: write a single combined JSON across all phase folders and both splits.
# Set to None to disable.
COMBINED_OUTPUT_JSON = RUNS_ROOT + "/all_results.json"


# =============================================================================
# ✏️  PHASE TEMPLATES (unchanged)
# =============================================================================

PHASE_A_BEST = {
    "alpha_start": 1.0,
    "alpha_end": 1.0,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "small_obj_boost": 1.0,
    "small_obj_px": 48,
}

PHASE_B_BEST = {
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
}

PHASE_C_BEST = {
    "iou_clip_start": 1000.0,
    "iou_clip_end": 1000.0,
    "dfl_clip_start": 1000.0,
    "dfl_clip_end": 1000.0,
}

PHASE_TEMPLATES = {
    "A": {
        "inherits": [],
        "own_params": {
            "alpha_start": 0.5,
            "alpha_end": 0.3,
            "alpha_min": 0.3,
            "alpha_max": 1.0,
            "small_obj_px": 48,
            "small_obj_boost": 1.0,
        },
    },
    "B": {
        "inherits": [PHASE_A_BEST],
        "own_params": {
            "center_loss_weight_init": 0.01,
            "center_loss_weight_min": 0.01,
            "center_loss_decay_epochs": 35,
        },
    },
    "C": {
        "inherits": [PHASE_A_BEST, PHASE_B_BEST],
        "own_params": {
            "iou_clip_start": 6.0,
            "iou_clip_end": 2.0,
            "dfl_clip_start": 1000.0,
            "dfl_clip_end": 1000.0,
        },
    },
    "D": {
        "inherits": [PHASE_A_BEST, PHASE_B_BEST, PHASE_C_BEST],
        "own_params": {
            "tal_topk": 10,
            "tal_alpha": 0.5,
            "tal_beta": 6.0,
        },
    },
    "E": {
        "inherits": [PHASE_A_BEST, PHASE_B_BEST, PHASE_C_BEST],
        "own_params": {},
    },
}

DEFAULTS = {
    "alpha_start": 0.5,
    "alpha_end": 0.3,
    "alpha_min": 0.3,
    "alpha_max": 1.0,
    "small_obj_px": 48,
    "small_obj_boost": 1.0,
}


# =============================================================================
# YAML PATCHING (val: field swap for valid/test passes)
# =============================================================================

@contextmanager
def patch_yaml_val(data_yaml: str, new_val_path: str):
    """
    Temporarily rewrite the `val:` field of the data.yaml.

    Saves a backup copy, writes the patched version, yields, then restores the
    original text exactly — even if an exception is raised inside the block.
    """
    yaml_path = Path(data_yaml)
    backup_path = yaml_path.with_suffix(yaml_path.suffix + ".bak")

    with open(yaml_path) as f:
        original_text = f.read()

    # Always-overwrite backup so re-runs after a crash are still safe
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
        # Restore the exact original text (preserves comments/formatting)
        with open(yaml_path, "w") as f:
            f.write(original_text)
        print(f"  [YAML] restored original val: field")


# =============================================================================
# HELPERS
# =============================================================================

def get_phase_letter(phase_str: str) -> str:
    m = re.match(r'^([A-Za-z]+)', phase_str)
    return m.group(1).upper() if m else ""


def build_phase_base(phase_letter: str) -> dict:
    cfg = {**DEFAULTS}
    template = PHASE_TEMPLATES.get(phase_letter)
    if template is None:
        return cfg
    for inherited in template.get("inherits", []):
        cfg.update(inherited)
    cfg.update(template.get("own_params", {}))
    return cfg


def parse_folder_name(name: str) -> dict:
    cfg = {}
    phase_m = re.match(r'^([A-Za-z]+\d+)', name)
    if phase_m:
        cfg["phase"] = phase_m.group(1)

    patterns = {
        "alpha_start":     (r'_as([\d.]+)', float),
        "alpha_end":       (r'_ae([\d.]+)', float),
        "small_obj_px":    (r'_px(\d+)',    int),
        "small_obj_boost": (r'_b([\d.]+)',  float),
        "tal_topk":        (r'_topk(\d+)',  int),
        "tal_alpha":       (r'_ta([\d.]+)', float),
        "tal_beta":        (r'_tb([\d.]+)', float),
        "iou_clip_start":  (r'_ics([\d.]+)', float),
        "iou_clip_end":    (r'_ice([\d.]+)', float),
        "dfl_clip_start":  (r'_dcs([\d.]+)', float),
        "dfl_clip_end":    (r'_dce([\d.]+)', float),
    }
    for key, (pat, cast) in patterns.items():
        m = re.search(pat, name)
        if m:
            cfg[key] = cast(m.group(1))
    return cfg


# =============================================================================
# CONFIG EXTRACTION — SA-TAL parameters only
# =============================================================================

# The exact set of parameters reported in each result's "config" field.
# Defaults match your "works well" baseline.
SATAL_PARAMS = {
    "use_satal":          True,
    "satal_alpha_small":  1.2,
    "satal_beta_small":   4.5,
    "satal_alpha_large":  1.0,
    "satal_beta_large":   6.0,
    "satal_small_area":   0.002500,
    "satal_large_area":   0.022500,
    "satal_topk_factor":  1.3,
    "tal_topk":           12,
    "tal_alpha":          0.6,
    "tal_beta":           5.0,
}


def load_args_yaml(run_dir: Path) -> dict:
    """Load the args.yaml that Ultralytics saves alongside each training run."""
    for candidate in [run_dir / "args.yaml", run_dir / "hyp.yaml"]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def build_config(run_dir: Path) -> dict:
    """
    Build the config dict reported for this run.

    Returns ONLY the SA-TAL parameters listed in SATAL_PARAMS, pulling
    values from the run's args.yaml when present, falling back to defaults
    otherwise. No phase parsing, no folder-name parsing.
    """
    from_yaml = load_args_yaml(run_dir)
    cfg = {}
    for key, default in SATAL_PARAMS.items():
        cfg[key] = from_yaml.get(key, default)
    return cfg


# =============================================================================
# PRECISION/RECALL @ BEST-F1 HELPERS  (NEW)
# =============================================================================

def _pr_best_f1_from_curve(pr_curve_1d: np.ndarray,
                           recall_pts: np.ndarray) -> tuple[float, float]:
    """
    Given a 1-D precision array indexed along the 101 COCO recall points,
    walk the PR curve and return (precision, recall) at the best-F1 point.

    Returns (-1.0, -1.0) if no valid points exist.
    """
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
    """
    Walk the PR curve at fixed (IoU, area) and return (precision, recall)
    at the best-F1 point, averaged across classes.

    precision_tensor shape: [T, R, K, A, M]   (COCO standard)
    """
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

        # AP50 (mean precision over the PR curve = average precision)
        ap_values = precision_50[0, :, :, area_idx, -1]
        ap_values = ap_values[ap_values > -1]
        ap = np.mean(ap_values) if len(ap_values) > 0 else -1.0
        metrics[f"mAP50_{area_name}"] = float(ap)

        # AR50 (COCO "average recall" — max recall achieved)
        ar_values = recall_50[0, :, area_idx, -1]
        ar_values = ar_values[ar_values > -1]
        ar = np.mean(ar_values) if len(ar_values) > 0 else -1.0
        metrics[f"AR50_{area_name}"] = float(ar)

        # NEW: precision/recall at best-F1 operating point
        p_bf1, r_bf1 = _best_f1_precision_recall_mean(
            precision_50, iou_idx=0, area_idx=area_idx
        )
        metrics[f"P50_{area_name}"] = p_bf1
        metrics[f"R50_{area_name}"] = r_bf1

    for cat_id in cat_ids:
        cat_name = cat_id_to_name[cat_id]
        cat_idx = cat_id_to_idx[cat_id]
        if cat_name not in per_class_metrics:
            per_class_metrics[cat_name] = {}

        for area_idx, area_name in enumerate(area_labels):
            # Per-class AP50
            ap_values = precision_50[0, :, cat_idx, area_idx, -1]
            ap_values = ap_values[ap_values > -1]
            ap = np.mean(ap_values) if len(ap_values) > 0 else -1.0
            per_class_metrics[cat_name][f"AP50_{area_name}"] = float(ap)

            # Per-class AR50
            ar_value = recall_50[0, cat_idx, area_idx, -1]
            ar = float(ar_value) if ar_value > -1 else -1.0
            per_class_metrics[cat_name][f"AR50_{area_name}"] = float(ar)

            # NEW: per-class P/R at best-F1 (per size, IoU=0.5)
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

        # mAP50:95
        ap_values = precision_5095[:, :, :, area_idx, -1]
        ap_values = ap_values[ap_values > -1]
        ap = np.mean(ap_values) if len(ap_values) > 0 else -1.0
        metrics[f"mAP50_95_{area_name}"] = float(ap)

        # AR50:95
        ar_values = recall_5095[:, :, area_idx, -1]
        ar_values = ar_values[ar_values > -1]
        ar = np.mean(ar_values) if len(ar_values) > 0 else -1.0
        metrics[f"AR50_95_{area_name}"] = float(ar)

        # NEW: precision/recall at best-F1, averaged across all 10 IoU thresholds
        ps, rs = [], []
        for t in range(precision_5095.shape[0]):
            p_t, r_t = _best_f1_precision_recall_mean(
                precision_5095, iou_idx=t, area_idx=area_idx
            )
            if p_t >= 0:
                ps.append(p_t)
                rs.append(r_t)
        metrics[f"P50_95_{area_name}"] = float(np.mean(ps)) if ps else -1.0
        metrics[f"R50_95_{area_name}"] = float(np.mean(rs)) if rs else -1.0

    for cat_id in cat_ids:
        cat_name = cat_id_to_name[cat_id]
        cat_idx = cat_id_to_idx[cat_id]
        for area_idx, area_name in enumerate(area_labels):
            # Per-class AP50:95
            ap_values = precision_5095[:, :, cat_idx, area_idx, -1]
            ap_values = ap_values[ap_values > -1]
            ap = np.mean(ap_values) if len(ap_values) > 0 else -1.0
            per_class_metrics[cat_name][f"AP50_95_{area_name}"] = float(ap)

            # Per-class AR50:95
            ar_values = recall_5095[:, cat_idx, area_idx, -1]
            ar_values = ar_values[ar_values > -1]
            ar = np.mean(ar_values) if len(ar_values) > 0 else -1.0
            per_class_metrics[cat_name][f"AR50_95_{area_name}"] = float(ar)

            # NEW: per-class P/R at best-F1, averaged over IoU thresholds (per size)
            cps, crs = [], []
            for t in range(precision_5095.shape[0]):
                pr_curve = precision_5095[t, :, cat_idx, area_idx, -1]
                p_bf1, r_bf1 = _pr_best_f1_from_curve(pr_curve, recall_pts_5095)
                if p_bf1 >= 0:
                    cps.append(p_bf1)
                    crs.append(r_bf1)
            per_class_metrics[cat_name][f"P50_95_{area_name}"] = (
                float(np.mean(cps)) if cps else -1.0
            )
            per_class_metrics[cat_name][f"R50_95_{area_name}"] = (
                float(np.mean(crs)) if crs else -1.0
            )

    print("\n" + "─" * 100)
    print("Per-Class × Size Breakdown")
    print("─" * 100)
    header = (f"{'Class':<12} │ {'AP50':<6} │ {'AP50_S':<6} │ {'AP50_M':<6} │ {'AP50_L':<6} │ "
              f"{'AP5095':<6} │ {'AP5095_S':<8} │ {'AP5095_M':<8} │ {'AP5095_L':<8}")
    print(header)
    print("─" * 100)
    for cat_name, cat_metrics in per_class_metrics.items():
        row = (
            f"{cat_name:<12} │ "
            f"{cat_metrics.get('AP50_all', -1):>6.3f} │ "
            f"{cat_metrics.get('AP50_small', -1):>6.3f} │ "
            f"{cat_metrics.get('AP50_medium', -1):>6.3f} │ "
            f"{cat_metrics.get('AP50_large', -1):>6.3f} │ "
            f"{cat_metrics.get('AP50_95_all', -1):>6.3f} │ "
            f"{cat_metrics.get('AP50_95_small', -1):>8.3f} │ "
            f"{cat_metrics.get('AP50_95_medium', -1):>8.3f} │ "
            f"{cat_metrics.get('AP50_95_large', -1):>8.3f}"
        )
        print(row)
    print("─" * 100)

    # NEW: per-size precision/recall breakdown (overall, not per-class)
    print("\n" + "─" * 80)
    print("Per-Size Precision/Recall @ best-F1")
    print("─" * 80)
    print(f"{'Size':<8} │ {'P50':>7} │ {'R50':>7} │ {'P50:95':>8} │ {'R50:95':>8} │ "
          f"{'AR50':>7} │ {'AR50:95':>8}")
    print("─" * 80)
    for area_name in ["small", "medium", "large"]:
        print(
            f"{area_name:<8} │ "
            f"{metrics.get(f'P50_{area_name}', -1):>7.3f} │ "
            f"{metrics.get(f'R50_{area_name}', -1):>7.3f} │ "
            f"{metrics.get(f'P50_95_{area_name}', -1):>8.3f} │ "
            f"{metrics.get(f'R50_95_{area_name}', -1):>8.3f} │ "
            f"{metrics.get(f'AR50_{area_name}', -1):>7.3f} │ "
            f"{metrics.get(f'AR50_95_{area_name}', -1):>8.3f}"
        )
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
        return {"mAP50_all": None, "mAP50_95_all": None, "precision": None, "recall": None}


def get_ultralytics_per_class_metrics(val_results, class_names: list) -> dict:
    """
    Pull per-class precision/recall (at best-F1 confidence) plus AP50 and AP50:95
    from the Ultralytics validator. These are the values shown in the
    Ultralytics validation summary table.
    """
    per_class = {}
    try:
        box = val_results.box
        ap50_per_class   = box.ap50
        ap5095_per_class = box.ap
        p_per_class      = box.p   # per-class precision @ best-F1 conf
        r_per_class      = box.r   # per-class recall    @ best-F1 conf
        for idx, name in enumerate(class_names):
            per_class[name] = {
                "AP50_all_ultralytics":    float(ap50_per_class[idx])   if idx < len(ap50_per_class)   else None,
                "AP50_95_all_ultralytics": float(ap5095_per_class[idx]) if idx < len(ap5095_per_class) else None,
                "precision_ultralytics":   float(p_per_class[idx])      if idx < len(p_per_class)      else None,
                "recall_ultralytics":      float(r_per_class[idx])      if idx < len(r_per_class)      else None,
            }
    except Exception as e:
        print(f"[WARN] Could not extract per-class Ultralytics metrics: {e}")
    return per_class


# =============================================================================
# PER-RUN EVALUATION (now split-aware)
# =============================================================================

def eval_run(run_dir: Path, data_yaml: str, coco_ann: str,
             img_size: int, skip_done: bool, done_names: set,
             split_name: str) -> dict | None:
    name = run_dir.name
    weights = run_dir / "weights" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(f"best.pt not found: {weights}")

    if skip_done and name in done_names:
        print(f"  [SKIP] {name} already in {split_name} results.")
        return None

    cfg = build_config(run_dir)

    print(f"\n{'=' * 70}")
    print(f"  RUN   : {name}")
    print(f"  SPLIT : {split_name}")
    print(f"  CFG   : {cfg}")
    print(f"{'=' * 70}")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    val_results = model.val(
        data=data_yaml,
        imgsz=896,
        save_json=True,
        verbose=True, split="test"
    )

    save_dir = Path(val_results.save_dir)
    pred_json = save_dir / "predictions.json"

    if not pred_json.exists():
        raise FileNotFoundError(f"predictions.json not found in {save_dir}")

    class_names = list(model.names.values()) if hasattr(model, 'names') else []

    ul_metrics = get_ultralytics_metrics(val_results)
    ul_per_class = get_ultralytics_per_class_metrics(val_results, class_names)

    coco_metrics, coco_per_class = run_coco_evaluation(
        pred_json, coco_ann, SMALL_THRESH, LARGE_THRESH
    )

    per_class_final = {}
    all_class_names = set(coco_per_class.keys()) | set(ul_per_class.keys())
    for cls_name in sorted(all_class_names):
        per_class_final[cls_name] = {}

        if cls_name in ul_per_class:
            per_class_final[cls_name]["AP50_all"]    = ul_per_class[cls_name].get("AP50_all_ultralytics")
            per_class_final[cls_name]["AP50_95_all"] = ul_per_class[cls_name].get("AP50_95_all_ultralytics")
            per_class_final[cls_name]["precision"]   = ul_per_class[cls_name].get("precision_ultralytics")
            per_class_final[cls_name]["recall"]      = ul_per_class[cls_name].get("recall_ultralytics")

        # Defensive defaults in case Ultralytics didn't expose this class
        per_class_final[cls_name].setdefault("precision", None)
        per_class_final[cls_name].setdefault("recall", None)

        if cls_name in coco_per_class:
            coco_cls = coco_per_class[cls_name]

            # AP per size
            per_class_final[cls_name]["AP50_small"]     = coco_cls.get("AP50_small")
            per_class_final[cls_name]["AP50_medium"]    = coco_cls.get("AP50_medium")
            per_class_final[cls_name]["AP50_large"]     = coco_cls.get("AP50_large")
            per_class_final[cls_name]["AP50_95_small"]  = coco_cls.get("AP50_95_small")
            per_class_final[cls_name]["AP50_95_medium"] = coco_cls.get("AP50_95_medium")
            per_class_final[cls_name]["AP50_95_large"]  = coco_cls.get("AP50_95_large")

            # AR per size (COCO average recall)
            per_class_final[cls_name]["AR50_all"]       = coco_cls.get("AR50_all")
            per_class_final[cls_name]["AR50_small"]     = coco_cls.get("AR50_small")
            per_class_final[cls_name]["AR50_medium"]    = coco_cls.get("AR50_medium")
            per_class_final[cls_name]["AR50_large"]     = coco_cls.get("AR50_large")
            per_class_final[cls_name]["AR50_95_all"]    = coco_cls.get("AR50_95_all")
            per_class_final[cls_name]["AR50_95_small"]  = coco_cls.get("AR50_95_small")
            per_class_final[cls_name]["AR50_95_medium"] = coco_cls.get("AR50_95_medium")
            per_class_final[cls_name]["AR50_95_large"]  = coco_cls.get("AR50_95_large")

            # NEW: precision/recall per size (best-F1 on the PR curve)
            per_class_final[cls_name]["P50_small"]      = coco_cls.get("P50_small")
            per_class_final[cls_name]["P50_medium"]     = coco_cls.get("P50_medium")
            per_class_final[cls_name]["P50_large"]      = coco_cls.get("P50_large")
            per_class_final[cls_name]["R50_small"]      = coco_cls.get("R50_small")
            per_class_final[cls_name]["R50_medium"]     = coco_cls.get("R50_medium")
            per_class_final[cls_name]["R50_large"]      = coco_cls.get("R50_large")
            per_class_final[cls_name]["P50_95_small"]   = coco_cls.get("P50_95_small")
            per_class_final[cls_name]["P50_95_medium"]  = coco_cls.get("P50_95_medium")
            per_class_final[cls_name]["P50_95_large"]   = coco_cls.get("P50_95_large")
            per_class_final[cls_name]["R50_95_small"]   = coco_cls.get("R50_95_small")
            per_class_final[cls_name]["R50_95_medium"]  = coco_cls.get("R50_95_medium")
            per_class_final[cls_name]["R50_95_large"]   = coco_cls.get("R50_95_large")

    metrics = {**ul_metrics, **coco_metrics}
    ordered_metrics = {
        # Overall
        "mAP50_all":       metrics.get("mAP50_all"),
        "mAP50_95_all":    metrics.get("mAP50_95_all"),
        "precision":       metrics.get("precision"),
        "recall":          metrics.get("recall"),

        # mAP per size
        "mAP50_small":     metrics.get("mAP50_small"),
        "mAP50_medium":    metrics.get("mAP50_medium"),
        "mAP50_large":     metrics.get("mAP50_large"),
        "mAP50_95_small":  metrics.get("mAP50_95_small"),
        "mAP50_95_medium": metrics.get("mAP50_95_medium"),
        "mAP50_95_large":  metrics.get("mAP50_95_large"),

        # AR per size
        "AR50_small":      metrics.get("AR50_small"),
        "AR50_medium":     metrics.get("AR50_medium"),
        "AR50_large":      metrics.get("AR50_large"),
        "AR50_95_small":   metrics.get("AR50_95_small"),
        "AR50_95_medium":  metrics.get("AR50_95_medium"),
        "AR50_95_large":   metrics.get("AR50_95_large"),

        # NEW: precision per size (best-F1)
        "P50_small":       metrics.get("P50_small"),
        "P50_medium":      metrics.get("P50_medium"),
        "P50_large":       metrics.get("P50_large"),
        "P50_95_small":    metrics.get("P50_95_small"),
        "P50_95_medium":   metrics.get("P50_95_medium"),
        "P50_95_large":    metrics.get("P50_95_large"),

        # NEW: recall per size (best-F1)
        "R50_small":       metrics.get("R50_small"),
        "R50_medium":      metrics.get("R50_medium"),
        "R50_large":       metrics.get("R50_large"),
        "R50_95_small":    metrics.get("R50_95_small"),
        "R50_95_medium":   metrics.get("R50_95_medium"),
        "R50_95_large":    metrics.get("R50_95_large"),
    }

    # Pretty per-class P/R/AP summary in the log
    print("\n" + "─" * 70)
    print(f"{'Class':<12} │ {'Precision':>9} │ {'Recall':>7} │ {'AP50':>6} │ {'AP50_95':>7}")
    print("─" * 70)
    for cls_name, m in per_class_final.items():
        p  = m.get("precision");   p_s  = f"{p:>9.3f}"  if p  is not None else f"{'—':>9}"
        r  = m.get("recall");      r_s  = f"{r:>7.3f}"  if r  is not None else f"{'—':>7}"
        a  = m.get("AP50_all");    a_s  = f"{a:>6.3f}"  if a  is not None else f"{'—':>6}"
        a2 = m.get("AP50_95_all"); a2_s = f"{a2:>7.3f}" if a2 is not None else f"{'—':>7}"
        print(f"{cls_name:<12} │ {p_s} │ {r_s} │ {a_s} │ {a2_s}")
    print("─" * 70)

    return {
        "name": name,
        "split": split_name,
        "config": cfg,
        "metrics": ordered_metrics,
        "per_class": per_class_final,
    }


# =============================================================================
# PROCESS ONE PHASE FOLDER ON ONE SPLIT
# =============================================================================

def process_phase_folder_on_split(
    phase_folder: Path,
    split_name: str,
    coco_ann: str,
    data_yaml: str,
    img_size: int,
    skip_done: bool,
    output_suffix: str = None,
) -> tuple[int, int, list, list]:
    """
    Run all models in `phase_folder` against the chosen split.
    Writes <phase_folder>/<phase_folder>__<output_suffix>.json.
    Returns (num_new, num_failed, results, failed_list).

    Caller is responsible for patching the YAML's `val:` field beforehand.
    """
    phase_name = phase_folder.name
    suffix = output_suffix or split_name
    output_json = phase_folder / f"{phase_name}__{suffix}.json"

    print(f"\n{'#' * 80}")
    print(f"# PHASE FOLDER : {phase_name}")
    print(f"# SPLIT        : {split_name}")
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

    run_dirs = sorted([
        d for d in phase_folder.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    ])
    print(f"[INFO] {len(run_dirs)} run(s) with best.pt in {phase_name}")

    if not run_dirs:
        return 0, 0, list(existing), []

    print("\n[INFO] Runs to process:")
    for rd in run_dirs:
        status = "[DONE]" if rd.name in done_names and skip_done else "[TODO]"
        print(f"  {status} {rd.name}")
    print()

    results = list(existing)
    failed = []

    for run_dir in run_dirs:
        try:
            entry = eval_run(run_dir, data_yaml, coco_ann,
                             img_size, skip_done, done_names, split_name)
            if entry is None:
                continue
            results.append(entry)
            done_names.add(entry["name"])

            with open(output_json, "w") as f:
                json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)
            print(f"  ✅  Saved → {output_json}")
        except Exception as e:
            print(f"\n  ❌  FAILED: {run_dir.name}\n  {e}")
            traceback.print_exc()
            failed.append({"name": run_dir.name, "error": str(e)})
            with open(output_json, "w") as f:
                json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)

    with open(output_json, "w") as f:
        json.dump({"split": split_name, "results": results, "failed": failed}, f, indent=2)

    num_new = len(results) - len(existing)
    print(f"\n[INFO] {phase_name}/{split_name}: {num_new} new, {len(failed)} failed, "
          f"total {len(results)}")
    return num_new, len(failed), results, failed


# =============================================================================
# COMBINED OUTPUT WRITER (now split-aware)
# =============================================================================

def write_combined_json(combined_path: Path, runs_root: Path,
                        by_phase_split: dict) -> None:
    """
    by_phase_split layout:
        {
            "phase_folder_name": {
                "valid": {"results": [...], "failed": [...]},
                "test":  {"results": [...], "failed": [...]},
            },
            ...
        }
    """
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    flat_results = []
    flat_failed = []
    for phase_name, splits in by_phase_split.items():
        for split_name, data in splits.items():
            for r in data.get("results", []):
                r_copy = dict(r)
                r_copy["phase_folder"] = phase_name
                r_copy["split"] = split_name
                flat_results.append(r_copy)
            for f in data.get("failed", []):
                f_copy = dict(f)
                f_copy["phase_folder"] = phase_name
                f_copy["split"] = split_name
                flat_failed.append(f_copy)

    payload = {
        "metadata": {
            "runs_root": str(runs_root),
            "phase_folder_pattern": PHASE_FOLDER_PATTERN,
            "data_yamls": {
                "ds1": DATA_YAMLDS1,
                "full": DATA_YAMLfull,
            },
            "splits": {
                k: {"coco_ann": v["coco_ann"], "yaml_val_path": v["yaml_val_path"],
                     "data_yaml": v["data_yaml"]}
                for k, v in EVAL_SPLITS.items()
            },
            "img_size": IMG_SIZE,
            "small_thresh": SMALL_THRESH,
            "large_thresh": LARGE_THRESH,
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

    phase_folders = sorted([
        d for d in runs_root.iterdir()
        if d.is_dir() and re.match(PHASE_FOLDER_PATTERN, d.name)
    ])

    if not phase_folders:
        print(f"[ERROR] No phase folders matching '{PHASE_FOLDER_PATTERN}' in {runs_root}")
        return

    # Validate all data YAMLs exist
    all_yamls = {DATA_YAMLDS1, DATA_YAMLfull}
    for yaml_path in all_yamls:
        if not Path(yaml_path).exists():
            print(f"[ERROR] DATA_YAML not found: {yaml_path}")
            return

    for split_name, info in EVAL_SPLITS.items():
        if not Path(info["coco_ann"]).exists():
            print(f"[WARN] COCO annotations missing for split '{split_name}': {info['coco_ann']}")

    print(f"{'=' * 80}")
    print(f"BATCH COCO EVALUATION  —  3 DATASETS × 2 SPLITS  (DS1/DS2/Full × valid/test)")
    print(f"{'=' * 80}")
    print(f"Root folder      : {runs_root}")
    print(f"Phase pattern    : {PHASE_FOLDER_PATTERN}")
    print(f"Data YAML DS1    : {DATA_YAMLDS1}")
    print(f"Data YAML Full   : {DATA_YAMLfull}")
    for split_name, info in EVAL_SPLITS.items():
        print(f"Split '{split_name}':")
        print(f"  ann      : {info['coco_ann']}")
        print(f"  data_yaml: {info['data_yaml']}")
        print(f"  yaml val : -> {info['yaml_val_path']}")
    print(f"Image size       : {IMG_SIZE}")
    print(f"Skip done        : {SKIP_DONE}")
    print(f"Size thresholds  : small < {SMALL_THRESH}px, large > {LARGE_THRESH}px")
    print(f"Combined output  : {COMBINED_OUTPUT_JSON or 'DISABLED'}")
    print(f"{'=' * 80}")

    print(f"\n[INFO] Found {len(phase_folders)} phase folder(s):")
    for pf in phase_folders:
        num_runs = sum(
            1 for d in pf.iterdir()
            if d.is_dir() and (d / "weights" / "best.pt").exists()
        )
        print(f"  • {pf.name:<30} ({num_runs} runs)")

    by_phase_split: dict = {pf.name: {} for pf in phase_folders}
    total_evaluated = 0
    total_failed = 0

    # Outer loop: split (each with its own data_yaml). Inner loop: phase folder.
    # We patch the correct YAML once per split and process all phase folders.
    for split_name, info in EVAL_SPLITS.items():
        coco_ann = info["coco_ann"]
        yaml_val_path = info["yaml_val_path"]
        data_yaml = info["data_yaml"]
        output_suffix = info["output_suffix"]

        print(f"\n\n{'=' * 80}")
        print(f"=== SPLIT: {split_name.upper()} (yaml: {data_yaml}) ===")
        print(f"{'=' * 80}")

        if not Path(coco_ann).exists():
            print(f"[SKIP] Annotations not found, skipping split: {coco_ann}")
            continue

        with patch_yaml_val(data_yaml, yaml_val_path):
            for phase_folder in phase_folders:
                try:
                    num_new, num_failed, results, failed_list = process_phase_folder_on_split(
                        phase_folder=phase_folder,
                        split_name=split_name,
                        coco_ann=coco_ann,
                        data_yaml=data_yaml,
                        img_size=IMG_SIZE,
                        skip_done=SKIP_DONE,
                        output_suffix=output_suffix,
                    )
                    by_phase_split[phase_folder.name][split_name] = {
                        "results": results,
                        "failed": failed_list,
                    }
                    total_evaluated += num_new
                    total_failed += num_failed
                except Exception as e:
                    print(f"\n❌ FAILED: {phase_folder.name} on split {split_name}\n  {e}")
                    traceback.print_exc()
                    by_phase_split[phase_folder.name][split_name] = {
                        "results": [],
                        "failed": [{"name": phase_folder.name, "error": str(e)}],
                    }
                    total_failed += 1

    if COMBINED_OUTPUT_JSON:
        try:
            combined_path = Path(COMBINED_OUTPUT_JSON)
            write_combined_json(combined_path, runs_root, by_phase_split)
            total_results = sum(
                len(splits.get(s, {}).get("results", []))
                for splits in by_phase_split.values()
                for s in EVAL_SPLITS.keys()
            )
            print(f"\n[INFO] Combined results saved to {combined_path}")
            print(f"[INFO] Total results across all phases × splits: {total_results}")
        except Exception as e:
            print(f"\n[WARN] Failed to write combined JSON: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"ALL DONE")
    print(f"{'=' * 80}")
    print(f"Total new runs evaluated: {total_evaluated}")
    print(f"Total failed: {total_failed}")
    print(f"\nOutput files:")
    for pf in phase_folders:
        for split_name, info in EVAL_SPLITS.items():
            suffix = info["output_suffix"]
            output_json = pf / f"{pf.name}__{suffix}.json"
            if output_json.exists():
                with open(output_json) as f:
                    data = json.load(f)
                    num_results = len(data.get("results", []))
                print(f"  • {output_json} ({num_results} results)")
    if COMBINED_OUTPUT_JSON and Path(COMBINED_OUTPUT_JSON).exists():
        with open(COMBINED_OUTPUT_JSON) as f:
            data = json.load(f)
            num_combined = len(data.get("all_results", []))
        print(f"  • {COMBINED_OUTPUT_JSON} ({num_combined} results, combined)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()