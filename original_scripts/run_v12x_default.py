#!/usr/bin/env python3
"""
70% Ablation — YOLOv12x OFFICIAL baseline, PLUS the ls_shift + sqrt arm.

TWO runs, same data / recipe (imgsz 640, 70 epochs, seed 0, close_mosaic 10):

    yolov12x_official_70    stock yolov12.yaml @ x, stock TAL loss, all OFF
    yolov12x_lsshift_sqrt    yolov12x-lsshift.yaml, SWA sqrt 0.7->0.3 px48 b2.0

STOCK ARM: unchanged from before. Builds ultralytics/cfg/models/v12/yolov12.yaml
at x scale + .load(yolov12x.pt). This local yaml is NOT the broken-duplicate
situation yolo26's was -- it has already built and trained correctly earlier
in this project (nl=3, strides [8,16,32], finite loss from epoch 1).

LS_SHIFT ARM (new): builds yolov12x-lsshift.yaml -- a 4-level detect graph
(P2 THROUGH P5, strides [4,8,16,32]; this ADDS a P2 output rather than
trading P5 for P2 like the earlier p234rich family). Rows 0-20 of that yaml
are structurally identical in position to stock's own layout (only
nn.Upsample -> DySample, which carries zero state-dict keys either way), so a
plain .load(yolov12x.pt) transfers the backbone + P3-P5 head with NO remap
step needed. Only the new P2 branch, the three ZGDSConv snake convs, and the
4-input Detect head come out randomly initialized -- the printed pretrained
fraction tells you exactly how much.

Loss for this arm is the SWA sqrt config already validated on this dataset in
the model-scale campaign (alpha 0.7->0.3, area_weight_mode="sqrt", px48,
boost 2.0 -- scored +0.86 mAP50-95 over the stock-loss anchor on yolov12s).
Every other custom loss knob is off (the same _ALL_OFF baseline used there),
so the ONLY things that differ between the two runs here are the graph and
that one loss family.

BATCH: each run keeps ITS OWN batch (deliberately NOT a shared global this
time). Forcing one batch across differently-sized graphs is what caused
Ultralytics to silently auto-halve mid-run for the p234rich x-scale arm
earlier in this project, which changes weight decay / grad accumulation
without telling you. ls_shift adds a P2 (stride-4) level ON TOP OF stock's
P3-P5, which is more memory than the P2-P4-only graphs that already needed
batch 22->11 on this card -- batch 8 here is a conservative, UNVERIFIED
starting point. Watch GPU_mem on epoch 1 and adjust.

Usage:
    python run_yolov12x_official_70.py                          # both
    python run_yolov12x_official_70.py --preflight               # build only
    python run_yolov12x_official_70.py yolov12x_official_70      # one by name
    python run_yolov12x_official_70.py yolov12x_lsshift_sqrt --batch 6
"""

import argparse
import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
PROJECT_DIR = "runs_v12x_default"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12x.pt"   # official COCO weights (auto-downloaded)
EPOCHS = 70
SEED = 0
CLOSE_MOSAIC = 10
PATIENCE = 100

CFG_STOCK = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/ultralytics/cfg/models/v12/yolov12.yaml"
CFG_LSSHIFT = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/ultralytics/cfg/models/v12/yolov12x-lsshift.yaml"

MIN_PRETRAINED_FRAC = 0.45
MIN_PARAMS_X = 30_000_000

# Full "everything off" baseline -- same dict as the model-scale campaign, so
# this script's stock loss matches that project's anchor exactly, not just a
# subset of the knobs.
_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    small_obj_px=0, small_obj_boost=1.0, area_weight_mode="inv",
    center_loss_weight_init=0.0, center_loss_weight_min=0.0,
    center_loss_decay_epochs=35,
    iou_clip_start=999.0, iou_clip_end=999.0,
    dfl_clip_start=999.0, dfl_clip_end=999.0,
    use_nwd=False, nwd_weight=0.0, nwd_C=4.0, dfl_entropy_weight=0.0,
    use_satal=False, use_snatal=False, use_artal=False,
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    cls_mode="bce", use_class_weighting=False, class_weight_mode="sqrt",
    use_pos_boost=False, use_freq_weight=False, use_cls_swa=False,
    use_bag_penalty=False, use_repulsion=False, use_loss_clip=False,
    use_ardfl=False, use_peu=False, use_lba=False,
    box_loss_type="ciou", swa_smooth=False,
    box=7.5, cls=0.5, dfl=1.5,
)

# The validated "sqrt" SWA config from the model-scale campaign (+0.86
# mAP50-95 over the stock-loss anchor on yolov12s, v6i data). Only these six
# keys differ from _ALL_OFF.
_SQRT_0703 = dict(
    _ALL_OFF,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.3, alpha_max=0.7,
    small_obj_px=48, small_obj_boost=2.0, area_weight_mode="sqrt",
)

RUNS = [
    # {"name": "yolov12x_official_70", "order": 1,
    #  "cfg": CFG_STOCK, "weights_only": False, "remap": False,
    #  "strides": [8, 16, 32], "batch": 16, "params": dict(_ALL_OFF),
    #  "label": "yolov12x, stock graph, stock loss (all OFF)"},

    # {"name": "yolov12x_lsshift_sqrt", "order": 2,
    #  "cfg": CFG_LSSHIFT, "weights_only": False, "remap": False,
    #  "strides": [4, 8, 16, 32], "batch": 8, "params": dict(_SQRT_0703),
    #  "label": "yolov12x, ls_shift graph (P2-P5, DySample), SWA sqrt 0.7->0.3 px48 b2.0"},
]
# =============================================================================


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


def build_model(rc):
    """Build from the yaml and transfer pretrained weights. No remap needed
    for either graph here: ls_shift's rows 0-20 line up 1:1 by position with
    stock's own layout, so a plain .load() finds every matching name/shape."""
    model = YOLO(rc["cfg"])
    sd_before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    model.load(PRETRAINED)
    sd_after = model.model.state_dict()
    total = sum(v.numel() for v in sd_after.values())
    moved = sum(sd_after[k].numel() for k in sd_before
                if not torch.equal(sd_before[k].float(), sd_after[k].float()))
    frac = moved / total if total else 0.0
    print(f"  [load ] {moved / 1e6:.2f}M of {total / 1e6:.2f}M params transferred "
          f"from {PRETRAINED}  ({100 * frac:.1f}%)")
    if frac < MIN_PRETRAINED_FRAC:
        raise RuntimeError(
            f"only {100 * frac:.1f}% of parameters came from {PRETRAINED}; the graph is "
            f"mostly RANDOM and this run would train from near-scratch.")
    return model, moved, frac


# ---------------------------------------------------------------- preflight --
def preflight(todo, epochs, seed):
    print("=" * 92)
    print("  PREFLIGHT — building each graph")
    print("=" * 92)
    ok = True
    try:
        from ultralytics.nn.tasks import DetectionModel
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False

    for rc in todo:
        c = rc["cfg"]
        if not os.path.exists(c):
            print(f"  [ABORT] {c} not found")
            return False
        try:
            m = DetectionModel(cfg=c, nc=3, verbose=False)
            det = m.model[-1]
            n_par = sum(x.numel() for x in m.parameters())
            nl = int(getattr(det, "nl", -1))
            st = [int(s) for s in det.stride.tolist()]
            m.eval()
            with torch.no_grad():
                m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        except Exception as ex:
            print(f"  [ABORT] {c} failed to build: {type(ex).__name__}: {ex}")
            return False
        good = nl == len(rc["strides"]) and st == rc["strides"] and n_par >= MIN_PARAMS_X
        ok &= good
        print(f"  {rc['name']:<24} params {n_par:>12,}  nl {nl}  {str(st):<16} "
              f"{'OK' if good else 'WRONG (want nl=' + str(len(rc['strides'])) + ', strides ' + str(rc['strides']) + ')'}")

    print()
    for rc in todo:
        print(f"  RUN  {rc['name']:<24} batch {rc['batch']:<4} {rc['label']}")
    print(f"\n  {len(todo)} runs, imgsz {IMG_SIZE}, {epochs} epochs, seed {seed}")
    return ok


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """The graph and (for ls_shift) the sqrt SWA family are the only things
    allowed to vary between the two runs."""
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 2:
            return
        det = trainer.model.model[-1]
        nl, st = int(getattr(det, "nl", -1)), [int(s) for s in det.stride.tolist()]
        if nl != len(rc["strides"]) or st != rc["strides"]:
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}, expected "
                               f"{len(rc['strides'])} / {rc['strides']} — wrong graph")
        print(f"  [guard] detect levels {nl}, strides {st}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


def run_one(rc, epochs, seed):
    print("\n" + "=" * 92)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 92)
    print(f"  {os.path.basename(rc['cfg'])} | b{rc['batch']} imgsz{IMG_SIZE} ep{epochs} "
          f"seed{seed}\n")

    t0 = time.time()
    model, n_load, frac_load = build_model(rc)
    state = attach_guard(model, rc)

    model.train(
        data=DATA_YAML, epochs=epochs, imgsz=IMG_SIZE, batch=rc["batch"],
        device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=rc["name"],
        patience=PATIENCE, close_mosaic=CLOSE_MOSAIC, seed=seed,
        deterministic=True, **rc["params"],
    )
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")

    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE,
                               batch=rc["batch"], device=DEVICE, workers=WORKERS,
                               project=PROJECT_DIR, name=f"{rc['name']}_test")
        print(f"\n  TEST  mAP50 {tm.box.map50 * 100:.2f}   mAP50-95 {tm.box.map * 100:.2f}")
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")

    print(f"\n  DONE: {rc['name']} ({hours:.2f}h)  ->  {save_dir}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--batch", type=int, default=None,
                    help="override batch for ALL selected runs (default: each run's own)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    epochs, seed = a.epochs, a.seed

    runs = [dict(r) for r in RUNS]
    if a.batch is not None:
        for r in runs:
            r["batch"] = a.batch
    todo = order([r for r in runs if not a.names or r["name"] in a.names])
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 92)
    print(f"  YOLOv12x — stock vs ls_shift+sqrt  ({len(todo)} runs)")
    print("=" * 92)
    ok = preflight(todo, epochs, seed)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph failed preflight.")
        return

    for rc in todo:
        try:
            run_one(rc, epochs, seed)
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")


if __name__ == "__main__":
    main()