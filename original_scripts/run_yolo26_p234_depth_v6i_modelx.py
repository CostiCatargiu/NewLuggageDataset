#!/usr/bin/env python3
r"""
YOLO26x — p234rich vs the stock (official) model

Two runs, same data, same 70-epoch recipe, same seed, stock loss on both.
The ONLY variable is the graph:

    y26x_stock      loaded directly from yolo26x.pt   3 levels, strides [8, 16, 32]
    y26x_p234rich   yolo26-p2dys-p234rich.yaml (x)     3 levels, strides [4, 8, 16],
                                                        head depth (4, 4) on rows 19/22

STOCK MODEL: NOT built from any local yaml. This repo's ultralytics/cfg/models/26/
yolo26.yaml AND yolo26x.yaml are both, byte-for-byte apart from one scale number,
the YOLOE-26 open-vocabulary SEGMENTATION config (YOLOESegment26 head, needs a text
embedding input) -- there is no usable plain-detect yolo26 yaml in that directory.
The correct and much simpler fix is to skip yaml reconstruction entirely for the
stock arm: `YOLO(WEIGHTS)` loads the architecture that ships INSIDE the .pt
checkpoint, which is the actual official stock detector, no remap/.load() needed.

Scale handling for p234rich: Ultralytics picks the scale from the yaml FILENAME
(yolo26x-...). CFG_RICH is written with the 'x' in it; if that exact file does not
exist Ultralytics falls back to the unscaled yolo26-p2dys-p234rich.yaml and applies
scale x. The preflight prints the parameter count so a silent wrong-scale build
(the 'n' default) is visible: x should be ~50M+, not ~3M.

BATCH: the s-scale runs used 48. The x model is several times larger and the P2
graph adds a stride-4 head, so the default here is 16. Override with --batch.
Batch changes the optimisation, so keep it IDENTICAL for both runs (this script
enforces that; it is one global).

    Usage:
        python run_yolo26x_p234rich_vs_stock_v6i.py                    # both
        python run_yolo26x_p234rich_vs_stock_v6i.py --preflight        # build only
        python run_yolo26x_p234rich_vs_stock_v6i.py y26x_p234rich      # one by name
        python run_yolo26x_p234rich_vs_stock_v6i.py --batch 24 --seed 1
"""

import argparse
import gc
import json
import os
import re
import time

import torch
from ultralytics import YOLO

# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml"
WEIGHTS = "yolo26x.pt"
PROJECT_DIR = "runs_yolo26x_p234rich_vs_stock_v6i"
EPOCHS = 70
IMG_SIZE = 640
BATCH = 12
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLOSE_MOSAIC = 10
PATIENCE = 100
OVERWRITE_EXISTING = False

CFG_DIR = "ultralytics/cfg/models/26"
CFG_RICH = f"{CFG_DIR}/yolo26x-p2dys-p234rich.yaml"
# No CFG_STOCK: yolo26.yaml and yolo26x.yaml in this repo are both the YOLOE-26
# segmentation config, not a detector -- see docstring. The stock arm loads
# straight from WEIGHTS below instead of any yaml.

# p234rich is a 3-level graph: stock head rows 17-19 -> 23-25 (shift 6).
REMAP_SHIFT = 6
REMAP_3LVL = range(17, 20)

# YOLO(<yaml>) builds a RANDOM model; .load() transfers the checkpoint.
MIN_PRETRAINED_FRAC = 0.45
# An x-scale model is ~50M+ params. The 'n' default is ~3M. Catch a wrong scale.
MIN_PARAMS_X = 30_000_000

_ALL_OFF = dict(
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    area_weight_mode="inv", area_weight_norm="max",
    small_obj_px=0, small_obj_boost=1.0,
    use_lbtal=False,
    tal_alpha=0.5, tal_beta=6.0, tal_beta_small=None, tal_beta_ref_px=64.0,
    tal_beta_o2m=None, tal_beta_o2o=None, tal_alpha_o2m=None, tal_alpha_o2o=None,
    tal_beta_sel=None, tal_beta_tgt=None, tcn_p=1.0, tal_beta_level=None,
    scb_branch="both",
    o2m_start=0.8, o2m_final=0.1, o2m_decay=True,
    l1_scale_p=0.0,
    sbb_q=0.0, sbb_ref_px=64.0, sbb_invert=False,
    snt_tau=0.0, snt_gamma=2.0, snt_min_iou=0.5,
    sharp_rho=1.0, cls_pw=0.0,
    nwd=0.0, nwd_c=24.0, iou_type="ciou", scale_balance=0.0,
    box=7.5, cls=0.5, dfl=1.5,
    multi_scale=0.0, scale=0.5, close_mosaic=CLOSE_MOSAIC, cos_lr=False,
)
# =============================================================================


def cfg(**over):
    d = dict(_ALL_OFF)
    d.update(over)
    return d


RUNS = [
    # {"name": "y26x_stock", "order": 1, "seed": 0,
    #  "cfg": None, "weights_only": True, "remap": False, "depth": None, "beta": 6.0,
    #  "strides": [8, 16, 32], "params": cfg(),
    #  "label": "yolo26x, official architecture (loaded straight from yolo26x.pt)",
    #  "why": "Reference for the x scale. Architecture + weights both come from the "
    #         "checkpoint -- no local yaml, no remap, no graph edits."},

    {"name": "y26x_p234rich", "order": 2, "seed": 0,
     "cfg": CFG_RICH, "weights_only": False, "remap": True, "depth": (4, 4), "beta": 6.0,
     "strides": [4, 8, 16], "params": cfg(),
     "label": "yolo26x, p234rich (P2-P4 graph, head depth (4,4), stock loss)",
     "why": "Same recipe and loss as the stock run; only the graph differs."},
]


def order(runs):
    return sorted(runs, key=lambda r: r["order"])


def build_model(rc):
    """Build the model. Stock: load the checkpoint's own architecture+weights
    directly, no yaml. p234rich: build the custom yaml, then transfer + remap."""
    if rc["weights_only"]:
        model = YOLO(WEIGHTS)  # architecture ships inside the .pt; already loaded
        n_par = sum(p.numel() for p in model.model.parameters())
        print(f"  [load ] {n_par / 1e6:.2f}M params loaded directly from {WEIGHTS} "
              f"(official architecture, no yaml, no .load() transfer)")
        return model, n_par, 1.0, 0
    model = YOLO(rc["cfg"])
    n_load, frac = load_pretrained(model)     # MUST precede remap_pan
    n_moved = remap_pan(model, REMAP_3LVL) if rc["remap"] else 0
    return model, n_load, frac, n_moved


# ---------------------------------------------------------------- preflight --
def preflight(todo):
    print("=" * 92)
    print("  PREFLIGHT — building each graph")
    print("=" * 92)
    ok = True
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import DetectionModel
    except Exception as ex:
        print(f"  [ABORT] cannot import ultralytics: {ex}")
        return False
    if "tal_beta" not in DEFAULT_CFG_DICT:
        print("  [ABORT] default.yaml does not accept tal_beta (patched fork required)")
        return False

    for rc in todo:
        c = rc["cfg"]
        if rc["weights_only"]:
            # No yaml at all: load straight from the checkpoint and inspect it.
            try:
                m = YOLO(WEIGHTS).model
                det = m.model[-1]
                n_par = sum(x.numel() for x in m.parameters())
                nl = int(getattr(det, "nl", -1))
                st = [int(s) for s in det.stride.tolist()]
                m.eval()
                with torch.no_grad():
                    m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
            except Exception as ex:
                print(f"  [ABORT] {WEIGHTS} failed to load: {type(ex).__name__}: {ex}")
                return False
            good = nl == 3 and st == rc["strides"] and n_par >= MIN_PARAMS_X
            ok &= good
            print(f"  {rc['name']:<16} params {n_par:>12,}  nl {nl}  {str(st):<12} "
                  f"{'OK' if good else 'WRONG (want nl=3, strides ' + str(rc['strides']) + ', x-scale params)'}"
                  f"  [from {WEIGHTS}, no yaml]")
            continue
        if rc["remap"] and not os.path.exists(c):
            fallback = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", c)
            if not os.path.exists(fallback):
                print(f"  [ABORT] neither {c} nor {fallback} found")
                return False
            print(f"  [note ] {c} not found; Ultralytics will use {fallback} at scale x")
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
        good = nl == 3 and st == rc["strides"] and n_par >= MIN_PARAMS_X
        ok &= good
        print(f"  {rc['name']:<16} params {n_par:>12,}  nl {nl}  {str(st):<12} "
              f"{'OK' if good else 'WRONG (want nl=3, strides ' + str(rc['strides']) + ', x-scale params)'}")

    print()
    for rc in todo:
        print(f"  RUN  {rc['name']:<16} seed {rc['seed']}  depth {rc['depth']}  STOCK LOSS")
    print(f"\n  {len(todo)} runs, b{BATCH}, imgsz {IMG_SIZE}, {EPOCHS} epochs")
    return ok


# ------------------------------------------------------------------- load --
def load_pretrained(model):
    """Transfer yolo26x.pt into the graph. MUST RUN BEFORE remap_pan()."""
    sd_before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    model.load(WEIGHTS)
    sd_after = model.model.state_dict()
    total = sum(v.numel() for v in sd_after.values())
    moved = sum(sd_after[k].numel() for k in sd_before
                if not torch.equal(sd_before[k].float(), sd_after[k].float()))
    frac = moved / total if total else 0.0
    print(f"  [load ] {moved / 1e6:.2f}M of {total / 1e6:.2f}M params transferred "
          f"from {WEIGHTS}  ({100 * frac:.1f}%)")
    if frac < MIN_PRETRAINED_FRAC:
        raise RuntimeError(
            f"only {100 * frac:.1f}% of parameters came from {WEIGHTS}; the graph is "
            f"mostly RANDOM and this run would train from near-scratch.")
    return moved, frac


# -------------------------------------------------------------------- remap --
def remap_pan(model, rows):
    """Copy stock head rows 17-19 into rows 23-25 of the shifted P2 graph."""
    ckpt = YOLO(WEIGHTS).model.float().state_dict()
    sd = model.model.state_dict()
    moved, skipped = {}, 0
    for k, v in ckpt.items():
        m = re.match(r"model\.(\d+)\.(.+)", k)
        if not m or int(m.group(1)) not in rows:
            continue
        k2 = f"model.{int(m.group(1)) + REMAP_SHIFT}.{m.group(2)}"
        if k2 in sd and sd[k2].shape == v.shape:
            moved[k2] = v
        else:
            skipped += 1
    if not moved:
        raise RuntimeError("remap matched nothing — the row shift or the graph is not "
                           "what this run assumes")
    if skipped:
        raise RuntimeError(f"remap skipped {skipped} tensors inside rows "
                           f"{rows.start}-{rows.stop - 1}")
    model.model.load_state_dict(moved, strict=False)
    after = model.model.state_dict()
    for k2, v in moved.items():
        if not torch.equal(after[k2].float(), v.float()):
            raise RuntimeError(f"remap did not stick on {k2}")
    print(f"  [remap] {len(moved)} tensors, "
          f"{sum(v.numel() for v in moved.values()) / 1e6:.2f}M params, 0 skipped")
    return len(moved)


# -------------------------------------------------------------------- guard --
def attach_guard(model, rc):
    """Stock loss on both runs; the graph is the only thing allowed to differ."""
    state = {"verified": False}

    def on_epoch_start(trainer):
        crit = getattr(getattr(trainer, "model", None), "criterion", None)
        if crit is None or state["verified"] or trainer.epoch < 2:
            return
        o2m, o2o = getattr(crit, "one2many", None), getattr(crit, "one2one", None)
        if o2m is None or o2o is None:
            raise RuntimeError(f"{rc['name']}: criterion is not E2ELoss")
        for tag, br in (("one2many", o2m), ("one2one", o2o)):
            a = br.assigner
            if abs(float(a.beta) - rc["beta"]) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} beta={a.beta}, expected {rc['beta']}")
            if abs(float(a.alpha) - 0.5) > 1e-6:
                raise RuntimeError(f"{rc['name']}: {tag} alpha={a.alpha}, expected 0.5")
            for attr in ("beta_small", "beta_sel", "beta_tgt", "beta_level"):
                if getattr(a, attr, None) is not None:
                    raise RuntimeError(f"{rc['name']}: {tag} {attr} live")
            if float(getattr(a, "tcn_p", 1.0)) != 1.0:
                raise RuntimeError(f"{rc['name']}: tcn_p is live")
            if a.scb_enabled() or a.snt_enabled() or a.tsh_enabled():
                raise RuntimeError(f"{rc['name']}: SCB/SNT/TSH live — not requested")
            b = br.bbox_loss
            if float(getattr(b, "sbb_q", 0.0)) != 0.0 or b.swa_enabled() \
                    or b.snl1_enabled() or float(getattr(b, "nwd", 0.0)) != 0.0:
                raise RuntimeError(f"{rc['name']}: SBB/SWA/SNL1/NWD live on {tag}")
        if bool(getattr(o2o.hyp, "use_lbtal", False)):
            raise RuntimeError(f"{rc['name']}: use_lbtal set")
        det = trainer.model.model[-1]
        nl, st = int(getattr(det, "nl", -1)), [int(s) for s in det.stride.tolist()]
        if nl != 3 or st != rc["strides"]:
            raise RuntimeError(f"{rc['name']}: detect nl={nl} strides={st}, expected 3 / "
                               f"{rc['strides']} — wrong graph")
        print(f"  [guard] stock loss (beta={rc['beta']}, alpha=0.5), nothing else live")
        print(f"  [guard] detect levels {nl}, strides {st}")
        state["verified"] = True

    model.add_callback("on_train_epoch_start", on_epoch_start)
    return state


# ---------------------------------------------------------------- run / main --
def run_one(rc):
    print("\n" + "=" * 92)
    print(f"  RUN {rc['name']}")
    print(f"  {rc['label']}")
    print("=" * 92)
    src = f"{WEIGHTS} (checkpoint architecture, no yaml)" if rc["weights_only"] \
        else os.path.basename(rc["cfg"])
    print(f"  {src} | b{BATCH} imgsz{IMG_SIZE} ep{EPOCHS} "
          f"seed{rc['seed']} | STOCK LOSS\n")
    t0 = time.time()
    model, n_load, frac_load, n_moved = build_model(rc)
    state = attach_guard(model, rc)
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
                          device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                          name=rc["name"], patience=PATIENCE, seed=rc["seed"],
                          deterministic=True, exist_ok=OVERWRITE_EXISTING, **rc["params"])
    if not state["verified"]:
        raise RuntimeError(f"{rc['name']}: the guard never ran — cannot certify this run")
    hours = (time.time() - t0) / 3600
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, rc["name"])))
    weights = os.path.join(save_dir, "weights", "best.pt")
    n_par = sum(p.numel() for p in model.model.parameters())
    out = {"name": rc["name"], "cfg": rc["cfg"] or f"{WEIGHTS} (no yaml)",
           "head_depth": rc["depth"],
           "seed": rc["seed"], "batch": BATCH, "imgsz": IMG_SIZE, "epochs": EPOCHS,
           "tal_beta": rc["beta"], "strides": rc["strides"],
           "remap_tensors": n_moved, "pretrained_params": n_load,
           "pretrained_frac": round(frac_load, 4), "n_params": n_par,
           "loss_params": rc["params"], "hours": hours, "weights": weights,
           "test_map50": float("nan"), "test_map5095": float("nan")}
    try:
        with open(os.path.join(save_dir, "x_run_params.json"), "w") as f:
            json.dump({**out, "why": rc["why"], "label": rc["label"]}, f, indent=2)
    except Exception as ex:
        print(f"  [warn] params json not saved: {ex}")
    try:
        tm = YOLO(weights).val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                               device=DEVICE, workers=WORKERS, project=PROJECT_DIR,
                               name=f"{rc['name']}_test")
        out["test_map50"] = float(tm.box.map50)
        out["test_map5095"] = float(tm.box.map)
    except Exception as ex:
        print(f"  [warn] test eval failed: {ex}")
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="applied to BOTH runs (default %(default)s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override seed for both runs; non-zero adds an _s<seed> suffix")
    a = ap.parse_args()
    BATCH = a.batch

    runs = [dict(r) for r in RUNS]
    if a.seed is not None:
        for r in runs:
            r["seed"] = a.seed
            if a.seed != 0:
                r["name"] = f"{r['name']}_s{a.seed}"
    todo = order([r for r in runs
                  if not a.names or r["name"] in a.names
                  or any(r["name"].startswith(n) for n in a.names)])
    if not todo:
        print("nothing selected.")
        return

    print()
    print("=" * 92)
    print(f"  YOLO26x — p234rich vs stock  ({len(todo)} runs, batch {BATCH})")
    print("=" * 92)
    ok = preflight(todo)
    if a.preflight:
        return
    if not ok:
        print("\n  [ABORT] a graph failed preflight.")
        return

    res, path = [], "runs_yolo26x_p234rich_vs_stock__partial.json"
    for rc in todo:
        try:
            res.append(run_one(rc))
        except Exception as ex:
            print(f"  [FAIL] {rc['name']}: {ex}")
            res.append({"name": rc["name"], "error": str(ex)})
        with open(path, "w") as f:
            json.dump({"batch": BATCH, "imgsz": IMG_SIZE, "results": res}, f, indent=2)

    print("\n" + "=" * 92)
    print("  RESULTS (test split; mAP50-95 shown — SCORE mAP50_small / AR50_95_small yourself)")
    print("=" * 92)
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<20} FAILED  {r['error']}")
        else:
            print(f"  {r['name']:<20} mAP50 {r['test_map50']*100:6.2f}  "
                  f"mAP50-95 {r['test_map5095']*100:6.2f}  "
                  f"{r['n_params']/1e6:6.2f}M  {r['hours']:.2f} h")
    print(f"\n  written to {path}")
    print("  n=1 per arm: differences under ~1.0 on small are inside run-to-run noise.")
    print("  Re-run with --seed 1 to get a second draw of each arm.")


if __name__ == "__main__":
    main()