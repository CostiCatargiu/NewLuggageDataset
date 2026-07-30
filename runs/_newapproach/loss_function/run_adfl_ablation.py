#!/usr/bin/env python3
"""
Anisotropic-DFL (A-DFL) ablation.

=============================================================================
THE MECHANISM
=============================================================================
Stock DFL encodes all four box edges into the same reg_max bins in units of the
anchor's stride, so every edge gets identical ABSOLUTE precision (1 bin = 1
stride = 8/16/32 px). A-DFL introduces a per-edge range scale s_e:

    encode:  t_e = d_e / s_e        decode:  d_e = E[bin] * s_e

reg_max is unchanged (pretrained weights load, head channels unchanged); only
the span the bins cover changes. s_w < 1 gives the width edges finer bins over
a shorter range.

=============================================================================
WHY WIDTH
=============================================================================
IoU is scale-relative. For a box w x h, an error of e px on one edge costs:

    width  edge (left/right):  e / w        height edge (top/bottom):  e / h

Ratio = h/w = 2.69 on this dataset (bag 2.23, backpack 2.55, trolley 2.96).
So width edges are ~2.7x more IoU-sensitive while receiving the same absolute
resolution, and being short they occupy only ~1.6-2.6 of the 16 available bins.
The rest of the width budget covers a range the data never reaches.

PRINCIPLED SCALE: choose s_w / s_h = w_mean / h_mean = 1 / 2.69 = 0.372, which
equalises each axis's quantisation contribution to IoU. At mean box 41x110 @640
stride 8 this moves the width-edge error cost from 9.76% IoU to 3.41%, matching
the height edge's 3.64%. That is a derivation, not a tuned hyperparameter — the
sweep below exists to show the dose-response around it.

=============================================================================
THE COST, AND THE CONTROL THAT MATTERS
=============================================================================
Compressing a range risks SATURATION (targets clipped at the last bin).
Estimated upper bounds on width-target saturation at stride 8: s_w=0.75 -> 2.1%,
0.5 -> 5.5%, 0.375 -> ~9%, 0.25 -> 19%. Real rates are lower because TAL sends
wide boxes to coarser levels. Live per-edge clamp rates are logged every epoch.

CONTROL: `adfl_iso050` compresses BOTH axes by 0.5. If it matches the
anisotropic run, the gain is just "finer bins everywhere" and the anisotropy
claim is dead. Without this control a reviewer will ask, and rightly.

=============================================================================
REQUIRES
=============================================================================
  * loss.py (this folder, with A-DFL) copied to ultralytics/utils/loss.py
  * adfl_patch_dfl.py importable — it patches DFL.forward so INFERENCE applies
    the same scale. Without it the model trains and evaluates on different box
    parameterisations and mAP collapses.

Usage:
  python run_adfl_ablation.py                    # full ablation
  python run_adfl_ablation.py adfl_w0375         # one run
  python run_adfl_ablation.py --seeds            # 3-seed replication of anchor + best
  python run_adfl_ablation.py --with-test        # also eval the test split
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time

import torch

# --- patch DFL BEFORE the model is built ------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adfl_patch_dfl  # noqa: E402

adfl_patch_dfl.install()

from ultralytics import YOLO  # noqa: E402

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(m):
        return m.module if hasattr(m, "module") else m

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/LuggageDataset.v5i.yolov12/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_adfl"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 58
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEED = 0

MEAN_AR = 2.69          # dataset mean h/w -> the principled scale is 1/MEAN_AR

# Baseline: plain TAL, stock CIoU, NO NWD. Deliberately clean.
# The prior "best" (r10_nwd_fixedc) used a DIFFERENT NWD implementation
# (small-only, adaptive C, annealed ratio) that loss.py does not reproduce, so
# stacking on it would compare against a baseline that was never actually run.
# A-DFL is measured against stock. NWD can be layered afterwards if it wins.
_BASE = dict(
    use_satal=False, tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    swa_mode="scale", swa_alpha=0.0, swa_boost=1.0, swa_size_axis="width",
    box_loss_type="ciou",
    use_nwd=False,
    use_class_weights=False, use_vfl=False, use_loss_clip=False,
    use_ardfl=False,
)

_ADFL_OFF = dict(use_adfl=False, adfl_w_scale=1.0, adfl_h_scale=1.0, adfl_log_clamp=True)


def cfg(**over):
    d = dict(_BASE)
    d.update(_ADFL_OFF)
    d.update(over)
    return d


def adfl(w, h=1.0):
    return cfg(use_adfl=True, adfl_w_scale=w, adfl_h_scale=h)


PRINCIPLED = round(1.0 / MEAN_AR, 3)      # 0.372

RUNS = [
    {"name": "adfl_anchor", "rank": 0, "kind": "baseline",
     "label": "stock DFL (s_w=s_h=1.0) — the beat-target",
     "params": cfg()},

    {"name": "adfl_w0375", "rank": 1, "kind": "anisotropic",
     "label": f"s_w={PRINCIPLED} (=1/AR, equalises per-axis IoU quantisation cost)",
     "params": adfl(PRINCIPLED)},

    {"name": "adfl_w050", "rank": 2, "kind": "anisotropic",
     "label": "s_w=0.50 — 2x finer width bins, conservative saturation (~5%)",
     "params": adfl(0.50)},

    {"name": "adfl_iso050", "rank": 3, "kind": "CONTROL",
     "label": "s_w=s_h=0.50 — isotropic compression. THE control: separates "
              "'anisotropy' from 'finer bins everywhere'",
     "params": adfl(0.50, 0.50)},

    {"name": "adfl_w075", "rank": 4, "kind": "anisotropic",
     "label": "s_w=0.75 — mild dose, lowest saturation risk",
     "params": adfl(0.75)},

    {"name": "adfl_w0375_h125", "rank": 5, "kind": "anisotropic",
     "label": f"s_w={PRINCIPLED}, s_h=1.25 — also give height the reach it uses",
     "params": adfl(PRINCIPLED, 1.25)},

    {"name": "adfl_w025", "rank": 6, "kind": "anisotropic",
     "label": "s_w=0.25 — aggressive; expected to saturate (~19% upper bound)",
     "params": adfl(0.25)},
]

SEED_RUNS = ["adfl_anchor", "adfl_w0375"]     # replicated over SEEDS_LIST
SEEDS_LIST = [0, 42, 123]


# =============================================================================
def _loss_file_fingerprint():
    """Record WHICH loss.py actually ran. The prior ablations did not, and the
    saved configs became uninterpretable as a result."""
    try:
        import ultralytics.utils.loss as L
        p = L.__file__
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
        return {"loss_path": p, "loss_md5": h,
                "has_adfl": hasattr(L, "adfl_encode")}
    except Exception as e:
        return {"error": str(e)}


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    m = de_parallel(trainer.model)
    try:
        m.current_epoch = epoch
    except Exception:
        pass
    try:
        from ultralytics.utils.loss import set_epoch
        set_epoch(epoch, getattr(trainer, "epochs", EPOCHS))
    except Exception:
        pass


def _make_scale_enforcer(w_scale, h_scale):
    """Re-apply the edge scale to the TRAINER's model.

    Ultralytics may rebuild/wrap the model between YOLO() and the training loop
    (get_model, DDP wrapping, EMA). If that happens the buffer we set on the
    original module is not the one used, and training would silently run stock
    DFL while the config banner claims A-DFL. This hook re-asserts it on the
    live model and verifies, loudly, that it took.
    """
    def _hook(trainer):
        for obj in (getattr(trainer, "model", None), getattr(getattr(trainer, "ema", None), "ema", None)):
            if obj is None:
                continue
            try:
                adfl_patch_dfl.set_scales(de_parallel(obj), w_scale, h_scale, verbose=False)
            except Exception as e:
                print(f"  [A-DFL] WARNING could not set scales: {e}")
        got = adfl_patch_dfl.get_scales(de_parallel(trainer.model))
        want = [w_scale, h_scale, w_scale, h_scale]
        ok = bool(got) and all(g == want for g in got)
        print(f"  [A-DFL] live DFL edge_scale = {got}  expected {want}  -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise RuntimeError("A-DFL edge_scale did not apply to the training model — aborting "
                               "rather than producing a mislabelled run.")
    return _hook


def on_train_epoch_end(trainer):
    """Report per-edge DFL saturation — the one real failure mode of A-DFL."""
    try:
        from ultralytics.utils.loss import adfl_clamp_report
        r = adfl_clamp_report(reset=True)
        if r:
            worst = max(r.values())
            flag = "  <-- HIGH, consider a larger s_w" if worst > 0.15 else ""
            print("  [A-DFL] edge saturation: " +
                  "  ".join(f"{k}={v*100:.2f}%" for k, v in r.items()) + flag)
    except Exception:
        pass


def run_one(rc, seed=SEED, with_test=False):
    name = rc["name"] if seed == SEED else f"{rc['name']}_s{seed}"
    p = rc["params"]
    print(f"\n{'=' * 74}\n  RUN {name}   [{rc.get('kind','')}]  seed={seed}\n  {rc['label']}\n{'=' * 74}\n")

    t0 = time.time()
    model = YOLO(MODEL_WEIGHTS)

    # write the scale into the DFL buffer so it is saved in the checkpoint and
    # inference/val decode matches training
    ws = p.get("adfl_w_scale", 1.0) if p.get("use_adfl") else 1.0
    hs = p.get("adfl_h_scale", 1.0) if p.get("use_adfl") else 1.0
    adfl_patch_dfl.set_scales(model, w_scale=ws, h_scale=hs)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    # re-assert on the trainer's live model (survives get_model / DDP / EMA)
    model.add_callback("on_train_start", _make_scale_enforcer(ws, hs))

    kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
              device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=name,
              patience=100, close_mosaic=10, seed=seed, deterministic=True,
              exist_ok=False)
    kw.update(copy.deepcopy(p))

    results = model.train(**kw)
    hours = (time.time() - t0) / 3600

    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir",
                           os.path.join(PROJECT_DIR, name)))
    meta = {"name": name, "kind": rc.get("kind"), "label": rc["label"], "params": p,
            "seed": seed, "epochs": EPOCHS, "imgsz": IMG_SIZE, "batch": BATCH,
            "mean_ar": MEAN_AR, "principled_w_scale": PRINCIPLED,
            "loss_file": _loss_file_fingerprint(),
            "dfl_edge_scale": adfl_patch_dfl.get_scales(model)}
    try:
        with open(os.path.join(save_dir, "adfl_params.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  [warn] params json not saved: {e}")

    def _m(rd, *keys):
        for k in keys:
            if k in rd:
                return float(rd[k])
        return float("nan")

    rd = getattr(results, "results_dict", {}) or {}
    out = {"name": name, "kind": rc.get("kind"), "seed": seed, "hours": hours,
           "val_map50": _m(rd, "metrics/mAP50(B)", "metrics/mAP50"),
           "val_map5095": _m(rd, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
           "test_map50": float("nan"), "test_map5095": float("nan")}

    if with_test:
        try:
            tm = YOLO(os.path.join(save_dir, "weights", "best.pt")).val(
                data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=BATCH,
                device=DEVICE, workers=WORKERS, project=PROJECT_DIR, name=f"{name}_test")
            out["test_map50"], out["test_map5095"] = float(tm.box.map50), float(tm.box.map)
        except Exception as e:
            print(f"  [warn] test eval failed: {e}")

    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(res, path):
    anchor = next((r for r in res if r["name"] == "adfl_anchor"), None)
    a = anchor["val_map5095"] if anchor else None
    print(f"\n{'=' * 74}\n  RESULTS (val — selection basis)\n{'=' * 74}")
    print(f"{'run':<22s}{'kind':<14s}{'mAP50':>8s}{'mAP50-95':>11s}{'delta':>9s}{'h':>6s}")
    print("-" * 74)
    for r in sorted(res, key=lambda x: -(x["val_map5095"] if x["val_map5095"] == x["val_map5095"] else -9)):
        d = f"{(r['val_map5095']-a)*100:+.2f}" if a and r["name"] != "adfl_anchor" else "—"
        print(f"{r['name']:<22s}{str(r['kind']):<14s}{r['val_map50']*100:>8.2f}"
              f"{r['val_map5095']*100:>11.2f}{d:>9s}{r['hours']:>5.1f}")
    print("\nREAD THE CONTROL FIRST: if adfl_iso050 >= adfl_w050, the gain is")
    print("resolution, not anisotropy, and the paper's claim must change.")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="subset of run names")
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--seeds", action="store_true",
                    help=f"replicate {SEED_RUNS} over seeds {SEEDS_LIST}")
    args = ap.parse_args()

    print(f"\n{'=' * 74}")
    print(f"  A-DFL ABLATION   @{IMG_SIZE}px, {EPOCHS} epochs")
    print(f"  dataset mean h/w = {MEAN_AR}  ->  principled s_w = {PRINCIPLED}")
    print(f"  loss file: {_loss_file_fingerprint()}")
    print(f"{'=' * 74}")

    if args.seeds:
        todo = [(r, s) for r in RUNS if r["name"] in SEED_RUNS for s in SEEDS_LIST]
    else:
        sel = RUNS if not args.runs else [r for r in RUNS if r["name"] in set(args.runs)]
        unknown = set(args.runs) - {r["name"] for r in RUNS}
        if unknown:
            sys.exit(f"unknown run(s): {sorted(unknown)}\navailable: {[r['name'] for r in RUNS]}")
        todo = [(r, SEED) for r in sel]

    for r, s in todo:
        print(f"  {r.get('rank','?'):>2}  {r['name']:<20s} seed={s}  {r['label']}")
    print(f"{'=' * 74}\n")

    res = [run_one(r, seed=s, with_test=args.with_test) for r, s in todo]

    os.makedirs(PROJECT_DIR, exist_ok=True)
    out = os.path.join(PROJECT_DIR, "adfl_summary_seeds.json" if args.seeds else "adfl_summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    summarise(res, out)
