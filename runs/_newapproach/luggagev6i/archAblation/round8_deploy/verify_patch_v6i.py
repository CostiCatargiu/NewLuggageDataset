#!/usr/bin/env python3
"""
Verify (and optionally install) the SWA / LB-TAL / custom-module patch on the
machine that actually TRAINS.

WHY THIS EXISTS
===============
The training scripts' preflight checked that `tal.py` has the assigner class and
that `default.yaml` accepts the keys. It never checked that `loss.py` READS the
flag. Those are different files, and only loss.py decides whether the assigner
is ever constructed:

    loss.py:456   use_lbtal = bool(getattr(h, "use_lbtal", False)) and tal_topk2 is None
    loss.py:465   if use_lbtal:
    loss.py:466       self.assigner = LevelBalancedTaskAlignedAssigner(...)

If loss.py is the stock file, `use_lbtal=True` is accepted by the config system,
printed in the run header, validated by preflight — and silently ignored. Every
run becomes the same configuration under a different name.

That is not hypothetical. It is the failure mode that would explain rounds 4-6
of this project: ten runs with different budget labels whose overall mAP50-95
spans 55.89..56.46 (sd 0.19) and whose large mAP spans 54.51..60.66 (sd 2.11).

RUN THIS BEFORE AND AFTER EVERY REINSTALL OR `pip install -e .` OF ULTRALYTICS.
A reinstall silently reverts every patched file.

Usage
=====
    python verify_patch_v6i.py --ref /path/to/ultralytics26/ultralytics
    python verify_patch_v6i.py --ref /path/to/ultralytics26/ultralytics --install
    python verify_patch_v6i.py --ref ... --runtime      # also build a live assigner

`--ref` points at the ultralytics PACKAGE directory of the patched reference
tree (the one containing utils/, cfg/, nn/), not at the repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys

# file -> (marker, minimum count) that must be present for the patch to be live
FILES = {
    "utils/loss.py": [
        ("def swa_weight", 1),
        ("use_lbtal", 2),
        ("LevelBalancedTaskAlignedAssigner", 2),
    ],
    "utils/tal.py": [
        ("class LevelBalancedTaskAlignedAssigner", 1),
        ("def set_strides", 1),
    ],
    "cfg/default.yaml": [
        ("use_lbtal", 1),
        ("lbtal_level_topk", 1),
        ("alpha_start", 1),
        ("area_weight_mode", 1),
    ],
    "nn/modules/block.py": [
        ("class DySample", 1),
        ("class ZGGlobalContext2", 1),
        ("class ZGDSConv", 1),
    ],
    "nn/modules/__init__.py": [("DySample", 1)],
    "nn/tasks.py": [("elif m is DySample:", 1)],
}

# loss.py is the one that decides whether ANY of the loss work takes effect
CRITICAL = "utils/loss.py"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def live_package():
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def check(live_dir, ref_dir):
    rows, missing = [], []
    for rel, markers in FILES.items():
        lp, rp = os.path.join(live_dir, rel), os.path.join(ref_dir, rel)
        if not os.path.exists(lp):
            rows.append((rel, "ABSENT", "-", "-", "-"))
            missing.append(rel)
            continue
        try:
            src = open(lp, encoding="utf-8", errors="ignore").read()
        except Exception as e:
            rows.append((rel, f"UNREADABLE {e}", "-", "-", "-"))
            missing.append(rel)
            continue
        bad = [m for m, n in markers if src.count(m) < n]
        state = "PATCHED" if not bad else "STOCK"
        if bad:
            missing.append(rel)
        same = "-"
        if os.path.exists(rp):
            same = "yes" if md5(lp) == md5(rp) else "no"
        rows.append((rel, state, str(len(src.splitlines())), md5(lp), same))
    return rows, missing


def runtime_check():
    """Build a live assigner and confirm loss.py would actually install it."""
    import inspect

    try:
        import ultralytics.utils.tal as T
        from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
    except Exception as e:
        print(f"  [runtime] import failed: {e}")
        return False
    ok = True
    has_cls = hasattr(T, "LevelBalancedTaskAlignedAssigner")
    reads = "use_lbtal" in inspect.getsource(v8DetectionLoss.__init__)
    swa = hasattr(BboxLoss, "swa_weight")
    print(f"  [runtime] tal.py  has LevelBalancedTaskAlignedAssigner : {has_cls}")
    print(f"  [runtime] loss.py reads use_lbtal in v8DetectionLoss   : {reads}   <-- THE ONE THAT MATTERS")
    print(f"  [runtime] BboxLoss has swa_weight (SWA port)           : {swa}")
    ok = has_cls and reads and swa
    if has_cls:
        A = T.LevelBalancedTaskAlignedAssigner
        b = A(topk=10, level_topk_mode="fixed", level_topk={4: 2, 8: 3, 16: 4, 32: 4},
              min_level_k=1)._per_level_budget([4.0, 8.0, 16.0, 32.0])
        got = {int(k): v for k, v in b.items()}
        good = got == {4: 2, 8: 3, 16: 4, 32: 4}
        print(f"  [runtime] budget resolves on a 4-level model          : {got}  {'OK' if good else 'WRONG'}")
        ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="patched ultralytics PACKAGE dir (contains utils/, cfg/, nn/)")
    ap.add_argument("--install", action="store_true", help="copy missing/stock files from --ref, with .bak backups")
    ap.add_argument("--runtime", action="store_true", help="also import and build a live assigner")
    a = ap.parse_args()

    ref = os.path.abspath(a.ref)
    if not os.path.isdir(os.path.join(ref, "utils")):
        sys.exit(f"[ABORT] --ref does not look like an ultralytics package dir: {ref}")
    live = live_package()
    if live is None:
        sys.exit("[ABORT] cannot import ultralytics on this machine")

    print(f"\n  live ultralytics : {live}")
    print(f"  reference (patch): {ref}\n")
    rows, missing = check(live, ref)
    print(f"  {'file':<24}{'state':<10}{'lines':>7}{'md5':>14}{'==ref':>8}")
    print("  " + "-" * 63)
    for rel, state, n, h, same in rows:
        print(f"  {rel:<24}{state:<10}{n:>7}{h:>14}{same:>8}")

    if not missing:
        print("\n  ALL PATCHED.")
        if a.runtime and not runtime_check():
            sys.exit("\n  [FAIL] files look patched but the runtime check disagrees.")
        print("\n  Loss-axis runs on this machine are REAL.")
        return 0

    print(f"\n  NOT PATCHED: {missing}")
    if CRITICAL in missing:
        print(f"\n  *** {CRITICAL} is stock. ***")
        print("  use_lbtal / alpha_start / area_weight_mode are accepted by the config")
        print("  system and IGNORED by the loss. Any run whose only difference is a loss")
        print("  or assigner hyperparameter is a REPLICATE of the stock run, not a variant.")
        print("  Architecture (yaml) differences are unaffected and still real.")

    if not a.install:
        print("\n  Re-run with --install to copy the patched files (originals kept as .bak).")
        return 1

    print()
    for rel in missing:
        src, dst = os.path.join(ref, rel), os.path.join(live, rel)
        if not os.path.exists(src):
            print(f"  [skip] no reference copy of {rel}")
            continue
        if os.path.exists(dst) and not os.path.exists(dst + ".bak"):
            shutil.copy2(dst, dst + ".bak")
            print(f"  backed up {rel} -> {rel}.bak")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  installed {rel}")

    print("\n  re-checking:")
    rows, missing = check(live, ref)
    for rel, state, n, h, same in rows:
        print(f"  {rel:<24}{state:<10}{n:>7}{h:>14}{same:>8}")
    if missing:
        sys.exit(f"\n  [FAIL] still not patched: {missing}")
    print("\n  ALL PATCHED.")
    if a.runtime and not runtime_check():
        sys.exit("\n  [FAIL] runtime check disagrees.")
    print("\n  Confirm on the next training run: the log must contain")
    print("    LB-TAL active (per-level top-k) | mode=fixed budget={...}")
    print("  If that banner is absent, the assigner is still not installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
