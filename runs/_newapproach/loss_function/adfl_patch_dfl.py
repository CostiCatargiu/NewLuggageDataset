#!/usr/bin/env python3
"""
Anisotropic-DFL inference patch — makes DFL.forward apply the same per-edge
range scale that loss.py applies during training.

WHY THIS FILE EXISTS
--------------------
A-DFL encodes edge distances as  t_e = d_e / s_e  and decodes them as
d_e = E[bin] * s_e. loss.py handles the training side. But every detection
path — val, predict, track, and all export formats — decodes boxes through
`DFL.forward` in ultralytics/nn/modules/block.py. If that path does not apply
the same s_e, the model trains on one box parameterisation and is evaluated on
another, and mAP collapses in a way that looks like "the method doesn't work".

HOW IT WORKS
------------
The scale is stored as a REGISTERED BUFFER on the DFL module. That means it is
written into the checkpoint's state_dict, so `YOLO('best.pt')` restores it
automatically and inference is correct without any flag. Default is all-ones,
which is bit-identical to stock DFL.

USAGE
-----
1. Install the patch once per process, BEFORE building the model:

       import adfl_patch_dfl
       adfl_patch_dfl.install()

       from ultralytics import YOLO
       model = YOLO("yolov12s.pt")
       adfl_patch_dfl.set_scales(model, w_scale=0.5, h_scale=1.0)
       model.train(..., use_adfl=True, adfl_w_scale=0.5, adfl_h_scale=1.0)

   (run_adfl_ablation.py does all of this for you.)

2. Verify a trained checkpoint carries the expected scale:

       python adfl_patch_dfl.py --check runs_adfl/adfl_w050/weights/best.pt

3. Self-test that the patch is exact at neutral settings:

       python adfl_patch_dfl.py --selftest

IMPORTANT: `install()` must be called before the model is constructed, because
it replaces `DFL.__init__` to add the buffer.
"""

import argparse
import sys

import torch
import torch.nn as nn

_INSTALLED = False
_ORIG_INIT = None
_ORIG_FORWARD = None


def install():
    """Patch ultralytics DFL to carry a per-edge range scale buffer."""
    global _INSTALLED, _ORIG_INIT, _ORIG_FORWARD
    if _INSTALLED:
        return True

    from ultralytics.nn.modules import block as _block

    DFL = _block.DFL
    _ORIG_INIT = DFL.__init__
    _ORIG_FORWARD = DFL.forward

    def __init__(self, c1=16):
        _ORIG_INIT(self, c1)
        # (1, 4, 1) broadcast over (b, 4, anchors); order = left, top, right, bottom
        self.register_buffer("edge_scale", torch.ones(1, 4, 1), persistent=True)

    def forward(self, x):
        b, _, a = x.shape
        d = self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        es = getattr(self, "edge_scale", None)
        if es is None:
            return d                     # checkpoint predates the patch
        return d * es.to(dtype=d.dtype, device=d.device)

    DFL.__init__ = __init__
    DFL.forward = forward
    _INSTALLED = True
    print("[A-DFL] DFL patched: per-edge range scale active (default 1.0 = stock)")
    return True


def uninstall():
    global _INSTALLED
    if not _INSTALLED:
        return
    from ultralytics.nn.modules import block as _block
    _block.DFL.__init__ = _ORIG_INIT
    _block.DFL.forward = _ORIG_FORWARD
    _INSTALLED = False


def _iter_dfl(model):
    m = getattr(model, "model", model)
    m = getattr(m, "model", m)
    for mod in m.modules():
        if type(mod).__name__ == "DFL":
            yield mod


def set_scales(model, w_scale=1.0, h_scale=1.0, verbose=True):
    """Write the per-edge scale into every DFL module of `model`.

    Edge order is bbox2dist's: (left, top, right, bottom), so width edges are
    columns 0 and 2, height edges 1 and 3.
    """
    vec = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=torch.float).view(1, 4, 1)
    n = 0
    for mod in _iter_dfl(model):
        if not hasattr(mod, "edge_scale"):
            mod.register_buffer("edge_scale", torch.ones(1, 4, 1), persistent=True)
        mod.edge_scale.data.copy_(vec.to(mod.edge_scale.device))
        n += 1
    if n == 0:
        raise RuntimeError("no DFL module found — did you call install() before building the model?")
    if verbose:
        print(f"[A-DFL] set edge_scale on {n} DFL module(s): "
              f"w={w_scale} h={h_scale} -> {vec.flatten().tolist()}")
    return n


def get_scales(model):
    out = []
    for mod in _iter_dfl(model):
        es = getattr(mod, "edge_scale", None)
        out.append(None if es is None else es.flatten().tolist())
    return out


# ---------------------------------------------------------------------------
def _check(path):
    install()
    from ultralytics import YOLO
    m = YOLO(path)
    sc = get_scales(m)
    print(f"\ncheckpoint: {path}")
    if not sc:
        print("  no DFL module found")
        return 1
    for i, s in enumerate(sc):
        if s is None:
            print(f"  DFL[{i}]: NO edge_scale buffer  -> checkpoint predates the patch (stock DFL)")
        else:
            w, h = s[0], s[1]
            tag = "STOCK (all ones)" if s == [1.0, 1.0, 1.0, 1.0] else f"A-DFL  w={w}  h={h}"
            print(f"  DFL[{i}]: edge_scale={s}   {tag}")
            if s[0] != s[2] or s[1] != s[3]:
                print("      !! left/right or top/bottom disagree — that is almost certainly a bug")
    return 0


def _selftest():
    install()
    from ultralytics.nn.modules.block import DFL

    torch.manual_seed(0)
    reg_max, b, a = 16, 2, 37
    x = torch.randn(b, reg_max * 4, a)

    d = DFL(reg_max)
    out_neutral = d(x)

    # reference: unpatched maths
    ref = d.conv(x.view(b, 4, reg_max, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    ok1 = torch.equal(out_neutral, ref)
    print(f"neutral == stock (bit-identical): {ok1}")

    set_scales_module(d, 0.5, 1.25)
    out_scaled = d(x)
    expect = ref * torch.tensor([0.5, 1.25, 0.5, 1.25]).view(1, 4, 1)
    ok2 = torch.allclose(out_scaled, expect, atol=0, rtol=0)
    print(f"scaled == ref * [w,h,w,h]:       {ok2}")

    # encode/decode round trip
    s_w, s_h = 0.5, 1.25
    d_true = torch.tensor([[3.7, 9.1, 4.2, 8.8]])
    scales = torch.tensor([s_w, s_h, s_w, s_h]).view(1, 4)
    enc = d_true / scales
    dec = enc * scales
    ok3 = torch.allclose(dec, d_true, atol=1e-6)
    print(f"encode/decode round trip exact:  {ok3}")

    # buffer survives a state_dict round trip
    sd = d.state_dict()
    d2 = DFL(reg_max)
    d2.load_state_dict(sd)
    ok4 = torch.equal(d2.edge_scale, d.edge_scale)
    print(f"edge_scale survives state_dict:  {ok4}")

    allok = ok1 and ok2 and ok3 and ok4
    print("\nSELFTEST:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def set_scales_module(dfl_mod, w_scale, h_scale):
    vec = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=torch.float).view(1, 4, 1)
    if not hasattr(dfl_mod, "edge_scale"):
        dfl_mod.register_buffer("edge_scale", torch.ones(1, 4, 1), persistent=True)
    dfl_mod.edge_scale.data.copy_(vec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="WEIGHTS", help="print the edge_scale stored in a checkpoint")
    ap.add_argument("--selftest", action="store_true", help="verify the patch maths")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if args.check:
        sys.exit(_check(args.check))
    ap.print_help()
