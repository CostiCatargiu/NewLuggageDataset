#!/usr/bin/env python3
"""
PATCH the installed ultralytics with the three custom NN modules.

=============================================================================
WHY THIS EXISTS
=============================================================================
The loss port (SWA / LB-TAL) and the NN-module port went into DIFFERENT copies
of ultralytics. run_yolo26_arch_v6i.py aborted with

    custom classes not importable: ['DySample','ZGGlobalContext2','ZGDSConv']

because the tree Python imports is not the tree the modules were added to.
This script finds the ultralytics that is ACTUALLY on the import path and
applies the same three edits to it:

  1. nn/modules/block.py      + DySample, ZGGlobalContext2, ZGDSConv, + __all__
  2. nn/modules/__init__.py   + re-export
  3. nn/tasks.py              + import, + ZG* in base_modules,
                                + a DySample branch in parse_model

It is IDEMPOTENT: run it twice and the second run reports "already patched"
and changes nothing. Every file is backed up to <file>.prepatch once.

Run it with the SAME interpreter you train with:

    python patch_ultralytics_modules.py            # patch + verify
    python patch_ultralytics_modules.py --check    # verify only, no writes
    python patch_ultralytics_modules.py --revert   # restore the .prepatch files
"""

import importlib
import os
import shutil
import sys

CLASSES = '''

# =============================================================================
# CUSTOM SMALL-OBJECT MODULES — ported from the YOLOv12 fork (arch_best/nn)
# =============================================================================
# DySample          content-aware upsampler, P3 -> P2 in the FPN top-down path
# ZGGlobalContext2  gated avg+max global-context broadcast, gamma = 0 at init
# ZGDSConv          zero-gated dynamic snake conv, shape prior for elongated GTs
#
# All three are CHANNEL-PRESERVING and IDENTITY AT INITIALISATION, so adding
# them to a pretrained graph does not perturb it at epoch 0.
# =============================================================================

class ZGGlobalContext2(nn.Module):
    """globalctx + max-pool branch (avg+max global descriptor) -- the last-arch
    refinement of the round-38 winner.

    ZGGlobalContext built its global descriptor from AVERAGE pool only (whole-scene
    context). This adds a MAX-pool branch: avg captures context, max captures the
    single most SALIENT activation -- the cue small/rare weapon instances spike on
    but that averages away in avg-pool (the CBAM/BAM channel-attention insight).
    Both descriptors are concatenated -> MLP -> gated additive broadcast.

    Stays gentle/gated/identity-init (gamma=0 -> identity at epoch 0), the property
    that made globalctx generalize and the aggressive variants (gather, wfv2_p3)
    fail. It ENRICHES the winning module rather than stacking a second one (which
    is what sank r39). Channel-preserving, ~0 inference cost.

    YAML: drop-in single-input, e.g.  - [21, 1, ZGGlobalContext2, [512]]
    """

    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        assert c1 == c2, "ZGGlobalContext2 preserves channels"
        hidden = max(8, c1 // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(2 * c1, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)           # scene context
        mx = x.amax(dim=(2, 3), keepdim=True)            # most salient activation
        ctx = self.fc(torch.cat([avg, mx], dim=1))       # (B, C, 1, 1)
        return x + self.gamma * ctx                      # gated additive broadcast


class DySample(nn.Module):
    """Dynamic content-aware upsampler (DySample, ICCV 2023), 'lp' style.

    Drop-in replacement for nn.Upsample(scale_factor=2) in the FPN top-down path.
    Instead of fixed nearest/bilinear interpolation, it predicts per-location
    sampling offsets and gathers via grid_sample -> recovers fine spatial detail
    when upsampling toward the P3 (small-object) level, where nearest-neighbour
    upsampling blurs exactly the detail small objects need. Channel-preserving;
    the offset conv is near-zero-init so it starts ~ bilinear (safe transfer).

    YAML: drop-in for nn.Upsample ->  - [-1, 1, DySample, [2]]   (scale=2)
    """

    def __init__(self, c1, scale=2, groups=4):
        super().__init__()
        assert c1 % groups == 0, "DySample: channels must be divisible by groups"
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(c1, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(
            1, self.groups, 1, 1).reshape(1, -1, 1, 1)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(
            1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        xg = x.reshape(B * self.groups, -1, H, W)
        return F.grid_sample(xg, coords, mode="bilinear", align_corners=False,
                             padding_mode="border").view(B, -1, self.scale * H, self.scale * W)


class ZGDSConv(nn.Module):
    """Zero-gated Dynamic Snake Convolution — shape prior for elongated objects.

    y = x + gamma * pw(act(bn(DSConv_x(x) + DSConv_y(x)))), gamma = 0 at init.

    Dynamic Snake Convolution (Qi et al., 2023, originally for tubular
    vessel segmentation) deforms a 1D kernel along a single axis with
    CUMULATIVE per-tap offsets, so the sampling path "snakes" along whatever
    elongated structure is present. weapon_noaug's long_gun/knife classes are
    intrinsically elongated/thin -- this encodes a different adaptivity prior
    than ZGDCN (independent unconstrained 2D offsets per tap, no path
    continuity) or ZGLSKA (fixed kernel shape). Implemented with
    F.grid_sample (pure PyTorch, no torchvision.ops -> avoids the
    deform_conv2d crash seen with ZGDCN).

    Two branches (kernel snaking along x, kernel snaking along y), each:
      1. predict per-tap offsets (1 scalar per tap) from a 3x3 conv,
         zero-initialized so offsets start at 0 (taps sit on the regular
         grid at init);
      2. cumulative-sum offsets outward from the center tap (snake path);
      3. bilinear-sample the input along the deformed 1D path;
      4. depthwise-combine the K sampled taps -> 1 output per channel.

    YAML args: [c2, k]  e.g. [512, 9]
    """

    def __init__(self, c1, c2, k=9):
        super().__init__()
        assert c1 == c2, "ZGDSConv preserves channels"
        assert k % 2 == 1, "k must be odd"
        self.c1 = c1
        self.k = k
        self.offset_x = nn.Conv2d(c1, k, 3, 1, 1)
        self.offset_y = nn.Conv2d(c1, k, 3, 1, 1)
        nn.init.zeros_(self.offset_x.weight)
        nn.init.zeros_(self.offset_x.bias)
        nn.init.zeros_(self.offset_y.weight)
        nn.init.zeros_(self.offset_y.bias)
        self.weight_x = nn.Parameter(torch.randn(c1, k) * 0.02)
        self.weight_y = nn.Parameter(torch.randn(c1, k) * 0.02)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def _snake_sample(self, x, offsets, weight, axis):
        B, C, H, W = x.shape
        K = self.k
        device, dtype = x.device, x.dtype
        off = torch.tanh(offsets.float())  # (B,K,H,W), bounded to (-1,1) taps
        center = K // 2
        cum = torch.zeros_like(off)

        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center, K):
            run = run + off[:, i]
            cum[:, i] = run

        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center - 1, -1, -1):
            run = run - off[:, i]
            cum[:, i] = run

        ys = torch.linspace(-1, 1, H, device=device, dtype=off.dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=off.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        step_x = 2.0 / max(W - 1, 1)
        step_y = 2.0 / max(H - 1, 1)

        out = torch.zeros_like(x)
        x32 = x.float()
        for i in range(K):
            tap = i - center
            if axis == "x":
                grid_x = base_x.unsqueeze(0) + tap * step_x + cum[:, i] * step_x
                grid_y = base_y.unsqueeze(0).expand(B, -1, -1)
            else:
                grid_x = base_x.unsqueeze(0).expand(B, -1, -1)
                grid_y = base_y.unsqueeze(0) + tap * step_y + cum[:, i] * step_y
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)
            sampled = F.grid_sample(x32, grid, mode="bilinear", padding_mode="border", align_corners=True)
            w = weight[:, i].view(1, C, 1, 1)
            out = out + (sampled * w).to(dtype)
        return out

    def forward(self, x):
        sx = self._snake_sample(x, self.offset_x(x), self.weight_x, "x")
        sy = self._snake_sample(x, self.offset_y(x), self.weight_y, "y")
        y = self.act(self.bn(sx + sy))
        y = self.pw(y)
        return x + self.gamma * y
'''

NAMES = ("DySample", "ZGDSConv", "ZGGlobalContext2")


def find_pkg():
    """Directory of the ultralytics package Python will actually import."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: ultralytics is not importable by this interpreter.")
    return list(spec.submodule_search_locations)[0]


def backup(path):
    bak = path + ".prepatch"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"    backed up -> {os.path.basename(bak)}")


def patch_block(path):
    s = open(path, encoding="utf-8").read()
    if "class DySample(" in s:
        print("    block.py       already has the classes")
        return False
    backup(path)
    s += CLASSES
    i = s.index("__all__ = (")
    j = s.index(")", i)
    s = s[:j] + "".join(f'    "{n}",\n' for n in NAMES) + s[j:]
    open(path, "w", encoding="utf-8").write(s)
    print("    block.py       + 3 classes, + __all__")
    return True


def patch_init(path):
    s = open(path, encoding="utf-8").read()
    if "DySample" in s:
        print("    __init__.py    already exports them")
        return False
    backup(path)
    i = s.index("from .block import (")
    j = s.index(")", i)
    s = s[:j] + "".join(f"    {n},\n" for n in NAMES) + s[j:]
    k = s.index("__all__ = (")
    l = s.index(")", k)
    s = s[:l] + "".join(f'    "{n}",\n' for n in NAMES) + s[l:]
    open(path, "w", encoding="utf-8").write(s)
    print("    __init__.py    + re-export")
    return True


def patch_tasks(path):
    s = open(path, encoding="utf-8").read()
    changed = False
    if "elif m is DySample:" in s and "ZGDSConv," in s:
        print("    tasks.py       already patched")
        return False
    backup(path)

    if "    ZGDSConv,\n" not in s:
        i = s.index("from ultralytics.nn.modules import (")
        j = s.index(")", i)
        s = s[:j] + "".join(f"    {n},\n" for n in NAMES) + s[j:]
        changed = True

    # channel-preserving c1/c2 modules -> base_modules gives args=[c1,c2,*rest]
    if "ZGGlobalContext2,\n            ZGDSConv," not in s:
        anchor = None
        for a in ("            A2C2f,\n        }", "            C2fCIB,\n        }",
                  "            C3x,\n        }"):
            if a in s:
                anchor = a
                break
        if anchor is None:
            sys.exit("ERROR: could not locate base_modules in parse_model. Patch tasks.py by hand.")
        s = s.replace(anchor, anchor.replace("\n        }",
                      "\n            ZGGlobalContext2,\n            ZGDSConv,\n        }"), 1)
        changed = True

    # DySample takes (c1, scale): channel-preserving, no c2 in the YAML args
    if "elif m is DySample:" not in s:
        branch = ("        elif m is DySample:\n"
                  "            # content-aware upsampler, drop-in for nn.Upsample.\n"
                  "            # Channel-preserving: the YAML carries only the scale.\n"
                  "            c1 = c2 = ch[f]\n"
                  "            args = [c1, *args]\n")
        anchor = None
        for a in ("        elif m is Depth:\n", "        elif m is v10Detect:\n",
                  "        elif m is CBLinear:\n"):
            if a in s:
                anchor = a
                break
        if anchor is None:
            sys.exit("ERROR: could not locate the parse_model elif chain. Patch tasks.py by hand.")
        s = s.replace(anchor, branch + anchor, 1)
        changed = True

    open(path, "w", encoding="utf-8").write(s)
    print("    tasks.py       + import, + base_modules, + DySample branch")
    return changed


def verify():
    for m in list(sys.modules):
        if m.startswith("ultralytics"):
            del sys.modules[m]
    ok = True
    try:
        import ultralytics.nn.modules as M
        for n in NAMES:
            good = hasattr(M, n)
            print(f"    import {n:<18}{'OK' if good else 'FAIL'}")
            ok &= good
    except Exception as e:
        print(f"    import FAILED: {e}")
        return False
    try:
        from ultralytics import YOLO
        mdl = YOLO("yolo26-lsshift.yaml")
        found = sorted({type(x).__name__ for x in mdl.model.modules()} & set(NAMES))
        print(f"    build yolo26-lsshift.yaml -> custom layers {found}")
        ok &= len(found) == 3
    except Exception as e:
        print(f"    build skipped/failed: {e}")
        print("    (fine if the yaml is not in cfg/models/26 — the runner writes its own)")
    return ok


def main():
    pkg = find_pkg()
    print(f"\n  ultralytics on THIS interpreter's import path:\n    {pkg}\n")
    files = {"block": os.path.join(pkg, "nn", "modules", "block.py"),
             "init": os.path.join(pkg, "nn", "modules", "__init__.py"),
             "tasks": os.path.join(pkg, "nn", "tasks.py")}
    for k, v in files.items():
        if not os.path.isfile(v):
            sys.exit(f"ERROR: {v} not found — is this really an ultralytics package?")

    if "--revert" in sys.argv:
        for v in files.values():
            bak = v + ".prepatch"
            if os.path.exists(bak):
                shutil.copy2(bak, v)
                print(f"    reverted {os.path.basename(v)}")
        return

    if "--check" in sys.argv:
        print("  verify only, no writes:")
        sys.exit(0 if verify() else 1)

    print("  patching:")
    patch_block(files["block"])
    patch_init(files["init"])
    patch_tasks(files["tasks"])
    print("\n  verifying:")
    ok = verify()
    print("\n  " + ("DONE — run_yolo26_arch_v6i.py should now pass preflight."
                    if ok else "FAILED — see above; --revert restores the originals."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
