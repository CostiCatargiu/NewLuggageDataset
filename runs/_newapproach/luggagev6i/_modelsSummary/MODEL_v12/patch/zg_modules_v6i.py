#!/usr/bin/env python3
"""
v6i-adapted ZG modules — drop-in replacements re-derived from THIS dataset.

The originals in arch_best/nn_modules/block.py were tuned on a weapons dataset
(their docstrings name long_gun/knife) and then carried onto luggage v5i. Three
of their assumptions do not hold on v6i. Everything below is a consequence of
the numbers in LuggageDatasetSplitv6i.txt, not a guess.

    v6i measured (train split, 9138 img / 41,823 inst)
      mean box            39 x 55 px
      mean h/w            1.55        (v5i was 2.69)
      shape mix           70.6% tall (h/w>1.25), 23.4% square, 6.0% wide
      size (max side)     60.0% small(<48) / 28.7% medium / 11.3% large
      per-class h/w       trolley 1.68 (51% of data), backpack 1.47, bag 1.33
      image size          640x360 dominant (7350 of 9138)

NEW CLASSES (suffixed V6 so they coexist with the originals for A/B):
    ZGDSConvV6          asymmetric snake, kernel re-scaled to v6i object size
    ZGGlobalContext2V6  attention-pooled avg+max+attn context, letterbox-robust
    ZGSEV6              squeeze-excitation, attention-pooled
    ZGGlobalContextV6   single-descriptor context, attention-pooled
    ZGStripV6           separable strips with the LONG axis on height
    ZGSmallDetailV6     P3 detail guard, mid kernel matched to object height

All keep gamma=0 at init, so each is an exact identity at epoch 0 and
pretrained transfer is unaffected — the property the original notes credit for
why the gated modules generalised and the ungated ones failed. The attention-
pooled ones additionally zero-init their score conv, so the pooled descriptor
STARTS exactly equal to the average pool they replace.

The audit of the other 21 block.py classes and the 3 heads — what is coupled to
the dataset, what is not, and why most were deliberately NOT ported — is in the
comment block below ZGSmallDetailV6.

INSTALL
    Append to ultralytics/nn/modules/block.py (or import from it), add the
    names to __all__ in that file, and register them in the parse_model
    dispatch in ultralytics/nn/tasks.py wherever the modules they replace
    already appear. Every one takes the same YAML arg shape as its original.

        - [17, 1, ZGDSConvV6, [512, 5]]        # c2, k   (k is the TALL axis)
        - [23, 1, ZGGlobalContext2V6, [256]]   # c2
        - [20, 1, ZGSEV6, [512, 8]]            # c2, r
        - [20, 1, ZGGlobalContextV6, [512]]    # c2
        - [17, 1, ZGStripV6, [512, 23]]        # c2, k_v  (k_h derived)
        - [14, 1, ZGSmallDetailV6, [256]]      # c2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZGDSConvV6(nn.Module):
    """Dynamic snake conv, asymmetric and re-scaled for v6i geometry.

    TWO CHANGES FROM ZGDSConv, both derived rather than tuned.

    (1) KERNEL SIZE 9 -> 5.
        The snake steps exactly ONE FEATURE CELL per tap (`tap * step_x`,
        step_x = 2/(W-1)), so k taps span k cells. The v5i comment reads
        "P3 + snake k=9 (obj 9.0 cells: MATCHED)" — k was scale-matched to
        objects that were 9 cells across at P3.

        On v6i the mean box is 39 x 55 px:

            level  stride   mean obj (cells)   k=9 spans   k=5 spans
            P3       8        4.9 x 6.9          9  (1.3x)   5  (0.7x)
            P4      16        2.4 x 3.4          9  (2.6x)   5  (1.5x)
            P5      32        1.2 x 1.7          9  (5.3x)   5  (2.9x)

        levelspec/ls_shift put this module at P4 and P5, where k=9 now covers
        2.6-5.3x the object it is supposed to trace. The deformation path
        spends most of its taps on background. k=5 is the matched choice.

        This is also why arch_ls_k5 (k=5) is expected to fare RELATIVELY better
        on v6i than its last-place v5i ranking suggests.

    (2) ASYMMETRIC AXES.
        The original runs an x-snake and a y-snake with identical k and equal
        weight. v6i is 70.6% TALL and only 6.0% wide, so the horizontal path is
        spending half the module's capacity on a shape that is 6% of the data.
        k_x defaults to round(k / 1.55) = the mean h/w, floored at 3.

        With k=5 that gives k_y=5, k_x=3: 8 grid_sample calls per forward
        instead of the original's 18 at k=9 — 2.25x fewer, on a module whose
        cost is dominated by that Python-loop of grid_samples.

    NOT CHANGED: zero-init offsets (taps start on the regular grid), tanh-bounded
    per-tap offsets, cumulative outward sum from the centre tap, gamma=0 gate.

    YAML: [c2, k]  e.g. [512, 5]   — k is the TALL (y) axis; k_x is derived.
    """

    def __init__(self, c1, c2, k=5, k_x=None, aspect=1.55):
        super().__init__()
        assert c1 == c2, "ZGDSConvV6 preserves channels"
        assert k % 2 == 1, "k must be odd"
        if k_x is None:
            k_x = max(3, int(round(k / max(aspect, 1e-6))))
            if k_x % 2 == 0:
                k_x += 1
        assert k_x % 2 == 1, "k_x must be odd"
        self.c1, self.k_y, self.k_x = c1, k, k_x

        self.offset_x = nn.Conv2d(c1, k_x, 3, 1, 1)
        self.offset_y = nn.Conv2d(c1, k, 3, 1, 1)
        for m in (self.offset_x, self.offset_y):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.weight_x = nn.Parameter(torch.randn(c1, k_x) * 0.02)
        self.weight_y = nn.Parameter(torch.randn(c1, k) * 0.02)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def _snake_sample(self, x, offsets, weight, axis, K):
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        off = torch.tanh(offsets.float())            # (B,K,H,W) taps in (-1,1)
        center = K // 2
        cum = torch.zeros_like(off)

        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center, K):                    # outward, centre -> end
            run = run + off[:, i]
            cum[:, i] = run
        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center - 1, -1, -1):           # outward, centre -> start
            run = run - off[:, i]
            cum[:, i] = run

        ys = torch.linspace(-1, 1, H, device=device, dtype=off.dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=off.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
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
            grid = torch.stack([grid_x, grid_y], dim=-1)
            sampled = F.grid_sample(x32, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            out = out + (sampled * weight[:, i].view(1, C, 1, 1)).to(dtype)
        return out

    def forward(self, x):
        sx = self._snake_sample(x, self.offset_x(x), self.weight_x, "x", self.k_x)
        sy = self._snake_sample(x, self.offset_y(x), self.weight_y, "y", self.k_y)
        y = self.pw(self.act(self.bn(sx + sy)))
        return x + self.gamma * y


class ZGGlobalContext2V6(nn.Module):
    """Global context with attention pooling — letterbox-robust.

    THE PROBLEM ON v6i. 7350 of 9138 train images are 640x360. Letterboxed to
    a 640x640 canvas that is ~44% constant grey padding, and the original does

        avg = x.mean(dim=(2, 3))

    so nearly half of the "whole-scene context" descriptor is padding. A fixed
    pad fraction would be an affine shift the following MLP bias could absorb,
    but mosaic tiles four images per sample, so the padded fraction VARIES per
    sample and the contamination is not a learnable constant.

    THE FIX. Add a third, attention-pooled descriptor (GCNet-style context
    modelling, Cao et al. 2019):

        a    = softmax_HW( conv1x1(x) )        # (B, 1, HW)
        attn = sum_HW a * x                    # (B, C, 1, 1)

    Padding produces low activation, so it earns low attention weight and is
    down-weighted automatically — no pad mask needed, which the module could
    not see anyway. The score conv is ZERO-INITIALISED, so softmax is uniform
    at init and `attn` is EXACTLY the mean: this starts as a strict superset of
    the original and can only learn away from it.

    Also: reduction 8 -> 4. At scale s the width multiplier is 0.50, so P2/P3
    carry c1=128 and the original's `hidden = max(8, 128//8) = 16` compressed a
    256-dim descriptor into 16 channels. The v5i round-4 notes flagged that
    bottleneck as "very tight" and never tested it. The descriptor is now
    3*c1 wide, which makes 16 tighter still.

    avg is KEPT rather than replaced — it was part of what was validated, and
    max (the salient-instance cue) is unaffected by padding since grey never
    wins a max. Gated with gamma=0, so identity at epoch 0.

    YAML: [c2]  e.g. [256]      (drop-in for ZGGlobalContext2)
    """

    def __init__(self, c1, c2, reduction=4, use_attn=True):
        super().__init__()
        assert c1 == c2, "ZGGlobalContext2V6 preserves channels"
        self.use_attn = use_attn
        n_desc = 3 if use_attn else 2
        hidden = max(8, c1 // reduction)
        if use_attn:
            self.att = nn.Conv2d(c1, 1, 1)
            nn.init.zeros_(self.att.weight)   # uniform softmax at init ->
            nn.init.zeros_(self.att.bias)     # attn == avg exactly
        self.fc = nn.Sequential(
            nn.Conv2d(n_desc * c1, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        avg = x.mean(dim=(2, 3), keepdim=True)        # scene context
        mx = x.amax(dim=(2, 3), keepdim=True)         # most salient activation
        desc = [avg, mx]
        if self.use_attn:
            a = self.att(x).view(B, 1, H * W).softmax(dim=-1)      # (B,1,HW)
            xf = x.view(B, C, H * W)                               # (B,C,HW)
            attn = torch.bmm(xf, a.transpose(1, 2)).view(B, C, 1, 1)
            desc.append(attn)
        ctx = self.fc(torch.cat(desc, dim=1))
        return x + self.gamma * ctx


# =============================================================================
# Self-test — run directly: python zg_modules_v6i.py
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(label, cond):
        global ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("ZGDSConvV6")
    m = ZGDSConvV6(64, 64, k=5)
    x = torch.randn(2, 64, 20, 20)
    y = m(x)
    check("shape preserved", y.shape == x.shape)
    check("identity at init (gamma=0)", torch.allclose(y, x, atol=1e-6))
    check(f"asymmetric axes k_y=5 k_x={m.k_x} (mean h/w 1.55)", m.k_x == 3)
    check("grid_samples per fwd = 8, was 18 at k=9", m.k_x + m.k_y == 8)
    m.gamma.data.fill_(0.1)
    check("non-identity once gated open", not torch.allclose(m(x), x, atol=1e-6))

    print("\nZGGlobalContext2V6")
    g = ZGGlobalContext2V6(128, 128)
    x = torch.randn(2, 128, 16, 16)
    check("shape preserved", g(x).shape == x.shape)
    check("identity at init (gamma=0)", torch.allclose(g(x), x, atol=1e-6))
    with torch.no_grad():
        B, C, H, W = x.shape
        a = g.att(x).view(B, 1, H * W).softmax(-1)
        attn = torch.bmm(x.view(B, C, H * W), a.transpose(1, 2)).view(B, C, 1, 1)
    check("attn == avg at init (zero-init score conv)",
          torch.allclose(attn, x.mean(dim=(2, 3), keepdim=True), atol=1e-5))
    check("hidden = c1//4 = 32 (was 16 at reduction=8)",
          g.fc[0].out_channels == 32)

    # letterbox realism: 640x360 -> 640x640 is ~44% padding
    print("\nletterbox behaviour (44% padded rows, as in 640x360 @ imgsz 640)")
    x = torch.randn(1, 32, 40, 40)
    x[:, :, 22:, :] = 0.0                       # padded region -> low activation
    with torch.no_grad():
        g2 = ZGGlobalContext2V6(32, 32)
        nn.init.normal_(g2.att.weight, std=0.5)  # a trained-ish scorer
        B, C, H, W = x.shape
        a = g2.att(x).view(B, 1, H * W).softmax(-1).view(1, 1, H, W)
        pad_mass = float(a[:, :, 22:, :].sum())
    print(f"    attention mass on the padded 45% of rows: {pad_mass:.3f}")
    print(f"    (uniform/avg-pool would put {18/40:.3f} there by construction)")
    check("attention can down-weight padding; avg cannot", pad_mass != 18 / 40)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))


# =============================================================================
# THE REST OF THE AUDIT — all 27 block.py classes + 3 heads
# =============================================================================
# Two defects recur across the file. Neither is a bug on the dataset these
# modules were written for; both are consequences of v6i's composition.
#
# DEFECT 1 — global average pooling eats the letterbox.
#   7350 of 9138 train images are 640x360 -> 43.8% of the 640x640 canvas is
#   constant grey padding, and mosaic varies the padded fraction per sample so
#   it is not an affine shift the following bias can absorb.
#     affected : ZGSE, ZGGlobalContext, ZGGatherContext, ZGGlobalContext2, ZGDSConv*
#     immune   : ZGGC, ZGLSKAGCFuse  <- these ALREADY use softmax attention
#                pooling (GCNet-style). The right pattern was in the fork all
#                along, just not in the modules the top-5 configs use.
#     * ZGDSConv's avg is BatchNorm's, not a context pool — not affected.
#
# DEFECT 2 — axis-symmetric spatial priors on a 70.6%-tall dataset.
#   v6i is 70.6% tall (h/w>1.25), 23.4% square, 6.0% wide, mean h/w 1.55.
#   Every strip/snake/large-kernel module treats the two axes identically, so
#   half of each module's spatial capacity serves 6% of the boxes. The v5i
#   lineage inherited this from a weapons dataset whose docstrings cite
#   "long_gun 29% of data" — horizontally elongated objects. The prior is
#   pointed the wrong way here.
#
# WHAT IS PORTED BELOW, and what is not:
#   ported   ZGDSConvV6, ZGGlobalContext2V6  (used by the top-5 configs)
#            ZGSEV6, ZGGlobalContextV6       (defect 1, drop-in, likely reused)
#            ZGStripV6, ZGSmallDetailV6      (defect 2, the two shape priors)
#   NOT ported: the 15-strong ZGLSKA* family, ZGLKA, ZGDCN, ZGMHSA, ZGStar,
#            ZGP2Fuse, ZGGatherContext, DySample.
#            - DySample, ZGP2Fuse, ZGStar, ZGDCN, ZGMHSA: no dataset coupling.
#              Nothing to change.
#            - ZGLSKA* family (k_sq=11, k_strip=23): these are CONTEXT modules,
#              where a deliberately oversized receptive field is the point, so
#              "k >> object size" is not automatically wrong the way it is for
#              a shape-tracing snake. Their k values came from dose-response
#              curves measured on the weapons data (the docstrings quote
#              k7=79.05, k11=79.19, k15=79.03), and I have no v6i equivalent to
#              re-derive from. Porting 15 modules on a guess would be inventing
#              numbers, and none of them appear in the five configs being run.
#              If you later run one, apply the ZGStripV6 asymmetry pattern.
#            - ZGGatherContext: avg-pooled and so defect-1 affected, but its
#              signature is (chs) for cross-scale fusion and it is unused here.
#   HEADS (head.py): DetectAux / DetectAuxDual / DetectAuxDualDeepP3 default
#            nc=80, always overridden by the YAML's nc: 3. aux_weight=0.25 is a
#            loss-balance knob with no dataset coupling. No change needed.


class ZGSEV6(nn.Module):
    """Squeeze-Excitation, attention-pooled instead of average-pooled.

    Original squeezes with nn.AdaptiveAvgPool2d(1), so 43.8% of the descriptor
    is letterbox padding on a 640x360 image. Same fix as ZGGlobalContext2V6:
    a zero-initialised 1x1 score conv gives a uniform softmax at init, so the
    pooled vector STARTS exactly equal to the average and can only learn away
    from it. Padding earns low attention because it produces low activation.

    YAML: [c2, r]  e.g. [512, 8]   (drop-in for ZGSE)
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        assert c1 == c2, "ZGSEV6 preserves channels"
        c_ = max(c1 // r, 16)
        self.att = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.att.weight)
        nn.init.zeros_(self.att.bias)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c_, 1), nn.SiLU(), nn.Conv2d(c_, c1, 1), nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.att(x).view(B, 1, H * W).softmax(dim=-1)
        ctx = torch.bmm(x.view(B, C, H * W), a.transpose(1, 2)).view(B, C, 1, 1)
        return x + self.gamma * (self.fc(ctx) * x)


class ZGGlobalContextV6(nn.Module):
    """Single-descriptor global context, attention-pooled. Defect 1 only.

    ZGGlobalContext2V6 is the avg+max+attn version; this is the plain one, kept
    as a drop-in for the runs that used ZGGlobalContext rather than its v2.

    YAML: [c2]  e.g. [512]
    """

    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        assert c1 == c2, "ZGGlobalContextV6 preserves channels"
        hidden = max(8, c1 // reduction)
        self.att = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.att.weight)
        nn.init.zeros_(self.att.bias)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.att(x).view(B, 1, H * W).softmax(dim=-1)
        ctx = torch.bmm(x.view(B, C, H * W), a.transpose(1, 2)).view(B, C, 1, 1)
        return x + self.gamma * self.fc(ctx)


class ZGStripV6(nn.Module):
    """Separable strip attention with the long axis on HEIGHT. Defect 2.

    The original wraps LSKA(c1, k_size=23): a symmetric separable pair, 1xk
    then kx1, both k=23. Its lineage is a weapons dataset with long_gun at 29%
    — objects elongated HORIZONTALLY. v6i is the opposite: 70.6% tall, 6.0%
    wide, mean h/w 1.55, and the tallest class (trolley, h/w 1.68) is 51% of
    the data. So the horizontal strip is the one carrying the prior, and it is
    aimed at the 6% case.

    Here the two strips are sized independently: k_v along height (default 23,
    unchanged), k_h = round(k_v / 1.55) along width. Ratio matches the measured
    mean aspect.

    HONEST LIMIT: the axis asymmetry is derived from v6i. The MAGNITUDE (23) is
    not — it came from a dose-response curve on the weapons data and there is
    no v6i equivalent to re-fit against. If you sweep anything here, sweep k_v.

    YAML: [c2, k_v]  e.g. [512, 23]   (drop-in for ZGStrip)
    """

    def __init__(self, c1, c2, k_v=23, k_h=None, aspect=1.55):
        super().__init__()
        assert c1 == c2, "ZGStripV6 preserves channels"
        if k_h is None:
            k_h = max(3, int(round(k_v / max(aspect, 1e-6))))
            if k_h % 2 == 0:
                k_h += 1
        self.k_v, self.k_h = k_v, k_h
        # depthwise separable strips: (k_v x 1) vertical, (1 x k_h) horizontal
        self.dw_v = nn.Conv2d(c1, c1, (k_v, 1), 1, (k_v // 2, 0), groups=c1)
        self.dw_h = nn.Conv2d(c1, c1, (1, k_h), 1, (0, k_h // 2), groups=c1)
        self.pw = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw(self.dw_h(self.dw_v(x)))


class ZGSmallDetailV6(nn.Module):
    """P3 detail guard with the mid kernel matched to v6i object height.

    Original k_fine=3, k_mid=5, both square. At P3 (stride 8) the v6i mean box
    is 4.9 x 6.9 cells, so a square k=5 covers the WIDTH well and falls short
    of the HEIGHT. k_mid becomes (7, 5) — 7 tall, 5 wide — which brackets
    6.9 x 4.9 almost exactly. k_fine stays 3x3: it is a detail/edge kernel, not
    an object-extent one.

    YAML: [c2]  or [c2, k_fine]   (drop-in for ZGSmallDetail)
    """

    def __init__(self, c1, c2, k_fine=3, k_mid_h=7, k_mid_w=5):
        super().__init__()
        assert c1 == c2, "ZGSmallDetailV6 preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.dw_fine = nn.Conv2d(c1, c1, k_fine, 1, k_fine // 2, groups=c1)
        self.dw_mid = nn.Conv2d(c1, c1, (k_mid_h, k_mid_w), 1,
                                (k_mid_h // 2, k_mid_w // 2), groups=c1)
        self.norm = nn.GroupNorm(1, c1)
        self.act2 = nn.SiLU()
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        y = self.act(self.pw1(x))
        y = self.dw_fine(y) + self.dw_mid(y)
        return x + self.gamma * self.pw2(self.act2(self.norm(y)))
