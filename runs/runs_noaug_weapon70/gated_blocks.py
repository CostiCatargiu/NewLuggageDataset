"""
Zero-Init Gated (ZG) blocks for YOLOv12s — designed from the failure analysis
of the previous 14 architecture experiments.

WHY THE PREVIOUS ROUND FAILED
-----------------------------
1. Every change REPLACED a pretrained block (A2C2f -> C2fLSKA etc.), so the
   new block started from random init while the rest of the net was
   pretrained. In 80 epochs on 13k images it never recovered the lost
   pretrained knowledge -> gains <= +0.58%.
2. Capacity was added at P3, but the dataset has only ~2.2% small objects
   (374 train instances). 68% of boxes are LARGE -> P4/P5 is where the data is.

THE FIX: zero-init gated residual branches, APPENDED after the existing
layers instead of replacing them:

    y = x + gamma * branch(x),   gamma initialized to 0 (per-channel)

Consequences:
  * At epoch 0 the network is EXACTLY the baseline (gamma=0 -> passthrough).
  * Layers 0-20 keep their YAML indices -> `model.load("yolov12s.pt")`
    transfers ALL backbone+head weights (check the "Transferred x/y items"
    log line — it should be near-complete except Detect).
  * The optimizer only "opens the gate" where the new branch actually
    reduces loss. Worst realistic case is baseline performance, not -1.3%.
  (Same idea as ReZero / zero-init gamma in GCNet & ViT adapters.)

REGISTRATION — ALREADY DONE in the fork at runs/ultralytics/
(blocks added to nn/modules/block.py, exported in nn/modules/__init__.py,
imported + registered in nn/tasks.py parse_model — outer tuple only).
This file remains as standalone documentation + self-test:
run `python gated_blocks.py` to verify shapes and identity-at-init.

Reference procedure (if porting to another fork):
--------------------------------------------------
1. Copy these classes into ultralytics/nn/modules/block.py
   (or `from .gated_blocks import *` there) and add
   "ZGLSKA", "ZGGC", "ZGSE", "ZGMHSA" to __all__.
2. Import them in ultralytics/nn/modules/__init__.py and ultralytics/nn/tasks.py.
3. In parse_model() in tasks.py, add ZGLSKA, ZGGC, ZGSE, ZGMHSA to the OUTER
   module tuple where C2fLSKA is registered (the branch doing
   `c1, c2 = ch[f], args[0]` + width scaling, then `args = [c1, c2, *args[1:]]`).
   IMPORTANT: do NOT add them to the inner tuple that does `args.insert(2, n)`
   (the repeats tuple for C2f-like blocks) — ZG blocks take no `n` argument.
   They take (c1, c2, ...) and require c2 == c1 (they preserve channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ZGLSKA", "ZGGC", "ZGSE", "ZGMHSA", "ZGStar", "ZGDSConv", "ZGLSKASG",
           "ZGLSKAStripFuse", "ZGLSKAMultiDil"]


class LKA(nn.Module):
    """Decomposed Large-Kernel Attention (VAN-style).

    5x5 depthwise -> kxk depthwise dilated(3) -> 1x1 pointwise, used as a
    multiplicative attention map. Effective RF ~ 4 + 3*(k-1) + 1 px at the
    feature stride (k=7 -> ~23 cells; at P4/stride16 that is ~368 px of
    context in the 640 image).
    """

    def __init__(self, c, k=7):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c)
        self.dwd = nn.Conv2d(c, c, k, 1, ((k - 1) // 2) * 3, groups=c, dilation=3)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.pw(self.dwd(self.dw(x))) * x


class ZGLSKA(nn.Module):
    """Zero-gated large-kernel context branch.  y = x + gamma * f(x), gamma=0.

    f = 1x1 -> SiLU -> LKA(k) -> 1x1. Unlike C2fLSKA this does NOT replace
    the pretrained A2C2f block — it is appended after it.

    YAML args: [c2, k]   e.g.  [512, 7]  (c2 is width-scaled by parse_model)
    """

    def __init__(self, c1, c2, k=7):
        super().__init__()
        assert c1 == c2, "ZGLSKA preserves channels (set YAML c2 = input channels)"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = LKA(c1, k)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.lka(self.act(self.pw1(x))))


class ZGGC(nn.Module):
    """Zero-gated Global Context block (GCNet-style) — for P5 / large objects.

    Softmax-pooled global context vector -> bottleneck transform ->
    broadcast-added back, behind a zero gate. Adds image-level context
    (scene type, co-occurring cues) at negligible cost.

    YAML args: [c2, r]   e.g.  [1024, 8]
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        assert c1 == c2, "ZGGC preserves channels"
        self.attn = nn.Conv2d(c1, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(c1, max(c1 // r, 16), 1),
            nn.GroupNorm(1, max(c1 // r, 16)),
            nn.SiLU(),
            nn.Conv2d(max(c1 // r, 16), c1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        w_ = self.attn(x).view(b, 1, h * w).softmax(dim=-1)          # b,1,hw
        ctx = (x.view(b, c, h * w) @ w_.transpose(1, 2)).view(b, c, 1, 1)
        return x + self.gamma * self.transform(ctx)


class ZGSE(nn.Module):
    """Zero-gated Squeeze-Excitation. Cheapest control variant.

    y = x + gamma * (SE(x) * x). If even this opens its gates and helps,
    gating works; if heavier blocks don't beat it, complexity isn't paying.

    YAML args: [c2, r]   e.g.  [512, 8]
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        assert c1 == c2, "ZGSE preserves channels"
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // r, 16), 1),
            nn.SiLU(),
            nn.Conv2d(max(c1 // r, 16), c1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * (self.fc(x) * x)


class ZGMHSA(nn.Module):
    """Zero-gated multi-head self-attention — P5 only (20x20 = 400 tokens).

    With 1.3 obj/img and 68% large boxes, global token mixing at P5 can
    relate the weapon to its surroundings. DW 3x3 on V as positional encoding
    (as in YOLO PSA blocks).

    YAML args: [c2, num_heads]   e.g.  [1024, 4]
    """

    def __init__(self, c1, c2, num_heads=4):
        super().__init__()
        assert c1 == c2, "ZGMHSA preserves channels"
        assert c1 % num_heads == 0
        self.nh = num_heads
        self.scale = (c1 // num_heads) ** -0.5
        self.qkv = nn.Conv2d(c1, c1 * 3, 1)
        self.pe = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1)
        self.proj = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.nh, c // self.nh, h * w)
        q, k, v = qkv.unbind(1)                                # each: b,nh,d,hw
        attn = (q.transpose(-2, -1) @ k) * self.scale          # b,nh,hw,hw
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        out = out + self.pe(v.reshape(b, c, h, w))
        return x + self.gamma * self.proj(out)


class ZGStar(nn.Module):
    """Zero-gated STAR block — multiplicative feature mixing (StarNet, 2024).

    y = x + gamma * proj_out(act(proj1(z)) * proj2(z)), z = BN(DWConv7x7(x)).
    NO spatial-attention map (unlike every block above): two 1x1 projections
    to a wide hidden dim are multiplied element-wise (the "star operation"),
    implicitly realizing a high-dimensional polynomial feature expansion in
    low-dim space (Ma et al., StarNet 2024). Multiplicative interaction
    instead of additive attention -- an orthogonal mechanism to the LSKA
    family.

    YAML args: [c2, hidden_mult]   e.g.  [512, 4]
    """

    def __init__(self, c1, c2, hidden_mult=4):
        super().__init__()
        assert c1 == c2, "ZGStar preserves channels"
        c_hidden = c1 * hidden_mult
        self.dw = nn.Conv2d(c1, c1, 7, 1, 3, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.proj1 = nn.Conv2d(c1, c_hidden, 1)
        self.proj2 = nn.Conv2d(c1, c_hidden, 1)
        self.act = nn.SiLU()
        self.proj_out = nn.Conv2d(c_hidden, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z = self.bn(self.dw(x))
        y = self.act(self.proj1(z)) * self.proj2(z)
        y = self.proj_out(y)
        return x + self.gamma * y


class ZGDSConv(nn.Module):
    """Zero-gated Dynamic Snake Convolution — shape prior for elongated objects.

    y = x + gamma * pw(act(bn(DSConv_x(x) + DSConv_y(x)))).
    Dynamic Snake Convolution (Qi et al., 2023, tubular vessel segmentation):
    two 1D kernels (along x, along y) with CUMULATIVE per-tap offsets that
    "snake" along whatever elongated structure is present. long_gun/knife
    are intrinsically elongated/thin -- a different adaptivity axis than
    ZGDCN (unconstrained per-tap 2D offsets, no path continuity) or ZGLSKA
    (fixed kernel shape). Pure-PyTorch via F.grid_sample (no torchvision.ops
    -> avoids the deform_conv2d crash seen with ZGDCN).

    YAML args: [c2, k]   e.g.  [512, 7]
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
        b, c, h, w = x.shape
        k = self.k
        device, dtype = x.device, x.dtype
        off = torch.tanh(offsets.float())
        center = k // 2
        cum = torch.zeros_like(off)

        run = torch.zeros(b, h, w, device=device, dtype=off.dtype)
        for i in range(center, k):
            run = run + off[:, i]
            cum[:, i] = run

        run = torch.zeros(b, h, w, device=device, dtype=off.dtype)
        for i in range(center - 1, -1, -1):
            run = run - off[:, i]
            cum[:, i] = run

        ys = torch.linspace(-1, 1, h, device=device, dtype=off.dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=off.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
        step_x = 2.0 / max(w - 1, 1)
        step_y = 2.0 / max(h - 1, 1)

        out = torch.zeros_like(x)
        x32 = x.float()
        for i in range(k):
            tap = i - center
            if axis == "x":
                grid_x = base_x.unsqueeze(0) + tap * step_x + cum[:, i] * step_x
                grid_y = base_y.unsqueeze(0).expand(b, -1, -1)
            else:
                grid_x = base_x.unsqueeze(0).expand(b, -1, -1)
                grid_y = base_y.unsqueeze(0) + tap * step_y + cum[:, i] * step_y
            grid = torch.stack([grid_x, grid_y], dim=-1)
            sampled = F.grid_sample(x32, grid, mode="bilinear", padding_mode="border", align_corners=True)
            wgt = weight[:, i].view(1, c, 1, 1)
            out = out + (sampled * wgt).to(dtype)
        return out

    def forward(self, x):
        sx = self._snake_sample(x, self.offset_x(x), self.weight_x, "x")
        sy = self._snake_sample(x, self.offset_y(x), self.weight_y, "y")
        y = self.act(self.bn(sx + sy))
        y = self.pw(y)
        return x + self.gamma * y


class ZGLSKASG(nn.Module):
    """Spatially-gated ZGLSKA -- round 9, built on the round-7 dose-response winner.

    y = x + gamma * sigmoid(spatial_gate(x)) * f(x), gamma = 0 at init.
    f = 1x1 -> SiLU -> LKA(k) -> 1x1   (identical branch to ZGLSKA, k=11
    confirmed as the dose-response peak: k7=79.05, k11=79.19, k15=79.03).

    ZGLSKA's gate is per-channel only -- it applies UNIFORMLY across the
    whole P4 feature map (the network learns "how much" large-kernel
    context to mix in per channel, but not "where"). ZGLSKASG adds a
    per-pixel sigmoid mask from a small 3x3 conv, so the same k=11 branch
    can be suppressed over background and emphasized around object-shaped
    regions.

    Identity-at-init is preserved the same way as ZGLSKA: gamma is
    zero-initialized (y = x regardless of the spatial gate's value at
    step 0). spatial_gate is ALSO zero-initialized (weight+bias=0), so
    sigmoid(.) = 0.5 uniformly at init -- a constant no-op multiplier that
    only starts differentiating spatially once gamma opens.

    YAML args: [c2, k]   e.g.  [512, 11]
    """

    def __init__(self, c1, c2, k=11):
        super().__init__()
        assert c1 == c2, "ZGLSKASG preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = LKA(c1, k)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.spatial_gate = nn.Conv2d(c1, 1, 3, 1, 1)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        f = self.pw2(self.lka(self.act(self.pw1(x))))
        g = torch.sigmoid(self.spatial_gate(x))  # (B,1,H,W), starts at 0.5 everywhere
        return x + self.gamma * g * f


class LSKAStrip(nn.Module):
    """Separable strip-kernel attention (1xk + kx1), mirrors conv.LSKA used by
    ZGStrip in the fork: channel-mix 1x1 -> horizontal kxk dw -> vertical kxk
    dw -> 1x1 -> sigmoid -> multiplicative gate on the input.
    """

    def __init__(self, c, k_size=23):
        super().__init__()
        pad = k_size // 2
        self.conv0 = nn.Conv2d(c, c, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(c)
        self.conv_h = nn.Conv2d(c, c, (1, k_size), padding=(0, pad), groups=c, bias=False)
        self.conv_v = nn.Conv2d(c, c, (k_size, 1), padding=(pad, 0), groups=c, bias=False)
        self.conv1 = nn.Conv2d(c, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        attn = self.act(self.bn0(self.conv0(x)))
        attn = self.conv_h(attn)
        attn = self.conv_v(attn)
        attn = self.bn1(self.conv1(attn))
        return x * torch.sigmoid(attn)


class ZGLSKAStripFuse(nn.Module):
    """Round 10 -- fuse the two best single-branch shapes (k11 square + strip23)
    in ONE gated branch via channel-split, instead of stacking two gates.

    y = x + gamma * pw2( cat[ LKA(k_sq)(z1), LSKAStrip(k_strip)(z2) ] ),
    gamma = 0 at init, z = act(pw1(x)) split in half along channels.

    Round 7's dose-response found k=11 (square, dilated) is the peak
    (79.19%) and strip k=23 (1x23+23x1, for elongated objects) is a close
    second (79.07%) -- two DIFFERENT receptive-field shapes, both near-best.
    Stacking them as two SEPARATE gated branches hurt (k11+GC@P4 = 78.66,
    two competing gammas). This instead routes half the channels through
    each shape inside a SINGLE branch under ONE gamma -- a different failure
    mode than stacking.

    YAML args: [c2, k_sq, k_strip]   e.g.  [512, 11, 23]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23):
        super().__init__()
        assert c1 == c2, "ZGLSKAStripFuse preserves channels"
        assert c1 % 2 == 0, "ZGLSKAStripFuse requires an even channel count"
        c_half = c1 // 2
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = LKA(c_half, k_sq)
        self.strip = LSKAStrip(c_half, k_size=k_strip)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        y = torch.cat([self.lka(z1), self.strip(z2)], dim=1)
        return x + self.gamma * self.pw2(y)


class LKAMultiDil(nn.Module):
    """Multi-dilation LKA primitive for ZGLSKAMultiDil: 5x5 depthwise -> two
    parallel dilated depthwise convs at different (k, dilation) -> summed ->
    1x1 pointwise, used as a multiplicative attention map.
    """

    def __init__(self, c, k_a=7, d_a=2, k_b=11, d_b=3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c)
        self.dwd_a = nn.Conv2d(c, c, k_a, 1, ((k_a - 1) // 2) * d_a, groups=c, dilation=d_a)
        self.dwd_b = nn.Conv2d(c, c, k_b, 1, ((k_b - 1) // 2) * d_b, groups=c, dilation=d_b)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        z = self.dw(x)
        return self.pw(self.dwd_a(z) + self.dwd_b(z)) * x


class ZGLSKAMultiDil(nn.Module):
    """Round 10 -- single-branch multi-scale LKA: k7/dilation2 AND k11/dilation3
    fused into one attention map under one gamma, instead of picking one k.

    y = x + gamma * pw2(LKAMultiDil(act(pw1(x)))), gamma = 0 at init.

    Round 7's dose-response over a single LKA(k, dilation=3): k7=79.05,
    k11=79.19 (peak), k15=79.03 -- fairly flat near the peak, suggesting
    both the k7 (RF~17 cells) and k11 (RF~35 cells) scales carry useful
    signal individually. Fuses both scales inside ONE branch/gate as a
    single-branch "ensemble" instead of an either/or choice.

    YAML args: [c2, k_a, d_a, k_b, d_b]   e.g.  [512, 7, 2, 11, 3]
    """

    def __init__(self, c1, c2, k_a=7, d_a=2, k_b=11, d_b=3):
        super().__init__()
        assert c1 == c2, "ZGLSKAMultiDil preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = LKAMultiDil(c1, k_a, d_a, k_b, d_b)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.lka(self.act(self.pw1(x))))


if __name__ == "__main__":
    # Sanity: forward shapes + exact identity at init (gamma == 0).
    torch.manual_seed(0)
    for cls, c, args in [
        (ZGLSKA, 128, (5,)), (ZGLSKA, 256, (7,)),
        (ZGGC, 512, ()), (ZGSE, 256, ()), (ZGMHSA, 512, (4,)),
        (ZGStar, 256, (4,)), (ZGDSConv, 64, (7,)),
        (ZGLSKASG, 256, (11,)),
        (ZGLSKAStripFuse, 256, (11, 23)), (ZGLSKAMultiDil, 256, (7, 2, 11, 3)),
    ]:
        m = cls(c, c, *args).eval()
        x = torch.randn(2, c, 16, 16)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape
        assert torch.equal(y, x), f"{cls.__name__} not identity at init!"
        n = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__:8s} c={c:4d} args={args}  params={n:,}  identity-at-init: OK")
