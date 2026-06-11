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

__all__ = ["ZGLSKA", "ZGGC", "ZGSE", "ZGMHSA"]


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


if __name__ == "__main__":
    # Sanity: forward shapes + exact identity at init (gamma == 0).
    torch.manual_seed(0)
    for cls, c, args in [
        (ZGLSKA, 128, (5,)), (ZGLSKA, 256, (7,)),
        (ZGGC, 512, ()), (ZGSE, 256, ()), (ZGMHSA, 512, (4,)),
    ]:
        m = cls(c, c, *args).eval()
        x = torch.randn(2, c, 16, 16)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape
        assert torch.equal(y, x), f"{cls.__name__} not identity at init!"
        n = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__:8s} c={c:4d} args={args}  params={n:,}  identity-at-init: OK")
