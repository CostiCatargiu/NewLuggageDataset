"""Person re-identification features (OSNet) for InferenceAndSaveLabelsLuggage.py.

Why this exists: the colour histogram the tracker used to decide "is this the same person?"
cannot tell people apart on AVSSS07 (dark clothes, one light). Measured on ~1,900 pairs from
that video, different people scored a median 0.87 against 0.95 for the same person. OSNet is
a network trained to recognise the same person across cameras, poses and occlusions.

The model code below is OSNet from deep-person-reid / torchreid (MIT licence):
    Zhou et al., Omni-Scale Feature Learning for Person Re-Identification, ICCV 2019.
    Zhou et al., Learning Generalisable Omni-Scale Representations for Person Re-ID, TPAMI 2021.
It is copied unchanged so no extra package is needed besides torch (already required by
ultralytics). Weights: osnet_x1_0 trained on MSMT17, from the torchreid model zoo; they are
downloaded once with gdown (pip install gdown), or put the file next to this script by hand.

MIT License -- Copyright (c) Kaiyang Zhou and torchreid contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions: The above copyright notice
and this permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import division, absolute_import

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_URLS = {
    # torchreid model zoo (the same files boxmot uses)
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
}
INPUT_HW = (256, 128)  # height, width the model was trained on
MIN_BOX_H = 24  # px; smaller crops give meaningless features
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


##########
# OSNet (unchanged from torchreid)
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        IN=False,
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))



def osnet_x1_0(num_classes=1000, **kwargs):
    return OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                 channels=[64, 256, 384, 512], **kwargs)


def osnet_x0_25(num_classes=1000, **kwargs):
    return OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                 channels=[16, 64, 96, 128], **kwargs)


ARCHS = {"osnet_x1_0": osnet_x1_0, "osnet_x0_25": osnet_x0_25}


##########
# Loading and feature extraction
##########
def _download(path):
    name = os.path.basename(path)
    url = WEIGHT_URLS.get(name)
    if url is None:
        raise FileNotFoundError(f"re-ID weights not found: {path}")
    try:
        import gdown
    except ImportError:
        raise FileNotFoundError(
            f"re-ID weights not found: {path}\n"
            f"    install gdown to download them automatically:  pip install gdown\n"
            f"    or download by hand from {url}\n"
            f"    and save the file as {path}")
    print(f"  re-ID   : downloading {name} (once) ...")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    gdown.download(url, path, quiet=False)
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        raise FileNotFoundError(f"download of {name} failed -- get it by hand from {url}")


def _load_weights(model, path):
    """Load a torchreid checkpoint; the identity classifier (training-only) is skipped."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    matched = OrderedDict()
    for k, v in state.items():
        k = k[7:] if k.startswith("module.") else k
        if k in own and own[k].size() == v.size():
            matched[k] = v
    missing = [k for k in own if k not in matched and not k.startswith("classifier.")]
    if missing:
        raise RuntimeError(f"{os.path.basename(path)} does not fit this architecture "
                           f"({len(missing)} layers missing, e.g. {missing[:3]})")
    own.update(matched)
    model.load_state_dict(own)


class PersonReID:
    """embed(frame_bgr, boxes_xyxy) -> list of L2-normalised float32 vectors (None if too small)."""

    def __init__(self, weights, arch="osnet_x1_0", device="0", batch=32):
        if not os.path.exists(weights):
            _download(weights)
        self.model = ARCHS[arch]()
        _load_weights(self.model, weights)
        dev = str(device)
        if dev.isdigit():
            dev = f"cuda:{dev}"
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.batch = batch
        self.dim = 512  # both architectures end in a 512-d embedding
        self.name = f"{arch} ({os.path.basename(weights)}) on {self.device}"

    def _crop(self, frame, bb):
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bb)
        if y2 - y1 < MIN_BOX_H or x2 - x1 < 6:
            return None
        xi1, yi1 = int(max(0, x1)), int(max(0, y1))
        xi2, yi2 = int(min(fw, x2)), int(min(fh, y2))
        if xi2 - xi1 < 4 or yi2 - yi1 < 8:
            return None
        crop = cv2.resize(frame[yi1:yi2, xi1:xi2], (INPUT_HW[1], INPUT_HW[0]),
                          interpolation=cv2.INTER_LINEAR)
        crop = (crop[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        return crop.transpose(2, 0, 1)

    @torch.no_grad()
    def embed(self, frame, boxes):
        crops = [self._crop(frame, bb) for bb in boxes]
        idx = [i for i, c in enumerate(crops) if c is not None]
        out = [None] * len(boxes)
        for s in range(0, len(idx), self.batch):
            part = idx[s:s + self.batch]
            x = torch.from_numpy(np.stack([crops[i] for i in part])).to(self.device)
            x = x.half() if self.half else x
            f = F.normalize(self.model(x).float(), dim=1).cpu().numpy()
            for j, i in enumerate(part):
                out[i] = f[j].astype(np.float32)
        return out
