# ultralytics/nn/modules/fsus.py
"""
Feature Sharpening via Upsampled Supervision (FSUS)

Training-only auxiliary loss that forces upsampled FPN features to preserve
the spatial structure of native-resolution backbone features.
Zero inference cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class FeatureSharpener(nn.Module):
    """
    Auxiliary module: computes sharpening loss between upsampled and native features.
    Pure passthrough at inference. Zero cost.

    Args:
        c_up (int): Channels in upsampled features
        c_native (int): Channels in native backbone features
        c_proj (int): Shared projection dimension (default 64)
        loss_weight (float): Weight for sharpening loss (default 0.05)
    """

    def __init__(self, c_up, c_native, c_proj=64, loss_weight=0.05):
        super().__init__()
        self.loss_weight = loss_weight
        self.c_proj = c_proj

        self.proj_up = nn.Sequential(
            nn.Conv2d(c_up, c_proj, 1, bias=False),
            nn.BatchNorm2d(c_proj),
        )
        self.proj_native = nn.Sequential(
            nn.Conv2d(c_native, c_proj, 1, bias=False),
            nn.BatchNorm2d(c_proj),
        )

        self.edge_conv = nn.Conv2d(c_proj, c_proj, 3, padding=1, groups=c_proj, bias=False)
        self._init_edge_weights()

        # Use register_buffer for the loss flag — avoids deepcopy issues
        # Store loss value as a plain Python float, not a tensor
        self._aux_loss_value = None

    def _init_edge_weights(self):
        """Initialize with Sobel-like kernels using .data to keep leaf status."""
        channels = self.edge_conv.weight.shape[0]
        w = torch.zeros(channels, 1, 3, 3)
        for i in range(channels):
            if i % 2 == 0:
                w[i, 0, 0, :] = torch.tensor([-1.0, -2.0, -1.0])
                w[i, 0, 2, :] = torch.tensor([1.0, 2.0, 1.0])
            else:
                w[i, 0, :, 0] = torch.tensor([-1.0, -2.0, -1.0])
                w[i, 0, :, 2] = torch.tensor([1.0, 2.0, 1.0])
        self.edge_conv.weight.data.copy_(w)

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying non-leaf tensors."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Deep copy all attributes except _aux_loss_value
        for k, v in self.__dict__.items():
            if k == '_aux_loss_value':
                setattr(result, k, None)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result

    def get_loss(self):
        """Retrieve and clear stored auxiliary loss."""
        loss = self._aux_loss_value
        self._aux_loss_value = None
        return loss

    def forward(self, x):
        """
        Args:
            x: list of [upsampled_features, native_features]
        Returns:
            upsampled_features unchanged (passthrough)
        """
        upsampled, native = x[0], x[1]

        if self.training:
            self._compute_sharpening_loss(upsampled, native)

        return upsampled

    def _compute_sharpening_loss(self, upsampled, native):
        """Compute spatial structure + channel correlation loss."""
        proj_up = self.proj_up(upsampled)
        proj_nat = self.proj_native(native)

        # Spatial structure loss (edge matching)
        edges_up = self.edge_conv(proj_up)
        edges_nat = self.edge_conv(proj_nat)

        e_up = F.normalize(edges_up.flatten(2), dim=-1)
        e_nat = F.normalize(edges_nat.flatten(2), dim=-1)
        spatial_loss = 1.0 - (e_up * e_nat).sum(dim=-1).mean()

        # Channel correlation loss (Gram matrix)
        B, C, H, W = proj_up.shape
        n_pixels = H * W

        up_flat = proj_up.flatten(2)
        nat_flat = proj_nat.flatten(2)

        gram_up = torch.bmm(up_flat, up_flat.transpose(1, 2)) / n_pixels
        gram_nat = torch.bmm(nat_flat, nat_flat.transpose(1, 2)) / n_pixels

        channel_loss = F.mse_loss(gram_up, gram_nat)

        # Store as a tensor but detach-and-reattach pattern to avoid graph issues
        combined = self.loss_weight * (spatial_loss + 0.5 * channel_loss)
        self._aux_loss_value = combined
