"""Early segmentation head used on intermediate encoder features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EarlySegmentationHead(nn.Module):
    """Reference FeaturesSegmenter with configurable encoder level upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        level: int = 3,
    ) -> None:
        super().__init__()
        if level not in {0, 1, 2, 3}:
            raise ValueError("EarlySegmentationHead level must be between 0 and 3.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.upsample_factor = 2**level
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, out_channels, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected ESH input with shape [B, {self.in_channels}, H, W], "
                f"received {tuple(features.shape)}."
            )
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        logits = self.conv3(x)
        if self.upsample_factor == 1:
            return logits
        return F.interpolate(
            logits,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=False,
        )


FeaturesSegmenter = EarlySegmentationHead

__all__ = ["EarlySegmentationHead", "FeaturesSegmenter"]
