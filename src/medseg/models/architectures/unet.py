"""Conventional U-Net baseline for binary segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from medseg.models.base import BaseSegmentationModel
from medseg.typing import ModelOutput, TensorLike


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block with max-pooling followed by a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution, skip fusion, and double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 projection to segmentation logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetRonneberger2015(BaseSegmentationModel):
    """Conventional four-level U-Net baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        feature_channels: Sequence[int] | None = None,
        encoder_depth: int = 4,
    ) -> None:
        super().__init__()

        # Keep the current config-based builder compatible without importing
        # the config package into this standalone architecture module.
        if not isinstance(in_channels, int):
            config = in_channels
            in_channels = int(getattr(config, "in_channels"))
            out_channels = int(getattr(config, "out_channels"))
            feature_channels = tuple(getattr(config, "feature_channels"))
            encoder_depth = int(getattr(config, "encoder_depth"))

        if encoder_depth != 4:
            raise ValueError("The baseline U-Net implementation currently expects encoder_depth=4.")

        channels = list(feature_channels or (64, 128, 256, 512, 1024))
        if len(channels) < 5:
            raise ValueError("feature_channels must contain at least 5 channel sizes.")

        channels = channels[:5]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = tuple(channels)
        self.encoder_depth = encoder_depth
        self.stem = DoubleConv(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.up1 = UpBlock(channels[4], channels[3])
        self.up2 = UpBlock(channels[3], channels[2])
        self.up3 = UpBlock(channels[2], channels[1])
        self.up4 = UpBlock(channels[1], channels[0])
        self.outc = OutConv(channels[0], out_channels)

    def forward(self, inputs: TensorLike) -> ModelOutput:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("UNetRonneberger2015 expects a torch.Tensor input.")

        x1 = self.stem(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return {"logits": logits}


UNet = UNetRonneberger2015

__all__ = ["UNet", "UNetRonneberger2015"]
