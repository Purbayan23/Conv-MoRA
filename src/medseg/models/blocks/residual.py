"""Residual convolution blocks used by the reference-style U-Net2D."""

from __future__ import annotations

import torch
from torch import nn


class PreActivationConv2d(nn.Module):
    """BatchNorm-ReLU-convolution pre-activation unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU()
        self.layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(self.activation(self.norm(inputs)))


class ResidualBlock2d(nn.Module):
    """Two pre-activation convolutions with an identity or 1x1 shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            PreActivationConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            PreActivationConv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.shortcut(inputs) + self.conv_path(inputs)


__all__ = ["PreActivationConv2d", "ResidualBlock2d"]
