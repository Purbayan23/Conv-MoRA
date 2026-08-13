"""Native convolutional LoRA layers and encoder insertion helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from medseg.models.base import BaseSegmentationModel
from medseg.models.extensions.base import Adapter


class ConvLoRA(nn.Conv2d):
    """A convolution with the reference ConvLoRA low-rank weight update.

    The update follows the reference layout exactly for integer square kernels:
    ``W_eff = W0 + (alpha / r) * reshape(B @ A)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(kernel_size, int):
            raise TypeError("ConvLoRA requires an integer kernel_size.")
        if r < 0:
            raise ValueError("LoRA rank must be non-negative.")
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0
        self.lora_enabled = self.r > 0
        self.merge_weights = bool(merge_weights)
        self.merged = False

        if self.r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((self.r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, self.r * kernel_size))
            )
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_conv(
        cls,
        layer: nn.Conv2d,
        r: int,
        lora_alpha: int,
    ) -> "ConvLoRA":
        """Create an adapted convolution while preserving base parameters."""

        if layer.kernel_size[0] != layer.kernel_size[1]:
            raise ValueError("ConvLoRA requires square convolution kernels.")
        adapted = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size[0],
            r=r,
            lora_alpha=lora_alpha,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
        with torch.no_grad():
            adapted.weight.copy_(layer.weight)
            if layer.bias is not None and adapted.bias is not None:
                adapted.bias.copy_(layer.bias)
        return adapted

    @property
    def delta_weight(self) -> torch.Tensor:
        """Return the unscaled low-rank convolution weight update."""

        if self.r == 0:
            return torch.zeros_like(self.weight)
        kernel_size = self.kernel_size[0]
        return (self.lora_B @ self.lora_A).view(self.weight.shape)

    def set_lora_enabled(self, enabled: bool) -> None:
        """Enable or disable the low-rank update without changing base weights."""

        enabled = bool(enabled) and self.r > 0
        if not enabled and self.merged:
            self._unmerge()
        self.lora_enabled = enabled
        if enabled and not self.training and self.merge_weights:
            self._merge()

    def train(self, mode: bool = True) -> "ConvLoRA":
        """Toggle training mode while merging or unmerging the LoRA update."""

        nn.Conv2d.train(self, mode)
        if self.r == 0 or not self.merge_weights or not self.lora_enabled:
            return self
        if mode and self.merged:
            self._unmerge()
        elif not mode and not self.merged:
            self._merge()
        return self

    def _merge(self) -> None:
        if self.r > 0 and not self.merged:
            with torch.no_grad():
                self.weight.add_(self.delta_weight * self.scaling)
            self.merged = True

    def _unmerge(self) -> None:
        if self.r > 0 and self.merged:
            with torch.no_grad():
                self.weight.sub_(self.delta_weight * self.scaling)
            self.merged = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and self.lora_enabled and not self.merged:
            weight = self.weight + self.delta_weight * self.scaling
            return F.conv2d(
                inputs,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, inputs)


def apply_convlora(
    model: nn.Module,
    rank: int = 2,
    alpha: int = 2,
    scope: Sequence[str] | str | None = None,
) -> nn.Module:
    """Replace Conv2d layers in selected encoder modules with ConvLoRA layers."""

    selected_scope = _resolve_scope(scope)
    for module_name in selected_scope:
        try:
            module = getattr(model, module_name)
        except AttributeError as error:
            raise ValueError(f"Model has no ConvLoRA insertion module '{module_name}'.") from error
        _replace_convolutions(module, rank=rank, alpha=alpha)
    return model


class ConvLoRAAdapter(Adapter):
    """Attach ConvLoRA to a model and expose only adapter parameters for training."""

    def __init__(
        self,
        rank: int = 2,
        alpha: int = 2,
        scope: Sequence[str] | str | None = None,
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.scope = scope

    def attach(self, model: BaseSegmentationModel) -> BaseSegmentationModel:
        apply_convlora(model, rank=self.rank, alpha=self.alpha, scope=self.scope)
        mark_only_adapter_as_trainable(model)
        return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze every parameter before enabling a selected adapter."""

    for parameter in model.parameters():
        parameter.requires_grad = False


def mark_only_adapter_as_trainable(model: nn.Module) -> None:
    """Freeze the base model and enable only ConvLoRA parameters."""

    freeze_base_model(model)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and trainable ConvLoRA parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    convlora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "convlora_trainable": convlora_trainable,
    }


def _resolve_scope(scope: Sequence[str] | str | None) -> tuple[str, ...]:
    if scope is None or scope == "full_encoder":
        return ("init_path", "down1", "down2", "down3")
    if isinstance(scope, str):
        return (scope,)
    return tuple(scope)


def _replace_convolutions(module: nn.Module, rank: int, alpha: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, ConvLoRA):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, ConvLoRA.from_conv(child, r=rank, lora_alpha=alpha))
        else:
            _replace_convolutions(child, rank=rank, alpha=alpha)


__all__ = [
    "ConvLoRA",
    "ConvLoRAAdapter",
    "apply_convlora",
    "freeze_base_model",
    "mark_only_adapter_as_trainable",
    "parameter_counts",
]
