"""Type-6 ConvMoRA layers for the controlled 3x3 encoder ablation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from medseg.models.base import BaseSegmentationModel
from medseg.models.extensions.base import Adapter


class ConvMoRA(nn.Conv2d):
    """A convolution with a Type-6 ConvMoRA weight update.

    The base convolution remains frozen. The only trainable parameter is the
    zero-initialized square matrix ``M``. The expanded update uses the same
    C-order ``D x I`` to convolution-kernel reshape as ConvLoRA.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r_conv: int = 2,
        merge_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(kernel_size, int):
            raise TypeError("ConvMoRA requires an integer kernel_size.")
        if r_conv <= 0:
            raise ValueError("ConvMoRA requires a positive r_conv.")

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        if self.groups != 1:
            raise ValueError("ConvMoRA currently supports only ungrouped convolutions.")

        self.r_conv = int(r_conv)
        self.input_dim = in_channels * kernel_size
        self.output_dim = out_channels * kernel_size
        self.q = self.r_conv * kernel_size
        self.parameter_budget = self.q * (self.input_dim + self.output_dim)
        self.m_raw = int(math.sqrt(self.parameter_budget) + 0.5)
        self.m = self.m_raw if self.m_raw % 2 == 0 else self.m_raw + 1
        self.n_blocks = math.ceil(self.input_dim / self.m)
        self.scaling = 1.0
        self.merge_weights = bool(merge_weights)
        self.merged = False

        self.M = nn.Parameter(self.weight.new_zeros((self.m, self.m)))
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @classmethod
    def from_conv(
        cls,
        layer: nn.Conv2d,
        r_conv: int = 2,
        merge_weights: bool = True,
    ) -> "ConvMoRA":
        """Create a ConvMoRA layer while preserving the source convolution."""

        if layer.kernel_size[0] != layer.kernel_size[1]:
            raise ValueError("ConvMoRA requires square convolution kernels.")
        adapted = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size[0],
            r_conv=r_conv,
            merge_weights=merge_weights,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
        adapted.to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            adapted.weight.copy_(layer.weight)
            if layer.bias is not None and adapted.bias is not None:
                adapted.bias.copy_(layer.bias)
        return adapted

    def _type6_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply prefix repetition, Type-6 RoPE, blockwise M, and reconstruction."""

        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected the final dimension to be {self.input_dim}, "
                f"got {inputs.shape[-1]}."
            )

        padding = self.n_blocks * self.m - self.input_dim
        if padding:
            inputs = torch.cat((inputs, inputs[..., :padding]), dim=-1)
        blocks = inputs.reshape(*inputs.shape[:-1], self.n_blocks, self.m)

        half = self.m // 2
        positions = torch.arange(self.n_blocks, device=inputs.device, dtype=inputs.dtype)
        indices = torch.arange(half, device=inputs.device, dtype=inputs.dtype)
        frequencies = torch.exp(-2.0 * math.log(10000.0) * indices / self.m)
        angles = positions[:, None] * frequencies[None, :]
        cosine = angles.cos()
        sine = angles.sin()

        first = blocks[..., :half]
        second = blocks[..., half:]
        rotated = torch.cat(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        )
        transformed = torch.einsum("ij,...nj->...ni", self.M, rotated)
        flattened = transformed.reshape(*transformed.shape[:-2], -1)

        if flattened.shape[-1] < self.output_dim:
            repeats = math.ceil(self.output_dim / flattened.shape[-1])
            flattened = torch.cat([flattened] * repeats, dim=-1)
        return flattened[..., : self.output_dim]

    @property
    def delta_weight_tilde(self) -> torch.Tensor:
        """Return the expanded Type-6 update with shape ``(D, I)``."""

        basis = torch.eye(
            self.input_dim,
            device=self.M.device,
            dtype=self.M.dtype,
        )
        transformed_basis = self._type6_transform(basis)
        return transformed_basis.transpose(0, 1).contiguous()

    @property
    def delta_weight(self) -> torch.Tensor:
        """Return the expanded update reshaped to the source kernel layout."""

        return self.delta_weight_tilde.view_as(self.weight)

    def train(self, mode: bool = True) -> "ConvMoRA":
        """Toggle training mode while maintaining a merged effective weight."""

        nn.Conv2d.train(self, mode)
        if not self.merge_weights:
            return self
        if mode and self.merged:
            self._unmerge()
        elif not mode and not self.merged:
            self._merge()
        return self

    def _merge(self) -> None:
        if not self.merged:
            with torch.no_grad():
                self.weight.add_(self.delta_weight)
            self.merged = True

    def _unmerge(self) -> None:
        if self.merged:
            with torch.no_grad():
                self.weight.sub_(self.delta_weight)
            self.merged = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.merged:
            weight = self.weight + self.delta_weight
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


class ConvMoRAAdapter(Adapter):
    """Attach 3x3 ConvMoRA layers to selected model modules."""

    def __init__(
        self,
        r_conv: int = 2,
        scope: Sequence[str] | str | None = None,
        kernel_size: int | None = 3,
    ) -> None:
        self.r_conv = r_conv
        self.scope = scope
        self.kernel_size = kernel_size

    def attach(self, model: BaseSegmentationModel) -> BaseSegmentationModel:
        apply_convmora(
            model,
            r_conv=self.r_conv,
            scope=self.scope,
            kernel_size=self.kernel_size,
        )
        mark_only_convmora_as_trainable(model)
        return model


def apply_convmora(
    model: nn.Module,
    r_conv: int = 2,
    scope: Sequence[str] | str | None = None,
    kernel_size: int | None = 3,
) -> nn.Module:
    """Replace selected Conv2d layers with standalone ConvMoRA layers."""

    if kernel_size is not None and kernel_size <= 0:
        raise ValueError("kernel_size must be positive when provided.")
    for module_name in _resolve_scope(scope):
        try:
            module = getattr(model, module_name)
        except AttributeError as error:
            raise ValueError(f"Model has no ConvMoRA insertion module '{module_name}'.") from error
        _replace_convolutions(module, r_conv=r_conv, kernel_size=kernel_size)
    return model


def mark_only_convmora_as_trainable(model: nn.Module) -> None:
    """Freeze the base model and enable only ConvMoRA matrices."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, ConvMoRA):
            module.M.requires_grad = True


def _resolve_scope(scope: Sequence[str] | str | None) -> tuple[str, ...]:
    if scope is None or scope == "full_encoder":
        return ("init_path", "down1", "down2", "down3")
    if isinstance(scope, str):
        return (scope,)
    return tuple(scope)


def _replace_convolutions(
    module: nn.Module,
    r_conv: int,
    kernel_size: int | None,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, ConvMoRA):
            continue
        if isinstance(child, nn.Conv2d):
            if kernel_size is None or child.kernel_size == (kernel_size, kernel_size):
                setattr(module, name, ConvMoRA.from_conv(child, r_conv=r_conv))
        else:
            _replace_convolutions(child, r_conv=r_conv, kernel_size=kernel_size)


__all__ = [
    "ConvMoRA",
    "ConvMoRAAdapter",
    "apply_convmora",
    "mark_only_convmora_as_trainable",
]
