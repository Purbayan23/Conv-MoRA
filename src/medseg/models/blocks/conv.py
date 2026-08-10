"""Convolution block specifications."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.models.blocks.base import BlockSpec


@dataclass(frozen=True)
class ConvBlockSpec(BlockSpec):
    """Description of a convolution block family."""

    kernel_size: int = 3
    repeats: int = 2
    bias: bool = True
    dropout: float = 0.0
