"""Bottleneck block specifications."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.models.blocks.base import BlockSpec


@dataclass(frozen=True)
class BottleneckBlockSpec(BlockSpec):
    """Description of a bottleneck stage family."""

    dropout: float = 0.0
