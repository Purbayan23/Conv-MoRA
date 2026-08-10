"""Encoder block specifications."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.models.blocks.base import BlockSpec


@dataclass(frozen=True)
class EncoderBlockSpec(BlockSpec):
    """Description of an encoder stage family."""

    downsample: str = "max_pool"
    skip_connections: bool = True
