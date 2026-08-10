"""Decoder block specifications."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.models.blocks.base import BlockSpec


@dataclass(frozen=True)
class DecoderBlockSpec(BlockSpec):
    """Description of a decoder stage family."""

    upsample_mode: str = "transposed_conv"
    merge_mode: str = "concat"
