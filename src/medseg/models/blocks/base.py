"""Base specifications for replaceable model blocks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSpec:
    """Descriptive metadata for a replaceable model block."""

    name: str
