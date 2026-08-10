"""Reusable data structures for a training step."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingStepResult:
    """Summary of one optimization step."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
