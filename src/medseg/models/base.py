"""Base model contracts for segmentation architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.typing import ModelOutput, TensorLike

try:
    from torch import nn
except ImportError:  # pragma: no cover - fallback only used in bare environments
    class _ModuleBase:
        def __init__(self) -> None:
            pass

    ModuleBase = _ModuleBase
else:
    ModuleBase = nn.Module


class BaseSegmentationModel(ModuleBase, ABC):
    """Abstract base class for segmentation models."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, inputs: TensorLike) -> ModelOutput:
        """Run a forward pass and return the canonical model output."""
