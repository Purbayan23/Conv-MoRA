"""Base hook for future model augmentation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.models.base import BaseSegmentationModel


class Adapter(ABC):
    """Generic hook for independently attachable model adapters."""

    @abstractmethod
    def attach(self, model: BaseSegmentationModel) -> BaseSegmentationModel:
        """Return an augmented model instance."""


# Keep the original scaffold name available to existing callers.
ModelExtension = Adapter
