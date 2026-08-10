"""Base hook for future model augmentation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.models.base import BaseSegmentationModel


class ModelExtension(ABC):
    """Hook interface for future model augmentation modules."""

    @abstractmethod
    def attach(self, model: BaseSegmentationModel) -> BaseSegmentationModel:
        """Return an augmented model instance."""
