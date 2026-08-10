"""Segmentation head exports."""

from medseg.models.heads.base import BaseSegmentationHead
from medseg.models.heads.segmentation import BinarySegmentationHead

__all__ = ["BaseSegmentationHead", "BinarySegmentationHead"]
