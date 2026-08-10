"""Shared type contracts used across the project."""

from __future__ import annotations

from typing import Any, TypedDict

TensorLike = Any
MetadataDict = dict[str, Any]


class SegmentationSampleRequired(TypedDict):
    """Required fields for all segmentation dataset samples."""

    image: TensorLike
    mask: TensorLike
    sample_id: str
    metadata: MetadataDict


class SegmentationSample(SegmentationSampleRequired):
    """Canonical dataset sample contract."""


class ModelOutputRequired(TypedDict):
    """Required fields for model outputs."""

    logits: TensorLike


class ModelOutput(ModelOutputRequired, total=False):
    """Canonical model output contract."""

    aux: MetadataDict
