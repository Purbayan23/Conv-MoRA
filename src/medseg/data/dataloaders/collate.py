"""Collation helpers for segmentation batches."""

from __future__ import annotations

from medseg.typing import SegmentationSample


def passthrough_collate(batch: list[SegmentationSample]) -> list[SegmentationSample]:
    """Return the batch unchanged.

    A task-specific tensor collation function can replace this later without
    changing dataset or trainer code.
    """

    return batch
