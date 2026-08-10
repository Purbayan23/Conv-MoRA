"""Collation helpers for segmentation batches."""

from __future__ import annotations

from typing import Any

import torch

from medseg.typing import SegmentationSample


def segmentation_collate(batch: list[SegmentationSample]) -> dict[str, Any]:
    """Stack a list of segmentation samples into one batch."""

    images = torch.stack([sample["image"] for sample in batch], dim=0)
    masks = torch.stack([sample["mask"] for sample in batch], dim=0)
    sample_ids = [sample["sample_id"] for sample in batch]
    metadata = [sample["metadata"] for sample in batch]
    return {
        "image": images,
        "mask": masks,
        "sample_id": sample_ids,
        "metadata": metadata,
    }


passthrough_collate = segmentation_collate
