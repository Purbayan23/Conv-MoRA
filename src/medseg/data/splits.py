"""Dataset split helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any


class DatasetSplit(str, Enum):
    """Supported dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def normalize_split(split: str | DatasetSplit) -> DatasetSplit:
    """Normalize a split string to the supported enum."""

    if isinstance(split, DatasetSplit):
        return split
    try:
        return DatasetSplit(split)
    except ValueError as error:
        allowed = ", ".join(item.value for item in DatasetSplit)
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {allowed}") from error


def resolve_split_config(dataset_config: Any, split: str | DatasetSplit) -> Any:
    """Return the explicit split definition from dataset configuration."""

    normalized = normalize_split(split)
    return getattr(dataset_config.splits, normalized.value)
