"""Dataloader construction helpers."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from medseg.data.dataloaders.collate import segmentation_collate
from medseg.data.datasets.builder import build_dataset


def build_dataloader_kwargs(
    data_config: Any,
    runtime_config: Any = None,
    split: str = "train",
) -> dict[str, Any]:
    """Resolve DataLoader keyword arguments from config."""

    dataset = getattr(data_config, "dataset", None)
    if dataset is not None:
        runtime = data_config.runtime
    else:
        dataset = data_config
        runtime = runtime_config
        if runtime is None:
            from medseg.config.schema import RuntimeConfig

            runtime = RuntimeConfig()
        dataset = data_config

    num_workers = (
        runtime.num_workers if runtime.num_workers is not None else dataset.loader.num_workers
    )
    pin_memory = runtime.pin_memory if runtime.pin_memory is not None else dataset.loader.pin_memory
    return {
        "batch_size": dataset.loader.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": split == "train",
        "persistent_workers": dataset.loader.persistent_workers and num_workers > 0,
        "drop_last": False,
    }


def build_dataloaders(
    config: Any,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """Build dataloaders for the requested dataset splits."""

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        dataset = build_dataset(config, split=split)
        kwargs = build_dataloader_kwargs(config, split=split)
        loaders[split] = DataLoader(
            dataset,
            collate_fn=segmentation_collate,
            **kwargs,
        )
    return loaders
