"""Dataloader configuration helpers."""

from __future__ import annotations

from typing import Any

from medseg.config.schema import AppConfig, DatasetConfig, RuntimeConfig


def build_dataloader_kwargs(
    data_config: AppConfig | DatasetConfig,
    runtime_config: RuntimeConfig | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Resolve dataloader keyword arguments from centralized config."""

    if isinstance(data_config, AppConfig):
        runtime = data_config.runtime
        dataset = data_config.dataset
    else:
        runtime = runtime_config or RuntimeConfig()
        dataset = data_config

    num_workers = (
        runtime.num_workers if runtime.num_workers is not None else dataset.loader.num_workers
    )
    pin_memory = runtime.pin_memory if runtime.pin_memory is not None else dataset.loader.pin_memory
    persistent_workers = dataset.loader.persistent_workers and num_workers > 0
    return {
        "batch_size": dataset.loader.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "shuffle": split == "train",
    }
