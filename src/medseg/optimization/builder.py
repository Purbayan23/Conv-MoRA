"""Optimizer and scheduler configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from medseg.config.schema import AppConfig, OptimizerConfig, SchedulerConfig


@dataclass(frozen=True)
class OptimizerSpec:
    """Serializable optimizer description derived from config."""

    name: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SchedulerSpec:
    """Serializable scheduler description derived from config."""

    name: str
    kwargs: dict[str, Any]


def build_optimizer_spec(config: AppConfig | OptimizerConfig) -> OptimizerSpec:
    """Return the optimizer request described by the config."""

    optimizer_config = config.optimizer if isinstance(config, AppConfig) else config
    return OptimizerSpec(
        name=optimizer_config.name,
        kwargs={
            "lr": optimizer_config.lr,
            "weight_decay": optimizer_config.weight_decay,
            "betas": tuple(optimizer_config.betas),
        },
    )


def build_scheduler_spec(config: AppConfig | SchedulerConfig) -> SchedulerSpec:
    """Return the scheduler request described by the config."""

    scheduler_config = config.scheduler if isinstance(config, AppConfig) else config
    return SchedulerSpec(
        name=scheduler_config.name,
        kwargs={
            "interval": scheduler_config.interval,
            "monitor": scheduler_config.monitor,
            "mode": scheduler_config.mode,
            "factor": scheduler_config.factor,
            "patience": scheduler_config.patience,
            "min_lr": scheduler_config.min_lr,
        },
    )


def build_scheduler(
    optimizer: Optimizer,
    config: AppConfig | SchedulerConfig,
) -> Any | None:
    """Build the configured scheduler for an optimizer.

    ReduceLROnPlateau is stepped by the trainer after validation so its
    monitored value is always the current validation Dice score.
    """

    scheduler_config = config.scheduler if isinstance(config, AppConfig) else config
    name = scheduler_config.name.lower()
    if name in {"none", "disabled"}:
        return None
    if name not in {"reduce_on_plateau", "reduce_lr_on_plateau", "reducelronplateau"}:
        raise ValueError(f"Unknown scheduler '{scheduler_config.name}'.")
    if scheduler_config.interval != "epoch":
        raise ValueError("ReduceLROnPlateau must use interval='epoch'.")

    import torch

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.mode,
        factor=scheduler_config.factor,
        patience=scheduler_config.patience,
        min_lr=scheduler_config.min_lr,
    )
