"""Optimizer and scheduler configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    return SchedulerSpec(name=scheduler_config.name, kwargs={"interval": scheduler_config.interval})
