"""Validation loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.config.schema import AppConfig


@dataclass
class Evaluator:
    """Model-agnostic validation coordinator."""

    config: AppConfig

    @classmethod
    def from_config(cls, config: AppConfig) -> "Evaluator":
        return cls(config=config)

    def evaluate(self) -> None:
        raise NotImplementedError("Validation loop implementation is intentionally deferred.")
