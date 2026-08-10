"""Model-agnostic trainer scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.config.schema import AppConfig
from medseg.experiment.identity import build_experiment_identity
from medseg.models.builder import describe_model_request
from medseg.optimization.builder import build_optimizer_spec, build_scheduler_spec


@dataclass
class Trainer:
    """Central training coordinator.

    This scaffold preserves the configuration flow and the model-agnostic API
    while leaving the actual training loop for the baseline implementation step.
    """

    config: AppConfig
    component_summary: dict[str, str]

    @classmethod
    def from_config(cls, config: AppConfig) -> "Trainer":
        """Create a trainer from the composed experiment config."""

        identity = build_experiment_identity(config)
        optimizer_spec = build_optimizer_spec(config)
        scheduler_spec = build_scheduler_spec(config)
        summary = {
            "dataset": identity.dataset_name,
            "dataset_version": identity.dataset_version,
            "split_definition": identity.split_definition_name,
            "model": describe_model_request(config),
            "preprocessing": config.preprocessing.name,
            "augmentation": config.augmentation.name,
            "loss": config.loss.name,
            "metrics": ", ".join(config.metrics.names),
            "optimizer": optimizer_spec.name,
            "scheduler": scheduler_spec.name,
            "checkpoint_load_enabled": str(config.checkpoint.load_enabled).lower(),
        }
        return cls(config=config, component_summary=summary)

    def fit(self) -> None:
        """Run model training.

        The trainer interface is intentionally stable before the concrete loop is
        implemented so future models can plug in without trainer refactors.
        """

        raise NotImplementedError("Training loop implementation is intentionally deferred.")
