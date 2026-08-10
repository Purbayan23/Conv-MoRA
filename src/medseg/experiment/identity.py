"""Experiment identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medseg.config.schema import AppConfig


@dataclass(frozen=True)
class ExperimentIdentity:
    """Minimal identity needed to distinguish independent experiments."""

    experiment_name: str
    dataset_name: str
    dataset_version: str
    split_definition_name: str
    model_name: str
    model_variant: str
    preprocessing_name: str
    augmentation_name: str
    loss_name: str
    optimizer_name: str
    scheduler_name: str
    runtime_profile: str
    seed: int


def build_experiment_identity(config: AppConfig) -> ExperimentIdentity:
    """Extract the stable identity of one experiment configuration."""

    return ExperimentIdentity(
        experiment_name=config.experiment.name,
        dataset_name=config.dataset.name,
        dataset_version=config.dataset.version,
        split_definition_name=config.dataset.split_definition_name,
        model_name=config.model.architecture,
        model_variant=config.model.variant,
        preprocessing_name=config.preprocessing.name,
        augmentation_name=config.augmentation.name,
        loss_name=config.loss.name,
        optimizer_name=config.optimizer.name,
        scheduler_name=config.scheduler.name,
        runtime_profile=config.runtime.profile_name,
        seed=config.seed,
    )


def build_experiment_manifest(config: AppConfig) -> dict[str, Any]:
    """Return a serializable manifest for experiment tracking."""

    identity = build_experiment_identity(config)
    return {
        "experiment": {
            "name": identity.experiment_name,
            "tags": list(config.experiment.tags),
            "seed": identity.seed,
        },
        "dataset": {
            "name": identity.dataset_name,
            "version": identity.dataset_version,
            "adapter": config.dataset.adapter,
            "split_definition_name": identity.split_definition_name,
            "splits": {
                "train": {
                    "manifest_path": config.dataset.splits.train.manifest_path,
                    "definition_source": config.dataset.splits.train.definition_source,
                },
                "val": {
                    "manifest_path": config.dataset.splits.val.manifest_path,
                    "definition_source": config.dataset.splits.val.definition_source,
                },
                "test": {
                    "manifest_path": config.dataset.splits.test.manifest_path,
                    "definition_source": config.dataset.splits.test.definition_source,
                },
            },
        },
        "model": {
            "architecture": identity.model_name,
            "variant": identity.model_variant,
            "stable_module_naming": config.model.stable_module_naming,
        },
        "preprocessing": {
            "name": identity.preprocessing_name,
            "size": {
                "height": config.preprocessing.size.height,
                "width": config.preprocessing.size.width,
                "require_spatial_divisible_by": config.preprocessing.size.require_spatial_divisible_by,
            },
            "normalization_enabled": config.preprocessing.normalization.enabled,
        },
        "augmentation": {
            "name": identity.augmentation_name,
            "train_allow_random": config.augmentation.train.allow_random,
            "eval_allow_random": config.augmentation.eval.allow_random,
        },
        "optimization": {
            "loss": identity.loss_name,
            "optimizer": identity.optimizer_name,
            "scheduler": identity.scheduler_name,
        },
        "runtime": {
            "profile_name": identity.runtime_profile,
            "device": config.runtime.device,
            "amp": config.runtime.amp,
            "deterministic": config.runtime.deterministic,
        },
        "checkpoint": {
            "load_enabled": config.checkpoint.load_enabled,
            "load_path": config.checkpoint.load_path,
            "strict": config.checkpoint.strict,
            "allow_dataset_mismatch": config.checkpoint.allow_dataset_mismatch,
            "allow_model_mismatch": config.checkpoint.allow_model_mismatch,
        },
    }
