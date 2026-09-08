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
    manifest: dict[str, Any] = {
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
    stage2 = getattr(config, "stage2", None)
    if stage2 is not None and stage2.enabled:
        if getattr(stage2, "adaptation_mode", "convlora") == "bn_only":
            manifest["stage2"] = {
                "protocol_name": stage2.protocol_name,
                "adaptation_mode": "bn_only",
                "paper_mapping": "frozen_unet_with_batchnorm_running_statistics_only",
                "executable_reference_mapping": (
                    "frozen source U-Net in train mode; only BatchNorm running buffers update; "
                    "BN affine parameters remain frozen"
                ),
                "insertion_scope": [],
                "convlora_enabled": False,
                "bn_affine_trainable": False,
                "adabn_train_affine": stage2.adabn_train_affine,
                "source_checkpoint": stage2.source_checkpoint,
                "duplicate_audit_manifest": stage2.duplicate_audit_manifest,
                "target_adaptation_manifest": stage2.target_adaptation_manifest,
                "target_eval_manifest": stage2.target_eval_manifest,
                "target_labels_allowed_for_training": stage2.target_labels_allowed_for_training,
            }
        else:
            manifest["stage2"] = {
                "protocol_name": stage2.protocol_name,
                "paper_mapping": (
                    "full_encoder_convlora_without_adabn"
                    if stage2.freeze_bn_running_stats
                    else "full_encoder_convlora_plus_adabn"
                ),
                "executable_reference_mapping": (
                    "init_path,down1,down2,down3 ConvLoRA with eval-mode BN buffers; "
                    "BN affine parameters frozen"
                    if stage2.freeze_bn_running_stats
                    else "init_path,down1,down2,down3 ConvLoRA with train-mode BN buffers; "
                    "BN affine parameters frozen"
                ),
                "reference_discrepancy": (
                    "reference constrained_lora_down3 omits adapter insertion; reference "
                    "test.py lora:down3 is the closest executable four-stage mapping"
                ),
                "insertion_scope": list(stage2.insertion_scope),
                "convlora_rank": stage2.convlora_rank,
                "convlora_alpha": stage2.convlora_alpha,
                "freeze_bn_running_stats": stage2.freeze_bn_running_stats,
                "esh_level": stage2.esh_level,
                "source_checkpoint": stage2.source_checkpoint,
                "esh_checkpoint": stage2.esh_checkpoint,
                "adaptation_checkpoint": stage2.adaptation_checkpoint,
                "duplicate_audit_manifest": stage2.duplicate_audit_manifest,
                "target_adaptation_manifest": stage2.target_adaptation_manifest,
                "target_eval_manifest": stage2.target_eval_manifest,
                "target_labels_allowed_for_training": stage2.target_labels_allowed_for_training,
                "consistency_metric": stage2.consistency_metric,
            }
    return manifest
