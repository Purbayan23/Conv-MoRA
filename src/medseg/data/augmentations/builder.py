"""Augmentation builder helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medseg.config.schema import AppConfig, AugmentationConfig
from medseg.data.augmentations.base import IdentityTransform, Transform


@dataclass(frozen=True)
class TransformPlan:
    """Description of how one transform stage should behave."""

    stage: str
    spatial_transforms: tuple[str, ...]
    image_only_transforms: tuple[str, ...]
    allow_random: bool
    shared_spatial_parameters: bool
    image_interpolation: str
    mask_interpolation: str
    apply_image_only_transforms_to_mask: bool


def describe_transform_plan(
    config: AppConfig | AugmentationConfig,
    stage: Literal["train", "eval"],
) -> TransformPlan:
    """Return the transform policy for one stage."""

    augmentation_config = config.augmentation if isinstance(config, AppConfig) else config
    stage_config = getattr(augmentation_config, stage)
    return TransformPlan(
        stage=stage,
        spatial_transforms=tuple(stage_config.spatial),
        image_only_transforms=tuple(stage_config.image_only),
        allow_random=stage_config.allow_random,
        shared_spatial_parameters=augmentation_config.policy.shared_spatial_parameters,
        image_interpolation=augmentation_config.policy.image_interpolation,
        mask_interpolation=augmentation_config.policy.mask_interpolation,
        apply_image_only_transforms_to_mask=augmentation_config.policy.apply_image_only_transforms_to_mask,
    )


def build_transforms(
    config: AppConfig | AugmentationConfig,
    stage: Literal["train", "eval"],
) -> Transform:
    """Build the transform pipeline for one stage.

    The scaffold currently returns an identity transform so the data API is
    stable before the actual augmentation implementation is added.
    """

    _ = describe_transform_plan(config=config, stage=stage)
    return IdentityTransform()
