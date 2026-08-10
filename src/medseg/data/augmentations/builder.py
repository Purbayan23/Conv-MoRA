"""Augmentation and preprocessing for segmentation samples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageOps

from medseg.data.contracts import build_model_ready_sample
from medseg.data.augmentations.base import Transform
from medseg.typing import SegmentationSample


@dataclass(frozen=True)
class TransformPlan:
    """Description of how one transform stage should behave."""

    stage: str
    spatial_transforms: tuple[str, ...]
    allow_random: bool
    image_interpolation: str
    mask_interpolation: str
    normalize_images: bool
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_size: tuple[int, int]


class SegmentationTransformPipeline:
    """Minimal paired image/mask preprocessing and augmentation pipeline."""

    def __init__(self, plan: TransformPlan) -> None:
        self.plan = plan

    def __call__(self, sample: SegmentationSample) -> SegmentationSample:
        image = _ensure_rgb_image(sample["image"])
        mask = _ensure_mask_image(sample["mask"])

        if self.plan.allow_random:
            image, mask = _apply_random_spatial_transforms(
                image,
                mask,
                self.plan.spatial_transforms,
                image_resample=_resolve_resample_mode(self.plan.image_interpolation),
                mask_resample=_resolve_resample_mode(self.plan.mask_interpolation),
            )

        image, mask = _resize_pair(
            image,
            mask,
            self.plan.image_size,
            image_resample=_resolve_resample_mode(self.plan.image_interpolation),
            mask_resample=_resolve_resample_mode(self.plan.mask_interpolation),
        )
        image_tensor = _to_image_tensor(image)
        if self.plan.normalize_images and self.plan.mean and self.plan.std:
            image_tensor = _normalize_image_tensor(image_tensor, self.plan.mean, self.plan.std)

        mask_tensor = _to_mask_tensor(mask)
        transformed: SegmentationSample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample["sample_id"],
            "metadata": dict(sample["metadata"]),
        }
        build_model_ready_sample(
            image=transformed["image"],
            mask=transformed["mask"],
            sample_id=transformed["sample_id"],
            metadata=transformed["metadata"],
        )
        return transformed


def describe_transform_plan(config: Any, stage: Literal["train", "eval"]) -> TransformPlan:
    """Return the transform policy for one stage."""

    augmentation_config = getattr(config, "augmentation", config)
    stage_config = getattr(augmentation_config, stage)
    return TransformPlan(
        stage=stage,
        spatial_transforms=tuple(stage_config.spatial),
        allow_random=stage_config.allow_random,
        image_interpolation=augmentation_config.policy.image_interpolation,
        mask_interpolation=augmentation_config.policy.mask_interpolation,
        normalize_images=config.preprocessing.normalization.enabled,
        mean=tuple(config.preprocessing.normalization.mean),
        std=tuple(config.preprocessing.normalization.std),
        image_size=(config.preprocessing.size.height, config.preprocessing.size.width),
    )


def build_transforms(config: Any, stage: Literal["train", "eval"]) -> Transform:
    """Build the transform pipeline for one stage."""

    plan = describe_transform_plan(config=config, stage=stage)
    return SegmentationTransformPipeline(plan)


def _ensure_rgb_image(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        array = np.asarray(array)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255.0).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _ensure_mask_image(mask: object) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        array = np.asarray(array)
        if array.dtype != np.uint8:
            array = np.where(array > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(array).convert("L")
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.dtype != np.uint8:
        array = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(array).convert("L")


def _apply_random_spatial_transforms(
    image: Image.Image,
    mask: Image.Image,
    spatial_transforms: tuple[str, ...],
    image_resample: int,
    mask_resample: int,
) -> tuple[Image.Image, Image.Image]:
    if "random_horizontal_flip" in spatial_transforms and random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if "random_vertical_flip" in spatial_transforms and random.random() < 0.5:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)
    if "random_rotate_90" in spatial_transforms:
        k = random.randint(0, 3)
        if k:
            image = image.rotate(90 * k, resample=image_resample, expand=True)
            mask = mask.rotate(90 * k, resample=mask_resample, expand=True)
    return image, mask


def _resize_pair(
    image: Image.Image,
    mask: Image.Image,
    image_size: tuple[int, int],
    image_resample: int,
    mask_resample: int,
) -> tuple[Image.Image, Image.Image]:
    width, height = image_size[1], image_size[0]
    image = image.resize((width, height), resample=image_resample)
    mask = mask.resize((width, height), resample=mask_resample)
    return image, mask


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(dtype=torch.float32)


def _normalize_image_tensor(
    image_tensor: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32)
    array = np.where(array > 0, 1.0, 0.0).astype(np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
    return tensor.to(dtype=torch.float32)


def _resolve_resample_mode(mode_name: str) -> int:
    normalized = mode_name.strip().lower()
    if normalized == "nearest":
        return Image.Resampling.NEAREST
    if normalized == "bilinear":
        return Image.Resampling.BILINEAR
    if normalized == "bicubic":
        return Image.Resampling.BICUBIC
    if normalized == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unsupported interpolation mode '{mode_name}'.")
