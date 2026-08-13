"""Frozen-manifest adapter for unlabeled-target ISIC 2017 experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from medseg.data.contracts import build_model_ready_sample
from medseg.data.augmentations.builder import ImageTransformPipeline
from medseg.data.augmentations.builder import SegmentationTransformPipeline
from medseg.utils.paths import get_project_root


TARGET_MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "mask_available",
    "mask_path",
    "split",
    "target_labels_allowed_for_training",
    "seed",
)


@dataclass(frozen=True)
class ISIC2017Record:
    """One immutable row from a frozen target manifest."""

    image_id: str
    image_path: str
    mask_available: bool
    mask_path: str | None
    split: str
    target_labels_allowed_for_training: bool
    seed: int
    row: dict[str, str]


class ISIC2017ManifestDataset(Dataset[dict[str, Any]]):
    """Read a frozen ISIC 2017 manifest without generating or editing splits.

    Adaptation datasets are intentionally image-only. Masks are resolved and opened
    only when ``include_mask=True`` for the held-out evaluation split.
    """

    def __init__(
        self,
        config: Any,
        manifest_path: str | Path,
        images_dir: str | Path,
        masks_dir: str | Path | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | ImageTransformPipeline | SegmentationTransformPipeline | None = None,
        include_mask: bool = False,
    ) -> None:
        if transform is None:
            raise ValueError("ISIC2017ManifestDataset requires an explicit transform pipeline.")
        self.config = config
        self.manifest_path = _resolve_existing_path(manifest_path)
        self.images_dir = _resolve_directory(images_dir)
        self.masks_dir = _resolve_directory(masks_dir) if masks_dir is not None else None
        self.transform = transform
        self.include_mask = include_mask
        self.records = self._read_manifest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = _resolve_record_path(record.image_path, self.images_dir)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        metadata: dict[str, Any] = {
            "dataset_name": "isic2017",
            "dataset_version": "2017",
            "split": record.split,
            "source_image_path": str(image_path),
            "source_mask_path": None,
            "split_manifest": str(self.manifest_path),
            "target_labels_allowed_for_training": record.target_labels_allowed_for_training,
            "manifest_seed": record.seed,
        }
        sample: dict[str, Any] = {
            "image": image,
            "sample_id": record.image_id,
            "metadata": metadata,
        }

        if self.include_mask:
            if not record.mask_available or not record.mask_path:
                raise FileNotFoundError(
                    f"Evaluation row '{record.image_id}' does not declare an available mask."
                )
            if self.masks_dir is None:
                raise ValueError("masks_dir is required when include_mask=True.")
            mask_path = _resolve_record_path(record.mask_path, self.masks_dir)
            with Image.open(mask_path) as mask_file:
                sample["mask"] = mask_file.convert("L")
            metadata["source_mask_path"] = str(mask_path)

        transformed = self.transform(sample)
        if self.include_mask:
            build_model_ready_sample(
                image=transformed["image"],
                mask=transformed["mask"],
                sample_id=transformed["sample_id"],
                metadata=transformed["metadata"],
            )
        else:
            image_tensor = transformed["image"]
            if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3:
                raise ValueError("Image-only target preprocessing must return a CHW tensor.")
        return transformed

    def _read_manifest(self) -> list[ISIC2017Record]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Target manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in TARGET_MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"Target manifest '{self.manifest_path}' is missing columns: {missing}"
                )
            records: list[ISIC2017Record] = []
            for row in reader:
                labels_allowed = _parse_bool(row["target_labels_allowed_for_training"])
                if not self.include_mask and labels_allowed:
                    raise ValueError(
                        "Target adaptation manifest permits labels for training; refusing to "
                        f"load row '{row['image_id']}'."
                    )
                records.append(
                    ISIC2017Record(
                        image_id=_normalise_id(row["image_id"]),
                        image_path=row["image_path"],
                        mask_available=_parse_bool(row["mask_available"]),
                        mask_path=row["mask_path"] or None,
                        split=row["split"],
                        target_labels_allowed_for_training=labels_allowed,
                        seed=int(row["seed"]),
                        row=dict(row),
                    )
                )
        if not records:
            raise ValueError(f"Target manifest is empty: {self.manifest_path}")
        return records


def _resolve_existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    project_candidate = (get_project_root() / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return project_candidate


def _resolve_directory(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    project_candidate = (get_project_root() / path).resolve()
    return project_candidate


def _resolve_record_path(value: str, fallback_directory: Path) -> Path:
    raw = str(value)
    direct = Path(raw).expanduser()
    candidates = [direct]
    if not direct.is_absolute():
        candidates.append(get_project_root() / direct)

    names = {
        Path(raw).name,
        PurePosixPath(raw.replace("\\", "/")).name,
        PureWindowsPath(raw.replace("/", "\\")).name,
    }
    candidates.extend(fallback_directory / name for name in names if name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve target file '{value}'. Checked fallback directory '{fallback_directory}'."
    )


def _normalise_id(value: str) -> str:
    name = PurePosixPath(str(value).replace("\\", "/")).name
    return Path(name).stem


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean manifest value, received '{value}'.")


__all__ = ["ISIC2017ManifestDataset", "ISIC2017Record", "TARGET_MANIFEST_COLUMNS"]
