"""Generic binary segmentation dataset implementation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from medseg.data.contracts import validate_segmentation_sample
from medseg.data.augmentations.builder import build_transforms
from medseg.data.datasets.base import BaseSegmentationDataset, SampleTransform
from medseg.data.splits import DatasetSplit
from medseg.utils.paths import get_project_root
from medseg.typing import SegmentationSample

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    """Resolved sample record for one image/mask pair."""

    image_path: Path
    mask_path: Path
    sample_id: str
    metadata: dict[str, Any]


class BinaryFolderSegmentationDataset(BaseSegmentationDataset):
    """Folder- or manifest-based dataset for binary segmentation."""

    def __init__(
        self,
        config: Any,
        split: str,
        transform: SampleTransform | None = None,
    ) -> None:
        super().__init__(config=config, split=split, transform=transform)
        if self.transform is None:
            raise ValueError(
                "A transform pipeline is required for binary segmentation samples. "
                "Use the configuration-driven transform builder."
            )
        self.records = self._load_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SegmentationSample:
        record = self.records[index]
        with Image.open(record.image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(record.mask_path) as mask_file:
            mask = mask_file.convert("L")

        sample: SegmentationSample = {
            "image": image,
            "mask": mask,
            "sample_id": record.sample_id,
            "metadata": dict(record.metadata),
        }
        transformed = self.transform(sample) if self.transform is not None else sample
        validate_segmentation_sample(transformed)
        return transformed

    def _load_records(self) -> list[SampleRecord]:
        manifest_path = self._resolve_path(self.split_config.manifest_path)
        if manifest_path is not None and manifest_path.exists():
            records = self._load_from_manifest(manifest_path)
            if records:
                return records

        return self._load_from_directories()

    def _load_from_manifest(self, manifest_path: Path) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = self._resolve_record_path(
                    row,
                    keys=("image_path", "image", "image_filename", "image_name"),
                    default_directory=self._resolve_path(self.split_config.images_dir),
                    manifest_path=manifest_path,
                )
                mask_path = self._resolve_record_path(
                    row,
                    keys=("mask_path", "mask", "mask_filename", "mask_name"),
                    default_directory=self._resolve_path(self.split_config.masks_dir),
                    manifest_path=manifest_path,
                )
                sample_id = (
                    row.get("sample_id")
                    or row.get("id")
                    or row.get("name")
                    or image_path.stem
                )
                records.append(self._build_record(image_path, mask_path, sample_id, manifest_path))
        return records

    def _load_from_directories(self) -> list[SampleRecord]:
        images_dir = self._resolve_path(self.split_config.images_dir)
        masks_dir = self._resolve_path(self.split_config.masks_dir)
        if images_dir is None or not images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        if masks_dir is None or not masks_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

        mask_lookup = {path.stem: path for path in _iter_files(masks_dir, MASK_EXTENSIONS)}
        records: list[SampleRecord] = []
        for image_path in _iter_files(images_dir, IMAGE_EXTENSIONS):
            mask_path = mask_lookup.get(image_path.stem)
            if mask_path is None:
                continue
            records.append(self._build_record(image_path, mask_path, image_path.stem, None))

        if not records:
            raise FileNotFoundError(
                f"No image/mask pairs found for split '{self.split.value}' in "
                f"{images_dir} and {masks_dir}."
            )
        return records

    def _build_record(
        self,
        image_path: Path,
        mask_path: Path,
        sample_id: str,
        manifest_path: Path | None,
    ) -> SampleRecord:
        with Image.open(image_path) as image_file:
            original_shape = (image_file.height, image_file.width)

        metadata: dict[str, Any] = {
            "dataset_name": self.config.name,
            "dataset_version": self.config.version,
            "split": self.split.value,
            "source_image_path": str(image_path),
            "source_mask_path": str(mask_path),
            "original_spatial_shape": original_shape,
            "split_manifest": str(manifest_path) if manifest_path is not None else None,
        }
        return SampleRecord(
            image_path=image_path,
            mask_path=mask_path,
            sample_id=str(sample_id),
            metadata=metadata,
        )

    def _resolve_record_path(
        self,
        row: dict[str, str | None],
        keys: tuple[str, ...],
        default_directory: Path | None,
        manifest_path: Path,
    ) -> Path:
        for key in keys:
            value = row.get(key)
            if value:
                return self._resolve_path(value, base=manifest_path.parent, fallback_directory=default_directory)

        sample_id = row.get("sample_id") or row.get("id") or row.get("name")
        if sample_id and default_directory is not None:
            for extension in IMAGE_EXTENSIONS:
                candidate = default_directory / f"{sample_id}{extension}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Could not resolve any of {keys} from manifest '{manifest_path}'. Row: {row}"
        )

    def _resolve_path(
        self,
        path_value: str | None,
        base: Path | None = None,
        fallback_directory: Path | None = None,
    ) -> Path | None:
        if path_value is None:
            return None
        path = Path(path_value)
        if path.is_absolute():
            return path
        if base is not None:
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        if fallback_directory is not None:
            candidate = (fallback_directory / path.name).resolve()
            if candidate.exists():
                return candidate
        project_root = get_project_root()
        candidate = (project_root / path).resolve()
        if candidate.exists():
            return candidate
        return (project_root / path).resolve()


def _iter_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
