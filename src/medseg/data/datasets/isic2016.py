"""ISIC 2016 dataset adapter."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from medseg.data.datasets.base import SampleTransform
from medseg.data.datasets.binary_folder import (
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    BinaryFolderSegmentationDataset,
    SampleRecord,
    _iter_files,
)


class ISIC2016Dataset(BinaryFolderSegmentationDataset):
    """Dataset adapter for the ISIC 2016 lesion segmentation benchmark."""

    def __init__(
        self,
        config: Any,
        split: str,
        transform: SampleTransform | None = None,
    ) -> None:
        super().__init__(config=config, split=split, transform=transform)

    def _load_records(self) -> list[SampleRecord]:
        """Load an existing manifest or resolve the configured ISIC split."""

        manifest_path = self._resolve_path(self.split_config.manifest_path)
        if manifest_path is not None and manifest_path.exists():
            records = self._load_isic_manifest(manifest_path)
            if records:
                return records

        if self.split.value in {"train", "val"}:
            return self._load_or_create_train_val_split()

        return self._load_isic_pairs(
            self._resolve_path(self.split_config.images_dir),
            self._resolve_path(self.split_config.masks_dir),
            manifest_path=None,
        )

    def _load_or_create_train_val_split(self) -> list[SampleRecord]:
        """Create deterministic train/validation manifests from official training data."""

        train_config = self.config.splits.train
        images_dir = self._resolve_path(train_config.images_dir)
        masks_dir = self._resolve_path(train_config.masks_dir)
        pairs = self._find_isic_pairs(images_dir, masks_dir)
        if not pairs:
            raise FileNotFoundError(
                "No ISIC training image/mask pairs found. "
                f"Images: {images_dir}; masks: {masks_dir}."
            )

        shuffled_pairs = list(pairs)
        random.Random(self.config.split_seed).shuffle(shuffled_pairs)
        train_count = int(len(shuffled_pairs) * self.config.train_val_ratio)
        if len(shuffled_pairs) > 1:
            train_count = min(max(train_count, 1), len(shuffled_pairs) - 1)
        train_pairs = shuffled_pairs[:train_count]
        val_pairs = shuffled_pairs[train_count:]

        self._write_manifest(self.config.splits.train.manifest_path, train_pairs)
        self._write_manifest(self.config.splits.val.manifest_path, val_pairs)
        selected_pairs = train_pairs if self.split.value == "train" else val_pairs
        manifest_path = self._resolve_path(self.split_config.manifest_path)
        return self._records_from_pairs(selected_pairs, manifest_path)

    def _load_isic_manifest(self, manifest_path: Path) -> list[SampleRecord]:
        """Resolve sample IDs from an ISIC split manifest."""

        images_dir = self._resolve_path(self.split_config.images_dir)
        masks_dir = self._resolve_path(self.split_config.masks_dir)
        records: list[SampleRecord] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row.get("sample_id") or row.get("id") or row.get("name")
                if not sample_id:
                    image_value = row.get("image_path") or row.get("image")
                    sample_id = Path(image_value).stem if image_value else None
                if not sample_id:
                    raise ValueError(f"Manifest row has no ISIC sample identifier: {row}")
                sample_id = _normalize_sample_id(sample_id)
                records.extend(
                    self._records_from_pairs(
                        self._find_isic_pairs(images_dir, masks_dir, sample_ids={sample_id}),
                        manifest_path,
                    )
                )
        return records

    def _load_isic_pairs(
        self,
        images_dir: Path | None,
        masks_dir: Path | None,
        manifest_path: Path | None,
    ) -> list[SampleRecord]:
        """Resolve all ISIC image/mask pairs for one configured directory pair."""

        return self._records_from_pairs(
            self._find_isic_pairs(images_dir, masks_dir),
            manifest_path,
        )

    def _find_isic_pairs(
        self,
        images_dir: Path | None,
        masks_dir: Path | None,
        sample_ids: set[str] | None = None,
    ) -> list[tuple[Path, Path, str]]:
        if images_dir is None or not images_dir.exists():
            raise FileNotFoundError(f"ISIC image directory not found: {images_dir}")
        if masks_dir is None or not masks_dir.exists():
            raise FileNotFoundError(f"ISIC mask directory not found: {masks_dir}")

        image_lookup = {path.stem: path for path in _iter_files(images_dir, IMAGE_EXTENSIONS)}
        mask_lookup = {
            _normalize_sample_id(path.stem): path
            for path in _iter_files(masks_dir, MASK_EXTENSIONS)
        }
        selected_ids = sorted(sample_ids if sample_ids is not None else image_lookup.keys())
        pairs = [
            (image_lookup[sample_id], mask_lookup[sample_id], sample_id)
            for sample_id in selected_ids
            if sample_id in image_lookup and sample_id in mask_lookup
        ]
        if sample_ids is not None and len(pairs) != len(sample_ids):
            missing = sorted(sample_ids - {sample_id for _, _, sample_id in pairs})
            raise FileNotFoundError(f"Missing ISIC image/mask pair(s): {missing}")
        if not pairs:
            raise FileNotFoundError(
                f"No ISIC image/mask pairs found in {images_dir} and {masks_dir}."
            )
        return pairs

    def _records_from_pairs(
        self,
        pairs: list[tuple[Path, Path, str]],
        manifest_path: Path | None,
    ) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for image_path, mask_path, sample_id in pairs:
            records.append(self._build_record(image_path, mask_path, sample_id, manifest_path))
        return records

    def _write_manifest(
        self,
        manifest_value: str | None,
        pairs: list[tuple[Path, Path, str]],
    ) -> None:
        manifest_path = self._resolve_path(manifest_value)
        if manifest_path is None or manifest_path.exists():
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id"])
            writer.writeheader()
            writer.writerows({"sample_id": sample_id} for _, _, sample_id in pairs)


def _normalize_sample_id(value: str) -> str:
    """Convert an image or ISIC segmentation filename to its shared sample ID."""

    sample_id = Path(value).stem
    suffix = "_Segmentation"
    if sample_id.endswith(suffix):
        sample_id = sample_id[: -len(suffix)]
    return sample_id
