"""Explicit dataset and preprocessing contracts for segmentation research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medseg.typing import SegmentationSample

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is an install-time dependency
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is an install-time dependency
    torch = None


@dataclass(frozen=True)
class TensorContract:
    """Descriptive contract for one tensor-like object."""

    layout: str
    symbolic_shape: tuple[str, ...]
    dtype: str
    value_convention: str
    channel_convention: str


@dataclass(frozen=True)
class MaskContract:
    """Descriptive contract for segmentation masks."""

    layout: str
    symbolic_shape: tuple[str, ...]
    dtype: str
    valid_values: tuple[float, ...]
    encoding: str
    resize_interpolation: str
    future_multiclass_policy: str


@dataclass(frozen=True)
class SegmentationSampleContract:
    """Complete common contract for dataset adapters."""

    required_keys: tuple[str, ...]
    image: TensorContract
    mask: MaskContract
    sample_id_type: str
    metadata_policy: str


DEFAULT_SEGMENTATION_SAMPLE_CONTRACT = SegmentationSampleContract(
    required_keys=("image", "mask", "sample_id", "metadata"),
    image=TensorContract(
        layout="CHW",
        symbolic_shape=("C", "H", "W"),
        dtype="float32",
        value_convention="[0.0, 1.0] before optional normalization",
        channel_convention="RGB for the current baseline",
    ),
    mask=MaskContract(
        layout="1HW",
        symbolic_shape=("1", "H", "W"),
        dtype="float32",
        valid_values=(0.0, 1.0),
        encoding="binary foreground mask",
        resize_interpolation="nearest",
        future_multiclass_policy=(
            "Future multi-class datasets should remain channel-first and use semantic "
            "class IDs before any loss-specific conversion."
        ),
    ),
    sample_id_type="str",
    metadata_policy=(
        "Serializable dataset-specific metadata only, such as dataset name, split, "
        "source paths, original spatial shape, or acquisition metadata."
    ),
)


class ContractValidationError(ValueError):
    """Raised when a sample or tensor violates the documented contract."""


def _extract_shape(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ContractValidationError("Tensor-like object is missing a 'shape' attribute.")
    return tuple(int(dim) for dim in shape)


def _extract_dtype_name(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise ContractValidationError("Tensor-like object is missing a 'dtype' attribute.")
    return str(dtype)


def _is_float32_dtype(dtype_name: str) -> bool:
    normalized = dtype_name.lower()
    return normalized == "float32" or normalized.endswith(".float32")


def _extract_unique_values(tensor: Any) -> tuple[float, ...] | None:
    if hasattr(tensor, "unique_values"):
        values = getattr(tensor, "unique_values")
        return tuple(float(value) for value in values)
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tuple(float(value) for value in torch.unique(tensor.detach().cpu()).flatten().tolist())
    if np is not None:
        try:
            return tuple(float(value) for value in np.unique(np.asarray(tensor)).flatten().tolist())
        except Exception:  # pragma: no cover - fallback for non-array-like inputs
            return None
    return None


def mask_values_are_binary(values: Sequence[float | int]) -> bool:
    """Return whether the values stay within the binary mask vocabulary."""

    return set(float(value) for value in values).issubset({0.0, 1.0})


def categorical_values_preserved(
    original_values: Sequence[float | int],
    transformed_values: Sequence[float | int],
) -> bool:
    """Return whether a transformed categorical mask stayed within the original label set."""

    original = {float(value) for value in original_values}
    transformed = {float(value) for value in transformed_values}
    return transformed.issubset(original)


def validate_segmentation_sample(
    sample: Mapping[str, Any],
    contract: SegmentationSampleContract = DEFAULT_SEGMENTATION_SAMPLE_CONTRACT,
) -> None:
    """Validate one dataset sample against the common contract."""

    missing_keys = [key for key in contract.required_keys if key not in sample]
    if missing_keys:
        raise ContractValidationError(f"Sample is missing required keys: {missing_keys}")

    sample_id = sample["sample_id"]
    metadata = sample["metadata"]
    if not isinstance(sample_id, str) or sample_id == "":
        raise ContractValidationError("sample_id must be a non-empty string.")
    if not isinstance(metadata, Mapping):
        raise ContractValidationError("metadata must be a mapping.")

    image_shape = _extract_shape(sample["image"])
    mask_shape = _extract_shape(sample["mask"])
    image_dtype = _extract_dtype_name(sample["image"])
    mask_dtype = _extract_dtype_name(sample["mask"])

    if len(image_shape) != 3:
        raise ContractValidationError(f"Image must be 3D CHW, received shape {image_shape}.")
    if len(mask_shape) != 3:
        raise ContractValidationError(f"Mask must be 3D 1HW, received shape {mask_shape}.")
    if mask_shape[0] != 1:
        raise ContractValidationError(f"Mask channel dimension must be 1, received {mask_shape[0]}.")
    if image_shape[1:] != mask_shape[1:]:
        raise ContractValidationError(
            "Image and mask spatial dimensions must match after preprocessing."
        )
    if not _is_float32_dtype(image_dtype):
        raise ContractValidationError(
            f"Image dtype must be compatible with {contract.image.dtype}, received {image_dtype}."
        )
    if not _is_float32_dtype(mask_dtype):
        raise ContractValidationError(
            f"Mask dtype must be compatible with {contract.mask.dtype}, received {mask_dtype}."
        )

    unique_mask_values = _extract_unique_values(sample["mask"])
    if unique_mask_values is not None and not mask_values_are_binary(unique_mask_values):
        raise ContractValidationError(
            f"Mask contains non-binary values after preprocessing: {unique_mask_values}"
        )


def build_model_ready_sample(
    image: Any,
    mask: Any,
    sample_id: str,
    metadata: dict[str, Any] | None = None,
) -> SegmentationSample:
    """Construct one common sample instance for tests or adapter implementations."""

    payload: SegmentationSample = {
        "image": image,
        "mask": mask,
        "sample_id": sample_id,
        "metadata": metadata or {},
    }
    validate_segmentation_sample(payload)
    return payload
