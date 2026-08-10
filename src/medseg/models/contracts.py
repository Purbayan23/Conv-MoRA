"""Stable model input and output contracts for segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from medseg.typing import ModelOutput


@dataclass(frozen=True)
class ModelInputContract:
    """Descriptive contract for model-ready input tensors."""

    batch_layout: str
    sample_layout: str
    dtype: str
    channel_order: str
    spatial_policy: str
    value_convention: str


@dataclass(frozen=True)
class ModelOutputContract:
    """Descriptive contract for model outputs."""

    required_key: str
    layout: str
    symbolic_shape: tuple[str, ...]
    dtype: str
    returns_logits: bool
    applies_sigmoid_inside_model: bool
    thresholding_location: str
    auxiliary_outputs_key: str | None


DEFAULT_MODEL_INPUT_CONTRACT = ModelInputContract(
    batch_layout="NCHW",
    sample_layout="CHW",
    dtype="float32",
    channel_order="RGB for the current baseline",
    spatial_policy="H and W are defined by preprocessing.size and should be divisible by 16.",
    value_convention=(
        "Images enter the model as float32 tensors. If normalization is disabled they remain "
        "in [0.0, 1.0]; otherwise they are normalized according to preprocessing.normalization."
    ),
)

DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT = ModelOutputContract(
    required_key="logits",
    layout="NCHW",
    symbolic_shape=("N", "1", "H", "W"),
    dtype="floating point",
    returns_logits=True,
    applies_sigmoid_inside_model=False,
    thresholding_location="outside the model, in loss/evaluation/inference code",
    auxiliary_outputs_key="aux",
)


class ModelContractError(ValueError):
    """Raised when a model input or output violates the documented contract."""


def _is_float32_dtype(dtype_name: str) -> bool:
    normalized = dtype_name.lower()
    return normalized == "float32" or normalized.endswith(".float32")


def validate_model_input_tensor(tensor: Any, expected_channels: int | None = None) -> None:
    """Validate a model-ready input tensor-like object."""

    shape = getattr(tensor, "shape", None)
    dtype = getattr(tensor, "dtype", None)
    if shape is None or len(tuple(shape)) != 4:
        raise ModelContractError("Model input tensor must have NCHW rank-4 shape.")
    if dtype is None or not _is_float32_dtype(str(dtype)):
        raise ModelContractError(
            f"Model input tensor must have dtype {DEFAULT_MODEL_INPUT_CONTRACT.dtype}."
        )
    if expected_channels is not None and int(tuple(shape)[1]) != expected_channels:
        raise ModelContractError(
            f"Model input tensor channel dimension must be {expected_channels}, received {tuple(shape)[1]}."
        )


def validate_model_output(
    output: Mapping[str, Any] | ModelOutput,
    expected_batch_size: int | None = None,
) -> None:
    """Validate a segmentation model output mapping."""

    if DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT.required_key not in output:
        raise ModelContractError("Model output is missing required 'logits' key.")
    logits = output[DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT.required_key]
    shape = getattr(logits, "shape", None)
    if shape is None or len(tuple(shape)) != 4:
        raise ModelContractError("Model logits must have rank-4 NCHW shape.")
    if int(tuple(shape)[1]) != 1:
        raise ModelContractError("Binary segmentation logits must have channel dimension 1.")
    if expected_batch_size is not None and int(tuple(shape)[0]) != expected_batch_size:
        raise ModelContractError(
            f"Expected batch size {expected_batch_size}, received {tuple(shape)[0]}."
        )
