"""Model layer exports."""

from medseg.models.base import BaseSegmentationModel
from medseg.models.builder import build_model, describe_model_request
from medseg.models.contracts import (
    DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT,
    DEFAULT_MODEL_INPUT_CONTRACT,
    validate_model_input_tensor,
    validate_model_output,
)

__all__ = [
    "BaseSegmentationModel",
    "DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT",
    "DEFAULT_MODEL_INPUT_CONTRACT",
    "build_model",
    "describe_model_request",
    "validate_model_input_tensor",
    "validate_model_output",
]
