"""Model builder entry points."""

from __future__ import annotations

from medseg.config.schema import AppConfig, ModelConfig
from medseg.models.base import BaseSegmentationModel


def describe_model_request(config: AppConfig | ModelConfig) -> str:
    """Return a compact description of the requested model."""

    model_config = config.model if isinstance(config, AppConfig) else config
    return f"{model_config.architecture}(in_channels={model_config.in_channels}, out_channels={model_config.out_channels})"


def build_model(config: AppConfig | ModelConfig) -> BaseSegmentationModel:
    """Build the configured model.

    The builder entry point exists now so future architectures can be swapped by
    configuration only. The baseline U-Net implementation is intentionally not
    provided in this scaffold step.
    """

    model_config = config.model if isinstance(config, AppConfig) else config
    if model_config.architecture == "unet_ronneberger2015":
        raise NotImplementedError(
            "The scaffold is ready for 'unet_ronneberger2015', but the model has not been implemented yet."
        )

    raise ValueError(f"Unknown model architecture '{model_config.architecture}'.")
