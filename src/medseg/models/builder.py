"""Model builder entry points."""

from __future__ import annotations

from medseg.config.schema import AppConfig, ModelConfig
from medseg.models.base import BaseSegmentationModel
from medseg.models.architectures.unet import UNetRonneberger2015
from medseg.models.architectures.unet2d import UNet2D


def describe_model_request(config: AppConfig | ModelConfig) -> str:
    """Return a compact description of the requested model."""

    model_config = config.model if isinstance(config, AppConfig) else config
    return f"{model_config.architecture}(in_channels={model_config.in_channels}, out_channels={model_config.out_channels})"


def build_model(config: AppConfig | ModelConfig) -> BaseSegmentationModel:
    """Build the configured model.
    """

    model_config = config.model if isinstance(config, AppConfig) else config
    if model_config.architecture == "unet_ronneberger2015":
        return UNetRonneberger2015(model_config)
    if model_config.architecture == "unet2d":
        return UNet2D.from_config(model_config)
    if model_config.architecture == "unet2d_convlora":
        return UNet2D.from_config(model_config)

    raise ValueError(f"Unknown model architecture '{model_config.architecture}'.")
