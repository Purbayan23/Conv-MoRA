"""Helpers for composing and validating Hydra/OmegaConf configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from medseg.config.schema import AppConfig


def get_project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def get_config_dir() -> Path:
    """Return the default Hydra configuration directory."""

    return get_project_root() / "configs"


def compose_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> DictConfig:
    """Compose a Hydra config and merge it with the structured schema."""

    resolved_config_dir = config_dir or get_config_dir()
    with initialize_config_dir(version_base=None, config_dir=str(resolved_config_dir)):
        raw_config = compose(
            config_name=config_name,
            overrides=list(overrides or ()),
            return_hydra_config=False,
        )

    raw_config.pop("hydra", None)
    structured = OmegaConf.structured(AppConfig())
    merged = OmegaConf.merge(structured, raw_config)
    OmegaConf.resolve(merged)
    return cast(DictConfig, merged)


def load_typed_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load the composed config as nested dataclass instances."""

    config = compose_config(config_name=config_name, overrides=overrides, config_dir=config_dir)
    typed_config = OmegaConf.to_object(config)
    if not isinstance(typed_config, AppConfig):
        raise TypeError("Expected composed configuration to resolve to an AppConfig instance.")
    return typed_config
