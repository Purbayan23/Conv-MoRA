"""Plain YAML configuration loading helpers."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints

import yaml

from medseg.config.schema import AppConfig

_INTERPOLATION_PATTERN = re.compile(r"\$\{([^{}]+)\}")


def get_project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def get_config_dir() -> Path:
    """Return the default YAML configuration directory."""

    return get_project_root() / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration file must contain a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    current = mapping
    parts = path.replace("/", ".").split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _lookup(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _resolve_expression(expression: str, config: dict[str, Any]) -> Any:
    expression = expression.strip()
    if expression.startswith("oc.env:"):
        body = expression[len("oc.env:") :]
        name, separator, default = body.partition(",")
        if name.strip() in os.environ:
            return os.environ[name.strip()]
        return default.strip() if separator else ""
    return _lookup(config, expression)


def _resolve_string(value: str, config: dict[str, Any]) -> Any:
    exact = _INTERPOLATION_PATTERN.fullmatch(value.strip())
    if exact:
        try:
            return _resolve_expression(exact.group(1), config)
        except KeyError:
            return value

    def replace(match: re.Match[str]) -> str:
        try:
            resolved = _resolve_expression(match.group(1), config)
        except KeyError:
            return match.group(0)
        return str(resolved)

    return _INTERPOLATION_PATTERN.sub(replace, value)


def _resolve_interpolations(config: dict[str, Any]) -> None:
    for _ in range(10):
        changed = False

        def resolve(value: Any) -> Any:
            nonlocal changed
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, str):
                resolved = _resolve_string(value, config)
                changed = changed or resolved != value
                return resolved
            return value

        resolved_config = resolve(config)
        config.clear()
        config.update(resolved_config)
        if not changed:
            return
    raise ValueError("Configuration interpolation did not converge.")


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Configuration override must use key=value syntax: {override}")
    key, raw_value = override.split("=", 1)
    key = key.lstrip("+").strip()
    if not key:
        raise ValueError(f"Configuration override has an empty key: {override}")
    return key, yaml.safe_load(raw_value)


def _apply_overrides(config: dict[str, Any], overrides: Sequence[str], config_dir: Path) -> None:
    for override in overrides:
        key, value = _parse_override(override)
        group_path = config_dir / Path(*key.split(".")) / f"{value}.yaml"
        if "." not in key and group_path.exists():
            _set_nested(config, key, _load_yaml(group_path))
        else:
            _set_nested(config, key, value)


def _compose_dict(
    config_name: str,
    overrides: Sequence[str],
    config_dir: Path,
) -> dict[str, Any]:
    root_path = config_dir / f"{config_name.removesuffix('.yaml')}.yaml"
    root = _load_yaml(root_path)
    composed: dict[str, Any] = {}

    for default in root.pop("defaults", []) or []:
        if default == "_self_" or not isinstance(default, dict):
            continue
        group, name = next(iter(default.items()))
        group_path = config_dir / Path(*str(group).split("/")) / f"{name}.yaml"
        _set_nested(composed, str(group), _load_yaml(group_path))

    _deep_merge(composed, root)
    _apply_overrides(composed, overrides, config_dir)
    _resolve_interpolations(composed)
    return composed


def _coerce_value(value: Any, annotation: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = [option for option in get_args(annotation) if option is not type(None)]
        return _coerce_value(value, options[0]) if options else value
    if origin is list:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_coerce_value(item, item_type) for item in value]
    if origin is tuple:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return tuple(_coerce_value(item, item_type) for item in value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError(f"Expected a mapping for {annotation.__name__}.")
        return _to_dataclass(annotation, value)
    if annotation in (bool, int, float, str):
        return annotation(value)
    return value


def _to_dataclass(dataclass_type: type[Any], values: dict[str, Any]) -> Any:
    annotations = get_type_hints(dataclass_type)
    kwargs = {
        field.name: _coerce_value(values[field.name], annotations.get(field.name, field.type))
        for field in fields(dataclass_type)
        if field.name in values
    }
    return dataclass_type(**kwargs)


def compose_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load grouped YAML files and return the typed application configuration."""

    resolved_config_dir = config_dir or get_config_dir()
    raw_config = _compose_dict(config_name, list(overrides or ()), resolved_config_dir)
    return _to_dataclass(AppConfig, raw_config)


def load_typed_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load the composed YAML configuration as nested dataclasses."""

    return compose_config(config_name=config_name, overrides=overrides, config_dir=config_dir)
