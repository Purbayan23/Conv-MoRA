"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path
