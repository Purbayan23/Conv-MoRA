"""Lightweight registry utilities for pluggable components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Simple named registry for components that will be selected by config."""

    name: str
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, key: str, item: T) -> None:
        if key in self._items:
            raise KeyError(f"'{key}' is already registered in '{self.name}'.")
        self._items[key] = item

    def get(self, key: str) -> T:
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown key '{key}' for registry '{self.name}'. Available: {available}")
        return self._items[key]

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def make_registration_callback(registry: Registry[T], key: str) -> Callable[[T], T]:
    """Return a decorator-style callback for future component registration."""

    def _callback(item: T) -> T:
        registry.register(key, item)
        return item

    return _callback
