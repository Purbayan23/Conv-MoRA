"""Helpers for future parameter-group strategies."""

from __future__ import annotations


def describe_param_group_strategy(strategy: str = "all") -> str:
    """Describe how trainable parameter groups should be selected."""

    if strategy == "all":
        return "Train all parameters in the selected model."
    return f"Custom parameter-group strategy requested: {strategy}"
