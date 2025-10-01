"""Utilities for converting evaluation dataclasses into JSON-serializable structures."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses, enums, and sets to plain Python types."""
    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_serializable(item) for item in value)
    return value


__all__ = ["to_serializable"]
