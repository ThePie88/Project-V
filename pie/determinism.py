"""Deterministic helpers shared across the kernel."""

from __future__ import annotations

from typing import Iterable, Callable, TypeVar, List

T = TypeVar("T")


def clamp_round(value: float, min_value: float, max_value: float, decimals: int) -> float:
    """Clamp and round a float deterministically."""
    v = min(max(value, min_value), max_value)
    return round(v, decimals)


def stable_sort(items: Iterable[T], key: Callable[[T], str]) -> List[T]:
    """Return items sorted by a deterministic string key."""
    return sorted(items, key=key)
