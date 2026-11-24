"""Defensive value coercion shared by trace parsers.

Missing or non-numeric values count as 0, matching the frozen extractor's
``value or 0`` behaviour for absent fields.
"""

from __future__ import annotations

from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def as_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
