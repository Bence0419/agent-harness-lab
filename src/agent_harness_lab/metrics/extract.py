"""Derive ahl-metrics-v1 trajectory metrics from a stream of normalized events."""

from __future__ import annotations

from collections.abc import Iterable

from agent_harness_lab.metrics.models import TrajectoryMetrics
from agent_harness_lab.metrics.tools import classify_tool
from agent_harness_lab.metrics.verification import is_verification_command
from agent_harness_lab.traces.models import (
    ModelTurnEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)

_CLASS_FIELD = {
    "read": "read_calls",
    "bash": "bash_calls",
    "edit": "edit_calls",
    "write": "write_calls",
    "other": "other_tool_calls",
}


def extract_trajectory(events: Iterable[TraceEvent]) -> TrajectoryMetrics:
    """Single streaming pass over ``events``. An empty stream yields genuine zeros;
    callers decide whether "no events" means unavailable."""
    counts = dict.fromkeys(
        (
            "model_turns", "tool_calls", "tool_errors", "read_calls", "bash_calls",
            "edit_calls", "write_calls", "other_tool_calls", "verification_calls",
            "input_tokens", "output_tokens", "reasoning_tokens",
            "cache_read_tokens", "cache_write_tokens",
        ),
        0,
    )  # fmt: skip
    cost = 0.0

    for event in events:
        if isinstance(event, ModelTurnEvent):
            counts["model_turns"] += 1
            counts["input_tokens"] += event.input_tokens
            counts["output_tokens"] += event.output_tokens
            counts["reasoning_tokens"] += event.reasoning_tokens
            counts["cache_read_tokens"] += event.cache_read_tokens
            counts["cache_write_tokens"] += event.cache_write_tokens
            cost += event.cost_usd
        elif isinstance(event, ToolCallEvent):
            counts["tool_calls"] += 1
            counts[_CLASS_FIELD[classify_tool(event.name)]] += 1
            if is_verification_command(event.command):
                counts["verification_calls"] += 1
        elif isinstance(event, ToolResultEvent) and event.is_error:
            counts["tool_errors"] += 1

    return TrajectoryMetrics(**counts, cost_usd=round(cost, 6))
