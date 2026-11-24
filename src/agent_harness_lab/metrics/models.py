"""Metric schema identifier and the trajectory metrics object."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# The semantics of every metric below are defined by the frozen research
# extractor (scripts/extract-trajectory-metrics.py). Any change to those
# semantics must introduce a new schema id, never alter this one.
METRICS_SCHEMA_V1 = "ahl-metrics-v1"


class TrajectoryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics_schema: Literal["ahl-metrics-v1"] = METRICS_SCHEMA_V1

    model_turns: int
    tool_calls: int
    tool_errors: int
    read_calls: int
    bash_calls: int
    edit_calls: int
    write_calls: int
    other_tool_calls: int
    verification_calls: int

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    cost_usd: float  # sum of per-turn cost, rounded to 6 decimals as in v1
