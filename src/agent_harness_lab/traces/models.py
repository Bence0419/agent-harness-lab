"""Harness-neutral trace events.

Harness-specific parsers translate raw JSONL objects into these events; metric
extraction and reporting only ever see these types.

* ``ToolCallEvent``  - the agent invoked a tool. ``command`` is set only when the
  harness exposes a shell command for the tool (used for the verification heuristic).
* ``ToolResultEvent`` - a tool invocation finished; ``is_error`` says whether it failed.
* ``ModelTurnEvent`` - one model response with its token usage and cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallEvent:
    name: str | None
    command: str | None = None


@dataclass(frozen=True)
class ToolResultEvent:
    is_error: bool


@dataclass(frozen=True)
class ModelTurnEvent:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


TraceEvent = ToolCallEvent | ToolResultEvent | ModelTurnEvent
