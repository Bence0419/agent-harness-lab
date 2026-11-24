"""OpenCode trace format -> normalized events (rules from the frozen v1 extractor).

* ``tool_use`` is both a tool call and its result; the result is an error if
  ``part.state.status == "error"``. A shell command is exposed for the tools
  named ``shell`` and ``bash``.
* ``step_finish`` with a non-empty ``part.tokens`` is one model turn; usage comes
  from ``part.tokens`` and cost from ``part.cost``.
"""

from __future__ import annotations

from typing import Any

from agent_harness_lab.traces.models import (
    ModelTurnEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)
from agent_harness_lab.traces.values import as_cost, as_count, as_dict

SHELL_TOOLS = {"shell", "bash"}


def normalize_event(e: dict[str, Any]) -> tuple[TraceEvent, ...]:
    kind = e.get("type")

    if kind == "step_finish":
        part = as_dict(e.get("part"))
        tokens = part.get("tokens")
        if tokens and isinstance(tokens, dict):
            cache = as_dict(tokens.get("cache"))
            return (
                ModelTurnEvent(
                    input_tokens=as_count(tokens.get("input")),
                    output_tokens=as_count(tokens.get("output")),
                    reasoning_tokens=as_count(tokens.get("reasoning")),
                    cache_read_tokens=as_count(cache.get("read")),
                    cache_write_tokens=as_count(cache.get("write")),
                    cost_usd=as_cost(part.get("cost")),
                ),
            )

    elif kind == "tool_use":
        part = as_dict(e.get("part"))
        name = part.get("tool", "unknown")
        state = as_dict(part.get("state"))
        command = None
        if name in SHELL_TOOLS:
            raw = as_dict(state.get("input")).get("command")
            command = raw if isinstance(raw, str) else None
        return (
            ToolCallEvent(name=name, command=command),
            ToolResultEvent(is_error=state.get("status") == "error"),
        )

    return ()
