"""Pi trace format -> normalized events (rules from the frozen v1 extractor).

* ``tool_execution_start`` is a tool call. A shell command is exposed only for
  the tool named ``bash``.
* ``tool_execution_end`` is a tool result; it is an error if ``isError`` is true,
  ``error`` is present, or ``result.isError`` is true.
* An assistant ``message_end`` that names a model is one model turn; usage comes
  from ``message.usage`` (cost from ``usage.cost.total``).
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


def normalize_event(e: dict[str, Any]) -> tuple[TraceEvent, ...]:
    kind = e.get("type")

    if kind == "message_end":
        msg = as_dict(e.get("message"))
        if msg.get("role") == "assistant" and msg.get("model"):
            usage = as_dict(msg.get("usage"))
            cost = as_dict(usage.get("cost"))
            return (
                ModelTurnEvent(
                    input_tokens=as_count(usage.get("input")),
                    output_tokens=as_count(usage.get("output")),
                    reasoning_tokens=as_count(usage.get("reasoning")),
                    cache_read_tokens=as_count(usage.get("cacheRead")),
                    cache_write_tokens=as_count(usage.get("cacheWrite")),
                    cost_usd=as_cost(cost.get("total")),
                ),
            )

    elif kind == "tool_execution_start":
        name = e.get("toolName", "unknown")
        command = None
        if name == "bash":
            raw = as_dict(e.get("args")).get("command")
            command = raw if isinstance(raw, str) else None
        return (ToolCallEvent(name=name, command=command),)

    elif kind == "tool_execution_end":
        is_error = (
            e.get("isError") is True
            or e.get("error") is not None
            or as_dict(e.get("result")).get("isError") is True
        )
        return (ToolResultEvent(is_error=is_error),)

    return ()
