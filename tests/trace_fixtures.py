"""Trimmed trace lines in the real Pi and OpenCode formats (verified on historical traces)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def jl(*events: dict[str, Any]) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


# ---- Pi ------------------------------------------------------------------
PI_SYSTEM_END = {"type": "message_end", "message": {"role": "system", "content": ""}}
PI_USER_END = {"type": "message_end", "message": {"role": "user", "content": "fix it"}}


def pi_turn(inp=3, out=45, reasoning=0, cache_read=0, cache_write=1351, cost=0.0191675, **extra):
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "provider": "openai",
            "model": "gpt-6-astra",
            "usage": {
                "input": inp,
                "output": out,
                "cacheRead": cache_read,
                "cacheWrite": cache_write,
                "reasoning": reasoning,
                "totalTokens": 1399,
                "cost": {"total": cost, "input": 3e-05},
            },
            **extra,
        },
    }


def pi_start(tool, **args):
    return {"type": "tool_execution_start", "toolCallId": "c1", "toolName": tool, "args": args}


def pi_end(tool="bash", is_error=False):
    return {
        "type": "tool_execution_end",
        "toolCallId": "c1",
        "toolName": tool,
        "result": {"content": [], "isError": is_error},
        "isError": is_error,
    }


# ---- OpenCode ------------------------------------------------------------
def oc_finish(inp=3, out=23, reasoning=0, read=0, write=4565, cost=0.0582425):
    return {
        "type": "step_finish",
        "part": {
            "type": "step-finish",
            "cost": cost,
            "tokens": {
                "input": inp,
                "output": out,
                "reasoning": reasoning,
                "cache": {"read": read, "write": write},
            },
        },
    }


def oc_tool(tool, status="completed", **tool_input):
    return {
        "type": "tool_use",
        "part": {"type": "tool", "tool": tool, "state": {"status": status, "input": tool_input}},
    }


def write_trace(run_dir: Path, text: str) -> Path:
    path = run_dir / "trace.jsonl"
    path.write_text(text, encoding="utf-8")
    return path
