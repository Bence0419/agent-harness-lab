from __future__ import annotations

from agent_harness_lab.harnesses.base import HarnessAdapter, HarnessError
from agent_harness_lab.harnesses.opencode import OpenCodeAdapter
from agent_harness_lab.harnesses.pi import PiAdapter

_ADAPTERS: dict[str, HarnessAdapter] = {a.name: a for a in (PiAdapter(), OpenCodeAdapter())}


def available_harnesses() -> list[str]:
    return sorted(_ADAPTERS)


def get_harness(name: str) -> HarnessAdapter:
    try:
        return _ADAPTERS[name.lower()]
    except KeyError:
        raise HarnessError(
            f"unknown harness {name!r}; available: {', '.join(available_harnesses())}"
        ) from None
