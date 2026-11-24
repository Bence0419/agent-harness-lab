"""Registry and entry point: raw trace file -> stream of normalized events."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from agent_harness_lab.traces import opencode, pi
from agent_harness_lab.traces.jsonl import TraceReport, iter_json_objects
from agent_harness_lab.traces.models import TraceEvent

Normalizer = Callable[[dict[str, Any]], tuple[TraceEvent, ...]]

# Keyed by RunRecord.harness, which comes from structured run metadata,
# never from directory names or trace size.
NORMALIZERS: dict[str, Normalizer] = {
    "Pi": pi.normalize_event,
    "OpenCode": opencode.normalize_event,
}


class UnsupportedHarness(Exception):
    pass


def read_events(path: Path, harness: str, report: TraceReport) -> Iterator[TraceEvent]:
    """Stream normalized events from ``path``. Lazy: nothing is read until iterated."""
    normalizer = NORMALIZERS.get(harness)
    if normalizer is None:
        raise UnsupportedHarness(harness)
    for obj in iter_json_objects(path, report):
        yield from normalizer(obj)
