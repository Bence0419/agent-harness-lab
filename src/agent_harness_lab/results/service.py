"""Load, enrich and filter normalized runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from agent_harness_lab.artifacts.discovery import discover_runs
from agent_harness_lab.artifacts.models import LoadIssue, RunRecord
from agent_harness_lab.results.native_metrics import NativeTraceMetrics
from agent_harness_lab.results.trace_metrics import AstraCsvMetrics, TraceMetricsSource

ASTRA_CSV_RELATIVE = Path("..") / "analysis" / "astra-trajectory-metrics.csv"


class ResultsDirNotFound(Exception):
    pass


@dataclass
class LoadedRuns:
    records: list[RunRecord] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)


class MetricsSourceName(StrEnum):
    NATIVE = "native"  # parse trace.jsonl / model.patch (default)
    FROZEN_CSV = "frozen-csv"  # diagnostic/regression oracle: committed Astra CSV
    NONE = "none"  # skip trace-derived metrics entirely (fast)


def build_metrics_source(name: MetricsSourceName, raw_dir: Path) -> TraceMetricsSource | None:
    if name is MetricsSourceName.NATIVE:
        return NativeTraceMetrics()
    if name is MetricsSourceName.FROZEN_CSV:
        return AstraCsvMetrics((raw_dir / ASTRA_CSV_RELATIVE).resolve())
    return None


def load_runs(raw_dir: Path) -> LoadedRuns:
    """Lightweight load: run metadata and evaluation only. Never reads traces."""
    if not raw_dir.is_dir():
        raise ResultsDirNotFound(str(raw_dir))
    found = discover_runs(raw_dir)
    records = sorted(
        found.records,
        key=lambda r: (r.task, r.harness, r.started_at is None, r.started_at, r.run_id),
    )
    return LoadedRuns(records=records, issues=found.issues)


def attach_metrics(
    records: list[RunRecord], source: TraceMetricsSource | None
) -> tuple[list[RunRecord], list[str]]:
    """Attach trace-derived metrics. Call this AFTER filtering, on selected runs only."""
    if source is None or not records:
        return records, []
    return source.enrich(records)


def _any_substring(needles: Sequence[str], value: str | None) -> bool:
    if not needles:
        return True
    return value is not None and any(n.lower() in value.lower() for n in needles)


def filter_runs(
    records: Sequence[RunRecord],
    *,
    tasks: Sequence[str] = (),
    harnesses: Sequence[str] = (),
    models: Sequence[str] = (),
    references: Sequence[str] = (),
) -> list[RunRecord]:
    """Filters AND together; repeated values for one filter OR together.

    ``harness`` matches exactly (case-insensitive); task, model and reference
    match as case-insensitive substrings.
    """
    wanted_harness = {h.lower() for h in harnesses}
    return [
        r
        for r in records
        if _any_substring(tasks, r.task)
        and (not wanted_harness or r.harness.lower() in wanted_harness)
        and _any_substring(models, r.model)
        and _any_substring(references, r.reference)
    ]
