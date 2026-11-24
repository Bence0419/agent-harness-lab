"""Native trace metrics: parse each selected run's ``trace.jsonl`` and ``model.patch``.

Lazy by construction: nothing is read until ``enrich`` is called, and callers
pass only the runs that survived filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness_lab.artifacts.models import RunRecord
from agent_harness_lab.metrics.extract import extract_trajectory
from agent_harness_lab.metrics.models import METRICS_SCHEMA_V1
from agent_harness_lab.metrics.patch import GitApplyNumstat, NumstatProbe, PatchStatError, patch_stats
from agent_harness_lab.traces.jsonl import TraceReport
from agent_harness_lab.traces.normalize import NORMALIZERS, read_events

TRACE_FILE = "trace.jsonl"
PATCH_FILE = "model.patch"
EXTRA_TRACE_FILES = ("verification-trace.jsonl",)  # additional passes the v1 metrics never read


class NativeTraceMetrics:
    SOURCE_NAME = "native-trace"

    def __init__(self, patch_probe: NumstatProbe | None = None) -> None:
        self._probe = patch_probe or GitApplyNumstat()

    def enrich(self, records: list[RunRecord]) -> tuple[list[RunRecord], list[str]]:
        out: list[RunRecord] = []
        warnings: list[str] = []
        for record in records:
            update, notes = self._metrics_for(record)
            warnings.extend(f"{record.run_id}: {n}" for n in notes)
            out.append(record.model_copy(update=update) if update else record)
        return out, warnings

    def _metrics_for(self, record: RunRecord) -> tuple[dict[str, Any], list[str]]:
        update: dict[str, Any] = {}
        notes: list[str] = []

        # --- trajectory metrics from the trace --------------------------------
        trace_path = record.artifact_dir / TRACE_FILE
        if record.harness not in NORMALIZERS:
            notes.append(f"trace metrics unavailable: unsupported harness {record.harness!r}")
        else:
            report = TraceReport(path=trace_path)
            try:
                metrics = extract_trajectory(read_events(trace_path, record.harness, report))
            except OSError as exc:
                metrics = None
                notes.append(f"trace metrics unavailable: cannot read {TRACE_FILE}: {exc}")
            else:
                problem = report.summary()
                if problem:
                    suffix = "" if report.usable else " (trace metrics unavailable)"
                    notes.append(f"{TRACE_FILE}: {problem}{suffix}")
                if not report.usable:
                    metrics = None  # missing/empty/unparseable trace: unavailable, not zero
            if metrics is not None:
                update.update(metrics.model_dump())

        for extra in EXTRA_TRACE_FILES:
            if (record.artifact_dir / extra).is_file():
                notes.append(
                    f"metrics cover {TRACE_FILE} only; {extra} (additional agent pass) is not included"
                )

        # --- patch metrics ----------------------------------------------------
        try:
            stats = patch_stats(record.artifact_dir / PATCH_FILE, self._probe)
        except (PatchStatError, OSError) as exc:
            stats = None
            notes.append(f"patch metrics unavailable: {exc}")
        if stats is not None:
            update.update(stats.model_dump())

        if update:
            update["metrics_source"] = self.SOURCE_NAME
            update.setdefault("metrics_schema", METRICS_SCHEMA_V1)
        return update, notes
