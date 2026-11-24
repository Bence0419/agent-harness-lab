"""Sources of trace-derived metrics for normalized runs.

Phase 2 joins the committed Astra CSV. Phase 3 will add a source that parses
``trace.jsonl`` directly and implements the same ``enrich`` interface, so the
CSV dependency can be dropped without touching RunRecord or the reports.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

from agent_harness_lab.artifacts.models import RunRecord, TraceMetrics

_INT_FIELDS = (
    "model_turns",
    "tool_calls",
    "tool_errors",
    "read_calls",
    "bash_calls",
    "edit_calls",
    "write_calls",
    "other_tool_calls",
    "verification_calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "files_changed",
    "test_files_changed",
    "non_test_files_changed",
    "insertions",
    "deletions",
)


class TraceMetricsSource(Protocol):
    def enrich(self, records: list[RunRecord]) -> tuple[list[RunRecord], list[str]]:
        """Return records with metrics attached where available, plus warnings."""
        ...


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


class AstraCsvMetrics:
    """Regression oracle: join ``astra-trajectory-metrics.csv`` onto the runs it describes.

    The CSV has no run id or model column, so coverage is defined the same way
    the extractor that produced it selected runs (model ``gpt-6-astra`` and a
    ``frontier-astra-v1`` reference), then matched on (task, harness). A row is
    attached only if that key is unique and the CSV's outcome and duration agree
    with the run's own metadata/evaluation; otherwise it is skipped with a warning.
    """

    SOURCE_NAME = "frozen-csv"
    MODEL = "gpt-6-astra"
    REFERENCE_MARKER = "frontier-astra-v1"

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def _load_rows(self) -> dict[tuple[str, str], dict[str, str]]:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            return {(r["task"], r["harness"]): r for r in csv.DictReader(f)}

    def _covers(self, r: RunRecord) -> bool:
        return r.model == self.MODEL and self.REFERENCE_MARKER in (r.reference or "")

    def enrich(self, records: list[RunRecord]) -> tuple[list[RunRecord], list[str]]:
        if not self.csv_path.is_file():
            return records, []
        try:
            rows = self._load_rows()
        except (OSError, KeyError, csv.Error) as exc:
            return records, [f"could not read {self.csv_path.name}: {exc}"]

        covered: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(records):
            if self._covers(r):
                covered.setdefault((r.task, r.harness), []).append(i)

        out = list(records)
        warnings: list[str] = []
        for key, idxs in covered.items():
            row = rows.get(key)
            if row is None:
                continue
            if len(idxs) > 1:
                warnings.append(
                    f"trace metrics not attached for {key[0]}/{key[1]}: "
                    f"{len(idxs)} runs share the CSV key"
                )
                continue
            rec = records[idxs[0]]
            if not self._consistent(rec, row):
                warnings.append(
                    f"trace metrics not attached for {rec.run_id}: "
                    f"{self.csv_path.name} disagrees with run metadata/evaluation"
                )
                continue
            metrics = TraceMetrics(
                **{f: _to_int(row.get(f)) for f in _INT_FIELDS},
                cost_usd=_to_float(row.get("cost_usd")),
                metrics_source=self.SOURCE_NAME,
            )
            out[idxs[0]] = rec.model_copy(update=metrics.model_dump())
        return out, warnings

    @staticmethod
    def _consistent(rec: RunRecord, row: dict[str, str]) -> bool:
        csv_resolved = {"True": True, "False": False}.get(row.get("resolved", ""))
        if rec.resolved is not None and csv_resolved is not None and rec.resolved != csv_resolved:
            return False
        csv_duration = _to_int(row.get("duration_s"))
        return not (
            rec.duration_seconds is not None
            and csv_duration is not None
            and rec.duration_seconds != csv_duration
        )
