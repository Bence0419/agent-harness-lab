"""Experiment summary: counts by outcome, correctness with numerator/denominator, totals, and
groupings by harness and by model.

Pure derivation from persisted records (matrix.json plus each run's canonical run.json,
evaluation.json and trace), so it can be recomputed at any time and after a crash. Metric
aggregation is the existing ``summarize_runs`` (also used by ``ahl compare``): nothing is
re-implemented here, and missing data stays ``None`` rather than becoming zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agent_harness_lab.artifacts.ahl_run import load_ahl_run
from agent_harness_lab.artifacts.legacy import LegacyFormatError
from agent_harness_lab.artifacts.models import RunRecord
from agent_harness_lab.experiments.store import (
    SUMMARY_SCHEMA,
    CellRecord,
    CellState,
    ExperimentRecord,
    MatrixDocument,
)
from agent_harness_lab.reporting.compare import SUMMED_METRICS, summarize_runs
from agent_harness_lab.results.service import TraceMetricsSource, attach_metrics

NOTE = (
    "Counts and rates are descriptive. No statistical significance is claimed. Cells without a "
    "benchmark verdict (infrastructure failures, harness failures, timeouts, ambiguous results) "
    "are excluded from the correctness denominator. Only the latest attempt of a cell counts."
)

_PENDING = (CellState.PLANNED, CellState.RUNNING, CellState.AGENT_COMPLETED)
_EXECUTED_STATES = ("completed", "harness_failed", "timed_out")  # the agent really ran


def _counts(cells: Sequence[CellRecord]) -> dict[str, int]:
    by_state = Counter(c.state for c in cells)
    evaluated = [c for c in cells if c.current and c.current.evaluation_state is not None]
    infra_eval = sum(
        1
        for c in cells
        if c.state is CellState.NO_VERDICT
        and c.current
        and c.current.classification == "evaluation_failure"
    )
    return {
        "planned": len(cells),
        "executed": sum(1 for c in cells if c.current and c.current.run_id is not None),
        "evaluated": len(evaluated),
        "passed": by_state[CellState.PASSED],
        "failed": by_state[CellState.FAILED],
        "no_verdict": by_state[CellState.NO_VERDICT],
        # execution infrastructure failures + evaluation infrastructure failures
        "infrastructure_failures": by_state[CellState.INFRA_FAILURE] + infra_eval,
        "harness_failures": by_state[CellState.HARNESS_FAILED],
        "timeouts": by_state[CellState.TIMED_OUT],
        "completed_without_evaluation": by_state[CellState.COMPLETED],
        "pending": sum(by_state[s] for s in _PENDING),
    }


def _durations(cells: Sequence[CellRecord]) -> dict[str, float | None]:
    def total(values: list[float | None]) -> float | None:
        known = [v for v in values if v is not None]
        return round(sum(known), 3) if known else None

    return {
        "agent_duration_seconds": total(
            [c.current.agent_duration_seconds if c.current else None for c in cells]
        ),
        "evaluation_duration_seconds": total(
            [c.current.evaluation_duration_seconds if c.current else None for c in cells]
        ),
    }


def _load_records(
    record: ExperimentRecord,
    cells: Sequence[CellRecord],
    raw_dir: Path,
    metrics: TraceMetricsSource | None,
) -> tuple[dict[str, RunRecord], list[str]]:
    """RunRecords (via the existing loaders) for cells whose agent actually ran."""
    loaded: dict[str, RunRecord] = {}
    notes: list[str] = []
    for c in cells:
        a = c.current
        if a is None or a.run_path is None or a.execution_state not in _EXECUTED_STATES:
            continue
        try:
            loaded[c.cell.cell_id] = load_ahl_run(raw_dir / a.run_path)
        except LegacyFormatError as exc:
            notes.append(f"{c.cell.cell_id}: run record unreadable ({exc})")
    ids = list(loaded)
    enriched, warnings = attach_metrics([loaded[i] for i in ids], metrics)
    notes.extend(warnings)
    return dict(zip(ids, enriched, strict=True)), notes


def _group(
    key: str,
    cells: Sequence[CellRecord],
    records: dict[str, RunRecord],
) -> dict:
    group_records = [records[c.cell.cell_id] for c in cells if c.cell.cell_id in records]
    s = summarize_runs(key, group_records)
    counts = _counts(cells)
    return {
        "cells": len(cells),
        **{k: counts[k] for k in ("passed", "failed", "no_verdict", "infrastructure_failures")},
        "harness_failures": counts["harness_failures"],
        "timeouts": counts["timeouts"],
        "resolved": s.resolved,
        "evaluated": s.evaluated,
        "success_rate": s.success_rate,
        "runs_with_metrics": len(group_records),
        "totals": s.totals,
    }


def build_summary(
    record: ExperimentRecord,
    matrix: MatrixDocument,
    raw_dir: Path,
    *,
    metrics: TraceMetricsSource | None,
    status: str,
    now: str,
) -> dict:
    cells = matrix.cells
    records, notes = _load_records(record, cells, raw_dir, metrics)
    overall = summarize_runs("all", list(records.values()))
    by_harness = {
        h: _group(h, [c for c in cells if c.cell.harness == h], records)
        for h in sorted({c.cell.harness for c in cells})
    }

    def model_key(c: CellRecord) -> str:
        return f"{c.cell.model_ref} (reasoning={c.cell.reasoning or 'none'})"

    by_model = {
        m: _group(m, [c for c in cells if model_key(c) == m], records)
        for m in sorted({model_key(c) for c in cells})
    }
    totals = {attr: overall.totals[attr] for _, attr in SUMMED_METRICS}
    return {
        "schema_version": SUMMARY_SCHEMA,
        "experiment_id": record.experiment_id,
        "definition_sha256": record.definition_sha256,
        "status": status,  # running | interrupted | incomplete | complete
        "updated_at": now,
        "overrides": list(record.overrides),
        "counts": _counts(cells),
        "correctness": {
            "resolved": overall.resolved,
            "denominator": overall.evaluated,
            "success_rate": overall.success_rate,
        },
        "durations": _durations(cells),
        "metrics": {
            "runs_with_metrics": len(records),
            "totals": totals,  # None = at least one included run lacks it (never zero-filled)
        },
        "by_harness": by_harness,
        "by_model": by_model,
        "notes": [NOTE, *notes],
    }
