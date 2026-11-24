"""Normalized, read-only view of a completed run.

legacy artifact -> legacy loader -> RunRecord

This is NOT the future ``run.json`` schema. It is an internal model that lets
historical runs (several ``metadata.json`` shapes) be queried uniformly.
Every field that older runs cannot supply is Optional; nothing is invented.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class EvaluationState(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    EMPTY_PATCH = "empty_patch"
    PATCH_APPLY_FAILED = "patch_apply_failed"  # the evaluator could not apply model.patch
    INFRA_FAILURE = "infra_failure"
    AMBIGUOUS_FAILURE = "ambiguous_failure"
    ERROR = "error"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"  # evaluation.json present but no recognised outcome flag
    NOT_EVALUATED = "not_evaluated"  # evaluation.json missing
    INVALID = "invalid"  # evaluation.json malformed or for a different run


class TraceMetrics(BaseModel):
    """Trace/patch-derived metrics. Filled by a metrics source, never by metadata."""

    model_config = ConfigDict(frozen=True)

    # Provenance: which metric schema/source produced the values below (None = none attached).
    metrics_schema: str | None = None
    metrics_source: str | None = None

    model_turns: int | None = None
    tool_calls: int | None = None
    tool_errors: int | None = None
    read_calls: int | None = None
    bash_calls: int | None = None
    edit_calls: int | None = None
    write_calls: int | None = None
    other_tool_calls: int | None = None
    verification_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    files_changed: int | None = None
    test_files_changed: int | None = None
    non_test_files_changed: int | None = None
    insertions: int | None = None
    deletions: int | None = None


class RunRecord(TraceMetrics):
    run_id: str
    task: str
    harness: str
    harness_version: str | None = None
    model: str | None = None
    provider: str | None = None
    reasoning: str | None = None
    reference: str | None = None
    started_at: datetime | None = None
    duration_seconds: int | None = None
    resolved: bool | None = None
    evaluation_state: EvaluationState = EvaluationState.NOT_EVALUATED
    harness_commit: str | None = None
    harness_dirty: bool | None = None
    task_repo_commit: str | None = None
    base_commit: str | None = None
    docker_image: str | None = None
    network_policy: str | None = None
    artifact_dir: Path
    legacy_format: str  # which metadata shape / run schema this was parsed from
    execution_state: str | None = None  # ahl-run-v1 runs only (historical runs have none)
    purpose: str | None = None  # ahl-run-v1 runs only, e.g. "canary"


class LoadIssue(BaseModel):
    """A run directory that could not be loaded. Reported, never hidden."""

    path: Path
    reason: str
