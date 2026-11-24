"""Persisted experiment state: ``experiment.yaml`` (byte copy), ``experiment.json``,
``matrix.json`` and ``summary.json`` under ``results/experiments/<experiment-id>/``.

Every write is atomic. matrix.json is rewritten after EVERY state transition, so an interrupted
orchestrator leaves a consistent, resumable record. Reading never needs Docker, an API key or
any external service.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agent_harness_lab.artifacts.writer import write_bytes_atomic, write_json_atomic
from agent_harness_lab.experiments.errors import ExperimentError
from agent_harness_lab.experiments.matrix import MatrixCell
from agent_harness_lab.experiments.models import NormalizedExperiment

EXPERIMENT_FILE = "experiment.json"
EXPERIMENT_YAML = "experiment.yaml"
MATRIX_FILE = "matrix.json"
SUMMARY_FILE = "summary.json"
MATRIX_SCHEMA = "ahl-matrix-v1"
RECORD_SCHEMA = "ahl-experiment-record-v1"
SUMMARY_SCHEMA = "ahl-experiment-summary-v1"


class CellState(StrEnum):
    PLANNED = "planned"  # nothing started
    RUNNING = "running"  # an attempt was launched; no outcome recorded (crash/interrupt ⇒ stale)
    AGENT_COMPLETED = "agent_completed"  # agent finished OK; evaluation still to do
    COMPLETED = "completed"  # agent finished OK; evaluation disabled (nothing more to do)
    PASSED = "passed"  # evaluated: benchmark PASS
    FAILED = "failed"  # evaluated: benchmark FAIL (counted in the denominator)
    NO_VERDICT = "no_verdict"  # evaluated, but infrastructure/ambiguous: neither pass nor fail
    HARNESS_FAILED = "harness_failed"  # harness exited non-zero; NOT evaluated; evidence kept
    TIMED_OUT = "timed_out"  # agent timeout; NOT evaluated; evidence kept
    INFRA_FAILURE = "infra_failure"  # execution infrastructure failed, or interrupted


TERMINAL = frozenset(
    {
        CellState.COMPLETED,
        CellState.PASSED,
        CellState.FAILED,
        CellState.NO_VERDICT,
        CellState.HARNESS_FAILED,
        CellState.TIMED_OUT,
        CellState.INFRA_FAILURE,
    }
)
# Terminal states that a retry (a NEW attempt, never an overwrite) may replace on request.
RETRYABLE = frozenset(
    {
        CellState.INFRA_FAILURE,
        CellState.HARNESS_FAILED,
        CellState.TIMED_OUT,
        CellState.NO_VERDICT,
    }
)


class AttemptRecord(BaseModel):
    attempt: int
    run_id: str | None = None  # None until the executor allocated a run directory
    run_path: str | None = None  # relative to the raw directory
    started_at: str
    finished_at: str | None = None
    execution_state: str | None = None
    failed_stage: str | None = None
    exit_code: int | None = None
    agent_duration_seconds: float | None = None
    evaluation_state: str | None = None
    classification: str | None = None
    resolved: bool | None = None
    evaluation_reason: str | None = None
    evaluation_duration_seconds: float | None = None
    error: str | None = None  # already redacted by the executor/evaluator


class CellRecord(BaseModel):
    cell: MatrixCell
    state: CellState = CellState.PLANNED
    attempts: list[AttemptRecord] = Field(default_factory=list)

    @property
    def current(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None


class MatrixDocument(BaseModel):
    schema_version: Literal["ahl-matrix-v1"] = MATRIX_SCHEMA
    experiment_id: str
    cells: list[CellRecord]


class AhlProvenance(BaseModel):
    commit: str | None = None
    dirty: bool | None = None


class TaskRepoProvenance(BaseModel):
    commit: str | None = None
    dirty: bool | None = None
    pinned_commit: str | None = None


class ExperimentRecord(BaseModel):
    schema_version: Literal["ahl-experiment-record-v1"] = RECORD_SCHEMA
    experiment_id: str
    definition_sha256: str
    started_at: str
    finished_at: str | None = None
    raw_dir: str
    # development overrides taken at launch (NOT part of the definition hash); empty = strict
    overrides: list[str] = Field(default_factory=list)
    ahl: AhlProvenance = Field(default_factory=AhlProvenance)
    task_repository: TaskRepoProvenance = Field(default_factory=TaskRepoProvenance)
    definition: NormalizedExperiment


class ExperimentStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def experiment_id(self) -> str:
        return self.directory.name

    # ---- creation ---------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        experiments_dir: Path,
        record: ExperimentRecord,
        raw_yaml: bytes,
        cells: list[MatrixCell],
    ) -> ExperimentStore:
        experiments_dir.mkdir(parents=True, exist_ok=True)
        directory = experiments_dir / record.experiment_id
        directory.mkdir()  # FileExistsError: an experiment directory is never reused
        store = cls(directory)
        write_bytes_atomic(directory / EXPERIMENT_YAML, raw_yaml)  # byte-for-byte
        store.save_record(record)
        store.save_matrix(
            MatrixDocument(
                experiment_id=record.experiment_id, cells=[CellRecord(cell=c) for c in cells]
            )
        )
        return store

    @classmethod
    def open(cls, experiments_dir: Path, experiment_id: str) -> ExperimentStore:
        directory = experiments_dir / experiment_id
        if not (directory / EXPERIMENT_FILE).is_file():
            raise ExperimentError(f"experiment not found: {experiment_id} (in {experiments_dir})")
        return cls(directory)

    # ---- read ---------------------------------------------------------------------------
    def _read(self, name: str, model: type[BaseModel]) -> Any:
        try:
            return model.model_validate_json((self.directory / name).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExperimentError(f"cannot read {name} of {self.experiment_id}") from exc
        except (ValidationError, ValueError) as exc:
            raise ExperimentError(
                f"{name} of {self.experiment_id} is malformed or has an unsupported schema"
            ) from exc

    def load_record(self) -> ExperimentRecord:
        return self._read(EXPERIMENT_FILE, ExperimentRecord)

    def load_matrix(self) -> MatrixDocument:
        return self._read(MATRIX_FILE, MatrixDocument)

    def load_summary(self) -> dict | None:
        path = self.directory / SUMMARY_FILE
        if not path.is_file():
            return None
        import json

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ExperimentError(f"{SUMMARY_FILE} of {self.experiment_id} is unreadable") from exc

    # ---- write (atomic) -------------------------------------------------------------------
    def save_record(self, record: ExperimentRecord) -> None:
        write_json_atomic(self.directory / EXPERIMENT_FILE, record.model_dump(mode="json"))

    def save_matrix(self, matrix: MatrixDocument) -> None:
        write_json_atomic(self.directory / MATRIX_FILE, matrix.model_dump(mode="json"))

    def save_summary(self, summary: dict) -> None:
        write_json_atomic(self.directory / SUMMARY_FILE, summary)
