"""Derive a cell's true state from CANONICAL records (run.json, evaluation.json).

matrix.json is a cache of what the orchestrator last saw. On resume it is never trusted on its
own: completion is established by validating the run's ``run.json`` (schema, terminal state, and
that its experiment provenance names THIS experiment, cell and attempt) and, for evaluated
cells, a valid ``evaluation.json``. File names alone prove nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from agent_harness_lab.artifacts.run_record import RUN_FILE, RunDocument
from agent_harness_lab.evaluation.models import EVALUATION_FILE, Classification, EvaluationDocument
from agent_harness_lab.evaluation.swebench_multilingual import WORK_DIRNAME
from agent_harness_lab.execution.executor import SWEBENCH_RAW_SUBDIR
from agent_harness_lab.execution.models import ExecutionState
from agent_harness_lab.experiments.store import AttemptRecord, CellRecord, CellState


@dataclass
class Derived:
    state: CellState
    needs_evaluation: bool = False
    force_evaluation: bool = False  # an unusable previous evaluation.json must be archived first


def load_run_doc(run_dir: Path) -> RunDocument | None:
    try:
        return RunDocument.model_validate_json((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None


def load_evaluation_doc(run_dir: Path) -> EvaluationDocument | None:
    try:
        return EvaluationDocument.model_validate_json(
            (run_dir / EVALUATION_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return None


def find_attempt_run(
    raw_dir: Path, task: str, experiment_id: str, cell_id: str, attempt: int
) -> Path | None:
    """The run directory whose canonical provenance names this experiment cell attempt.
    Needed when the orchestrator died before it could record the run id."""
    task_dir = raw_dir / SWEBENCH_RAW_SUBDIR / task
    if not task_dir.is_dir():
        return None
    for run_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        doc = load_run_doc(run_dir)
        exp = doc.plan.experiment if doc else None
        if (
            exp
            and exp.experiment_id == experiment_id
            and exp.matrix_cell_id == cell_id
            and exp.attempt == attempt
        ):
            return run_dir
    return None


def attempt_run_dir(
    raw_dir: Path, cr: CellRecord, attempt: AttemptRecord, experiment_id: str
) -> Path | None:
    if attempt.run_path:
        candidate = raw_dir / attempt.run_path
        if candidate.is_dir():
            return candidate
    return find_attempt_run(raw_dir, cr.cell.task, experiment_id, cr.cell.cell_id, attempt.attempt)


def derive_state(
    raw_dir: Path,
    cr: CellRecord,
    experiment_id: str,
    *,
    evaluation_enabled: bool,
) -> tuple[Derived, Path | None, RunDocument | None, EvaluationDocument | None]:
    attempt = cr.current
    if attempt is None:
        return Derived(CellState.PLANNED), None, None, None
    run_dir = attempt_run_dir(raw_dir, cr, attempt, experiment_id)
    doc = load_run_doc(run_dir) if run_dir else None
    if run_dir is None or doc is None:
        return Derived(CellState.INFRA_FAILURE), run_dir, None, None
    exp = doc.plan.experiment
    if (
        exp is None
        or exp.experiment_id != experiment_id
        or exp.matrix_cell_id != cr.cell.cell_id
        or exp.attempt != attempt.attempt
    ):
        return Derived(CellState.INFRA_FAILURE), run_dir, None, None

    if doc.state in (ExecutionState.PREPARED, ExecutionState.RUNNING):
        return Derived(CellState.INFRA_FAILURE), run_dir, doc, None  # interrupted mid-run
    if doc.state is ExecutionState.FAILED:
        return Derived(CellState.INFRA_FAILURE), run_dir, doc, None
    if doc.state is ExecutionState.HARNESS_FAILED:
        return Derived(CellState.HARNESS_FAILED), run_dir, doc, None
    if doc.state is ExecutionState.TIMED_OUT:
        return Derived(CellState.TIMED_OUT), run_dir, doc, None

    # COMPLETED
    if not evaluation_enabled:
        return Derived(CellState.COMPLETED), run_dir, doc, None
    ev = load_evaluation_doc(run_dir)
    if ev is None:
        # an invalid evaluation.json, or the leftovers of an evaluation that died midway:
        # archive them (never delete) and evaluate again
        unusable = (run_dir / EVALUATION_FILE).exists() or (run_dir / WORK_DIRNAME).exists()
        return Derived(CellState.AGENT_COMPLETED, True, unusable), run_dir, doc, None
    if ev.interrupted or ev.run_id != doc.run_id:
        return Derived(CellState.AGENT_COMPLETED, True, True), run_dir, doc, ev
    return Derived(state_from_classification(ev.classification)), run_dir, doc, ev


def state_from_classification(classification: Classification) -> CellState:
    if classification is Classification.BENCHMARK_PASS:
        return CellState.PASSED
    if classification is Classification.BENCHMARK_FAIL:
        return CellState.FAILED
    return CellState.NO_VERDICT  # evaluation_failure / ambiguous: never a pass or a fail
