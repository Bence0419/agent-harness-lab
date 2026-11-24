"""Sequential experiment orchestration over the existing run and evaluation layers.

Cell lifecycle::

    planned --run--> running --> completed (evaluation disabled)
                            \\--> agent_completed --evaluate--> passed | failed | no_verdict
                            \\--> harness_failed | timed_out   (terminal; NOT evaluated)
                            \\--> infra_failure               (terminal; also: interrupted)

Evaluation rule: only a cleanly COMPLETED agent run (harness exit 0, patch captured) is
evaluated automatically. After a non-zero harness exit or a timeout the patch (if any) is kept
as evidence, but the cell is reported as harness_failed / timed_out: it is an agent outcome,
not a benchmark verdict, and is excluded from correctness denominators. A person can still score
the preserved patch with ``ahl evaluate <run-id>``.

Cells run strictly one after another (see docs/experiments.md for why).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_harness_lab.evaluation.base import EvaluationContext
from agent_harness_lab.evaluation.registry import get_evaluator
from agent_harness_lab.evaluation.service import run_evaluation
from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.executor import RunRequest, execute_run, utc_now
from agent_harness_lab.execution.models import ExecutionDisabled
from agent_harness_lab.execution.preflight import CheckStatus
from agent_harness_lab.experiments.planning import CellPlan
from agent_harness_lab.experiments.reconcile import (
    Derived,
    derive_state,
    state_from_classification,
)
from agent_harness_lab.experiments.store import (
    RETRYABLE,
    TERMINAL,
    AttemptRecord,
    CellRecord,
    CellState,
    ExperimentRecord,
    ExperimentStore,
    MatrixDocument,
)
from agent_harness_lab.experiments.summary import build_summary
from agent_harness_lab.harnesses import get_harness
from agent_harness_lab.harnesses.provisioning import HostEnvironment
from agent_harness_lab.results.service import TraceMetricsSource
from agent_harness_lab.runplan.prompt import build_agent_prompt


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Action:
    kind: str  # "execute" | "evaluate"
    attempt: int  # the attempt to create (execute) or to evaluate
    force: bool = False  # evaluate: archive an existing/unusable evaluation first


@dataclass
class RunnerDeps:
    docker: DockerClient
    environ: Mapping[str, str]
    host: HostEnvironment
    raw_dir: Path
    benchmark: object  # BenchmarkAdapter (get_task)
    make_evaluation_context: Callable[[Path, bool], EvaluationContext] | None
    metrics: TraceMetricsSource | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = utc_now
    progress: Callable[[str], None] = lambda message: None


# ---------------------------------------------------------------------------------------------
# Reconciliation: what must be done for each cell?
# ---------------------------------------------------------------------------------------------
def reconcile(
    store: ExperimentStore,
    record: ExperimentRecord,
    matrix: MatrixDocument,
    raw_dir: Path,
    *,
    retry_failed: bool = False,
) -> dict[str, Action]:
    """Bring matrix.json in line with the canonical records and decide what is left to do.

    Completed and evaluated cells are left alone. Failed/infrastructure cells are NOT rerun
    unless ``retry_failed`` (which creates a NEW attempt, or re-evaluates a no-verdict cell:
    a flaky evaluator must not cost another agent run)."""
    actions: dict[str, Action] = {}
    evaluation_enabled = record.definition.evaluation.enabled
    for cr in matrix.cells:
        derived, run_dir, doc, ev = derive_state(
            raw_dir, cr, record.experiment_id, evaluation_enabled=evaluation_enabled
        )
        _apply_derived(cr, derived, run_dir, raw_dir, doc, ev)
        n = len(cr.attempts)
        if cr.state is CellState.PLANNED:
            actions[cr.cell.cell_id] = Action("execute", 1)
        elif cr.state is CellState.AGENT_COMPLETED:
            actions[cr.cell.cell_id] = Action("evaluate", n, derived.force_evaluation)
        elif retry_failed and cr.state in RETRYABLE:
            if cr.state is CellState.NO_VERDICT:
                actions[cr.cell.cell_id] = Action("evaluate", n, force=True)
            else:
                actions[cr.cell.cell_id] = Action("execute", n + 1)
    store.save_matrix(matrix)
    return actions


def _apply_derived(cr, derived: Derived, run_dir, raw_dir, doc, ev) -> None:
    cr.state = derived.state
    attempt = cr.current
    if attempt is None:
        return
    if run_dir is not None:
        attempt.run_path = (
            str(run_dir.relative_to(raw_dir)) if run_dir.is_relative_to(raw_dir) else str(run_dir)
        )
        attempt.run_id = run_dir.name
    if doc is not None:
        attempt.execution_state = doc.state.value
        attempt.failed_stage = doc.failed_stage
        attempt.exit_code = doc.exit_code
        attempt.agent_duration_seconds = doc.duration_seconds
        attempt.finished_at = doc.finished_at
        attempt.error = doc.error
    elif derived.state is CellState.INFRA_FAILURE:
        attempt.execution_state = attempt.execution_state or "unknown"
        attempt.error = attempt.error or "interrupted before a complete run record was written"
    if ev is not None and not derived.needs_evaluation:
        attempt.evaluation_state = ev.state.value
        attempt.classification = ev.classification.value
        attempt.resolved = ev.resolved
        attempt.evaluation_reason = ev.reason
        attempt.evaluation_duration_seconds = ev.duration_seconds


# ---------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------
@dataclass
class Orchestrator:
    store: ExperimentStore
    record: ExperimentRecord
    matrix: MatrixDocument
    deps: RunnerDeps
    plans: dict[str, CellPlan]  # cells that need an agent run
    actions: dict[str, Action]
    processed: list[str] = field(default_factory=list)

    def run(self) -> dict:
        self._summary("running")
        try:
            for cr in self.matrix.cells:
                action = self.actions.get(cr.cell.cell_id)
                if action is None:
                    continue
                self.deps.progress(f"{cr.cell.cell_id}: {action.kind}")
                if action.kind == "execute":
                    self._execute(cr, action)
                else:
                    self._evaluate(cr, action)
                self.processed.append(cr.cell.cell_id)
                self._summary("running")
        except BaseException:
            self._summary("interrupted")
            raise
        done = all(c.state in TERMINAL for c in self.matrix.cells)
        summary = self._summary("complete" if done else "incomplete")
        self.record = self.record.model_copy(update={"finished_at": _iso(self.deps.now())})
        self.store.save_record(self.record)
        return summary

    # -- one agent run -----------------------------------------------------------------------
    def _execute(self, cr: CellRecord, action: Action) -> None:
        cp = self.plans[cr.cell.cell_id]
        plan = cp.result.plan  # type: ignore[union-attr]
        assert plan.experiment is not None and plan.experiment.attempt == action.attempt
        attempt = AttemptRecord(attempt=action.attempt, started_at=_iso(self.deps.now()))
        cr.attempts.append(attempt)  # earlier attempts stay exactly as they were
        cr.state = CellState.RUNNING
        self.store.save_matrix(self.matrix)

        adapter = get_harness(plan.harness.name)
        task = self.deps.benchmark.get_task(plan.task.instance_id)  # type: ignore[attr-defined]
        request = RunRequest(
            plan=plan,
            prompt=build_agent_prompt(task.problem_statement),
            provisioner=lambda staged: adapter.provisioning(
                plan.harness.executable_path, self.deps.host, staged
            ),
            raw_dir=self.deps.raw_dir,
            plan_warnings=list(cp.warnings),
        )
        try:
            doc = execute_run(
                request,
                docker=self.deps.docker,
                environ=self.deps.environ,
                sleep=self.deps.sleep,
                now=self.deps.now,
            )
        except ExecutionDisabled:
            cr.attempts.pop()  # nothing ran: do not leave a phantom attempt
            cr.state = CellState.PLANNED if action.attempt == 1 else cr.state
            self.store.save_matrix(self.matrix)
            raise
        # KeyboardInterrupt propagates: the run directory holds the (finalized) evidence and
        # the next resume reconciles it from the canonical records.
        attempt.run_id = doc.run_id
        attempt.run_path = str(
            (self.deps.raw_dir / "swebench" / plan.task.instance_id / doc.run_id).relative_to(
                self.deps.raw_dir
            )
        )
        self._reconcile_cell(cr)
        if cr.state is CellState.AGENT_COMPLETED:
            self.store.save_matrix(self.matrix)
            self._evaluate(cr, Action("evaluate", action.attempt))

    def _reconcile_cell(self, cr: CellRecord) -> None:
        derived, run_dir, doc, ev = derive_state(
            self.deps.raw_dir,
            cr,
            self.record.experiment_id,
            evaluation_enabled=self.record.definition.evaluation.enabled,
        )
        _apply_derived(cr, derived, run_dir, self.deps.raw_dir, doc, ev)
        self.store.save_matrix(self.matrix)

    # -- one evaluation ---------------------------------------------------------------------
    def _evaluate(self, cr: CellRecord, action: Action) -> None:
        attempt = cr.current
        assert attempt is not None and attempt.run_path is not None
        run_dir = self.deps.raw_dir / attempt.run_path
        factory = self.deps.make_evaluation_context
        if factory is None:
            attempt.error = "evaluation is not configured"
            self.store.save_matrix(self.matrix)
            return
        try:
            adapter = get_evaluator(self.record.definition.benchmark.name)
            ctx = factory(run_dir, action.force)
            pre, doc = run_evaluation(adapter, ctx)
        except KeyboardInterrupt:
            attempt.error = "evaluation interrupted; resume re-evaluates this cell"
            cr.state = CellState.AGENT_COMPLETED
            self.store.save_matrix(self.matrix)
            raise
        except Exception as exc:  # never lose the matrix because one evaluation misbehaved
            attempt.error = f"evaluation did not run: {type(exc).__name__}: {exc}"
            cr.state = CellState.AGENT_COMPLETED  # still awaiting evaluation
            self.store.save_matrix(self.matrix)
            return
        if doc is None:
            failed = "; ".join(
                f"{c.name}: {c.detail}" for c in pre.checks if c.status is CheckStatus.ERROR
            )
            attempt.error = f"evaluation pre-flight failed: {failed}"
            cr.state = CellState.AGENT_COMPLETED
            self.store.save_matrix(self.matrix)
            return
        attempt.error = None
        attempt.evaluation_state = doc.state.value
        attempt.classification = doc.classification.value
        attempt.resolved = doc.resolved
        attempt.evaluation_reason = doc.reason
        attempt.evaluation_duration_seconds = doc.duration_seconds
        cr.state = state_from_classification(doc.classification)
        self.store.save_matrix(self.matrix)

    def _summary(self, status: str) -> dict:
        summary = build_summary(
            self.record,
            self.matrix,
            self.deps.raw_dir,
            metrics=self.deps.metrics,
            status=status,
            now=_iso(self.deps.now()),
        )
        self.store.save_summary(summary)
        return summary
