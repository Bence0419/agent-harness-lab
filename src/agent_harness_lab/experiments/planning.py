"""Experiment-wide planning and pre-flight.

Every cell that still needs an agent run gets a RunPlan and the SAME static pre-flight a single
``ahl run --preflight`` performs, plus the experiment-level rules (pinned task repository,
harness versions equal to the repository pins). All failures are collected and reported
together; if there is any, the caller starts zero runs. Read-only: no container, network or
agent is ever created here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent_harness_lab.benchmarks import BenchmarkAdapter, BenchmarkError
from agent_harness_lab.benchmarks.base import RepositoryInfo, verify_expected_commit
from agent_harness_lab.evaluation.registry import get_evaluator
from agent_harness_lab.execution.docker import CommandRunner, DockerClient
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.execution.preflight import CheckStatus, PreflightReport, run_preflight
from agent_harness_lab.experiments.errors import ExperimentError
from agent_harness_lab.experiments.matrix import MatrixCell
from agent_harness_lab.experiments.models import NormalizedExperiment
from agent_harness_lab.harnesses import get_harness
from agent_harness_lab.harnesses.base import HarnessError
from agent_harness_lab.harnesses.provisioning import HostEnvironment
from agent_harness_lab.providers import ProviderSpec
from agent_harness_lab.runplan.builder import PlanProbe, build_run_plan
from agent_harness_lab.runplan.models import ExperimentRef, PlanResult

OVERRIDE_DIRTY = "allow-dirty-task-repo"
OVERRIDE_VERSION = "allow-harness-version-mismatch"
GATE_CHECK = "real-execution gate"


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str
    cell_id: str | None = None  # None = experiment-wide


@dataclass
class CellPlan:
    cell: MatrixCell
    attempt: int
    result: PlanResult | None = None
    preflight: PreflightReport | None = None
    failures: list[Failure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExperimentPlan:
    experiment: NormalizedExperiment
    definition_sha256: str
    experiment_id: str
    repository: RepositoryInfo | None
    cells: list[CellPlan]
    global_failures: list[Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # prominent, e.g. active overrides
    overrides: list[str] = field(default_factory=list)
    evaluation_checks: list[tuple[str, str, str]] = field(default_factory=list)
    # (name, status, detail)
    gate_enabled: bool = False

    @property
    def failures(self) -> list[Failure]:
        return [*self.global_failures, *(f for c in self.cells for f in c.failures)]

    @property
    def ready(self) -> bool:
        return not self.failures


@dataclass
class PlanningDeps:
    benchmark: BenchmarkAdapter
    providers: Mapping[str, ProviderSpec]
    probe: PlanProbe
    git: object  # GitProbe
    docker: DockerClient
    host: HostEnvironment
    environ: Mapping[str, str]
    runner: CommandRunner
    swebench_python: object | None  # Path | None


def _check_repository(
    exp: NormalizedExperiment, info: RepositoryInfo, overrides: Sequence[str]
) -> tuple[list[Failure], list[str]]:
    failures: list[Failure] = []
    notes: list[str] = []
    pin = exp.benchmark.task_repository_commit
    if pin is None:
        notes.append(
            "task_repository_commit is not pinned in the experiment: the result is only as "
            f"reproducible as the checkout (currently {info.commit or 'unknown'})"
        )
    else:
        try:
            verify_expected_commit(info, pin)
        except BenchmarkError as exc:
            failures.append(Failure("task repository pin", str(exc)))
    if info.dirty is not False:
        what = "has uncommitted changes" if info.dirty else "dirty state cannot be determined"
        if OVERRIDE_DIRTY in overrides:
            notes.append(f"DEVELOPMENT OVERRIDE --{OVERRIDE_DIRTY}: the task repository {what}")
        else:
            failures.append(
                Failure(
                    "task repository state",
                    f"the task repository {what}; a reproducible benchmark needs a clean "
                    f"checkout (development only: --{OVERRIDE_DIRTY})",
                )
            )
    return failures, notes


def plan_experiment(
    exp: NormalizedExperiment,
    *,
    definition_sha256: str,
    experiment_id: str,
    cells: Sequence[tuple[MatrixCell, int]],  # (cell, attempt number) that need an agent run
    evaluate_only_tasks: Sequence[str] = (),  # tasks whose cells only need evaluation
    deps: PlanningDeps,
    require_gate: bool,
    overrides: Sequence[str] = (),
    evaluation_task_ids: Sequence[str] | None = None,
) -> ExperimentPlan:
    plan = ExperimentPlan(
        experiment=exp,
        definition_sha256=definition_sha256,
        experiment_id=experiment_id,
        repository=None,
        cells=[CellPlan(cell=c, attempt=a) for c, a in cells],
        overrides=list(overrides),
        gate_enabled=real_execution_enabled(deps.environ),
    )
    if require_gate and not plan.gate_enabled:
        plan.global_failures.append(
            Failure(GATE_CHECK, f"{GATE_ENV}=1 is not set (required to execute real runs)")
        )

    info = deps.benchmark.repository_info()
    plan.repository = info
    failures, notes = _check_repository(exp, info, overrides)
    plan.global_failures.extend(failures)
    plan.notes.extend(notes)

    # ---- tasks and harnesses resolve before anything else ---------------------------------
    unresolved_tasks: set[str] = set()
    for task_id in sorted({c.cell.task for c in plan.cells}):
        try:
            deps.benchmark.get_task(task_id)
        except BenchmarkError as exc:
            unresolved_tasks.add(task_id)
            plan.global_failures.append(Failure("task", str(exc)))
    bad_harnesses: set[str] = set()
    for name in sorted({c.cell.harness for c in plan.cells}):
        try:
            get_harness(name)
        except HarnessError as exc:
            bad_harnesses.add(name)
            plan.global_failures.append(Failure("harness", str(exc)))

    # ---- one RunPlan + the single-run static pre-flight per cell ---------------------------
    for cp in plan.cells:
        c = cp.cell
        if c.task in unresolved_tasks or c.harness in bad_harnesses:
            continue
        _plan_cell(cp, exp, definition_sha256, experiment_id, deps, overrides, require_gate)

    # ---- the evaluator must be able to score these tasks (checked BEFORE any run) -----------
    if exp.evaluation.enabled:
        ids = list(evaluation_task_ids if evaluation_task_ids is not None else exp.tasks)
        ids = sorted(set(ids) | set(evaluate_only_tasks))
        try:
            checks = get_evaluator(exp.benchmark.name).readiness(
                benchmark=deps.benchmark,
                task_ids=ids,
                python=deps.swebench_python,  # type: ignore[arg-type]
                runner=deps.runner,
                environ=deps.environ,
            )
        except Exception as exc:  # unsupported benchmark etc.: a pre-flight failure, not a crash
            checks = []
            plan.global_failures.append(Failure("evaluator", str(exc)))
        for chk in checks:
            plan.evaluation_checks.append((chk.name, chk.status.value, chk.detail))
            if chk.status is CheckStatus.ERROR:
                plan.global_failures.append(Failure(f"evaluation: {chk.name}", chk.detail))
    return plan


def _plan_cell(
    cp: CellPlan,
    exp: NormalizedExperiment,
    definition_sha256: str,
    experiment_id: str,
    deps: PlanningDeps,
    overrides: Sequence[str],
    require_gate: bool,
) -> None:
    c = cp.cell
    ref = ExperimentRef(
        experiment_id=experiment_id,
        definition_sha256=definition_sha256,
        matrix_cell_id=c.cell_id,
        repetition=c.repetition,
        attempt=cp.attempt,
    )
    try:
        result = build_run_plan(
            benchmark=deps.benchmark,
            task_id=c.task,
            harness_name=c.harness,
            model_ref=c.model_ref,
            reasoning=c.reasoning,
            timeout_seconds=exp.execution.timeout_seconds,
            network_mode=exp.execution.network,
            providers=deps.providers,
            probe=deps.probe,
            git=deps.git,  # type: ignore[arg-type]
            purpose="experiment",
            experiment=ref,
        )
    except (BenchmarkError, ExperimentError) as exc:
        cp.failures.append(Failure("plan", str(exc), c.cell_id))
        return
    cp.result = result
    cp.warnings = list(result.warnings)
    plan = result.plan

    h = plan.harness
    if h.expected_version is not None and h.expected_version != h.version:
        msg = f"{h.display_name} {h.version} differs from the repository pin {h.expected_version}"
        if OVERRIDE_VERSION in overrides:
            cp.warnings.append(f"DEVELOPMENT OVERRIDE --{OVERRIDE_VERSION}: {msg}")
        else:
            cp.failures.append(
                Failure(
                    "harness version",
                    f"{msg} (development only: --{OVERRIDE_VERSION})",
                    c.cell_id,
                )
            )

    adapter = get_harness(plan.harness.name)
    report = run_preflight(
        plan,
        provisioner=lambda staged: adapter.provisioning(
            plan.harness.executable_path, deps.host, staged
        ),
        docker=deps.docker,
        path_exists=lambda p: deps.host.is_file(p) or deps.host.is_dir(p),
        environ=deps.environ,
        plan_warnings=result.warnings,
    )
    cp.preflight = report
    for chk in report.checks:
        if chk.status is not CheckStatus.ERROR:
            continue
        if chk.name == GATE_CHECK and not require_gate:
            continue  # dry-run: the gate is reported once, experiment-wide, never per cell
        cp.failures.append(Failure(chk.name, chk.detail, c.cell_id))


def group_failures(failures: Sequence[Failure]) -> list[tuple[str, str, list[str]]]:
    """Identical (check, detail) pairs collapse into one line listing the affected cells."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for f in failures:
        grouped.setdefault((f.check, f.detail), [])
        if f.cell_id:
            grouped[(f.check, f.detail)].append(f.cell_id)
    return [(check, detail, cells) for (check, detail), cells in grouped.items()]
