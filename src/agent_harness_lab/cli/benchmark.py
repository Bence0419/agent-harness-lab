from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from agent_harness_lab.benchmarks import BenchmarkError
from agent_harness_lab.benchmarks.git import GitHead, GitProbe
from agent_harness_lab.cli import common
from agent_harness_lab.cli.results import RawDirOpt
from agent_harness_lab.cli.tasks import TasksRepoOpt, fail, open_benchmark
from agent_harness_lab.config import resolve_path_setting
from agent_harness_lab.doctor.checks import HostProbe
from agent_harness_lab.evaluation.base import EvaluationContext
from agent_harness_lab.evaluation.service import load_subject
from agent_harness_lab.execution.executor import utc_now
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.experiments.errors import ExperimentError
from agent_harness_lab.experiments.loader import load_experiment, make_experiment_id
from agent_harness_lab.experiments.matrix import expand_matrix
from agent_harness_lab.experiments.planning import (
    OVERRIDE_DIRTY,
    OVERRIDE_VERSION,
    ExperimentPlan,
    PlanningDeps,
    group_failures,
    plan_experiment,
)
from agent_harness_lab.experiments.runner import Action, Orchestrator, RunnerDeps, reconcile
from agent_harness_lab.experiments.store import (
    AhlProvenance,
    ExperimentRecord,
    ExperimentStore,
    TaskRepoProvenance,
)
from agent_harness_lab.paths import default_experiments_dir, default_raw_dir
from agent_harness_lab.providers import (
    PROVIDERS_FILE_ENV,
    ProviderError,
    ProviderSpec,
    load_providers_file,
)
from agent_harness_lab.reporting.experiment import plan_to_json, render_plan, render_status
from agent_harness_lab.results.native_metrics import NativeTraceMetrics

SWEBENCH_PYTHON_ENV = "AHL_SWEBENCH_PYTHON"

ExperimentsDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--experiments-dir", help="Where experiment directories live (results/experiments)."
    ),
]


def _probe():
    """Host probe override point; tests patch this."""
    return HostProbe()


def _git() -> GitProbe:
    """Git probe override point; tests patch this."""
    return GitHead()


def _load_providers(cli_value: Path | None) -> dict[str, ProviderSpec]:
    resolved = resolve_path_setting(cli_value, PROVIDERS_FILE_ENV)
    return load_providers_file(resolved.path) if resolved.path is not None else {}


def benchmark_command(
    experiment_file: Annotated[Path, typer.Argument(help="Experiment definition (YAML).")],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate, expand and pre-flight the matrix; run/evaluate nothing."
        ),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Continue the experiment with this id (same file)."),
    ] = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="With --resume: give failed/infrastructure cells a NEW attempt (no-verdict "
            "cells are re-evaluated instead). Never overwrites earlier attempts.",
        ),
    ] = False,
    tasks_repo: TasksRepoOpt = None,
    providers_file: Annotated[
        Path | None,
        typer.Option("--providers-file", help=f"Providers file (else ${PROVIDERS_FILE_ENV})."),
    ] = None,
    swebench_python: Annotated[
        Path | None,
        typer.Option(
            "--swebench-python",
            help=f"Python of the evaluator environment (else ${SWEBENCH_PYTHON_ENV}).",
        ),
    ] = None,
    raw_dir: RawDirOpt = None,
    experiments_dir: ExperimentsDirOpt = None,
    allow_dirty_task_repo: Annotated[
        bool,
        typer.Option(
            f"--{OVERRIDE_DIRTY}", help="DEVELOPMENT ONLY: accept a dirty task repository."
        ),
    ] = False,
    allow_harness_version_mismatch: Annotated[
        bool,
        typer.Option(
            f"--{OVERRIDE_VERSION}",
            help="DEVELOPMENT ONLY: accept a harness version that differs from the repo pin.",
        ),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON only (--dry-run).")
    ] = False,
) -> None:
    """Expand an experiment into a run matrix, pre-flight ALL of it, then run it sequentially.

    Execution requires AHL_ENABLE_REAL_EXECUTION=1. --dry-run needs neither the gate nor any
    secret and executes nothing.
    """
    err = Console(stderr=True, highlight=False)
    env = common.environ()
    if dry_run and (resume or retry_failed):
        err.print("[red]error:[/] --dry-run cannot be combined with --resume/--retry-failed")
        raise typer.Exit(2)
    if retry_failed and not resume:
        err.print("[red]error:[/] --retry-failed only makes sense with --resume")
        raise typer.Exit(2)
    if json_output and not dry_run:
        err.print("[red]error:[/] --json is only available with --dry-run")
        raise typer.Exit(2)
    if not dry_run and not real_execution_enabled(env):
        err.print(
            f"[red]error:[/] real execution is disabled: set {GATE_ENV}=1 (a temporary "
            "pre-release safety gate), or use --dry-run. No run was started."
        )
        raise typer.Exit(2)

    try:
        loaded = load_experiment(experiment_file)
    except ExperimentError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(2) from None
    exp = loaded.normalized
    overrides = [
        name
        for name, on in (
            (OVERRIDE_DIRTY, allow_dirty_task_repo),
            (OVERRIDE_VERSION, allow_harness_version_mismatch),
        )
        if on
    ]

    git = _git()
    try:
        providers = _load_providers(providers_file)
        benchmark = open_benchmark(exp.benchmark.name, tasks_repo, git)
    except ProviderError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from None
    except BenchmarkError as exc:
        raise fail(err, exc) from None

    raw = raw_dir or default_raw_dir()
    experiments = experiments_dir or default_experiments_dir()
    docker = common.docker_client()
    python = resolve_path_setting(swebench_python, SWEBENCH_PYTHON_ENV).path
    now = utc_now()

    # ---- fresh launch vs resume --------------------------------------------------------------
    store: ExperimentStore | None = None
    record: ExperimentRecord | None = None
    actions: dict[str, Action] = {}
    try:
        if resume:
            store = ExperimentStore.open(experiments, resume)
            record = store.load_record()
            if record.definition_sha256 != loaded.sha256:
                raise ExperimentError(
                    "the experiment file does not match the experiment being resumed "
                    f"(file {loaded.sha256[:12]}, recorded {record.definition_sha256[:12]}); "
                    "a changed definition is a different experiment"
                )
            matrix = store.load_matrix()
            raw = Path(record.raw_dir)
            actions = reconcile(store, record, matrix, raw, retry_failed=retry_failed)
            experiment_id = record.experiment_id
            by_id = {c.cell.cell_id: c for c in matrix.cells}
            cells = [
                (by_id[cid].cell, a.attempt) for cid, a in actions.items() if a.kind == "execute"
            ]
            eval_only = [by_id[cid].cell.task for cid, a in actions.items() if a.kind == "evaluate"]
            eval_tasks = sorted({c.task for c, _ in cells} | set(eval_only))
        else:
            experiment_id = make_experiment_id(exp.experiment.name, now, loaded.sha256)
            cells = [(c, 1) for c in expand_matrix(exp)]
            eval_only, eval_tasks = [], list(exp.tasks)
    except ExperimentError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(2) from None

    if resume and not actions:
        console = Console(highlight=False)
        console.print(f"Nothing to do: every cell of {experiment_id} is already complete.")
        assert store is not None
        render_status(record, store.load_matrix(), store.load_summary(), console)  # type: ignore[arg-type]
        raise typer.Exit(0)

    plan = plan_experiment(
        exp,
        definition_sha256=loaded.sha256,
        experiment_id=experiment_id,
        cells=cells,
        evaluate_only_tasks=eval_only,
        evaluation_task_ids=eval_tasks,
        deps=PlanningDeps(
            benchmark=benchmark,
            providers=providers,
            probe=_probe(),
            git=git,
            docker=docker,
            host=common.host_environment(),
            environ=env,
            runner=common.command_runner(),
            swebench_python=python,
        ),
        require_gate=not dry_run,
        overrides=overrides,
    )

    if dry_run:
        if json_output:
            typer.echo(json.dumps(plan_to_json(plan, dry_run=True), indent=2))
        else:
            render_plan(plan, Console(highlight=False, width=None), dry_run=True)
        raise typer.Exit(0 if plan.ready else 1)

    if not plan.ready:
        console = Console(stderr=True, highlight=False)
        console.print(
            f"[bold red]pre-flight failed: zero runs were started "
            f"({len(plan.failures)} failure(s)):[/]"
        )
        for check, detail, cells_ in group_failures(plan.failures):
            where = f" ({len(cells_)} cell(s))" if cells_ else " (experiment-wide)"
            console.print(f"  [red]-[/] {escape(check)}: {escape(detail)}{where}")
        raise typer.Exit(1)

    # ---- everything validated: only now create artifacts and start ----------------------------
    out = Console(highlight=False)
    for note in plan.notes:
        out.print(f"[bold yellow]note:[/] {escape(note)}")
    if store is None:
        assert plan.repository is not None
        record = ExperimentRecord(
            experiment_id=experiment_id,
            definition_sha256=loaded.sha256,
            started_at=now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            raw_dir=str(raw),
            overrides=[f"--{o}" for o in overrides],
            ahl=_ahl_provenance(plan),
            task_repository=TaskRepoProvenance(
                commit=plan.repository.commit,
                dirty=plan.repository.dirty,
                pinned_commit=exp.benchmark.task_repository_commit,
            ),
            definition=exp,
        )
        store = ExperimentStore.create(experiments, record, loaded.raw_bytes, [c for c, _ in cells])
        matrix = store.load_matrix()
        actions = {c.cell_id: Action("execute", 1) for c, _ in cells}
    assert record is not None
    out.print(f"Experiment {experiment_id}: {len(actions)} cell(s) to process, sequentially")

    def make_ctx(run_dir: Path, force: bool) -> EvaluationContext:
        return EvaluationContext(
            subject=load_subject(run_dir),
            benchmark=benchmark,
            repository=benchmark.repository_info(),
            python=python,
            docker=docker,
            runner=common.command_runner(),
            environ=env,
            output_dir=run_dir,
            force=force,
        )

    orchestrator = Orchestrator(
        store=store,
        record=record,
        matrix=matrix,
        deps=RunnerDeps(
            docker=docker,
            environ=env,
            host=common.host_environment(),
            raw_dir=raw,
            benchmark=benchmark,
            make_evaluation_context=make_ctx if exp.evaluation.enabled else None,
            metrics=NativeTraceMetrics(),
            sleep=common.sleep,
            progress=out.print,
        ),
        plans={cp.cell.cell_id: cp for cp in plan.cells},
        actions=actions,
    )
    summary = orchestrator.run()
    render_status(orchestrator.record, matrix, summary, out)
    out.print(f"Artifacts:  {store.directory}")
    raise typer.Exit(0 if summary["status"] == "complete" else 1)


def _ahl_provenance(plan: ExperimentPlan) -> AhlProvenance:
    for c in plan.cells:
        if c.result is not None:
            return AhlProvenance(commit=c.result.plan.ahl.commit, dirty=c.result.plan.ahl.dirty)
    return AhlProvenance()


# ---- ahl experiments ----------------------------------------------------------------------
experiments_app = typer.Typer(help="Inspect persisted experiments (offline).", no_args_is_help=True)


@experiments_app.command("show")
def show_command(
    experiment_id: Annotated[str, typer.Argument(help="Experiment id.")],
    experiments_dir: ExperimentsDirOpt = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON only.")] = False,
) -> None:
    """Show the persisted matrix states and summary. Needs no Docker, API key or network."""
    err = Console(stderr=True, highlight=False)
    try:
        store = ExperimentStore.open(experiments_dir or default_experiments_dir(), experiment_id)
        record, matrix, summary = store.load_record(), store.load_matrix(), store.load_summary()
    except ExperimentError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from None
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "experiment": record.model_dump(mode="json"),
                    "matrix": matrix.model_dump(mode="json"),
                    "summary": summary,
                },
                indent=2,
            )
        )
        return
    render_status(record, matrix, summary, Console(highlight=False))


@experiments_app.command("list")
def list_command(experiments_dir: ExperimentsDirOpt = None) -> None:
    """List experiment ids in the experiments directory."""
    root = experiments_dir or default_experiments_dir()
    ids = (
        sorted(p.name for p in root.iterdir() if (p / "experiment.json").is_file())
        if root.is_dir()
        else []
    )
    for i in ids:
        typer.echo(i)
    if not ids:
        typer.echo(f"no experiments in {root}", err=True)
