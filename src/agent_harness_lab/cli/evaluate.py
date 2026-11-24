from __future__ import annotations

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
from agent_harness_lab.evaluation.base import (
    EvaluationContext,
    EvaluationError,
    EvaluationPreflight,
)
from agent_harness_lab.evaluation.models import Classification
from agent_harness_lab.evaluation.registry import get_evaluator
from agent_harness_lab.evaluation.service import load_subject, resolve_run_dir, run_evaluation
from agent_harness_lab.evaluation.service import preflight as run_preflight_checks
from agent_harness_lab.execution.preflight import CheckStatus
from agent_harness_lab.paths import default_raw_dir
from agent_harness_lab.reporting.evaluation import render_evaluation
from agent_harness_lab.reporting.preflight import render_preflight

SWEBENCH_PYTHON_ENV = "AHL_SWEBENCH_PYTHON"


def _git() -> GitProbe:
    """Git probe override point; tests patch this."""
    return GitHead()


def evaluate_command(
    run: Annotated[str, typer.Argument(help="Run id (or a run directory path).")],
    raw_dir: RawDirOpt = None,
    tasks_repo: TasksRepoOpt = None,
    swebench_python: Annotated[
        Path | None,
        typer.Option(
            "--swebench-python",
            help="Python interpreter of the environment that has the evaluator installed "
            f"(else ${SWEBENCH_PYTHON_ENV}).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Write evaluation artifacts here instead of the run directory "
            "(e.g. to re-check a historical run without touching it).",
        ),
    ] = None,
    preflight: Annotated[
        bool, typer.Option("--preflight", help="Check everything; evaluate nothing.")
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Re-evaluate: the previous evaluation is archived, not deleted."
        ),
    ] = False,
    keep_evaluator_image: Annotated[
        bool,
        typer.Option(
            "--keep-evaluator-image",
            help="Diagnostic: keep the image the evaluator built (multi-GB) under an "
            "ahl-eval-image/ reference. By default it is removed after verification.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON only.")] = False,
) -> None:
    """Score a run's persisted model.patch with the benchmark's official evaluator.

    Needs no API key and no execution gate: no model is called and no agent runs.
    """
    err = Console(stderr=True, highlight=False)
    store = raw_dir or default_raw_dir()

    def refuse(message: str, code: int = 2) -> typer.Exit:
        if preflight:
            report = EvaluationPreflight()
            report.add("evaluation request", CheckStatus.ERROR, message)
            _emit(report, json_output)
            return typer.Exit(1)
        err.print(f"[red]error:[/] {escape(message)}")
        return typer.Exit(code)

    try:
        run_dir = resolve_run_dir(store, run)
        subject = load_subject(run_dir)
        adapter = get_evaluator(subject.benchmark)
        benchmark = open_benchmark(subject.benchmark, tasks_repo, _git())
        repository = benchmark.repository_info()
    except (EvaluationError, BenchmarkError) as exc:
        if isinstance(exc, BenchmarkError) and not preflight:
            raise fail(err, exc) from None
        raise refuse(str(exc)) from None

    python = resolve_path_setting(swebench_python, SWEBENCH_PYTHON_ENV).path
    ctx = EvaluationContext(
        subject=subject,
        benchmark=benchmark,
        repository=repository,
        python=python,
        docker=common.docker_client(),
        runner=common.command_runner(),
        environ=common.environ(),
        output_dir=output_dir or run_dir,
        force=force,
        keep_evaluator_image=keep_evaluator_image,
    )
    if preflight:
        report = run_preflight_checks(adapter, ctx)
        _emit(report, json_output)
        raise typer.Exit(0 if report.ready else 1)

    try:
        report, doc = run_evaluation(adapter, ctx)
    except EvaluationError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(2) from None
    if doc is None:
        err.print("[red]error:[/] evaluation pre-flight failed; nothing was evaluated:")
        for c in report.checks:
            if c.status is CheckStatus.ERROR:
                err.print(f"  - {escape(c.name)}: {escape(c.detail)}")
        raise typer.Exit(2)
    if json_output:
        typer.echo(doc.model_dump_json(indent=2))
    else:
        render_evaluation(doc, Console(highlight=False))
    if doc.image.restoration == "failed":  # shared Docker state was left changed: say so loudly
        err.print(
            f"[bold red]error:[/] the task image tag {escape(doc.image.reference)} was NOT "
            "restored; see the warning above for the exact recovery command"
        )
        raise typer.Exit(1)
    # Exit 0 whenever the evaluator gave a clear benchmark verdict (pass OR fail);
    # 1 when there is no verdict (infrastructure failure / ambiguous).
    raise typer.Exit(
        0
        if doc.classification in (Classification.BENCHMARK_PASS, Classification.BENCHMARK_FAIL)
        else 1
    )


def _emit(report: EvaluationPreflight, json_output: bool) -> None:
    if json_output:
        typer.echo(report.model_dump_for_json())
    else:
        render_preflight(report, Console(highlight=False))  # type: ignore[arg-type]
