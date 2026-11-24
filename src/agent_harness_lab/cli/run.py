from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from agent_harness_lab.benchmarks import DEFAULT_BENCHMARK, BenchmarkError
from agent_harness_lab.benchmarks.git import GitHead, GitProbe
from agent_harness_lab.cli import common
from agent_harness_lab.cli.results import RawDirOpt
from agent_harness_lab.cli.tasks import TasksRepoOpt, fail, open_benchmark
from agent_harness_lab.config import resolve_path_setting
from agent_harness_lab.doctor.checks import HostProbe
from agent_harness_lab.execution.executor import RunRequest, execute_run
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.execution.models import ExecutionDisabled, ExecutionState
from agent_harness_lab.execution.preflight import CheckStatus, PreflightReport, run_preflight
from agent_harness_lab.harnesses import get_harness
from agent_harness_lab.paths import default_raw_dir
from agent_harness_lab.providers import (
    PROVIDERS_FILE_ENV,
    ProviderError,
    ProviderSpec,
    load_providers_file,
)
from agent_harness_lab.reporting.preflight import render_preflight
from agent_harness_lab.reporting.runplan import render_plan
from agent_harness_lab.runplan.builder import PlanProbe, build_run_plan
from agent_harness_lab.runplan.models import DEFAULT_TIMEOUT_SECONDS, NetworkMode, PlanResult
from agent_harness_lab.runplan.prompt import build_agent_prompt


def _probe() -> PlanProbe:
    """Host probe override point; tests patch this."""
    return HostProbe()


def _git() -> GitProbe:
    """Git probe override point; tests patch this."""
    return GitHead()


def run_command(
    task: Annotated[str, typer.Option("--task", help="Task instance id.")],
    harness: Annotated[str, typer.Option("--harness", help="Harness: pi or opencode.")],
    model: Annotated[str, typer.Option("--model", help="PROVIDER/MODEL, e.g. openai/gpt-6-astra.")],
    reasoning: Annotated[
        str | None, typer.Option("--reasoning", help="Reasoning level, e.g. medium.")
    ] = None,
    timeout: Annotated[
        int, typer.Option("--timeout", min=1, help="Agent timeout in seconds.")
    ] = DEFAULT_TIMEOUT_SECONDS,
    network: Annotated[
        NetworkMode, typer.Option("--network", help="Network policy for the agent container.")
    ] = NetworkMode.RESTRICTED_PROVIDER,
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="Benchmark backend.")
    ] = DEFAULT_BENCHMARK,
    tasks_repo: TasksRepoOpt = None,
    providers_file: Annotated[
        Path | None,
        typer.Option(
            "--providers-file", help=f"Providers definition file (else ${PROVIDERS_FILE_ENV})."
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Resolve and print the plan; execute nothing.")
    ] = False,
    preflight: Annotated[
        bool,
        typer.Option(
            "--preflight",
            help="Stricter than --dry-run: also check files, images and secrets. Creates nothing.",
        ),
    ] = False,
    raw_dir: RawDirOpt = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the plan/pre-flight report as JSON only.")
    ] = False,
) -> None:
    """Plan (--dry-run), verify (--preflight) or execute ONE run of a task.

    Real execution calls the model API (it costs money) and requires the pre-release safety
    gate AHL_ENABLE_REAL_EXECUTION=1. Single runs are recorded with purpose "canary"; use
    `ahl benchmark` for experiments. --dry-run and --preflight create and execute nothing.
    """
    err = Console(stderr=True, highlight=False)
    env = common.environ()
    if dry_run and preflight:
        err.print("[red]error:[/] --dry-run and --preflight are mutually exclusive")
        raise typer.Exit(2)
    executing = not dry_run and not preflight
    if executing and not real_execution_enabled(env):
        err.print(
            f"[red]error:[/] real execution is disabled: set {GATE_ENV}=1 (a temporary "
            "pre-release safety gate), or use --dry-run / --preflight"
        )
        raise typer.Exit(2)

    git = _git()
    benchmark_adapter = None
    try:
        providers = _load_providers(providers_file)
        benchmark_adapter = open_benchmark(benchmark, tasks_repo, git)
        result = build_run_plan(
            benchmark=benchmark_adapter,
            task_id=task,
            harness_name=harness,
            model_ref=model,
            reasoning=reasoning,
            timeout_seconds=timeout,
            network_mode=network,
            providers=providers,
            probe=_probe(),
            git=git,
            purpose="canary",
        )
    except (BenchmarkError, ProviderError) as exc:
        if preflight:
            report = PreflightReport()
            report.add("plan resolution", CheckStatus.ERROR, str(exc))
            _emit_preflight(report, json_output)
            raise typer.Exit(1) from None
        if isinstance(exc, ProviderError):
            err.print(f"[red]error:[/] {escape(str(exc))}")
            raise typer.Exit(1) from None
        raise fail(err, exc) from None  # type: ignore[arg-type]

    for w in result.warnings:
        err.print(f"[yellow]warning:[/] {escape(w)}")

    if dry_run:
        if json_output:
            typer.echo(result.plan.model_dump_json(indent=2))
            return
        render_plan(result, Console(highlight=False))
        return

    report = _preflight(result, env)
    if preflight:
        _emit_preflight(report, json_output)
        raise typer.Exit(0 if report.ready else 1)

    # ---- real execution (gate already verified) -----------------------------------------------
    if not report.ready:
        err.print("[red]error:[/] pre-flight failed; nothing was created:")
        for c in report.checks:
            if c.status is CheckStatus.ERROR:
                err.print(f"  - {escape(c.name)}: {escape(c.detail)}")
        raise typer.Exit(1)
    assert benchmark_adapter is not None
    plan = result.plan
    adapter = get_harness(plan.harness.name)
    host = common.host_environment()
    request = RunRequest(
        plan=plan,
        prompt=build_agent_prompt(benchmark_adapter.get_task(task).problem_statement),
        provisioner=lambda staged: adapter.provisioning(plan.harness.executable_path, host, staged),
        raw_dir=raw_dir or default_raw_dir(),
        plan_warnings=list(result.warnings),
    )
    try:
        doc = execute_run(request, docker=common.docker_client(), environ=env, sleep=common.sleep)
    except ExecutionDisabled as exc:  # defence in depth: the executor re-checks the gate
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(2) from None

    out = Console(highlight=False)
    out.print(f"Run:        {doc.run_id}")
    out.print(
        f"State:      {doc.state.value}"
        + (f" (stage: {doc.failed_stage})" if doc.failed_stage else "")
    )
    out.print(f"Exit code:  {doc.exit_code}")
    out.print(f"Cleanup:    {doc.cleanup.status}")
    out.print(f"Artifacts:  {request.raw_dir / 'swebench' / task / doc.run_id}")
    if doc.error:
        err.print(f"[red]error:[/] {escape(doc.error)}")
    raise typer.Exit(0 if doc.state is ExecutionState.COMPLETED else 1)


def _preflight(result: PlanResult, env) -> PreflightReport:
    plan = result.plan
    adapter = get_harness(plan.harness.name)
    host = common.host_environment()
    return run_preflight(
        plan,
        provisioner=lambda staged: adapter.provisioning(plan.harness.executable_path, host, staged),
        docker=common.docker_client(),
        path_exists=lambda p: host.is_file(p) or host.is_dir(p),
        environ=env,
        plan_warnings=result.warnings,
    )


def _emit_preflight(report: PreflightReport, json_output: bool) -> None:
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        render_preflight(report, Console(highlight=False))


def _load_providers(cli_value: Path | None) -> dict[str, ProviderSpec]:
    resolved = resolve_path_setting(cli_value, PROVIDERS_FILE_ENV)
    return load_providers_file(resolved.path) if resolved.path is not None else {}
