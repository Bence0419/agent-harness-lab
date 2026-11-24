from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent_harness_lab.benchmarks import (
    DEFAULT_BENCHMARK,
    BenchmarkAdapter,
    BenchmarkError,
    TaskRepoNotConfigured,
    create_benchmark,
)
from agent_harness_lab.benchmarks.git import GitProbe
from agent_harness_lab.config import TASKS_REPO_ENV, resolve_tasks_repo

tasks_app = typer.Typer(help="Inspect benchmark tasks.", no_args_is_help=True)

BenchmarkOpt = Annotated[str, typer.Option("--benchmark", help="Benchmark backend.")]
TasksRepoOpt = Annotated[
    Path | None,
    typer.Option(
        "--tasks-repo",
        help=f"Path to the task repository checkout (else ${TASKS_REPO_ENV}).",
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit JSON only.")]


def _git_probe() -> GitProbe | None:
    """Git probe override; ``None`` means the real one. Tests patch this."""
    return None


def open_benchmark(
    name: str, tasks_repo: Path | None, git: GitProbe | None = None
) -> BenchmarkAdapter:
    """Resolve the repository path (CLI > env > unset) and build the adapter."""
    resolved = resolve_tasks_repo(tasks_repo)
    if resolved.path is None:
        raise TaskRepoNotConfigured(
            f"task repository not configured: pass --tasks-repo PATH or set {TASKS_REPO_ENV}"
        )
    return create_benchmark(name, resolved.path, git if git is not None else _git_probe())


def fail(err: Console, exc: BenchmarkError) -> typer.Exit:
    err.print(f"[red]error:[/] {escape(str(exc))}")
    return typer.Exit(2 if isinstance(exc, TaskRepoNotConfigured) else 1)


@tasks_app.command("list")
def list_command(
    benchmark: BenchmarkOpt = DEFAULT_BENCHMARK,
    tasks_repo: TasksRepoOpt = None,
    repo: Annotated[
        str | None, typer.Option("--repo", help="Exact repository name, e.g. babel/babel.")
    ] = None,
    task: Annotated[
        str | None, typer.Option("--task", help="Case-insensitive substring of the instance id.")
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """List tasks (no problem statements; works without Docker)."""
    err = Console(stderr=True, highlight=False)
    try:
        listing = open_benchmark(benchmark, tasks_repo).list_tasks()
    except BenchmarkError as exc:
        raise fail(err, exc) from None

    tasks = [
        t
        for t in listing.tasks
        if (repo is None or t.repo == repo)
        and (task is None or task.lower() in t.instance_id.lower())
    ]
    # Malformed tasks are never dropped silently, whatever the filters.
    for p in listing.problems:
        err.print(f"[yellow]warning:[/] malformed task {escape(p.instance_id)}: {escape(p.reason)}")
    if listing.repository.commit is None:
        err.print("[yellow]warning:[/] could not read the task repository's git commit")
    if listing.repository.dirty:
        err.print(
            "[yellow]warning:[/] task repository has uncommitted changes (affects reproducibility)"
        )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "benchmark": listing.repository.benchmark,
                    "task_repository": {
                        "path": str(listing.repository.path),
                        "commit": listing.repository.commit,
                    },
                    "tasks": [
                        t.model_dump(mode="json", exclude={"problem_statement"}) for t in tasks
                    ],
                    "problems": [p.model_dump(mode="json") for p in listing.problems],
                },
                indent=2,
            )
        )
        return

    console = Console(highlight=False)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Instance", no_wrap=True)
    table.add_column("Repository", no_wrap=True)
    table.add_column("Base commit", no_wrap=True)
    table.add_column("Docker image", overflow="fold")
    for t in tasks:
        table.add_row(
            escape(t.instance_id), escape(t.repo), t.base_commit[:12], escape(t.docker_image)
        )
    console.print(table)
    commit = listing.repository.commit or "unknown"
    console.print(
        f"{len(tasks)} task(s) shown, {len(listing.tasks)} loaded, "
        f"{len(listing.problems)} malformed  |  {listing.repository.benchmark} @ {commit}"
    )


@tasks_app.command("show")
def show_command(
    instance_id: Annotated[str, typer.Argument(help="Task instance id.")],
    benchmark: BenchmarkOpt = DEFAULT_BENCHMARK,
    tasks_repo: TasksRepoOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Show one task, including its problem statement."""
    err = Console(stderr=True, highlight=False)
    try:
        task = open_benchmark(benchmark, tasks_repo).get_task(instance_id)
    except BenchmarkError as exc:
        raise fail(err, exc) from None

    if json_output:
        typer.echo(json.dumps(task.model_dump(mode="json"), indent=2))
        return

    console = Console(highlight=False)
    rows = [
        ("Benchmark", task.benchmark),
        ("Instance id", task.instance_id),
        ("Repository", task.repo),
        ("Base commit", task.base_commit),
        ("Docker image", task.docker_image),
        ("Task repo commit", task.task_repository_commit or "unknown"),
    ]
    for label, value in (("Split", task.split), ("Version", task.version)):
        if value is not None:
            rows.append((label, value))
    for label, value in rows:
        console.print(f"[bold]{label + ':':<18}[/]{escape(value)}")
    console.print("\n[bold]Problem statement[/]")
    console.print(task.problem_statement, markup=False, soft_wrap=True)


@tasks_app.command("history")
def history_command(
    instance_id: Annotated[str, typer.Argument(help="Task instance id.")],
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw-dir",
            help="Result store (default: ./results/raw).",
        ),
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """Has this task ever been run (complete or not) in the result store?"""
    from agent_harness_lab.cli.results import default_raw_dir
    from agent_harness_lab.paths import find_project_root
    from agent_harness_lab.results.history import frozen_task_ids, task_history

    err = Console(stderr=True, highlight=False)
    store = raw_dir or default_raw_dir()
    if not store.is_dir():
        err.print(f"[red]error:[/] no result store at {store}; history cannot be established")
        raise typer.Exit(1)
    hist = task_history(store, instance_id, frozen_task_ids(find_project_root()))
    if json_output:
        typer.echo(hist.model_dump_json(indent=2))
        return
    console = Console(highlight=False)
    verdict = "[yellow]SEEN[/]" if hist.seen else "[green]UNSEEN[/]"
    console.print(f"{escape(instance_id)}: {verdict}")
    console.print(f"  run directories:       {hist.run_directories}")
    console.print(f"  frozen research task:  {'yes' if hist.frozen_research_task else 'no'}")
    if hist.harnesses:
        console.print(f"  harnesses:             {', '.join(hist.harnesses)}")
        console.print(f"  models:                {escape(', '.join(hist.models))}")


@tasks_app.command("candidates")
def candidates_command(
    benchmark: BenchmarkOpt = DEFAULT_BENCHMARK,
    tasks_repo: TasksRepoOpt = None,
    count: Annotated[int, typer.Option("--count", min=1, help="How many candidates.")] = 3,
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw-dir",
            help="Result store (default: ./results/raw).",
        ),
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """Deterministic, outcome-independent canary candidates (unseen tasks, image local)."""
    from agent_harness_lab.benchmarks.selection import select_candidates
    from agent_harness_lab.cli import common
    from agent_harness_lab.cli.results import default_raw_dir
    from agent_harness_lab.paths import find_project_root
    from agent_harness_lab.results.history import frozen_task_ids

    err = Console(stderr=True, highlight=False)
    store = raw_dir or default_raw_dir()
    if not store.is_dir():
        err.print(f"[red]error:[/] no result store at {store}; unseen tasks cannot be determined")
        raise typer.Exit(1)
    try:
        listing = open_benchmark(benchmark, tasks_repo).list_tasks()
        local = common.docker_client().list_image_refs()  # type: ignore[attr-defined]
    except BenchmarkError as exc:
        raise fail(err, exc) from None
    frozen = frozen_task_ids(find_project_root())
    run_root = store / "swebench"
    selection = select_candidates(
        listing.tasks,
        is_seen=lambda i: i in frozen or (run_root / i).is_dir(),
        image_local=lambda ref: ref in local,
        count=count,
    )
    if json_output:
        typer.echo(selection.model_dump_json(indent=2))
        return
    console = Console(highlight=False)
    console.print(f"Rule: {selection.rule}\n")
    table = Table(show_header=True, header_style="bold")
    for col in ("Instance", "Repository", "Image local", "Docker image"):
        table.add_column(col, overflow="fold")
    for c in [*selection.ready, *selection.needs_image]:
        table.add_row(
            escape(c.instance_id), escape(c.repo), "yes" if c.image_local else "NO",
            escape(c.docker_image),
        )  # fmt: skip
    console.print(table)
    console.print(
        f"{len(selection.ready)} ready of {count} requested; {selection.unseen_total} unseen "
        f"tasks in total"
    )
    if selection.shortfall:
        console.print(
            f"[yellow]Shortfall: {selection.shortfall} -- only {len(selection.ready)} unseen "
            "task(s) have their image locally; the rest need an image that AHL cannot pull yet.[/]"
        )
