from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from agent_harness_lab.artifacts.models import RunRecord
from agent_harness_lab.paths import default_raw_dir
from agent_harness_lab.reporting.results import records_to_json, render_results
from agent_harness_lab.results.service import (
    LoadedRuns,
    MetricsSourceName,
    ResultsDirNotFound,
    attach_metrics,
    build_metrics_source,
    filter_runs,
    load_runs,
)

TaskOpt = Annotated[
    list[str] | None, typer.Option("--task", help="Task id substring (repeatable).")
]
HarnessOpt = Annotated[
    list[str] | None, typer.Option("--harness", help="Harness name, e.g. Pi (repeatable).")
]
ModelOpt = Annotated[
    list[str] | None, typer.Option("--model", help="Model name substring (repeatable).")
]
ReferenceOpt = Annotated[
    list[str] | None,
    typer.Option("--reference", help="Reference/experiment substring (repeatable)."),
]
RawDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--raw-dir",
        help="Raw results directory (default: ./results/raw).",
    ),
]


MetricsSourceOpt = Annotated[
    MetricsSourceName,
    typer.Option(
        "--metrics-source",
        help="native: parse traces (default); frozen-csv: regression oracle; none: skip traces.",
    ),
]


def attach_selected_metrics(
    records: list[RunRecord], source: MetricsSourceName, raw_dir: Path | None, err: Console
) -> list[RunRecord]:
    """Parse traces for the already-filtered records only; report warnings on stderr."""
    enriched, warnings = attach_metrics(
        records, build_metrics_source(source, raw_dir or default_raw_dir())
    )
    for w in warnings:
        err.print(f"[yellow]warning:[/] {w}", highlight=False)
    return enriched


def load_or_exit(raw_dir: Path | None, err: Console) -> LoadedRuns:
    path = raw_dir or default_raw_dir()
    try:
        loaded = load_runs(path)
    except ResultsDirNotFound:
        err.print(
            f"[red]No results directory found at {path}.[/]\n"
            "Run an experiment first, or pass --raw-dir to point at existing results."
        )
        raise typer.Exit(1) from None
    if loaded.issues:
        n = len(loaded.issues)
        err.print(f"[yellow]warning:[/] skipped {n} unreadable run directory(ies):")
        for issue in loaded.issues:
            err.print(f"  {issue.path.name}: {issue.reason}", markup=False)
    return loaded


def results_command(
    task: TaskOpt = None,
    harness: HarnessOpt = None,
    model: ModelOpt = None,
    reference: ReferenceOpt = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON only.")] = False,
    raw_dir: RawDirOpt = None,
    metrics_source: MetricsSourceOpt = MetricsSourceName.NATIVE,
) -> None:
    """List runs found under results/raw."""
    err = Console(stderr=True, highlight=False)
    loaded = load_or_exit(raw_dir, err)
    records = filter_runs(
        loaded.records,
        tasks=task or (),
        harnesses=harness or (),
        models=model or (),
        references=reference or (),
    )
    records = attach_selected_metrics(records, metrics_source, raw_dir, err)
    if json_output:
        typer.echo(records_to_json(records))
        return
    render_results(records, Console(highlight=False))
