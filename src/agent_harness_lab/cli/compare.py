from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from agent_harness_lab.cli.results import (
    MetricsSourceOpt,
    ModelOpt,
    RawDirOpt,
    ReferenceOpt,
    TaskOpt,
    attach_selected_metrics,
    load_or_exit,
)
from agent_harness_lab.reporting.compare import (
    NoComparableRuns,
    build_comparison,
    render_comparison,
    render_warnings,
)
from agent_harness_lab.results.service import MetricsSourceName, filter_runs


def compare_command(
    harness: Annotated[
        list[str] | None, typer.Option("--harness", help="Harness to compare (repeat, min. 2).")
    ] = None,
    task: TaskOpt = None,
    model: ModelOpt = None,
    reference: ReferenceOpt = None,
    raw_dir: RawDirOpt = None,
    metrics_source: MetricsSourceOpt = MetricsSourceName.NATIVE,
    include_canaries: Annotated[
        bool,
        typer.Option(
            "--include-canaries",
            help="Include canary runs (excluded by default so they never mix into experiments).",
        ),
    ] = False,
) -> None:
    """Compare harnesses over the tasks they have in common."""
    err = Console(stderr=True, highlight=False)
    if not harness or len(harness) < 2:
        err.print("[red]Specify at least two harnesses, e.g. --harness Pi --harness OpenCode[/]")
        raise typer.Exit(2)
    loaded = load_or_exit(raw_dir, err)
    # Filter first (including harness) so traces are parsed only for selected runs.
    records = filter_runs(
        loaded.records,
        tasks=task or (),
        harnesses=harness,
        models=model or (),
        references=reference or (),
    )
    canaries = [r for r in records if r.purpose == "canary"]
    if canaries and not include_canaries:
        records = [r for r in records if r.purpose != "canary"]
        err.print(
            f"[yellow]note:[/] {len(canaries)} canary run(s) excluded from the comparison "
            "(use --include-canaries to include them)"
        )
    records = attach_selected_metrics(records, metrics_source, raw_dir, err)
    try:
        comparison = build_comparison(records, harness)
    except NoComparableRuns as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from None
    console = Console(highlight=False)
    render_comparison(comparison, console)
    render_warnings(comparison.warnings, console)
