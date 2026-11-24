"""Human and JSON rendering for experiment plans and persisted experiment state."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent_harness_lab.execution.gate import GATE_ENV
from agent_harness_lab.experiments.planning import ExperimentPlan, group_failures
from agent_harness_lab.experiments.store import (
    CellRecord,
    ExperimentRecord,
    MatrixDocument,
)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def plan_to_json(plan: ExperimentPlan, *, dry_run: bool) -> dict:
    exp = plan.experiment
    return {
        "experiment": exp.experiment.name,
        "experiment_id": plan.experiment_id,
        "dry_run": dry_run,
        "definition_sha256": plan.definition_sha256,
        "ready": plan.ready,
        "tasks": len(exp.tasks),
        "harnesses": len(exp.harnesses),
        "models": len(exp.models),
        "repetitions": exp.repetitions,
        "runs": len(plan.cells),
        "evaluation_enabled": exp.evaluation.enabled,
        "real_execution_gate": {"env": GATE_ENV, "enabled": plan.gate_enabled},
        "task_repository": None
        if plan.repository is None
        else {
            "path": str(plan.repository.path),
            "commit": plan.repository.commit,
            "dirty": plan.repository.dirty,
            "pinned_commit": exp.benchmark.task_repository_commit,
        },
        "overrides": plan.overrides,
        "notes": plan.notes,
        "failures": [
            {"check": check, "detail": detail, "cells": cells}
            for check, detail, cells in group_failures(plan.failures)
        ],
        "evaluation_checks": [
            {"name": n, "status": s, "detail": d} for n, s, d in plan.evaluation_checks
        ],
        "cells": [
            {
                "cell_id": c.cell.cell_id,
                "task": c.cell.task,
                "harness": c.cell.harness,
                "model": c.cell.model_ref,
                "reasoning": c.cell.reasoning,
                "repetition": c.cell.repetition,
                "attempt": c.attempt,
                "ready": not c.failures,
                "failures": [{"check": f.check, "detail": f.detail} for f in c.failures],
                "warnings": c.warnings,
                "plan": None if c.result is None else c.result.plan.model_dump(mode="json"),
            }
            for c in plan.cells
        ],
    }


def render_plan(plan: ExperimentPlan, console: Console, *, dry_run: bool) -> None:
    exp = plan.experiment
    p = console.print
    p(f"Experiment: {escape(exp.experiment.name)}")
    p(f"Tasks:      {len(exp.tasks)}")
    p(f"Harnesses:  {len(exp.harnesses)}")
    p(f"Models:     {len(exp.models)}")
    p(f"Repeats:    {exp.repetitions}")
    p(f"Runs:       {len(plan.cells)}")
    p(f"Evaluation: {'enabled' if exp.evaluation.enabled else 'disabled'}")
    timeout = exp.execution.timeout_seconds
    if timeout is None:
        p("Estimate:   no execution timeout configured: duration unbounded")
    else:
        p(
            f"Estimate:   {len(plan.cells)} sequential run(s), at most {timeout}s of agent time "
            f"each (<= {_hours(len(plan.cells) * timeout)} worst case, before evaluation)"
        )
    p(f"Definition: sha256 {plan.definition_sha256}")
    p(
        f"Would be:   {escape(plan.experiment_id)}"
        if dry_run
        else f"ID:         {plan.experiment_id}"
    )
    gate = "set" if plan.gate_enabled else "NOT set"
    p(f"Gate:       {GATE_ENV}=1 is {gate}" + (" (not needed for --dry-run)" if dry_run else ""))
    if plan.repository is not None:
        r = plan.repository
        pin = exp.benchmark.task_repository_commit
        p(
            f"Task repo:  {r.commit or 'unknown'}"
            + (" (dirty)" if r.dirty else "")
            + (f"; pinned {pin}" if pin else "; NOT pinned")
        )
    for note in plan.notes:
        console.print(f"[bold yellow]note:[/] {escape(note)}")

    table = Table(show_header=True, header_style="bold")
    for col in ("Task", "Harness", "Model", "Reasoning", "Rep", "Pre-flight"):
        table.add_column(col)
    for c in plan.cells:
        status = "[green]ready[/]" if not c.failures else "[red]NOT READY[/]"
        table.add_row(
            c.cell.task,
            c.cell.harness,
            escape(c.cell.model_ref),
            c.cell.reasoning or "-",
            str(c.cell.repetition),
            status,
        )
    console.print(table)

    if plan.evaluation_checks:
        for name, status, detail in plan.evaluation_checks:
            color = {"ok": "green", "warning": "yellow", "error": "red"}[status]
            console.print(f"  evaluation {escape(name)}: [{color}]{status}[/] {escape(detail)}")
    failures = group_failures(plan.failures)
    if failures:
        console.print(f"[bold red]{len(plan.failures)} pre-flight failure(s):[/]")
        for check, detail, cells in failures:
            where = f" ({len(cells)} cell(s))" if cells else " (experiment-wide)"
            console.print(f"  [red]-[/] {escape(check)}: {escape(detail)}{where}")
        console.print("[red]NOT READY:[/] zero runs would be started.")
    elif dry_run:
        console.print("[green]READY[/]: nothing was executed or evaluated (--dry-run).")


STATE_COLORS = {
    "passed": "green",
    "failed": "red",
    "completed": "green",
    "no_verdict": "yellow",
    "infra_failure": "yellow",
    "harness_failed": "red",
    "timed_out": "red",
}


def _cell_row(c: CellRecord) -> tuple[str, ...]:
    a = c.current
    color = STATE_COLORS.get(c.state.value, "white")
    return (
        c.cell.task,
        c.cell.harness,
        escape(c.cell.model_ref),
        c.cell.reasoning or "-",
        str(c.cell.repetition),
        f"[{color}]{c.state.value}[/]",
        str(len(c.attempts)),
        (a.run_id or "-") if a else "-",
    )


def render_status(
    record: ExperimentRecord, matrix: MatrixDocument, summary: dict | None, console: Console
) -> None:
    p = console.print
    p(f"Experiment: {escape(record.experiment_id)}")
    p(
        f"Started:    {record.started_at}"
        + (f"   Finished: {record.finished_at}" if record.finished_at else "")
    )
    p(f"Definition: sha256 {record.definition_sha256}")
    if record.overrides:
        console.print(
            "[bold yellow]DEVELOPMENT OVERRIDES: "
            + escape(", ".join(record.overrides))
            + " (not a strict reproducible run)[/]"
        )
    table = Table(show_header=True, header_style="bold")
    for col in ("Task", "Harness", "Model", "Reasoning", "Rep", "State", "Attempts", "Run"):
        table.add_column(col)
    for c in matrix.cells:
        table.add_row(*_cell_row(c))
    console.print(table)
    if summary is None:
        p("[dim]No summary.json yet.[/]")
        return
    counts = summary["counts"]
    corr = summary["correctness"]
    p(f"Status:     {summary['status']}")
    p(
        f"Cells:      {counts['planned']} planned, {counts['executed']} executed, "
        f"{counts['evaluated']} evaluated, {counts['pending']} pending"
    )
    p(
        f"Outcomes:   {counts['passed']} PASS, {counts['failed']} FAIL, "
        f"{counts['no_verdict']} no verdict, {counts['infrastructure_failures']} infrastructure, "
        f"{counts['harness_failures']} harness failure, {counts['timeouts']} timeout"
    )
    rate = "N/A" if corr["success_rate"] is None else f"{corr['success_rate'] * 100:.1f}%"
    p(f"Correct:    {corr['resolved']} / {corr['denominator']} ({rate}) of evaluated cells")
    d = summary["durations"]
    p(
        f"Time:       agent {_fmt(d['agent_duration_seconds'])}s, "
        f"evaluation {_fmt(d['evaluation_duration_seconds'])}s"
    )
    t = summary["metrics"]["totals"]
    p(
        f"Metrics:    cost {_fmt(t.get('cost_usd'), 3)} USD, turns {_fmt(t.get('model_turns'))}, "
        f"tool calls {_fmt(t.get('tool_calls'))} "
        f"(over {summary['metrics']['runs_with_metrics']} runs)"
    )
    for title, groups in (("harness", summary["by_harness"]), ("model", summary["by_model"])):
        g = Table(show_header=True, header_style="bold", title=f"By {title}")
        for col in (
            title.capitalize(),
            "Resolved",
            "No verdict",
            "Infra",
            "Harness fail",
            "Timeout",
        ):
            g.add_column(col)
        for key, v in groups.items():
            g.add_row(
                escape(key),
                f"{v['resolved']} / {v['evaluated']}",
                str(v["no_verdict"]),
                str(v["infrastructure_failures"]),
                str(v["harness_failures"]),
                str(v["timeouts"]),
            )
        console.print(g)
    console.print(f"[dim]{escape(summary['notes'][0])}[/]")


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"
