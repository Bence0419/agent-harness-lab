from __future__ import annotations

import json
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from agent_harness_lab.artifacts.models import RunRecord


def records_to_json(records: Sequence[RunRecord]) -> str:
    return json.dumps([r.model_dump(mode="json") for r in records], indent=2)


def _resolved_cell(r: RunRecord) -> str:
    if r.resolved is None:
        return "[dim]N/A[/]"
    return "[green]PASS[/]" if r.resolved else "[red]FAIL[/]"


def render_results(records: Sequence[RunRecord], console: Console) -> None:
    table = Table(show_header=True, header_style="bold")
    for name, kw in (
        ("Task", {"no_wrap": True}),
        ("Harness", {"no_wrap": True}),
        ("Model", {"no_wrap": True}),
        ("Reasoning", {"no_wrap": True}),
        ("Resolved", {"no_wrap": True}),
        ("Duration", {"justify": "right", "no_wrap": True}),
        ("Cost", {"justify": "right", "no_wrap": True}),
        ("Run ID", {"overflow": "fold"}),
    ):
        table.add_column(name, **kw)
    for r in records:
        table.add_row(
            r.task,
            r.harness,
            r.model or "N/A",
            r.reasoning or "N/A",
            _resolved_cell(r),
            "N/A" if r.duration_seconds is None else f"{r.duration_seconds} s",
            "N/A" if r.cost_usd is None else f"${r.cost_usd:.4f}",
            r.run_id,
        )
    console.print(table)
    console.print(f"{len(records)} run(s)")
