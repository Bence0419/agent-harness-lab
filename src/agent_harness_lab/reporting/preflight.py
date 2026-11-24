from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent_harness_lab.execution.preflight import CheckStatus, PreflightReport

_STYLE = {CheckStatus.OK: "green", CheckStatus.WARNING: "yellow", CheckStatus.ERROR: "bold red"}


def render_preflight(report: PreflightReport, console: Console) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for c in report.checks:
        table.add_row(
            escape(c.name), f"[{_STYLE[c.status]}]{c.status.value.upper()}[/]", escape(c.detail)
        )
    console.print(table)

    if report.provisioning:
        console.print("\n[bold]Host paths that would be mounted or copied into the container[/]")
        prov = Table(show_header=True, header_style="bold")
        for col in ("Kind", "Host source", "Container destination", "Mode"):
            prov.add_column(col, overflow="fold")
        for p in report.provisioning:
            prov.add_row(
                p.kind, escape(p.source), escape(p.destination), "ro" if p.read_only else "rw"
            )
        console.print(prov)
    for g in report.generated_files:
        console.print(
            f"[dim]generated {escape(g.container_path)}: {g.bytes} bytes, "
            f"sha256 {g.sha256[:16]}…[/]"
        )
    verdict = "[green]READY[/]" if report.ready else "[bold red]NOT READY[/]"
    console.print(f"\nPre-flight: {verdict} (nothing was created or executed)")
