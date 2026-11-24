import typer
from rich.console import Console
from rich.table import Table

from agent_harness_lab.doctor import DoctorReport, Status, SystemProbe, run_checks

_STYLES = {Status.OK: "green", Status.WARNING: "yellow", Status.ERROR: "bold red"}


def _make_probe() -> SystemProbe | None:
    """Return a probe override; ``None`` means use the real host. Tests patch this."""
    return None


def render_report(report: DoctorReport, console: Console) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for r in report.results:
        table.add_row(r.name, f"[{_STYLES[r.status]}]{r.status.value}[/]", r.detail)
    console.print(table)
    console.print(
        f"{report.count(Status.OK)} ok, "
        f"{report.count(Status.WARNING)} warning(s), "
        f"{report.count(Status.ERROR)} error(s)"
    )


def doctor_command() -> None:
    """Check the environment. Exits non-zero only on foundational errors."""
    report = run_checks(_make_probe())
    render_report(report, Console(highlight=False))
    raise typer.Exit(1 if report.has_errors else 0)
