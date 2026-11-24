"""Compare harnesses over matching tasks. Pure computation first, rendering second."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from agent_harness_lab.artifacts.models import RunRecord

NOT_RECORDED = "not recorded"

# (label, attribute) for metrics summed across runs; None when any run lacks it.
SUMMED_METRICS: tuple[tuple[str, str], ...] = (
    ("Total duration (s)", "duration_seconds"),
    ("Total cost (USD)", "cost_usd"),
    ("Tool calls", "tool_calls"),
    ("Model turns", "model_turns"),
    ("Verification calls", "verification_calls"),
    ("Output tokens", "output_tokens"),
    ("Reasoning tokens", "reasoning_tokens"),
    ("Cache read tokens", "cache_read_tokens"),
    ("Cache write tokens", "cache_write_tokens"),
)


class HarnessSummary(BaseModel):
    harness: str
    tasks: int
    runs: int
    evaluated: int
    resolved: int
    success_rate: float | None  # 0..1, over evaluated runs
    totals: dict[str, float | None]  # attribute -> total, None if any run lacks it


class Comparison(BaseModel):
    summaries: list[HarnessSummary]
    tasks_compared: list[str]
    warnings: list[str]


class NoComparableRuns(Exception):
    pass


def _total(runs: Sequence[RunRecord], attr: str) -> float | None:
    values = [getattr(r, attr) for r in runs]
    if not values or any(v is None for v in values):
        return None  # missing data is not zero; never report a partial sum
    return sum(values)


def _summarize(harness: str, runs: Sequence[RunRecord]) -> HarnessSummary:
    evaluated = [r for r in runs if r.resolved is not None]
    resolved = sum(1 for r in evaluated if r.resolved)
    return HarnessSummary(
        harness=harness,
        tasks=len({r.task for r in runs}),
        runs=len(runs),
        evaluated=len(evaluated),
        resolved=resolved,
        success_rate=(resolved / len(evaluated)) if evaluated else None,
        totals={attr: _total(runs, attr) for _, attr in SUMMED_METRICS},
    )


def summarize_runs(label: str, runs: Sequence[RunRecord]) -> HarnessSummary:
    """The one aggregation (counts, success rate, summed metrics) shared by ``ahl compare`` and
    experiment summaries. ``label`` names the group (a harness, a model, ...)."""
    return _summarize(label, runs)


def _distribution(values: Sequence[str | None]) -> str:
    counts = Counter(NOT_RECORDED if v is None else v for v in values)
    return ", ".join(f"{k} ({n})" for k, n in sorted(counts.items()))


def _warnings(runs: Sequence[RunRecord], by_harness: dict[str, list[RunRecord]]) -> list[str]:
    out: list[str] = []
    policies = [r.network_policy for r in runs]
    if len(set(policies)) > 1:
        out.append(f"mixed network policies: {_distribution(policies)}")
    elif policies and policies[0] is None:
        out.append("network policy not recorded for any selected run")
    dirty = sum(1 for r in runs if r.harness_dirty is True)
    if dirty:
        out.append(f"{dirty} run(s) were produced with a dirty harness working tree")
    unknown_dirty = sum(1 for r in runs if r.harness_dirty is None)
    if unknown_dirty:
        out.append(f"harness dirty state not recorded for {unknown_dirty} run(s)")
    for label, attr in (("models", "model"), ("reasoning", "reasoning")):
        values = [getattr(r, attr) for r in runs]
        if len(set(values)) > 1:
            out.append(f"mixed {label}: {_distribution(values)}")
    # Each harness normally has its own reference id; only differences *within*
    # one harness indicate mixed experiments.
    for h, rs in by_harness.items():
        refs = [r.reference for r in rs]
        if len(set(refs)) > 1:
            out.append(f"mixed references for {h}: {_distribution(refs)}")
    multi = {h: rs for h, rs in by_harness.items() if len(rs) > len({r.task for r in rs})}
    for h, rs in multi.items():
        out.append(
            f"{h}: {len(rs)} runs over {len({r.task for r in rs})} task(s); "
            "multiple attempts per task are counted as separate runs"
        )
    unevaluated = sum(1 for r in runs if r.resolved is None)
    if unevaluated:
        out.append(
            f"{unevaluated} run(s) have no valid evaluation and are excluded from success rate"
        )
    return out


def build_comparison(records: Sequence[RunRecord], harnesses: Sequence[str]) -> Comparison:
    """Compare ``harnesses`` over the tasks that every one of them has runs for."""
    canonical = {r.harness.lower(): r.harness for r in records}
    names: list[str] = []
    for h in harnesses:
        name = canonical.get(h.lower())
        if name is None:
            raise NoComparableRuns(f"no runs for harness {h!r} in the selection")
        if name not in names:
            names.append(name)
    if len(names) < 2:
        raise NoComparableRuns("at least two distinct harnesses are required")

    task_sets = [{r.task for r in records if r.harness == n} for n in names]
    common = sorted(set.intersection(*task_sets))
    if not common:
        raise NoComparableRuns("the selected harnesses have no tasks in common")

    by_harness = {n: [r for r in records if r.harness == n and r.task in common] for n in names}
    selected = [r for rs in by_harness.values() for r in rs]
    warnings = _warnings(selected, by_harness)
    excluded = sorted(set().union(*task_sets) - set(common))
    if excluded:
        warnings.append(f"{len(excluded)} task(s) not run by every harness were excluded")

    return Comparison(
        summaries=[_summarize(n, by_harness[n]) for n in names],
        tasks_compared=common,
        warnings=warnings,
    )


def _fmt_number(value: float | None, attr: str) -> str:
    if value is None:
        return "N/A"
    if attr == "cost_usd":
        return f"{value:.3f}"
    return f"{int(round(value))}"


def _relative_difference(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "N/A"
    return f"{(b - a) / a * 100:+.1f}%"


def _rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}%"


def _pp_difference(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "N/A"
    return f"{(b - a) * 100:+.1f} percentage points"


def render_comparison(comparison: Comparison, console: Console) -> None:
    s = comparison.summaries
    two = len(s) == 2
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    for item in s:
        table.add_column(item.harness, justify="right")
    if two:
        table.add_column(f"{s[1].harness} vs {s[0].harness}", justify="right")

    def row(label: str, cells: list[str], diff: str | None = None) -> None:
        table.add_row(label, *cells, *([diff] if two and diff is not None else [""] * two))

    row("Tasks", [str(i.tasks) for i in s])
    row("Runs", [str(i.runs) for i in s])
    row("Resolved", [f"{i.resolved} / {i.evaluated}" for i in s])
    row(
        "Success rate",
        [_rate(i.success_rate) for i in s],
        _pp_difference(s[0].success_rate, s[1].success_rate) if two else None,
    )
    for label, attr in SUMMED_METRICS:
        values = [i.totals[attr] for i in s]
        row(
            label,
            [_fmt_number(v, attr) for v in values],
            _relative_difference(values[0], values[1]) if two else None,
        )
    console.print(f"Compared {len(comparison.tasks_compared)} task(s) run by every harness")
    console.print(table)
    console.print(
        "[dim]Differences are descriptive; no statistical significance is claimed. "
        "N/A = not available in the historical data (not zero).[/]"
    )


def render_warnings(warnings: Sequence[str], console: Console) -> None:
    for w in warnings:
        console.print(f"[yellow]warning:[/] {w}")

