from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent_harness_lab.benchmarks.base import BenchmarkAdapter, BenchmarkError
from agent_harness_lab.benchmarks.git import GitProbe
from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual

DEFAULT_BENCHMARK = SWEBenchMultilingual.name

_FACTORIES: dict[str, Callable[[Path, GitProbe | None], BenchmarkAdapter]] = {
    SWEBenchMultilingual.name: SWEBenchMultilingual,
}


class UnknownBenchmark(BenchmarkError):
    pass


def available_benchmarks() -> list[str]:
    return sorted(_FACTORIES)


def create_benchmark(name: str, repo_path: Path, git: GitProbe | None = None) -> BenchmarkAdapter:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise UnknownBenchmark(
            f"unknown benchmark {name!r}; available: {', '.join(available_benchmarks())}"
        ) from None
    return factory(repo_path, git)
