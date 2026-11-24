from __future__ import annotations

from agent_harness_lab.evaluation.base import EvaluationAdapter, UnsupportedBenchmark
from agent_harness_lab.evaluation.swebench_multilingual import SWEBenchMultilingualEvaluator

_EVALUATORS: dict[str, EvaluationAdapter] = {
    SWEBenchMultilingualEvaluator.benchmark: SWEBenchMultilingualEvaluator(),
}


def available_evaluators() -> list[str]:
    return sorted(_EVALUATORS)


def get_evaluator(benchmark: str) -> EvaluationAdapter:
    try:
        return _EVALUATORS[benchmark]
    except KeyError:
        raise UnsupportedBenchmark(
            f"no evaluator for benchmark {benchmark!r}; "
            f"available: {', '.join(available_evaluators())}"
        ) from None
