# Contributing

## Setup

```bash
uv sync                      # Python 3.12+; installs AHL and the dev tools
uv run ahl doctor
```

## Checks (all must pass)

```bash
uv run ruff check .
uv run pytest                          # default suite
uv run python tools/check_public_release.py
```

The default suite needs **no Docker, no API key, no network, no Pi/OpenCode and no external task
repository**. CI runs exactly this. **Tests must never make a real LLM/API call** or spend
money; use the fake Docker client, the mock harness and fake evaluator in `tests/`.

## Optional suites

* Docker integration: `uv run pytest -m docker` (needs a Docker daemon; creates only labelled,
  throwaway resources).
* Real evaluator regression: needs `AHL_REAL_EVALUATOR=1`, `AHL_SWEBENCH_PYTHON`,
  `AHL_SWEBENCH_TASKS_REPO`, the historical task images and the ids of two historical runs
  (`AHL_REGRESSION_PASS_RUN`, `AHL_REGRESSION_FAIL_RUN`); see
  `tests/test_real_evaluator_regression.py`. It re-scores existing runs and calls no model.
* Real-data regression tests skip themselves without the frozen research data.

## Style

ruff (line length 100), type-annotated Pydantic models, small protocols for seams that tests
replace. Never print or store secret values; only environment variable **names**.

## Adding a harness

Implement `HarnessAdapter` (`harnesses/base.py`): `build_invocation` (argv, env, generated
config, secret env names), `provisioning` (mounts/copies) and reasoning validation; register it
in `harnesses/registry.py`; add a trace normalizer in `traces/` if its output format is new.
Do not add harness-specific branches to `execution/`, `evaluation/` or `experiments/` (tests
enforce this). Add adapter and parity tests.

## Adding a benchmark

Implement `BenchmarkAdapter` (`benchmarks/base.py`, with a normalized `Task` that carries no
evaluator material) and an `EvaluationAdapter` (`evaluation/base.py`), and register both.
Evaluator-side files must be read only by the evaluation adapter.

## Schema compatibility

Persisted documents (`run.json`, `evaluation.json`, experiment files) are public formats: see
`docs/schemas.md`. Adding an optional field is fine; changing the meaning of a field needs a new
schema id and a loader for the old one. Never edit historical research data or the frozen
`scripts/`, `configs/` and `docs/research-v1.md` except for documented public-release
redactions.
