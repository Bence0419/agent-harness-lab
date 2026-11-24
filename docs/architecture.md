# Architecture

AHL is layered so that each concern has one owner. Arrows are the only allowed dependencies.

```
cli ──> experiments ──> runplan ──> benchmarks (tasks), harnesses (adapters), providers
  │         │               │
  │         ├──────────────> execution (executor, Docker, network, workspace)
  │         └──────────────> evaluation (adapter per benchmark, image guard, baseline check)
  └──> results / metrics / traces / artifacts (read-only analysis of persisted runs)
```

| Package | Responsibility | Must not |
|---|---|---|
| `benchmarks` | Discover tasks from a task repository; expose a normalized, evaluator-free `Task`. | Open gold patches, hints, tests or eval scripts. Only `benchmarks/evaluator_data.py` touches evaluator files, and only for the evaluator. |
| `providers` | Provider/model references; credentials are environment-variable **names**. | Hold a secret value. |
| `harnesses` | One adapter per harness: renders the invocation, generated config and host provisioning. | Execute anything. |
| `runplan` | Resolve user intent into a validated, secret-free `RunPlan`. | Touch Docker or start an agent. |
| `execution` | Run one plan in an isolated container; capture patch, trace, hashes; always clean up. | Know how Pi/OpenCode are invoked (enforced by a test). |
| `evaluation` | Score a persisted `model.patch` with the benchmark's official evaluator; verify the evaluator image baseline; protect the user's image tag. | Know about harnesses; read model output other than `model.patch`. |
| `experiments` | Validate an experiment file, expand the matrix, pre-flight every cell, orchestrate sequentially, resume, summarize. | Contain harness- or evaluator-specific logic or any concurrency. |
| `results`, `metrics`, `traces`, `artifacts` | Read persisted runs (current and legacy), parse traces into `ahl-metrics-v1`, aggregate. | Modify run artifacts. |
| `doctor` | Capability-oriented environment checks. | Require a source checkout. |
| `cli`, `reporting` | Thin Typer commands and Rich/JSON rendering. | Hold domain logic. |

## Data flow of one run

1. `build_run_plan` loads the `Task`, discovers the harness (PATH, then a source checkout's
   `node_modules/.bin`), validates provider, model and reasoning, renders the harness invocation
   and records provenance in a `RunPlan`.
2. `run_preflight` statically checks the plan (read-only).
3. `execute_run` creates a run directory, stages the prompt and generated config, creates the
   network (an internal network plus a pinned Squid proxy for `restricted-provider`) and the
   task container, resets the workspace to the task's base commit, runs the harness with the
   timeout, captures `model.patch`, and removes everything it created. `run.json` is written
   atomically at the start, when the harness starts, and at the end.
4. `ahl evaluate` hands `model.patch` to the official evaluator and writes `evaluation.json`.

An experiment is a deterministic loop over step 1-4 for every matrix cell, with the
experiment-wide pre-flight in front and resumable state behind it.

## Extension points

* New harness: implement `HarnessAdapter` and register it (see `CONTRIBUTING.md`).
* New benchmark: implement `BenchmarkAdapter` (tasks) and an `EvaluationAdapter` (scoring).
* Neither requires changing the executor.

Detail per layer: [run-plan.md](run-plan.md), [execution.md](execution.md),
[evaluation.md](evaluation.md), [experiments.md](experiments.md).
